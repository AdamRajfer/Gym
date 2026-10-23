# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

import nemo_gym.exporters
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.cli.eval import _health_check_collected_rollouts
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.health.timeouts import (
    ELAPSED_S_KEY,
    LEGACY_TIMEOUT_SIGNALS,
    export_keys,
    read_rollout_timing,
    summarize_timeouts,
)
from nemo_gym.health.types import QUALITY_SUMMARY_FILENAME, RunHealthLimits
from nemo_gym.path_utils import failures_path_for
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
from nemo_gym.rollout_health import health_check_run_dir
from tests.unit_tests.test_rollout_collection import (  # noqa: F401 - fixture
    FakeResponse,
    empty_global_config,
    install_fake_server_client,
)


REPO_ROOT = Path(__file__).parents[2]
T0 = 1_000_000.0  # Unix time at which a rollout's first call starts.


def event(event_id: str = "e1", **fields) -> dict:
    """A timeout event as the component that enforced the limit reports it."""
    return {"event_id": event_id, "enforcer": "gym.environment_server", "evidence": "internal", **fields}


def model_call(start: float, end: float, *, model: str = "policy_model", tokens_out: int | None = 100) -> dict:
    return {
        "model_ref": {"type": "responses_api_models", "name": model},
        "started_at": T0 + start,
        "completed_at": T0 + end,
        "latency_total_ms": (end - start) * 1000,
        "tokens_out": tokens_out,
    }


def tool_call(start: float, end: float) -> dict:
    return {"started_at": T0 + start, "completed_at": T0 + end, "duration_ms": (end - start) * 1000}


def timed_out_row(elapsed: float = 100.0, calls: list | None = None, tools: list | None = None, **fields) -> dict:
    """A rollout that ran out of time on the agent's own deadline."""
    row = {"agent_timed_out": True, ELAPSED_S_KEY: elapsed, "reward": 0.0, **fields}
    if calls is not None:
        row["ng_model_call_capture"] = {"calls": calls}
    if tools is not None:
        row["ng_trajectory"] = {"tool_calls": tools}
    return row


def section(*rows: dict) -> dict:
    return summarize_timeouts(read_rollout_timing(row) for row in rows).model_dump()


class TestEvidence:
    def test_a_typed_event_names_the_limit_and_its_enforcer(self) -> None:
        row = {"timeouts": [event(limit="default_episode_timeout_seconds")]}

        result = section(row)

        assert result["by_evidence"] == {"typed": 1, "legacy": 0}
        assert result["by_limit"] == {"default_episode_timeout_seconds": 1}
        assert result["by_enforcer"] == {"gym.environment_server": 1}

    def test_a_typed_event_wins_over_a_legacy_signal(self) -> None:
        row = {"agent_timed_out": True, "timeouts": [event(limit="agent_timeout_seconds")]}

        assert section(row)["by_limit"] == {"agent_timeout_seconds": 1}

    def test_an_event_listed_twice_counts_once(self) -> None:
        assert section({"timeouts": [event("e1"), event("e1")]})["events"] == 1

    def test_only_the_row_carries_events(self) -> None:
        """Collection drops a trajectory whose operation records carry `timeouts`, so they are not read there."""
        trajectory = {"invocations": [{"timeouts": [event("e1")]}], "tool_calls": [{"timeouts": [event("e2")]}]}

        assert section({"ng_trajectory": trajectory})["timed_out"] == 0

    def test_an_invalid_event_is_counted_but_is_not_evidence(self) -> None:
        row = {"timeouts": [{"event_id": "e1"}]}  # No enforcer: not a valid report.

        result = section(row)

        assert (result["timed_out"], result["invalid_events"]) == (0, 1)

    @pytest.mark.parametrize("signal", LEGACY_TIMEOUT_SIGNALS, ids=lambda signal: signal.name)
    def test_each_legacy_signal_counts_as_a_legacy_timeout(self, signal) -> None:
        result = section({signal.field: signal.value if signal.value is not None else 1})

        assert result["by_limit"] == {signal.name: 1}

    def test_a_flag_set_to_zero_is_not_a_timeout(self) -> None:
        assert section({"agent_timeout_error": 0})["timed_out"] == 0

    def test_a_terminal_bench_termination_by_timeout_is_a_legacy_timeout(self) -> None:
        assert section({"termination": {"reason": "timeout"}})["by_limit"] == {"termination.reason=timeout": 1}

    def test_a_slow_failure_with_no_signal_is_never_counted_as_timed_out(self) -> None:
        row = {NG_FAILURE_CLASS_KEY: "agent_run_error", ELAPSED_S_KEY: 10_000.0}

        result = section(row)

        assert (result["timed_out"], result["failed_without_signal"]) == (0, 1)


@pytest.mark.parametrize("signal", LEGACY_TIMEOUT_SIGNALS, ids=lambda signal: signal.name)
def test_each_legacy_signal_is_still_written_by_its_producer(signal) -> None:
    """A legacy signal is only evidence while a Gym component writes it."""
    assert signal.written_as in (REPO_ROOT / signal.producer).read_text()


class TestTimeSplit:
    def test_policy_judge_and_tool_time_divide_the_elapsed_time(self) -> None:
        row = timed_out_row(
            elapsed=100.0,
            calls=[model_call(0, 40), model_call(50, 60, model="judge_model")],
            tools=[tool_call(40, 50)],
        )

        split = section(row)["time_split"]

        assert (split["policy_model"], split["other_model"], split["tool"], split["untracked"]) == (0.4, 0.1, 0.1, 0.4)

    def test_untracked_is_the_time_in_which_no_call_ran(self) -> None:
        """A tool that runs the policy model overlaps it; the stretch the two share is not counted twice."""
        row = timed_out_row(elapsed=100.0, calls=[model_call(0, 40)], tools=[tool_call(0, 40)])

        split = section(row)["time_split"]

        assert (split["policy_model"], split["tool"], split["untracked"]) == (0.4, 0.4, 0.6)

    def test_overlapping_calls_count_once(self) -> None:
        row = timed_out_row(elapsed=100.0, calls=[model_call(0, 30), model_call(10, 40)], tools=[])

        assert section(row)["time_split"]["policy_model"] == 0.4

    def test_busy_time_never_exceeds_the_elapsed_time(self) -> None:
        row = timed_out_row(elapsed=10.0, calls=[model_call(0, 30)], tools=[])

        assert section(row)["time_split"]["policy_model"] == 1.0

    def test_a_call_without_timing_makes_only_its_kind_unknown(self) -> None:
        untimed_judge = {"model_ref": {"type": "responses_api_models", "name": "judge_model"}}
        row = timed_out_row(calls=[model_call(0, 40), untimed_judge], tools=[])

        split = section(row)["time_split"]

        assert (split["policy_model"], split["other_model"], split["untracked"]) == (0.4, None, None)

    @pytest.mark.parametrize(
        "trajectory",
        [{}, {"tool_calls": [{"started_at": T0}]}, {"tool_calls": [], "gaps": [{"code": "tool_timing_unavailable"}]}],
        ids=["no-tool-list", "untimed-tool-call", "reported-gap"],
    )
    def test_unmeasured_tool_time_is_unknown(self, trajectory: dict) -> None:
        row = timed_out_row(calls=[model_call(0, 40)], ng_trajectory=trajectory)

        assert section(row)["time_split"]["tool"] is None

    def test_an_empty_tool_list_is_zero_tool_time(self) -> None:
        assert section(timed_out_row(calls=[model_call(0, 40)], tools=[]))["time_split"]["tool"] == 0.0

    def test_a_call_from_a_capture_without_model_ref_is_the_policy(self) -> None:
        call = model_call(0, 40)
        del call["model_ref"]

        assert section(timed_out_row(calls=[call], tools=[]))["time_split"]["policy_model"] == 0.4

    def test_serving_speed_is_the_policy_output_rate(self) -> None:
        row = timed_out_row(calls=[model_call(0, 10, tokens_out=500), model_call(10, 20, tokens_out=100)], tools=[])

        speed = section(row)["time_split"]["policy_output_tokens_per_s"]

        assert speed == {"p50": 30.0, "p5": 12.0}

    def test_rollouts_without_captured_calls_are_not_split(self) -> None:
        assert section(timed_out_row())["time_split"]["rollouts_with_captures"] == 0


def test_an_empty_run_has_an_empty_section() -> None:
    result = section()

    assert (result["rollouts"], result["timed_out"], result["elapsed_s"]) == (0, 0, {"p50": None, "p95": None})


# --- A malformed record never stops the health report ---------------------------------------------------------

_MALFORMED = {
    "trajectory_invocations_not_a_list": {"ng_trajectory": {"invocations": 7}},
    "tool_call_not_an_object": {"ng_trajectory": {"tool_calls": [7]}},
    "trajectory_not_an_object": {"ng_trajectory": "x"},
    "capture_calls_not_a_list": {"ng_model_call_capture": {"calls": 7}},
    "capture_not_an_object": {"ng_model_call_capture": [1]},
    "model_ref_not_an_object": {"ng_model_call_capture": {"calls": [{"model_ref": "policy_model"}]}},
    "timeouts_not_a_list": {"timeouts": 7},
    "termination_not_an_object": {"termination": "timeout"},
    "timing_not_a_number": {
        ELAPSED_S_KEY: "12",
        "ng_model_call_capture": {"calls": [{"started_at": "0", "completed_at": 1.0, "latency_total_ms": "5"}]},
    },
}


@pytest.mark.parametrize("malformed", _MALFORMED.values(), ids=_MALFORMED.keys())
def test_a_malformed_row_still_gets_a_health_report(tmp_path: Path, malformed: dict) -> None:
    rows = [
        {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
        {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, **malformed},
    ]
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    timeouts = health_check_run_dir(tmp_path).summary["run"]["stats"]["timeouts"]

    assert timeouts["rollouts"] == 2
    assert (tmp_path / QUALITY_SUMMARY_FILENAME).is_file()


def test_a_malformed_failure_row_still_gets_a_health_report(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    output.write_text(json.dumps({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}) + "\n")
    failure = {
        TASK_INDEX_KEY_NAME: 1,
        ROLLOUT_INDEX_KEY_NAME: 0,
        NG_FAILURE_CLASS_KEY: "agent_run_error",
        "ng_model_call_capture": {"calls": 7},
    }
    failures_path_for(output).write_text(json.dumps(failure) + "\n")

    timeouts = health_check_run_dir(tmp_path).summary["run"]["stats"]["timeouts"]

    assert (timeouts["rollouts"], timeouts["failed_without_signal"]) == (2, 1)


def exchange(start: float, end: float, *, model: str = "policy_model") -> dict:
    """One model exchange as the model server's capture middleware records it."""
    return {
        "dialect": "chat",
        "model_ref": {"type": "responses_api_models", "name": model},
        "started_at": T0 + start,
        "completed_at": T0 + end,
        "latency_ms": (end - start) * 1000,
        "status_code": 200,
        "request": {},
        "response": {},
    }


def deadline_failure(task: int, *, attempt: int = 0, elapsed: float = 100.0, **fields) -> dict:
    """A rollout the environment server's episode deadline cut: its last attempt's failure row, with no capture."""
    return {
        TASK_INDEX_KEY_NAME: task,
        ROLLOUT_INDEX_KEY_NAME: 0,
        **({ATTEMPT_INDEX_KEY_NAME: attempt} if attempt else {}),
        NG_FAILURE_CLASS_KEY: "environment_server_failed",
        "_ng_failure_message": "Episode timed out",
        ELAPSED_S_KEY: elapsed,
        **fields,
    }


class TestCapturesOfFailedRollouts:
    """A rollout that failed without a result has no capture on its row; the capture store holds its calls."""

    @staticmethod
    def timeouts(tmp_path: Path, failures: list[dict], captures: dict[str, list[dict]], *, with_dir: bool = True):
        output = tmp_path / "rollouts.jsonl"
        output.write_text("")
        failures_path_for(output).write_text("".join(json.dumps(row) + "\n" for row in failures))
        capture_dir = tmp_path / "model-calls"
        store = CaptureStore(capture_dir)
        for rollout_id, exchanges in captures.items():
            for recorded in exchanges:
                store.record(rollout_id, recorded)
        result = health_check_run_dir(tmp_path, model_call_capture_dir=capture_dir if with_dir else None)
        return result.summary["run"]["stats"]["timeouts"]

    def test_a_failure_row_takes_its_split_from_the_capture_file(self, tmp_path: Path) -> None:
        captures = {"0-0": [exchange(0, 40), exchange(50, 70), exchange(60, 80, model="judge_model")]}

        timeouts = self.timeouts(tmp_path, [deadline_failure(0)], captures)

        split = timeouts["time_split"]
        assert (timeouts["timed_out"], split["rollouts_with_captures"]) == (1, 1)
        assert (split["policy_model"], split["other_model"]) == (0.6, 0.2)
        assert split["policy_calls_per_rollout"]["p50"] == 2
        # The failure row records no tool calls, so tool time and the untracked share stay unknown.
        assert (split["tool"], split["untracked"]) == (None, None)

    def test_the_last_attempt_reads_its_own_capture(self, tmp_path: Path) -> None:
        failures = [deadline_failure(0, elapsed=90.0), deadline_failure(0, attempt=1)]
        captures = {"0-0": [exchange(0, 90)], "0-0-a1": [exchange(0, 30)]}

        split = self.timeouts(tmp_path, failures, captures)["time_split"]

        assert (split["rollouts_with_captures"], split["policy_model"]) == (1, 0.3)

    def test_a_row_with_its_own_capture_is_not_counted_twice(self, tmp_path: Path) -> None:
        row = deadline_failure(0, ng_model_call_capture={"calls": [model_call(0, 10)]})

        split = self.timeouts(tmp_path, [row], {"0-0": [exchange(0, 90)]})["time_split"]

        assert (split["rollouts_with_captures"], split["policy_model"]) == (1, 0.1)

    def test_a_rollout_without_a_capture_file_stays_unmeasured(self, tmp_path: Path) -> None:
        timeouts = self.timeouts(tmp_path, [deadline_failure(0), deadline_failure(1)], {"1-0": [exchange(0, 50)]})

        assert (timeouts["timed_out"], timeouts["time_split"]["rollouts_with_captures"]) == (2, 1)
        assert timeouts["time_split"]["policy_model"] == 0.5

    def test_without_the_capture_dir_the_split_is_not_measured(self, tmp_path: Path) -> None:
        timeouts = self.timeouts(tmp_path, [deadline_failure(0)], {"0-0": [exchange(0, 40)]}, with_dir=False)

        assert timeouts["time_split"]["rollouts_with_captures"] == 0


def test_export_keys_are_flat_and_leave_out_unknowns() -> None:
    keys = export_keys(section(timed_out_row()))

    assert keys["health/timeouts/timed_out"] == 1
    assert keys["health/timeouts/by_evidence/legacy"] == 1
    assert "health/timeouts/time_split/policy_model" not in keys


# --- Placement: the section is written wherever a run's health is computed, and exported from there ----------


def _input(tmp_path: Path, count: int) -> Path:
    fpath = tmp_path / "input.jsonl"
    rows = [
        {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "my_agent"}, "i": i}
        for i in range(count)
    ]
    fpath.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return fpath


_REPLIES = [
    {"reward": 1.0},
    {"reward": 0.0, "timeouts": [event(limit="default_episode_timeout_seconds", configured_s=60)]},
    {NG_FAILURE_CLASS_KEY: "timeout_exceeded", "error": "agent ran out of time"},
]


def _fake_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(json.loads(json.dumps(_REPLIES)))

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        if url_path == "/run":
            return FakeResponse(200, next(replies))
        return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))


def _exported_keys(export_metrics: MagicMock) -> dict:
    return {key: value for call in export_metrics.call_args_list for key, value in call.args[0].items()}


async def test_collection_writes_and_exports_the_section_and_a_recheck_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config
) -> None:
    _fake_servers(monkeypatch)
    export_metrics = MagicMock()
    monkeypatch.setattr(nemo_gym.exporters, "export_metrics", export_metrics)
    output = tmp_path / "rollouts.jsonl"

    await RolloutCollectionHelper().run_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(_input(tmp_path, 3)), output_jsonl_fpath=str(output), num_samples_in_parallel=1
        )
    )

    timeouts = orjson.loads((tmp_path / QUALITY_SUMMARY_FILENAME).read_bytes())["run"]["stats"]["timeouts"]
    assert (timeouts["rollouts"], timeouts["timed_out"]) == (3, 2)
    assert timeouts["by_limit"] == {"_ng_failure_class=timeout_exceeded": 1, "default_episode_timeout_seconds": 1}
    assert _exported_keys(export_metrics)["health/timeouts/timed_out"] == 2
    # The score is what it was: the failure row is not in it.
    [entry] = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
    assert entry["key_metrics"]["mean/reward"] == 0.5
    assert health_check_run_dir(tmp_path).summary["run"]["stats"]["timeouts"] == timeouts


async def test_offline_aggregation_writes_the_same_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config
) -> None:
    _fake_servers(monkeypatch)
    output = tmp_path / "rollouts.jsonl"
    await RolloutCollectionHelper().run_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(_input(tmp_path, 3)), output_jsonl_fpath=str(output), num_samples_in_parallel=1
        )
    )
    after_run = orjson.loads((tmp_path / QUALITY_SUMMARY_FILENAME).read_bytes())["run"]["stats"]["timeouts"]
    (tmp_path / QUALITY_SUMMARY_FILENAME).unlink()

    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(input_glob=str(output), output_jsonl_fpath=str(output))
    )

    assert orjson.loads((tmp_path / QUALITY_SUMMARY_FILENAME).read_bytes())["run"]["stats"]["timeouts"] == after_run


def _saved_run(tmp_path: Path) -> Path:
    output = tmp_path / "rollouts.jsonl"
    rows = [
        {**timed_out_row(), TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
        {"reward": 1.0, ELAPSED_S_KEY: 5.0, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output


_HEALTH_CONFIG = SimpleNamespace(
    run_health_limits=RunHealthLimits(), health_check_workers=1, health_check_ignored_checks=()
)


def test_gym_eval_run_exports_the_section_after_its_health_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gym eval run` computes health after its servers stop, outside rollout collection, and exports from there."""
    export_metrics = MagicMock()
    monkeypatch.setattr(nemo_gym.exporters, "export_metrics", export_metrics)

    _health_check_collected_rollouts(_saved_run(tmp_path), _HEALTH_CONFIG)

    assert _exported_keys(export_metrics)["health/timeouts/timed_out"] == 1


def test_a_failed_export_is_logged_and_does_not_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(nemo_gym.exporters, "export_metrics", MagicMock(side_effect=ConnectionError("tracking down")))

    _health_check_collected_rollouts(_saved_run(tmp_path), _HEALTH_CONFIG)

    assert "Could not export the run health timeouts" in caplog.text
    assert (tmp_path / QUALITY_SUMMARY_FILENAME).is_file()


def test_the_elapsed_stamp_changes_no_score_or_metric() -> None:
    rows = [
        {"reward": 1.0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "a"}},
        {"reward": 0.0, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "a"}},
    ]
    stamped = [{**row, ELAPSED_S_KEY: 12.5} for row in rows]

    assert compute_aggregate_metrics(stamped).model_dump() == compute_aggregate_metrics(rows).model_dump()


async def test_run_examples_output_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training reads run_examples: its rows carry no elapsed stamp and nothing of the health section."""
    reply = {"reward": 1.0, "response": {"output": []}}

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        return FakeResponse(200, dict(reply))

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    rows = [
        {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0} for i in range(2)
    ]

    results = [result for _, result in [await future for future in RolloutCollectionHelper().run_examples(rows)]]

    assert results == [reply, reply]
