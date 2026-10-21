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
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

import nemo_gym.rollout_collection
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    NG_ENVIRONMENT_SERVER_KEY,
    NG_FAILURE_CLASS_KEY,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _write_time_limit_sections,
)
from nemo_gym.time_limit_diagnostics import NG_ELAPSED_S_KEY, ran_out_of_time, rollout_elapsed_s, summarize_time_limits
from tests.unit_tests.test_rollout_collection import (  # noqa: F401 - fixture
    FakeResponse,
    empty_global_config,
    install_fake_server_client,
)


def rollout(elapsed: float, **fields) -> dict:
    return {NG_ELAPSED_S_KEY: elapsed, "reward": 0.0, **fields}


def timed_out_policy_call() -> dict:
    return {"model_ref": {"type": "responses_api_models", "name": "policy_model"}, "error_category": "timeout"}


T0 = 1_000_000.0  # Unix time the rollout's first call starts.


def call(start: float, end: float, *, model: str = "policy_model", tokens_out: int | None = 100) -> dict:
    return {
        "model_ref": {"type": "responses_api_models", "name": model},
        "started_at": T0 + start,
        "completed_at": T0 + end,
        "latency_total_ms": (end - start) * 1000,
        "tokens_out": tokens_out,
    }


def tool(start: float, end: float) -> dict:
    return {"started_at": T0 + start, "completed_at": T0 + end, "duration_ms": (end - start) * 1000}


def timed_out(
    elapsed: float,
    calls: list[dict],
    tools: list[dict] | None = None,
    window_end: float | None = None,
    window_start: float = 0.0,
):
    """A rollout that ran out of its time budget, with its captured calls and, optionally, timed tool calls."""
    row = rollout(elapsed, termination={"reason": "timeout"}, ng_model_call_capture={"calls": calls})
    if tools is not None:
        row["ng_trajectory"] = {"tool_calls": tools, "gaps": []}
    if window_end is not None:
        row["timings"] = {"agent_execution": {"started_at": T0 + window_start, "finished_at": T0 + window_end}}
    return row


class TestElapsed:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ({NG_ELAPSED_S_KEY: 12.5}, 12.5),
            ({NG_ELAPSED_S_KEY: 0}, None),
            ({NG_ELAPSED_S_KEY: True}, None),
            # Agent-reported elapsed time is not read: only the collector's own wall clock is.
            ({"elapsed_seconds": 7.0}, None),
            ({}, None),
        ],
        ids=["stamp", "zero", "bool", "agent field ignored", "absent"],
    )
    def test_reads_only_the_collector_stamp(self, row: dict, expected: float | None) -> None:
        assert rollout_elapsed_s(row) == expected


class TestDetection:
    @pytest.mark.parametrize(
        "fields",
        [
            {NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
            {"failure_kind": "agent_timeout"},
            {"failure_kind": "transport_timeout"},
            {"timed_out": True},
            {"agent_timed_out": 1},
            {"_ng_agent_timed_out": True},
            {"agent_timeout_error": 1},
            {"eval_timed_out": True},
            {NG_FAILURE_CLASS_KEY: "environment_server_failure", "_ng_failure_message": "Episode timed out"},
            {"termination": {"reason": "timeout", "detail": None}},
            {NG_FAILURE_CLASS_KEY: "agent_run_error", "ng_model_call_capture": {"calls": [timed_out_policy_call()]}},
        ],
    )
    def test_labels_the_row_already_carries(self, fields: dict) -> None:
        assert ran_out_of_time(rollout(10, **fields))

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"timed_out": False, "agent_timeout_error": 0},
            {"termination": {"reason": "completed"}},
            {NG_FAILURE_CLASS_KEY: "environment_server_failure", "_ng_failure_message": "boom"},
            # A call that timed out inside a rollout the agent finished does not end the rollout.
            {"ng_model_call_capture": {"calls": [timed_out_policy_call()]}},
        ],
        ids=["nothing", "false flags", "completed termination", "other failure", "finished rollout"],
    )
    def test_rows_that_did_not_run_out_of_time(self, fields: dict) -> None:
        assert not ran_out_of_time(rollout(10, **fields))


class TestSummary:
    def test_counts_and_elapsed_without_captures(self) -> None:
        rows = [rollout(100, timed_out=True), rollout(300, timed_out=True), rollout(10), rollout(20)]
        section = summarize_time_limits(rows)
        assert section["timed_out"] == 2
        assert section["timed_out_share"] == 0.5
        assert section["timed_out_elapsed_s_p50"] == 200.0
        assert section["timed_out_elapsed_s_p95"] == 290.0
        # No captures: nothing to split.
        assert section["rollouts_with_captures"] == 0
        assert section["policy_call_time_share"] is None
        assert section["untracked_time_share"] is None

    def test_no_rollout_ran_out_of_time(self) -> None:
        section = summarize_time_limits([rollout(10, ng_model_call_capture={"calls": [call(0, 5)]})])
        assert section["timed_out"] == 0
        assert section["timed_out_elapsed_s_p50"] is None
        assert section["rollouts_with_captures"] == 0

    def test_no_rows(self) -> None:
        assert summarize_time_limits([])["timed_out_share"] is None


def _metrics_file(tmp_path: Path, entries: list[dict]) -> Path:
    fpath = tmp_path / "m.json"
    fpath.write_bytes(orjson.dumps(entries))
    return fpath


class TestWriteSections:
    def test_reads_captures_back_for_rows_that_carry_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure row is written without its captures; the store still holds the call that timed out."""
        merge = MagicMock(
            side_effect=lambda record, dirs: {**record, "ng_model_call_capture": {"calls": [timed_out_policy_call()]}}
        )
        monkeypatch.setattr(nemo_gym.rollout_collection, "merge_model_call_capture_into_record", merge)
        fpath = _metrics_file(tmp_path, [{AGENT_REF_KEY_NAME: {"name": "a"}, NG_ENVIRONMENT_SERVER_KEY: "e"}])
        failed = rollout(100, **{NG_FAILURE_CLASS_KEY: "agent_run_error"})
        captured = rollout(10, ng_model_call_capture={"calls": []})

        _write_time_limit_sections(fpath, [failed, captured], lambda row: "e", [tmp_path])

        assert merge.call_count == 1  # only the row without captures is read back
        assert "ng_model_call_capture" not in failed  # the caller's row is not changed
        [entry] = orjson.loads(fpath.read_bytes())
        assert entry["time_limit"]["timed_out"] == 1

    def test_adds_a_section_per_server_and_exports_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        exported = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "export_metrics", exported)
        fpath = _metrics_file(
            tmp_path,
            [
                {AGENT_REF_KEY_NAME: {"name": "a"}, NG_ENVIRONMENT_SERVER_KEY: "env_a", "key_metrics": {"m": 1}},
                {AGENT_REF_KEY_NAME: {"name": "b"}, NG_ENVIRONMENT_SERVER_KEY: "env_b", "aggregation_error": {}},
            ],
        )
        rows = [
            rollout(100, timed_out=True, **{NG_ENVIRONMENT_SERVER_KEY: "env_a"}),
            rollout(10, **{NG_ENVIRONMENT_SERVER_KEY: "env_b"}),
        ]
        _write_time_limit_sections(fpath, rows, lambda row: row.get(NG_ENVIRONMENT_SERVER_KEY))

        written = orjson.loads(fpath.read_bytes())
        assert written[0]["key_metrics"] == {"m": 1}
        assert written[0]["time_limit"]["timed_out"] == 1
        assert "time_limit" not in written[1]
        [[metrics], _] = exported.call_args
        assert metrics == {f"time_limit/a/{key}": value for key, value in written[0]["time_limit"].items()}
        assert metrics["time_limit/a/timed_out"] == 1
        assert metrics["time_limit/a/timed_out_elapsed_s_p50"] == 100.0

    def test_a_broken_section_never_fails_the_run(self, tmp_path: Path) -> None:
        fpath = _metrics_file(tmp_path, [{AGENT_REF_KEY_NAME: {"name": "a"}, NG_ENVIRONMENT_SERVER_KEY: "e"}])
        before = fpath.read_bytes()

        def broken(row):
            raise RuntimeError("boom")

        _write_time_limit_sections(fpath, [{}], broken)
        assert fpath.read_bytes() == before
        _write_time_limit_sections(None, [{}], broken)


def _input(tmp_path: Path, count: int) -> Path:
    fpath = tmp_path / "input.jsonl"
    fpath.write_text(
        "".join(
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "my_agent"}, "i": i})
            + "\n"
            for i in range(count)
        )
    )
    return fpath


async def test_collection_and_aggregate_write_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config
) -> None:
    """gym eval run stamps elapsed time on scored and failed rows; both commands write the section."""
    replies = iter(
        [
            FakeResponse(200, {"reward": 1.0}),
            FakeResponse(200, {"reward": 0.0, "timed_out": True}),
            FakeResponse(200, {NG_FAILURE_CLASS_KEY: "timeout_exceeded", "error": "agent ran out of time"}),
        ]
    )

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        if url_path == "/run":
            return next(replies)
        return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    output = tmp_path / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(_input(tmp_path, 3)),
        output_jsonl_fpath=str(output),
        disable_health_check=True,
        num_samples_in_parallel=1,
    )
    await RolloutCollectionHelper().run_from_config(config)

    rows = [orjson.loads(line) for line in output.read_bytes().splitlines()]
    failures = [orjson.loads(line) for line in (tmp_path / "rollouts_failures.jsonl").read_bytes().splitlines()]
    assert all(row[NG_ELAPSED_S_KEY] >= 0 for row in rows + failures)

    [entry] = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
    assert entry["time_limit"]["timed_out"] == 2
    assert entry["time_limit"]["timed_out_share"] == round(2 / 3, 4)
    # The score is computed as before: the failure row is not part of it.
    assert entry["key_metrics"]["mean/reward"] == 0.5

    (tmp_path / "rollouts_aggregate_metrics.json").unlink()
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(input_glob=str(output), output_jsonl_fpath=str(output), disable_health_check=True)
    )
    [entry] = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
    assert entry["time_limit"]["timed_out"] == 2


async def test_run_examples_output_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training reads run_examples: its rows carry no elapsed stamp and no time-limit section."""
    reply = {"reward": 1.0, "response": {"output": []}}

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        return FakeResponse(200, dict(reply))

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    rows = [
        {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "x": i}
        for i in range(2)
    ]
    summarize = AsyncMock()
    monkeypatch.setattr(nemo_gym.rollout_collection, "summarize_time_limits", summarize)
    monkeypatch.setattr(nemo_gym.rollout_collection, "_write_time_limit_sections", summarize)

    results = [result for _, result in [await future for future in RolloutCollectionHelper().run_examples(rows)]]

    assert results == [reply, reply]
    summarize.assert_not_called()


class TestTimeSplit:
    def test_every_field(self) -> None:
        # 100 s rollouts: policy 0-40 and 40-70 s (10 and 20 tokens/s), a tool 70-75 s, the agent window
        # ends at 85 s; the other 15 s are setup and verification, which no field names.
        rows = [
            timed_out(
                100,
                [call(0, 40, tokens_out=400), call(40, 70, tokens_out=600)],
                [tool(70, 75)],
                window_end=85,
            )
            for _ in range(2)
        ] + [rollout(10, ng_model_call_capture={"calls": [call(0, 1)]})]
        assert summarize_time_limits(rows) == {
            "timed_out": 2,
            "timed_out_share": round(2 / 3, 4),
            "timed_out_elapsed_s_p50": 100.0,
            "timed_out_elapsed_s_p95": 100.0,
            "rollouts_with_captures": 2,
            "policy_call_time_share": 0.7,
            "other_model_call_time_share": 0.0,
            "tool_call_time_share": 0.05,
            "unrecorded_time_share": 0.1,
            "untracked_time_share": 0.15,
            "policy_e2e_output_tokens_per_s_p50": 15.0,
            "policy_e2e_output_tokens_per_s_p5": 10.0,
            "policy_calls_per_rollout_p50": 2.0,
            "policy_calls_per_rollout_p95": 2.0,
        }

    def test_setup_before_the_first_call_is_not_unrecorded(self) -> None:
        # 300 s of setup, then calls until the window ends: nothing after the last call is unrecorded.
        row = timed_out(400, [call(300, 390)], [tool(390, 400)], window_end=400)
        section = summarize_time_limits([row])
        assert section["unrecorded_time_share"] == 0.0
        assert section["untracked_time_share"] == round(1 - 90 / 400 - 10 / 400, 4)

    def test_time_after_the_last_call_is_unrecorded(self) -> None:
        # The call in flight at the deadline was cancelled before the capture wrote it.
        row = timed_out(480, [call(0, 150)], [tool(150, 151)], window_end=421)
        assert summarize_time_limits([row])["unrecorded_time_share"] == round(270 / 480, 4)

    def test_no_window_end_means_unknown_unrecorded_time(self) -> None:
        row = timed_out(480, [call(0, 150)], [tool(150, 151)])
        section = summarize_time_limits([row])
        assert section["unrecorded_time_share"] is None
        assert section["untracked_time_share"] is None

    def test_an_empty_tool_call_list_is_zero_tool_time(self) -> None:
        # The agent reported its tool calls and there were none: a measured zero, not an unknown share.
        rows = [
            timed_out(100, [call(0, 50)], [], window_end=60),
            timed_out(100, [call(0, 50)], [tool(50, 60)], window_end=60),
        ]
        section = summarize_time_limits(rows)
        assert section["tool_call_time_share"] == 0.05
        # policy 100 of 200 s, tool 10 s, after the last call 10 s (first rollout only): 80 s untracked.
        assert section["unrecorded_time_share"] == 0.05
        assert section["untracked_time_share"] == 0.4

    def test_a_row_with_no_trajectory_has_no_tool_share(self) -> None:
        section = summarize_time_limits([timed_out(100, [call(0, 50)], window_end=60)])
        assert section["tool_call_time_share"] is None
        assert section["untracked_time_share"] is None

    def test_reported_tool_timing_gap_means_no_tool_share(self) -> None:
        row = timed_out(100, [call(0, 50)], [tool(50, 60)], window_end=60)
        row["ng_trajectory"]["gaps"] = [{"code": "tool_timing_unavailable"}]
        assert summarize_time_limits([row])["tool_call_time_share"] is None

    def test_tool_calls_without_any_timing_have_no_tool_share(self) -> None:
        # A harness that records which tools ran but not when: no measured tool time, so no zero share.
        row = timed_out(100, [call(0, 50)], [{"tool_call_id": "a"}, {"tool_call_id": "b"}], window_end=60)
        assert summarize_time_limits([row])["tool_call_time_share"] is None

    def test_a_tool_end_time_alone_still_bounds_unrecorded_time(self) -> None:
        # Only completed_at is known: no tool share, but the tool's end is the last recorded event.
        row = timed_out(480, [call(0, 150)], [{"completed_at": T0 + 151}], window_end=421)
        section = summarize_time_limits([row])
        assert section["tool_call_time_share"] is None
        assert section["unrecorded_time_share"] == round(270 / 480, 4)

    def test_one_rollout_without_tool_timing_makes_the_share_unknown(self) -> None:
        rows = [timed_out(100, [call(0, 50)], [tool(50, 60)], window_end=60), timed_out(100, [call(0, 50)])]
        assert summarize_time_limits(rows)["tool_call_time_share"] is None

    def test_overlapping_calls_count_once_and_never_exceed_elapsed(self) -> None:
        # Two parallel subagent calls cover 0-60 s together.
        assert summarize_time_limits([timed_out(50, [call(0, 40), call(20, 60)])])["policy_call_time_share"] == 1.0
        assert summarize_time_limits([timed_out(100, [call(0, 40), call(20, 60)])])["policy_call_time_share"] == 0.6

    def test_a_call_without_timestamps_counts_its_duration(self) -> None:
        raw = {"model_ref": {"name": "policy_model"}, "latency_total_ms": 30000, "tokens_out": 300}
        section = summarize_time_limits([timed_out(60, [raw])])
        assert section["policy_call_time_share"] == 0.5
        assert section["policy_e2e_output_tokens_per_s_p50"] == 10.0

    def test_judge_calls_are_other_model_calls(self) -> None:
        section = summarize_time_limits([timed_out(100, [call(0, 10), call(10, 70, model="judge_model")])])
        assert section["policy_call_time_share"] == 0.1
        assert section["other_model_call_time_share"] == 0.6
        assert section["policy_calls_per_rollout_p50"] == 1.0

    def test_a_policy_call_without_timing_makes_only_the_policy_share_unknown(self) -> None:
        """Review finding 4338902666 on #4842: an untimed call is unknown time, not zero time."""
        section = summarize_time_limits([timed_out(100, [timed_out_policy_call(), call(0, 20, model="judge_model")])])
        assert section["policy_call_time_share"] is None
        assert section["other_model_call_time_share"] == 0.2
        assert section["untracked_time_share"] is None

    def test_a_judge_call_without_timing_makes_only_the_other_model_share_unknown(self) -> None:
        untimed_judge = {**timed_out_policy_call(), "model_ref": {"name": "judge_model"}}
        section = summarize_time_limits([timed_out(100, [call(0, 30), untimed_judge])])
        assert section["policy_call_time_share"] == 0.3
        assert section["other_model_call_time_share"] is None

    def test_a_call_without_model_ref_is_the_policy(self) -> None:
        raw = call(0, 60)
        raw.pop("model_ref")
        assert summarize_time_limits([timed_out(100, [raw])])["policy_call_time_share"] == 0.6
