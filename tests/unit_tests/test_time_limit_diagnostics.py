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
from nemo_gym.health.types import QUALITY_SUMMARY_FILENAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _attach_trajectory_record,
)
from nemo_gym.rollout_health import health_check_run_dir
from nemo_gym.time_limit_diagnostics import (
    NG_ELAPSED_S_KEY,
    TimeLimitVerdict,
    classify_time_limit,
    flatten_for_export,
    rollout_elapsed_s,
    summarize_timeouts,
    timeout_view,
)
from tests.unit_tests.test_rollout_collection import (  # noqa: F401 - fixture
    FakeResponse,
    empty_global_config,
    install_fake_server_client,
)


def rollout(elapsed: float, **fields) -> dict:
    return {NG_ELAPSED_S_KEY: elapsed, "reward": 0.0, **fields}


def event(event_id: str = "e1", **fields) -> dict:
    """A timeout record as a component reports it."""
    return {"event_id": event_id, "enforcer": "gym.environment_server", "evidence": "internal", **fields}


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


def split(rows: list[dict]) -> dict:
    return summarize_timeouts(rows)["time_split"]


class TestElapsed:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ({NG_ELAPSED_S_KEY: 12.5}, 12.5),
            ({NG_ELAPSED_S_KEY: 0}, None),
            ({NG_ELAPSED_S_KEY: "12"}, None),
            ({NG_ELAPSED_S_KEY: True}, None),
            ({"elapsed_seconds": 12.5}, None),
            ({}, None),
        ],
    )
    def test_reads_only_the_collector_stamp(self, row: dict, expected: float | None) -> None:
        assert rollout_elapsed_s(row) == expected


class TestEvidenceOrder:
    """One function decides: a typed record first, then a legacy signal, then unclassified."""

    def test_a_typed_record_names_the_limit_and_its_enforcer(self) -> None:
        row = rollout(60, timeouts=[event(limit="default_episode_timeout_seconds", configured_s=60)])
        verdict = classify_time_limit(row)
        assert verdict.evidence == "typed"
        assert (verdict.limit, verdict.enforcer) == ("default_episode_timeout_seconds", "gym.environment_server")

    def test_a_typed_record_wins_over_a_legacy_label(self) -> None:
        row = rollout(
            60,
            **{NG_FAILURE_CLASS_KEY: "timeout_exceeded"},
            ng_trajectory={"tool_calls": [{"timeouts": [event(limit="step_timeout_sec", enforcer="gym.shell")]}]},
        )
        assert classify_time_limit(row)[:3] == ("typed", "step_timeout_sec", "gym.shell")

    def test_a_typed_record_without_a_limit_name_keeps_it_unknown(self) -> None:
        row = rollout(60, timeouts=[event(enforcer="aws.codebuild", evidence="external")])
        assert classify_time_limit(row)[:3] == ("typed", None, "aws.codebuild")

    def test_an_event_forwarded_onto_several_records_counts_once(self) -> None:
        shared = event("same", limit="llm_request_timeout")
        row = rollout(60, timeouts=[shared], ng_model_call_capture={"calls": [{"timeouts": [shared]}]})
        assert len(classify_time_limit(row).events) == 1

    def test_an_invalid_event_is_not_evidence_but_is_counted(self) -> None:
        row = rollout(60, **{NG_FAILURE_CLASS_KEY: "timeout_exceeded"}, timeouts=[{"event_id": "x"}])
        assert classify_time_limit(row).evidence == "legacy"
        assert summarize_timeouts([row])["invalid_events"] == 1

    @pytest.mark.parametrize(
        ("fields", "limit"),
        [
            ({NG_FAILURE_CLASS_KEY: "timeout_exceeded"}, "_ng_failure_class=timeout_exceeded"),
            ({NG_FAILURE_CLASS_KEY: "agent_timeout"}, "_ng_failure_class=agent_timeout"),
            ({"failure_kind": "transport_timeout"}, "failure_kind=transport_timeout"),
            (
                {NG_FAILURE_CLASS_KEY: "agent_run_error", "_ng_failure_message": "Episode timed out"},
                "episode_deadline",
            ),
            ({"timed_out": True}, "timed_out"),
            ({"agent_timed_out": True}, "agent_timed_out"),
            ({"eval_timed_out": 1}, "eval_timed_out"),
            ({"termination": {"reason": "timeout"}}, "termination.reason=timeout"),
            (
                {
                    NG_FAILURE_CLASS_KEY: "agent_run_error",
                    "ng_model_call_capture": {"calls": [timed_out_policy_call()]},
                },
                "model_call.error_category=timeout",
            ),
        ],
    )
    def test_signals_rows_already_carry_are_legacy_evidence(self, fields: dict, limit: str) -> None:
        assert classify_time_limit(rollout(100, **fields)) == TimeLimitVerdict("legacy", limit, None)

    def test_a_failure_that_reached_its_limit_with_no_signal_is_unclassified(self) -> None:
        # The invocation measured its interval past the configured limit, but nothing reports a firing.
        row = rollout(
            700,
            **{NG_FAILURE_CLASS_KEY: "agent_run_error"},
            ng_trajectory={"invocations": [{"configured_s": 600, "elapsed_s": 650}]},
        )
        assert classify_time_limit(row) == TimeLimitVerdict("unclassified", None, None)
        section = summarize_timeouts([row])
        assert section["by_evidence"] == {"typed": 0, "legacy": 0, "unclassified": 1}
        assert section["failed_without_signal"] == 0

    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"timed_out": False, "agent_timed_out": 0},
            {"termination": {"reason": "completed"}},
            # A model call can time out and be retried inside a rollout that then succeeds.
            {"ng_model_call_capture": {"calls": [timed_out_policy_call()]}},
            # Elapsed past a limit on a row that succeeded is not a timeout.
            {"ng_trajectory": {"invocations": [{"configured_s": 10, "elapsed_s": 20}]}},
        ],
    )
    def test_rows_that_did_not_run_out_of_time(self, fields: dict) -> None:
        assert classify_time_limit(rollout(100, **fields)) is None

    def test_a_failure_with_no_signal_is_counted_apart_never_as_not_timed_out(self) -> None:
        section = summarize_timeouts([rollout(10, **{NG_FAILURE_CLASS_KEY: "agent_run_error"}), rollout(10)])
        assert (section["timed_out"], section["failed_without_signal"]) == (0, 1)


class TestSection:
    def test_fields(self) -> None:
        rows = [
            rollout(60, timeouts=[event("a", limit="default_episode_timeout_seconds")]),
            rollout(100, timeouts=[event("b", limit="default_episode_timeout_seconds")]),
            rollout(80, **{NG_FAILURE_CLASS_KEY: "timeout_exceeded"}),
            rollout(5),
        ]
        section = summarize_timeouts(rows)
        assert {k: v for k, v in section.items() if k != "time_split"} == {
            "rollouts": 4,
            "timed_out": 3,
            "by_evidence": {"typed": 2, "legacy": 1, "unclassified": 0},
            "by_limit": {"_ng_failure_class=timeout_exceeded": 1, "default_episode_timeout_seconds": 2},
            "by_enforcer": {"gym.environment_server": 2},
            "events": 2,
            "invalid_events": 0,
            "failed_without_signal": 0,
            "elapsed_s": {"p50": 80.0, "p95": 98.0},
        }

    def test_no_rows(self) -> None:
        section = summarize_timeouts([])
        assert (section["rollouts"], section["timed_out"], section["elapsed_s"]) == (0, 0, {"p50": None, "p95": None})
        assert section["time_split"]["share"]["policy_model"] is None

    def test_the_view_a_worker_returns_gives_the_same_section_as_the_row(self) -> None:
        rows = [
            timed_out(100, [call(0, 40, tokens_out=400)], [tool(40, 45)], window_end=60),
            rollout(30, **{NG_FAILURE_CLASS_KEY: "agent_run_error"}, response={"output": ["large payload"] * 50}),
            rollout(5),
        ]
        views = [view for row in rows if (view := timeout_view(row)) is not None]
        assert len(views) == 2
        assert "response" not in views[1]
        assert summarize_timeouts(views, rollouts=len(rows)) == summarize_timeouts(rows)


class TestExport:
    def test_numeric_leaves_become_flat_keys_and_nulls_are_left_out(self) -> None:
        section = summarize_timeouts([rollout(60, timeouts=[event(limit="llm_request_timeout")])])
        flat = flatten_for_export(section)
        assert flat["health/timeouts/timed_out"] == 1
        assert flat["health/timeouts/by_evidence/typed"] == 1
        assert flat["health/timeouts/by_limit/llm_request_timeout"] == 1
        assert flat["health/timeouts/elapsed_s/p50"] == 60.0
        assert "health/timeouts/time_split/share/policy_model" not in flat
        assert all(isinstance(value, (int, float)) for value in flat.values())


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
        assert split(rows) == {
            "rollouts_with_captures": 2,
            "share": {"policy_model": 0.7, "other_model": 0.0, "tool": 0.05, "unrecorded": 0.1, "untracked": 0.15},
            "policy_output_tokens_per_s": {"p50": 15.0, "p5": 10.0},
            "policy_calls_per_rollout": {"p50": 2.0, "p95": 2.0},
        }

    def test_setup_before_the_first_call_is_not_unrecorded(self) -> None:
        share = split([timed_out(400, [call(300, 390)], [tool(390, 400)], window_end=400)])["share"]
        assert share["unrecorded"] == 0.0
        assert share["untracked"] == round(1 - 90 / 400 - 10 / 400, 4)

    def test_time_after_the_last_call_is_unrecorded(self) -> None:
        # The call in flight at the deadline was cancelled before the capture wrote it.
        row = timed_out(480, [call(0, 150)], [tool(150, 151)], window_end=421)
        assert split([row])["share"]["unrecorded"] == round(270 / 480, 4)

    def test_a_call_after_the_agent_window_does_not_hide_its_tail(self) -> None:
        # A judge call after the window ends is not agent activity.
        share = split([timed_out(100, [call(0, 10), call(70, 90, model="judge_model")], [], window_end=60)])["share"]
        assert (share["unrecorded"], share["other_model"], share["untracked"]) == (0.5, 0.2, 0.2)

    def test_a_call_spanning_the_window_end_leaves_no_tail(self) -> None:
        assert split([timed_out(100, [call(40, 80)], [], window_end=60)])["share"]["unrecorded"] == 0.0

    def test_no_window_end_means_unknown_unrecorded_time(self) -> None:
        share = split([timed_out(480, [call(0, 150)], [tool(150, 151)])])["share"]
        assert (share["unrecorded"], share["untracked"]) == (None, None)

    def test_an_empty_tool_call_list_is_zero_tool_time(self) -> None:
        rows = [
            timed_out(100, [call(0, 50)], [], window_end=60),
            timed_out(100, [call(0, 50)], [tool(50, 60)], window_end=60),
        ]
        share = split(rows)["share"]
        assert (share["tool"], share["unrecorded"], share["untracked"]) == (0.05, 0.05, 0.4)

    def test_a_trajectory_the_collector_projected_without_tool_evidence_has_no_tool_share(self) -> None:
        row = {**timed_out(100, [call(0, 50)], window_end=60), TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        _attach_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, row)
        assert row["ng_trajectory"]["tool_calls"] == []
        assert split([row])["share"]["tool"] is None

    @pytest.mark.parametrize(
        "tools",
        [None, [{"tool_call_id": "a"}, {"tool_call_id": "b"}]],
        ids=["no_trajectory", "tools_without_timing"],
    )
    def test_unmeasured_tool_time_has_no_tool_share(self, tools: list | None) -> None:
        assert split([timed_out(100, [call(0, 50)], tools, window_end=60)])["share"]["tool"] is None

    def test_reported_tool_timing_gap_means_no_tool_share(self) -> None:
        row = timed_out(100, [call(0, 50)], [tool(50, 60)], window_end=60)
        row["ng_trajectory"]["gaps"] = [{"code": "tool_timing_unavailable"}]
        assert split([row])["share"]["tool"] is None

    def test_a_tool_end_time_alone_still_bounds_unrecorded_time(self) -> None:
        share = split([timed_out(480, [call(0, 150)], [{"completed_at": T0 + 151}], window_end=421)])["share"]
        assert (share["tool"], share["unrecorded"]) == (None, round(270 / 480, 4))

    def test_overlapping_calls_count_once_and_never_exceed_elapsed(self) -> None:
        assert split([timed_out(50, [call(0, 40), call(20, 60)])])["share"]["policy_model"] == 1.0
        assert split([timed_out(100, [call(0, 40), call(20, 60)])])["share"]["policy_model"] == 0.6

    def test_a_call_without_timestamps_counts_its_duration(self) -> None:
        raw = {"model_ref": {"name": "policy_model"}, "latency_total_ms": 30000, "tokens_out": 300}
        section = split([timed_out(60, [raw])])
        assert section["share"]["policy_model"] == 0.5
        assert section["policy_output_tokens_per_s"]["p50"] == 10.0

    def test_judge_calls_are_other_model_calls(self) -> None:
        section = split([timed_out(100, [call(0, 10), call(10, 70, model="judge_model")])])
        assert (section["share"]["policy_model"], section["share"]["other_model"]) == (0.1, 0.6)
        assert section["policy_calls_per_rollout"]["p50"] == 1.0

    def test_an_untimed_call_makes_only_its_own_share_unknown(self) -> None:
        share = split([timed_out(100, [timed_out_policy_call(), call(0, 20, model="judge_model")])])["share"]
        assert (share["policy_model"], share["other_model"], share["untracked"]) == (None, 0.2, None)
        untimed_judge = {**timed_out_policy_call(), "model_ref": {"name": "judge_model"}}
        share = split([timed_out(100, [call(0, 30), untimed_judge])])["share"]
        assert (share["policy_model"], share["other_model"]) == (0.3, None)

    def test_a_call_without_model_ref_is_the_policy(self) -> None:
        raw = call(0, 60)
        raw.pop("model_ref")
        assert split([timed_out(100, [raw])])["share"]["policy_model"] == 0.6


# --- End to end: collection writes the section into the health summary, and a recheck gives the same -----------


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


_REPLIES = [
    {"reward": 1.0},
    # Typed: the component reports the limit that fired.
    {"reward": 0.0, "timeouts": [event(limit="default_episode_timeout_seconds", configured_s=60)]},
    # Legacy: a failure label only.
    {NG_FAILURE_CLASS_KEY: "timeout_exceeded", "error": "agent ran out of time"},
]


def _fake_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(json.loads(json.dumps(_REPLIES)))

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        if url_path == "/run":
            return FakeResponse(200, next(replies))
        return FakeResponse(200, compute_aggregate_metrics([dict(r) for r in json.verify_responses]).model_dump())

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))


async def test_collection_writes_the_section_to_the_health_summary_and_a_recheck_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_global_config
) -> None:
    _fake_servers(monkeypatch)
    exported = MagicMock()
    monkeypatch.setattr(nemo_gym.rollout_collection, "export_metrics", exported)
    output = tmp_path / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(_input(tmp_path, 3)), output_jsonl_fpath=str(output), num_samples_in_parallel=1
    )

    await RolloutCollectionHelper().run_from_config(config)

    summary = orjson.loads((tmp_path / QUALITY_SUMMARY_FILENAME).read_bytes())
    timeouts = summary["run"]["stats"]["timeouts"]
    assert (timeouts["rollouts"], timeouts["timed_out"]) == (3, 2)
    assert timeouts["by_evidence"] == {"typed": 1, "legacy": 1, "unclassified": 0}
    assert timeouts["by_limit"] == {"_ng_failure_class=timeout_exceeded": 1, "default_episode_timeout_seconds": 1}
    # Aggregate metrics stay scores only, and the score is what it was: the failure row is not in it.
    [entry] = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
    assert not any("time" in key and "limit" in key for key in entry)
    assert entry["key_metrics"]["mean/reward"] == 0.5
    # Tracking servers get the section as flat keys.
    sent = {key: value for c in exported.call_args_list for key, value in c.args[0].items()}
    assert sent["health/timeouts/timed_out"] == 2
    assert sent["health/timeouts/by_evidence/typed"] == 1

    recheck = health_check_run_dir(tmp_path)

    assert recheck.summary["run"]["stats"]["timeouts"] == timeouts


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

    summary = orjson.loads((tmp_path / QUALITY_SUMMARY_FILENAME).read_bytes())
    assert summary["run"]["stats"]["timeouts"] == after_run
    [entry] = orjson.loads((tmp_path / "rollouts_aggregate_metrics.json").read_bytes())
    assert entry["key_metrics"]["mean/reward"] == 0.5


def test_the_elapsed_stamp_changes_no_score_or_metric() -> None:
    """Rows now carry the `/run` wall clock; aggregation must read the same scores from them."""
    rows = [
        {"reward": 1.0, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "a"}},
        {"reward": 0.0, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, AGENT_REF_KEY_NAME: {"name": "a"}},
    ]
    stamped = [{**row, NG_ELAPSED_S_KEY: 12.5} for row in rows]

    assert compute_aggregate_metrics(stamped).model_dump() == compute_aggregate_metrics(rows).model_dump()


async def test_run_examples_output_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training reads run_examples: its rows carry no elapsed stamp and nothing of the health section."""
    reply = {"reward": 1.0, "response": {"output": []}}

    async def post(server_name: str, url_path: str, json, **kwargs) -> FakeResponse:
        return FakeResponse(200, dict(reply))

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    rows = [
        {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "x": i}
        for i in range(2)
    ]

    results = [result for _, result in [await future for future in RolloutCollectionHelper().run_examples(rows)]]

    assert results == [reply, reply]
