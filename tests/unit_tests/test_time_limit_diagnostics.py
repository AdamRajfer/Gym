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
    def test_fields(self) -> None:
        rows = [rollout(100, timed_out=True), rollout(300, timed_out=True), rollout(10), rollout(20)]
        assert summarize_time_limits(rows) == {
            "timed_out": 2,
            "timed_out_share": 0.5,
            "timed_out_elapsed_s_p50": 200.0,
            "timed_out_elapsed_s_p95": 290.0,
        }

    def test_no_rollout_ran_out_of_time(self) -> None:
        assert summarize_time_limits([rollout(10)]) == {
            "timed_out": 0,
            "timed_out_share": 0.0,
            "timed_out_elapsed_s_p50": None,
            "timed_out_elapsed_s_p95": None,
        }

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
        assert metrics == {
            "time_limit/a/timed_out": 1,
            "time_limit/a/timed_out_share": 1.0,
            "time_limit/a/timed_out_elapsed_s_p50": 100.0,
            "time_limit/a/timed_out_elapsed_s_p95": 100.0,
        }

    def test_a_broken_section_never_fails_the_run(self, tmp_path: Path) -> None:
        fpath = _metrics_file(tmp_path, [{AGENT_REF_KEY_NAME: {"name": "a"}, NG_ENVIRONMENT_SERVER_KEY: "e"}])
        before = fpath.read_bytes()

        def broken(row):
            raise RuntimeError("boom")

        _write_time_limit_sections(fpath, [{}], broken)
        assert fpath.read_bytes() == before
        _write_time_limit_sections(None, [{}], broken)

    def test_a_write_that_fails_partway_leaves_the_metrics_file_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review finding 4339034971 on #4841: a failed diagnostic write must not truncate the aggregate metrics."""
        exported = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "export_metrics", exported)
        fpath = _metrics_file(tmp_path, [{AGENT_REF_KEY_NAME: {"name": "a"}, NG_ENVIRONMENT_SERVER_KEY: "e"}])
        before = fpath.read_bytes()
        write_bytes = Path.write_bytes

        def disk_full(self: Path, data: bytes) -> int:
            write_bytes(self, data[:10])
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(Path, "write_bytes", disk_full)
        _write_time_limit_sections(fpath, [rollout(100, **{NG_FAILURE_CLASS_KEY: "agent_run_error"})], lambda row: "e")
        monkeypatch.undo()

        assert fpath.read_bytes() == before
        assert [path.name for path in tmp_path.iterdir()] == [fpath.name]
        exported.assert_not_called()


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
