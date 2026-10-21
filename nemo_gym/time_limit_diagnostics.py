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
"""Count the rollouts of a run that ran out of time, and show where their time went.

A rollout that runs out of time scores as a failure of the evaluated model, whatever made it slow.
This module reads what a run already recorded about each rollout (its failure labels, its timeout
flags, its wall clock and its captured model and tool calls) into a ``time_limit`` section of the
aggregate metrics. It only reads rows: it changes no classification, reward or row.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, NamedTuple, Optional

from nemo_gym.metrics_config import Stat
from nemo_gym.reward_profile import _stat


# Collector key with the `/run` wall clock of one rollout, success or failure, in seconds.
NG_ELAPSED_S_KEY = "_ng_elapsed_s"

# Gym's name for the model server under evaluation. Calls to any other model server (a judge, a user
# simulator) count as other-model calls.
POLICY_MODEL_SERVER_NAME = "policy_model"

# Labels that already say a rollout ran out of time.
_TIMEOUT_LABELS = frozenset({"timeout_exceeded", "agent_timeout", "transport_timeout"})
# The failure an environment server's own episode deadline records (BaseEnvironmentServer).
_EPISODE_DEADLINE_REASON = "Episode timed out"
# Agent flags that already say the agent, the task or a test run ran out of time.
_TIMEOUT_FLAGS = ("timed_out", "agent_timed_out", "_ng_agent_timed_out", "agent_timeout_error", "eval_timed_out")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def _quantile(values: list[float], stat: str) -> Optional[float]:
    return _round(_stat(values, stat)) if values else None


def rollout_elapsed_s(row: Mapping[str, Any]) -> Optional[float]:
    """The `/run` wall clock rollout collection stamped on this row, in seconds."""
    value = _number(row.get(NG_ELAPSED_S_KEY))
    return value if value is not None and value > 0 else None


def _policy_call_timed_out(row: Mapping[str, Any]) -> bool:
    capture = row.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, Mapping) else None
    return any(isinstance(call, Mapping) and call.get("error_category") == "timeout" for call in calls or [])


def ran_out_of_time(row: Mapping[str, Any]) -> bool:
    """Whether a label, flag or termination record the row already carries says it ran out of time.

    A failed rollout one of whose captured model calls timed out counts too.
    """
    failed = isinstance(row.get("_ng_failure_class"), str)
    labels = {row.get("_ng_failure_class"), row.get("failure_kind")}
    termination = row.get("termination")
    return (
        bool(labels & _TIMEOUT_LABELS)
        or (failed and row.get("_ng_failure_message") == _EPISODE_DEADLINE_REASON)
        or any(bool(row.get(name)) and _number(row.get(name)) != 0 for name in _TIMEOUT_FLAGS)
        # A sandboxed agent's own termination record, as Terminal-Bench writes it.
        or (isinstance(termination, Mapping) and termination.get("reason") == "timeout")
        or (failed and _policy_call_timed_out(row))
    )


class _Call(NamedTuple):
    start: Optional[float]
    end: Optional[float]
    duration_s: Optional[float]
    tokens_out: Optional[float]
    is_policy: bool


def _model_calls(row: Mapping[str, Any]) -> list[_Call]:
    capture = row.get("ng_model_call_capture")
    raw_calls = capture.get("calls") if isinstance(capture, Mapping) else None
    calls = []
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(raw, Mapping):
            continue
        model_ref = raw.get("model_ref")
        latency_ms = _number(raw.get("latency_total_ms"))
        calls.append(
            _Call(
                start=_number(raw.get("started_at")),
                end=_number(raw.get("completed_at")),
                duration_s=latency_ms / 1000 if latency_ms is not None else None,
                tokens_out=_number(raw.get("tokens_out")),
                # A call with no model_ref predates the field; the policy was then the only model captured.
                is_policy=not isinstance(model_ref, Mapping) or model_ref.get("name") == POLICY_MODEL_SERVER_NAME,
            )
        )
    return calls


def _tool_calls(row: Mapping[str, Any]) -> list[_Call]:
    trajectory = row.get("ng_trajectory")
    raw_calls = trajectory.get("tool_calls") if isinstance(trajectory, Mapping) else None
    calls = []
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(raw, Mapping):
            continue
        duration_ms = _number(raw.get("duration_ms"))
        calls.append(
            _Call(
                start=_number(raw.get("started_at")),
                end=_number(raw.get("completed_at")),
                duration_s=duration_ms / 1000 if duration_ms is not None else None,
                tokens_out=None,
                is_policy=False,
            )
        )
    return calls


def _tool_time_measured(row: Mapping[str, Any], calls: list[_Call]) -> bool:
    """Whether the row measures its tool time: a ``tool_calls`` list, possibly empty, whose calls all have a
    duration, and no reported gap. An empty list is a measured zero; a row without the list measures nothing."""
    trajectory = row.get("ng_trajectory")
    if not isinstance(trajectory, Mapping) or not isinstance(trajectory.get("tool_calls"), list):
        return False
    if any(
        isinstance(gap, Mapping) and gap.get("code") == "tool_timing_unavailable"
        for gap in trajectory.get("gaps") or []
    ):
        return False
    return all((c.start is not None and c.end is not None) or c.duration_s is not None for c in calls)


def _busy_s(calls: list[_Call]) -> float:
    """Time covered by at least one call: overlapping calls (parallel subagents) count once."""
    busy = sum(call.duration_s or 0.0 for call in calls if call.start is None or call.end is None)
    intervals = sorted(
        (c.start, c.end) for c in calls if c.start is not None and c.end is not None and c.end >= c.start
    )
    current: Optional[list[float]] = None
    for start, end in intervals:
        if current is not None and start <= current[1]:
            current[1] = max(current[1], end)
        else:
            if current is not None:
                busy += current[1] - current[0]
            current = [start, end]
    if current is not None:
        busy += current[1] - current[0]
    return busy


def _agent_window_end(row: Mapping[str, Any]) -> Optional[float]:
    """When the agent's execution window ended, as a Unix timestamp, if the row records it."""
    window = (row.get("timings") or {}).get("agent_execution") if isinstance(row.get("timings"), Mapping) else None
    finished = window.get("finished_at") if isinstance(window, Mapping) else None
    if isinstance(finished, str):
        try:
            return datetime.fromisoformat(finished.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return _number(finished)


def _unrecorded_s(row: Mapping[str, Any], events: list[_Call]) -> Optional[float]:
    """Agent time after the last recorded call: typically a call cancelled in flight at the deadline."""
    window_end = _agent_window_end(row)
    ends = [call.end for call in events if call.end is not None]
    if window_end is None or not ends:
        return None
    return max(0.0, window_end - max(ends))


def _share(part: float, whole: float) -> Optional[float]:
    return _round(part / whole) if whole > 0 else None


def _time_split(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the elapsed time of these rollouts went.

    Every share divides by the summed elapsed time of the rollouts that have both an elapsed stamp and
    model-call captures. A share is null when a rollout in that set does not report what it measures.
    """
    measured = [
        (row, elapsed, calls) for row in rows if (elapsed := rollout_elapsed_s(row)) and (calls := _model_calls(row))
    ]
    total = sum(elapsed for _, elapsed, _ in measured)
    policy = sum(min(_busy_s([c for c in calls if c.is_policy]), elapsed) for _, elapsed, calls in measured)
    other = sum(min(_busy_s([c for c in calls if not c.is_policy]), elapsed) for _, elapsed, calls in measured)
    tools = [_tool_calls(row) for row, _, _ in measured]
    tool = (
        sum(min(_busy_s(t), elapsed) for t, (_, elapsed, _) in zip(tools, measured))
        if measured and all(_tool_time_measured(row, t) for t, (row, _, _) in zip(tools, measured))
        else None
    )

    def model_time_known(is_policy: bool) -> bool:
        # A call with neither bounds nor a duration is unknown time; that category's share is then unknown too.
        return all(
            (c.start is not None and c.end is not None) or c.duration_s is not None
            for _, _, calls in measured
            for c in calls
            if c.is_policy == is_policy
        )

    unrecorded_parts = [_unrecorded_s(row, calls + t) for (row, _, calls), t in zip(measured, tools)]
    unrecorded = sum(unrecorded_parts) if measured and all(u is not None for u in unrecorded_parts) else None
    shares = {
        "policy_call_time_share": _share(policy, total) if model_time_known(True) else None,
        "other_model_call_time_share": _share(other, total) if model_time_known(False) else None,
        "tool_call_time_share": _share(tool, total) if tool is not None else None,
        "unrecorded_time_share": _share(unrecorded, total) if unrecorded is not None else None,
    }
    known = list(shares.values())
    shares["untracked_time_share"] = (
        _round(max(0.0, 1 - sum(known))) if measured and all(v is not None for v in known) else None
    )

    policy_calls = [c for _, _, calls in measured for c in calls if c.is_policy]
    # End to end: a call's duration includes its time in the server queue and in prefill.
    speeds = [c.tokens_out / c.duration_s for c in policy_calls if (c.tokens_out or 0) > 0 and (c.duration_s or 0) > 0]
    per_rollout = [float(sum(c.is_policy for c in calls)) for _, _, calls in measured]
    return {
        "rollouts_with_captures": len(measured),
        **shares,
        "policy_e2e_output_tokens_per_s_p50": _quantile(speeds, Stat.MEDIAN),
        # The slow tail of serving speed is its low end.
        "policy_e2e_output_tokens_per_s_p5": _quantile(speeds, "p5"),
        "policy_calls_per_rollout_p50": _quantile(per_rollout, Stat.MEDIAN),
        "policy_calls_per_rollout_p95": _quantile(per_rollout, "p95"),
    }


def summarize_time_limits(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the rollouts that ran out of time and summarize where their time went.

    ``rows`` are the final record of every rollout: the scored rows and the last failures-sidecar
    attempt of each rollout that never succeeded. Every field after ``timed_out_share`` describes the
    rollouts that ran out of time.
    """
    rows = list(rows)
    timed_out = [row for row in rows if ran_out_of_time(row)]
    elapsed = [value for value in map(rollout_elapsed_s, timed_out) if value is not None]
    return {
        "timed_out": len(timed_out),
        "timed_out_share": _round(len(timed_out) / len(rows)) if rows else None,
        "timed_out_elapsed_s_p50": _quantile(elapsed, Stat.MEDIAN),
        "timed_out_elapsed_s_p95": _quantile(elapsed, "p95"),
        **_time_split(timed_out),
    }
