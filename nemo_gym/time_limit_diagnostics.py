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
"""Which rollouts of a run ran out of time, on what evidence, and where their time went.

A run-health diagnostic: ``gym eval health-check`` writes it to ``quality_summary.json`` as
``run.stats.timeouts`` and recomputes it from the saved rows. It only reads rows; it changes no
classification, reward, score or row.

:func:`classify_time_limit` is the one place that decides whether a row ran out of time. Evidence, in order:

1. ``typed``: a :class:`TimeoutEvent` on the row or on an operation record it carries (an agent
   invocation, a tool call, a model call). A component that reports its own timeouts this way is trusted
   over any label.
2. ``legacy``: a failure label, agent flag, termination reason, episode-deadline message or timed-out
   model call that rows already carry. Nothing ties these names to the components that set them, so they
   are reported apart from typed evidence.
3. ``unclassified``: no signal says the row timed out, yet a typed record shows its measured interval
   reached its configured limit. Timing never confirms a firing, so such a row is counted on its own and
   never as "not timed out".
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Iterator, Literal, Mapping, NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.metrics_config import Stat
from nemo_gym.reward_profile import _stat


# Collector key with the `/run` wall clock of one rollout, success or failure, in seconds.
NG_ELAPSED_S_KEY = "_ng_elapsed_s"

# Gym's name for the model server under evaluation. Calls to any other model server (a judge, a user
# simulator) count as other-model calls.
POLICY_MODEL_SERVER_NAME = "policy_model"

# Labels that rows already carry when they ran out of time.
_TIMEOUT_LABELS = frozenset({"timeout_exceeded", "agent_timeout", "transport_timeout"})
# The failure an environment server's own episode deadline records (BaseEnvironmentServer).
_EPISODE_DEADLINE_REASON = "Episode timed out"
# Agent flags that say the agent, the task or a test run ran out of time.
_TIMEOUT_FLAGS = ("timed_out", "agent_timed_out", "_ng_agent_timed_out", "agent_timeout_error", "eval_timed_out")

TimeLimitEvidence = Literal["typed", "legacy", "unclassified"]


class TimeoutEvent(BaseModel):
    """One confirmed firing of one time limit, as a component reports it on the record it bounds.

    The shape a shared timeout record takes in Gym: an event's presence means the limit fired, and the
    enforcer is required. Unknown measurements stay null.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    limit: Optional[str] = Field(default=None, min_length=1)
    enforcer: str = Field(min_length=1)
    evidence: Literal["internal", "external"]
    configured_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    elapsed_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    fired_at: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)


class TimeLimitVerdict(NamedTuple):
    """Why a row counts as having run out of time; ``limit`` and ``enforcer`` are None when unknown."""

    evidence: TimeLimitEvidence
    limit: Optional[str]
    enforcer: Optional[str]
    events: tuple[TimeoutEvent, ...] = ()


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _records(row: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """The row and every operation record on it that can own a time limit."""
    yield row
    trajectory = row.get("ng_trajectory")
    if isinstance(trajectory, Mapping):
        for key in ("invocations", "tool_calls"):
            yield from (r for r in trajectory.get(key) or [] if isinstance(r, Mapping))
    capture = row.get("ng_model_call_capture")
    if isinstance(capture, Mapping):
        yield from (call for call in capture.get("calls") or [] if isinstance(call, Mapping))


def timeout_events(row: Mapping[str, Any]) -> tuple[list[TimeoutEvent], list[Mapping[str, Any]]]:
    """The valid timeout events on the row, once per ``event_id``, and the entries that are not valid events.

    A forwarded event keeps its ``event_id``, so a firing copied onto several records counts once.
    """
    events: dict[str, TimeoutEvent] = {}
    invalid: list[Mapping[str, Any]] = []
    for record in _records(row):
        raw_events = record.get("timeouts")
        for raw in raw_events if isinstance(raw_events, list) else []:
            try:
                event = TimeoutEvent.model_validate(raw)
            except ValidationError:
                invalid.append(raw if isinstance(raw, Mapping) else {})
                continue
            events.setdefault(event.event_id, event)
    return list(events.values()), invalid


def _policy_call_timed_out(row: Mapping[str, Any]) -> bool:
    capture = row.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, Mapping) else None
    return any(isinstance(call, Mapping) and call.get("error_category") == "timeout" for call in calls or [])


def _legacy_limit(row: Mapping[str, Any]) -> Optional[str]:
    """The legacy signal that says the row ran out of time, named by where it was found, or None."""
    failed = isinstance(row.get("_ng_failure_class"), str)
    for key in ("_ng_failure_class", "failure_kind"):
        if row.get(key) in _TIMEOUT_LABELS:
            return f"{key}={row[key]}"
    if failed and row.get("_ng_failure_message") == _EPISODE_DEADLINE_REASON:
        return "episode_deadline"
    for name in _TIMEOUT_FLAGS:
        if bool(row.get(name)) and _number(row.get(name)) != 0:
            return name
    termination = row.get("termination")
    # A sandboxed agent's own termination record, as Terminal-Bench writes it.
    if isinstance(termination, Mapping) and termination.get("reason") == "timeout":
        return "termination.reason=timeout"
    if failed and _policy_call_timed_out(row):
        return "model_call.error_category=timeout"
    return None


def _reached_its_limit(row: Mapping[str, Any]) -> bool:
    """Whether a record on the row measured an interval at or past its configured limit, with no firing on it."""
    for record in _records(row):
        configured = _number(record.get("configured_s"))
        elapsed = _number(record.get("elapsed_s"))
        if configured is not None and configured > 0 and elapsed is not None and elapsed >= configured:
            return True
    return False


def classify_time_limit(row: Mapping[str, Any]) -> Optional[TimeLimitVerdict]:
    """Whether the row ran out of time, which limit it hit, and on what evidence; None when nothing says so."""
    events, _ = timeout_events(row)
    if events:
        first = events[0]
        return TimeLimitVerdict("typed", first.limit, first.enforcer, tuple(events))
    legacy = _legacy_limit(row)
    if legacy is not None:
        return TimeLimitVerdict("legacy", legacy, None)
    if isinstance(row.get("_ng_failure_class"), str) and _reached_its_limit(row):
        return TimeLimitVerdict("unclassified", None, None)
    return None


# What the section reads of a row: everything the classifier and the time split use, and nothing else.
_ROW_KEYS = (
    "_ng_failure_class",
    "_ng_failure_message",
    "failure_kind",
    "termination",
    "timeouts",
    "timings",
    NG_ELAPSED_S_KEY,
    *_TIMEOUT_FLAGS,
)
_CALL_KEYS = (
    "timeouts",
    "configured_s",
    "elapsed_s",
    "error_category",
    "model_ref",
    "started_at",
    "completed_at",
    "latency_total_ms",
    "tokens_out",
    "duration_ms",
)


def _slim(record: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: record[key] for key in keys if key in record}


def timeout_view(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The part of a row :func:`summarize_timeouts` reads, or None for a row the section does not count.

    A succeeded row that did not run out of time adds only to `rollouts`, so it needs no view. The view
    carries no payloads, so health-check workers can return it for every failed or timed-out row.
    """
    counted = classify_time_limit(row) is not None or isinstance(row.get("_ng_failure_class"), str)
    if not counted and not any(timeout_events(row)):
        return None
    view = _slim(row, _ROW_KEYS)
    trajectory = row.get("ng_trajectory")
    if isinstance(trajectory, Mapping):
        view["ng_trajectory"] = {
            key: [_slim(r, _CALL_KEYS) for r in trajectory[key] if isinstance(r, Mapping)]
            for key in ("invocations", "tool_calls")
            if isinstance(trajectory.get(key), list)
        }
        if isinstance(trajectory.get("gaps"), list):
            view["ng_trajectory"]["gaps"] = trajectory["gaps"]
    capture = row.get("ng_model_call_capture")
    if isinstance(capture, Mapping) and isinstance(capture.get("calls"), list):
        view["ng_model_call_capture"] = {
            "calls": [_slim(call, _CALL_KEYS) for call in capture["calls"] if isinstance(call, Mapping)]
        }
    return view


def rollout_elapsed_s(row: Mapping[str, Any]) -> Optional[float]:
    """The `/run` wall clock rollout collection stamped on this row, in seconds."""
    value = _number(row.get(NG_ELAPSED_S_KEY))
    return value if value is not None and value > 0 else None


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def _quantiles(values: list[float], *stats: str) -> dict[str, Optional[float]]:
    """Each named quantile of ``values`` (``p50`` is the median), or null when there are none."""
    return {stat: _round(_stat(values, Stat.MEDIAN if stat == "p50" else stat)) if values else None for stat in stats}


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


def _timed(call: _Call) -> bool:
    return (call.start is not None and call.end is not None) or call.duration_s is not None


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
    return all(_timed(call) for call in calls)


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
    """Agent time after the last recorded call: typically a call cancelled in flight at the deadline.

    Only calls inside the agent's window bound it: a judge call that starts after the window is not agent
    activity, and a call spanning the window end leaves no tail. An end-only call after the window cannot
    be placed, so the tail is then unknown.
    """
    window_end = _agent_window_end(row)
    if window_end is None:
        return None
    ends = []
    for call in events:
        if call.end is None or (call.start is not None and call.start > window_end):
            continue
        if call.start is None and call.end > window_end:
            return None
        ends.append(min(call.end, window_end))
    return max(0.0, window_end - max(ends)) if ends else None


def _share(part: float, whole: float) -> Optional[float]:
    return _round(part / whole) if whole > 0 else None


def time_split(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the elapsed time of these rollouts went: the policy model, other models (a judge), tools.

    Every share divides by the summed elapsed time of the rollouts that have both an elapsed stamp and
    model-call captures. A share is null when a rollout in that set does not measure that kind of time.
    """
    measured = [
        (row, elapsed, calls) for row in rows if (elapsed := rollout_elapsed_s(row)) and (calls := _model_calls(row))
    ]
    total = sum(elapsed for _, elapsed, _ in measured)
    tools = [_tool_calls(row) for row, _, _ in measured]

    def busy(is_policy: bool) -> Optional[float]:
        # A call with neither bounds nor a duration is unknown time, so that kind's share is unknown too.
        if not all(_timed(c) for _, _, calls in measured for c in calls if c.is_policy == is_policy):
            return None
        return sum(min(_busy_s([c for c in calls if c.is_policy == is_policy]), e) for _, e, calls in measured)

    tool = (
        sum(min(_busy_s(t), elapsed) for t, (_, elapsed, _) in zip(tools, measured))
        if measured and all(_tool_time_measured(row, t) for t, (row, _, _) in zip(tools, measured))
        else None
    )
    unrecorded_parts = [_unrecorded_s(row, calls + t) for (row, _, calls), t in zip(measured, tools)]
    unrecorded = sum(unrecorded_parts) if measured and all(u is not None for u in unrecorded_parts) else None
    parts = {"policy_model": busy(True), "other_model": busy(False), "tool": tool, "unrecorded": unrecorded}
    shares = {name: _share(value, total) if value is not None else None for name, value in parts.items()}
    known = list(shares.values())
    shares["untracked"] = _round(max(0.0, 1 - sum(known))) if measured and all(v is not None for v in known) else None

    policy_calls = [c for _, _, calls in measured for c in calls if c.is_policy]
    # End to end: a call's duration includes its time in the server queue and in prefill.
    speeds = [c.tokens_out / c.duration_s for c in policy_calls if (c.tokens_out or 0) > 0 and (c.duration_s or 0) > 0]
    per_rollout = [float(sum(c.is_policy for c in calls)) for _, _, calls in measured]
    return {
        "rollouts_with_captures": len(measured),
        "share": shares,
        # The slow tail of serving speed is its low end.
        "policy_output_tokens_per_s": _quantiles(speeds, "p50", "p5"),
        "policy_calls_per_rollout": _quantiles(per_rollout, "p50", "p95"),
    }


def summarize_timeouts(rows: Iterable[Mapping[str, Any]], *, rollouts: Optional[int] = None) -> dict[str, Any]:
    """The ``run.stats.timeouts`` section: how many rollouts ran out of time, on what evidence, and where
    their time went.

    ``rows`` are the final record of every rollout: its saved row, or the last failures-sidecar attempt of
    a rollout that never succeeded. They may be :func:`timeout_view` copies, in which case ``rollouts``
    counts every rollout, including those with no view. ``failed_without_signal`` counts failures that no
    evidence ties to a time limit: they are not known to be timeouts, and not known not to be.
    """
    rows = list(rows)
    rollouts = len(rows) if rollouts is None else rollouts
    verdicts = [(row, classify_time_limit(row)) for row in rows]
    timed_out = [(row, verdict) for row, verdict in verdicts if verdict is not None]
    by_evidence = Counter(verdict.evidence for _, verdict in timed_out)
    elapsed = [value for value in (rollout_elapsed_s(row) for row, _ in timed_out) if value is not None]
    invalid = sum(len(timeout_events(row)[1]) for row in rows)
    return {
        "rollouts": rollouts,
        "timed_out": len(timed_out),
        "by_evidence": {name: by_evidence[name] for name in ("typed", "legacy", "unclassified")},
        "by_limit": dict(sorted(Counter(verdict.limit or "unknown" for _, verdict in timed_out).items())),
        "by_enforcer": dict(
            sorted(Counter(v.enforcer for _, v in timed_out if v.evidence == "typed" and v.enforcer).items())
        ),
        "events": sum(len(verdict.events) for _, verdict in timed_out),
        "invalid_events": invalid,
        "failed_without_signal": sum(
            1 for row, verdict in verdicts if verdict is None and isinstance(row.get("_ng_failure_class"), str)
        ),
        "elapsed_s": _quantiles(elapsed, "p50", "p95"),
        "time_split": time_split([row for row, _ in timed_out]),
    }


def flatten_for_export(section: Mapping[str, Any], prefix: str = "health/timeouts") -> dict[str, float]:
    """The numeric leaves of the section as flat exporter keys, e.g. ``health/timeouts/by_evidence/typed``.

    A null value is left out, not exported as zero: unknown stays unknown on the tracking server.
    """
    flat: dict[str, float] = {}
    for key, value in section.items():
        name = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_for_export(value, name))
        elif _number(value) is not None:
            flat[name] = value
    return flat
