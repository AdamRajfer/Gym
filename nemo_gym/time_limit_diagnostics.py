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
"""Count the rollouts of a run that ran out of time, and how long they took.

A rollout that runs out of time scores as a failure of the evaluated model, whatever made it slow.
This module reads what a run already recorded about each rollout (its failure labels, its timeout
flags and its wall clock) into a ``time_limit`` section of the aggregate metrics. It only reads rows:
it changes no classification, reward or row.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional

from nemo_gym.metrics_config import Stat
from nemo_gym.reward_profile import _stat


# Collector key with the `/run` wall clock of one rollout, success or failure, in seconds.
NG_ELAPSED_S_KEY = "_ng_elapsed_s"
# Set by collection on a failures-sidecar row written without its captures when a captured policy call of that
# rollout timed out, so `gym eval aggregate`, which has no capture store, sees the same evidence.
NG_FAILURE_POLICY_CALL_TIMED_OUT_KEY = "_ng_failure_policy_call_timed_out"

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
    if row.get(NG_FAILURE_POLICY_CALL_TIMED_OUT_KEY) is True:
        return True
    capture = row.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, Mapping) else None
    return any(isinstance(call, Mapping) and call.get("error_category") == "timeout" for call in calls or [])


def captured_policy_call_timed_out(row: Mapping[str, Any]) -> bool:
    """Whether one of the row's captured model calls timed out."""
    return _policy_call_timed_out(row)


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


def summarize_time_limits(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the rollouts that ran out of time and summarize their wall clock.

    ``rows`` are the final record of every rollout: the scored rows and the last failures-sidecar
    attempt of each rollout that never succeeded.
    """
    rows = list(rows)
    timed_out = [row for row in rows if ran_out_of_time(row)]
    elapsed = [value for value in map(rollout_elapsed_s, timed_out) if value is not None]
    return {
        "timed_out": len(timed_out),
        "timed_out_share": _round(len(timed_out) / len(rows)) if rows else None,
        "timed_out_elapsed_s_p50": _quantile(elapsed, Stat.MEDIAN),
        "timed_out_elapsed_s_p95": _quantile(elapsed, "p95"),
    }
