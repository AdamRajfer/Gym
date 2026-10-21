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
"""The sandbox rule: once the model ran a command, losing its sandbox is the model's, unless the platform shows otherwise.

A sandbox that dies, wedges, OOMs or loses its exec daemon while the model's commands run is what a model can cause
(a memory or fork bomb, ``kill -9 -1``, a command that outlives its timeout). Terminal-Bench and SWE-bench Pro score
it 0. Gym retries a sandbox failure only before the model's first command (create, start, a verifier start timeout)
or with platform evidence (``SandboxPlatformEvictionError``: an eviction, node loss or control-plane error).
"""

import pytest

from nemo_gym.failure_kinds import INFRA, MODEL, failure_cause_of, failure_cause_of_text
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, is_terminal_failure
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxOperationTimeoutError,
    SandboxPlatformEvictionError,
    SandboxPtyError,
)


class TestExecTimeSandboxErrorsAreTheModels:
    @pytest.mark.parametrize(
        "error",
        [
            SandboxOperationTimeoutError("Timed out during OpenSandbox command run after 660s; sandbox_id='x'"),
            SandboxPtyError("PTY connection closed unexpectedly (close code 1006)"),
        ],
        ids=["command-outlived-its-timeout", "shell-ended"],
    )
    def test_a_live_exec_time_sandbox_error_is_the_models(self, error: BaseException) -> None:
        assert failure_cause_of(error) == MODEL

    @pytest.mark.parametrize(
        "text",
        [
            "SandboxApiException: Failed to run command failed: Sandbox <id> has ended (OOMKilled): 1/1 observed pods",
            "SandboxTimeoutException: command timed out",
            "SandboxOperationTimeoutError: Timed out during OpenSandbox command status after 10s",
            "SandboxPtyError: PTY connection closed unexpectedly",
            "AgentInfrastructureError: mini-SWE produced no trajectory",
            "SandboxBackendUnreachableError: Sandbox was OOM-killed. OpenSandbox status: state='Error'",
        ],
    )
    def test_relayed_exec_time_sandbox_text_is_the_models(self, text: str) -> None:
        assert failure_cause_of_text(text) == MODEL

    def test_a_tb4_row_whose_agent_run_lost_its_sandbox_is_a_final_zero(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "AgentInfrastructureError: mini-SWE produced no trajectory",
            "infrastructure_error": "AgentInfrastructureError",
        }
        assert is_terminal_failure(row)

    async def test_the_exec_hard_cap_raised_inside_the_provider_is_the_models(self) -> None:
        # A command that wedges past every timeout: the provider raises a bare TimeoutError from its own module.
        import nemo_gym.sandbox.providers.opensandbox.provider as provider

        namespace = {"__name__": provider.__name__}
        exec(
            "def wedge():\n    raise TimeoutError('OpenSandbox exec exceeded hard cap of 1350s; the command wedged')",
            namespace,
        )
        with pytest.raises(TimeoutError) as caught:
            namespace["wedge"]()
        assert failure_cause_of(caught.value) == MODEL


class TestPlatformEvidenceAndCreateStayInfra:
    def test_platform_eviction_is_infra(self) -> None:
        error = SandboxPlatformEvictionError("Sandbox was evicted by the platform: reason='Evicted'")
        assert failure_cause_of(error) == INFRA
        assert failure_cause_of_text(f"SandboxPlatformEvictionError: {error}") == INFRA

    @pytest.mark.parametrize(
        "text",
        [
            "SandboxCreateError: could not create",
            "OpenSandboxCreateError: Timed out creating OpenSandbox sandbox after 1200s",
            "EnvironmentStartTimeoutError",
            "HealthcheckError",
            "AgentSetupTimeoutError",
            "VerifierStartTimeoutError",
        ],
    )
    def test_a_failure_before_the_models_first_command_stays_infra(self, text: str) -> None:
        assert failure_cause_of_text(text) == INFRA

    def test_a_live_create_error_stays_infra(self) -> None:
        assert failure_cause_of(SandboxCreateError("quota")) == INFRA

    def test_a_tb4_sandbox_create_failure_row_is_retried(self) -> None:
        row = {
            NG_FAILURE_CLASS_KEY: "infrastructure_error",
            "failure_reason": "SandboxCreateError",
            "infrastructure_error": "SandboxCreateError",
        }
        assert not is_terminal_failure(row)
