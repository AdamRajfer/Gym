# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The driver's Gym install carries the `ray` extra exactly when a configured server declares `ray_enabled`."""

from pathlib import Path

import pytest

from nemo_gym.orchestration.api import SubmitConfig
from nemo_gym.orchestration.executors.kubernetes_script import build_job_manifest
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script


_RAY_INSTALL = 'uv pip install -e ".[ray]"'
_PLAIN_INSTALL = "uv pip install -e .\n"


def _config(config_paths: list[str], compute: dict | None = None) -> SubmitConfig:
    return SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": compute or {"type": "slurm", "account": "acct"}},
            "driver": {
                "container": "gym:latest",
                "gym_install": {"ref": "main"},
                "benchmarks": {"bench": {"run": {"config_paths": config_paths}}},
            },
            "job": {"output_path": "/jobs"},
            "otel": {"enabled": False},
        }
    )


def _slurm_script(config: SubmitConfig) -> str:
    bench = config.driver.benchmarks["bench"]
    return build_sbatch_script(config, "bench", bench, config.compute["cluster"], Path("/jobs/x/bench"))


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        ("benchmarks/wmt24pp/config.yaml", _RAY_INSTALL),  # COMET scoring runs on Ray.
        ("benchmarks/gpqa/config.yaml", _PLAIN_INSTALL),  # No Ray: the install line is unchanged.
        ("benchmarks/lmarena_v3/config.yaml", _PLAIN_INSTALL),  # Interpolates ${policy_model_name} in a list.
    ],
)
def test_shipped_benchmarks(config_path, expected):
    assert expected in _slurm_script(_config([config_path]))


def test_the_kubernetes_driver_gets_the_ray_extra(tmp_path):
    config = _config(["benchmarks/wmt24pp/config.yaml"], {"type": "kubernetes", "namespace": "ns", "pvc_name": "p"})
    job = build_job_manifest(
        config,
        "bench",
        config.driver.benchmarks["bench"],
        config.compute["cluster"],
        tmp_path / "run",
        name="gym-bench",
        gym_job_id="gym-job-x",
        resolved_config="",
        manifest="",
    )

    assert _RAY_INSTALL in job["spec"]["template"]["spec"]["containers"][0]["command"][2]


def _write_ray_server(root: Path, extra_yaml: str = "") -> str:
    server_dir = root / "resources_servers" / "uses_ray"
    server_dir.mkdir(parents=True)
    (server_dir / "requirements.txt").write_text("nemo-gym\n")
    (server_dir / "app.py").write_text("class Server:\n    ray_enabled = True\n\nServer.run_webserver()\n")
    config = root / "benchmarks" / "uses_ray" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "uses_ray_resources_server:\n  resources_servers:\n    uses_ray:\n      entrypoint: app.py\n" + extra_yaml
    )
    return str(config)


def test_a_value_set_only_in_the_job_does_not_hide_a_ray_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_ray_server(tmp_path, "      addresses:\n      - ${oc.env:SET_ONLY_IN_THE_JOB}\n")

    assert _RAY_INSTALL in _slurm_script(_config([config_path]))


def test_a_config_that_does_not_load_keeps_the_plain_install(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _write_ray_server(tmp_path, "      addresses: [unclosed\n")

    assert _PLAIN_INSTALL in _slurm_script(_config([config_path]))


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [("benchmarks/gpqa/config.yaml", _PLAIN_INSTALL), ("benchmarks/wmt24pp/config.yaml", _RAY_INSTALL)],
)
def test_an_empty_inherited_template_does_not_break_rendering(tmp_path, config_path, expected):
    # Runtime consumes an empty `_inherit_from` template before it lists servers; detection must accept it too.
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("empty_defaults: {}\nresponses_create_params:\n  _inherit_from: empty_defaults\n")

    assert expected in _slurm_script(_config([config_path, str(overlay)]))
