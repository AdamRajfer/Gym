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
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

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


def _write_server(root: Path, name: str, *, ray_enabled: bool) -> None:
    server_dir = root / "resources_servers" / name
    server_dir.mkdir(parents=True)
    (server_dir / "requirements.txt").write_text("nemo-gym\n")
    (server_dir / "app.py").write_text(f"class Server:\n    ray_enabled = {ray_enabled}\n\nServer.run_webserver()\n")


@pytest.mark.parametrize(
    ("file_server", "file_entrypoint", "overrides", "expected"),
    [
        # The entrypoint interpolates a value the run sets.
        ("uses_ray", "${server_entrypoint}", {"server_entrypoint": "app.py"}, _RAY_INSTALL),
        # The run points a non-Ray server at a Ray entrypoint, and the reverse.
        (
            "no_ray",
            "app.py",
            {"s": {"resources_servers": {"no_ray": {"entrypoint": "../uses_ray/app.py"}}}},
            _RAY_INSTALL,
        ),
        (
            "uses_ray",
            "app.py",
            {"s": {"resources_servers": {"uses_ray": {"entrypoint": "../no_ray/app.py"}}}},
            _PLAIN_INSTALL,
        ),
        # A dotted key reaches `gym eval run` as the same nested path.
        ("no_ray", "app.py", {"s.resources_servers.no_ray.entrypoint": "../uses_ray/app.py"}, _RAY_INSTALL),
    ],
)
def test_run_overrides_decide_the_entrypoint_as_at_runtime(
    tmp_path, monkeypatch, file_server, file_entrypoint, overrides, expected
):
    monkeypatch.chdir(tmp_path)
    _write_server(tmp_path, "uses_ray", ray_enabled=True)
    _write_server(tmp_path, "no_ray", ray_enabled=False)
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(f"s:\n  resources_servers:\n    {file_server}:\n      entrypoint: {file_entrypoint}\n")
    config = _config([str(config_path)])
    config.driver.benchmarks["bench"].run.update(overrides)

    assert expected in _slurm_script(config)


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [("benchmarks/gpqa/config.yaml", _PLAIN_INSTALL), ("benchmarks/wmt24pp/config.yaml", _RAY_INSTALL)],
)
def test_an_empty_inherited_template_does_not_break_rendering(tmp_path, config_path, expected):
    # Runtime consumes an empty `_inherit_from` template before it lists servers; detection must accept it too.
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("empty_defaults: {}\nresponses_create_params:\n  _inherit_from: empty_defaults\n")

    assert expected in _slurm_script(_config([config_path, str(overlay)]))


@pytest.mark.parametrize(
    ("base_server", "target_entrypoint", "expected"),
    [
        # The target overrides an inherited non-Ray entrypoint with the Ray one, and the reverse.
        ("no_ray", "../uses_ray/app.py", _RAY_INSTALL),
        ("uses_ray", "../no_ray/app.py", _PLAIN_INSTALL),
        # The target keeps the inherited Ray entrypoint.
        ("uses_ray", None, _RAY_INSTALL),
    ],
)
def test_inheritance_decides_the_entrypoint_as_at_runtime(
    tmp_path, monkeypatch, base_server, target_entrypoint, expected
):
    # Runtime applies `_inherit_from` and drops the source before it lists servers.
    monkeypatch.chdir(tmp_path)
    _write_server(tmp_path, "uses_ray", ray_enabled=True)
    _write_server(tmp_path, "no_ray", ray_enabled=False)
    target = "s:\n  _inherit_from: base\n"
    if target_entrypoint:
        target += f"  resources_servers:\n    {base_server}:\n      entrypoint: {target_entrypoint}\n"
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(
        f"base:\n  resources_servers:\n    {base_server}:\n      entrypoint: app.py\n      domain: other\n" + target
    )

    assert expected in _slurm_script(_config([str(config_path)]))


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize(
    ("config_path", "expected"),
    [("benchmarks/wmt24pp/config.yaml", _RAY_INSTALL), ("benchmarks/gpqa/config.yaml", _PLAIN_INSTALL)],
)
def test_a_top_level_value_set_only_in_the_job_does_not_hide_a_ray_server(
    tmp_path, monkeypatch, position, config_path, expected
):
    monkeypatch.delenv("ONLY_IN_THE_JOB", raising=False)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("job_only: ${oc.env:ONLY_IN_THE_JOB}\n")
    config_paths = [str(overlay), config_path] if position == "before" else [config_path, str(overlay)]

    assert expected in _slurm_script(_config(config_paths))


def test_the_server_listing_skips_an_unresolved_top_level_value_only_when_asked(monkeypatch):
    from nemo_gym.cli.env import _configured_servers

    monkeypatch.delenv("ONLY_IN_THE_JOB", raising=False)
    config = OmegaConf.create({"job_only": "${oc.env:ONLY_IN_THE_JOB}"})

    with pytest.raises(OmegaConfBaseException):
        list(_configured_servers(config))
    assert list(_configured_servers(config, skip_unresolved_roots=True)) == []


@pytest.mark.parametrize(
    ("base_server", "target_entrypoint", "expected"),
    [
        ("no_ray", "../uses_ray/app.py", _RAY_INSTALL),
        ("uses_ray", "../no_ray/app.py", _PLAIN_INSTALL),
        ("uses_ray", None, _RAY_INSTALL),
    ],
)
def test_an_unresolved_list_value_does_not_undo_inheritance(
    tmp_path, monkeypatch, base_server, target_entrypoint, expected
):
    # The inheritance step reads list items without resolving them, so a job-only list value cannot stop it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONLY_IN_THE_JOB", raising=False)
    _write_server(tmp_path, "uses_ray", ray_enabled=True)
    _write_server(tmp_path, "no_ray", ray_enabled=False)
    target = "s:\n  _inherit_from: base\n"
    if target_entrypoint:
        target += f"  resources_servers:\n    {base_server}:\n      entrypoint: {target_entrypoint}\n"
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(
        f"base:\n  resources_servers:\n    {base_server}:\n      entrypoint: app.py\n      domain: other\n"
        + target
        + 'job_only: ["${oc.env:ONLY_IN_THE_JOB}"]\n'
    )

    assert expected in _slurm_script(_config([str(config_path)]))


def test_the_detection_swap_still_applies_directives_inside_lists(monkeypatch):
    from nemo_gym.global_config import GlobalConfigDictParser

    monkeypatch.delenv("ONLY_IN_THE_JOB", raising=False)
    config = OmegaConf.create(
        {
            "defaults": {"value": 1},
            "holders": [{"cfg": {"_copy": "defaults"}}, "${oc.env:ONLY_IN_THE_JOB}"],
        }
    )

    GlobalConfigDictParser()._recursively_swap_keys(config, resolve_list_items=False)

    assert config.holders[0].cfg.value == 1
    assert config.defaults.value == 1


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize(
    ("config_path", "expected"),
    [("benchmarks/wmt24pp/config.yaml", _RAY_INSTALL), ("benchmarks/gpqa/config.yaml", _PLAIN_INSTALL)],
)
def test_an_unresolved_entrypoint_does_not_hide_another_ray_server(
    tmp_path, monkeypatch, position, config_path, expected
):
    monkeypatch.delenv("PLAIN_ENTRYPOINT", raising=False)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "plain:\n  resources_servers:\n    mcqa:\n      entrypoint: ${oc.env:PLAIN_ENTRYPOINT}\n      domain: knowledge\n"
    )
    config_paths = [str(overlay), config_path] if position == "before" else [config_path, str(overlay)]

    assert expected in _slurm_script(_config(config_paths))


def test_the_server_listing_skips_an_unresolved_branch_only_when_asked(monkeypatch):
    from nemo_gym.cli.env import _configured_servers

    monkeypatch.delenv("PLAIN_ENTRYPOINT", raising=False)
    config = OmegaConf.create(
        {
            "plain": {"resources_servers": {"mcqa": {"entrypoint": "${oc.env:PLAIN_ENTRYPOINT}"}}},
            "other": {"resources_servers": {"mcqa": {"entrypoint": "app.py"}}},
        }
    )

    with pytest.raises(OmegaConfBaseException):
        list(_configured_servers(config))
    assert [server.top_level_path for server in _configured_servers(config, skip_unresolved_roots=True)] == ["other"]

    monkeypatch.setenv("PLAIN_ENTRYPOINT", "app.py")
    assert [server.top_level_path for server in _configured_servers(config)] == ["plain", "other"]
