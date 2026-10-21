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

import logging
import shlex
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import CONFIG_PATHS_KEY_NAME


logger = logging.getLogger(__name__)


def ray_extra(run: dict[str, Any]) -> tuple[str, ...]:
    """`("ray",)` when a server in the run's `config_paths` declares `ray_enabled = True`, else `()`.

    Ray is an optional extra, so a driver that installs Gym must add it for such a server. The check reads the
    configs and server sources of the Gym checkout doing the submit; a config that fails to load adds nothing.
    """
    # Lazy: the CLI module is heavy, and only a submit that installs Gym needs it.
    from nemo_gym.cli.env import _configured_servers
    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.server_utils import entrypoint_ray_enabled

    try:
        _, configs = GlobalConfigDictParser().load_extra_config_paths(list(run.get(CONFIG_PATHS_KEY_NAME) or []))
        servers = []
        if configs:
            merged = OmegaConf.merge(*configs)
            # The run's overrides win over the files, as at runtime, where each reaches `gym eval run` as a
            # dotted `+key.path=value` (see flatten_run_args). Values stay unresolved.
            for path, value in _run_override_paths(run):
                OmegaConf.update(merged, path, value, force_add=True)
            merged = OmegaConf.to_container(merged, resolve=False)
            servers = list(_configured_servers(OmegaConf.create(_without_empty_mappings(merged))))
    except (ConfigError, OmegaConfBaseException) as e:
        logger.warning("Could not load the benchmark configs to check whether it needs Ray: %s", e)
        return ()
    uses_ray = any(entrypoint_ray_enabled(s.dir_path / s.entrypoint_fpath) is True for s in servers)
    return ("ray",) if uses_ray else ()


def _run_override_paths(run: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """The (dotted path, value) pairs that flatten_run_args turns the run config into."""
    pairs = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            pairs.extend(_run_override_paths(value, full_key))
        elif full_key != CONFIG_PATHS_KEY_NAME:
            pairs.append((full_key, value))
    return pairs


def _without_empty_mappings(value: object) -> object:
    """`value` without empty mappings, which hold no server. Runtime drops empty `_inherit_from` templates before
    it lists servers, and the server listing indexes the first key of every mapping."""
    if isinstance(value, dict):
        pruned = {key: _without_empty_mappings(item) for key, item in value.items()}
        return {key: item for key, item in pruned.items() if item != {}}
    return value


def flatten_run_args(run: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten a nested run config dict into shell-quoted ++key.path=value Hydra override tokens."""
    args = []
    for key, value in run.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            args.extend(flatten_run_args(value, full_key))
        elif isinstance(value, list):
            items = ",".join(str(v) for v in value)
            args.append(shlex.quote(f"+{full_key}=[{items}]"))
        else:
            args.append(shlex.quote(f"+{full_key}={value}"))
    return args
