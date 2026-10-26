# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants in a borrowed-sandbox OpenClaw session."""

import json
import re
import shlex
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.openclaw_agent.tests.test_sandbox import close_body, seed, setup  # noqa: F401


def grant(name: str, *, required: bool) -> dict:
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": f"http://resources.example:8000/{name}/mcp", "headers": {"x-session": "token"}},
    }


_PROBLEM = {
    "diagnostic": "refused",
    "exit": "exit 1: No MCP server named x",
    "not-listed": "not connected",
    "unreadable": "unreadable probe output: not json",
    "timeout": "timeout: probe exceeded 120 s",
}


def probe_result(name: str, failing: set[str], failure: str) -> dict:
    """What the probe of `name` reports: stdout, stderr, exit code and whether its supervisor timed it out."""
    report = {"servers": {name: {"launch": "streamable-http", "tools": 4}}, "tools": [], "diagnostics": []}
    result = {"stderr": "", "return_code": 0, "timed_out": False}
    if name in failing and failure == "exit":
        return {**result, "stdout": "", "stderr": "No MCP server named x", "return_code": 1}
    if name in failing and failure == "timeout":
        return {**result, "stdout": "", "return_code": None, "timed_out": True}
    if name in failing and failure == "diagnostic":
        # OpenClaw 2026.6.11 probe --json exits 0 when the connection fails and reports it as a diagnostic.
        report = {"servers": {}, "tools": [], "diagnostics": [{"serverName": name, "message": "refused"}]}
    if name in failing and failure == "not-listed":
        report["servers"] = {}
    stdout = "probe log line\n" + json.dumps(report, indent=2)
    if name in failing and failure == "unreadable":
        stdout = "not json"
    return {**result, "stdout": stdout}


def install_probe_fake(sandbox, failing: set[str], failure: str) -> list[dict]:
    """Answer each supervised `openclaw mcp probe` launch the way its supervisor and runner would.

    The probe writes stdout.log/stderr.log in its own directory and its supervisor writes cleanup.json there.
    """
    probes = []
    run_exec = sandbox.run_exec

    async def probe_or_run(command, **kwargs):
        match = re.search(r"(/\S*/mcp-probe-\d+)/cleanup\.json", command)
        if match and "process_supervisor.py" in command:
            directory = match.group(1)
            runner_input = json.loads(sandbox.files.pop(f"{directory}/input.json"))
            name = runner_input["command"][4]
            config = next(text for path, text in sandbox.files.items() if path.endswith("/.openclaw/openclaw.json"))
            probes.append({"name": name, "config": json.loads(config), "input": runner_input, "launch": command})
            result = probe_result(name, failing, failure)
            sandbox.files[f"{directory}/stdout.log"] = result["stdout"]
            sandbox.files[f"{directory}/stderr.log"] = result["stderr"]
            sandbox.files[f"{directory}/cleanup.json"] = json.dumps(
                {
                    "cleanup_confirmed": True,
                    "return_code": result["return_code"],
                    "timed_out": result["timed_out"],
                    "error": None,
                }
            )
            return SimpleNamespace(error_type=None, return_code=0, stdout="", stderr="")
        argv = shlex.split(command)
        if argv[2:4] == ["mcp", "probe"]:
            # The probe as an unsupervised exec, before supervised probes; kept so these tests also run on it.
            config = next(text for path, text in sandbox.files.items() if path.endswith("/.openclaw/openclaw.json"))
            probes.append({"name": argv[4], "config": json.loads(config), "input": kwargs, "launch": command})
            result = probe_result(argv[4], failing, failure)
            if result["timed_out"]:
                return SimpleNamespace(error_type="timeout", return_code=-1, stdout=None, stderr="timed out")
            return SimpleNamespace(error_type=None, **{k: result[k] for k in ("return_code", "stdout", "stderr")})
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = probe_or_run
    return probes


def skipped(closed) -> list[str]:
    return [
        gap["detail"] for gap in closed.json()["agent_observations"]["gaps"] if gap["code"] == "mcp_server_skipped"
    ]


@pytest.mark.parametrize("failure", list(_PROBLEM))
@pytest.mark.parametrize("unreachable", [None, "task_tools", "extra"])
def test_mcp_grants_are_probed_before_the_agent_runs(setup, unreachable, failure):  # noqa: F811
    agent, sandbox = setup
    agent.config.mcp_tool_timeout_seconds = 900
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True), grant("extra", required=False)]
    probes = install_probe_fake(sandbox, {unreachable} - {None}, failure)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        directory = agent._session_records[session_id].state.session.session_dir
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert [probe["name"] for probe in probes] == ["task_tools", "extra"]
        assert probes[0]["config"]["mcp"]["servers"]["task_tools"] == {
            "url": "http://resources.example:8000/task_tools/mcp",
            "transport": "streamable-http",
            "headers": {"x-session": "token"},
            "timeout": 900,
        }
        assert probes[0]["input"]["env"]["OPENCLAW_CONFIG_PATH"] == f"{directory}/home/.openclaw/openclaw.json"
        # Each probe runs under its own supervisor and receipt, in its own directory, bounded at 120 s.
        for index, probe in enumerate(probes):
            assert f"--receipt {directory}/mcp-probe-{index}/cleanup.json" in probe["launch"]
            assert "--timeout 120" in probe["launch"]
            assert probe["input"]["command"][2:] == ["mcp", "probe", probe["name"], "--json"]
        if unreachable == "task_tools":
            assert result.status_code == 502
            assert result.json()["detail"] == (
                "Required Gym MCP tools could not be initialized: task_tools: " + _PROBLEM[failure]
            )
            # The model is never called: the agent run does not start.
            sandbox.launch.assert_not_awaited()
        else:
            assert result.status_code == 200, result.text
            assert result.json()["status"] == "completed"
            config = json.loads(sandbox.files[f"{directory}/home/.openclaw/openclaw.json"])
            expected = {"task_tools"} if unreachable == "extra" else {"task_tools", "extra"}
            assert set(config["mcp"]["servers"]) == expected
            assert config["tools"]["allow"] == ["read", "write", "edit", "exec", "process", "bundle-mcp"]
            sandbox.launch.assert_awaited_once()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    assert skipped(closed) == (["extra: " + _PROBLEM[failure]] if unreachable == "extra" else [])
    # The borrowed sandbox belongs to the resources server: close releases it without stopping it.
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("failure", ["diagnostic", "timeout"])
@pytest.mark.parametrize("required_first", [True, False])
def test_required_failure_keeps_the_optional_failure_gap(setup, required_first, failure):  # noqa: F811
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    grants = [grant("task_tools", required=True), grant("extra", required=False)]
    body["tool_accesses"] = grants if required_first else grants[::-1]
    probes = install_probe_fake(sandbox, {"task_tools", "extra"}, failure)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert len(probes) == 2
        assert result.status_code == 502
        assert result.json()["detail"] == (
            "Required Gym MCP tools could not be initialized: task_tools: " + _PROBLEM[failure]
        )
        sandbox.launch.assert_not_awaited()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    codes = [gap["code"] for gap in closed.json()["agent_observations"]["gaps"]]
    assert skipped(closed) == ["extra: " + _PROBLEM[failure]]
    assert codes.count("agent_activation_interrupted") == 1
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_required_non_mcp_grant_is_still_rejected(setup):  # noqa: F811
    agent, sandbox = setup
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [
        grant("task_tools", required=True),
        {"kind": "direct_http", "name": "tools", "base_url": "http://resources", "required": True},
    ]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=body)
    assert response.status_code == 422
    assert response.json()["detail"].endswith("required grants it cannot use: tools")
    sandbox.exec.assert_not_awaited()
