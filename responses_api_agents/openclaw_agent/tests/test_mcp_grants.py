# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants in a borrowed-sandbox OpenClaw session."""

import json
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
}


def probe_result(name: str, *, unreachable: str | None, failure: str) -> SimpleNamespace:
    report = {"servers": {name: {"launch": "streamable-http", "tools": 4}}, "tools": [], "diagnostics": []}
    if name == unreachable and failure == "exit":
        return SimpleNamespace(error_type=None, return_code=1, stdout="", stderr="No MCP server named x")
    if name == unreachable and failure == "diagnostic":
        # OpenClaw 2026.6.11 probe --json exits 0 when the connection fails and reports it as a diagnostic.
        report = {"servers": {}, "tools": [], "diagnostics": [{"serverName": name, "message": "refused"}]}
    if name == unreachable and failure == "not-listed":
        report["servers"] = {}
    stdout = "probe log line\n" + json.dumps(report, indent=2)
    if name == unreachable and failure == "unreadable":
        stdout = "not json"
    return SimpleNamespace(error_type=None, return_code=0, stdout=stdout, stderr="")


@pytest.mark.parametrize("failure", list(_PROBLEM))
@pytest.mark.parametrize("unreachable", [None, "task_tools", "extra"])
def test_mcp_grants_are_probed_before_the_agent_runs(setup, unreachable, failure):  # noqa: F811
    agent, sandbox = setup
    agent.config.mcp_tool_timeout_seconds = 900
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=True), grant("extra", required=False)]
    probes = []
    run_exec = sandbox.run_exec

    async def probe_or_run(command, **kwargs):
        argv = shlex.split(command)
        if argv[2:4] == ["mcp", "probe"]:
            config = next(text for path, text in sandbox.files.items() if path.endswith("/.openclaw/openclaw.json"))
            probes.append((argv[4], json.loads(config), kwargs))
            return probe_result(argv[4], unreachable=unreachable, failure=failure)
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = probe_or_run
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body)
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        directory = agent._session_records[session_id].state.session.session_dir
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert [name for name, _, _ in probes] == ["task_tools", "extra"]
        _, probed_config, probe_kwargs = probes[0]
        assert probed_config["mcp"]["servers"]["task_tools"] == {
            "url": "http://resources.example:8000/task_tools/mcp",
            "transport": "streamable-http",
            "headers": {"x-session": "token"},
            "timeout": 900,
        }
        assert probe_kwargs["env"]["OPENCLAW_CONFIG_PATH"] == f"{directory}/home/.openclaw/openclaw.json"
        assert probe_kwargs["timeout_s"] == 120
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
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200, closed.text
    skipped = [gap for gap in closed.json()["agent_observations"]["gaps"] if gap["code"] == "mcp_server_skipped"]
    assert [gap["detail"] for gap in skipped] == (["extra: " + _PROBLEM[failure]] if unreachable == "extra" else [])
    # The borrowed sandbox belongs to the resources server: close releases it without stopping it.
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
