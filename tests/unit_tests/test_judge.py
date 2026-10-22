# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pydantic import BaseModel, ConfigDict

from nemo_gym.judge import JudgeError, call_judge, judge_failsafe, normalize_math_judge_verdict, reraise_judge_errors
from nemo_gym.server_utils import ServerClient


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[[ A=B ]]", "[[A=B]]"),
        ("[[ A != B ]]", "[[A!=B]]"),
        ("[[\nA\t! = B\n]]", "[[A!=B]]"),
        ("[[A=B]] then [[ A != B ]]", "[[A=B]] then [[A!=B]]"),
        ("An explanation with spaces: [[ A=B ]]", "An explanation with spaces: [[A=B]]"),
        ("A=B without brackets", "A=B without brackets"),
        ("[[A==B]] [[A≠B]] [[a=b]] [[A=C]]", "[[A==B]] [[A≠B]] [[a=b]] [[A=C]]"),
        ("[ [ A=B ] ]", "[ [ A=B ] ]"),
        ("", ""),
    ],
)
def test_normalize_math_judge_verdict(text: str, expected: str) -> None:
    assert normalize_math_judge_verdict(text) == expected


class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")

    response: dict = {}


class _Verdict(BaseModel):
    verdict: str


def _client(*, status: int, body: bytes) -> MagicMock:
    """A ServerClient stand-in whose post() returns an aiohttp-shaped response."""
    resp = MagicMock(ok=status < 400, status=status)
    resp.read = AsyncMock(return_value=body)
    resp.content.read = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock(side_effect=RuntimeError(f"HTTP {status}"))
    return MagicMock(post=AsyncMock(return_value=resp))


async def _call(client) -> _Verdict:
    return await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)


class TestCallJudge:
    @pytest.mark.asyncio
    async def test_success_parses_response_model(self) -> None:
        assert (await _call(_client(status=200, body=b'{"verdict": "CORRECT"}'))).verdict == "CORRECT"

    @pytest.mark.asyncio
    async def test_http_error_reported_as_http_not_validation_error(self) -> None:
        # A 401 body is not a valid _Verdict. Without the status check the failure
        # would surface as an opaque pydantic ValidationError on the error payload.
        with pytest.raises(JudgeError, match="HTTP 401"):
            await _call(_client(status=401, body=b'{"error": {"code": "401"}}'))

    @pytest.mark.asyncio
    async def test_transport_error_becomes_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="judge unreachable"):
            await _call(MagicMock(post=AsyncMock(side_effect=ConnectionError("judge unreachable"))))


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_success_returns_result(self) -> None:
        async def ok():
            return "verdict"

        assert await reraise_judge_errors(ok()) == "verdict"

    @pytest.mark.asyncio
    async def test_exception_reraised_as_judge_error(self) -> None:
        async def boom():
            raise RuntimeError("judge timeout")

        with pytest.raises(JudgeError, match="RuntimeError: judge timeout"):
            await reraise_judge_errors(boom())


class TestJudgeFailsafe:
    @pytest.mark.asyncio
    async def test_success_passes_through(self) -> None:
        async def verify(body):
            return {"reward": 1.0}

        assert await judge_failsafe(verify)(_Req()) == {"reward": 1.0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("by_keyword", [False, True])
    async def test_judge_error_routed_to_sidecar(self, by_keyword: bool) -> None:
        async def verify(body):
            raise JudgeError("RuntimeError: judge 401")

        req = _Req(response={"final": "answer"})
        # FastAPI injects by keyword (kwargs["body"]); direct callers pass positionally.
        out = await (judge_failsafe(verify)(body=req) if by_keyword else judge_failsafe(verify)(req))
        data = orjson.loads(out.body)
        assert data["reward"] == 0.0
        assert data["mask_sample"] is True
        assert data["failure_kind"] == "judge_failed"
        assert data["failure_reason"] == "RuntimeError: judge 401"
        assert data["instance_config"]["mask_sample"] is True
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["_ng_failure_judge_error"] == "RuntimeError: judge 401"
        # The model's final output is carried for a later judge-only replay.
        assert data["response"] == {"final": "answer"}
        # Transient: never terminal, so resume re-dispatches it.
        assert "_ng_failure_terminal" not in data

    @pytest.mark.asyncio
    @pytest.mark.parametrize("instance_config", [None, "invalid", {"task": "keep", "mask_sample": False}])
    async def test_mask_contract_overrides_request_defaults_without_mutation(self, instance_config) -> None:
        async def verify(body):
            raise JudgeError("judge offline " + "x" * 3000)

        request = _Req(
            response={"final": "answer"},
            mask_sample=False,
            failure_kind=None,
            instance_config=instance_config,
        )
        original = request.model_dump()
        data = orjson.loads((await judge_failsafe(verify)(request)).body)
        assert data["mask_sample"] is True and data["failure_kind"] == "judge_failed"
        assert data["instance_config"] == {
            **(instance_config if isinstance(instance_config, dict) else {}),
            "mask_sample": True,
        }
        assert data["failure_reason"].startswith("judge offline")
        assert len(data["failure_reason"]) == 2000
        assert data["_ng_failure_judge_error"] == data["failure_reason"]
        assert request.model_dump() == original


def _overflow_client(status: int, body: bytes) -> MagicMock:
    """A judge whose model server rejects the request the way nemo_gym.server_utils.raise_for_status raises it."""
    from types import SimpleNamespace

    from aiohttp import ClientResponseError

    def reject() -> None:
        request_info = SimpleNamespace(method="POST", url="http://judge/v1", real_url="http://judge/v1", headers={})
        raise ClientResponseError(request_info=request_info, history=(), status=status, message="Bad Request")

    resp = MagicMock(ok=False, status=status)
    resp.read = AsyncMock(return_value=body)
    resp.content.read = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock(side_effect=reject)
    return MagicMock(post=AsyncMock(return_value=resp))


_OVERFLOW_BODY = (
    b'{"error": {"message": "This model\'s maximum context length is 32768 tokens. However, you requested 40000 '
    b'tokens.", "code": "context_length_exceeded"}}'
)


class TestJudgeInputTooLong:
    """A judge input the model's own answer made too long is the model's final 0, where the server can tell."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The judge-overflow rule and the judge's reply on the row are evaluation-only.
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)

    @pytest.mark.asyncio
    async def test_training_keeps_an_answer_caused_overflow_judge_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: False)

        async def verify(body):
            return await call_judge(
                _overflow_client(400, _OVERFLOW_BODY),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        data = orjson.loads((await judge_failsafe(verify)(_Req())).body)
        assert (data["_ng_failure_class"], data["reward"], data["mask_sample"]) == ("judge_failed", 0.0, True)

    @pytest.mark.asyncio
    async def test_an_answer_caused_overflow_is_the_models_final_zero(self) -> None:
        from nemo_gym.judge import JudgeInputTooLongError

        async def verify(body):
            return await call_judge(
                _overflow_client(400, _OVERFLOW_BODY),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        with pytest.raises(JudgeInputTooLongError):
            await verify(None)
        data = orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "a" * 10}))).body)
        assert data["_ng_failure_class"] == "judge_input_too_long"
        assert data["failure_kind"] == "judge_input_too_long"
        assert data["reward"] == 0.0
        assert data["mask_sample"] is False
        from nemo_gym.failure_kinds import FAILURE_RULES, FINAL

        assert FAILURE_RULES["judge_input_too_long"] == FINAL

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overflow_is_the_models", "status", "body"),
        [(False, 400, _OVERFLOW_BODY), (True, 400, b'{"error": "invalid request"}'), (True, 503, _OVERFLOW_BODY)],
        ids=["server-cannot-tell", "other-4xx", "5xx"],
    )
    async def test_other_judge_errors_stay_judge_failed(
        self, overflow_is_the_models: bool, status: int, body: bytes
    ) -> None:
        async def verify(body_):
            return await call_judge(
                _overflow_client(status, body),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=overflow_is_the_models,
            )

        data = orjson.loads((await judge_failsafe(verify)(_Req())).body)
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["mask_sample"] is True


class _JudgeRelayClient(ServerClient):
    """A ServerClient whose post() reaches a real Gym judge model server over its HTTP app, as the real hop does."""

    judge: object

    async def post(self, server_name: str, url_path: str, json=None, **kwargs):
        from types import SimpleNamespace

        from aiohttp import ClientResponseError

        payload = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        reply = self.judge.post(url_path, json=payload)

        def reject() -> None:
            url = f"http://judge{url_path}"
            info = SimpleNamespace(method="POST", url=url, real_url=url, headers={})
            raise ClientResponseError(request_info=info, history=(), status=reply.status_code, message="Error")

        relayed = MagicMock(ok=reply.is_success, status=reply.status_code, headers=dict(reply.headers))
        relayed.read = AsyncMock(return_value=reply.content)
        relayed.content.read = AsyncMock(return_value=reply.content)
        relayed.raise_for_status = MagicMock(side_effect=None if reply.is_success else reject)
        return relayed


def _JudgeRelay(judge_app) -> "_JudgeRelayClient":
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    from nemo_gym.server_utils import BaseServerConfig

    return _JudgeRelayClient.model_construct(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        judge=TestClient(judge_app, raise_server_exceptions=False),
    )


def _judge_model_server(upstream_error: BaseException):
    """Gym's own OpenAI-compatible judge model server, whose upstream engine raises upstream_error."""
    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
    from nemo_gym.server_utils import ServerClient
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://judge.example/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="judge",
        entrypoint="",
        name="judge",
    )
    server = SimpleModelServer(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=OmegaConf.create({}))
    )
    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(side_effect=upstream_error)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    return app


class TestJudgeReasonThroughTheModelServer:
    """The judge's own reason survives Gym's judge model server, which answers 500 for an upstream 4xx it keeps."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The judge-overflow rule and the judge's reply on the row are evaluation-only.
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)

    @pytest.mark.asyncio
    async def test_training_keeps_the_row_as_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: False)

        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=True)

        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_failed", 0.0, True)
        assert (
            row["_ng_failure_judge_error"]
            == "ClientResponseError: 500, message='Error', url='http://judge/v1/responses'"
        )

    @staticmethod
    def _upstream(status: int, body: bytes) -> BaseException:
        from types import SimpleNamespace

        from aiohttp import ClientResponseError

        info = SimpleNamespace(method="POST", url="https://judge.example/v1", real_url="https://judge.example/v1")
        error = ClientResponseError(request_info=info, history=(), status=status, message="Bad Request")
        error.response_content = body
        return error

    async def _row(self, upstream_error: BaseException, *, overflow_is_the_models: bool) -> dict:
        relay = _JudgeRelay(_judge_model_server(upstream_error))

        async def verify(body):
            return await call_judge(
                relay,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade this"},
                response_model=_Verdict,
                overflow_is_the_models=overflow_is_the_models,
            )

        return orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "answer"}))).body)

    @pytest.mark.asyncio
    async def test_an_overflow_relayed_as_a_500_is_still_the_models_final_zero(self) -> None:
        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=True)

        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_input_too_long", 0.0, False)
        assert "context_length_exceeded" in row["_ng_failure_judge_error"]

    @pytest.mark.asyncio
    async def test_a_server_that_cannot_tell_keeps_the_judges_reason_on_the_row(self) -> None:
        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=False)

        assert row["_ng_failure_class"] == "judge_failed"
        # The row says why the judge failed, so it no longer looks like any other judge failure.
        assert "context_length_exceeded" in row["_ng_failure_judge_error"]

    @pytest.mark.asyncio
    async def test_a_relayed_server_error_without_an_overflow_stays_judge_failed(self) -> None:
        row = await self._row(self._upstream(503, b'{"error": "overloaded"}'), overflow_is_the_models=True)

        assert row["_ng_failure_class"] == "judge_failed"
        assert row["mask_sample"] is True
