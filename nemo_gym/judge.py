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
"""Shared LLM-as-judge failure abstraction.

A failed judge call is a distinct outcome, not a wrong answer. Resources servers
issue judge calls through ``call_judge``, or wrap a bespoke call in
``reraise_judge_errors``; ``judge_failsafe`` wraps every verify endpoint so a
JudgeError becomes a row tagged ``_ng_failure_class="judge_failed"``, which
rollout_collection routes to ``<output>_failures.jsonl`` — excluded from the
aggregate metrics and from the file-based re-aggregation (``gym eval
aggregate``), and retryable on resume.

Boundary: a failed *call* (transport/timeout/auth/HTTP) → JudgeError → sidecar, and
so is a *received-but-unparseable* verdict: the judge, not the model, failed to
produce a usable answer, so the row is retried as ``judge_failed`` and never scored
0 for the model. A server raises it with ``raise_unparseable_verdict`` from the
branch that finds no verdict. Empty output is a per-benchmark call, so servers differ.

One judge error is the model's: a judge that rejects its input as too long when the
model's own answer made it so. A server that knows this (its judge input carries no
documents, or the input without the answer fits) passes ``overflow_is_the_models=True``
to ``call_judge``; the row is then ``judge_input_too_long``, a final 0. A server that
cannot tell keeps the default, so the overflow stays a retried ``judge_failed``.

Wrap the call, not the scoring around it: ``reraise_judge_errors`` relabels every
exception it sees, so covering a whole ``verify()`` would misfile ordinary bugs
(a KeyError in prompt assembly, say) as judge failures.
"""

import functools
import re
from typing import Any, Awaitable, Callable, Type, TypeVar

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


ResponseT = TypeVar("ResponseT", bound=BaseModel)


def normalize_math_judge_verdict(text: str) -> str:
    """Canonicalize whitespace inside A=B/A!=B markers without changing other text."""
    return re.sub(
        r"\[\[\s*A\s*(!\s*)?=\s*B\s*\]\]",
        lambda match: "[[A!=B]]" if match.group(1) else "[[A=B]]",
        text,
    )


class JudgeError(Exception):
    """A judge call failed; judge_failsafe routes the row to the failures sidecar."""


class JudgeInputTooLongError(JudgeError):
    """The judge rejected an input the model's own answer made too long: the model's final 0."""


# How an OpenAI-compatible engine (vLLM, SGLang, the OpenAI API) words a request that exceeds its context window.
_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"context_length_exceeded|maximum context length|context window|prompt is too long|input length"
    r"|too many (?:input )?tokens",
    re.IGNORECASE,
)


# How a Gym model server relays an upstream failure it does not propagate: a 500 whose body carries the upstream
# reply verbatim (``SimpleServer.setup_exception_middleware``).
_GYM_RELAY_MARKER = "calling an inner server:"


def _error_body(error: BaseException) -> str:
    body = getattr(error, "response_content", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return body or ""


def _is_context_overflow(error: BaseException) -> bool:
    """Whether a judge call failed because the request exceeds the judge's context window.

    The judge's engine answers with a 4xx. A Gym model server in front of it that does not propagate that status
    answers with a 500 that carries the engine's reply, which says the same.
    """
    status = getattr(error, "status", None)
    body = _error_body(error)
    if not isinstance(status, int) or not (400 <= status < 500 or (status == 500 and _GYM_RELAY_MARKER in body)):
        return False
    return bool(_CONTEXT_OVERFLOW_PATTERN.search(f"{getattr(error, 'message', '')} {body}"))


def raise_unparseable_verdict(server: str, judge_output: str) -> None:
    """Raise the JudgeError for a judge reply that carries no verdict the server can read.

    ``judge_failsafe`` turns it into a ``judge_failed`` row: retried on resume, outside the score.
    """
    raise JudgeError(f"{server}: judge verdict is unparseable: {judge_output[:500]!r}")


async def reraise_judge_errors(coro: Awaitable[Any]) -> Any:
    """Await a judge call; re-raise any exception as JudgeError (recorded verbatim)."""
    try:
        return await coro
    except JudgeError:
        raise
    except Exception as e:
        # The judge's own reply (an engine's 400 body, relayed through a model server's 500) says why it failed;
        # it goes on the row, so a context overflow and an unparseable verdict do not look alike.
        body = _error_body(e)
        reply = f" | judge reply: {body[:1000]}" if body else ""
        raise JudgeError(f"{type(e).__name__}: {e}{reply}") from e


async def call_judge(
    server_client: ServerClient,
    *,
    server_name: str,
    url_path: str,
    json: Any,
    response_model: Type[ResponseT],
    overflow_is_the_models: bool = False,
) -> ResponseT:
    """POST to a judge model server and parse the reply; raise JudgeError on failure.

    ``url_path`` is the judge's API surface (``/v1/responses`` or
    ``/v1/chat/completions``), paired with the matching ``response_model``. The
    status check comes before parsing so an auth or server error reports the HTTP
    failure itself, not a validation error against the error body.
    ``overflow_is_the_models`` says the judge input carries nothing long but the model's answer, so a
    context-length rejection is the model's final 0 (``JudgeInputTooLongError``), not a retried judge failure.
    """

    async def _call() -> ResponseT:
        http_response = await server_client.post(server_name=server_name, url_path=url_path, json=json)
        try:
            await raise_for_status(http_response)
        except Exception as e:
            if overflow_is_the_models and _is_context_overflow(e):
                raise JudgeInputTooLongError(f"{type(e).__name__}: {e} | judge reply: {_error_body(e)[:1000]}") from e
            raise
        return response_model.model_validate(await get_response_json(http_response))

    return await reraise_judge_errors(_call())


def judge_failsafe(verify_fn: Callable) -> Callable:
    """Wrap verify() so a JudgeError returns a sidecar-routed row (reward 0.0, the
    routing keys, the request's ``response`` carried) instead of propagating.
    functools.wraps keeps verify's signature so FastAPI injects the same params
    (``body``, and ``request`` for servers that take it); ``*args, **kwargs`` pass
    them straight through."""

    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        try:
            return await verify_fn(*args, **kwargs)
        except JudgeError as e:
            body = kwargs.get("body") or next(
                (a for a in (*kwargs.values(), *args) if hasattr(a, "model_dump") and hasattr(a, "response")),
                None,
            )
            if body is None:  # verify always has a request body; guard against an opaque 500
                raise RuntimeError("judge_failsafe: could not locate the verify request body") from e
            error = str(e)[:2000]
            data = body.model_dump()
            if isinstance(e, JudgeInputTooLongError):
                # The model's own answer did not fit the judge: a final failure, scored 0, in training too.
                data |= {
                    "reward": 0.0,
                    "mask_sample": False,
                    "failure_kind": "judge_input_too_long",
                    "failure_reason": error,
                    "_ng_failure_class": "judge_input_too_long",
                    "_ng_failure_judge_error": error,
                }
                return JSONResponse(content=jsonable_encoder(data))
            instance_config = data.get("instance_config")
            data |= {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "judge_failed",
                "failure_reason": error,
                # Keep older NeMo RL consumers working during top-level field migration.
                "instance_config": {
                    **(instance_config if isinstance(instance_config, dict) else {}),
                    "mask_sample": True,
                },
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": error,
            }
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
