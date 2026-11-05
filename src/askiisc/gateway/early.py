"""Admission before the route runs: the FR-6 fast shed path.

A refused question should cost the server as little as possible. Without
this middleware a chat question is admitted only after the session's
account checks, the chat's history is loaded and decrypted, its access
check (an HTTP call) is made and a gateway token is signed and verified.
`EarlyAdmission` admits it as soon as the session cookie names the user,
and so the tenant: one store lookup. A refused request gets its audited
429 at once; an admitted one carries its claim to the route, which still
makes every check (a failed check refuses it as before, fail closed).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request
from starlette.responses import Response

from askiisc.gateway.app import Claim, GatewayDeps, Origin, admit_early
from askiisc.gateway.auth import Caller
from askiisc.observability.timing import ask_kind

CLAIM_KEY = "askiisc.admission"
WEB_KINDS = frozenset({"temporary", "chat"})
MAX_EARLY_BODY = 512 * 1024
"""Bodies larger than this skip the early path (the route refuses them)."""

Identify = Callable[[Any], Awaitable[tuple[Caller, Origin] | None]]


def claim_of(request: Request) -> Claim | None:
    """The claim `EarlyAdmission` stored on a request, if any."""
    claim = request.scope.get("state", {}).get(CLAIM_KEY)
    return claim if isinstance(claim, Claim) else None


def question_of(body: bytes) -> str:
    """The question field of a JSON body ("" if there is none)."""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    question = data.get("question") if isinstance(data, dict) else None
    return question if isinstance(question, str) else ""


async def read_body(
    receive: Callable[[], Awaitable[Any]],
) -> tuple[bytes | None, Callable[[], Awaitable[Any]]]:
    """Reads a request body and returns a receive that replays it.

    Args:
        receive: The ASGI receive callable.

    Returns:
        The body (None if it is longer than MAX_EARLY_BODY) and a receive
        that gives the messages already read, then the rest.
    """
    seen: list[Any] = []
    size = 0
    while True:
        message = await receive()
        seen.append(message)
        if message["type"] != "http.request":
            break
        size += len(message.get("body", b""))
        if size > MAX_EARLY_BODY or not message.get("more_body", False):
            break
    complete = seen[-1].get("type") == "http.request" and not seen[-1].get(
        "more_body", False
    )
    body = (
        b"".join(m.get("body", b"") for m in seen)
        if complete and size <= MAX_EARLY_BODY
        else None
    )

    async def replay() -> Any:
        if seen:
            return seen.pop(0)
        return await receive()

    return body, replay


class EarlyAdmission:
    """ASGI middleware: admit web questions before their route."""

    def __init__(self, app: Any, deps: GatewayDeps, identify: Identify) -> None:
        """Wraps an ASGI app.

        Args:
            app: The app.
            deps: The gateway's dependencies (admission, audit).
            identify: The session's caller and origin, or None (the route
                then refuses the request itself).
        """
        self._app = app
        self._deps = deps
        self._identify = identify

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Admits or sheds a question request, then runs the route."""
        if not _is_question(scope):
            await self._app(scope, receive, send)
            return
        who = await self._identify(scope)
        body, replay = await read_body(receive)
        if who is None or body is None:
            await self._app(scope, replay, send)
            return
        caller, origin = who
        outcome = admit_early(self._deps, caller, question_of(body), origin)
        if isinstance(outcome, Response):
            await outcome(scope, replay, send)
            return
        scope.setdefault("state", {})[CLAIM_KEY] = outcome
        try:
            await self._app(scope, replay, send)
        finally:
            if not outcome.used:
                self._deps.admission.release(outcome.ticket, completed=False)


def _is_question(scope: Any) -> bool:
    return (
        scope["type"] == "http"
        and scope.get("method") == "POST"
        and ask_kind(scope.get("path", "")) in WEB_KINDS
    )
