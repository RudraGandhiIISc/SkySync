"""Streamed responses whose clean-up runs however they end.

A StreamingResponse's generator runs its `finally` only once it has
started, and a route that fails after building a stream never sends it.
What a stream holds (the gateway's admission ticket and audit record, the
chat app's per-chat guard) is therefore released by close hooks, which
`ManagedStream` runs exactly once: after the response is sent, however
that ended, or when the route drops it with `aclose`.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

CloseHook = Callable[[], Awaitable[None] | None]


class ManagedStream(StreamingResponse):
    """A StreamingResponse with close hooks that always run, once."""

    def __init__(
        self,
        content: Any,
        *,
        on_close: tuple[CloseHook, ...] = (),
        **kwargs: Any,
    ) -> None:
        """Creates the response.

        Args:
            content: The body iterator.
            on_close: Hooks to run when the stream ends, in order.
            **kwargs: StreamingResponse arguments (headers, media type).
        """
        super().__init__(content, **kwargs)
        self._hooks = list(on_close)
        self._closed = False

    def add_close_hook(self, hook: CloseHook) -> None:
        """Adds a hook that runs after the existing ones."""
        self._hooks.append(hook)

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Sends the stream, then closes it whatever happened."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Closes the body iterator, then runs every hook (once).

        Raises:
            Exception: The first error a hook raised, after all have run.
        """
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            await _run_all(self._hooks)


async def _run_all(hooks: list[CloseHook]) -> None:
    failure: Exception | None = None
    for hook in hooks:
        try:
            result = hook()
            if inspect.isawaitable(result):
                await result
        except Exception as err:  # the rest still run; the first is raised
            failure = failure or err
    if failure is not None:
        raise failure
