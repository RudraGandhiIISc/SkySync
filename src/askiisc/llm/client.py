"""Async client for an OpenAI-compatible chat server (vLLM or the fake).

The model name is the only thing that changes between models (FR-8).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field

import httpx

from askiisc.common import requestid
from askiisc.common.errors import AskIIScError, LLMError
from askiisc.common.targets import LLM_MAX_TOKENS, LLM_TEMPERATURE

Message = dict[str, str]


@dataclass
class Usage:
    """Token counts reported by the server."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamState:
    """Filled in while a stream runs; read after it ends."""

    usage: Usage = field(default_factory=Usage)
    text: list[str] = field(default_factory=list)


class LLMClient:
    """Streams chat completions."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str | Callable[[], str],
        model: str,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        timeout_s: float = 60.0,
        extra_body: Mapping[str, object] | None = None,
    ) -> None:
        self._client = client
        self._extra = dict(extra_body or {})
        self._base = base_url
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_s

    @property
    def url(self) -> str:
        """The chat endpoint; a callable base URL picks a worker per call."""
        base = self._base() if callable(self._base) else self._base
        return base.rstrip("/") + "/v1/chat/completions"

    def request_body(
        self, messages: Sequence[Message], cache_salt: str | None, stream: bool
    ) -> dict[str, object]:
        """Builds the request JSON.

        Args:
            messages: Chat messages.
            cache_salt: Per-tenant prefix-cache salt, if any.
            stream: Whether to stream.

        Returns:
            The body sent to the server.
        """
        body: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if cache_salt:
            body["cache_salt"] = cache_salt
        body.update(self._extra)
        return body

    async def stream(
        self,
        messages: Sequence[Message],
        state: StreamState,
        cache_salt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yields generated text pieces as they arrive.

        Args:
            messages: Chat messages.
            state: Receives the usage and the full text.
            cache_salt: Per-tenant prefix-cache salt.

        Yields:
            Text pieces.

        Raises:
            LLMError: On HTTP errors, timeouts or malformed events.
        """
        body = self.request_body(messages, cache_salt, stream=True)
        try:
            url = self.url
        except AskIIScError as err:
            raise LLMError(f"no LLM endpoint: {err}") from err
        try:
            async with self._client.stream(
                "POST",
                url,
                json=body,
                headers=requestid.headers(),
                timeout=self._timeout,
            ) as response:
                if response.status_code != httpx.codes.OK:
                    raise LLMError(f"LLM server said {response.status_code}")
                async for line in response.aiter_lines():
                    piece = _parse_event(line, state)
                    if piece:
                        state.text.append(piece)
                        yield piece
        except httpx.HTTPError as err:
            raise LLMError(f"LLM request failed: {err}") from err

    async def complete(
        self, messages: Sequence[Message], cache_salt: str | None = None
    ) -> tuple[str, Usage]:
        """Runs a whole completion by consuming the stream."""
        state = StreamState()
        async for _ in self.stream(messages, state, cache_salt):
            pass
        return "".join(state.text), state.usage


def _parse_event(line: str, state: StreamState) -> str:
    if not line.startswith("data: "):
        return ""
    data = line.removeprefix("data: ").strip()
    if data == "[DONE]":
        return ""
    try:
        event = json.loads(data)
        usage = event.get("usage")
        if usage:
            state.usage = Usage(
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
        choices = event.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("delta", {}).get("content") or "")
    except (ValueError, AttributeError, TypeError) as err:
        raise LLMError("malformed stream event") from err
