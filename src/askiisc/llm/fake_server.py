"""A deterministic stand-in for vLLM's OpenAI-compatible server.

It speaks POST /v1/chat/completions (streaming and not), GET /v1/models,
GET /health and GET /metrics with vLLM's metric names, so the whole stack
can be tested without a GPU. Time comes from an injected Clock.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict

from askiisc.common.clock import Clock
from askiisc.common.targets import REFUSAL_TEXT

NO_ANSWER_MARKER = "[[NO_ANSWER]]"
_CHUNK_ID = re.compile(r'<chunk id="([^"]+)"')


@dataclass(frozen=True)
class FakeLLMConfig:
    """Behaviour of the fake server.

    Attributes:
        model: The served model name.
        tokens_per_second: Streaming rate, enforced through the Clock.
        delay_start_seconds: /health returns 503 until this long after start.
        block_tokens: Prefix-cache block size in whitespace tokens.
        cache_blocks: LRU capacity of the prefix cache, in blocks.
    """

    model: str = "fake-model"
    tokens_per_second: float = 100.0
    delay_start_seconds: float = 0.0
    block_tokens: int = 16
    cache_blocks: int = 4096


class ChatMessage(BaseModel):
    """One OpenAI chat message."""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class ChatRequest(BaseModel):
    """The subset of the OpenAI chat request the fake server reads."""

    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.0
    cache_salt: str | None = None


class PrefixCache:
    """An LRU of hashed prompt blocks, like vLLM's automatic prefix cache."""

    def __init__(self, block_tokens: int, capacity: int) -> None:
        self._block = block_tokens
        self._capacity = capacity
        self._blocks: OrderedDict[str, None] = OrderedDict()
        self.hits = 0
        self.queries = 0

    def lookup_and_insert(self, tokens: list[str], salt: str) -> int:
        """Counts cached leading blocks of a prompt, then caches all blocks.

        Args:
            tokens: The prompt tokens.
            salt: Per-tenant cache salt; different salts never share blocks.

        Returns:
            The number of leading full blocks that were already cached.
        """
        digest = hashlib.sha256(salt.encode())
        hit_blocks, still_hitting = 0, True
        full = len(tokens) // self._block
        for i in range(full):
            block = tokens[i * self._block : (i + 1) * self._block]
            digest.update(" ".join(block).encode())
            key = digest.hexdigest()
            self.queries += 1
            if still_hitting and key in self._blocks:
                hit_blocks += 1
                self.hits += 1
                self._blocks.move_to_end(key)
            else:
                still_hitting = False
                self._blocks[key] = None
                if len(self._blocks) > self._capacity:
                    self._blocks.popitem(last=False)
        return hit_blocks

    @property
    def hit_rate(self) -> float:
        """Fraction of looked-up blocks that were cached."""
        return self.hits / self.queries if self.queries else 0.0


def answer_for(prompt: str) -> str:
    """Builds the deterministic answer text for a prompt.

    Args:
        prompt: All message contents joined.

    Returns:
        The refusal for the no-answer marker or a prompt without chunks,
        otherwise a sentence citing the first chunk.
    """
    if NO_ANSWER_MARKER in prompt:
        return REFUSAL_TEXT
    match = _CHUNK_ID.search(prompt)
    if match is None:
        return REFUSAL_TEXT
    return f"According to the documents, see [chunk:{match.group(1)}]."


class _Metrics:
    def __init__(self, cache: PrefixCache) -> None:
        self.registry = CollectorRegistry()
        self.prompt = Counter(
            "vllm:prompt_tokens", "Prompt tokens.", registry=self.registry
        )
        self.generation = Counter(
            "vllm:generation_tokens",
            "Generated tokens.",
            registry=self.registry,
        )
        self.running = Gauge(
            "vllm:num_requests_running",
            "Requests running.",
            registry=self.registry,
        )
        hit_rate = Gauge(
            "vllm:gpu_prefix_cache_hit_rate",
            "Prefix cache hit rate.",
            registry=self.registry,
        )
        hit_rate.set_function(lambda: cache.hit_rate)


def create_app(config: FakeLLMConfig, clock: Clock) -> FastAPI:
    """Builds the fake vLLM app.

    Args:
        config: Behaviour settings.
        clock: Time source for streaming pace and start-up delay.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="fake-vllm")
    started = clock.now()
    cache = PrefixCache(config.block_tokens, config.cache_blocks)
    metrics = _Metrics(cache)
    app.state.prefix_cache = cache

    def ready() -> bool:
        return clock.now() - started >= config.delay_start_seconds

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200 if ready() else 503)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": config.model}]}

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(
            generate_latest(metrics.registry), media_type="text/plain"
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(request: ChatRequest) -> Response:
        if not ready():
            return JSONResponse({"error": "loading"}, status_code=503)
        prompt = "\n".join(m.content for m in request.messages)
        prompt_tokens = prompt.split()
        cache.lookup_and_insert(prompt_tokens, request.cache_salt or "")
        metrics.prompt.inc(len(prompt_tokens))
        words = answer_for(prompt).split(" ")[: request.max_tokens]
        if request.stream:
            return StreamingResponse(
                _stream(
                    words, len(prompt_tokens), request, config, clock, metrics
                ),
                media_type="text/event-stream",
            )
        metrics.generation.inc(len(words))
        return JSONResponse(
            _completion(words, len(prompt_tokens), request.model)
        )

    return app


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion(words: list[str], prompt_tokens: int, model: str) -> object:
    return {
        "id": "cmpl-fake",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": " ".join(words)},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(prompt_tokens, len(words)),
    }


async def _stream(
    words: list[str],
    prompt_tokens: int,
    request: ChatRequest,
    config: FakeLLMConfig,
    clock: Clock,
    metrics: _Metrics,
) -> AsyncIterator[bytes]:
    metrics.running.inc()
    try:
        for i, word in enumerate(words):
            await clock.sleep(1.0 / config.tokens_per_second)
            piece = word if i == 0 else " " + word
            metrics.generation.inc()
            yield _sse(_delta(request.model, {"content": piece}, None))
        final = _delta(request.model, {}, "stop")
        final["usage"] = _usage(prompt_tokens, len(words))
        yield _sse(final)
        yield b"data: [DONE]\n\n"
    finally:
        metrics.running.dec()


def _delta(
    model: str, delta: dict[str, str], finish: str | None
) -> dict[str, object]:
    return {
        "id": "cmpl-fake",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(payload: object) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
