import json

import httpx
import pytest

from askiisc.common.clock import FakeClock
from askiisc.common.targets import REFUSAL_TEXT
from askiisc.llm.fake_server import (
    NO_ANSWER_MARKER,
    FakeLLMConfig,
    PrefixCache,
    answer_for,
    create_app,
)

SYSTEM = " ".join(f"rule{i}" for i in range(64))


def _client(
    config: FakeLLMConfig | None = None, clock: FakeClock | None = None
) -> httpx.AsyncClient:
    app = create_app(config or FakeLLMConfig(), clock or FakeClock())
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://fake")


def _body(
    user: str, stream: bool, salt: str | None = None
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "fake-model",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": stream,
    }
    if salt is not None:
        body["cache_salt"] = salt
    return body


def test_answer_for_quotes_first_chunk() -> None:
    prompt = '<chunk id="a1" doc="d">x</chunk><chunk id="b2">y</chunk>'
    assert answer_for(prompt) == "According to the documents, see [chunk:a1]."
    assert answer_for("no chunks") == REFUSAL_TEXT
    assert answer_for(f'{NO_ANSWER_MARKER} <chunk id="a">') == REFUSAL_TEXT


async def test_non_streaming_completion() -> None:
    async with _client() as client:
        response = await client.post(
            "/v1/chat/completions", json=_body('<chunk id="c9">', False)
        )
    data = response.json()
    assert data["choices"][0]["message"]["content"].endswith("[chunk:c9].")
    assert data["usage"]["completion_tokens"] == 6


async def test_streaming_format_and_pace() -> None:
    clock = FakeClock()
    config = FakeLLMConfig(tokens_per_second=10.0)
    async with _client(config, clock) as client:
        response = await client.post(
            "/v1/chat/completions", json=_body('<chunk id="c1">', True)
        )
    lines = [ln for ln in response.text.split("\n\n") if ln]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(ln.removeprefix("data: ")) for ln in lines[:-1]]
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "According to the documents, see [chunk:c1]."
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"]["completion_tokens"] == 6
    assert clock.now() == pytest.approx(0.6)


async def test_no_answer_marker_streams_refusal() -> None:
    async with _client() as client:
        response = await client.post(
            "/v1/chat/completions", json=_body(NO_ANSWER_MARKER, False)
        )
    content = response.json()["choices"][0]["message"]["content"]
    assert content == REFUSAL_TEXT


async def test_metrics_count_tokens() -> None:
    async with _client() as client:
        await client.post("/v1/chat/completions", json=_body("q", False))
        text = (await client.get("/metrics")).text
    assert "vllm:prompt_tokens_total 65.0" in text
    assert "vllm:generation_tokens_total" in text
    assert "vllm:num_requests_running 0.0" in text


async def test_prefix_cache_hits_on_repeated_prefix() -> None:
    async with _client() as client:
        await client.post("/v1/chat/completions", json=_body("first", False))
        await client.post("/v1/chat/completions", json=_body("second", False))
        text = (await client.get("/metrics")).text
    line = next(
        ln
        for ln in text.splitlines()
        if ln.startswith("vllm:gpu_prefix_cache_hit_rate ")
    )
    assert float(line.split()[1]) == pytest.approx(0.5)


def test_prefix_cache_salt_separates_tenants() -> None:
    cache = PrefixCache(block_tokens=4, capacity=100)
    tokens = [str(i) for i in range(16)]
    assert cache.lookup_and_insert(tokens, "tenant-a") == 0
    assert cache.lookup_and_insert(tokens, "tenant-a") == 4
    assert cache.lookup_and_insert(tokens, "tenant-b") == 0


def test_prefix_cache_evicts_lru() -> None:
    cache = PrefixCache(block_tokens=1, capacity=2)
    cache.lookup_and_insert(["a", "b", "c"], "")
    assert cache.lookup_and_insert(["a"], "") == 0
    assert PrefixCache(1, 1).hit_rate == 0.0


async def test_delay_start_reports_loading() -> None:
    clock = FakeClock()
    config = FakeLLMConfig(delay_start_seconds=30.0)
    async with _client(config, clock) as client:
        assert (await client.get("/health")).status_code == 503
        busy = await client.post("/v1/chat/completions", json=_body("q", False))
        assert busy.status_code == 503
        clock.advance(30.0)
        assert (await client.get("/health")).status_code == 200
        models = (await client.get("/v1/models")).json()
    assert models["data"][0]["id"] == "fake-model"
