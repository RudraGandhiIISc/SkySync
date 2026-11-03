"""Queue, KV cache and token rate read from vLLM workers."""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from deploy.serve import read_workers

from askiisc.observability.vllm_stats import (
    ServingStats,
    WorkerReading,
    parse_metrics,
)

METRICS = """\
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="m"} 2.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 5.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="m"} 0.38
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{model_name="m"} 1000.0
"""


def test_metrics_are_read_from_new_and_old_names() -> None:
    assert parse_metrics(METRICS) == WorkerReading(2, 5, 0.38, 1000)
    old = METRICS.replace("kv_cache_usage_perc", "gpu_cache_usage_perc")
    assert parse_metrics(old).kv_cache == pytest.approx(0.38)
    assert parse_metrics("") == WorkerReading()


def test_token_rate_comes_from_the_counter() -> None:
    stats = ServingStats()
    assert stats.summary() == {
        "waiting": 0,
        "running": 0,
        "kv_cache": None,
        "tokens_per_s": None,
    }
    stats.update({"w1": WorkerReading(1, 2, 0.2, 100)}, now=10.0)
    assert stats.summary()["tokens_per_s"] is None  # one reading: no rate
    stats.update(
        {"w1": WorkerReading(0, 1, 0.5, 400), "w2": WorkerReading(3, 0, 0.1)},
        now=13.0,
    )
    assert stats.summary() == {
        "waiting": 3,
        "running": 1,
        "kv_cache": 0.5,
        "tokens_per_s": pytest.approx(100.0),
    }
    stats.update({"w1": WorkerReading(generated=50)}, now=14.0)  # restarted
    assert stats.summary()["tokens_per_s"] == 0.0
    stats.update({}, now=15.0)  # scaled to zero
    assert stats.summary()["tokens_per_s"] is None


async def test_ready_workers_are_polled_and_failures_skipped() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.host == "down":
            raise httpx.ConnectError("refused")
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.11.0"})
        return httpx.Response(200, text=METRICS)

    workers = [
        SimpleNamespace(worker_id="a", state="ready", url="http://up:1"),
        SimpleNamespace(worker_id="b", state="ready", url="http://down:1"),
        SimpleNamespace(worker_id="c", state="loading", url="http://up:2"),
    ]
    live: Any = SimpleNamespace(
        pool=SimpleNamespace(list_workers=lambda: workers),
        stats=ServingStats(),
        vllm_version="",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        await read_workers(live, http, now=1.0)
    assert live.vllm_version == "0.11.0"
    assert live.stats.summary()["waiting"] == 2  # only worker a  # noqa: PLR2004
