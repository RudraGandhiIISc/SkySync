"""Queue, KV cache and throughput of the vLLM workers, for the admin page.

The serving job reads each ready worker's Prometheus `/metrics` every few
seconds. Tokens per second come from the change in
`vllm:generation_tokens_total` between two reads of the same worker. The
KV cache gauge is `vllm:kv_cache_usage_perc` in recent vLLM and
`vllm:gpu_cache_usage_perc` in older releases; either is read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

_WAITING = "vllm:num_requests_waiting"
_RUNNING = "vllm:num_requests_running"
_KV = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
_GENERATED = "vllm:generation_tokens_total"


@dataclass(frozen=True)
class WorkerReading:
    """One worker's gauges and its generated-token counter."""

    waiting: float = 0.0
    running: float = 0.0
    kv_cache: float | None = None
    generated: float | None = None


def parse_metrics(text: str) -> WorkerReading:
    """Reads the gauges this page shows from a /metrics body.

    Values summed over label sets (one vLLM process can report several
    engines); the KV cache is the fullest one.
    """
    waiting = running = 0.0
    kv: float | None = None
    generated: float | None = None
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == _WAITING:
                waiting += sample.value
            elif sample.name == _RUNNING:
                running += sample.value
            elif sample.name in _KV:
                kv = max(kv or 0.0, sample.value)
            elif sample.name == _GENERATED:
                generated = (generated or 0.0) + sample.value
    return WorkerReading(waiting, running, kv, generated)


class ServingStats:
    """The latest readings of all workers, and their token rate."""

    def __init__(self) -> None:
        self._readings: dict[str, WorkerReading] = {}
        self._last: dict[str, tuple[float, float]] = {}
        self._rates: dict[str, float] = {}

    def update(self, readings: Mapping[str, WorkerReading], now: float) -> None:
        """Replaces the readings; workers not listed are gone."""
        self._readings = dict(readings)
        for worker in set(self._last) - set(readings):
            self._last.pop(worker)
            self._rates.pop(worker, None)
        for worker, reading in readings.items():
            if reading.generated is None:
                continue
            before = self._last.get(worker)
            self._last[worker] = (now, reading.generated)
            if before is not None and now > before[0]:
                delta = reading.generated - before[1]
                self._rates[worker] = max(0.0, delta / (now - before[0]))

    def summary(self) -> dict[str, float | int | None]:
        """Waiting and running requests, fullest KV cache, tokens/s."""
        kv = [
            r.kv_cache
            for r in self._readings.values()
            if r.kv_cache is not None
        ]
        return {
            "waiting": int(sum(r.waiting for r in self._readings.values())),
            "running": int(sum(r.running for r in self._readings.values())),
            "kv_cache": max(kv) if kv else None,
            "tokens_per_s": sum(self._rates.values()) if self._rates else None,
        }
