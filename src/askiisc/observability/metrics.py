"""Prometheus metrics. Each component gets its own registry (no globals).

No metric label or value ever carries query or answer text.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.8,
    1.0,
    2.0,
    3.5,
    5.0,
    10.0,
    30.0,
    90.0,
    150.0,
)


class GatewayMetrics:
    """Gateway counters, gauges and histograms."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "askiisc_requests",
            "Requests by outcome.",
            ["tenant", "status"],
            registry=r,
        )
        self.shed = Counter(
            "askiisc_shed",
            "Requests rejected with 429.",
            ["tenant"],
            registry=r,
        )
        self.latency = Histogram(
            "askiisc_latency_seconds",
            "End-to-end latency.",
            ["tenant"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "askiisc_ttft_seconds",
            "Time to first token.",
            ["tenant"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "askiisc_output_tokens",
            "Output tokens served.",
            ["tenant"],
            registry=r,
        )
        self.cache_hits = Counter(
            "askiisc_cache_hits", "Answer cache hits.", registry=r
        )
        self.queue_depth = Gauge(
            "askiisc_fair_queue_depth",
            "Requests waiting for an LLM slot.",
            registry=r,
        )
        self.cold_holds = Counter(
            "askiisc_cold_start_holds",
            "Requests held for a cold start.",
            registry=r,
        )

    def render(self) -> bytes:
        """Returns the Prometheus text exposition."""
        return generate_latest(self.registry)
