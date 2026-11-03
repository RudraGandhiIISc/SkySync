"""NVML GPU sampling at 1 Hz (stands in for DCGM, which needs root).

The NVML module is injected (`pynvml` in production, a fake in tests).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from prometheus_client import CollectorRegistry, Gauge

from askiisc.autoscaler.pool import GpuInfo
from askiisc.common.clock import Clock

_MIB = 1024 * 1024


def read_gpus(
    nvml: Any, allowed: frozenset[int] | None = None
) -> list[GpuInfo]:
    """Reads utilisation and memory of every card (or the allowed ones).

    Args:
        nvml: An initialised pynvml-compatible module.
        allowed: Card indexes this job may use; None means all.

    Returns:
        One entry per card.
    """
    gpus = []
    for index in range(nvml.nvmlDeviceGetCount()):
        if allowed is not None and index not in allowed:
            continue
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append(
            GpuInfo(
                index,
                int(util),
                int(memory.free // _MIB),
                int(memory.total // _MIB),
            )
        )
    return gpus


@dataclass(frozen=True)
class GpuSample:
    """One reading of one card."""

    time: float
    index: int
    utilization: int
    used_mib: int


class GpuSampler:
    """Samples chosen cards and exports Prometheus gauges."""

    def __init__(
        self,
        read: Callable[[], list[GpuInfo]],
        clock: Clock,
        cards: Callable[[], Sequence[int]],
    ) -> None:
        self._read = read
        self._clock = clock
        self._cards = cards
        self.samples: list[GpuSample] = []
        self.registry = CollectorRegistry()
        self._util = Gauge(
            "askiisc_gpu_utilization_percent",
            "GPU use.",
            ["gpu"],
            registry=self.registry,
        )
        self._mem = Gauge(
            "askiisc_gpu_memory_used_mib",
            "GPU memory used.",
            ["gpu"],
            registry=self.registry,
        )

    def sample_once(self) -> list[GpuSample]:
        """Reads the cards in use and records them."""
        wanted = set(self._cards())
        now = self._clock.now()
        taken = [
            GpuSample(now, g.index, g.utilization, g.total_mib - g.free_mib)
            for g in self._read()
            if g.index in wanted
        ]
        for s in taken:
            self._util.labels(str(s.index)).set(s.utilization)
            self._mem.labels(str(s.index)).set(s.used_mib)
        self.samples.extend(taken)
        return taken

    async def run(
        self, interval_s: float = 1.0, until: Callable[[], bool] = lambda: False
    ) -> None:
        """Samples every interval until `until()` is true."""
        while not until():
            self.sample_once()
            await self._clock.sleep(interval_s)
