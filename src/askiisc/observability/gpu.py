"""NVML GPU sampling at 1 Hz (stands in for DCGM, which needs root).

The NVML module is injected (`pynvml` in production, a fake in tests).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Gauge

from askiisc.autoscaler.pool import GpuInfo, GpuProcess
from askiisc.common.clock import Clock

LOG = logging.getLogger(__name__)
_MIB = 1024 * 1024
# Longest parent chain followed before giving up (guards against loops).
_MAX_DEPTH = 64


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


def read_processes(nvml: Any, index: int) -> list[GpuProcess] | None:
    """Lists the compute processes NVML shows on one card.

    Args:
        nvml: An initialised pynvml-compatible module.
        index: The card.

    Returns:
        The processes (memory None where NVML hides it), or None if NVML
        refused to list them; the caller then judges by memory alone.
    """
    handle = nvml.nvmlDeviceGetHandleByIndex(index)
    try:
        running = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except nvml.NVMLError as err:
        LOG.warning("cannot list processes on GPU %d: %s", index, err)
        return None
    return [
        GpuProcess(
            int(p.pid),
            None if p.usedGpuMemory is None else int(p.usedGpuMemory // _MIB),
        )
        for p in running
    ]


def proc_parent(pid: int) -> int | None:
    """The parent pid from /proc, or None if the process is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # Field 4 follows the command name, which is in parentheses and may
    # itself contain spaces.
    return int(stat.rsplit(")", 1)[1].split()[1])


def descends_from(
    root: int, parent_of: Callable[[int], int | None] = proc_parent
) -> Callable[[int], bool]:
    """A test for "this pid is `root` or one of its descendants".

    Args:
        root: The serving job's own pid.
        parent_of: Gives a pid's parent (None when unknown).

    Returns:
        A predicate on pids.
    """

    def ours(pid: int) -> bool:
        current: int | None = pid
        for _ in range(_MAX_DEPTH):
            if current is None:
                return False
            if current == root:
                return True
            parent = parent_of(current)
            if parent == current:
                return False
            current = parent
        return False

    return ours


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
