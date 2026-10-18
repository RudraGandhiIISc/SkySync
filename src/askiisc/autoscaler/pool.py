"""GPU worker pools.

On the CSA server Slurm reserves no GPU, so a "GPU node" is a `vllm serve`
process pinned to one H200 with CUDA_VISIBLE_DEVICES (docs/SERVER.md).
`ProcessPool` starts and stops those processes; `FakeWorkerPool` simulates
acquisition and load times for tests. A SlurmPool (one job per worker) is
not built: it needs the admin's permission first.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import httpx

from askiisc.common.clock import Clock
from askiisc.common.errors import AskIIScError
from askiisc.common.targets import MAX_GPU_WORKERS


class WorkerState(StrEnum):
    """Lifecycle of a worker."""

    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"


class PoolError(AskIIScError):
    """A worker could not be started."""


@dataclass
class Worker:
    """One GPU worker and its lifecycle timestamps (for the cost model)."""

    worker_id: str
    url: str = ""
    gpu: int | None = None
    state: WorkerState = WorkerState.STARTING
    requested_at: float = 0.0
    acquired_at: float | None = None
    ready_at: float | None = None
    released_at: float | None = None


class WorkerPool(Protocol):
    """Starts and stops GPU workers."""

    async def request_worker(self) -> Worker:
        """Starts a worker and returns it once ready."""
        ...

    async def release_worker(self, worker_id: str) -> None:
        """Stops a worker."""
        ...

    def list_workers(self) -> list[Worker]:
        """Returns workers that are not released."""
        ...


class FakeWorkerPool:
    """Simulated pool with configurable acquire and load times."""

    def __init__(
        self,
        clock: Clock,
        acquire_s: float = 5.0,
        load_s: float = 30.0,
        max_workers: int = MAX_GPU_WORKERS,
        base_url: str | None = None,
    ) -> None:
        self._clock = clock
        self._base_url = base_url
        self.acquire_s = acquire_s
        self.load_s = load_s
        self._max = max_workers
        self.fail_next = 0
        self._workers: dict[str, Worker] = {}
        self.history: list[Worker] = []
        self._next = 0

    async def request_worker(self) -> Worker:
        """Starts a simulated worker.

        Raises:
            PoolError: If the pool is full or a failure was scripted.
        """
        if len(self._workers) >= self._max:
            raise PoolError("pool is at its maximum size")
        self._next += 1
        worker = Worker(
            f"fake-{self._next}",
            url=self._base_url or f"http://fake-{self._next}",
            requested_at=self._clock.now(),
        )
        self._workers[worker.worker_id] = worker
        self.history.append(worker)
        await self._clock.sleep(self.acquire_s)
        worker.acquired_at, worker.state = (
            self._clock.now(),
            WorkerState.LOADING,
        )
        if self.fail_next > 0:
            self.fail_next -= 1
            await self.release_worker(worker.worker_id)
            raise PoolError("scripted start failure")
        await self._clock.sleep(self.load_s)
        worker.ready_at, worker.state = self._clock.now(), WorkerState.READY
        return worker

    async def release_worker(self, worker_id: str) -> None:
        """Stops a simulated worker."""
        worker = self._workers.pop(worker_id, None)
        if worker is not None:
            worker.released_at = self._clock.now()

    def list_workers(self) -> list[Worker]:
        """Returns live workers."""
        return list(self._workers.values())

    def kill(self, worker_id: str) -> None:
        """Simulates a node failure."""
        worker = self._workers.pop(worker_id)
        worker.released_at = self._clock.now()


@dataclass(frozen=True)
class GpuInfo:
    """One card as NVML reports it."""

    index: int
    utilization: int
    free_mib: int
    total_mib: int


def choose_gpu(gpus: Sequence[GpuInfo], exclude: set[int]) -> GpuInfo | None:
    """Picks the least busy card by utilisation, then free memory."""
    candidates = [g for g in gpus if g.index not in exclude]
    if not candidates:
        return None
    return min(candidates, key=lambda g: (g.utilization, -g.free_mib, g.index))


def memory_fraction(
    gpu: GpuInfo, headroom_mib: int = 4096, cap: float = 0.90
) -> float:
    """VLLM --gpu-memory-utilization that fits in the card's free memory."""
    return round(min(cap, (gpu.free_mib - headroom_mib) / gpu.total_mib), 2)


class Process(Protocol):
    """The subset of asyncio.subprocess.Process the pool uses."""

    @property
    def pid(self) -> int:
        """The process id (also its process group id)."""
        ...

    @property
    def returncode(self) -> int | None:
        """The exit code, or None while running."""
        ...

    async def wait(self) -> int:
        """Waits for exit."""
        ...


Launcher = Callable[[list[str], dict[str, str]], Awaitable[Process]]


async def launch(cmd: list[str], env: dict[str, str]) -> Process:
    """Starts a process in its own process group, output to a log file."""
    with open(env["ASKIISC_WORKER_LOG"], "ab") as log:  # noqa: ASYNC230
        return await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )


@dataclass
class ProcessPoolConfig:
    """How ProcessPool starts vLLM."""

    vllm_bin: str
    model_args: list[str]
    log_dir: str
    max_workers: int = MAX_GPU_WORKERS
    max_model_len: int = 4096
    min_memory_fraction: float = 0.25
    load_timeout_s: float = 900.0
    poll_s: float = 2.0
    acquire_delay_s: float = 0.0
    stop_grace_s: float = 30.0
    extra_env: dict[str, str] = field(default_factory=dict)


class ProcessPool:
    """Runs `vllm serve` processes on pinned GPUs, at most max_workers."""

    def __init__(
        self,
        config: ProcessPoolConfig,
        clock: Clock,
        http: httpx.AsyncClient,
        gpus: Callable[[], list[GpuInfo]],
        ports: Callable[[], int],
        launcher: Launcher = launch,
    ) -> None:
        if not 0 < config.max_workers <= MAX_GPU_WORKERS:
            raise ValueError("max_workers must be 1 or 2")
        self._config = config
        self._clock = clock
        self._http = http
        self._gpus = gpus
        self._ports = ports
        self._launch = launcher
        self._workers: dict[str, tuple[Worker, Process]] = {}
        self._next = 0

    def command(self, port: int, fraction: float) -> list[str]:
        """The vLLM command line for one worker."""
        c = self._config
        return [
            c.vllm_bin,
            "serve",
            *c.model_args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-prefix-caching",
            "--max-model-len",
            str(c.max_model_len),
            "--gpu-memory-utilization",
            f"{fraction:.2f}",
        ]

    async def request_worker(self) -> Worker:
        """Starts a worker on the least busy free card and waits until healthy.

        Raises:
            PoolError: If the pool is full, no card has room, or the worker
                exits or does not become healthy in time.
        """
        if len(self._workers) >= self._config.max_workers:
            raise PoolError("pool is at its maximum size")
        self._next += 1
        worker = Worker(f"vllm-{self._next}", requested_at=self._clock.now())
        gpu = self._pick(worker)
        await self._clock.sleep(self._config.acquire_delay_s)
        port = self._ports()
        worker.url, worker.acquired_at = (
            f"http://127.0.0.1:{port}",
            self._clock.now(),
        )
        env = {
            **os.environ,
            **self._config.extra_env,
            "CUDA_VISIBLE_DEVICES": str(gpu.index),
            "HF_HUB_OFFLINE": "1",
            "ASKIISC_WORKER_LOG": self._log_path(worker),
        }
        process = await self._launch(
            self.command(port, memory_fraction(gpu)), env
        )
        self._workers[worker.worker_id] = (worker, process)
        worker.state = WorkerState.LOADING
        await self._wait_healthy(worker, process)
        return worker

    def _log_path(self, worker: Worker) -> str:
        return f"{self._config.log_dir}/{worker.worker_id}.log"

    def _pick(self, worker: Worker) -> GpuInfo:
        used = {w.gpu for w, _ in self._workers.values() if w.gpu is not None}
        gpu = choose_gpu(self._gpus(), used)
        if (
            gpu is None
            or memory_fraction(gpu) < self._config.min_memory_fraction
        ):
            raise PoolError("no GPU with enough free memory")
        worker.gpu = gpu.index
        return gpu

    async def _wait_healthy(self, worker: Worker, process: Process) -> None:
        deadline = self._clock.now() + self._config.load_timeout_s
        while self._clock.now() < deadline:
            if process.returncode is not None:
                await self.release_worker(worker.worker_id)
                raise PoolError(f"{worker.worker_id} exited during start-up")
            with contextlib.suppress(httpx.HTTPError):
                response = await self._http.get(
                    worker.url + "/health", timeout=5.0
                )
                if response.status_code == httpx.codes.OK:
                    worker.ready_at, worker.state = (
                        self._clock.now(),
                        WorkerState.READY,
                    )
                    return
            await self._clock.sleep(self._config.poll_s)
        await self.release_worker(worker.worker_id)
        raise PoolError(f"{worker.worker_id} not healthy in time")

    async def release_worker(self, worker_id: str) -> None:
        """Terminates a worker's process group (SIGTERM, then SIGKILL)."""
        entry = self._workers.pop(worker_id, None)
        if entry is None:
            return
        worker, process = entry
        worker.state = WorkerState.DRAINING
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self._config.stop_grace_s
                )
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        worker.released_at = self._clock.now()

    def list_workers(self) -> list[Worker]:
        """Returns live workers."""
        return [w for w, _ in self._workers.values()]

    async def close(self) -> None:
        """Stops every worker."""
        for worker_id in list(self._workers):
            await self.release_worker(worker_id)
