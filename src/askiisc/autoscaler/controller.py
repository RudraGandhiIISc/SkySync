"""The autoscaler controller (FR-9, FR-10).

It tracks arrivals, asks the pure policy what to do every interval, starts
and releases workers through a WorkerPool, and tells the gateway whether a
worker is ready (it implements the gateway's WorkerStatus).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections import deque
from collections.abc import Callable

from askiisc.autoscaler.policy import (
    Decision,
    Hold,
    Observation,
    ScaleIn,
    ScaleOut,
    ScalingConfig,
    decide,
    needed_workers,
)
from askiisc.autoscaler.pool import PoolError, Worker, WorkerPool, WorkerState
from askiisc.common.clock import Clock

_LOG = logging.getLogger(__name__)


class DemandTracker:
    """Arrival times over a sliding window."""

    def __init__(self, clock: Clock, window_s: float = 30.0) -> None:
        self._clock = clock
        self._window = window_s
        self._arrivals: deque[float] = deque()
        self.last_arrival: float | None = None

    def note(self) -> None:
        """Records one arrival now."""
        now = self._clock.now()
        self._arrivals.append(now)
        self.last_arrival = now

    def rate(self) -> float:
        """Arrivals per second over the window."""
        cutoff = self._clock.now() - self._window
        while self._arrivals and self._arrivals[0] < cutoff:
            self._arrivals.popleft()
        return len(self._arrivals) / self._window


class Controller:
    """Drives a WorkerPool from demand."""

    def __init__(
        self,
        pool: WorkerPool,
        config: ScalingConfig,
        clock: Clock,
        in_flight: Callable[[], int] = lambda: 0,
        queue_depth: Callable[[], int] = lambda: 0,
        on_ready_change: Callable[[int], None] | None = None,
        window_s: float = 30.0,
        poll_s: float = 0.5,
    ) -> None:
        self._pool = pool
        self._config = config
        self._clock = clock
        self._in_flight = in_flight
        self._queue_depth = queue_depth
        self._on_ready_change = on_ready_change
        self._poll = poll_s
        self.demand = DemandTracker(clock, window_s)
        self.history: list[Worker] = []
        self.failures = 0
        self._starting = 0
        self._tasks: set[asyncio.Task[None]] = set()
        self._last_action: float | None = None
        self._low_since: float | None = None
        self._rr = itertools.count()
        self._ready_count = 0
        self._closed = False

    # WorkerStatus (used by the gateway).

    def note_request(self) -> None:
        """Records an arrival."""
        self.demand.note()

    def ready_workers(self) -> list[Worker]:
        """Workers that can serve now."""
        return [
            w for w in self._pool.list_workers() if w.state == WorkerState.READY
        ]

    def ready(self) -> bool:
        """True if a worker can serve now."""
        return bool(self.ready_workers())

    def ready_count(self) -> int:
        """How many workers can serve now (recorded in each audit record)."""
        return len(self.ready_workers())

    async def wait_ready(self, timeout_s: float) -> bool:
        """Waits until a worker is ready, or the timeout passes."""
        deadline = self._clock.now() + timeout_s
        while not self.ready():
            if self._clock.now() >= deadline:
                return False
            await self._clock.sleep(self._poll)
        return True

    def endpoint(self) -> str:
        """A ready worker's URL, round robin.

        Raises:
            PoolError: If no worker is ready.
        """
        ready = self.ready_workers()
        if not ready:
            raise PoolError("no ready worker")
        return ready[next(self._rr) % len(ready)].url

    # Control loop.

    def observe(self) -> Observation:
        """Builds the policy input from current state."""
        now = self._clock.now()
        active = [
            w
            for w in self._pool.list_workers()
            if w.state != WorkerState.DRAINING
        ]
        workers = max(len(active), self._starting)
        obs = Observation(
            now=now,
            request_rate=self.demand.rate(),
            queue_depth=self._queue_depth(),
            in_flight=self._in_flight(),
            workers=workers,
            last_traffic=self.demand.last_arrival,
            last_action=self._last_action,
            low_since=self._low_since,
        )
        low = needed_workers(obs, self._config) < workers
        self._low_since = (self._low_since or now) if low else None
        return obs

    async def step(self) -> Decision:
        """Reaps dead workers, then observes, decides and acts once."""
        reap = getattr(self._pool, "reap", None)
        if reap is not None:
            for worker_id in await reap():
                self.failures += 1
                _LOG.warning("worker %s died; removed", worker_id)
        obs = self.observe()
        if self._closed:
            return Hold()
        decision = decide(obs, self._config)
        if isinstance(decision, ScaleOut):
            for _ in range(decision.count):
                self._spawn()
            self._last_action = obs.now
        elif isinstance(decision, ScaleIn):
            await self._release(decision.count)
            self._last_action = obs.now
        self._notify()
        return decision

    def _spawn(self) -> None:
        self._starting += 1
        task = asyncio.create_task(self._start_one())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _start_one(self) -> None:
        try:
            worker = await self._pool.request_worker()
            self.history.append(worker)
        except PoolError as err:
            self.failures += 1
            _LOG.warning("worker start failed: %s", err)
        finally:
            self._starting -= 1
            self._notify()

    async def _release(self, count: int) -> None:
        ready = sorted(self.ready_workers(), key=lambda w: w.ready_at or 0.0)
        for worker in ready[-count:]:
            await self._pool.release_worker(worker.worker_id)

    def _notify(self) -> None:
        count = len(self.ready_workers())
        if count != self._ready_count:
            self._ready_count = count
            if self._on_ready_change is not None:
                self._on_ready_change(count)

    async def run(
        self, interval_s: float = 1.0, until: Callable[[], bool] = lambda: False
    ) -> None:
        """Steps every interval until `until()` is true."""
        while not until() and not self._closed:
            await self.step()
            await self._clock.sleep(interval_s)

    async def close(self) -> None:
        """Stops the loop, cancels start-ups and releases every worker."""
        self._closed = True
        for task in list(self._tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for worker in list(self._pool.list_workers()):
            await self._pool.release_worker(worker.worker_id)
