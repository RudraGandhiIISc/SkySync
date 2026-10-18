"""A discrete-event clock for simulating concurrent coroutines.

`sleep` parks the caller until the simulation reaches its wake-up time.
`run_until` lets every runnable task reach its next sleep, then jumps time
to the earliest wake-up, and repeats.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools


class SimClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        heapq.heappush(
            self._heap, (self._now + max(seconds, 0.0), next(self._seq), future)
        )
        await future

    async def _settle(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0)

    async def run_until(self, end: float) -> None:
        await self._settle()
        while self._heap and self._heap[0][0] <= end:
            when, _, future = heapq.heappop(self._heap)
            self._now = max(self._now, when)
            if not future.cancelled():
                future.set_result(None)
            await self._settle()
        self._now = max(self._now, end)
