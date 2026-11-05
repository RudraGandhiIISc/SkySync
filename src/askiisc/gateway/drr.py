"""Deficit Round Robin across tenant queues (FR-11).

The cost of a request is its estimated token count (prompt plus maximum
output). The scheduler is work-conserving: a tenant alone gets everything.
`FairDispatcher` uses it to hand out a fixed number of LLM slots.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Generic, TypeVar

from askiisc.common.types import TenantId

T = TypeVar("T")


class DeficitRoundRobin(Generic[T]):
    """Classic DRR: each turn adds `quantum` to a tenant's deficit."""

    def __init__(self, quantum: float) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        self._quantum = quantum
        self._queues: dict[TenantId, deque[tuple[T, float]]] = {}
        self._deficit: dict[TenantId, float] = {}
        self._active: deque[TenantId] = deque()
        self._in_turn = False

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def enqueue(self, tenant: TenantId, item: T, cost: float) -> None:
        """Adds an item to its tenant's queue.

        Raises:
            ValueError: If the cost is not positive.
        """
        if cost <= 0:
            raise ValueError("cost must be positive")
        queue = self._queues.setdefault(tenant, deque())
        if not queue and tenant not in self._active:
            self._active.append(tenant)
            self._deficit[tenant] = 0.0
        queue.append((item, cost))

    def pop(self) -> tuple[TenantId, T] | None:
        """Returns the next item in DRR order, or None if all are empty."""
        while self._active:
            tenant = self._active[0]
            if not self._in_turn:
                self._deficit[tenant] += self._quantum
                self._in_turn = True
            queue = self._queues[tenant]
            item, cost = queue[0]
            if cost <= self._deficit[tenant]:
                queue.popleft()
                self._deficit[tenant] -= cost
                if not queue:
                    self._deficit[tenant] = 0.0
                    self._active.popleft()
                    self._in_turn = False
                return tenant, item
            self._active.rotate(-1)
            self._in_turn = False
        return None

    def queued(self, tenant: TenantId) -> int:
        """Items waiting for one tenant."""
        return len(self._queues.get(tenant, ()))


FIFO = TenantId("*")
"""The one queue every request joins when fair queueing is off."""


class FairDispatcher:
    """Grants `capacity` concurrent slots in DRR order.

    With `fair=False` (ASKIISC_FAIR_QUEUE=off, the E4 arm) every request
    joins one queue, so slots go first come, first served.
    """

    def __init__(
        self, capacity: int, quantum: float, fair: bool = True
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._fair = fair
        self._free = capacity
        self._capacity = capacity
        self._drr: DeficitRoundRobin[asyncio.Future[None]] = DeficitRoundRobin(
            quantum
        )

    @property
    def waiting(self) -> int:
        """Requests waiting for a slot."""
        return len(self._drr)

    @property
    def in_use(self) -> int:
        """Slots held by requests being generated."""
        return self._capacity - self._free

    async def acquire(self, tenant: TenantId, cost: float) -> None:
        """Waits for a slot.

        Args:
            tenant: The requester's tenant.
            cost: Estimated tokens.
        """
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._drr.enqueue(tenant if self._fair else FIFO, future, cost)
        self._dispatch()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()
            raise

    @property
    def capacity(self) -> int:
        """Total slots."""
        return self._capacity

    def resize(self, capacity: int) -> None:
        """Changes the number of slots (for example as workers come and go).

        Slots in use are not revoked; a smaller size takes effect as they
        are released.

        Raises:
            ValueError: If capacity is below 1.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._free += capacity - self._capacity
        self._capacity = capacity
        self._dispatch()

    def release(self) -> None:
        """Frees a slot and wakes the next waiter."""
        self._free += 1
        self._dispatch()

    def _dispatch(self) -> None:
        while self._free > 0:
            nxt = self._drr.pop()
            if nxt is None:
                return
            future = nxt[1]
            if future.cancelled():
                continue
            self._free -= 1
            future.set_result(None)
