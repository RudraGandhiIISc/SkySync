"""Per-tenant load limits and admission control (FR-6).

A request is admitted only if its tenant's token bucket has a token, its
tenant queue is below the bound, its estimated wait is under the maximum
and, with a capacity (`max_in_flight`), the tenant is within its fair share
of it. Otherwise RateLimited is raised at once, with a Retry-After, and
nothing waits on the LLM.

Fair share (FR-6, FR-11): every tenant with requests in flight may hold
max_in_flight / (number of such tenants). A tenant below its share is
always admitted, so a quiet department is never locked out by a busy one
(the total can exceed the capacity by at most one request per tenant); a
tenant at its share is refused while others are active or the server is
full.

Pool-aware wait (FR-6, FR-10): the wait estimate divides by the LLM slots
of the GPU workers ready now (`concurrency` slots per worker, times
`ready_workers()`). With no ready worker the wait is not estimated: the
request goes on to the cold-start hold, which bounds it, so an empty pool
never causes a refusal by itself.

With `enabled=False` (ASKIISC_ADMISSION=off, the E4 arm) every request is
admitted and queues without bound.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from askiisc.common.clock import Clock
from askiisc.common.errors import RateLimited
from askiisc.common.types import TenantId

_EWMA_WEIGHT = 0.2


class TokenBucket:
    """A token bucket: `rate` tokens per second, at most `burst` saved."""

    def __init__(self, rate: float, burst: float, clock: Clock) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate and burst must be positive")
        self._rate = rate
        self._burst = burst
        self._clock = clock
        self._tokens = burst
        self._last = clock.now()

    def _refill(self) -> None:
        now = self._clock.now()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last = now

    def try_take(self, cost: float = 1.0) -> bool:
        """Takes `cost` tokens if available.

        Returns:
            True if the tokens were taken.
        """
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    def retry_after(self, cost: float = 1.0) -> float:
        """Seconds until `cost` tokens will be available."""
        self._refill()
        return max(0.0, (cost - self._tokens) / self._rate)


@dataclass(frozen=True)
class Ticket:
    """Proof of admission; hand it back to `release`."""

    tenant: TenantId
    admitted_at: float


class Admission:
    """Admission control across tenants."""

    def __init__(
        self,
        clock: Clock,
        rate_per_s: float,
        burst: float,
        queue_max: int,
        max_wait_s: float,
        concurrency: int,
        service_estimate_s: float = 1.0,
        max_in_flight: int | None = None,
        ready_workers: Callable[[], int] | None = None,
        enabled: bool = True,
    ) -> None:
        """Creates admission control.

        Args:
            clock: Time source.
            rate_per_s: Token-bucket rate per tenant.
            burst: Token-bucket size per tenant.
            queue_max: Most unfinished requests per tenant.
            max_wait_s: Longest estimated wait admitted.
            concurrency: LLM slots per ready worker (all slots when
                `ready_workers` is None).
            service_estimate_s: First guess of a request's duration.
            max_in_flight: The gateway's capacity, shared fairly between
                tenants (None: no capacity check).
            ready_workers: Ready GPU workers now (None: the slots do not
                depend on the pool).
            enabled: False admits everything (no limits at all).
        """
        self._clock = clock
        self._rate = rate_per_s
        self._burst = burst
        self._queue_max = queue_max
        self._max_wait = max_wait_s
        self._concurrency = concurrency
        self._service_s = service_estimate_s
        self._capacity = max_in_flight
        self._ready = ready_workers
        self._enabled = enabled
        self._buckets: dict[TenantId, TokenBucket] = {}
        self._depth: defaultdict[TenantId, int] = defaultdict(int)

    def _bucket(self, tenant: TenantId) -> TokenBucket:
        if tenant not in self._buckets:
            self._buckets[tenant] = TokenBucket(
                self._rate, self._burst, self._clock
            )
        return self._buckets[tenant]

    def slots(self) -> int:
        """LLM slots of the workers ready now."""
        if self._ready is None:
            return self._concurrency
        return self._concurrency * max(0, self._ready())

    def estimated_wait_s(self) -> float:
        """Expected queueing delay for a request admitted now.

        Returns:
            Seconds; 0 when no worker is ready (the cold hold bounds it).
        """
        slots = self.slots()
        if slots == 0:
            return 0.0
        in_system = sum(self._depth.values())
        waiting = max(0, in_system + 1 - slots)
        return waiting * self._service_s / slots

    def share(self, tenant: TenantId) -> int | None:
        """The tenant's fair share of the capacity now (None: no capacity)."""
        if self._capacity is None:
            return None
        active = {t for t, d in self._depth.items() if d > 0} | {tenant}
        return max(1, self._capacity // len(active))

    def _over_share(self, tenant: TenantId) -> bool:
        share = self.share(tenant)
        if share is None or self._depth[tenant] < share:
            return False
        others = any(d > 0 for t, d in self._depth.items() if t != tenant)
        full = sum(self._depth.values()) >= (self._capacity or 0)
        return others or full

    def _owed(self, tenant: TenantId) -> bool:
        """Whether a tenant is below its fair share while others are active.

        The fair queue serves such a tenant first (FR-11), so the wait
        estimate, which assumes first come first served, does not apply to
        it. Run 2259 (6 LLM slots): with it, the quiet department was
        refused like the busy one and got 11% of the tokens.
        """
        share = self.share(tenant)
        others = any(d > 0 for t, d in self._depth.items() if t != tenant)
        return share is not None and others and self._depth[tenant] < share

    def admit(self, tenant: TenantId) -> Ticket:
        """Admits a request or rejects it immediately.

        Args:
            tenant: The requester's tenant.

        Returns:
            A ticket to release when the request finishes.

        Raises:
            RateLimited: If any limit would be exceeded.
        """
        if not self._enabled:
            self._depth[tenant] += 1
            return Ticket(tenant, self._clock.now())
        if self._depth[tenant] >= self._queue_max:
            raise RateLimited("tenant queue full", self._max_wait)
        if self._over_share(tenant):
            raise RateLimited("server busy: tenant at its fair share", 1.0)
        wait = self.estimated_wait_s()
        if wait > self._max_wait and not self._owed(tenant):
            raise RateLimited("estimated wait too long", wait - self._max_wait)
        bucket = self._bucket(tenant)
        if not bucket.try_take():
            raise RateLimited("tenant rate limit", bucket.retry_after())
        self._depth[tenant] += 1
        return Ticket(tenant, self._clock.now())

    def release(self, ticket: Ticket, completed: bool = True) -> None:
        """Returns a ticket and updates the service-time estimate.

        Args:
            ticket: The admission ticket.
            completed: Whether the request was served (only served requests
                update the estimate).
        """
        self._depth[ticket.tenant] = max(0, self._depth[ticket.tenant] - 1)
        if completed:
            took = self._clock.now() - ticket.admitted_at
            self._service_s += _EWMA_WEIGHT * (took - self._service_s)

    def depth(self, tenant: TenantId) -> int:
        """Requests from a tenant currently admitted and unfinished."""
        return self._depth[tenant]
