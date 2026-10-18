"""Per-tenant load limits and admission control (FR-6).

A request is admitted only if its tenant's token bucket has a token, its
tenant queue is below the bound, and its estimated wait is under the
maximum. Otherwise RateLimited is raised at once, with a Retry-After, and
nothing waits on the LLM.
"""

from __future__ import annotations

from collections import defaultdict
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
    ) -> None:
        self._clock = clock
        self._rate = rate_per_s
        self._burst = burst
        self._queue_max = queue_max
        self._max_wait = max_wait_s
        self._concurrency = concurrency
        self._service_s = service_estimate_s
        self._buckets: dict[TenantId, TokenBucket] = {}
        self._depth: defaultdict[TenantId, int] = defaultdict(int)

    def _bucket(self, tenant: TenantId) -> TokenBucket:
        if tenant not in self._buckets:
            self._buckets[tenant] = TokenBucket(
                self._rate, self._burst, self._clock
            )
        return self._buckets[tenant]

    def estimated_wait_s(self) -> float:
        """Expected queueing delay for a request admitted now."""
        in_system = sum(self._depth.values())
        waiting = max(0, in_system + 1 - self._concurrency)
        return waiting * self._service_s / self._concurrency

    def admit(self, tenant: TenantId) -> Ticket:
        """Admits a request or rejects it immediately.

        Args:
            tenant: The requester's tenant.

        Returns:
            A ticket to release when the request finishes.

        Raises:
            RateLimited: If any limit would be exceeded.
        """
        if self._depth[tenant] >= self._queue_max:
            raise RateLimited("tenant queue full", self._max_wait)
        wait = self.estimated_wait_s()
        if wait > self._max_wait:
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
