"""FR-6 load limits and FR-11 fairness."""

import asyncio
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from askiisc.common.clock import FakeClock
from askiisc.common.errors import RateLimited
from askiisc.common.targets import (
    FR11_MAX_HEAVY_SHARE,
    FR11_MIN_LIGHT_SHARE,
)
from askiisc.common.types import TenantId
from askiisc.gateway.drr import DeficitRoundRobin, FairDispatcher
from askiisc.gateway.limits import Admission, TokenBucket

A, B, C = TenantId("a"), TenantId("b"), TenantId("c")


@settings(max_examples=50, deadline=None)
@given(
    rate=st.floats(0.5, 50),
    burst=st.floats(1, 50),
    gaps=st.lists(st.floats(0, 2), min_size=1, max_size=200),
)
def test_bucket_never_exceeds_rate_times_time_plus_burst(
    rate: float, burst: float, gaps: list[float]
) -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate, burst, clock)
    taken = 0
    for gap in gaps:
        clock.advance(gap)
        taken += bucket.try_take()
    assert taken <= rate * clock.now() + burst + 1e-9


def test_bucket_retry_after_and_validation() -> None:
    clock = FakeClock()
    bucket = TokenBucket(rate=2.0, burst=1.0, clock=clock)
    assert bucket.try_take()
    assert not bucket.try_take()
    assert bucket.retry_after() == pytest.approx(0.5)
    clock.advance(0.5)
    assert bucket.retry_after() == 0.0
    with pytest.raises(ValueError):
        TokenBucket(0, 1, clock)


def test_admission_rejects_on_rate() -> None:
    clock = FakeClock()
    adm = Admission(
        clock,
        rate_per_s=1,
        burst=2,
        queue_max=10,
        max_wait_s=100,
        concurrency=10,
    )
    adm.admit(A)
    adm.admit(A)
    with pytest.raises(RateLimited) as err:
        adm.admit(A)
    assert err.value.retry_after_s == pytest.approx(1.0)
    adm.admit(B)  # other tenants keep their own bucket


def test_admission_queue_bound_and_release() -> None:
    clock = FakeClock()
    adm = Admission(
        clock,
        rate_per_s=100,
        burst=100,
        queue_max=2,
        max_wait_s=100,
        concurrency=10,
    )
    t1, _ = adm.admit(A), adm.admit(A)
    with pytest.raises(RateLimited, match="queue full"):
        adm.admit(A)
    adm.release(t1, completed=False)
    assert adm.depth(A) == 1
    adm.admit(A)


def test_admission_rejects_when_wait_too_long() -> None:
    clock = FakeClock()
    adm = Admission(
        clock,
        rate_per_s=100,
        burst=100,
        queue_max=100,
        max_wait_s=2.0,
        concurrency=2,
        service_estimate_s=1.0,
    )
    for _ in range(5):
        adm.admit(A)  # estimated waits 0, 0, 0.5, 1.0, 1.5
    adm.admit(B)
    with pytest.raises(RateLimited, match="wait"):
        adm.admit(C)
    assert adm.estimated_wait_s() == pytest.approx(2.5)


def test_service_estimate_tracks_completions() -> None:
    clock = FakeClock()
    adm = Admission(
        clock, 100, 100, 100, 100, concurrency=1, service_estimate_s=1.0
    )
    ticket = adm.admit(A)
    clock.advance(6.0)
    adm.release(ticket)
    adm.admit(A)
    assert adm.estimated_wait_s() == pytest.approx(2.0)


def test_rejection_does_not_await() -> None:
    """admit() is synchronous: a 429 never waits on the LLM (FR-6)."""
    assert not asyncio.iscoroutinefunction(Admission.admit)


def test_drr_validation_and_empty() -> None:
    with pytest.raises(ValueError):
        DeficitRoundRobin[int](0)
    drr = DeficitRoundRobin[int](10)
    with pytest.raises(ValueError):
        drr.enqueue(A, 1, 0)
    assert drr.pop() is None


def test_drr_alone_tenant_gets_everything() -> None:
    drr = DeficitRoundRobin[int](100)
    for i in range(50):
        drr.enqueue(A, i, 300)
    order = [drr.pop() for _ in range(50)]
    assert [item for _, item in order] == list(range(50))  # type: ignore[misc]
    assert len(drr) == 0


def _served_shares(
    load: dict[TenantId, float], seconds: float
) -> dict[TenantId, float]:
    """Simulates one LLM slot under DRR with Poisson-free fixed arrivals.

    Each tenant offers `load` requests/s of 200 tokens; the slot serves
    1000 tokens/s. Returns each tenant's share of served tokens.
    """
    drr = DeficitRoundRobin[float](quantum=200)
    served: Counter[TenantId] = Counter()
    next_arrival = dict.fromkeys(load, 0.0)
    now, cost, speed = 0.0, 200.0, 1000.0
    while now < seconds:
        for t, rate in load.items():
            while next_arrival[t] <= now:
                drr.enqueue(t, next_arrival[t], cost)
                next_arrival[t] += 1.0 / rate
        nxt = drr.pop()
        if nxt is None:
            now = min(next_arrival.values())
            continue
        served[nxt[0]] += cost
        now += cost / speed
    total = sum(served.values())
    return {t: served[t] / total for t in load}


def test_fr11_ten_to_one_overload_shares() -> None:
    # Capacity 5 req/s; light offers 3 req/s, heavy 30 req/s (10x).
    shares = _served_shares({A: 30.0, B: 3.0}, seconds=600)
    assert shares[A] <= FR11_MAX_HEAVY_SHARE
    assert shares[B] >= FR11_MIN_LIGHT_SHARE


def test_fr11_three_tenants_and_idle_tenant() -> None:
    shares = _served_shares({A: 40.0, B: 4.0, C: 4.0}, seconds=600)
    assert max(shares.values()) <= FR11_MAX_HEAVY_SHARE
    alone = _served_shares({A: 40.0}, seconds=60)
    assert alone[A] == 1.0


@settings(max_examples=60, deadline=None)
@given(
    costs=st.lists(
        st.tuples(st.sampled_from([A, B, C]), st.integers(500, 1000)),
        min_size=1,
        max_size=150,
    )
)
def test_drr_no_queue_waits_more_than_n_rounds(
    costs: list[tuple[TenantId, int]],
) -> None:
    """With quantum >= every cost, each visit serves 1 to 3 items, so a
    backlogged tenant waits at most 2 other visits: at most 6 items."""
    drr = DeficitRoundRobin[int](quantum=1000)
    for i, (tenant, cost) in enumerate(costs):
        drr.enqueue(tenant, i, cost)
    last_served = dict.fromkeys((A, B, C), 0)
    served = 0
    while (nxt := drr.pop()) is not None:
        served += 1
        last_served[nxt[0]] = served
        for other in (A, B, C):
            if drr.queued(other):
                assert served - last_served[other] <= 7
    assert served == len(costs)


async def test_dispatcher_grants_in_order_and_releases() -> None:
    dispatcher = FairDispatcher(capacity=1, quantum=10)
    with pytest.raises(ValueError):
        FairDispatcher(capacity=0, quantum=1)
    order: list[str] = []
    await dispatcher.acquire(A, 5)

    async def worker(tenant: TenantId, name: str) -> None:
        await dispatcher.acquire(tenant, 5)
        order.append(name)
        dispatcher.release()

    tasks = [
        asyncio.create_task(worker(A, "a1")),
        asyncio.create_task(worker(A, "a2")),
        asyncio.create_task(worker(B, "b1")),
    ]
    await asyncio.sleep(0)
    assert dispatcher.waiting == 3
    dispatcher.release()
    await asyncio.gather(*tasks)
    assert sorted(order) == ["a1", "a2", "b1"]
    assert dispatcher.waiting == 0


async def test_dispatcher_skips_cancelled_waiters() -> None:
    dispatcher = FairDispatcher(capacity=1, quantum=10)
    await dispatcher.acquire(A, 1)
    waiter = asyncio.create_task(dispatcher.acquire(B, 1))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    dispatcher.release()
    await asyncio.wait_for(dispatcher.acquire(C, 1), timeout=1)


async def test_dispatcher_cancel_after_grant_releases_slot() -> None:
    dispatcher = FairDispatcher(capacity=1, quantum=10)
    await dispatcher.acquire(A, 1)
    waiter = asyncio.create_task(dispatcher.acquire(B, 1))
    await asyncio.sleep(0)
    dispatcher.release()  # grants the waiter's future
    waiter.cancel()  # cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(dispatcher.acquire(C, 1), timeout=1)


async def test_dispatcher_resize_follows_workers() -> None:
    dispatcher = FairDispatcher(capacity=1, quantum=10)
    await dispatcher.acquire(A, 1)
    waiter = asyncio.create_task(dispatcher.acquire(B, 1))
    await asyncio.sleep(0)
    assert dispatcher.waiting == 1
    dispatcher.resize(2)  # a second worker became ready
    await asyncio.wait_for(waiter, timeout=1)
    assert dispatcher.capacity == 2
    dispatcher.resize(1)  # shrinks as slots are released
    dispatcher.release()
    dispatcher.release()
    await asyncio.wait_for(dispatcher.acquire(C, 1), timeout=1)
    with pytest.raises(ValueError):
        dispatcher.resize(0)
