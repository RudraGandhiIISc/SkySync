"""FR-6: admission follows the GPU pool, and can be switched off (E4)."""

import asyncio

import pytest

from askiisc.autoscaler.pool import FakeWorkerPool, WorkerState
from askiisc.common.clock import FakeClock
from askiisc.common.config import Settings
from askiisc.common.errors import RateLimited
from askiisc.common.types import TenantId
from askiisc.gateway.limits import Admission

A, B = TenantId("a"), TenantId("b")
SLOTS = 4
SERVICE_S = 8.0
MAX_WAIT_S = 3.0


def ready(pool: FakeWorkerPool) -> int:
    return sum(w.state is WorkerState.READY for w in pool.list_workers())


def make(clock: FakeClock, pool: FakeWorkerPool, **kw: object) -> Admission:
    args: dict[str, object] = {
        "rate_per_s": 1000,
        "burst": 1000,
        "queue_max": 1000,
        "max_wait_s": MAX_WAIT_S,
        "concurrency": SLOTS,
        "service_estimate_s": SERVICE_S,
        "ready_workers": lambda: ready(pool),
    }
    args.update(kw)
    return Admission(clock, **args)  # type: ignore[arg-type]


async def start(pool: FakeWorkerPool, n: int) -> None:
    await asyncio.gather(*(pool.request_worker() for _ in range(n)))


def admit_until_refused(adm: Admission, limit: int = 100) -> int:
    for admitted in range(limit):
        try:
            adm.admit(A)
        except RateLimited:
            return admitted
    return limit


async def test_an_empty_pool_never_refuses_on_wait() -> None:
    clock = FakeClock()
    pool = FakeWorkerPool(clock)
    adm = make(clock, pool)
    assert adm.slots() == 0
    assert admit_until_refused(adm) == 100  # all go to the cold hold
    assert adm.estimated_wait_s() == 0.0


@pytest.mark.parametrize("workers", [1, 2])
async def test_slots_follow_ready_workers(workers: int) -> None:
    clock = FakeClock()
    pool = FakeWorkerPool(clock, acquire_s=0, load_s=0)
    await start(pool, workers)
    adm = make(clock, pool)
    assert adm.slots() == SLOTS * workers
    # Admitted while the wait stays within MAX_WAIT_S: one queued request
    # waits SERVICE_S / slots, so this many fit beyond the slots.
    extra = int(MAX_WAIT_S * SLOTS * workers / SERVICE_S)
    assert admit_until_refused(adm) == SLOTS * workers + extra


async def test_a_worker_becoming_ready_mid_run_raises_capacity() -> None:
    clock = FakeClock()
    pool = FakeWorkerPool(clock, acquire_s=5, load_s=30)
    adm = make(clock, pool)
    for _ in range(10):
        adm.admit(A)  # pool at zero: held, not refused
    await pool.request_worker()  # the fake clock runs the 35 s start
    assert adm.slots() == SLOTS
    # 10 in system on 4 slots: 7 would wait 7 * 8 / 4 = 14 s > 3 s.
    with pytest.raises(RateLimited, match="estimated wait"):
        adm.admit(A)
    await pool.request_worker()
    assert adm.slots() == 2 * SLOTS
    # 10 on 8 slots: 3 waiting * 8 / 8 = 3 s, within the limit.
    adm.admit(A)


async def test_a_worker_lost_mid_run_lowers_capacity() -> None:
    clock = FakeClock()
    pool = FakeWorkerPool(clock, acquire_s=0, load_s=0)
    await start(pool, 2)
    adm = make(clock, pool)
    for _ in range(9):
        adm.admit(A)
    pool.kill(pool.list_workers()[0].worker_id)
    assert adm.slots() == SLOTS
    with pytest.raises(RateLimited, match="estimated wait"):
        adm.admit(A)


async def test_without_a_pool_the_slots_are_fixed() -> None:
    adm = Admission(FakeClock(), 1000, 1000, 1000, 1000, concurrency=7)
    assert adm.slots() == 7  # noqa: PLR2004


async def test_switched_off_admits_everything() -> None:
    clock = FakeClock()
    pool = FakeWorkerPool(clock, acquire_s=0, load_s=0)
    await start(pool, 1)
    adm = make(
        clock,
        pool,
        rate_per_s=0.001,
        burst=1,
        queue_max=1,
        max_in_flight=1,
        enabled=False,
    )
    tickets = [adm.admit(t) for t in (A, B) * 50]
    assert adm.depth(A) == adm.depth(B) == 50  # noqa: PLR2004
    for ticket in tickets:
        adm.release(ticket)
    assert adm.depth(A) == 0


def test_the_switch_reads_on_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().admission is True
    monkeypatch.setenv("ASKIISC_ADMISSION", "off")
    assert Settings().admission is False
    monkeypatch.setenv("ASKIISC_ADMISSION", "on")
    assert Settings().admission is True
