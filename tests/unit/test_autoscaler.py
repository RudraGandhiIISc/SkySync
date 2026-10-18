"""FR-9 and FR-10: scaling policy and controller timelines."""

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from askiisc.autoscaler.controller import Controller, DemandTracker
from askiisc.autoscaler.policy import (
    Hold,
    Observation,
    ScaleIn,
    ScaleOut,
    ScalingConfig,
    decide,
)
from askiisc.autoscaler.pool import FakeWorkerPool, PoolError
from askiisc.common.targets import (
    FR9_SCALE_OUT_SECONDS,
    FR10_RELEASE_AFTER_IDLE_SECONDS,
    FR10_SCRIPTED_COLD_STARTS,
    MAX_GPU_WORKERS,
)
from tests.sim import SimClock

CONFIG = ScalingConfig(
    rate_per_worker=4.0,
    queue_per_worker=16,
    idle_timeout_s=60.0,
    hysteresis_s=30.0,
)


def obs(**kw: object) -> Observation:
    base: dict[str, object] = dict(
        now=1000.0,
        request_rate=0.0,
        queue_depth=0,
        in_flight=0,
        workers=1,
        last_traffic=990.0,
    )
    base.update(kw)
    return Observation(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (obs(workers=0, request_rate=0.1), ScaleOut(1)),
        (obs(workers=0, queue_depth=1, last_action=999.0), ScaleOut(1)),
        (obs(workers=0, last_traffic=None), Hold()),
        # Regression: old arrivals still in the rate window after a
        # scale-to-zero must not restart a worker (flapping).
        (
            obs(
                workers=0,
                request_rate=1.0,
                last_traffic=900.0,
                last_action=950.0,
            ),
            Hold(),
        ),
        (
            obs(
                workers=0,
                request_rate=1.0,
                last_traffic=960.0,
                last_action=950.0,
            ),
            ScaleOut(1),
        ),
        (obs(request_rate=6.0), ScaleOut(1)),
        (obs(request_rate=60.0, workers=1), ScaleOut(1)),
        (obs(queue_depth=40), ScaleOut(1)),
        (obs(request_rate=6.0, last_action=980.0), Hold()),
        (obs(request_rate=6.0, workers=2), Hold()),
        (obs(last_traffic=900.0), ScaleIn(1)),
        (obs(last_traffic=None, workers=2), ScaleIn(2)),
        (obs(last_traffic=900.0, in_flight=1), Hold()),
        (obs(last_traffic=900.0, queue_depth=1), Hold()),
        (obs(workers=2, request_rate=1.0, low_since=900.0), ScaleIn(1)),
        (obs(workers=2, request_rate=1.0, low_since=990.0), Hold()),
        (obs(request_rate=1.0), Hold()),
    ],
)
def test_policy_table(observation: Observation, expected: object) -> None:
    assert decide(observation, CONFIG) == expected


def test_config_rejects_more_than_two_workers() -> None:
    with pytest.raises(ValueError):
        ScalingConfig(max_workers=3)
    with pytest.raises(ValueError):
        ScalingConfig(max_workers=0)


observations = st.builds(
    Observation,
    now=st.floats(0, 1e5),
    request_rate=st.floats(0, 100),
    queue_depth=st.integers(0, 500),
    in_flight=st.integers(0, 100),
    workers=st.integers(0, MAX_GPU_WORKERS),
    last_traffic=st.none() | st.floats(0, 1e5),
    last_action=st.none() | st.floats(0, 1e5),
    low_since=st.none() | st.floats(0, 1e5),
)


@given(o=observations)
def test_policy_invariants(o: Observation) -> None:
    d = decide(o, CONFIG)
    if isinstance(d, ScaleOut):
        assert d.count >= 1
        assert o.workers + d.count <= MAX_GPU_WORKERS
    if isinstance(d, ScaleIn):
        assert o.in_flight == 0 and o.queue_depth == 0
        assert 1 <= d.count <= o.workers
    in_window = (
        o.last_action is not None
        and o.now - o.last_action < CONFIG.hysteresis_s
    )
    if in_window and o.workers > 0:
        assert d == Hold()


def test_demand_tracker_window() -> None:
    clock = SimClock()
    tracker = DemandTracker(clock, window_s=10.0)  # type: ignore[arg-type]
    for _ in range(20):
        tracker.note()
    assert tracker.rate() == 2.0
    assert tracker.last_arrival == 0.0


class Sim:
    """A controller, a fake pool and a load generator on one SimClock."""

    def __init__(self, acquire_s: float = 5.0, load_s: float = 30.0) -> None:
        self.clock = SimClock()
        self.pool = FakeWorkerPool(self.clock, acquire_s, load_s)  # type: ignore[arg-type]
        self.in_flight = 0
        self.max_seen = 0
        self.controller = Controller(
            self.pool,
            CONFIG,
            self.clock,  # type: ignore[arg-type]
            in_flight=lambda: self.in_flight,
        )
        self.done = False

    async def control(self) -> None:
        await self.controller.run(interval_s=1.0, until=lambda: self.done)

    async def traffic(self, rate: float, start: float, end: float) -> None:
        await self.clock.sleep(start - self.clock.now())
        while self.clock.now() < end:
            self.controller.note_request()
            self.max_seen = max(self.max_seen, len(self.pool.list_workers()))
            await self.clock.sleep(1.0 / rate)

    def ready_count(self) -> int:
        return len(self.controller.ready_workers())


async def _start(sim: Sim, *coros: object) -> list[asyncio.Task[None]]:
    return [asyncio.create_task(c) for c in (sim.control(), *coros)]  # type: ignore[misc]


async def _stop(sim: Sim, tasks: list[asyncio.Task[None]]) -> None:
    sim.done = True
    await sim.clock.run_until(sim.clock.now() + 2.0)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_burst_reaches_two_workers_within_180_s() -> None:
    sim = Sim()
    tasks = await _start(sim, sim.traffic(rate=6.0, start=0.0, end=400.0))
    reached_at = None
    for t in range(1, 400):
        await sim.clock.run_until(float(t))
        sim.max_seen = max(sim.max_seen, len(sim.pool.list_workers()))
        if reached_at is None and sim.ready_count() == 2:
            reached_at = sim.clock.now()
    await _stop(sim, tasks)
    assert reached_at is not None and reached_at <= FR9_SCALE_OUT_SECONDS
    assert sim.max_seen <= MAX_GPU_WORKERS
    assert len(sim.pool.history) == 2


async def test_scale_to_zero_after_idle_timeout() -> None:
    sim = Sim()
    tasks = await _start(sim, sim.traffic(rate=1.0, start=0.0, end=100.0))
    await sim.clock.run_until(100.0)
    assert sim.ready_count() == 1
    last = sim.controller.demand.last_arrival or 0.0
    await sim.clock.run_until(
        last + CONFIG.idle_timeout_s + FR10_RELEASE_AFTER_IDLE_SECONDS
    )
    assert sim.pool.list_workers() == []
    assert all(w.released_at is not None for w in sim.pool.history)
    await _stop(sim, tasks)


async def test_twenty_scripted_cold_starts_all_complete() -> None:
    sim = Sim(acquire_s=5.0, load_s=30.0)
    tasks = await _start(sim)
    holds: list[float] = []

    async def one_request() -> None:
        sim.controller.note_request()
        arrived = sim.clock.now()
        assert not sim.controller.ready()
        ok = await sim.controller.wait_ready(timeout_s=150.0)
        assert ok
        holds.append(sim.clock.now() - arrived)

    for i in range(FR10_SCRIPTED_COLD_STARTS):
        request = asyncio.create_task(one_request())
        await sim.clock.run_until(sim.clock.now() + 60.0)
        assert request.done() and request.exception() is None, i
        await sim.clock.run_until(sim.clock.now() + 150.0)  # back to zero
        assert sim.pool.list_workers() == []
    await _stop(sim, tasks)
    assert len(holds) == FR10_SCRIPTED_COLD_STARTS
    assert max(holds) <= 5.0 + 30.0 + 2.0


async def test_worker_failure_is_replaced_within_180_s() -> None:
    sim = Sim()
    tasks = await _start(sim, sim.traffic(rate=2.0, start=0.0, end=600.0))
    await sim.clock.run_until(100.0)
    victim = sim.controller.ready_workers()[0]
    sim.pool.kill(victim.worker_id)
    killed_at = sim.clock.now()
    restored = None
    for t in range(101, 400):
        await sim.clock.run_until(float(t))
        if restored is None and sim.ready_count() >= 1:
            restored = sim.clock.now()
    await _stop(sim, tasks)
    assert restored is not None
    assert restored - killed_at <= FR9_SCALE_OUT_SECONDS


async def test_failed_start_is_counted_and_retried() -> None:
    sim = Sim()
    sim.pool.fail_next = 1
    tasks = await _start(sim, sim.traffic(rate=1.0, start=0.0, end=200.0))
    await sim.clock.run_until(150.0)
    assert sim.controller.failures == 1
    assert sim.ready_count() == 1
    assert sim.controller.endpoint().startswith("http://fake-")
    await _stop(sim, tasks)


async def test_endpoint_without_workers_and_close() -> None:
    sim = Sim()
    with pytest.raises(PoolError):
        sim.controller.endpoint()
    assert not await sim.controller.wait_ready(0.0)
    tasks = await _start(sim, sim.traffic(rate=1.0, start=0.0, end=50.0))
    await sim.clock.run_until(3.0)
    closer = asyncio.create_task(sim.controller.close())
    await sim.clock.run_until(4.0)
    await closer
    assert sim.pool.list_workers() == []
    await _stop(sim, tasks)


async def test_ready_change_callback_resizes() -> None:
    clock = SimClock()
    pool = FakeWorkerPool(clock, 1.0, 1.0)  # type: ignore[arg-type]
    seen: list[int] = []
    controller = Controller(pool, CONFIG, clock, on_ready_change=seen.append)  # type: ignore[arg-type]
    controller.note_request()
    await controller.step()
    await clock.run_until(5.0)
    assert seen == [1]


async def test_fake_pool_refuses_third_worker() -> None:
    clock = SimClock()
    pool = FakeWorkerPool(clock, 0.0, 0.0)  # type: ignore[arg-type]
    tasks = [asyncio.create_task(pool.request_worker()) for _ in range(2)]
    await clock.run_until(1.0)
    await asyncio.gather(*tasks)
    with pytest.raises(PoolError):
        await pool.request_worker()
    await pool.release_worker("missing")
