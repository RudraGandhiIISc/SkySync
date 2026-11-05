"""The controller's ready-worker count (recorded in audit records, FR-12)."""

import asyncio

from askiisc.autoscaler.controller import Controller
from askiisc.autoscaler.policy import ScalingConfig
from askiisc.autoscaler.pool import FakeWorkerPool
from tests.sim import SimClock

CONFIG = ScalingConfig(
    rate_per_worker=4.0,
    queue_per_worker=16,
    idle_timeout_s=60.0,
    hysteresis_s=30.0,
)


async def test_ready_count_follows_the_pool_from_zero_to_two() -> None:
    clock = SimClock()
    pool = FakeWorkerPool(clock, 1.0, 2.0)  # type: ignore[arg-type]
    controller = Controller(pool, CONFIG, clock)  # type: ignore[arg-type]
    assert controller.ready_count() == 0
    first = asyncio.create_task(pool.request_worker())
    await clock.run_until(1.5)  # acquired, still loading
    assert controller.ready_count() == 0
    await clock.run_until(3.0)
    await first
    assert controller.ready_count() == 1
    second = asyncio.create_task(pool.request_worker())
    await clock.run_until(6.0)
    await second
    assert controller.ready_count() == 2  # noqa: PLR2004
    await pool.release_worker(pool.list_workers()[0].worker_id)
    assert controller.ready_count() == 1
