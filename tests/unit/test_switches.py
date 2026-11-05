"""E4 and FR-7 switches: fair queueing and the answer cache."""

import asyncio

import pytest

from askiisc.common.config import Settings
from askiisc.common.types import TenantId
from askiisc.gateway.drr import FairDispatcher

A, B = TenantId("a"), TenantId("b")


async def grant_order(fair: bool) -> list[str]:
    dispatcher = FairDispatcher(capacity=1, quantum=1.0, fair=fair)
    await dispatcher.acquire(A, 1.0)  # holds the only slot
    order: list[str] = []

    async def wait(tenant: TenantId, name: str) -> None:
        await dispatcher.acquire(tenant, 1.0)
        order.append(name)

    names = [(A, "a1"), (A, "a2"), (A, "a3"), (B, "b1")]
    tasks = [asyncio.create_task(wait(t, n)) for t, n in names]
    await asyncio.sleep(0)
    for _ in names:
        dispatcher.release()
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    return order


async def test_fair_queueing_interleaves_and_fifo_does_not() -> None:
    fair = await grant_order(fair=True)
    assert fair.index("b1") < fair.index("a3")  # b is not stuck behind a
    assert await grant_order(fair=False) == ["a1", "a2", "a3", "b1"]


def test_switch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    on = Settings()
    assert on.fair_queue and on.answer_cache
    monkeypatch.setenv("ASKIISC_FAIR_QUEUE", "off")
    monkeypatch.setenv("ASKIISC_ANSWER_CACHE", "off")
    off = Settings()
    assert not off.fair_queue and not off.answer_cache
