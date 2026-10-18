import asyncio
import importlib

import pytest

from askiisc.common import targets
from askiisc.common.clock import FakeClock, SystemClock

PACKAGES = [
    "askiisc",
    "askiisc.common",
    "askiisc.ingest",
    "askiisc.access",
    "askiisc.retrieval",
    "askiisc.llm",
    "askiisc.gateway",
    "askiisc.autoscaler",
    "askiisc.observability",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name).__doc__


def test_fake_clock_advances_and_sleeps_instantly() -> None:
    clock = FakeClock(start=10.0)
    clock.advance(5.0)
    asyncio.run(clock.sleep(2.5))
    assert clock.now() == 17.5


def test_fake_clock_refuses_to_go_back() -> None:
    with pytest.raises(ValueError, match="backwards"):
        FakeClock().advance(-1.0)


def test_system_clock_moves_forward() -> None:
    clock = SystemClock()
    first = clock.now()
    asyncio.run(clock.sleep(0))
    assert clock.now() >= first


def test_targets_match_prd() -> None:
    assert targets.MAX_GPU_WORKERS == 2
    assert targets.RETRIEVAL_TOP_K == 8
    assert targets.FR11_MAX_HEAVY_SHARE + targets.FR11_MIN_LIGHT_SHARE < 1
