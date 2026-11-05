"""Per-request stage timer (FR-12 latency breakdown)."""

import pytest

from askiisc.observability.stages import StageTimer


class Ticks:
    """A monotonic clock that advances only when told."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_measure_uses_only_the_injected_clock() -> None:
    ticks = Ticks()
    timer = StageTimer(ticks)
    with timer.measure("embed"):
        ticks.now += 0.25
    assert timer.totals() == {"embed": pytest.approx(0.25)}


def test_repeated_stages_add_up() -> None:
    ticks = Ticks()
    timer = StageTimer(ticks)
    for seconds in (0.1, 0.2):  # one search per as-of date
        with timer.measure("search"):
            ticks.now += seconds
    timer.add("search", 0.05)
    assert timer.totals()["search"] == pytest.approx(0.35)


def test_measure_records_even_when_the_stage_raises() -> None:
    ticks = Ticks()
    timer = StageTimer(ticks)
    with pytest.raises(RuntimeError), timer.measure("access"):
        ticks.now += 0.01
        raise RuntimeError("down")
    assert timer.totals()["access"] == pytest.approx(0.01)


def test_negative_durations_are_refused() -> None:
    timer = StageTimer(Ticks())
    with pytest.raises(ValueError):
        timer.add("auth", -0.001)


def test_totals_is_a_copy() -> None:
    timer = StageTimer(Ticks())
    timer.add("auth", 0.001)
    timer.totals()["auth"] = 99.0
    assert timer.totals()["auth"] == pytest.approx(0.001)
