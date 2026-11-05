"""Per-request stage timer for the audit record's latency breakdown (FR-12).

One timer belongs to one request; it is passed down the request path, so
nothing is shared between requests. Stage names are fixed identifiers
(never user text). Repeated stages add up: a question about two years runs
the search twice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager


class StageTimer:
    """Sums the seconds spent in each named stage of one request."""

    def __init__(self, monotonic: Callable[[], float]) -> None:
        """Creates an empty timer.

        Args:
            monotonic: Seconds from a monotonic clock (time.perf_counter in
                production, a fake in tests).
        """
        self._monotonic = monotonic
        self._totals: dict[str, float] = {}

    def now(self) -> float:
        """The injected clock's current reading."""
        return self._monotonic()

    def add(self, stage: str, seconds: float) -> None:
        """Adds time to a stage.

        Args:
            stage: The stage name.
            seconds: Time spent.

        Raises:
            ValueError: If seconds is negative.
        """
        if seconds < 0:
            raise ValueError(f"negative duration for stage {stage!r}")
        self._totals[stage] = self._totals.get(stage, 0.0) + seconds

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Times the enclosed block, also when it raises.

        Args:
            stage: The stage name.

        Yields:
            Nothing.
        """
        started = self._monotonic()
        try:
            yield
        finally:
            self.add(stage, self._monotonic() - started)

    def totals(self) -> dict[str, float]:
        """Seconds per stage (a copy)."""
        return dict(self._totals)
