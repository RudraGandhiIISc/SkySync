"""Injected time source, so no code under test reads the wall clock."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    """A source of time in seconds since an arbitrary epoch."""

    def now(self) -> float:
        """Returns the current time in seconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Waits for the given number of seconds."""
        ...


class SystemClock:
    """The real wall clock (Unix epoch seconds)."""

    def now(self) -> float:
        """Returns the current Unix time in seconds."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Sleeps for the given number of seconds."""
        await asyncio.sleep(seconds)


class FakeClock:
    """A manually advanced clock for deterministic tests.

    `sleep` advances the clock instead of waiting, so code that sleeps runs
    instantly under test.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        """Returns the fake current time."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Moves the clock forward.

        Args:
            seconds: How far to move; must not be negative.

        Raises:
            ValueError: If `seconds` is negative.
        """
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")
        self._now += seconds

    async def sleep(self, seconds: float) -> None:
        """Advances the clock by `seconds` and yields to the event loop."""
        self.advance(max(seconds, 0.0))
        await asyncio.sleep(0)
