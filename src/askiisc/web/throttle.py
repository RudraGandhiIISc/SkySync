"""Slows down password guessing: per address, and across all addresses.

Everyone reaches the app through the same SSH tunnel (one client address),
so limits are kept per typed e-mail address or id, whether or not an
account exists (the answer must not tell). After FREE_FAILURES failures
within WINDOW_S, each further attempt must wait, doubling from
FIRST_DELAY_S up to MAX_DELAY_S; a success clears the address. Across all
addresses, more than GLOBAL_FAILURES_PER_MIN failures in a minute make
every attempt wait (password spraying). State is in memory: a restart of
the serving job clears it.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from askiisc.common.clock import Clock

WINDOW_S = 15 * 60.0
FREE_FAILURES = 5
FIRST_DELAY_S = 30.0
MAX_DELAY_S = 15 * 60.0
GLOBAL_FAILURES_PER_MIN = 60
MAX_KEYS = 100_000


@dataclass
class _Key:
    failures: deque[float] = field(default_factory=deque)
    blocked_until: float = 0.0


class LoginThrottle:
    """Failure counts and waits; thread-safe."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._keys: dict[str, _Key] = {}
        self._recent: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_s(self, key: str) -> float:
        """Seconds to wait before this attempt may be checked (0: go)."""
        now = self._clock.now()
        with self._lock:
            while self._recent and self._recent[0] <= now - 60.0:
                self._recent.popleft()
            if len(self._recent) >= GLOBAL_FAILURES_PER_MIN:
                return max(1.0, self._recent[0] + 60.0 - now)
            entry = self._keys.get(key)
            if entry is None:
                return 0.0
            return max(0.0, entry.blocked_until - now)

    def failed(self, key: str) -> None:
        """Records a failure; sets the next wait once past the free ones."""
        now = self._clock.now()
        with self._lock:
            self._recent.append(now)
            if key not in self._keys and len(self._keys) >= MAX_KEYS:
                self._keys.clear()  # bounded memory under a flood
            entry = self._keys.setdefault(key, _Key())
            while entry.failures and entry.failures[0] <= now - WINDOW_S:
                entry.failures.popleft()
            entry.failures.append(now)
            extra = len(entry.failures) - FREE_FAILURES
            if extra > 0:
                delay = min(MAX_DELAY_S, FIRST_DELAY_S * 2 ** (extra - 1))
                entry.blocked_until = now + delay

    def succeeded(self, key: str) -> None:
        """Clears an address after a successful sign-in."""
        with self._lock:
            self._keys.pop(key, None)
