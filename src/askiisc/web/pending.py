"""Registrations waiting for their e-mail confirmation, in memory.

A person who registers chooses their password at once; it takes effect
only when they open the single-use confirmation link e-mailed to them,
which proves they read that mailbox. Only the link's SHA-256 and the
password's Argon2id hash are kept, in memory: a restart of the serving job
asks people to register again. A new registration voids the address's
earlier one; at most MAX_PER_HOUR links an address.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass

from askiisc.common.clock import Clock

CONFIRM_TTL_S = 10 * 60.0
MAX_PER_HOUR = 3


@dataclass(frozen=True)
class Pending:
    """A registration waiting for its link."""

    user_id: str
    password_hash: str
    expires: float


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class PendingRegistrations:
    """Confirmation links of registrations; thread-safe."""

    def __init__(self, clock: Clock, ttl_s: float = CONFIRM_TTL_S) -> None:
        self._clock = clock
        self._ttl_s = ttl_s
        self._pending: dict[str, Pending] = {}
        self._sent: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(self, user_id: str, password_hash: str) -> str | None:
        """A new confirmation link token, or None if sent too often."""
        now = self._clock.now()
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._drop_expired(now)
            recent = [t for t in self._sent.get(user_id, []) if t > now - 3600]
            if len(recent) >= MAX_PER_HOUR:
                return None
            self._sent[user_id] = [*recent, now]
            for key in [
                k for k, p in self._pending.items() if p.user_id == user_id
            ]:
                del self._pending[key]
            self._pending[_hash(token)] = Pending(
                user_id, password_hash, now + self._ttl_s
            )
        return token

    def take(self, token: str) -> Pending | None:
        """The registration of an unexpired link, used up; or None."""
        now = self._clock.now()
        with self._lock:
            self._drop_expired(now)
            return self._pending.pop(_hash(token), None)

    def _drop_expired(self, now: float) -> None:
        for key in [k for k, p in self._pending.items() if p.expires <= now]:
            del self._pending[key]
        for user_id in [
            u for u, ts in self._sent.items() if ts[-1] <= now - 3600
        ]:
            del self._sent[user_id]
