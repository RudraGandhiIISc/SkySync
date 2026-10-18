"""Answer cache keyed so that permissions and versions are part of the key.

key = H(permission-set fingerprint, policy version, corpus generation,
normalised question hash). A user with a different permission set, a policy
change or any new document version produces a different key, so stale or
foreign entries are never served.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from askiisc.common.clock import Clock
from askiisc.common.types import Answer

_SPACES = re.compile(r"\s+")


class CachedAnswer(BaseModel):
    """What the cache stores for one key."""

    model_config = ConfigDict(frozen=True)
    answer: Answer
    documents: tuple[tuple[str, int], ...]
    chunk_ids: tuple[str, ...]


def normalise(question: str) -> str:
    """Lower-cases and collapses whitespace."""
    return _SPACES.sub(" ", question.strip().lower())


def cache_key(
    allowed_docs: Iterable[str],
    policy_version: int,
    generation: int,
    question: str,
) -> str:
    """Builds the cache key (contains no raw question text).

    Args:
        allowed_docs: The user's readable documents.
        policy_version: Policy version the permission set came from.
        generation: Corpus generation (bumped on every new version).
        question: The question.

    Returns:
        A hex key.
    """
    fingerprint = hashlib.sha256(
        "\n".join(sorted(allowed_docs)).encode()
    ).hexdigest()
    question_hash = hashlib.sha256(normalise(question).encode()).hexdigest()
    raw = f"{fingerprint}|{policy_version}|{generation}|{question_hash}"
    return "askiisc:answer:" + hashlib.sha256(raw.encode()).hexdigest()


class AnswerCache(Protocol):
    """Stores answers by key."""

    async def get(self, key: str) -> CachedAnswer | None:
        """Returns a cached answer or None."""
        ...

    async def set(self, key: str, value: CachedAnswer) -> None:
        """Stores an answer."""
        ...


class InMemoryAnswerCache:
    """A TTL dictionary (tests and single-process runs)."""

    def __init__(self, clock: Clock, ttl_s: float = 600.0) -> None:
        self._clock = clock
        self._ttl = ttl_s
        self._items: dict[str, tuple[float, CachedAnswer]] = {}

    async def get(self, key: str) -> CachedAnswer | None:
        """Returns a live entry or None."""
        item = self._items.get(key)
        if item is None or self._clock.now() - item[0] >= self._ttl:
            self._items.pop(key, None)
            return None
        return item[1]

    async def set(self, key: str, value: CachedAnswer) -> None:
        """Stores an entry."""
        self._items[key] = (self._clock.now(), value)


class RedisAnswerCache:
    """Answers in Redis with a TTL. A Redis failure is a cache miss."""

    def __init__(self, client: Any, ttl_s: int = 600) -> None:
        self._client = client
        self._ttl = ttl_s

    async def get(self, key: str) -> CachedAnswer | None:
        """Returns a cached answer or None (also on Redis errors)."""
        from redis.exceptions import RedisError  # noqa: PLC0415

        try:
            raw = await self._client.get(key)
        except RedisError:
            return None
        return None if raw is None else CachedAnswer.model_validate_json(raw)

    async def set(self, key: str, value: CachedAnswer) -> None:
        """Stores an answer; Redis errors are ignored (cache is optional)."""
        from redis.exceptions import RedisError  # noqa: PLC0415

        try:
            await self._client.set(key, value.model_dump_json(), ex=self._ttl)
        except RedisError:
            return
