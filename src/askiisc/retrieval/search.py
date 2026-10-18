"""Permission- and version-filtered retrieval (FR-2).

The access service supplies the documents a user may read; that set and
the validity time go inside the index search. If the access service fails,
AccessDenied propagates: nothing is searched.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from askiisc.access.client import AccessAPI
from askiisc.common.clock import Clock
from askiisc.common.targets import RETRIEVAL_TOP_K
from askiisc.retrieval.embedder import Embedder
from askiisc.retrieval.store import ChunkIndex, ScoredChunk


@dataclass(frozen=True)
class SearchResult:
    """Retrieved chunks and the context they were retrieved under."""

    hits: tuple[ScoredChunk, ...]
    policy_version: int
    at: float


class Retriever:
    """Embeds a question and runs the filtered search."""

    def __init__(
        self,
        access: AccessAPI,
        embedder: Embedder,
        index: ChunkIndex,
        clock: Clock,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        self._access = access
        self._embedder = embedder
        self._index = index
        self._clock = clock
        self._top_k = top_k

    async def search(
        self, token: str, query: str, at: float | None = None
    ) -> SearchResult:
        """Finds the best chunks the user may read.

        Args:
            token: The user's verified SSO token.
            query: The question.
            at: As-of time; None means now.

        Returns:
            Up to top_k hits, best first.

        Raises:
            AccessDenied: If the access service cannot answer.
        """
        allowed = await self._access.filter(token, at)
        when = self._clock.now() if at is None else at
        if not allowed.doc_ids:
            return SearchResult((), allowed.policy_version, when)
        vector = (await self._embedder.embed([query]))[0]
        hits = await asyncio.to_thread(
            self._index.search, vector, allowed.doc_ids, when, self._top_k
        )
        return SearchResult(tuple(hits), allowed.policy_version, when)
