"""Permission- and version-filtered retrieval (FR-2).

The access service supplies the documents a user may read; that set and
the validity time go inside the index search, together with the question
text for indexes that also search by keyword. If the access service fails,
AccessDenied propagates: nothing is searched.

With a reranker, the search returns `candidates` chunks and a cross-encoder
picks the top_k among them (it only reorders permitted, valid chunks). If
the reranker fails, the search order is kept and a warning is logged.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass

from askiisc.access.client import AccessAPI
from askiisc.common.clock import Clock
from askiisc.common.targets import (
    PROMPT_TOP_K,
    RERANK_CANDIDATES,
    RERANK_PASSAGE_CHARS,
)
from askiisc.observability.stages import StageTimer
from askiisc.retrieval.embedder import Embedder
from askiisc.retrieval.rerank import Reranker, RerankError
from askiisc.retrieval.store import ChunkIndex, ScoredChunk

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Retrieved chunks and the context they were retrieved under."""

    hits: tuple[ScoredChunk, ...]
    policy_version: int
    at: float


class Retriever:
    """Embeds a question and runs the filtered search.

    `query_prefix` goes in front of the question before embedding only
    (BGE models want an instruction on queries, not on passages); the
    keyword search gets the question as asked. `executor` runs the index
    searches on their own threads, so searches waiting for the index never
    hold the threads that logins and chat storage need (None: the default
    pool). `monotonic` times the stages (time.perf_counter unless a test
    injects a fake); each stage goes to `observe` and, when the caller
    passes one, to the request's StageTimer.
    """

    def __init__(
        self,
        access: AccessAPI,
        embedder: Embedder,
        index: ChunkIndex,
        clock: Clock,
        top_k: int = PROMPT_TOP_K,
        observe: Callable[[str, float], None] | None = None,
        query_prefix: str = "",
        executor: Executor | None = None,
        reranker: Reranker | None = None,
        candidates: int = RERANK_CANDIDATES,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._access = access
        self._embedder = embedder
        self._index = index
        self._clock = clock
        self._top_k = top_k
        self._observe = observe or (lambda stage, seconds: None)
        self._query_prefix = query_prefix
        self._executor = executor
        self._reranker = reranker
        self._candidates = max(candidates, top_k)
        self._monotonic = monotonic

    def _now(self, timer: StageTimer | None) -> float:
        # A request's stages all use its timer's clock.
        return self._monotonic() if timer is None else timer.now()

    def _done(
        self, stage: str, started: float, timer: StageTimer | None
    ) -> None:
        seconds = self._now(timer) - started
        self._observe(stage, seconds)
        if timer is not None:
            timer.add(stage, seconds)

    async def search(
        self,
        token: str,
        query: str,
        at: float | None = None,
        timer: StageTimer | None = None,
    ) -> SearchResult:
        """Finds the best chunks the user may read.

        Args:
            token: The user's verified SSO token.
            query: The question.
            at: As-of time; None means now.
            timer: The request's stage timer, if it keeps one.

        Returns:
            Up to top_k hits, best first.

        Raises:
            AccessDenied: If the access service cannot answer.
        """
        started = self._now(timer)
        try:
            allowed = await self._access.filter(token, at)
        finally:
            self._done("access", started, timer)
        when = self._clock.now() if at is None else at
        if not allowed.doc_ids:
            return SearchResult((), allowed.policy_version, when)
        started = self._now(timer)
        vector = (await self._embedder.embed([self._query_prefix + query]))[0]
        self._done("embed", started, timer)
        started = self._now(timer)
        search = functools.partial(
            self._index.search,
            vector,
            allowed.doc_ids,
            when,
            self._candidates if self._reranker else self._top_k,
            query,
        )
        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(self._executor, search)
        self._done("search", started, timer)
        hits = await self._rerank(query, hits, timer)
        return SearchResult(tuple(hits), allowed.policy_version, when)

    async def _rerank(
        self, query: str, hits: list[ScoredChunk], timer: StageTimer | None
    ) -> list[ScoredChunk]:
        """The top_k hits by cross-encoder score (search order on failure)."""
        if self._reranker is None or len(hits) < 2:  # noqa: PLR2004
            return hits[: self._top_k]
        started = self._now(timer)
        texts = [h.chunk.text[:RERANK_PASSAGE_CHARS] for h in hits]
        try:
            scores = await self._reranker.scores(query, texts)
        except RerankError as err:
            LOG.warning("reranking failed, search order kept: %s", err)
            if timer is not None:
                timer.add("rerank", timer.now() - started)
            return hits[: self._top_k]
        self._done("rerank", started, timer)
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        return [hits[i] for i in order[: self._top_k]]
