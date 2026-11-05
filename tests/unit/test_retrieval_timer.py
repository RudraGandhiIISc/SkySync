"""Retriever stages go into the request's StageTimer (FR-12 breakdown)."""

from collections.abc import Sequence

import pytest

from askiisc.access.client import LocalAccess
from askiisc.access.policy import PolicyEngine
from askiisc.observability.stages import StageTimer
from askiisc.retrieval.embedder import FakeEmbedder, Vector
from askiisc.retrieval.rerank import FakeReranker, RerankError
from askiisc.retrieval.search import Retriever
from tests import corpus
from tests.access_fixtures import world


class Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SlowEmbedder(FakeEmbedder):
    def __init__(self, ticks: Ticks) -> None:
        super().__init__(dim=64)
        self._ticks = ticks

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self._ticks.now += 0.005
        return await super().embed(texts)


class SlowReranker(FakeReranker):
    def __init__(self, ticks: Ticks, fail: bool = False) -> None:
        super().__init__()
        self._ticks, self._fail = ticks, fail

    async def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        self._ticks.now += 0.009
        if self._fail:
            raise RerankError("down")
        return await super().scores(query, texts)


def _retriever(reranker: object) -> tuple[Retriever, str, Ticks]:
    w = world()
    ticks = Ticks()
    access = LocalAccess(
        PolicyEngine(w.directory, w.policies),
        w.issuer.verifier(w.clock),
        w.clock,
    )
    retriever = Retriever(
        access,
        SlowEmbedder(ticks),
        corpus.index(),
        w.clock,
        top_k=2,
        reranker=reranker(ticks),  # type: ignore[operator]
        candidates=10,
        monotonic=lambda: 1e9,  # unused: the request timer's clock wins
    )
    return retriever, w.token("u_ece_fac"), ticks


async def test_two_as_of_searches_add_up_in_one_timer() -> None:
    retriever, token, ticks = _retriever(SlowReranker)
    timer = StageTimer(ticks)
    await retriever.search(token, "hostel fees", None, timer)
    await retriever.search(token, "hostel fees", 1.0, timer)
    totals = timer.totals()
    assert totals["embed"] == pytest.approx(0.010)
    assert totals["rerank"] == pytest.approx(0.018)
    assert set(totals) == {"access", "embed", "search", "rerank"}


async def test_failed_rerank_is_still_timed() -> None:
    retriever, token, ticks = _retriever(lambda t: SlowReranker(t, fail=True))
    timer = StageTimer(ticks)
    result = await retriever.search(token, "hostel fees", None, timer)
    assert len(result.hits) == 2  # noqa: PLR2004
    assert timer.totals()["rerank"] == pytest.approx(0.009)


async def test_search_without_a_timer_still_reports_stages() -> None:
    seen: list[str] = []
    w = world()
    access = LocalAccess(
        PolicyEngine(w.directory, w.policies),
        w.issuer.verifier(w.clock),
        w.clock,
    )
    retriever = Retriever(
        access,
        corpus.EMBEDDER,
        corpus.index(),
        w.clock,
        observe=lambda stage, s: seen.append(stage),
        monotonic=lambda: 0.0,
    )
    await retriever.search(w.token("u_ece_fac"), "hostel fees")
    assert seen == ["access", "embed", "search"]
