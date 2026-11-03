"""FR-5: every version kept, as-of queries right at 60 checkpoints."""

import asyncio
from collections.abc import Sequence

import pytest

from askiisc.common.clock import FakeClock
from askiisc.common.errors import ExtractionError, NotFound
from askiisc.common.targets import FR5_REVISION_CHECKPOINTS
from askiisc.common.types import ChunkDraft, DocId
from askiisc.ingest.objects import InMemoryObjectStore
from askiisc.ingest.pipeline import Ingestor, extract_and_chunk
from askiisc.ingest.reconciler import (
    Reconciler,
    SourceDocument,
    content_fingerprint,
)
from askiisc.ingest.versions import InMemoryVersionStore
from askiisc.retrieval.embedder import FakeEmbedder
from askiisc.retrieval.store import InMemoryIndex


def _draft(text: str) -> ChunkDraft:
    return ChunkDraft(text=text, page=1, start=0, end=len(text))


def test_publish_retires_previous_at_same_instant() -> None:
    store = InMemoryVersionStore()
    doc = DocId("circular")
    v1 = store.publish(doc, [_draft("old")], at=10.0, fingerprint="a")
    v2 = store.publish(doc, [_draft("new")], at=20.0, fingerprint="b")
    assert v1.valid_to == v2.valid_from == 20.0
    assert store.current(doc).version == 2
    assert store.as_of(doc, 15.0) is v1
    assert store.as_of(doc, 20.0) is v2
    assert store.as_of(doc, 5.0) is None
    assert [r.version for r in store.history(doc)] == [1, 2]
    assert store.documents() == [doc]
    assert v1.chunks()[0].valid_to == 20.0


def test_publish_must_move_forward_and_unknown_doc() -> None:
    store = InMemoryVersionStore()
    store.publish(DocId("d"), [], at=10.0, fingerprint="a")
    with pytest.raises(ValueError):
        store.publish(DocId("d"), [], at=10.0, fingerprint="b")
    with pytest.raises(NotFound):
        store.current(DocId("missing"))


class FakeSource:
    """Documents whose content changes when the test says so."""

    def __init__(self, docs: int, use_etag: bool) -> None:
        self.revision = {DocId(f"doc{i}"): 1 for i in range(docs)}
        self.use_etag = use_etag
        self.fetches = 0
        self.broken: set[DocId] = set()

    def text(self, doc: DocId, revision: int) -> str:
        return f"Document {doc} revision {revision}. The rule says {revision}."

    async def list_documents(self) -> Sequence[SourceDocument]:
        return [
            SourceDocument(
                doc_id=d,
                content_type="text/plain",
                etag=f'"{r}"' if self.use_etag else None,
            )
            for d, r in self.revision.items()
        ]

    async def fetch(self, doc_id: DocId) -> bytes:
        self.fetches += 1
        if doc_id in self.broken:
            return b""
        return self.text(doc_id, self.revision[doc_id]).encode()


def _ingestor() -> tuple[Ingestor, InMemoryVersionStore, InMemoryIndex]:
    versions, index = InMemoryVersionStore(), InMemoryIndex()
    ingestor = Ingestor(
        versions,
        InMemoryObjectStore(),
        FakeEmbedder(dim=32),
        index,
        commit_delay_s=5.0,
    )
    return ingestor, versions, index


@pytest.mark.parametrize("use_etag", [True, False])
async def test_sixty_revisions_across_twenty_documents(use_etag: bool) -> None:
    clock = FakeClock(start=1_000.0)
    source = FakeSource(docs=20, use_etag=use_etag)
    ingestor, versions, index = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock, interval_s=300.0)
    await reconciler.run_once()
    checkpoints: list[tuple[float, DocId, int]] = []
    for step in range(FR5_REVISION_CHECKPOINTS):
        doc = DocId(f"doc{step % 20}")
        source.revision[doc] += 1
        clock.advance(300.0)
        report = await reconciler.run_once()
        assert report.changed == [doc]
        checkpoints.append((clock.now() + 10.0, doc, source.revision[doc]))
    for at, doc, revision in checkpoints:
        record = versions.as_of(doc, at)
        assert record is not None
        assert record.version == revision
        assert f"revision {record.version}." in record.drafts[0].text
    # Retired chunks never appear in current results.
    now = clock.now() + 10.0
    allowed = frozenset(versions.documents())
    vector = FakeEmbedder(dim=32).embed_one("rule")
    hits = index.search(vector, allowed, now, top_k=100)
    for hit in hits:
        assert hit.chunk.version == versions.current(hit.chunk.doc_id).version


async def test_atomic_switch_has_no_gap_or_overlap() -> None:
    clock = FakeClock(start=0.0)
    source = FakeSource(docs=1, use_etag=True)
    ingestor, _, index = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock)
    await reconciler.run_once()
    source.revision[DocId("doc0")] = 2
    clock.advance(100.0)
    await reconciler.run_once()
    allowed = frozenset({DocId("doc0")})
    vector = FakeEmbedder(dim=32).embed_one("document")
    switch = 105.0
    for at in (switch - 1e-6, switch, switch + 1e-6, 104.0, 106.0):
        versions_seen = {
            h.chunk.version for h in index.search(vector, allowed, at, 10)
        }
        assert len(versions_seen) == 1
        assert versions_seen == ({2} if at >= switch else {1})


async def test_unchanged_documents_are_skipped() -> None:
    clock = FakeClock()
    source = FakeSource(docs=3, use_etag=True)
    ingestor, _, _ = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock)
    await reconciler.run_once()
    fetches = source.fetches
    report = await reconciler.run_once()
    assert len(report.unchanged) == 3
    assert source.fetches == fetches  # ETag match: no fetch


async def test_content_hash_fallback_and_failures() -> None:
    clock = FakeClock()
    source = FakeSource(docs=2, use_etag=False)
    ingestor, _, _ = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock)
    first = await reconciler.run_once()
    assert len(first.changed) == 2
    source.broken.add(DocId("doc1"))
    source.revision[DocId("doc1")] += 1
    report = await reconciler.run_once()
    assert report.unchanged == [DocId("doc0")]
    assert report.failed == [DocId("doc1")]
    reconciler.seed({DocId("doc0"): "x"})
    assert content_fingerprint(b"a").startswith("sha256:")


async def test_run_forever_sleeps_between_passes() -> None:
    clock = FakeClock()
    source = FakeSource(docs=1, use_etag=True)
    ingestor, _, _ = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock, interval_s=300.0)
    await reconciler.run_forever(passes=3)
    assert clock.now() == 900.0


async def test_ingest_rejects_empty_document() -> None:
    ingestor, _, _ = _ingestor()
    doc = SourceDocument(doc_id=DocId("d"), content_type="text/plain")
    with pytest.raises(ExtractionError):
        await ingestor.ingest(doc, b"   ", "fp", 0.0)


def test_sqlite_store_keeps_versions_across_reopen(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from askiisc.ingest.versions import SQLiteVersionStore

    path = tmp_path / "state" / "versions.db"
    store = SQLiteVersionStore(path)
    doc = DocId("circular")
    store.publish(
        doc, [_draft("old")], at=10.0, fingerprint="a", acl=frozenset({"g"})
    )
    v2 = store.publish(doc, [_draft("new")], at=20.0, fingerprint="b")
    with pytest.raises(ValueError):
        store.publish(doc, [], at=20.0, fingerprint="c")
    store.close()
    reopened = SQLiteVersionStore(path)
    assert reopened.current(doc) == v2
    first = reopened.as_of(doc, 15.0)
    assert first is not None and first.valid_to == 20.0
    assert first.drafts[0].text == "old" and first.acl == {"g"}
    assert reopened.as_of(doc, 5.0) is None
    assert [r.version for r in reopened.history(doc)] == [1, 2]
    assert reopened.documents() == [doc]
    assert reopened.fingerprints() == {doc: "b"}
    with pytest.raises(NotFound):
        reopened.current(DocId("missing"))


async def test_seeded_reconciler_skips_ingested_documents(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from askiisc.ingest.versions import SQLiteVersionStore

    clock = FakeClock(start=100.0)
    source = FakeSource(docs=3, use_etag=False)
    store = SQLiteVersionStore(tmp_path / "v.db")
    ingestor = Ingestor(
        store,
        InMemoryObjectStore(),
        FakeEmbedder(dim=32),
        InMemoryIndex(),
        commit_delay_s=0.0,
    )
    first = Reconciler(source, ingestor.ingest, clock)
    assert len((await first.run_once()).changed) == 3
    second = Reconciler(source, ingestor.ingest, clock)  # a later job
    second.seed(store.fingerprints())
    source.revision[DocId("doc1")] += 1
    clock.advance(10.0)
    report = await second.run_once()
    assert report.changed == [DocId("doc1")]
    assert store.current(DocId("doc1")).version == 2


async def test_failed_embedding_leaves_document_unrecorded() -> None:
    """Crash safety: the version is recorded only after the chunks are in
    the index, so a killed or failed ingest is retried next time."""
    from askiisc.retrieval.embedder import EmbeddingError

    class Flaky(FakeEmbedder):
        fail = True

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            if self.fail:
                raise EmbeddingError("server died")
            return await super().embed(texts)

    clock = FakeClock(start=10.0)
    source = FakeSource(docs=1, use_etag=True)
    versions, index = InMemoryVersionStore(), InMemoryIndex()
    embedder = Flaky(dim=32)
    ingestor = Ingestor(
        versions, InMemoryObjectStore(), embedder, index, commit_delay_s=0.0
    )
    reconciler = Reconciler(source, ingestor.ingest, clock)
    assert (await reconciler.run_once()).failed == [DocId("doc0")]
    assert versions.history(DocId("doc0")) == []
    embedder.fail = False
    assert (await reconciler.run_once()).changed == [DocId("doc0")]
    assert versions.current(DocId("doc0")).version == 1
    assert len(index.all_chunks()) == 1


async def test_reconcile_pass_can_stop_between_documents() -> None:
    clock = FakeClock()
    source = FakeSource(docs=5, use_etag=True)
    ingestor, versions, _ = _ingestor()
    reconciler = Reconciler(source, ingestor.ingest, clock)
    calls = iter([False, False, True])
    report = await reconciler.run_once(should_stop=lambda: next(calls))
    assert report.stopped and len(report.changed) == 2
    rest = await reconciler.run_once()
    assert not rest.stopped and len(rest.changed) == 3
    assert len(versions.documents()) == 5


async def test_concurrent_pass_ingests_every_document_once() -> None:
    clock = FakeClock()
    source = FakeSource(docs=12, use_etag=True)
    ingestor, versions, _ = _ingestor()
    calls: list[str] = []

    async def chunker(
        data: bytes, content_type: str, template: frozenset[int]
    ) -> list[ChunkDraft]:
        calls.append(content_type)
        await asyncio.sleep(0)  # let other documents start
        return extract_and_chunk(data, content_type, template)

    ingestor.chunker = chunker
    reconciler = Reconciler(source, ingestor.ingest, clock, concurrency=4)
    report = await reconciler.run_once()
    assert sorted(report.changed) == sorted(versions.documents())
    assert len(report.changed) == 12 and len(calls) == 12  # noqa: PLR2004
    again = await reconciler.run_once()
    assert len(again.unchanged) == 12 and not again.changed  # noqa: PLR2004
