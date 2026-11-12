"""FR-5: the index never keeps a version the version store has ended."""

from pathlib import Path

from askiisc.common.types import Chunk, ChunkDraft, ChunkId, DocId, Version
from askiisc.ingest.pipeline import reconcile_retired
from askiisc.ingest.versions import InMemoryVersionStore, SQLiteVersionStore
from askiisc.retrieval.dense import DenseIndex
from askiisc.retrieval.store import FAR_FUTURE

DOC = DocId("loadtest-fresh-12")
DRAFTS = [ChunkDraft(text="code ALPHA", page=1, start=0, end=10)]


def _chunk(version: int, start: float) -> Chunk:
    return Chunk(
        id=ChunkId(f"{DOC}@v{version}#0"),
        doc_id=DOC,
        version=Version(version),
        text=f"code v{version}",
        page=1,
        valid_from=start,
        valid_to=None,
        embedding=(1.0, 0.0, 0.0, 0.0),
    )


def test_a_version_the_store_ended_is_retired_in_the_index() -> None:
    # Regression (job 2163, found by run 2234): the job was killed between
    # recording v2 and retiring v1's chunks, so v1 stayed searchable.
    versions = InMemoryVersionStore()
    versions.publish(DOC, DRAFTS, 10.0, "f1")
    versions.publish(DOC, DRAFTS, 20.0, "f2")  # v1 ends at 20
    index = DenseIndex(dim=4)
    index.upsert([_chunk(1, 10.0), _chunk(2, 20.0)])  # v1's retire was lost
    assert index.valid_to_of(DOC, Version(1)) == FAR_FUTURE
    hits = index.search([1.0, 0, 0, 0], frozenset({DOC}), 30.0, 5)
    assert {h.chunk.version for h in hits} == {1, 2}

    assert reconcile_retired(versions, index) == 1
    assert index.valid_to_of(DOC, Version(1)) == 20.0
    hits = index.search([1.0, 0, 0, 0], frozenset({DOC}), 30.0, 5)
    assert [h.chunk.version for h in hits] == [2]
    # As-of 15 still finds v1 (versions are retired, never deleted).
    past = index.search([1.0, 0, 0, 0], frozenset({DOC}), 15.0, 5)
    assert [h.chunk.version for h in past] == [1]
    assert reconcile_retired(versions, index) == 0  # nothing left to fix


def test_versions_unknown_to_the_index_are_left_alone() -> None:
    versions = InMemoryVersionStore()
    versions.publish(DOC, DRAFTS, 10.0, "f1")
    versions.publish(DOC, DRAFTS, 20.0, "f2")
    index = DenseIndex(dim=4)
    assert index.valid_to_of(DOC, Version(1)) is None
    assert reconcile_retired(versions, index) == 0


def test_sqlite_store_lists_ended_versions(tmp_path: Path) -> None:
    versions = SQLiteVersionStore(tmp_path / "versions.db")
    versions.publish(DOC, DRAFTS, 10.0, "f1")
    versions.publish(DOC, DRAFTS, 20.0, "f2")
    versions.publish(DocId("other"), DRAFTS, 10.0, "f3")
    assert versions.retired() == [(DOC, Version(1), 20.0)]
