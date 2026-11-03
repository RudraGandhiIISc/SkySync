"""The in-memory exact index agrees with the reference index (FR-2)."""

import random
from pathlib import Path
from typing import Any

import pytest

from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.dense import DenseIndex, load_from_milvus, open_dense
from askiisc.retrieval.store import FAR_FUTURE, InMemoryIndex

DIM = 8


def _chunks(n_docs: int = 30, per_doc: int = 4, seed: int = 1) -> list[Chunk]:
    rng = random.Random(seed)
    out = []
    for d in range(n_docs):
        for i in range(per_doc):
            start = float(rng.choice([0, 0, 50]))
            end = rng.choice([None, None, 80.0, 200.0])
            out.append(
                Chunk(
                    id=ChunkId(f"doc{d}@v1#{i}"),
                    doc_id=DocId(f"doc{d}"),
                    version=Version(1),
                    text=f"text {d} {i}",
                    page=i + 1,
                    valid_from=start,
                    valid_to=end,
                    embedding=tuple(rng.uniform(-1, 1) for _ in range(DIM)),
                )
            )
    return out


@pytest.mark.parametrize("at", [10.0, 60.0, 100.0, 300.0])
def test_dense_matches_the_reference(at: float) -> None:
    chunks = _chunks()
    dense, ref = DenseIndex(dim=DIM), InMemoryIndex()
    dense.upsert(chunks)
    ref.upsert(chunks)
    rng = random.Random(7)
    for _ in range(25):
        allowed = frozenset(DocId(f"doc{d}") for d in rng.sample(range(30), 12))
        q = [rng.uniform(-1, 1) for _ in range(DIM)]
        got = dense.search(q, allowed, at, 8)
        want = ref.search(q, allowed, at, 8)
        assert [h.chunk.id for h in got] == [h.chunk.id for h in want]
        for g, w in zip(got, want, strict=True):
            assert g.score == pytest.approx(w.score, abs=1e-5)
            assert g.chunk.doc_id in allowed
            assert g.chunk.valid_to == w.chunk.valid_to
    assert dense.search(q, frozenset(), at, 8) == []
    assert dense.search(q, frozenset({DocId("nope")}), at, 8) == []
    assert dense.search(q, allowed, at, 0) == []
    assert DenseIndex(dim=DIM).search(q, allowed, at, 8) == []


def test_upsert_replace_retire_and_persistent_first() -> None:
    calls: list[str] = []

    class Persistent:
        def upsert(self, chunks: Any) -> None:
            calls.append("upsert")

        def retire(self, doc_id: Any, version: Any, at: float) -> int:
            calls.append("retire")
            return 0

        def search(self, *a: Any) -> list[Any]:
            return []

    dense = DenseIndex(Persistent(), dim=DIM)
    chunks = _chunks(2, 2)
    dense.upsert(chunks)
    assert dense.dirty and len(dense) == 4 and dense.documents == 2
    moved = chunks[0].model_copy(
        update={"doc_id": DocId("doc1"), "text": "moved"}
    )
    dense.upsert([moved])
    assert dense.text(chunks[0].id) == "moved" and dense.text("x") is None
    assert dense.retire(DocId("doc1"), Version(1), 25.0) == 3
    assert dense.retire(DocId("doc1"), Version(9), 25.0) == 0
    assert dense.retire(DocId("ghost"), Version(1), 25.0) == 0
    everything = frozenset({DocId("doc0"), DocId("doc1")})
    hits = dense.search([1.0] * DIM, everything, 30.0, 8)
    assert {h.chunk.id for h in hits} <= {chunks[1].id}
    assert calls == ["upsert", "upsert", "retire", "retire", "retire"]
    with pytest.raises(ValueError, match="no embedding"):
        dense.upsert([chunks[0].model_copy(update={"embedding": None})])


def test_masks_are_cached_per_filter_object() -> None:
    dense = DenseIndex(dim=DIM)
    dense.upsert(_chunks(70, 1))
    allowed = frozenset({DocId("doc1")})
    q = [0.5] * DIM
    dense.search(q, allowed, 10.0, 3)
    dense.search(q, allowed, 10.0, 3)
    assert len(dense._masks) == 1  # noqa: SLF001
    for d in range(70):
        dense.search(q, frozenset({DocId(f"doc{d}")}), 10.0, 1)
    assert len(dense._masks) == 64  # noqa: SLF001
    dense.upsert(_chunks(71, 1)[70:])  # a new document clears them
    assert len(dense._masks) == 0  # noqa: SLF001


def test_snapshot_round_trip_and_milvus_rebuild(tmp_path: Path) -> None:
    chunks = _chunks(5, 3)
    rows = [
        {
            "chunk_id": c.id,
            "doc_id": c.doc_id,
            "version": int(c.version),
            "page": c.page,
            "text": c.text,
            "valid_from": c.valid_from,
            "valid_to": FAR_FUTURE if c.valid_to is None else c.valid_to,
            "vector": list(c.embedding or ()),
        }
        for c in chunks
    ]

    class Iterator:
        def __init__(self) -> None:
            self.batches = [rows[:10], rows[10:], []]
            self.closed = False

        def next(self) -> list[dict[str, Any]]:
            return self.batches.pop(0)

        def close(self) -> None:
            self.closed = True

    class Client:
        def query_iterator(self, *a: Any, **kw: Any) -> Iterator:
            return Iterator()

    built = open_dense(tmp_path, None, Client(), "stamp-1", DIM)
    assert (
        len(built) == 15
        and DenseIndex.stamp_of(tmp_path / "dense") == "stamp-1"
    )
    loaded = open_dense(tmp_path, None, None, "stamp-1", DIM)
    q = [0.3] * DIM
    everything = frozenset(c.doc_id for c in chunks)
    assert [h.chunk.id for h in loaded.search(q, everything, 60.0, 5)] == [
        h.chunk.id for h in built.search(q, everything, 60.0, 5)
    ]
    loaded.upsert(_chunks(6, 1)[5:])
    assert loaded.documents == 6
    assert DenseIndex.stamp_of(tmp_path / "missing") is None
    (tmp_path / "dense" / "stamp.json").write_text('{"format": 0}')
    assert DenseIndex.stamp_of(tmp_path / "dense") is None
    assert load_from_milvus(DenseIndex(dim=DIM), Client()) == 15


def _texted(i: int, doc: str, text: str, vec: list[float], **kw: Any) -> Chunk:
    return Chunk(
        id=ChunkId(f"{doc}@v1#{i}"),
        doc_id=DocId(doc),
        version=Version(1),
        text=text,
        page=1,
        valid_from=kw.get("valid_from", 0.0),
        valid_to=kw.get("valid_to"),
        embedding=tuple(vec),
    )


def _hybrid_index() -> DenseIndex:
    e = [[1.0 if j == k else 0.0 for j in range(DIM)] for k in range(DIM)]
    index = DenseIndex(dim=DIM)
    index.upsert(
        [
            _texted(0, "near", "General notice about campus life.", e[0]),
            _texted(1, "near", "Another general campus notice.", e[0]),
            _texted(0, "form", "Submit form PG-12 to change your guide.", e[3]),
            _texted(0, "secret", "Form PG-12 secret committee copy.", e[0]),
            _texted(0, "old", "Form PG-12 old rules.", e[0], valid_to=5.0),
            _texted(0, "dup1", "Mess rebate 2024: apply by Friday.", e[1]),
            _texted(0, "dup2", "Mess rebate 2025: apply by Friday.", e[1]),
        ]
    )
    return index


def test_hybrid_finds_keyword_matches_inside_the_filter() -> None:
    index = _hybrid_index()
    q = [1.0] + [0.0] * (DIM - 1)
    allowed = frozenset(DocId(d) for d in ("near", "form", "old", "dup1"))
    vector_only = [h.chunk.doc_id for h in index.search(q, allowed, 10.0, 2)]
    assert "form" not in vector_only
    hybrid = index.search(q, allowed, 10.0, 2, text="form PG-12")
    assert hybrid[0].chunk.doc_id == "form"
    everything = index.search(q, allowed, 10.0, 8, text="form PG-12 rebate")
    docs = {h.chunk.doc_id for h in everything}
    assert "secret" not in docs and "old" not in docs  # masks hold
    assert hybrid[0].score == pytest.approx(0.0)  # the cosine is reported


def test_hybrid_results_survive_a_snapshot(tmp_path: Path) -> None:
    index = _hybrid_index()
    q = [0.0, 1.0] + [0.0] * (DIM - 2)
    allowed = frozenset(DocId(d) for d in ("dup1", "dup2", "near"))
    hits = index.search(q, allowed, 10.0, 8, text="mess rebate 2025")
    assert hits[0].chunk.doc_id == "dup2"  # the keyword breaks the tie
    index.save(tmp_path / "s", "x")
    loaded = DenseIndex(dim=DIM)
    loaded.load(tmp_path / "s")
    again = loaded.search(q, allowed, 10.0, 8, text="mess rebate 2025")
    assert [h.chunk.id for h in again] == [h.chunk.id for h in hits]


def test_old_snapshots_get_a_keyword_index(tmp_path: Path) -> None:
    index = _hybrid_index()
    folder = tmp_path / "s"
    index.save(folder, "x")
    (folder / "keyword.npz").unlink()
    (folder / "keyword_vocab.txt").unlink()
    loaded = DenseIndex(dim=DIM)
    loaded.load(folder)
    assert (folder / "keyword.npz").exists()
    q = [1.0] + [0.0] * (DIM - 1)
    hits = loaded.search(q, frozenset({DocId("form")}), 10.0, 1, text="PG-12")
    assert hits[0].chunk.doc_id == "form"


def test_bulk_rows_get_their_keywords_in_one_pass() -> None:
    def row(cid: str, text: str, axis: int) -> dict[str, Any]:
        vec = [0.0] * DIM
        vec[axis] = 1.0
        return {
            "chunk_id": cid, "doc_id": cid.split("@", maxsplit=1)[0],
            "version": 1, "page": 1, "text": text, "valid_from": 0.0,
            "valid_to": FAR_FUTURE, "vector": vec,
        }  # fmt: skip

    index = DenseIndex(dim=DIM)
    index.load_rows(
        [row("a@v1#0", "Hostel rules", 0), row("b@v1#0", "Form PG-12", 1)]
    )
    q = [1.0] + [0.0] * (DIM - 1)
    allowed = frozenset({DocId("a"), DocId("b")})
    assert index.search(q, allowed, 1.0, 1, text="PG-12")[0].chunk.doc_id == "a"
    index.build_keywords()
    assert index.search(q, allowed, 1.0, 1, text="PG-12")[0].chunk.doc_id == "b"


def test_hybrid_lists_are_as_deep_as_the_candidates_asked_for() -> None:
    # Regression: each list stopped at HYBRID_CANDIDATES (20), so a
    # reranker asking for 60 candidates got at most 40, all shallow.
    chunks = _chunks(n_docs=40, per_doc=2)
    index = DenseIndex(dim=DIM)
    index.upsert([c.model_copy(update={"valid_to": None}) for c in chunks])
    allowed = frozenset(c.doc_id for c in chunks)
    q = [1.0] * DIM
    hits = index.search(q, allowed, 60.0, 60, text="unrelated words")
    assert len(hits) == 60  # noqa: PLR2004
    mixed = index.search(q, allowed, 60.0, 60, text="text 3")
    assert len(mixed) == 60  # noqa: PLR2004
