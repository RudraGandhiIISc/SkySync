"""The FR-2 recall check measures the serving index honestly."""

import random
from pathlib import Path
from typing import Any

import pytest
from eval.probes.recall import read_snapshot, readable_docs, run_probes

from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.dense import DenseIndex
from askiisc.retrieval.store import ScoredChunk

DIM = 8
PEOPLE = [
    {"user_id": "u1", "department": "cds", "role": "student", "groups": []},
    {"user_id": "u2", "department": "ece", "role": "faculty", "groups": []},
]
POLICIES: list[dict[str, Any]] = [
    {"doc_id": f"doc{d}", "visibility": "public"} for d in range(6)
] + [
    {"doc_id": "doc6", "visibility": "restricted", "departments": ["cds"]},
    {"doc_id": "doc7", "visibility": "restricted", "roles": ["faculty"]},
]


def _index(tmp_path: Path) -> tuple[DenseIndex, Path]:
    rng = random.Random(3)
    chunks = [
        Chunk(
            id=ChunkId(f"doc{d}@v1#{i}"),
            doc_id=DocId(f"doc{d}"),
            version=Version(1),
            text="t",
            page=1,
            valid_from=0.0,
            valid_to=50.0 if (d, i) == (0, 0) else None,
            embedding=tuple(rng.uniform(-1, 1) for _ in range(DIM)),
        )
        for d in range(8)
        for i in range(5)
    ]
    index = DenseIndex(dim=DIM)
    index.upsert(chunks)
    index.save(tmp_path / "dense", "s")
    return index, tmp_path / "dense"


def _inputs() -> tuple[dict[str, frozenset[DocId]], dict[str, set[str]]]:
    readable = {
        u["user_id"]: readable_docs(PEOPLE, POLICIES, u["user_id"], 100.0)
        for u in PEOPLE
    }
    filters = {u: frozenset(DocId(d) for d in r) for u, r in readable.items()}
    return filters, readable


def test_oracle_readable_sets() -> None:
    _, readable = _inputs()
    assert "doc6" in readable["u1"] and "doc7" not in readable["u1"]
    assert "doc7" in readable["u2"] and "doc6" not in readable["u2"]


def test_the_exact_index_has_full_recall_and_no_leaks(tmp_path: Path) -> None:
    index, folder = _index(tmp_path)
    filters, readable = _inputs()
    out = run_probes(
        index, read_snapshot(folder), filters, readable, 200, 8,
        random.Random(1), now=lambda: 100.0,
    )  # fmt: skip
    assert out["recall_at_k"] == pytest.approx(1.0)
    assert out["leaks"] == 0
    assert out["probes"] == 200 and out["users"] == 2


class _Broken:
    """Returns forbidden, expired and second-rate chunks."""

    def __init__(self, inner: DenseIndex) -> None:
        self._inner = inner

    def search(
        self, vector: Any, allowed: frozenset[DocId], at: float, k: int
    ) -> list[ScoredChunk]:
        everything = frozenset(DocId(f"doc{d}") for d in range(8))
        hits = self._inner.search(vector, everything, 10.0, 40)
        return hits[-k:]


def test_a_wrong_index_shows_low_recall_and_leaks(tmp_path: Path) -> None:
    index, folder = _index(tmp_path)
    filters, readable = _inputs()
    out = run_probes(
        _Broken(index), read_snapshot(folder), filters, readable, 50, 8,  # type: ignore[arg-type]
        random.Random(2), now=lambda: 100.0,
    )  # fmt: skip
    assert out["recall_at_k"] < 0.5
    assert out["leaks"] > 0


def test_a_user_who_may_read_nothing(tmp_path: Path) -> None:
    index, folder = _index(tmp_path)
    out = run_probes(
        index, read_snapshot(folder), {"x": frozenset()}, {"x": set()}, 5, 8,
        random.Random(3), now=lambda: 100.0,
    )  # fmt: skip
    assert out["recall_at_k"] == 1.0 and out["leaks"] == 0
