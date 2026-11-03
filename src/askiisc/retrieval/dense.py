"""An in-memory exact index over every chunk, persisted in Milvus.

Milvus Lite only does brute-force search, one query at a time, and the
permission filter is a 25k-id expression it parses on every query (about
840 ms on the real corpus). This index keeps the unit vectors in one numpy
matrix and the permission and validity filters as boolean masks, so one
query is one matrix-vector product (tens of milliseconds) and the recall is
exact. The filter is still applied inside the search: a forbidden or
expired chunk can never reach the top k (CLAUDE.md safety rule 2).

With the question's text, search is hybrid (NFR-7): the best vector
candidates and the best BM25 keyword candidates (retrieval/keyword.py),
both under the same masks, are fused by reciprocal rank.

Writes go to the persistent index first (Milvus), then to memory. A
snapshot (`state/dense/`) makes restarts fast; it is rebuilt from Milvus
when its stamp no longer matches the version store. A snapshot without the
keyword index gets one built at load.

Build a snapshot (inside Slurm):
    python -m askiisc.retrieval.dense build STATE_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from askiisc.common.targets import (
    EMBEDDING_DIM,
    HYBRID_CANDIDATES,
    KEYWORD_WEIGHT,
    RRF_K,
)
from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.keyword import KeywordIndex
from askiisc.retrieval.store import FAR_FUTURE, ChunkIndex, ScoredChunk

LOG = logging.getLogger(__name__)
SNAPSHOT_FORMAT = 1
_MASK_CACHE = 64

Floats = npt.NDArray[np.float32]


class DenseIndex:
    """Exact filtered cosine search over chunks held in memory.

    Args:
        persistent: Where writes go first (a MilvusIndex), or None.
        dim: Vector dimension.
    """

    def __init__(
        self, persistent: ChunkIndex | None = None, dim: int = EMBEDDING_DIM
    ) -> None:
        self._persistent = persistent
        self._dim = dim
        self._lock = threading.Lock()
        self._n = 0
        self._vectors: Floats = np.zeros((0, dim), np.float32)
        self._doc: npt.NDArray[np.int32] = np.zeros(0, np.int32)
        self._version: npt.NDArray[np.int32] = np.zeros(0, np.int32)
        self._page: npt.NDArray[np.int32] = np.zeros(0, np.int32)
        self._from: npt.NDArray[np.float64] = np.zeros(0, np.float64)
        self._to: npt.NDArray[np.float64] = np.zeros(0, np.float64)
        self._ids: list[ChunkId] = []
        self._texts: list[str] = []
        self._row: dict[ChunkId, int] = {}
        self._docs: list[DocId] = []
        self._doc_pos: dict[DocId, int] = {}
        self._doc_rows: dict[int, list[int]] = {}
        self._masks: OrderedDict[int, tuple[frozenset[DocId], Any]] = (
            OrderedDict()
        )
        self._keywords = KeywordIndex()
        self.dirty = False

    def __len__(self) -> int:
        return self._n

    @property
    def documents(self) -> int:
        """The number of distinct documents."""
        return len(self._docs)

    @property
    def dim(self) -> int:
        """The vector dimension of the index."""
        return self._dim

    # -- writes ----------------------------------------------------------

    def _grow(self, need: int) -> None:
        capacity = len(self._vectors)
        if need <= capacity:
            return
        new = max(need, int(capacity * 1.25) + 1024)

        def grown(a: Any, shape: tuple[int, ...]) -> Any:
            out = np.zeros(shape, a.dtype)
            out[: self._n] = a[: self._n]
            return out

        self._vectors = grown(self._vectors, (new, self._dim))
        self._doc = grown(self._doc, (new,))
        self._version = grown(self._version, (new,))
        self._page = grown(self._page, (new,))
        self._from = grown(self._from, (new,))
        self._to = grown(self._to, (new,))

    def _doc_index(self, doc_id: DocId) -> int:
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            pos = len(self._docs)
            self._docs.append(doc_id)
            self._doc_pos[doc_id] = pos
            self._masks.clear()  # cached masks are one document short
        return pos

    def _put(
        self,
        chunk_id: ChunkId,
        doc_id: DocId,
        version: int,
        page: int,
        text: str,
        valid_from: float,
        valid_to: float,
        vector: Sequence[float] | Floats,
        keywords: bool = True,
    ) -> None:
        row = self._row.get(chunk_id)
        if row is None:
            row = self._n
            self._grow(row + 1)
            self._n += 1
            self._row[chunk_id] = row
            self._ids.append(chunk_id)
            self._texts.append(text)
            if keywords:
                self._keywords.put(row, text)
        else:
            if keywords:
                self._keywords.put(row, text, old=self._texts[row])
            self._texts[row] = text
            self._doc_rows[int(self._doc[row])].remove(row)
        doc = self._doc_index(doc_id)
        self._doc_rows.setdefault(doc, []).append(row)
        v = np.asarray(vector, np.float32)
        norm = float(np.linalg.norm(v))
        self._vectors[row] = v / norm if norm else v
        self._doc[row] = doc
        self._version[row] = version
        self._page[row] = page
        self._from[row] = valid_from
        self._to[row] = valid_to

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Inserts or replaces chunks (persistent index first).

        Raises:
            ValueError: If a chunk has no embedding.
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id} has no embedding")
        if self._persistent is not None:
            self._persistent.upsert(chunks)
        with self._lock:
            for c in chunks:
                self._put(
                    c.id,
                    c.doc_id,
                    int(c.version),
                    c.page,
                    c.text,
                    c.valid_from,
                    FAR_FUTURE if c.valid_to is None else c.valid_to,
                    c.embedding or (),
                )
            self.dirty = True

    def retire(self, doc_id: DocId, version: Version, at: float) -> int:
        """Sets valid_to on every chunk of a version (persistent first)."""
        if self._persistent is not None:
            self._persistent.retire(doc_id, version, at)
        with self._lock:
            doc = self._doc_pos.get(doc_id)
            rows = [] if doc is None else self._doc_rows.get(doc, [])
            hits = [r for r in rows if int(self._version[r]) == int(version)]
            for r in hits:
                self._to[r] = at
            self.dirty = self.dirty or bool(hits)
            return len(hits)

    # -- search ----------------------------------------------------------

    def _doc_mask(self, allowed: frozenset[DocId]) -> Any:
        """Boolean mask over documents, cached per filter object.

        The access client hands out the same frozenset for a user for a few
        seconds, so most queries reuse a mask. The cache holds the set
        itself, so an id is never reused for a different set.
        """
        key = id(allowed)
        hit = self._masks.get(key)
        if hit is not None and hit[0] is allowed:
            self._masks.move_to_end(key)
            return hit[1]
        mask = np.zeros(len(self._docs), bool)
        positions = [self._doc_pos[d] for d in allowed if d in self._doc_pos]
        mask[np.asarray(positions, np.int64)] = True
        self._masks[key] = (allowed, mask)
        if len(self._masks) > _MASK_CACHE:
            self._masks.popitem(last=False)
        return mask

    def search(
        self,
        vector: Sequence[float],
        allowed_docs: frozenset[DocId],
        at: float,
        top_k: int,
        text: str = "",
    ) -> list[ScoredChunk]:
        """Returns the top_k allowed chunks valid at `at`, best first.

        Args:
            vector: The query vector.
            allowed_docs: Documents the user may read.
            at: Validity time.
            top_k: How many chunks.
            text: The query text; empty means vector search only.

        Returns:
            Chunks with their cosine scores, best (fused) rank first.
        """
        if not allowed_docs or top_k <= 0:
            return []
        q = np.asarray(vector, np.float32)
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        with self._lock:
            n = self._n
            if n == 0:
                return []
            rows = self._doc_mask(allowed_docs)[self._doc[:n]]
            rows &= self._from[:n] <= at
            rows &= self._to[:n] > at
            candidates = int(rows.sum())
            if candidates == 0:
                return []
            scores = self._vectors[:n] @ q
            scores[~rows] = -np.inf
            if not text:
                order = _best(scores, min(top_k, candidates))
            else:
                order = self._hybrid(scores, rows, text, top_k, candidates)
            return [self._hit(int(r), float(scores[r])) for r in order]

    def _hybrid(
        self, scores: Any, rows: Any, text: str, top_k: int, candidates: int
    ) -> list[int]:
        """Fuses vector and keyword candidates (caller holds the lock).

        Each list is as deep as the number of chunks asked for (at least
        HYBRID_CANDIDATES), so a reranker asking for many candidates gets
        deep candidates from both lists, not the top few of each.
        """
        depth = min(max(HYBRID_CANDIDATES, top_k), candidates)
        keyword = self._keywords.scores(text, len(scores))
        keyword[~rows] = 0.0
        lists = [(_best(scores, depth), 1.0)]
        hits = int(np.count_nonzero(keyword))
        if hits:
            lists.append((_best(keyword, min(depth, hits)), KEYWORD_WEIGHT))
        fused: dict[int, float] = {}
        for ranked, weight in lists:
            for rank, row in enumerate(ranked):
                fused[row] = fused.get(row, 0.0) + weight / (RRF_K + rank + 1)
        return sorted(fused, key=lambda r: (-fused[r], r))[:top_k]

    def _hit(self, row: int, score: float) -> ScoredChunk:
        valid_to = float(self._to[row])
        chunk = Chunk(
            id=self._ids[row],
            doc_id=self._docs[int(self._doc[row])],
            version=Version(int(self._version[row])),
            text=self._texts[row],
            page=int(self._page[row]),
            valid_from=float(self._from[row]),
            valid_to=None if valid_to >= FAR_FUTURE else valid_to,
        )
        return ScoredChunk(chunk, score)

    def text(self, chunk_id: str) -> str | None:
        """A chunk's text, or None if unknown."""
        with self._lock:
            row = self._row.get(ChunkId(chunk_id))
            return None if row is None else self._texts[row]

    # -- persistence -----------------------------------------------------

    def load_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        """Adds rows as Milvus returns them (no persistent write).

        The keyword index is not updated row by row; call build_keywords
        once all rows are in.
        """
        with self._lock:
            for r in rows:
                self._put(
                    ChunkId(r["chunk_id"]),
                    DocId(r["doc_id"]),
                    int(r["version"]),
                    int(r["page"]),
                    r["text"],
                    float(r["valid_from"]),
                    float(r["valid_to"]),
                    r["vector"],
                    keywords=False,
                )

    def build_keywords(self) -> None:
        """Rebuilds the keyword index from every row's text in one pass."""
        with self._lock:
            keywords = KeywordIndex()
            keywords.build(self._texts)
            self._keywords = keywords

    def save(self, folder: Path, stamp: str) -> None:
        """Writes a snapshot that `load` can read back."""
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "tmp"
        tmp.mkdir(exist_ok=True)
        with self._lock:
            n = self._n
            np.save(tmp / "vectors.npy", self._vectors[:n])
            np.savez(
                tmp / "meta.npz",
                doc=self._doc[:n],
                version=self._version[:n],
                page=self._page[:n],
                valid_from=self._from[:n],
                valid_to=self._to[:n],
            )
            with (tmp / "rows.jsonl").open("w") as f:
                for cid, text in zip(self._ids, self._texts, strict=True):
                    f.write(json.dumps([cid, text]) + "\n")
            (tmp / "docs.json").write_text(json.dumps(self._docs))
            self._keywords.save(tmp)
            (tmp / "stamp.json").write_text(
                json.dumps({"format": SNAPSHOT_FORMAT, "stamp": stamp})
            )
            self.dirty = False
        for path in tmp.iterdir():
            path.replace(folder / path.name)
        tmp.rmdir()

    @staticmethod
    def stamp_of(folder: Path) -> str | None:
        """The stamp a snapshot was saved with, or None."""
        try:
            meta = json.loads((folder / "stamp.json").read_text())
        except (OSError, ValueError):
            return None
        if meta.get("format") != SNAPSHOT_FORMAT:
            return None
        return str(meta.get("stamp"))

    def load(self, folder: Path) -> None:
        """Replaces the contents with a snapshot.

        A snapshot from before the keyword index gets one built from its
        texts and saved next to it.
        """
        vectors = np.load(folder / "vectors.npy")
        meta = np.load(folder / "meta.npz")
        docs = [
            DocId(d) for d in json.loads((folder / "docs.json").read_text())
        ]
        ids: list[ChunkId] = []
        texts: list[str] = []
        with (folder / "rows.jsonl").open() as f:
            for line in f:
                cid, text = json.loads(line)
                ids.append(ChunkId(cid))
                texts.append(text)
        doc_rows: dict[int, list[int]] = {}
        for row, doc in enumerate(meta["doc"].tolist()):
            doc_rows.setdefault(doc, []).append(row)
        keywords = KeywordIndex()
        if KeywordIndex.exists(folder):
            keywords.load(folder)
        else:
            started = time.monotonic()
            keywords.build(texts)
            keywords.save(folder)
            LOG.info(
                "keyword index built in %.0f s", time.monotonic() - started
            )
        with self._lock:
            self._n = len(ids)
            self._vectors = vectors.astype(np.float32, copy=False)
            self._dim = int(vectors.shape[1])
            self._doc = meta["doc"].astype(np.int32)
            self._version = meta["version"].astype(np.int32)
            self._page = meta["page"].astype(np.int32)
            self._from = meta["valid_from"].astype(np.float64)
            self._to = meta["valid_to"].astype(np.float64)
            self._ids, self._texts = ids, texts
            self._row = {c: i for i, c in enumerate(ids)}
            self._docs = docs
            self._doc_pos = {d: i for i, d in enumerate(docs)}
            self._doc_rows = doc_rows
            self._keywords = keywords
            self._masks.clear()
            self.dirty = False


def _best(scores: Any, k: int) -> list[int]:
    """Rows of the k highest scores, best first (ties by row)."""
    top = np.argpartition(-scores, k - 1)[:k]
    return [int(r) for r in top[np.lexsort((top, -scores[top]))]]


def load_from_milvus(
    index: DenseIndex, client: Any, collection: str = "chunks"
) -> int:
    """Copies every row of a Milvus collection into the index.

    Returns:
        The number of rows copied.
    """
    fields = [
        "chunk_id",
        "doc_id",
        "version",
        "page",
        "text",
        "valid_from",
        "valid_to",
        "vector",
    ]
    iterator = client.query_iterator(
        collection, batch_size=4096, filter="", output_fields=fields
    )
    copied = 0
    try:
        while batch := iterator.next():
            index.load_rows(batch)
            copied += len(batch)
            if copied % 65536 < len(batch):
                LOG.info("copied %d rows from milvus", copied)
    finally:
        iterator.close()
    return copied


def open_dense(
    state: Path,
    persistent: ChunkIndex | None,
    client: Any,
    stamp: str,
    dim: int = EMBEDDING_DIM,
) -> DenseIndex:
    """Loads the snapshot in `state/dense`, or rebuilds it from Milvus."""
    folder = state / "dense"
    index = DenseIndex(persistent, dim)
    started = time.monotonic()
    if DenseIndex.stamp_of(folder) == stamp:
        index.load(folder)
        LOG.info(
            "dense snapshot: %d chunks in %.0f s",
            len(index),
            time.monotonic() - started,
        )
        return index
    load_from_milvus(index, client)
    index.build_keywords()
    LOG.info(
        "dense index from milvus: %d chunks in %.0f s",
        len(index),
        time.monotonic() - started,
    )
    index.save(folder, stamp)
    return index


def main() -> None:  # pragma: no cover - needs the real state folder
    """Builds the snapshot of a state folder and times some searches."""
    from pymilvus import MilvusClient  # noqa: PLC0415

    from askiisc.ingest.versions import SQLiteVersionStore  # noqa: PLC0415
    from askiisc.retrieval.store import MilvusIndex  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = MilvusClient(str(args.state / "milvus.db"))
    versions = SQLiteVersionStore(args.state / "versions.db")
    index = open_dense(
        args.state, MilvusIndex(client), client, versions.stamp()
    )
    rng = np.random.default_rng(0)
    everything = frozenset(index._docs)  # noqa: SLF001 - benchmark only
    times = []
    for _ in range(50):
        q = rng.standard_normal(index._dim).astype(np.float32)  # noqa: SLF001
        t0 = time.perf_counter()
        index.search(q.tolist(), everything, time.time(), 8)
        times.append(time.perf_counter() - t0)
    times.sort()
    print(
        json.dumps(
            {
                "chunks": len(index),
                "documents": index.documents,
                "search_p50_ms": round(times[25] * 1e3, 1),
                "search_p95_ms": round(times[47] * 1e3, 1),
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
