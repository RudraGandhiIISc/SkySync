"""The chunk index: an in-memory exact index and a Milvus HNSW index.

Both apply the permission and version filters inside the search, never
after it (CLAUDE.md safety rule 2). This is the only module that talks to
Milvus.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from askiisc.common.targets import (
    EMBEDDING_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
)
from askiisc.common.types import Chunk, ChunkId, DocId, Version

FAR_FUTURE = 1.0e15
"""Stored as valid_to for current chunks (Milvus 2.4 has no null floats)."""

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/@#\-]{1,200}$")


@dataclass(frozen=True)
class ScoredChunk:
    """A search hit."""

    chunk: Chunk
    score: float


class ChunkIndex(Protocol):
    """Stores chunks with embeddings and searches them with filters."""

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Inserts or replaces chunks; each must carry an embedding."""
        ...

    def retire(self, doc_id: DocId, version: Version, at: float) -> int:
        """Sets valid_to on every chunk of a version; returns the count."""
        ...

    def search(
        self,
        vector: Sequence[float],
        allowed_docs: frozenset[DocId],
        at: float,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Returns the top_k chunks among allowed documents valid at `at`."""
        ...


def check_id(value: str) -> str:
    """Validates an id before it is placed in a filter expression.

    Args:
        value: A document or chunk id.

    Returns:
        The id unchanged.

    Raises:
        ValueError: If it contains characters outside the safe set.
    """
    if not _SAFE_ID.match(value):
        raise ValueError(f"unsafe id: {value!r}")
    return value


def _valid_at(chunk: Chunk, at: float) -> bool:
    end = FAR_FUTURE if chunk.valid_to is None else chunk.valid_to
    return chunk.valid_from <= at < end


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryIndex:
    """An exact, filtered brute-force index: the reference for recall."""

    def __init__(self) -> None:
        self._chunks: dict[ChunkId, Chunk] = {}
        self._lock = threading.Lock()

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Inserts or replaces chunks.

        Raises:
            ValueError: If a chunk has no embedding or an unsafe id.
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id} has no embedding")
            check_id(chunk.id)
            check_id(chunk.doc_id)
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

    def retire(self, doc_id: DocId, version: Version, at: float) -> int:
        """Sets valid_to on every chunk of a version."""
        with self._lock:
            hits = [
                c
                for c in self._chunks.values()
                if c.doc_id == doc_id and c.version == version
            ]
            for c in hits:
                self._chunks[c.id] = c.model_copy(update={"valid_to": at})
            return len(hits)

    def search(
        self,
        vector: Sequence[float],
        allowed_docs: frozenset[DocId],
        at: float,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Scans every chunk that passes the filter and ranks by cosine."""
        with self._lock:
            candidates = [
                c
                for c in self._chunks.values()
                if c.doc_id in allowed_docs and _valid_at(c, at)
            ]
        scored = [
            ScoredChunk(c, _cosine(vector, c.embedding or ()))
            for c in candidates
        ]
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        """Returns every stored chunk (for tests and audits)."""
        with self._lock:
            return list(self._chunks.values())


def milvus_filter(allowed_docs: Iterable[DocId], at: float) -> str:
    """Builds the Milvus boolean expression for permission and validity.

    Args:
        allowed_docs: Documents the user may read (must be non-empty).
        at: The time the chunks must be valid at.

    Returns:
        A Milvus filter expression.

    Raises:
        ValueError: If there are no documents or an id is unsafe.
    """
    ids = sorted(check_id(d) for d in allowed_docs)
    if not ids:
        raise ValueError("an empty permission set must not reach Milvus")
    quoted = ", ".join(f'"{d}"' for d in ids)
    return (
        f"doc_id in [{quoted}] and valid_from <= {at!r} and valid_to > {at!r}"
    )


class MilvusIndex:
    """A Milvus 2.4 collection with an HNSW index.

    Works with Milvus Lite (a local file URI, for tests) and with a Milvus
    standalone server URI. Milvus Lite indexes everything as FLAT.
    """

    _FIELDS = ("chunk_id", "doc_id", "version", "page", "text")
    _OUTPUT = (*_FIELDS, "valid_from", "valid_to", "acl_groups")

    def __init__(
        self,
        client: Any,
        collection: str = "chunks",
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self._client = client
        self._name = collection
        self._dim = dim
        if not client.has_collection(collection):
            self._create()
        client.load_collection(collection)

    def _create(self) -> None:
        from pymilvus import DataType  # noqa: PLC0415

        schema = self._client.create_schema(auto_id=False)
        schema.add_field(
            "chunk_id", DataType.VARCHAR, is_primary=True, max_length=256
        )
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("version", DataType.INT64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("valid_from", DataType.DOUBLE)
        schema.add_field("valid_to", DataType.DOUBLE)
        schema.add_field(
            "acl_groups",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=64,
            max_length=128,
        )
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        index = self._client.prepare_index_params()
        index.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        )
        self._client.create_collection(
            self._name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Inserts or replaces chunks.

        Raises:
            ValueError: If a chunk has no embedding or an unsafe id.
        """
        rows = [self._row(c) for c in chunks]
        if rows:
            self._client.upsert(self._name, rows)

    def _row(self, chunk: Chunk) -> dict[str, Any]:
        if chunk.embedding is None:
            raise ValueError(f"chunk {chunk.id} has no embedding")
        return {
            "chunk_id": check_id(chunk.id),
            "doc_id": check_id(chunk.doc_id),
            "version": int(chunk.version),
            "page": chunk.page,
            "text": chunk.text,
            "valid_from": chunk.valid_from,
            "valid_to": FAR_FUTURE
            if chunk.valid_to is None
            else chunk.valid_to,
            "acl_groups": sorted(chunk.acl),
            "vector": list(chunk.embedding),
        }

    def retire(self, doc_id: DocId, version: Version, at: float) -> int:
        """Sets valid_to on every chunk of a version (rows are kept)."""
        rows = self._client.query(
            self._name,
            filter=(
                f'doc_id == "{check_id(doc_id)}" and version == {int(version)}'
            ),
            output_fields=[*self._OUTPUT, "vector"],
            consistency_level="Strong",
        )
        for row in rows:
            row["valid_to"] = at
        if rows:
            self._client.upsert(self._name, [dict(r) for r in rows])
        return len(rows)

    def search(
        self,
        vector: Sequence[float],
        allowed_docs: frozenset[DocId],
        at: float,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Runs a filtered HNSW search; the filter is part of the query."""
        if not allowed_docs:
            return []
        hits = self._client.search(
            self._name,
            data=[list(vector)],
            filter=milvus_filter(allowed_docs, at),
            limit=top_k,
            output_fields=list(self._OUTPUT),
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": HNSW_EF_SEARCH},
            },
            consistency_level="Strong",
        )
        return [self._scored(hit) for hit in hits[0]]

    @staticmethod
    def _scored(hit: dict[str, Any]) -> ScoredChunk:
        entity = hit["entity"]
        valid_to = float(entity["valid_to"])
        chunk = Chunk(
            id=ChunkId(entity["chunk_id"]),
            doc_id=DocId(entity["doc_id"]),
            version=Version(int(entity["version"])),
            text=entity["text"],
            page=int(entity["page"]),
            valid_from=float(entity["valid_from"]),
            valid_to=None if valid_to >= FAR_FUTURE else valid_to,
            acl=frozenset(entity.get("acl_groups") or ()),
        )
        return ScoredChunk(chunk, float(hit["distance"]))
