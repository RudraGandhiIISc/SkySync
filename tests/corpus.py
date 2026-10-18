"""A small synthetic three-tenant corpus over the access-matrix documents."""

from __future__ import annotations

from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.embedder import FakeEmbedder
from askiisc.retrieval.store import InMemoryIndex
from tests.access_fixtures import MATRIX

TOPICS = [
    "hostel fees",
    "exam attendance",
    "library hours",
    "thesis submission",
    "scholarship stipend",
    "course registration",
    "lab safety",
    "leave rules",
    "travel grant",
    "convocation dress",
]

EMBEDDER = FakeEmbedder(dim=64)


def doc_text(doc: str, topic: str, n: int) -> str:
    return (
        f"Document {doc} section {n} covers {topic}. The {topic} rule for "
        f"section {n} of {doc} is number {n * 7}."
    )


def chunks(per_doc: int = 3, valid_from: float = 0.0) -> list[Chunk]:
    out = []
    for d, doc in enumerate(MATRIX["documents"]):
        for n in range(per_doc):
            text = doc_text(doc, TOPICS[(d + n) % len(TOPICS)], n)
            out.append(
                Chunk(
                    id=ChunkId(f"{doc}@v1#{n}"),
                    doc_id=DocId(doc),
                    version=Version(1),
                    text=text,
                    page=n + 1,
                    valid_from=valid_from,
                    embedding=EMBEDDER.embed_one(text),
                )
            )
    return out


def index(per_doc: int = 3) -> InMemoryIndex:
    idx = InMemoryIndex()
    idx.upsert(chunks(per_doc))
    return idx
