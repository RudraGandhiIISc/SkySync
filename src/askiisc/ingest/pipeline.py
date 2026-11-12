"""Ingests one document version end to end.

extract -> remove the site template -> chunk -> store original -> publish
version -> embed -> index.

Every indexed chunk starts with its document's context line (title,
edition, site), so the embedder, the keyword search, the reranker and the
model all know which document and edition a chunk comes from even when its
own words do not say (a course outline page that never repeats the course
code). The version store keeps the drafts without that line.

The new version becomes visible `commit_delay_s` after ingestion starts.
Both index writes (insert the new chunks, retire the old ones) carry that
same future instant, so searches see the old version until then and only
the new version after: never both, never neither (FR-5).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from askiisc.common.errors import AskIIScError, ExtractionError
from askiisc.common.types import ChunkDraft, DocId, Version
from askiisc.ingest import boilerplate
from askiisc.ingest.chunk import chunk_pages
from askiisc.ingest.extract import HTML, extract_text
from askiisc.ingest.objects import ObjectStore
from askiisc.ingest.reconciler import SourceDocument
from askiisc.ingest.versions import VersionRecord, VersionStore
from askiisc.retrieval.embedder import Embedder
from askiisc.retrieval.store import ChunkIndex

DEFAULT_COMMIT_DELAY_S = 5.0

Chunker = Callable[
    [bytes, str, boilerplate.Template], Awaitable[list[ChunkDraft]]
]


def extract_and_chunk(
    data: bytes,
    content_type: str,
    template: boilerplate.Template = frozenset(),
) -> list[ChunkDraft]:
    """Extracts a document's text and chunks it (runs in a worker process).

    Args:
        data: The document bytes.
        content_type: Its MIME type.
        template: Its site's template lines, removed from HTML pages.

    Raises:
        ExtractionError: If the document has no text (or only template
            text).
    """
    pages = extract_text(data, content_type)
    kind = content_type.split(";", maxsplit=1)[0].strip().lower()
    if template and kind == HTML:
        pages = [
            p.model_copy(update={"text": boilerplate.strip(p.text, template)})
            for p in pages
        ]
        pages = [p for p in pages if p.text.strip()]
        if not pages:
            raise ExtractionError("only site template text")
    return chunk_pages(pages)


@dataclass(frozen=True)
class IngestResult:
    """What ingesting one version produced."""

    doc_id: DocId
    version: Version
    chunks: int
    visible_from: float
    retired: int


def retire_edition(
    versions: VersionStore, index: ChunkIndex, doc_id: DocId, at: float
) -> bool:
    """Retires a document's current version at `at` (a newer edition).

    Nothing is deleted (safety rule 6): the version and its chunks get
    valid_to, in the version store and in the index.

    Returns:
        True if a current version was retired, False if it already was.
    """
    record = versions.retire(doc_id, at)
    if record is None:
        return False
    index.retire(doc_id, record.version, at)
    return True


class RetirableIndex(Protocol):
    """What the start-up reconciliation needs from the search index."""

    def valid_to_of(self, doc_id: DocId, version: Version) -> float | None:
        """The latest valid_to of a version's chunks; None if none."""
        ...

    def retire(self, doc_id: DocId, version: Version, at: float) -> int:
        """Sets valid_to on every chunk of a version."""
        ...


def reconcile_retired(versions: VersionStore, index: RetirableIndex) -> int:
    """Retires in the index every version the version store has ended.

    The version store is the record: a revision is recorded first and its
    old chunks are retired a moment later, so a job killed in between left
    a superseded version searchable, unflagged, for days (job 2163,
    2026-11-06; found by run 2234). The serving job runs this at start-up.

    Returns:
        How many versions were retired in the index now.
    """
    fixed = 0
    for doc_id, version, valid_to in versions.retired():
        end = index.valid_to_of(doc_id, version)
        if end is not None and end > valid_to:
            index.retire(doc_id, version, valid_to)
            fixed += 1
    return fixed


@dataclass
class Ingestor:
    """Holds the stores an ingestion writes to."""

    versions: VersionStore
    objects: ObjectStore
    embedder: Embedder
    index: ChunkIndex
    commit_delay_s: float = DEFAULT_COMMIT_DELAY_S
    # Extract and chunk elsewhere (a process pool); None: in this thread.
    chunker: Chunker | None = None

    async def ingest(
        self, doc: SourceDocument, data: bytes, fingerprint: str, now: float
    ) -> IngestResult:
        """Ingests a new version of a document.

        Args:
            doc: The source listing entry.
            data: The document bytes.
            fingerprint: ETag or content hash of `data`.
            now: Current time.

        Returns:
            The ingestion summary.

        Raises:
            ExtractionError: If the document has no text.
        """
        if self.chunker is None:
            drafts = extract_and_chunk(data, doc.content_type, doc.template)
        else:
            drafts = await self.chunker(data, doc.content_type, doc.template)
        previous = self.versions.history(doc.doc_id)
        # An edition's own start applies when it is first ingested; a
        # changed document becomes visible after the commit delay (FR-5).
        if doc.valid_from is not None and not previous:
            visible_from = doc.valid_from
        else:
            visible_from = now + self.commit_delay_s
        if previous and visible_from <= previous[-1].valid_from:
            raise ValueError("a new version must start after the old one")
        # Chunks go into the index before the version is recorded, so a job
        # killed half-way leaves the document unrecorded and the next run
        # ingests it again (chunk ids are stable, so the upsert is
        # idempotent). Recording first could mark it done with no chunks.
        draft = VersionRecord(
            doc.doc_id,
            Version(len(previous) + 1),
            visible_from,
            None,
            fingerprint,
            tuple(drafts),
            doc.acl,
        )
        chunks = [
            c.model_copy(update={"text": f"{doc.context}\n{c.text}"})
            if doc.context
            else c
            for c in draft.chunks()
        ]
        vectors = await self.embedder.embed([c.text for c in chunks])
        self.objects.put(
            f"{doc.doc_id}/v{draft.version}", data, doc.content_type
        )
        self.index.upsert(
            [
                c.model_copy(update={"embedding": v})
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )
        record = self.versions.publish(
            doc.doc_id, drafts, visible_from, fingerprint, doc.acl
        )
        if record.version != draft.version:
            raise AskIIScError(f"concurrent ingestion of {doc.doc_id}")
        retired = 0
        if previous:
            retired = self.index.retire(
                doc.doc_id, previous[-1].version, visible_from
            )
        if doc.valid_to is not None and doc.valid_to > visible_from:
            retire_edition(self.versions, self.index, doc.doc_id, doc.valid_to)
        return IngestResult(
            doc.doc_id, record.version, len(chunks), visible_from, retired
        )
