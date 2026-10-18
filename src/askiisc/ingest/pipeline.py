"""Ingests one document version end to end.

extract -> chunk -> store original -> publish version -> embed -> index.

The new version becomes visible `commit_delay_s` after ingestion starts.
Both index writes (insert the new chunks, retire the old ones) carry that
same future instant, so searches see the old version until then and only
the new version after: never both, never neither (FR-5).
"""

from __future__ import annotations

from dataclasses import dataclass

from askiisc.common.types import DocId, Version
from askiisc.ingest.chunk import chunk_pages
from askiisc.ingest.extract import extract_text
from askiisc.ingest.objects import ObjectStore
from askiisc.ingest.reconciler import SourceDocument
from askiisc.ingest.versions import VersionStore
from askiisc.retrieval.embedder import Embedder
from askiisc.retrieval.store import ChunkIndex

DEFAULT_COMMIT_DELAY_S = 5.0


@dataclass(frozen=True)
class IngestResult:
    """What ingesting one version produced."""

    doc_id: DocId
    version: Version
    chunks: int
    visible_from: float
    retired: int


@dataclass
class Ingestor:
    """Holds the stores an ingestion writes to."""

    versions: VersionStore
    objects: ObjectStore
    embedder: Embedder
    index: ChunkIndex
    commit_delay_s: float = DEFAULT_COMMIT_DELAY_S

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
        pages = extract_text(data, doc.content_type)
        drafts = chunk_pages(pages)
        visible_from = now + self.commit_delay_s
        previous = self.versions.history(doc.doc_id)
        record = self.versions.publish(
            doc.doc_id, drafts, visible_from, fingerprint, doc.acl
        )
        self.objects.put(
            f"{doc.doc_id}/v{record.version}", data, doc.content_type
        )
        chunks = record.chunks()
        vectors = await self.embedder.embed([c.text for c in chunks])
        self.index.upsert(
            [
                c.model_copy(update={"embedding": v})
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )
        retired = 0
        if previous:
            retired = self.index.retire(
                doc.doc_id, previous[-1].version, visible_from
            )
        return IngestResult(
            doc.doc_id, record.version, len(chunks), visible_from, retired
        )
