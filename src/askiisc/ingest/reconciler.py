"""Detects changed source documents and ingests their new versions (FR-5).

A document has changed when its ETag differs from the last one seen, or,
when it has no ETag, when the SHA-256 of its content differs (A3).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from askiisc.common.clock import Clock
from askiisc.common.errors import AskIIScError
from askiisc.common.targets import RECONCILE_INTERVAL_S
from askiisc.common.types import DocId

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDocument:
    """A document as listed by its source.

    Attributes:
        doc_id: Stable document id.
        content_type: MIME type of the content.
        etag: The source's ETag, if it gives one.
        acl: Access groups for this document.
        valid_from: For an edition of a yearly document, when it took
            effect (used when the document is first ingested); None means
            from the moment it is ingested.
        valid_to: When a newer edition replaced it; None while current.
        context: Title, edition and site, the first line of every chunk
            (seen by the embedder, keyword search, reranker and model).
        template: Hashes of its site's template lines, removed from HTML
            before chunking (ingest/boilerplate).
    """

    doc_id: DocId
    content_type: str
    etag: str | None = None
    acl: frozenset[str] = frozenset()
    valid_from: float | None = None
    valid_to: float | None = None
    context: str = ""
    template: frozenset[int] = frozenset()


class Source(Protocol):
    """A place documents come from (a website, a shared drive, a test)."""

    async def list_documents(self) -> Sequence[SourceDocument]:
        """Lists the documents currently published."""
        ...

    async def fetch(self, doc_id: DocId) -> bytes:
        """Fetches a document's current content."""
        ...


IngestFn = Callable[[SourceDocument, bytes, str, float], Awaitable[object]]


@dataclass
class ReconcileReport:
    """What one reconciliation pass did."""

    changed: list[DocId] = field(default_factory=list)
    unchanged: list[DocId] = field(default_factory=list)
    failed: list[DocId] = field(default_factory=list)
    stopped: bool = False


def content_fingerprint(data: bytes) -> str:
    """Returns the content hash used when a source gives no ETag."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class Reconciler:
    """Compares sources with what was ingested and ingests what changed."""

    def __init__(
        self,
        source: Source,
        ingest: IngestFn,
        clock: Clock,
        interval_s: float = RECONCILE_INTERVAL_S,
        concurrency: int = 1,
    ) -> None:
        self._source = source
        self._ingest = ingest
        self._clock = clock
        self._interval = interval_s
        self._concurrency = max(1, concurrency)
        self._seen: dict[DocId, str] = {}

    def seed(self, fingerprints: dict[DocId, str]) -> None:
        """Records fingerprints of versions ingested earlier."""
        self._seen.update(fingerprints)

    async def run_once(
        self, should_stop: Callable[[], bool] = lambda: False
    ) -> ReconcileReport:
        """Runs one pass over every source document.

        A failure on one document is logged and reported; the others still
        run. Up to `concurrency` documents are in progress at once (their
        extraction runs elsewhere, so a bulk ingest uses every core).

        Args:
            should_stop: Checked between documents; when it returns True the
                pass ends early (report.stopped), for example at a job's
                time limit.

        Returns:
            The pass report.
        """
        report = ReconcileReport()
        gate = asyncio.Semaphore(self._concurrency)

        async def one(doc: SourceDocument) -> None:
            async with gate:
                if report.stopped or should_stop():
                    report.stopped = True
                    return
                try:
                    changed = await self._reconcile(doc)
                except AskIIScError as err:
                    _LOG.warning("reconcile failed for %s: %s", doc.doc_id, err)
                    report.failed.append(doc.doc_id)
                    return
                target = report.changed if changed else report.unchanged
                target.append(doc.doc_id)

        docs = await self._source.list_documents()
        await asyncio.gather(*(one(doc) for doc in docs))
        return report

    async def _reconcile(self, doc: SourceDocument) -> bool:
        if doc.etag is not None and self._seen.get(doc.doc_id) == doc.etag:
            return False
        data = await self._source.fetch(doc.doc_id)
        fingerprint = doc.etag or content_fingerprint(data)
        if self._seen.get(doc.doc_id) == fingerprint:
            return False
        await self._ingest(doc, data, fingerprint, self._clock.now())
        self._seen[doc.doc_id] = fingerprint
        return True

    async def run_forever(self, passes: int | None = None) -> None:
        """Runs a pass every interval.

        Args:
            passes: Stop after this many passes; None runs until cancelled.
        """
        done = 0
        while passes is None or done < passes:
            await self.run_once()
            done += 1
            await self._clock.sleep(self._interval)
