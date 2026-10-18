"""Version lineage: every document version is kept, old ones are retired.

Nothing is ever deleted (CLAUDE.md safety rule 6). Publishing a new version
sets the old version's `valid_to` and the new version's `valid_from` to the
same instant, so at every moment exactly one version is current.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from askiisc.common.errors import NotFound
from askiisc.common.types import Chunk, ChunkDraft, ChunkId, DocId, Version


@dataclass
class VersionRecord:
    """One stored version of one document.

    Attributes:
        doc_id: The document.
        version: Its number, starting at 1.
        valid_from: When it became current.
        valid_to: When it was superseded, or None while current.
        fingerprint: ETag or content hash used to detect changes.
        drafts: Its chunk drafts, in order.
        acl: Access groups recorded at ingestion.
    """

    doc_id: DocId
    version: Version
    valid_from: float
    valid_to: float | None
    fingerprint: str
    drafts: tuple[ChunkDraft, ...]
    acl: frozenset[str] = field(default_factory=frozenset)

    def chunks(self) -> list[Chunk]:
        """Builds this version's chunks with its validity times."""
        return [
            Chunk(
                id=chunk_id(self.doc_id, self.version, i),
                doc_id=self.doc_id,
                version=self.version,
                text=d.text,
                page=d.page,
                valid_from=self.valid_from,
                valid_to=self.valid_to,
                acl=self.acl,
            )
            for i, d in enumerate(self.drafts)
        ]


def chunk_id(doc_id: DocId, version: Version, index: int) -> ChunkId:
    """Builds the stable id of a chunk.

    Args:
        doc_id: The document.
        version: The document version.
        index: Position of the chunk in the version.

    Returns:
        An id of the form "<doc>@v<version>#<index>".
    """
    return ChunkId(f"{doc_id}@v{version}#{index}")


class VersionStore(Protocol):
    """Stores document versions."""

    def publish(
        self,
        doc_id: DocId,
        drafts: Sequence[ChunkDraft],
        at: float,
        fingerprint: str,
        acl: frozenset[str] = frozenset(),
    ) -> VersionRecord:
        """Adds a version and retires the previous one at the same instant."""
        ...

    def current(self, doc_id: DocId) -> VersionRecord:
        """Returns the current version of a document."""
        ...

    def as_of(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Returns the version that was current at `at`, if any."""
        ...

    def history(self, doc_id: DocId) -> list[VersionRecord]:
        """Returns every version of a document, oldest first."""
        ...

    def documents(self) -> list[DocId]:
        """Returns every known document id."""
        ...


class InMemoryVersionStore:
    """A thread-safe in-memory VersionStore."""

    def __init__(self) -> None:
        self._docs: dict[DocId, list[VersionRecord]] = {}
        self._lock = threading.Lock()

    def publish(
        self,
        doc_id: DocId,
        drafts: Sequence[ChunkDraft],
        at: float,
        fingerprint: str,
        acl: frozenset[str] = frozenset(),
    ) -> VersionRecord:
        """Adds a version and retires the previous one at the same instant.

        Args:
            doc_id: The document.
            drafts: The new version's chunks.
            at: When the new version becomes current.
            fingerprint: ETag or content hash of the new version.
            acl: Access groups for the new version.

        Returns:
            The new version record.

        Raises:
            ValueError: If `at` is not after the current version's start.
        """
        with self._lock:
            history = self._docs.setdefault(doc_id, [])
            if history and at <= history[-1].valid_from:
                raise ValueError("a new version must start after the old one")
            if history:
                history[-1].valid_to = at
            record = VersionRecord(
                doc_id=doc_id,
                version=Version(len(history) + 1),
                valid_from=at,
                valid_to=None,
                fingerprint=fingerprint,
                drafts=tuple(drafts),
                acl=acl,
            )
            history.append(record)
            return record

    def current(self, doc_id: DocId) -> VersionRecord:
        """Returns the current version of a document.

        Raises:
            NotFound: If the document is unknown.
        """
        with self._lock:
            history = self._docs.get(doc_id)
            if not history:
                raise NotFound(f"unknown document {doc_id}")
            return history[-1]

    def as_of(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Returns the version that was current at `at`, if any."""
        with self._lock:
            for record in self._docs.get(doc_id, []):
                ended = record.valid_to is not None and record.valid_to <= at
                if record.valid_from <= at and not ended:
                    return record
            return None

    def history(self, doc_id: DocId) -> list[VersionRecord]:
        """Returns every version of a document, oldest first."""
        with self._lock:
            return list(self._docs.get(doc_id, []))

    def documents(self) -> list[DocId]:
        """Returns every known document id."""
        with self._lock:
            return sorted(self._docs)
