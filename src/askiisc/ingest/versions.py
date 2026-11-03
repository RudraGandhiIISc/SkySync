"""Version lineage: every document version is kept, old ones are retired.

Nothing is ever deleted (CLAUDE.md safety rule 6). Publishing a new version
sets the old version's `valid_to` and the new version's `valid_from` to the
same instant, so at every moment exactly one version is current.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
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

    def retire(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Ends the current version at `at` (a newer edition took over)."""
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

    def retire(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Ends the current version at `at`.

        Returns:
            The retired version, or None if the document is unknown or
            already retired.

        Raises:
            ValueError: If `at` is not after the version's start.
        """
        with self._lock:
            history = self._docs.get(doc_id)
            if not history or history[-1].valid_to is not None:
                return None
            if at <= history[-1].valid_from:
                raise ValueError("a version must end after it starts")
            history[-1].valid_to = at
            return history[-1]

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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    doc_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    valid_from REAL NOT NULL,
    valid_to REAL,
    fingerprint TEXT NOT NULL,
    acl TEXT NOT NULL,
    drafts TEXT NOT NULL,
    PRIMARY KEY (doc_id, version)
)
"""


class SQLiteVersionStore:
    """A VersionStore in one SQLite file, so versions outlive a job.

    Publishing (retire the old version, add the new one) is one
    transaction. Rows are never deleted.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Wait for locks rather than fail: SQLite locking over NFS is slow.
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, timeout=120.0
        )
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(_SCHEMA)

    def _row(self, row: tuple[object, ...]) -> VersionRecord:
        doc_id, version, start, end, fingerprint, acl, drafts = row
        return VersionRecord(
            doc_id=DocId(str(doc_id)),
            version=Version(int(str(version))),
            valid_from=float(str(start)),
            valid_to=None if end is None else float(str(end)),
            fingerprint=str(fingerprint),
            drafts=tuple(ChunkDraft(**d) for d in json.loads(str(drafts))),
            acl=frozenset(json.loads(str(acl))),
        )

    def _history(self, doc_id: DocId) -> list[VersionRecord]:
        rows = self._db.execute(
            "SELECT * FROM versions WHERE doc_id = ? ORDER BY version",
            (doc_id,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def publish(
        self,
        doc_id: DocId,
        drafts: Sequence[ChunkDraft],
        at: float,
        fingerprint: str,
        acl: frozenset[str] = frozenset(),
    ) -> VersionRecord:
        """Adds a version and retires the previous one in one transaction.

        Raises:
            ValueError: If `at` is not after the current version's start.
        """
        with self._lock, self._db:
            history = self._history(doc_id)
            if history and at <= history[-1].valid_from:
                raise ValueError("a new version must start after the old one")
            if history:
                self._db.execute(
                    "UPDATE versions SET valid_to = ? "
                    "WHERE doc_id = ? AND version = ?",
                    (at, doc_id, history[-1].version),
                )
            version = Version(len(history) + 1)
            self._db.execute(
                "INSERT INTO versions VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (
                    doc_id,
                    version,
                    at,
                    fingerprint,
                    json.dumps(sorted(acl)),
                    json.dumps([d.model_dump() for d in drafts]),
                ),
            )
        return VersionRecord(
            doc_id, version, at, None, fingerprint, tuple(drafts), acl
        )

    def retire(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Ends the current version at `at` (one transaction).

        Returns:
            The retired version, or None if the document is unknown or
            already retired.

        Raises:
            ValueError: If `at` is not after the version's start.
        """
        with self._lock, self._db:
            history = self._history(doc_id)
            if not history or history[-1].valid_to is not None:
                return None
            last = history[-1]
            if at <= last.valid_from:
                raise ValueError("a version must end after it starts")
            self._db.execute(
                "UPDATE versions SET valid_to = ? "
                "WHERE doc_id = ? AND version = ?",
                (at, doc_id, last.version),
            )
        last.valid_to = at
        return last

    def current(self, doc_id: DocId) -> VersionRecord:
        """Returns the current version of a document.

        Raises:
            NotFound: If the document is unknown.
        """
        with self._lock:
            history = self._history(doc_id)
        if not history:
            raise NotFound(f"unknown document {doc_id}")
        return history[-1]

    def as_of(self, doc_id: DocId, at: float) -> VersionRecord | None:
        """Returns the version that was current at `at`, if any."""
        with self._lock:
            history = self._history(doc_id)
        for record in history:
            ended = record.valid_to is not None and record.valid_to <= at
            if record.valid_from <= at and not ended:
                return record
        return None

    def history(self, doc_id: DocId) -> list[VersionRecord]:
        """Returns every version of a document, oldest first."""
        with self._lock:
            return self._history(doc_id)

    def documents(self) -> list[DocId]:
        """Returns every known document id."""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT doc_id FROM versions ORDER BY doc_id"
            ).fetchall()
        return [DocId(r[0]) for r in rows]

    def fingerprints(self) -> dict[DocId, str]:
        """Newest fingerprint of every document (to seed the reconciler).

        Retired editions count too: an edition superseded by a newer one is
        still ingested, and must not be ingested again on the next run.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT doc_id, fingerprint FROM versions AS v "
                "WHERE version = (SELECT max(version) FROM versions "
                "WHERE doc_id = v.doc_id)"
            ).fetchall()
        return {DocId(d): f for d, f in rows}

    def stamp(self) -> str:
        """A short summary that changes whenever any version is written.

        Snapshots of the index carry it, so a stale snapshot is detected.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT count(*), max(valid_from), total(valid_to) "
                "FROM versions"
            ).fetchone()
        return ":".join(repr(v) for v in row)

    def close(self) -> None:
        """Closes the database."""
        self._db.close()
