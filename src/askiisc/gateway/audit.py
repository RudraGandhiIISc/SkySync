"""Append-only, privacy-preserving audit records (FR-12, NFR-9).

Users and queries are stored only as salted SHA-256 hashes. Query and
answer text are never written.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

Decision = Literal["answered", "abstained", "rejected", "denied", "error"]


def salted_hash(salt: str, value: str) -> str:
    """Returns HMAC-SHA-256(salt, value) as hex.

    Args:
        salt: The secret salt.
        value: The value to hide.

    Raises:
        ValueError: If the salt is empty.
    """
    if not salt:
        raise ValueError("an audit salt is required")
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()


class DocVersion(BaseModel):
    """A document version used for an answer."""

    model_config = ConfigDict(frozen=True)
    doc_id: str
    version: int


class AuditRecord(BaseModel):
    """One request's record. No raw text fields exist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    time: float
    as_of: float
    user_hash: str
    query_hash: str
    tenant: str
    policy_version: int
    documents: tuple[DocVersion, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    model: str
    decision: Decision
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    stale: bool = False


class AuditSink(Protocol):
    """Somewhere audit records are appended."""

    def write(self, record: AuditRecord) -> None:
        """Appends a record."""
        ...


class InMemoryAuditSink:
    """Keeps records in a list (tests)."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def write(self, record: AuditRecord) -> None:
        """Appends a record."""
        self.records.append(record)


class FileAuditSink:
    """Appends JSON lines to a file, one record per line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        """Appends a record and flushes it."""
        line = record.model_dump_json() + "\n"
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def read(self) -> list[AuditRecord]:
        """Reads every record back."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return [AuditRecord.model_validate_json(ln) for ln in lines if ln]
