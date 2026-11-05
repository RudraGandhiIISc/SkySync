"""Append-only, privacy-preserving audit records (FR-12, NFR-9).

Users and queries are stored only as salted SHA-256 hashes. Query and
answer text are never written.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["answered", "abstained", "rejected", "denied", "error"]
RefusalReason = Literal[
    "invalid_token",
    "not_in_directory",
    "question_too_long",
    "rate_limited",
    "access_check_timeout",
    "access_check_failed",
]
"""Why a request was refused before an answer (fixed codes, never text)."""


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


STAGES = (
    "auth",
    "access",
    "start",
    "embed",
    "search",
    "rerank",
    "prompt",
    "cold_hold",
    "queue",
    "first_token",
    "generation",
    "finish",
    "filter",
    "deliver",
)
"""The named stages of a request, in path order (seconds in StageTimer)."""


class LatencyBreakdown(BaseModel):
    """Where one request's time went, in milliseconds (FR-12).

    `start_ms` is the wait from the gateway handing back the stream until
    the event loop starts running it; `prompt_ms` the time scope and
    prompt building around the retriever's stages; `first_token_ms` runs
    from dispatch to the model until its first token; `generation_ms` from
    the first token to the last; `finish_ms` parsing the answer and storing
    it in the answer cache; `deliver_ms` from sending the final answer
    event until the stream closes (the chat app enriching and saving it,
    and the client reading it). `other_ms` is the
    total minus the named stages (never below 0), so the fields always add
    up to the request's latency; a large `other_ms` means a stage is
    missing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auth_ms: float = Field(default=0.0, ge=0)
    access_ms: float = Field(default=0.0, ge=0)
    start_ms: float = Field(default=0.0, ge=0)
    embed_ms: float = Field(default=0.0, ge=0)
    search_ms: float = Field(default=0.0, ge=0)
    rerank_ms: float = Field(default=0.0, ge=0)
    prompt_ms: float = Field(default=0.0, ge=0)
    cold_hold_ms: float = Field(default=0.0, ge=0)
    queue_ms: float = Field(default=0.0, ge=0)
    first_token_ms: float = Field(default=0.0, ge=0)
    generation_ms: float = Field(default=0.0, ge=0)
    finish_ms: float = Field(default=0.0, ge=0)
    filter_ms: float = Field(default=0.0, ge=0)
    deliver_ms: float = Field(default=0.0, ge=0)
    other_ms: float = Field(default=0.0, ge=0)

    @classmethod
    def from_stages(
        cls, seconds: Mapping[str, float], total_ms: float
    ) -> LatencyBreakdown:
        """Builds a breakdown from a StageTimer's totals.

        Args:
            seconds: Seconds per stage name (names from STAGES).
            total_ms: The request's whole latency.

        Returns:
            The breakdown, with `other_ms` the unexplained remainder.

        Raises:
            ValueError: On a stage name not in STAGES.
        """
        unknown = set(seconds) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stages {sorted(unknown)}")
        named = {f"{s}_ms": seconds.get(s, 0.0) * 1e3 for s in STAGES}
        other = max(0.0, total_ms - sum(named.values()))
        return cls(**named, other_ms=other)

    def sum_ms(self) -> float:
        """The sum of every field, other_ms included."""
        return float(sum(self.model_dump().values()))


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
    active_workers: int = Field(default=0, ge=0)
    cold_start: bool = False
    latency: LatencyBreakdown = LatencyBreakdown()
    reason: RefusalReason | None = None


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
