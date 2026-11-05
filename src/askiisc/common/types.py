"""Immutable domain types shared by every package.

Times are seconds since the Unix epoch, as returned by an injected Clock.
"""

from __future__ import annotations

from typing import NewType, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

UserId = NewType("UserId", str)
TenantId = NewType("TenantId", str)
DocId = NewType("DocId", str)
ChunkId = NewType("ChunkId", str)
Version = NewType("Version", int)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Chunk(_Frozen):
    """One retrievable piece of one version of one document.

    Attributes:
        id: Unique chunk id, stable across re-ingestion of the same version.
        doc_id: The document this chunk belongs to.
        version: The document version (1, 2, ...).
        text: The chunk text. Untrusted data, never instructions.
        page: 1-based page number in the source.
        valid_from: When this version became current.
        valid_to: When it was superseded; None while it is current.
        acl: Access groups recorded at ingestion (informational; decisions
            come from the access service).
        embedding: Optional embedding vector.
    """

    id: ChunkId
    doc_id: DocId
    version: Version = Field(ge=1)
    text: str
    page: int = Field(ge=1)
    valid_from: float
    valid_to: float | None = None
    acl: frozenset[str] = frozenset()
    embedding: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_validity(self) -> Self:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self

    def is_current(self, at: float | None = None) -> bool:
        """Says whether this chunk's version is valid at a time.

        Args:
            at: The time to check; None means "now, with no end date".

        Returns:
            True if the chunk is valid at `at` (or has no end when None).
        """
        if at is None:
            return self.valid_to is None
        ended = self.valid_to is not None and self.valid_to <= at
        return self.valid_from <= at and not ended


class DocRef(_Frozen):
    """One version of one document."""

    doc_id: DocId
    version: Version = Field(ge=1)


class Citation(_Frozen):
    """A reference from an answer to the chunk that supports it.

    Attributes:
        is_stale: The cited version was superseded when the answer was made.
        superseded_by: For a stale citation, the version that replaced it
            (the next revision, or the newer edition's current version);
            None when it is current or the replacement is unknown (FR-5).
    """

    doc_id: DocId
    version: Version = Field(ge=1)
    chunk_id: ChunkId
    page: int = Field(ge=1)
    is_stale: bool = False
    superseded_by: DocRef | None = None


class Answer(_Frozen):
    """A grounded answer, or an explicit abstention."""

    text: str
    citations: tuple[Citation, ...] = ()
    abstained: bool = False


class Page(_Frozen):
    """Extracted text of one page of a source document."""

    number: int = Field(ge=1)
    text: str


class ChunkDraft(_Frozen):
    """A chunk before it gets ids, version and validity times."""

    text: str
    page: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
