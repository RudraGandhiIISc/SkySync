"""Typed exceptions. Code raises these instead of generic errors."""

from __future__ import annotations


class AskIIScError(Exception):
    """Base class for every Ask IISc error."""


class AccessDenied(AskIIScError):
    """The user may not read the requested content, or access is unknown.

    Raised on every failure of the access path, so callers fail closed.
    """


class RateLimited(AskIIScError):
    """A request was shed by the load limits (FR-6)."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AuthError(AskIIScError):
    """The login token is missing, malformed, expired or wrongly signed."""


class NotFound(AskIIScError):
    """A document, version or policy does not exist."""


class StaleContent(AskIIScError):
    """Content from a superseded document version was used."""


class ConfigError(AskIIScError):
    """Configuration is missing or invalid."""


class LLMError(AskIIScError):
    """The language model server failed or returned something unusable."""


class ExtractionError(AskIIScError):
    """A document could not be turned into text (for example a scanned PDF)."""
