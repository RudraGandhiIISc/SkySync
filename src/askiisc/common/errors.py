"""Typed exceptions. Code raises these instead of generic errors."""

from __future__ import annotations


class AskIIScError(Exception):
    """Base class for every Ask IISc error."""


class AccessDenied(AskIIScError):
    """The user may not read the requested content, or access is unknown.

    Raised on every failure of the access path, so callers fail closed.
    """


class AccessTimeout(AccessDenied):
    """The access service did not answer in time (still a deny).

    A subclass, so every caller that catches AccessDenied still fails
    closed; audit records use it to tell a timeout from a real denial.
    """


class RateLimited(AskIIScError):
    """A request was shed by the load limits (FR-6)."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class GpusFull(AskIIScError):
    """No GPU worker can start: other jobs fill every card (FR-10).

    A held request ends with this at once, carrying a retry time, instead
    of waiting out the whole hold.
    """

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AuthError(AskIIScError):
    """The login token is missing, malformed, expired or wrongly signed."""


class NotFound(AskIIScError):
    """A document, version or policy does not exist."""


class ConfigError(AskIIScError):
    """Configuration is missing or invalid."""


class LLMError(AskIIScError):
    """The language model server failed or returned something unusable."""


class ExtractionError(AskIIScError):
    """A document could not be turned into text (for example a scanned PDF)."""
