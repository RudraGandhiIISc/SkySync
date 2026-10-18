"""Clients of the access service. Every failure raises AccessDenied.

There is no fallback that allows (CLAUDE.md safety rule 1).
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx

from askiisc.access.policy import AccessFilter, PolicyEngine
from askiisc.common.clock import Clock
from askiisc.common.errors import AccessDenied, AuthError
from askiisc.common.targets import ACCESS_CLIENT_TIMEOUT_S
from askiisc.common.tokens import TokenVerifier
from askiisc.common.types import DocId, UserId

DEFAULT_CACHE_TTL_S = 10.0
"""Filter cache lifetime; well under the 60 s revocation target (FR-3)."""


class AccessAPI(Protocol):
    """The access interface every implementation must satisfy (FR-4)."""

    async def authorize(
        self, token: str, doc_id: DocId, at: float | None = None
    ) -> bool:
        """Says whether the token's user may read the document."""
        ...

    async def filter(self, token: str, at: float | None = None) -> AccessFilter:
        """Lists the documents the token's user may read."""
        ...


class LocalAccess:
    """In-process implementation over a PolicyEngine."""

    def __init__(
        self, engine: PolicyEngine, verifier: TokenVerifier, clock: Clock
    ) -> None:
        self._engine = engine
        self._verifier = verifier
        self._clock = clock

    def _user(self, token: str) -> UserId:
        try:
            return self._verifier.verify(token)
        except AuthError as err:
            raise AccessDenied("bad token") from err

    async def authorize(
        self, token: str, doc_id: DocId, at: float | None = None
    ) -> bool:
        """Says whether the token's user may read the document.

        Raises:
            AccessDenied: If the token is invalid.
        """
        when = self._clock.now() if at is None else at
        return self._engine.authorize(self._user(token), doc_id, when).allowed

    async def filter(self, token: str, at: float | None = None) -> AccessFilter:
        """Lists the documents the token's user may read.

        Raises:
            AccessDenied: If the token is invalid.
        """
        when = self._clock.now() if at is None else at
        return self._engine.filter(self._user(token), when)


class HttpAccessClient:
    """HTTP client of the access service.

    Timeouts are short (50 ms). A connection error is retried once; a deny is
    never retried. Anything unexpected raises AccessDenied.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        clock: Clock,
        timeout_s: float = ACCESS_CLIENT_TIMEOUT_S,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    ) -> None:
        self._client = client
        self._base = base_url.rstrip("/")
        self._clock = clock
        self._timeout = timeout_s
        self._ttl = cache_ttl_s
        self._cache: dict[str, tuple[float, AccessFilter]] = {}

    async def _post(
        self, path: str, token: str, body: dict[str, object]
    ) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {token}"}
        for attempt in range(2):
            try:
                response = await self._client.post(
                    self._base + path,
                    json=body,
                    headers=headers,
                    timeout=self._timeout,
                )
            except httpx.ConnectError as err:
                if attempt == 0:
                    continue
                raise AccessDenied("access service unreachable") from err
            except httpx.HTTPError as err:
                raise AccessDenied(f"access service error: {err}") from err
            if response.status_code != httpx.codes.OK:
                raise AccessDenied(
                    f"access service said {response.status_code}"
                )
            try:
                data = response.json()
            except ValueError as err:
                raise AccessDenied("malformed access response") from err
            if not isinstance(data, dict):
                raise AccessDenied("malformed access response")
            return data
        raise AccessDenied("access service unreachable")  # pragma: no cover

    async def authorize(
        self, token: str, doc_id: DocId, at: float | None = None
    ) -> bool:
        """Says whether the token's user may read the document.

        Raises:
            AccessDenied: On any failure.
        """
        data = await self._post(
            "/v1/authorize", token, {"doc_id": doc_id, "at": at}
        )
        return data.get("allowed") is True

    async def filter(self, token: str, at: float | None = None) -> AccessFilter:
        """Lists the documents the token's user may read.

        Results for "now" (at=None) are cached for cache_ttl_s.

        Raises:
            AccessDenied: On any failure.
        """
        key = hashlib.sha256(token.encode()).hexdigest()
        now = self._clock.now()
        if at is None and key in self._cache:
            stored, cached = self._cache[key]
            if now - stored < self._ttl:
                return cached
        data = await self._post("/v1/filter", token, {"at": at})
        doc_ids, version = data.get("doc_ids"), data.get("policy_version")
        if not isinstance(doc_ids, list) or not isinstance(version, int):
            raise AccessDenied("malformed access response")
        result = AccessFilter(
            frozenset(DocId(str(d)) for d in doc_ids), version
        )
        if at is None:
            self._cache[key] = (now, result)
        return result
