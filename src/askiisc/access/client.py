"""Clients of the access service. Every failure raises AccessDenied.

There is no fallback that allows (CLAUDE.md safety rule 1).
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import httpx

from askiisc.access.policy import AccessFilter, PolicyEngine
from askiisc.common import requestid
from askiisc.common.clock import Clock
from askiisc.common.errors import AccessDenied, AccessTimeout, AuthError
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


PUBLIC_LIST_TIMEOUT_S = 5.0
"""The plain public list is large and fetched once per policy version."""


class HttpAccessClient:
    """HTTP client of the access service.

    Timeouts are short (50 ms). A connection error is retried once; a deny is
    never retried. Anything unexpected raises AccessDenied.

    Filters are fetched in compact form: the plain public documents (the
    same for every user) are fetched once per policy version, and each
    request carries only the user's other readable documents.
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
        self._public: AccessFilter | None = None

    async def _post(
        self,
        path: str,
        token: str,
        body: dict[str, object] | None,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {token}", **requestid.headers()}
        for attempt in range(2):
            try:
                response = await self._client.request(
                    "GET" if body is None else "POST",
                    self._base + path,
                    json=body,
                    headers=headers,
                    timeout=timeout_s or self._timeout,
                )
            except httpx.ConnectError as err:
                if attempt == 0:
                    continue
                raise AccessDenied("access service unreachable") from err
            except httpx.TimeoutException as err:
                raise AccessTimeout("access service timed out") from err
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
        data = await self._post(
            "/v1/filter", token, {"at": at, "compact": True}
        )
        extra, version = _ids_and_version(data)
        result = AccessFilter(extra, version)
        if data.get("public") is True:
            public = await self._public_set(token, version)
            # Users with nothing extra share one set object, so the index
            # can reuse its mask for all of them.
            result = AccessFilter(public | extra if extra else public, version)
        if at is None:
            self._cache[key] = (now, result)
        return result

    async def _public_set(self, token: str, version: int) -> frozenset[DocId]:
        """The plain public documents at exactly this policy version.

        Raises:
            AccessDenied: If the service cannot give that version (a policy
                changed in between; the caller's request fails closed).
        """
        cached = self._public
        if cached is None or cached.policy_version != version:
            data = await self._post(
                "/v1/public", token, None, PUBLIC_LIST_TIMEOUT_S
            )
            doc_ids, got = _ids_and_version(data)
            cached = AccessFilter(doc_ids, got)
            self._public = cached
        if cached.policy_version != version:
            raise AccessDenied("policy changed during the request")
        return cached.doc_ids


def _ids_and_version(data: dict[str, object]) -> tuple[frozenset[DocId], int]:
    doc_ids, version = data.get("doc_ids"), data.get("policy_version")
    if not isinstance(doc_ids, list) or not isinstance(version, int):
        raise AccessDenied("malformed access response")
    return frozenset(DocId(str(d)) for d in doc_ids), version
