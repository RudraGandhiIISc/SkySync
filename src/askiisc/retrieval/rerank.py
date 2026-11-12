"""Cross-encoder reranking of retrieved chunks.

Hybrid search (vectors and BM25) finds candidates quickly; a cross-encoder
reads the question and each candidate together and scores how well the
passage answers it, which orders the final top_k far better. It only
reorders chunks the user may already read: permission and validity filters
stay inside the search (safety rule 2).

The server is the embedding server's POST /rerank
({"query": ..., "texts": [...]} -> one score per text).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from askiisc.common import requestid
from askiisc.common.errors import AskIIScError
from askiisc.common.servicekey import bearer

_TOKEN = re.compile(r"[a-z0-9]+")


class RerankError(AskIIScError):
    """The reranking server failed."""


class Reranker(Protocol):
    """Scores passages against a question."""

    async def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text (higher is better)."""
        ...


class HttpReranker:
    """Client for the embedding server's /rerank endpoint."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_s: float = 10.0,
        api_key: str = "",
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/rerank"
        self._timeout = timeout_s
        self._auth = bearer(api_key)

    async def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Scores the texts.

        Raises:
            RerankError: On HTTP errors, timeouts or a wrong shape.
        """
        if not texts:
            return []
        try:
            response = await self._client.post(
                self._url,
                json={"query": query, "texts": list(texts)},
                headers={**requestid.headers(), **self._auth},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as err:
            raise RerankError(f"rerank request failed: {err}") from err
        if not isinstance(data, list) or len(data) != len(texts):
            raise RerankError("rerank server returned wrong shape")
        return [float(x) for x in data]


class FakeReranker:
    """Word-overlap scores, for tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Fraction of the question's words found in each text."""
        self.calls += 1
        words = set(_TOKEN.findall(query.lower()))
        if not words:
            return [0.0] * len(texts)
        return [
            len(words & set(_TOKEN.findall(t.lower()))) / len(words)
            for t in texts
        ]
