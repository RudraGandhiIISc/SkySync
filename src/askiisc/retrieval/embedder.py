"""Text embeddings from an HTTP embedding server, or a deterministic fake.

The server speaks TEI's `POST /embed` contract: `{"inputs": [...]}` in, a
list of vectors out. On this server it is `retrieval/embed_server.py`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from askiisc.common import requestid
from askiisc.common.errors import AskIIScError
from askiisc.common.servicekey import bearer
from askiisc.common.targets import EMBEDDING_DIM

Vector = tuple[float, ...]
# BAAI/bge-*-en-v1.5 retrieves better with this instruction on queries;
# BAAI/bge-m3 is trained without one.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIXES = {
    "BAAI/bge-base-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-small-en-v1.5": BGE_QUERY_PREFIX,
    "BAAI/bge-large-en-v1.5": BGE_QUERY_PREFIX,
}


def query_prefix(model: str) -> str:
    """The instruction an embedding model wants in front of queries."""
    return QUERY_PREFIXES.get(model, "")


_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Turns texts into unit-length vectors."""

    @property
    def dim(self) -> int:
        """The vector dimension."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embeds each text."""
        ...


class EmbeddingError(AskIIScError):
    """The embedding server failed."""


class HttpEmbedder:
    """Client for a TEI-compatible embedding server."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        dim: int = EMBEDDING_DIM,
        batch_size: int = 32,
        timeout_s: float = 10.0,
        api_key: str = "",
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/embed"
        self._dim = dim
        self._batch = batch_size
        self._timeout = timeout_s
        self._auth = bearer(api_key)

    @property
    def dim(self) -> int:
        """The vector dimension."""
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embeds texts in batches.

        Args:
            texts: The texts.

        Returns:
            One vector per text, in order.

        Raises:
            EmbeddingError: On HTTP errors, timeouts or wrong shapes.
        """
        vectors: list[Vector] = []
        for i in range(0, len(texts), self._batch):
            vectors.extend(await self._embed_batch(texts[i : i + self._batch]))
        return vectors

    async def _post(self, texts: Sequence[str]) -> httpx.Response:
        """Posts once, retrying once on a transport error (idempotent)."""
        headers = {**requestid.headers(), **self._auth}
        try:
            return await self._client.post(
                self._url,
                json={"inputs": list(texts)},
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.TransportError:
            return await self._client.post(
                self._url,
                json={"inputs": list(texts)},
                headers=headers,
                timeout=self._timeout,
            )

    async def _embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        try:
            response = await self._post(texts)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as err:
            raise EmbeddingError(f"embedding request failed: {err}") from err
        if len(data) != len(texts) or any(len(v) != self._dim for v in data):
            raise EmbeddingError("embedding server returned wrong shape")
        return [tuple(float(x) for x in v) for v in data]


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for tests.

    Texts that share words get similar vectors, so retrieval tests behave
    sensibly without a model.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        """The vector dimension."""
        return self._dim

    def embed_one(self, text: str) -> Vector:
        """Embeds one text synchronously."""
        values = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            values[bucket] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            values[0], norm = 1.0, 1.0
        return tuple(v / norm for v in values)

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embeds each text."""
        return [self.embed_one(t) for t in texts]
