"""A per-job key for the services behind the gateway (vLLM, embeddings).

Loopback ports are reachable by every user of the shared server, so the
vLLM workers and the embedding server accept a request only with the key
the serving job generated for them (`Authorization: Bearer <key>`). The
key reaches each child through its environment (`ASKIISC_SERVICE_KEY`),
never its command line, which other users can read in `ps`.

`RequireKey` is a plain ASGI middleware with no project imports beyond the
standard library and Starlette, so vLLM can load it from the serve venv
with `--middleware askiisc.common.servicekey.RequireKey`. Unlike vLLM's own
`--api-key`, which guards only paths under /v1, it guards every path but
the open ones (health checks and Prometheus counters).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send

KEY_ENV = "ASKIISC_SERVICE_KEY"
"""The environment variable a child reads its key from."""
WORKER_OPEN_PATHS = ("/health", "/metrics")
"""vLLM paths without a key: health checks and Prometheus counters."""
EMBED_OPEN_PATHS = ("/health",)
"""Embedding-server paths without a key."""
_KEY_BYTES = 32


def new_key() -> str:
    """Returns a fresh random key (256 bits, URL-safe)."""
    return secrets.token_urlsafe(_KEY_BYTES)


def bearer(key: str) -> dict[str, str]:
    """The header that carries a key; empty when there is no key."""
    return {"Authorization": f"Bearer {key}"} if key else {}


class RequireKey:
    """Answers 401 to any request without the key, except on open paths.

    Args:
        app: The application it wraps.
        key: The key; read from `ASKIISC_SERVICE_KEY` when not given.
        open_paths: Exact paths served without a key.

    Raises:
        ValueError: If there is no key: a service never runs unprotected.
    """

    def __init__(
        self,
        app: ASGIApp,
        key: str | None = None,
        open_paths: Iterable[str] = WORKER_OPEN_PATHS,
    ) -> None:
        key = os.environ.get(KEY_ENV, "") if key is None else key
        if not key:
            raise ValueError(f"no service key: set {KEY_ENV}")
        self.app = app
        self._digest = hashlib.sha256(key.encode()).digest()
        self._open = frozenset(open_paths)

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Passes a request on if it is allowed, else answers 401."""
        if scope["type"] not in ("http", "websocket") or self._allowed(scope):
            await self.app(scope, receive, send)
            return
        await _unauthorized(scope, send)

    def _allowed(self, scope: Scope) -> bool:
        if scope["type"] == "http" and scope.get("path", "") in self._open:
            return True
        return self._valid(_authorization(scope))

    def _valid(self, header: str) -> bool:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        digest = hashlib.sha256(token.strip().encode()).digest()
        return hmac.compare_digest(digest, self._digest)


def _authorization(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() == b"authorization":
            return bytes(value).decode("latin-1")
    return ""


async def _unauthorized(scope: Scope, send: Send) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    body = json.dumps({"error": "unauthorized"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
