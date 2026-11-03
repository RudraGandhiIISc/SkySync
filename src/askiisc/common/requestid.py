"""The request id that ties one question's log lines together.

The gateway sets it at the start of a request; the HTTP clients of the
access service, the embedding server and vLLM send it as the
`X-Request-Id` header; each service logs it with one JSON line per call
(`log_requests`). Loki then finds every line of one request by its id.
Lines hold ids, paths, status codes and timings only, never text.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response

HEADER = "X-Request-Id"
REQUEST_ID: ContextVar[str | None] = ContextVar(
    "askiisc_request_id", default=None
)


def headers() -> dict[str, str]:
    """The header to send with an outgoing call, if a request is active."""
    current = REQUEST_ID.get()
    return {HEADER: current} if current else {}


def log_requests(app: FastAPI, service: str, logger: logging.Logger) -> None:
    """Logs one JSON line per HTTP call: id, method, path, status, time.

    Args:
        app: The service's FastAPI app.
        service: Its name in the log line (access, embed).
        logger: Where the lines go.
    """

    @app.middleware("http")
    async def request_line(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path not in {"/healthz", "/health", "/metrics"}:
            logger.info(
                json.dumps(
                    {
                        "service": service,
                        "request_id": request.headers.get(HEADER, ""),
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round((time.perf_counter() - started) * 1e3, 2),
                    }
                )
            )
        return response
