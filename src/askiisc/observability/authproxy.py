"""A basic-auth front door for a service listening on a Unix socket.

Loki has no login of its own. In the serving job it listens on a Unix
socket in a private folder (only this account can open it), and this proxy
on 127.0.0.1 checks a user name and password before forwarding, the way
Prometheus checks its own basic auth. Grafana and Alloy use the password.
"""

from __future__ import annotations

import base64
import binascii
import hmac

import httpx
from fastapi import FastAPI, Request, Response

HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "authorization",
    "content-length",
}


def check_basic(header: str | None, user: str, password: str) -> bool:
    """Whether an Authorization header carries these credentials."""
    scheme, _, encoded = (header or "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        given = base64.b64decode(encoded, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return False
    name, _, secret = given.partition(":")
    ok_user = hmac.compare_digest(name.encode(), user.encode())
    ok_pass = hmac.compare_digest(secret.encode(), password.encode())
    return ok_user and ok_pass


def create_proxy(
    upstream: httpx.AsyncClient, user: str, password: str
) -> FastAPI:
    """Builds the proxy.

    Args:
        upstream: Client whose base URL is the service (for Loki, a client
            with a Unix-socket transport).
        user: The accepted user name.
        password: The accepted password.

    Returns:
        A FastAPI app forwarding every authenticated request.
    """
    app = FastAPI(title="ask-iisc-auth-proxy", docs_url=None, redoc_url=None)

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    async def forward(path: str, request: Request) -> Response:
        if not check_basic(
            request.headers.get("authorization"), user, password
        ):
            return Response(
                "unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="ask-iisc"'},
            )
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in HOP_HEADERS
        }
        reply = await upstream.request(
            request.method,
            "/" + path,
            params=request.query_params,
            content=await request.body(),
            headers=headers,
            timeout=60.0,
        )
        out = {
            k: v
            for k, v in reply.headers.items()
            if k.lower() not in HOP_HEADERS
        }
        out.pop("content-encoding", None)
        return Response(
            reply.content, status_code=reply.status_code, headers=out
        )

    return app


def main() -> None:  # pragma: no cover - a process entry point
    """Serves the proxy on 127.0.0.1 in front of a Unix socket."""
    import argparse  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    args = parser.parse_args()
    upstream = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(args.socket)),
        base_url="http://upstream",
    )
    password = args.password_file.read_text().strip()
    app = create_proxy(upstream, args.user, password)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
