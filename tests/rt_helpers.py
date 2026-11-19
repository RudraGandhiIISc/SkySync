"""Helpers for the red-team regression tests (tests/unit/test_rt*.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from askiisc.common.types import TenantId
from askiisc.web.store import AppStore
from tests.gateway_fixtures import final
from tests.web_fixtures import HDR, Web, build_web

QUESTION = "What are the hostel fees rules?"
CDS = TenantId("cds")


async def chat_of(web: Web) -> str:
    r = await web.client.post("/api/chats", json={}, headers=HDR)
    return str(r.json()["chat_id"])


async def student(tmp_path: Path, **stack: Any) -> Web:
    web = build_web(tmp_path, **stack)
    web.account("u_cds_stu")
    assert (await web.login("u_cds_stu")).status_code == 200  # noqa: PLR2004
    return web


def orphans(store: AppStore) -> int:
    """Messages whose chat no longer exists."""
    rows = store._all(  # noqa: SLF001
        "SELECT COUNT(*) FROM messages "
        "WHERE chat_id NOT IN (SELECT chat_id FROM chats)"
    )
    return int(rows[0][0])


async def dropped(
    app: Any, path: str, body: dict[str, Any], headers: dict[str, str]
) -> list[dict[str, Any]]:
    """POSTs to the app; the client is gone before the answer starts.

    The server's receive() answers the disconnect at once, as uvicorn does
    for a closed connection (ASGI spec 2.3).
    """
    pending = [
        {
            "type": "http.request",
            "body": json.dumps(body).encode(),
            "more_body": False,
        }
    ]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return pending.pop(0) if pending else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (k.lower().encode(), v.encode())
            for k, v in {
                "host": "testserver",
                "content-type": "application/json",
                **headers,
            }.items()
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
        "state": {},
    }
    await app(scope, receive, send)
    return sent


def app_of(client: httpx.AsyncClient) -> Any:
    return client._transport.app  # type: ignore[attr-defined]  # noqa: SLF001


def retrieved(response: httpx.Response) -> set[str]:
    payload = final(response)
    return {str(doc) for _, doc in payload["retrieved"]}  # type: ignore[attr-defined]


def roles(web: Web, chat: str) -> list[str]:
    return [m.role for m in web.deps.store.messages(chat)]
