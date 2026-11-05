"""FR-6 fast shed path: web questions are admitted before their route."""

from pathlib import Path
from typing import Any

import pytest

from askiisc.common.types import TenantId
from askiisc.gateway.early import MAX_EARLY_BODY, question_of, read_body
from tests.web_fixtures import HDR, build_web

QUESTION = "What are the hostel fees rules?"


async def chat_of(web: Any) -> str:
    r = await web.client.post("/api/chats", json={}, headers=HDR)
    return str(r.json()["chat_id"])


async def test_a_shed_chat_question_skips_the_history_and_access_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = build_web(tmp_path, rate=0.001, burst=1.0)
    web.account("u_cds_stu")
    assert (await web.login("u_cds_stu")).status_code == 200  # noqa: PLR2004
    chat = await chat_of(web)
    first = await web.client.post(
        f"/api/chats/{chat}/messages", json={"question": QUESTION}, headers=HDR
    )
    assert first.status_code == 200  # noqa: PLR2004
    calls: list[str] = []
    real_filter = web.stack.deps.access.filter

    async def counted(token: str, at: float | None = None) -> Any:
        calls.append(token)
        return await real_filter(token, at)

    monkeypatch.setattr(web.stack.deps.access, "filter", counted)
    shed = await web.client.post(
        f"/api/chats/{chat}/messages", json={"question": QUESTION}, headers=HDR
    )
    assert shed.status_code == 429  # noqa: PLR2004
    assert shed.headers["Retry-After"]
    assert calls == []  # no history access check, no gateway access check
    record = web.stack.audit.records[-1]
    assert (record.decision, record.reason) == ("rejected", "rate_limited")
    assert shed.headers["X-Request-Id"] == record.request_id
    assert web.stack.deps.admission.depth(TenantId(record.tenant)) == 0


async def test_an_admitted_request_that_fails_later_frees_its_place(
    tmp_path: Path,
) -> None:
    web = build_web(tmp_path)
    web.account("u_cds_stu")
    await web.login("u_cds_stu")
    missing = await web.client.post(
        "/api/chats/nope/messages", json={"question": QUESTION}, headers=HDR
    )
    assert missing.status_code == 404  # noqa: PLR2004
    answered = await web.client.post(
        "/api/temporary", json={"question": QUESTION}, headers=HDR
    )
    assert answered.status_code == 200  # noqa: PLR2004
    record = web.stack.audit.records[-1]
    assert record.decision in {"answered", "abstained"}
    # Both the 404 and the answered request gave their place back.
    assert web.stack.deps.admission.depth(TenantId(record.tenant)) == 0


async def test_requests_without_a_good_session_go_to_the_route(
    tmp_path: Path,
) -> None:
    web = build_web(tmp_path, rate=0.001, burst=1.0)
    web.account("u_cds_stu")
    await web.login("u_cds_stu")
    cross_site = await web.client.post(
        "/api/temporary",
        json={"question": QUESTION},
        headers={"Content-Type": "application/json"},
    )
    assert cross_site.status_code == 403  # noqa: PLR2004 - the route's check
    await web.client.post("/api/auth/logout", headers=HDR)
    signed_out = await web.client.post(
        "/api/temporary", json={"question": QUESTION}, headers=HDR
    )
    assert signed_out.status_code == 401  # noqa: PLR2004
    assert web.stack.audit.records == []  # nothing was admitted or shed


def test_question_of_reads_only_the_question_field() -> None:
    assert question_of(b'{"question": "Fees?", "history": []}') == "Fees?"
    assert question_of(b"not json") == ""
    assert question_of(b"[1, 2]") == ""
    assert question_of(b'{"question": 3}') == ""


def receiver(messages: list[dict[str, Any]]) -> Any:
    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    return receive


async def test_read_body_replays_what_it_read() -> None:
    parts = [
        {"type": "http.request", "body": b'{"q', "more_body": True},
        {"type": "http.request", "body": b'": 1}', "more_body": False},
        {"type": "http.disconnect"},
    ]
    body, replay = await read_body(receiver(list(parts)))
    assert body == b'{"q": 1}'
    assert [await replay() for _ in range(3)] == parts


async def test_a_large_or_cut_body_is_not_read_early() -> None:
    big = [
        {
            "type": "http.request",
            "body": b"x" * MAX_EARLY_BODY,
            "more_body": True,
        },
        {"type": "http.request", "body": b"x", "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
    ]
    body, replay = await read_body(receiver(list(big)))
    assert body is None
    assert [await replay() for _ in range(3)] == big
    cut, _ = await read_body(receiver([{"type": "http.disconnect"}]))
    assert cut is None
