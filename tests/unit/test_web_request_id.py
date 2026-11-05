"""The chat app passes the gateway's audit record id to the client (FR-12)."""

from pathlib import Path

import pytest

from tests.web_fixtures import HDR, build_web, sse

QUESTION = "What are the hostel fees rules?"


async def test_chat_and_temporary_responses_carry_the_audit_id(
    tmp_path: Path,
) -> None:
    web = build_web(tmp_path, rate=1.0, burst=2.0)
    web.account("u_cds_stu")
    assert (await web.login("u_cds_stu")).status_code == 200
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    in_chat = await web.client.post(
        f"/api/chats/{chat['chat_id']}/messages",
        json={"question": QUESTION},
        headers=HDR,
    )
    temporary = await web.client.post(
        "/api/temporary", json={"question": QUESTION}, headers=HDR
    )
    shed = await web.client.post(
        "/api/temporary", json={"question": "Mess fees?"}, headers=HDR
    )
    assert [r.status_code for r in (in_chat, temporary, shed)] == [
        200,
        200,
        429,
    ]
    audited = [r.request_id for r in web.stack.audit.records]
    ids = [r.headers["X-Request-Id"] for r in (in_chat, temporary, shed)]
    assert ids == audited
    answer = dict(sse(in_chat))["answer"]
    assert answer["request_id"] == ids[0]


async def test_a_chat_refused_before_the_gateway_is_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (FR-12, run 2089/2090): the chat app answered 503 itself
    # when the history's access check timed out, and 30 such requests had
    # no audit record and no request id.
    from askiisc.common.errors import AccessTimeout

    web = build_web(tmp_path)
    web.account("u_cds_stu")
    assert (await web.login("u_cds_stu")).status_code == 200
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()

    async def timeout(token: str, at: float | None = None) -> None:
        raise AccessTimeout("slow")

    monkeypatch.setattr(web.stack.deps.access, "filter", timeout)
    r = await web.client.post(
        f"/api/chats/{chat['chat_id']}/messages",
        json={"question": QUESTION},
        headers=HDR,
    )
    assert r.status_code == 503  # noqa: PLR2004 - still fails closed
    record = web.stack.audit.records[-1]
    assert r.headers["X-Request-Id"] == record.request_id
    assert (record.decision, record.reason) == (
        "denied",
        "access_check_timeout",
    )
    assert QUESTION.lower() not in record.model_dump_json().lower()
