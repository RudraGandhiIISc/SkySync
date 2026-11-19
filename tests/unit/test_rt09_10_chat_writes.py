"""Issues #9 and #10: one answer at a time per chat; a chat deleted while it
is answered keeps nothing; streams free what they hold however they end.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from askiisc.common.errors import NotFound
from askiisc.web.store import AppStore, load_key
from tests.gateway_fixtures import build, events
from tests.rt_helpers import (
    CDS,
    QUESTION,
    app_of,
    chat_of,
    dropped,
    orphans,
    roles,
    student,
)
from tests.web_fixtures import HDR


async def test_issue9_a_chat_answers_one_question_at_a_time(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path)
    chat = await chat_of(web)
    url = f"/api/chats/{chat}/messages"
    sends = [
        web.client.post(url, json={"question": f"{QUESTION} {i}"}, headers=HDR)
        for i in range(4)
    ]
    statuses = [r.status_code for r in await asyncio.gather(*sends)]
    assert 409 in statuses  # noqa: PLR2004 - a second send while one runs
    assert set(statuses) <= {200, 409}
    answered = statuses.count(200)
    assert roles(web, chat) == ["user", "assistant"] * answered
    regenerate = f"/api/chats/{chat}/regenerate"
    twice = await asyncio.gather(
        web.client.post(regenerate, headers=HDR),
        web.client.post(regenerate, headers=HDR),
    )
    assert sorted(r.status_code for r in twice) == [200, 409]
    assert roles(web, chat) == ["user", "assistant"] * answered
    assert web.stack.deps.admission.depth(CDS) == 0


async def test_issue10_deleting_a_chat_during_its_answer_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = await student(tmp_path)
    chat = await chat_of(web)
    store = web.deps.store
    real_add = store.add_message

    def add_after_delete(chat_id: str, role: str, *rest: Any) -> str:
        if role == "assistant":  # deleted while the answer streams
            store.delete_chat("u_cds_stu", chat_id)
        return real_add(chat_id, role, *rest)

    monkeypatch.setattr(store, "add_message", add_after_delete)
    r = await web.client.post(
        f"/api/chats/{chat}/messages", json={"question": QUESTION}, headers=HDR
    )
    assert r.status_code == 200  # noqa: PLR2004
    kinds = [kind for kind, _ in events(r)]
    assert "answer" in kinds
    assert "saved" not in kinds  # nothing was saved: the chat is gone
    assert orphans(store) == 0
    assert web.stack.deps.admission.depth(CDS) == 0


async def test_issue10_a_chat_deleted_before_its_question_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = await student(tmp_path)
    chat = await chat_of(web)
    store = web.deps.store
    real_add = store.add_message

    def add_after_delete(chat_id: str, role: str, *rest: Any) -> str:
        if role == "user":  # deleted in another tab after admission
            store.delete_chat("u_cds_stu", chat_id)
        return real_add(chat_id, role, *rest)

    monkeypatch.setattr(store, "add_message", add_after_delete)
    r = await web.client.post(
        f"/api/chats/{chat}/messages", json={"question": QUESTION}, headers=HDR
    )
    assert r.status_code == 404  # noqa: PLR2004 - was a 500
    assert web.stack.deps.admission.depth(CDS) == 0  # was leaked for good
    assert web.stack.audit.records[-1].reason == "client_closed"
    assert orphans(store) == 0


def test_issue10_old_orphans_go_at_start_up_and_with_delete_all(
    tmp_path: Path,
) -> None:
    key = load_key(tmp_path / "chat.key")
    store = AppStore(tmp_path / "app.db", key)
    with pytest.raises(NotFound):
        store.add_message("gone", "user", "q", {}, 1.0)
    insert = (
        "INSERT INTO messages (message_id, chat_id, role, content, meta, "
        "created) VALUES (?, 'gone', 'user', ?, '{}', 1.0)"
    )
    with store._lock, store._db:  # noqa: SLF001 - an orphan from rt-baseline
        store._db.execute(insert, ("m1", b"x"))  # noqa: SLF001
    assert orphans(store) == 1
    store.delete_all_chats("someone")
    assert orphans(store) == 0
    with store._lock, store._db:  # noqa: SLF001
        store._db.execute(insert, ("m2", b"x"))  # noqa: SLF001
    store.close()
    reopened = AppStore(tmp_path / "app.db", key)
    assert orphans(reopened) == 0


async def test_a_stream_dropped_before_it_starts_is_closed() -> None:
    stack = build()
    token = stack.w.token("u_cds_stu")
    await dropped(
        app_of(stack.client),
        "/v1/ask",
        {"question": QUESTION},
        {"authorization": f"Bearer {token}"},
    )
    assert stack.deps.admission.depth(CDS) == 0
    record = stack.audit.records[-1]
    assert (record.decision, record.reason) == ("error", "client_closed")


class _Plain:
    """An async iterator without aclose."""

    def __aiter__(self) -> _Plain:
        return self

    async def __anext__(self) -> bytes:
        raise StopAsyncIteration


async def test_managed_stream_runs_every_hook_once_and_raises_the_first() -> (
    None
):
    ran: list[str] = []

    def boom() -> None:
        ran.append("boom")
        raise RuntimeError("first")

    async def later() -> None:
        ran.append("later")

    from askiisc.gateway.streams import ManagedStream  # new in the fix

    stream = ManagedStream(_Plain(), on_close=(boom, later))
    stream.add_close_hook(lambda: ran.append("added"))
    with pytest.raises(RuntimeError, match="first"):
        await stream.aclose()
    await stream.aclose()  # a second close does nothing
    assert ran == ["boom", "later", "added"]


async def test_a_failure_after_admission_frees_the_ticket_and_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web = await student(tmp_path)
    chat = await chat_of(web)
    store = web.deps.store
    real_add = store.add_message
    calls: list[str] = []

    def failing(chat_id: str, role: str, *rest: Any) -> str:
        calls.append(role)
        if len(calls) == 1:
            raise RuntimeError("disk full")
        return real_add(chat_id, role, *rest)

    monkeypatch.setattr(store, "add_message", failing)
    url = f"/api/chats/{chat}/messages"
    with pytest.raises(RuntimeError, match="disk full"):
        await web.client.post(url, json={"question": QUESTION}, headers=HDR)
    assert web.stack.deps.admission.depth(CDS) == 0
    retry = await web.client.post(url, json={"question": QUESTION}, headers=HDR)
    assert retry.status_code == 200  # noqa: PLR2004 - the chat is free again
    assert roles(web, chat) == ["user", "assistant"]
