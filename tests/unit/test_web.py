"""The web app: accounts, chats, temporary chats and the admin API."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from askiisc.common.errors import AuthError, NotFound
from askiisc.common.types import DocId
from askiisc.gateway.app import RequestEvent
from askiisc.web.app import SIGN_IN_FAILED
from askiisc.web.chat import REVOKED_TEXT
from askiisc.web.cli import seed_locust
from askiisc.web.passwords import (
    check_strength,
    hash_password,
    needs_rehash,
    verify_password,
)
from askiisc.web.requestlog import RequestLog, percentile
from askiisc.web.store import AccountExists, AppStore, load_key
from tests.web_fixtures import HDR, PASSWORD, Web, build_web, sse

QUESTION = "What are the hostel fees rules?"


@pytest.fixture
def web(tmp_path: Path) -> Web:
    return build_web(tmp_path)


async def _logged_in(web: Web, user: str = "u_cds_stu") -> Web:
    web.account(user)
    assert (await web.login(user)).status_code == 200
    return web


async def _send(
    web: Web, chat_id: str, question: str = QUESTION
) -> httpx.Response:
    return await web.client.post(
        f"/api/chats/{chat_id}/messages",
        json={"question": question},
        headers=HDR,
    )


# ------------------------------------------------------------- accounts


async def test_test_accounts_log_in_with_a_password(web: Web) -> None:
    c = web.client
    web.account("u_cds_stu")
    assert (await web.login("u_cds_stu")).status_code == 200
    me = (await c.get("/api/me")).json()
    assert me == {
        "user_id": "u_cds_stu",
        "department": "cds",
        "role": "student",
        "program": "",
        "program_name": "",
        "is_admin": False,
        "institute_admin": False,
        "synthetic": False,
    }
    admin = web.browser()
    web.account("u_cds_admin")
    await web.login("u_cds_admin", admin)
    users = (await admin.get("/api/admin/users")).json()["users"]
    assert [u["user_id"] for u in users] == ["u_cds_stu", "u_cds_admin"]
    me = (await admin.get("/api/me")).json()
    assert me["is_admin"] and not me["institute_admin"]
    r = await c.post(
        "/api/auth/login", json={"user_id": "u_cds_stu", "password": PASSWORD}
    )
    assert r.status_code == 403  # CSRF header required


def test_old_pending_accounts_are_activated(tmp_path: Path) -> None:
    store = AppStore(tmp_path / "a.db", load_key(tmp_path / "k"))
    store.create_account("a", "h", "pending", 1.0)
    store.create_account("b", "h", "disabled", 1.0)
    assert store.activate_pending() == 1
    assert [x.status for x in store.accounts()] == ["active", "disabled"]


async def test_unknown_or_wrong_logins(tmp_path: Path) -> None:
    web = build_web(tmp_path)
    web.account("u_phd")
    bad = {"user_id": "nobody", "password": PASSWORD}
    r = await web.client.post("/api/auth/login", json=bad, headers=HDR)
    assert r.status_code == 401
    assert (await web.login("u_phd")).status_code == 200


async def test_repeated_wrong_passwords_are_slowed_down(web: Web) -> None:
    # Five free failures, then each attempt must wait (30 s, then doubling);
    # the same for an address with no account, so nothing is revealed.
    web.account("u_cds_stu")
    for user in ("u_cds_stu", "nobody@iisc.example"):
        wrong = {"user_id": user, "password": "Wrong-password-1"}
        codes = []
        for _ in range(7):
            r = await web.client.post(
                "/api/auth/login", json=wrong, headers=HDR
            )
            codes.append(r.status_code)
        assert codes == [401] * 6 + [429], user
        assert r.headers["retry-after"] == "31"
    assert (await web.login("u_cds_stu")).status_code == 429  # even if right
    web.stack.w.clock.advance(31)
    assert (await web.login("u_cds_stu")).status_code == 200


async def test_sessions_need_cookie_csrf_and_an_active_account(
    web: Web,
) -> None:
    c = web.client
    assert (await c.get("/api/chats")).status_code == 401
    await _logged_in(web)
    assert (await c.post("/api/chats", json={})).status_code == 403
    assert (await c.post("/api/chats", json={}, headers=HDR)).status_code == 201
    web.deps.store.set_status("u_cds_stu", "disabled")
    assert (await c.get("/api/chats")).status_code == 401
    web.deps.store.set_status("u_cds_stu", "active")
    assert (await c.get("/api/chats")).status_code == 401  # sessions ended
    await web.login("u_cds_stu")
    web.stack.w.clock.advance(13 * 3600)
    assert (await c.get("/api/chats")).status_code == 401  # expired
    await web.login("u_cds_stu")
    assert (await c.post("/api/auth/logout", headers=HDR)).status_code == 200
    assert (await c.get("/api/chats")).status_code == 401


async def test_disabled_login_and_password_change(web: Web) -> None:
    web.account("u_ece_stu", status="disabled")
    r = await web.login("u_ece_stu")
    assert r.status_code == 401  # the same answer as a wrong password
    assert r.json()["error"] == SIGN_IN_FAILED
    await _logged_in(web)
    c = web.client
    r = await c.post(
        "/api/me/password",
        json={"old": "nope", "new": "Another-pass-1"},
        headers=HDR,
    )
    assert r.status_code == 403
    r = await c.post(
        "/api/me/password", json={"old": PASSWORD, "new": "short"}, headers=HDR
    )
    assert r.status_code == 400  # under 8 characters
    r = await c.post(
        "/api/me/password",
        json={"old": PASSWORD, "new": "Another-pass-1"},
        headers=HDR,
    )
    assert r.status_code == 200
    assert (await c.get("/api/chats")).status_code == 401
    login = {"user_id": "u_cds_stu", "password": "Another-pass-1"}
    assert (
        await c.post("/api/auth/login", json=login, headers=HDR)
    ).status_code == 200


async def test_delete_account_removes_chats(web: Web) -> None:
    await _logged_in(web)
    c = web.client
    await c.post("/api/chats", json={"title": "x"}, headers=HDR)
    r = await c.request(
        "DELETE", "/api/me", json={"password": "bad"}, headers=HDR
    )
    assert r.status_code == 403
    r = await c.request(
        "DELETE", "/api/me", json={"password": PASSWORD}, headers=HDR
    )
    assert r.json() == {"deleted": True}
    assert web.deps.store.account("u_cds_stu") is None
    assert web.deps.store.chats("u_cds_stu") == []


# ----------------------------------------------------------------- chats


async def test_chat_lifecycle(web: Web) -> None:
    await _logged_in(web)
    c = web.client
    chat = (await c.post("/api/chats", json={}, headers=HDR)).json()
    r = await _send(web, chat["chat_id"])
    events = sse(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "token" and kinds[-2:] == ["answer", "saved"]
    answer = dict(events)["answer"]
    assert answer["citations"][0]["title"].startswith("Doc ")
    assert "retrieved" not in answer
    saved = dict(events)["saved"]
    assert saved["title"] == QUESTION
    listed = (await c.get("/api/chats?q=hostel")).json()["chats"]
    assert [x["chat_id"] for x in listed] == [chat["chat_id"]]
    assert (await c.get("/api/chats?q=zzz")).json()["chats"] == []
    opened = (await c.get(f"/api/chats/{chat['chat_id']}")).json()
    assert [m["role"] for m in opened["messages"]] == ["user", "assistant"]
    r = await c.patch(
        f"/api/chats/{chat['chat_id']}", json={"title": "Fees"}, headers=HDR
    )
    assert r.status_code == 200
    r = await c.get(f"/api/chats/{chat['chat_id']}/export")
    assert r.text.startswith("# Fees") and "**Ask IISc:**" in r.text
    assert "[chunk:" not in r.text  # numbered sources, as in the app
    r = await c.post(
        f"/api/messages/{saved['answer_id']}/rating",
        json={"rating": 1},
        headers=HDR,
    )
    assert r.status_code == 200
    assert web.deps.store.ratings() == {"up": 1, "down": 0}
    bad = await c.post(
        f"/api/messages/{saved['question_id']}/rating",
        json={"rating": 1},
        headers=HDR,
    )
    assert bad.status_code == 404  # only answers can be rated
    r = await c.post(f"/api/chats/{chat['chat_id']}/regenerate", headers=HDR)
    assert [e for e, _ in sse(r)][-1] == "saved"
    opened = (await c.get(f"/api/chats/{chat['chat_id']}")).json()
    assert len(opened["messages"]) == 2
    assert (
        await c.delete(f"/api/chats/{chat['chat_id']}", headers=HDR)
    ).status_code == 200
    assert (await c.get(f"/api/chats/{chat['chat_id']}")).status_code == 404


async def test_follow_ups_carry_history_and_skip_the_cache(web: Web) -> None:
    await _logged_in(web)
    first = dict(sse(await _temporary(web)))["answer"]
    again = dict(sse(await _temporary(web)))["answer"]
    assert not first["cache_hit"] and again["cache_hit"]
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    opening = dict(sse(await _send(web, chat["chat_id"])))["answer"]
    assert opening["cache_hit"]  # no history yet: same as a single question
    follow = dict(sse(await _send(web, chat["chat_id"])))["answer"]
    assert follow["cache_hit"] is False
    opened = (await web.client.get(f"/api/chats/{chat['chat_id']}")).json()
    assert len(opened["messages"]) == 4


async def test_long_chats_and_long_questions_keep_working(web: Web) -> None:
    # Regression: every earlier turn went into the request, whose schema
    # allows 20 turns of 6,000 characters, so the 12th question of a chat
    # (or any follow-up to a 6,000-character question) failed with a 500.
    await _logged_in(web)
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    long_question = " ".join(["hostel" + "x" * 64] * 100)  # 7,100 chars
    assert len(long_question) > 6000
    for question in [QUESTION] * 11 + [long_question, QUESTION]:
        r = await _send(web, chat["chat_id"], question)
        assert r.status_code == 200
        assert "answer" in dict(sse(r))
    opened = (await web.client.get(f"/api/chats/{chat['chat_id']}")).json()
    assert len(opened["messages"]) == 26


async def _temporary(
    web: Web, history: list[dict[str, str]] | None = None
) -> httpx.Response:
    return await web.client.post(
        "/api/temporary",
        json={"question": QUESTION, "history": history or []},
        headers=HDR,
    )


async def test_temporary_chats_are_never_stored(web: Web) -> None:
    await _logged_in(web)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x [chunk:a]"},
    ]
    events = sse(await _temporary(web, history))
    assert [e for e, _ in events][-1] == "answer"
    assert web.deps.store.chats("u_cds_stu") == []
    web.deps.requests.flush()
    rows = web.deps.requests.rows(None)
    assert (
        rows[0]["source"] == "temporary" and rows[0]["user_id"] == "u_cds_stu"
    )


async def test_answers_citing_revoked_documents_are_hidden(web: Web) -> None:
    await _logged_in(web)
    c = web.client
    chat = (await c.post("/api/chats", json={}, headers=HDR)).json()
    answer = dict(sse(await _send(web, chat["chat_id"])))["answer"]
    doc = answer["citations"][0]["doc_id"]
    web.stack.w.policies.delete(DocId(doc))
    opened = (await c.get(f"/api/chats/{chat['chat_id']}")).json()
    hidden = opened["messages"][1]
    assert hidden["content"] == REVOKED_TEXT and hidden["meta"] == {
        "hidden": True
    }
    shown = [(m["content"], m["meta"]) for m in opened["messages"]]
    assert doc not in json.dumps(shown)  # ids are random hex: not checked


async def test_people_never_see_each_others_chats(web: Web) -> None:
    await _logged_in(web)
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    other = web.browser()
    web.account("u_ece_stu")
    await web.login("u_ece_stu", other)
    cid = chat["chat_id"]
    assert (await other.get(f"/api/chats/{cid}")).status_code == 404
    assert (await other.get(f"/api/chats/{cid}/export")).status_code == 404
    r = await other.patch(f"/api/chats/{cid}", json={"title": "x"}, headers=HDR)
    assert r.status_code == 404
    assert (
        await other.delete(f"/api/chats/{cid}", headers=HDR)
    ).status_code == 404
    assert (
        await other.post(f"/api/chats/{cid}/regenerate", headers=HDR)
    ).status_code == 404
    r = await other.post(
        f"/api/chats/{cid}/messages", json={"question": "q"}, headers=HDR
    )
    assert r.status_code == 404
    assert (await other.delete("/api/chats", headers=HDR)).json() == {
        "deleted": 0
    }
    assert web.deps.store.chats("u_cds_stu")


async def test_rejections_pass_through_and_regenerate_needs_an_answer(
    tmp_path: Path,
) -> None:
    web = build_web(tmp_path, rate=0.001, burst=1.0)
    await _logged_in(web)
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    r = await web.client.post(
        f"/api/chats/{chat['chat_id']}/regenerate", headers=HDR
    )
    assert r.status_code == 400
    assert (await _send(web, chat["chat_id"])).status_code == 200
    r = await _send(web, chat["chat_id"])
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert (await _temporary(web)).status_code == 429


async def test_a_refused_regeneration_keeps_the_answer(tmp_path: Path) -> None:
    # Regression: regenerate deleted the question and answer before asking
    # again, so a 429 lost both.
    web = build_web(tmp_path, rate=0.001, burst=1.0)
    await _logged_in(web)
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()
    assert (await _send(web, chat["chat_id"])).status_code == 200
    r = await web.client.post(
        f"/api/chats/{chat['chat_id']}/regenerate", headers=HDR
    )
    assert r.status_code == 429
    opened = (await web.client.get(f"/api/chats/{chat['chat_id']}")).json()
    assert [m["role"] for m in opened["messages"]] == ["user", "assistant"]


async def test_pages_are_served(web: Web) -> None:
    page = (await web.client.get("/")).text
    assert "Ask IISc" in page and "Forgot password?" in page
    admin = (await web.client.get("/admin")).text
    assert "People" in admin and "Upload new document" in admin
    assert "Access policies" not in admin  # merged into Documents
    for name in ("ui.js", "chat.js", "admin.js", "policy_editor.js"):
        r = await web.client.get(f"/static/{name}")
        assert r.status_code == 200, name
    headers = (await web.client.get("/")).headers
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert "script-src 'self';" in headers["content-security-policy"]
    # Only local addresses may be framed (Grafana through the tunnel).
    assert (
        "frame-src http://localhost:* http://127.0.0.1:*;"
        in (headers["content-security-policy"])
    )
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    api = await web.client.get("/api/me")
    assert api.headers["referrer-policy"] == "no-referrer"
    # Regression: a cached old admin.js ran against a new admin.html and
    # left the overview empty. Scripts are revalidated on every load.
    script = await web.client.get("/static/admin.js")
    assert script.headers["cache-control"] == "no-cache"
    again = await web.client.get(
        "/static/admin.js", headers={"If-None-Match": script.headers["etag"]}
    )
    assert again.status_code == 304  # unchanged: nothing re-sent
    assert headers["cache-control"] == "no-cache"


# ----------------------------------------------------------------- admin


async def test_admin_overview_requests_and_scope(
    web: Web, tmp_path: Path
) -> None:
    await _logged_in(web)
    await _temporary(web)
    web.account("u_ece_stu", synthetic=True)
    ece = web.browser()
    await web.login("u_ece_stu", ece)
    await ece.post("/api/temporary", json={"question": QUESTION}, headers=HDR)
    web.deps.requests.flush()
    assert (await web.client.get("/api/admin/overview")).status_code == 403
    inst = web.browser()
    web.account("u_admin")
    await web.login("u_admin", inst)
    o = (await inst.get("/api/admin/overview?window_s=600")).json()
    assert o["scope"] == "all" and o["requests"]["requests"] == 2
    # Test ids (no "@") are Locust's whether or not the account is marked.
    assert o["requests"]["real"] == 0 and o["requests"]["locust"] == 2  # noqa: PLR2004
    assert o["users"]["accounts"] == 3 and o["users"]["locust"] == 3  # noqa: PLR2004
    assert o["system"] == {"model": "fake-model"}
    rows = (await inst.get("/api/admin/requests?synthetic=true")).json()["rows"]
    assert [r["user_id"] for r in rows] == ["u_ece_stu", "u_cds_stu"]
    rows = (
        await inst.get("/api/admin/requests?status=200&department=cds")
    ).json()["rows"]
    assert [r["user_id"] for r in rows] == ["u_cds_stu"]
    assert QUESTION not in json.dumps(rows)
    csv = (await inst.get("/api/admin/requests.csv")).text
    assert csv.splitlines()[0].startswith("request_id,time,user_id")
    assert len(csv.splitlines()) == 3
    series = (await inst.get("/api/admin/series?window_s=600&step_s=60")).json()
    assert sum(p["real"] + p["locust"] for p in series["points"]) == 2
    cds = web.browser()
    web.account("u_cds_admin")
    await web.login("u_cds_admin", cds)
    o = (await cds.get("/api/admin/overview")).json()
    assert o["scope"] == "all" and o["requests"]["requests"] == 2
    rows = (await cds.get("/api/admin/requests?department=ece")).json()["rows"]
    assert [r["user_id"] for r in rows] == ["u_ece_stu"]  # admins view all
    actions = (await inst.get("/api/admin/actions")).json()["actions"]
    assert {a["action"] for a in actions} == {
        "view_requests",
        "export_requests",
    }
    assert (await inst.get("/api/admin/loadtests")).status_code == 404
    assert QUESTION.encode() not in (tmp_path / "requests.db").read_bytes()


async def test_policies_revisions_and_chaos_through_the_admin_api(
    web: Web,
) -> None:
    inst = web.browser()
    web.account("u_admin")
    await web.login("u_admin", inst)
    r = await inst.get("/api/admin/policies/d02")
    assert r.status_code == 200 and r.json()["departments"] == ["cds"]
    r = await inst.put(
        "/api/admin/policies/d02",
        json={"owner": "institute", "departments": ["ece"]},
        headers=HDR,
    )
    assert r.status_code == 200
    assert web.stack.w.policies.snapshot()[1][DocId("d02")].departments == {
        "ece"
    }
    assert (
        await inst.delete("/api/admin/policies/d02", headers=HDR)
    ).status_code == 200
    assert (await inst.get("/api/admin/policies/d02")).status_code == 404
    for bad in ("%2E%2E", "..%2E", "D02", "-d02", "x" * 201):
        r = await inst.get(f"/api/admin/policies/{bad}")
        # Refused here (the access service would say "unknown document");
        # ".." would otherwise climb out of /v1/policy/ in the service URL.
        assert r.json() == {"error": "no such document"}, bad
    r = await inst.post(
        "/api/admin/documents/d01/revise", json={"text": "new"}, headers=HDR
    )
    assert r.json()["version"] == 2 and web.revisions == [("d01", "new")]  # type: ignore[attr-defined]
    r = await inst.post(
        "/api/admin/documents/zz/revise", json={"text": "x"}, headers=HDR
    )
    assert r.status_code == 404
    assert (
        await inst.post("/api/admin/chaos/kill_worker", headers=HDR)
    ).json() == {"result": "killed vllm-1"}
    assert (
        await inst.post("/api/admin/chaos/nope", headers=HDR)
    ).status_code == 404
    cds = web.browser()
    web.account("u_cds_admin")
    await web.login("u_cds_admin", cds)
    assert (
        await cds.post("/api/admin/chaos/kill_worker", headers=HDR)
    ).status_code == 404
    web.deps.revise = None
    web.deps.access_http = None
    assert (
        await inst.post(
            "/api/admin/documents/d01/revise", json={"text": "x"}, headers=HDR
        )
    ).status_code == 503
    assert (await inst.get("/api/admin/policies/d01")).status_code == 503
    actions = {a["action"] for a in web.deps.store.actions()}
    assert {
        "policy_put",
        "policy_delete",
        "document_revise",
        "chaos",
    } <= actions


async def test_access_outage_gives_503_on_open(web: Web) -> None:
    await _logged_in(web)
    chat = (await web.client.post("/api/chats", json={}, headers=HDR)).json()

    class Down:
        async def filter(self, token: str, at: float | None = None) -> object:
            from askiisc.common.errors import AccessDenied

            raise AccessDenied("down")

    web.stack.deps.access = Down()  # type: ignore[assignment]
    assert (
        await web.client.get(f"/api/chats/{chat['chat_id']}")
    ).status_code == 503
    assert (
        await web.client.get(f"/api/chats/{chat['chat_id']}/export")
    ).status_code == 503
    assert (await _send(web, chat["chat_id"])).status_code == 503


# ------------------------------------------------------- store and parts


def test_passwords() -> None:
    stored = hash_password("Some-pass-123")
    assert stored.startswith("$argon2id$")
    assert verify_password("Some-pass-123", stored)
    assert not verify_password("other", stored)
    assert not verify_password("x", "bcrypt$1$2$3$aa$bb")
    assert not verify_password("x", "garbage")
    assert not needs_rehash(stored)
    for bad in ("short1A", "a" * 257, "u_cds_stu1", "Real.Person"):
        with pytest.raises(AuthError):
            check_strength(
                bad, "real.person@iisc.example" if "." in bad else "u_cds_stu1"
            )
    check_strength("eightchr", "someone")  # 8 is enough; no character rules
    check_strength("x" * 256, "someone")  # long passphrases are kept whole
    long = "correct horse battery staple " * 8
    assert verify_password(long, hash_password(long))
    assert not verify_password(long[:-1], hash_password(long))


def test_store_encrypts_chat_text(tmp_path: Path) -> None:
    key = load_key(tmp_path / "k" / "chat.key")
    assert oct((tmp_path / "k" / "chat.key").stat().st_mode)[-3:] == "600"
    store = AppStore(tmp_path / "app.db", key)
    store.create_account("u1", "h", "active", 1.0)
    with pytest.raises(AccountExists):
        store.create_account("u1", "h", "active", 1.0)
    chat = store.create_chat("u1", "Secret title", 1.0)
    store.add_message(chat.chat_id, "user", "my secret question", {}, 2.0)
    raw = (tmp_path / "app.db").read_bytes()
    assert b"secret" not in raw
    other = AppStore(tmp_path / "app.db", load_key(tmp_path / "other.key"))
    assert other.messages(chat.chat_id)[0].content.startswith("[unreadable")
    with pytest.raises(NotFound):
        store.set_status("nobody", "active")
    with pytest.raises(NotFound):
        store.rename_chat("u2", chat.chat_id, "x")
    store.end_session(store.start_session("u1", 1.0, 10.0))
    assert store.live_sessions(1.0) == 0
    store.close()


def _event(**kw: object) -> RequestEvent:
    base: dict[str, object] = dict(
        request_id="r", time=100.0, user_id="u", tenant="cds", source="chat",
        synthetic=False, decision="answered", status=200, latency_ms=100.0,
        ttft_ms=50.0, prompt_tokens=10, completion_tokens=5, cache_hit=False,
        stale=False, citations=1, documents=("d@v1",), model="m",
        policy_version=1, cold_hold=False,
    )  # fmt: skip
    base.update(kw)
    return RequestEvent(**base)  # type: ignore[arg-type]


def test_request_log_queries(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "r.db", role_of=lambda u: "student", now=0.0)
    for i in range(20):
        log.record(
            _event(request_id=f"r{i}", time=100.0 + i, latency_ms=float(i * 10))
        )
    log.record(_event(request_id="x1", time=110, decision="error", status=500))
    log.record(
        _event(
            request_id="x2",
            time=111,
            status=429,
            decision="rejected",
            synthetic=True,
            user_id="v",
            tenant="ece",
        )
    )
    assert log.flush() == 22
    s = log.summary(None, 0.0, 130.0)
    assert s["requests"] == 22 and s["errors"] == 1 and s["rejected_429"] == 1
    assert s["p50_ms"] == 90 and s["p99_ms"] == 190 and s["users_active"] == 2
    assert s["by_status"] == {"200": 20, "500": 1, "429": 1}
    assert log.summary({"ece"}, 0.0, 130.0)["requests"] == 1
    series = log.series(None, 100.0, 10.0)
    assert [p["real"] for p in series] == [10, 11]
    assert series[1]["errors"] == 1 and series[1]["rejected"] == 1
    users = log.per_user(None, 0.0)
    assert users[0]["user_id"] == "u" and users[0]["requests"] == 21
    assert users[1]["synthetic"] is True
    assert len(log.rows(None, limit=5, before=105.0)) == 5
    assert log.rows(None, user_id="v")[0]["role"] == "student"
    assert percentile([], 50) is None
    log.start()
    log.record(_event(request_id="late", time=120))
    log.close()
    again = RequestLog(tmp_path / "r.db", now=120 + 31 * 86400)
    assert again.rows(None) == []


def test_seed_locust_only_for_test_users(tmp_path: Path) -> None:
    from askiisc.access.directory import Directory

    directory = Directory.from_records(
        [
            {"user_id": "stu_csa", "department": "csa", "role": "student"},
            {"user_id": "fac_csa", "department": "csa", "role": "faculty"},
            {
                "user_id": "a.real@iisc.example",
                "department": "csa",
                "role": "student",
            },
        ]
    )
    store = AppStore(tmp_path / "app.db", load_key(tmp_path / "k"))
    store.create_account("fac_csa", "h", "active", 1.0)  # registered by hand
    out = tmp_path / "private" / "locust_accounts.json"
    assert seed_locust(store, directory.people(), out, 1.0) == 1
    data = json.loads(out.read_text())
    assert [u["user_id"] for u in data["users"]] == ["stu_csa"]
    assert oct(out.stat().st_mode)[-3:] == "600"
    stu = store.account("stu_csa")
    assert stu is not None and stu.synthetic and stu.status == "active"
    assert store.account("a.real@iisc.example") is None
    assert seed_locust(store, directory.people(), out, 2.0) == 1  # re-seed
    stored = store.password_hash("stu_csa")
    assert stored is not None and verify_password(
        json.loads(out.read_text())["password"], stored
    )


def test_internal_tokens_are_reused_for_a_while(web: Web) -> None:
    tokens = web.deps.tokens
    first = tokens.for_user("u_cds_stu")
    assert tokens.for_user("u_cds_stu") == first
    web.stack.w.clock.advance(301)
    assert tokens.for_user("u_cds_stu") != first


def test_prometheus_basic_auth_file(tmp_path: Path) -> None:
    import shutil

    from deploy.serve import Paths, prometheus_auth

    if shutil.which("htpasswd") is None:
        pytest.skip("htpasswd not installed")
    paths = Paths(tmp_path / "run", tmp_path, tmp_path, tmp_path / "data")
    paths.run.mkdir()
    password, config = prometheus_auth(paths)
    users = json.loads(config.read_text())["basic_auth_users"]
    assert list(users) == ["prometheus"] and users["prometheus"].startswith(
        "$2"
    )
    assert password not in config.read_text()
    stored = paths.app / "prometheus_password"
    assert oct(stored.stat().st_mode)[-3:] == "600"
    assert prometheus_auth(paths)[0] == password  # same password on restart


async def test_policy_preview_counts_who_could_read(web: Web) -> None:
    await _logged_in(web)
    draft = {"doc_id": "d1", "departments": ["ece"], "deny_users": ["u_hostel"]}
    url = "/api/admin/policies/preview"
    assert (
        await web.client.post(url, json=draft, headers=HDR)
    ).status_code == 403
    inst = web.browser()
    web.account("u_admin")
    await web.login("u_admin", inst)
    r = (await inst.post(url, json=draft, headers=HDR)).json()
    assert r == {"matches": 2, "of": 12, "by_department": {"ece": 2}}
    public = {"doc_id": "d1", "visibility": "public"}
    assert (await inst.post(url, json=public, headers=HDR)).json()[
        "matches"
    ] == 12
    for bad in ({"doc_id": "d1", "visibility": "x"}, ["not", "a", "dict"]):
        r = await inst.post(url, json=bad, headers=HDR)
        assert r.status_code == 422


def test_export_numbers_the_sources() -> None:
    from askiisc.web.chat import export_markdown

    messages = [
        {"role": "user", "content": "Credits?", "meta": {}},
        {
            "role": "assistant",
            "content": "Twelve credits [chunk:ds@v1#0, chunk:ds@v1#0] in all.",
            "meta": {
                "citations": [
                    {
                        "chunk_id": "ds@v1#0",
                        "doc_id": "ds",
                        "title": "DS252",
                        "edition": "Aug 2025, academic year 2025-26",
                        "link": "https://cds.iisc.ac.in/ds252/",
                    }
                ]
            },
        },
    ]
    text = export_markdown("Credits", messages)
    assert "**Ask IISc:** Twelve credits [1] in all." in text
    assert (
        "1. DS252, Aug 2025, academic year 2025-26 <https://cds.iisc.ac.in/ds252/>"
        in text
    )


async def test_overview_and_users_filter_real_and_locust(web: Web) -> None:
    inst = web.browser()
    web.account("u_admin")
    await web.login("u_admin", inst)
    web.account("u_cds_stu", synthetic=True)
    real = "real.person@iisc.example"
    web.account(real)
    for user, synthetic in (("u_cds_stu", True), (real, False)):
        web.deps.requests.record(
            _event(
                request_id=user,
                time=web.deps.clock.now(),
                user_id=user,
                synthetic=synthetic,
            )
        )
    web.deps.requests.flush()
    counts = {}
    for origin in ("", "real", "locust"):
        body = (await inst.get(f"/api/admin/overview?origin={origin}")).json()
        users = (await inst.get(f"/api/admin/users?origin={origin}")).json()
        series = (await inst.get(f"/api/admin/series?origin={origin}")).json()
        counts[origin] = (
            body["requests"]["requests"],
            body["users"]["accounts"],
            sorted(u["user_id"] for u in users["users"]),
            sum(p["real"] + p["locust"] for p in series["points"]),
        )
    assert counts[""] == (2, 3, [real, "u_admin", "u_cds_stu"], 2)
    assert counts["real"] == (1, 1, [real], 1)
    assert counts["locust"] == (1, 2, ["u_admin", "u_cds_stu"], 1)


async def test_request_and_admin_logs_come_in_pages(web: Web) -> None:
    inst = web.browser()
    web.account("u_admin")
    await web.login("u_admin", inst)
    for n in range(25):
        web.deps.requests.record(
            _event(request_id=f"r{n}", time=1000.0 + n, user_id="u_admin")
        )
    web.deps.requests.flush()
    first = (await inst.get("/api/admin/requests?limit=20")).json()
    second = (await inst.get("/api/admin/requests?limit=20&offset=20")).json()
    assert first["total"] == second["total"] == 25  # noqa: PLR2004
    assert len(first["rows"]) == 20 and len(second["rows"]) == 5  # noqa: PLR2004
    assert first["rows"][0]["request_id"] == "r24"
    assert second["rows"][-1]["request_id"] == "r0"
    page = (await inst.get("/api/admin/actions?limit=1&offset=1")).json()
    assert page["total"] == 2 and len(page["actions"]) == 1  # noqa: PLR2004


async def test_every_overview_number_follows_the_users_filter(
    web: Web,
) -> None:
    # Regression: "signed in now" counted everyone whatever the filter.
    real = "real.person@iisc.example"
    web.account(real)  # a real person (not marked synthetic)
    web.deps.store.allow_existing([real], "test", 1.0)
    person = web.browser()
    assert (await web.login(real, person)).status_code == 200
    r = await person.post(
        "/api/temporary", json={"question": QUESTION}, headers=HDR
    )
    assert r.status_code == 200
    await _logged_in(web)  # u_cds_stu, a test id (Locust)
    assert (await _temporary(web)).status_code == 200
    admin = web.browser()
    web.account("u_admin", synthetic=True)
    await web.login("u_admin", admin)
    web.deps.requests.flush()

    async def overview(origin: str) -> dict[str, Any]:
        url = f"/api/admin/overview?origin={origin}"
        return (await admin.get(url)).json()  # type: ignore[no-any-return]

    both, only_real, only_locust = [
        await overview(o) for o in ("", "real", "locust")
    ]
    assert both["users"]["live_sessions"] == 3  # noqa: PLR2004
    assert only_real["users"]["live_sessions"] == 1
    assert only_locust["users"]["live_sessions"] == 2  # noqa: PLR2004
    assert (only_real["users"]["accounts"], only_real["users"]["locust"]) == (
        1,
        0,
    )
    assert only_locust["users"]["locust"] == only_locust["users"]["accounts"]
    assert only_real["requests"]["requests"] == 1
    assert only_real["requests"]["locust"] == 0
    assert only_locust["requests"]["real"] == 0
    assert both["requests"]["requests"] == 2  # noqa: PLR2004
