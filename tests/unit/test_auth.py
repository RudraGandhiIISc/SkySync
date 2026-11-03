"""Local sign-in: passwords, the allowlist, sessions, links and limits."""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import httpx
import pytest
from deploy.serve import mailer as serve_mailer

from askiisc.common.clock import FakeClock
from askiisc.common.errors import ConfigError
from askiisc.web.app import COOKIE, SIGN_IN_FAILED
from askiisc.web.mailer import SmtpMailer, load_smtp
from askiisc.web.password_links import (
    NO_ACCOUNT,
    RESET_TTL_S,
    SEND_FAILED,
)
from askiisc.web.passwords import burn, needs_rehash, verify_password
from askiisc.web.pending import CONFIRM_TTL_S
from askiisc.web.registration import HAS_ACCOUNT, MISMATCH, NOT_ADDED
from askiisc.web.store import AppStore, load_key
from askiisc.web.throttle import (
    FIRST_DELAY_S,
    FREE_FAILURES,
    GLOBAL_FAILURES_PER_MIN,
    WINDOW_S,
    LoginThrottle,
)
from tests.web_fixtures import HDR, PASSWORD, Web, build_web

REAL = "real.person@iisc.example"  # in the web fixtures' directory


class FakeMailer:
    public_url = "http://ask.example"

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.fail = fail

    def send(self, to: str, subject: str, body: str) -> None:
        if self.fail:
            raise OSError("relay down")
        self.sent.append((to, subject, body))

    def token(self, marker: str = "#set-password=") -> str:
        body = self.sent[-1][2]
        return body.split(marker, 1)[1].split()[0]


@pytest.fixture
def web(tmp_path: Path) -> Web:
    return build_web(tmp_path)


def allow(web: Web, user: str = REAL) -> None:
    web.deps.store.allow_existing([user], "test", web.deps.clock.now())


async def sign_in(
    web: Web,
    user: str = REAL,
    password: str = PASSWORD,
    c: httpx.AsyncClient | None = None,
) -> httpx.Response:
    body = {"user_id": user, "password": password}
    return await (c or web.client).post(
        "/api/auth/login", json=body, headers=HDR
    )


# --------------------------------------------------------- sign-in


async def test_an_approved_active_person_signs_in(web: Web) -> None:
    web.account(REAL)
    allow(web)
    r = await sign_in(web, "  Real.Person@IISc.example ")  # any spelling
    assert r.status_code == 200 and r.json() == {"user_id": REAL}
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert (await web.client.get("/api/me")).json()["user_id"] == REAL


@pytest.mark.parametrize(
    "case",
    [
        "wrong password",
        "no account",
        "not approved",
        "revoked",
        "disabled",
        "no password yet",
    ],
)
async def test_every_refusal_looks_the_same(web: Web, case: str) -> None:
    password = PASSWORD
    if case == "wrong password":
        web.account(REAL)
        allow(web)
        password = "not-the-password"
    elif case == "not approved":
        web.account(REAL)  # right password, never allowed by an admin
    elif case == "revoked":
        web.account(REAL)
        allow(web)
        web.deps.store.set_access(REAL, "revoked", "admin", 1.0)
    elif case == "disabled":
        web.account(REAL, status="disabled")
        allow(web)
    elif case == "no password yet":
        with web.deps.store.allowing([REAL], "admin", 1.0):
            pass  # added by an admin, not registered: no password
    user = "nobody@iisc.example" if case == "no account" else REAL
    r = await sign_in(web, user, password)
    assert r.status_code == 401 and r.json() == {"error": SIGN_IN_FAILED}
    assert "set-cookie" not in r.headers


async def test_unknown_ids_cost_a_verification_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def spy(password: str, stored: str) -> bool:
        calls.append((password, stored))
        return False

    monkeypatch.setattr("askiisc.web.passwords.verify_password", spy)
    burn("guess")
    assert [c[0] for c in calls] == ["guess"]
    assert calls[0][1].startswith("$argon2id$")  # a real Argon2 check


def test_spraying_many_addresses_slows_everyone() -> None:
    clock = FakeClock(0.0)
    throttle = LoginThrottle(clock)
    for i in range(GLOBAL_FAILURES_PER_MIN):
        assert throttle.wait_s(f"u{i}@iisc.example") == 0
        throttle.failed(f"u{i}@iisc.example")
    assert throttle.wait_s("someone.else@iisc.example") == 60  # noqa: PLR2004
    clock.advance(61)
    assert throttle.wait_s("someone.else@iisc.example") == 0
    for _ in range(FREE_FAILURES + 1):
        throttle.failed("a@iisc.example")
    assert throttle.wait_s("a@iisc.example") == FIRST_DELAY_S
    throttle.succeeded("a@iisc.example")
    assert throttle.wait_s("a@iisc.example") == 0
    clock.advance(WINDOW_S + 1)  # old failures leave the window
    for _ in range(FREE_FAILURES):
        throttle.failed("b@iisc.example")
    clock.advance(WINDOW_S + 1)
    throttle.failed("b@iisc.example")
    assert throttle.wait_s("b@iisc.example") == 0


async def test_old_scrypt_passwords_work_once_then_become_argon2(
    web: Web,
) -> None:
    salt = b"0123456789abcdef"
    digest = hashlib.scrypt(PASSWORD.encode(), salt=salt, n=2**14, r=8, p=1)
    old = f"scrypt${2**14}$8$1${salt.hex()}${digest.hex()}"
    web.deps.store.create_account(REAL, old, "active", 1.0)
    allow(web)
    assert needs_rehash(old)
    assert (await sign_in(web)).status_code == 200
    stored = web.deps.store.password_hash(REAL)
    assert stored is not None and stored.startswith("$argon2id$")
    assert verify_password(PASSWORD, stored)


async def test_test_accounts_need_test_logins(web: Web) -> None:
    web.account("u_cds_stu", synthetic=True)
    assert (await sign_in(web, "u_cds_stu")).status_code == 200
    web.deps.test_logins = False
    assert (await web.client.get("/api/me")).status_code == 401  # session too
    assert (await sign_in(web, "u_cds_stu")).status_code == 401


# --------------------------------------------------------- sessions


async def test_signing_in_replaces_any_presented_session(web: Web) -> None:
    web.account(REAL)
    allow(web)
    planted = web.deps.store.start_session(REAL, 1.0, 3600)  # attacker's
    web.client.cookies.set(COOKIE, planted)
    r = await sign_in(web)
    fresh = r.cookies.get(COOKIE)
    assert fresh and fresh != planted
    assert web.deps.store.session_user(planted, 2.0) is None  # fixation fails
    assert web.deps.store.session_user(fresh, 2.0) == REAL


async def test_logout_ends_the_session_on_the_server(web: Web) -> None:
    web.account(REAL)
    allow(web)
    token = (await sign_in(web)).cookies.get(COOKIE)
    assert (
        await web.client.post("/api/auth/logout", headers=HDR)
    ).status_code == 200
    assert web.deps.store.session_user(str(token), 2.0) is None
    stolen = web.browser()
    stolen.cookies.set(COOKIE, str(token))
    assert (await stolen.get("/api/me")).status_code == 401


async def test_sessions_survive_a_restart_but_the_throttle_does_not(
    tmp_path: Path,
) -> None:
    web = build_web(tmp_path)
    web.account(REAL)
    allow(web)
    token = str((await sign_in(web)).cookies.get(COOKIE))
    reopened = AppStore(tmp_path / "app.db", load_key(tmp_path / "chat.key"))
    assert reopened.session_user(token, web.deps.clock.now()) == REAL
    web.deps.clock.advance(12 * 3600 + 1)  # type: ignore[attr-defined]
    assert reopened.session_user(token, web.deps.clock.now()) is None


async def test_rights_changes_and_suspension_end_sessions(web: Web) -> None:
    web.account(REAL)
    allow(web)
    assert (await sign_in(web)).status_code == 200
    web.deps.store.set_access(REAL, "revoked", "admin", 2.0)
    assert (await web.client.get("/api/chats")).status_code == 401
    web.deps.store.set_access(REAL, "active", "admin", 3.0)
    assert (await web.client.get("/api/chats")).status_code == 401  # signed out
    assert (await sign_in(web)).status_code == 200


# ------------------------------------------------------------- CSRF


async def test_state_changes_need_the_header_and_this_origin(web: Web) -> None:
    web.account(REAL)
    allow(web)
    await sign_in(web)
    for headers in ({}, {**HDR, "Origin": "https://evil.example"}):
        r = await web.client.post("/api/chats", json={}, headers=headers)
        assert r.status_code == 403, headers
        r = await web.client.post(
            "/api/auth/login",
            json={"user_id": REAL, "password": PASSWORD},
            headers=headers,
        )
        assert r.status_code == 403, headers
    same = {**HDR, "Origin": "http://testserver"}
    assert (
        await web.client.post("/api/chats", json={}, headers=same)
    ).status_code == 201


# --------------------------------------------------- access control


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/me"),
        ("get", "/api/chats"),
        ("post", "/api/chats"),
        ("post", "/api/temporary"),
        ("post", "/api/chats/x/messages"),
        ("get", "/api/admin/overview"),
        ("get", "/api/admin/people"),
        ("get", "/api/admin/requests"),
        ("get", "/api/admin/documents/x/versions"),
        ("post", "/v1/ask"),
    ],
)
async def test_anonymous_requests_are_refused(
    web: Web, method: str, path: str
) -> None:
    kwargs = {"json": {"question": "q"}} if method == "post" else {}
    r = await getattr(web.client, method)(path, headers=HDR, **kwargs)
    assert r.status_code == 401, path


async def test_nobody_reads_another_persons_chats(web: Web) -> None:
    web.account(REAL)
    allow(web)
    web.account("u_cds_stu")
    mine = web.deps.store.create_chat("u_cds_stu", "Mine", 1.0)
    answer = web.deps.store.add_message(mine.chat_id, "assistant", "x", {}, 1.0)
    await sign_in(web)
    for method, path, body in (
        ("get", f"/api/chats/{mine.chat_id}", None),
        ("get", f"/api/chats/{mine.chat_id}/export", None),
        ("patch", f"/api/chats/{mine.chat_id}", {"title": "taken"}),
        ("delete", f"/api/chats/{mine.chat_id}", None),
        ("post", f"/api/chats/{mine.chat_id}/messages", {"question": "q"}),
        ("post", f"/api/chats/{mine.chat_id}/regenerate", None),
        ("post", f"/api/messages/{answer}/rating", {"rating": 1}),
    ):
        kwargs = {"json": body} if body is not None else {}
        r = await web.client.request(
            method.upper(), path, headers=HDR, **kwargs
        )
        assert r.status_code == 404, path
    assert [c.title for c in web.deps.store.chats("u_cds_stu")] == ["Mine"]


async def test_users_cannot_make_themselves_admins(web: Web) -> None:
    web.account("u_cds_stu")
    await sign_in(web, "u_cds_stu")
    form = {
        "user_id": "u_cds_stu",
        "department": "cds",
        "role": "admin",
        "program": "",
        "groups": [],
    }
    for method, path, body in (
        ("put", "/api/admin/people/u_cds_stu", form),
        ("post", "/api/admin/people", {**form, "user_id": "x@iisc.example"}),
    ):
        kwargs = {"json": body} if body is not None else {}
        r = await web.client.request(
            method.upper(), path, headers=HDR, **kwargs
        )
        assert r.status_code == 403, path
    me = (await web.client.get("/api/me")).json()
    assert not me["is_admin"]


# ------------------------------------------------- password links


async def test_forgot_password_mails_a_ten_minute_link(web: Web) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    web.account(REAL)
    allow(web)
    for user in ("nobody@iisc.example", "u_cds_stu"):
        r = await web.client.post(
            "/api/auth/forgot", json={"user_id": user}, headers=HDR
        )
        assert r.status_code == 404 and r.json() == {"error": NO_ACCOUNT}
    r = await web.client.post(
        "/api/auth/forgot", json={"user_id": REAL.upper()}, headers=HDR
    )
    assert r.status_code == 200 and "10 minutes" in r.json()["message"]
    assert [m[0] for m in mail.sent] == [REAL]  # only to a real account
    assert "for 10 minutes" in mail.sent[0][2]
    token = mail.token()
    assert (
        web.deps.store.link_user(token, web.deps.clock.now() + RESET_TTL_S + 1)
        is None
    )
    new = {"token": token, "password": "a brand new phrase"}
    r = await web.client.post("/api/auth/set-password", json=new, headers=HDR)
    assert r.status_code == 200 and r.json()["user_id"] == REAL
    assert (
        await sign_in(web, password="a brand new phrase")
    ).status_code == 200
    assert (await sign_in(web)).status_code == 401  # old password gone
    r = await web.client.post("/api/auth/set-password", json=new, headers=HDR)
    assert r.status_code == 400  # single use


async def test_forgot_password_limits_and_mail_failures(web: Web) -> None:
    web.deps.mailer = FakeMailer(fail=True)
    web.account(REAL)
    allow(web)
    r = await web.client.post(
        "/api/auth/forgot", json={"user_id": REAL}, headers=HDR
    )
    assert r.status_code == 502 and r.json() == {"error": SEND_FAILED}
    web.deps.mailer = FakeMailer()
    for _ in range(3):
        r = await web.client.post(
            "/api/auth/forgot", json={"user_id": REAL}, headers=HDR
        )
        assert r.status_code == 200
    r = await web.client.post(
        "/api/auth/forgot", json={"user_id": REAL}, headers=HDR
    )
    assert r.status_code == 429  # 3 links an hour
    for _ in range(7):  # guessing addresses is slowed like sign-in
        r = await web.client.post(
            "/api/auth/forgot", json={"user_id": "x@iisc.example"}, headers=HDR
        )
    assert r.status_code == 429


# ------------------------------------------------------ registration


def register_body(
    user: str = REAL,
    password: str = "my long passphrase",  # noqa: S107 - a test password
    again: str = "",
) -> dict[str, str]:
    return {"user_id": user, "password": password, "confirm": again or password}


async def register(web: Web, **kw: str) -> httpx.Response:
    return await web.client.post(
        "/api/auth/register", json=register_body(**kw), headers=HDR
    )


async def confirm(web: Web, token: str) -> httpx.Response:
    return await web.client.post(
        "/api/auth/register/confirm", json={"token": token}, headers=HDR
    )


async def test_registration_explains_bad_passwords(web: Web) -> None:
    web.deps.mailer = FakeMailer()
    allow(web)
    r = await register(web, again="something else")
    assert r.status_code == 400 and r.json() == {"error": MISMATCH}
    r = await register(web, password="short")
    assert r.status_code == 400 and "at least 8 characters" in r.json()["error"]
    r = await register(web, password="real.person")
    assert r.status_code == 400 and "e-mail" in r.json()["error"]
    assert web.deps.mailer.sent == []  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "case",
    ["not in directory", "not allowed", "revoked", "disabled", "test id"],
)
async def test_only_allowlisted_people_can_register(
    web: Web, case: str
) -> None:
    web.deps.mailer = FakeMailer()
    user = REAL
    if case == "not in directory":
        user = "nobody@iisc.example"
        allow(web, user)
    elif case == "revoked":
        allow(web)
        web.deps.store.set_access(REAL, "revoked", "admin", 1.0)
    elif case == "disabled":
        allow(web)
        web.deps.store.create_account(REAL, "", "disabled", 1.0)
    elif case == "test id":
        user = "u_cds_stu"
    r = await register(web, user=user)
    assert r.status_code == 403 and r.json() == {"error": NOT_ADDED}


async def test_registration_is_confirmed_by_an_e_mailed_link(web: Web) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    with web.deps.store.allowing([REAL], "admin", 1.0):
        pass  # added by an admin: an account without a password
    r = await register(web, user="Real.Person@IISc.example")
    assert r.status_code == 202 and "10 minutes" in r.json()["message"]
    assert [m[0] for m in mail.sent] == [REAL]
    token = mail.token("#confirm=")
    assert (
        await sign_in(web, password="my long passphrase")
    ).status_code == 401
    r = await confirm(web, token)
    assert r.status_code == 200 and r.json()["user_id"] == REAL
    assert (
        await sign_in(web, password="my long passphrase")
    ).status_code == 200
    assert (await confirm(web, token)).status_code == 400  # single use
    r = await register(web)
    assert r.status_code == 409 and r.json() == {"error": HAS_ACCOUNT}
    actions = [a["action"] for a in web.deps.store.actions()]
    assert "registered" in actions


async def test_a_listed_person_without_an_account_gets_one(web: Web) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    allow(web)  # on the allowlist, no account row at all
    assert web.deps.store.account(REAL) is None
    assert (await register(web)).status_code == 202
    assert (await confirm(web, mail.token("#confirm="))).status_code == 200
    account = web.deps.store.account(REAL)
    assert account is not None and account.has_password


async def test_confirmation_links_expire_and_are_limited(web: Web) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    allow(web)
    assert (await register(web)).status_code == 202
    first = mail.token("#confirm=")
    assert (await register(web)).status_code == 202
    second = mail.token("#confirm=")
    assert (await confirm(web, first)).status_code == 400  # voided
    web.deps.clock.advance(CONFIRM_TTL_S + 1)  # type: ignore[attr-defined]
    assert (await confirm(web, second)).status_code == 400  # expired
    assert (await register(web)).status_code == 202
    r = await register(web)
    assert r.status_code == 429  # 3 an hour
    web.deps.clock.advance(3601)  # type: ignore[attr-defined]
    assert (await register(web)).status_code == 202


async def test_access_is_checked_again_when_confirming(web: Web) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    allow(web)
    assert (await register(web)).status_code == 202
    web.deps.store.set_access(REAL, "revoked", "admin", 2.0)
    r = await confirm(web, mail.token("#confirm="))
    assert r.status_code == 403 and r.json() == {"error": NOT_ADDED}
    assert web.deps.store.account(REAL) is None


async def test_registration_needs_working_e_mail(web: Web) -> None:
    allow(web)
    r = await register(web)
    assert r.status_code == 503 and "admin" in r.json()["error"]
    web.deps.mailer = FakeMailer(fail=True)
    r = await register(web)
    assert r.status_code == 502 and r.json() == {"error": SEND_FAILED}


async def test_links_expire_and_newer_ones_void_older(web: Web) -> None:
    web.account(REAL)
    allow(web)
    store, now = web.deps.store, web.deps.clock.now()
    first = store.create_link(REAL, now, RESET_TTL_S)
    second = store.create_link(REAL, now, RESET_TTL_S)
    assert (
        store.link_user(first, now) is None
        and store.link_user(second, now) == REAL
    )
    assert store.link_user(second, now + RESET_TTL_S + 1) is None
    short = {"token": second, "password": "short"}
    r = await web.client.post("/api/auth/set-password", json=short, headers=HDR)
    assert r.status_code == 400 and "8" in r.json()["error"]
    assert store.link_user(second, now) == REAL  # a refused try keeps it


async def test_adding_sends_nothing_and_the_person_registers(
    web: Web, tmp_path: Path
) -> None:
    mail = FakeMailer()
    web.deps.mailer = mail
    await web.institute_admin(tmp_path / "users")
    new = "new.one@iisc.example"
    form = {
        "user_id": "New.One@IISc.example",
        "department": "cds",
        "role": "student",
        "program": "phd",
        "joined": 2025,
        "groups": [],
    }
    r = await web.client.post("/api/admin/people", json=form, headers=HDR)
    assert r.status_code == 201 and "invite" not in r.json()
    assert mail.sent == []  # adding only allows them
    person = web.browser()
    body = register_body(user=new)
    r = await person.post("/api/auth/register", json=body, headers=HDR)
    assert r.status_code == 202
    r = await person.post(
        "/api/auth/register/confirm",
        json={"token": mail.token("#confirm=")},
        headers=HDR,
    )
    assert r.status_code == 200
    # Admins send no links: no invitations, no resets.
    for path in (
        f"/api/admin/people/{new}/password-link",
        f"/api/admin/users/{new}/disable",
    ):
        r = await web.client.post(path, headers=HDR)
        assert r.status_code in {404, 405}, path


async def test_only_allowed_domains_can_be_added(
    web: Web, tmp_path: Path
) -> None:
    await web.institute_admin(tmp_path / "users")
    form = {
        "user_id": "someone@gmail.com",
        "department": "cds",
        "role": "student",
        "program": "phd",
        "joined": 2025,
        "groups": [],
    }
    r = await web.client.post("/api/admin/people", json=form, headers=HDR)
    assert r.status_code == 422
    assert "only @iisc.example addresses" in r.json()["error"]
    csv = (
        "user_id,department,role,program,joined\n"
        "ok@iisc.example,cds,student,phd,2025\n"
        "x@iisc.exmaple,cds,student,phd,2025\n"
    )
    r = await web.client.post(
        "/api/admin/people/bulk", json={"csv": csv}, headers=HDR
    )
    assert r.status_code == 422 and r.json()["error"].startswith("row 2:")
    assert web.deps.store.account("someone@gmail.com") is None


async def test_forgot_without_mail_says_to_ask_the_admin(web: Web) -> None:
    r = await web.client.post(
        "/api/auth/forgot", json={"user_id": REAL}, headers=HDR
    )
    assert r.status_code == 503 and "admin" in r.json()["error"]
    assert (await web.client.get("/api/auth/options")).json()[
        "reset_by_email"
    ] is False


# ------------------------------------------- database and config


def test_existing_accounts_and_chats_survive_the_upgrade(
    tmp_path: Path,
) -> None:
    path = tmp_path / "app.db"
    old = sqlite3.connect(path)
    old.executescript(
        """CREATE TABLE accounts (user_id TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL, status TEXT NOT NULL,
        synthetic INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,
        approved_by TEXT, last_login REAL, failed INTEGER NOT NULL DEFAULT 0,
        locked_until REAL NOT NULL DEFAULT 0);
        CREATE TABLE chats (chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        title BLOB NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
        INSERT INTO accounts (user_id, password_hash, status, created)
        VALUES ('a@iisc.example', 'scrypt$1$1$1$00$00', 'active', 1.0);
        INSERT INTO chats VALUES ('c1', 'a@iisc.example', x'00', 1.0, 1.0);"""
    )
    old.commit()
    old.close()
    store = AppStore(path, load_key(tmp_path / "k"))
    account = store.account("a@iisc.example")
    assert account is not None and account.has_password
    assert [c.chat_id for c in store.chats("a@iisc.example")] == ["c1"]
    assert store.schema_version() == 2  # noqa: PLR2004


def test_an_older_release_keeps_a_newer_schema_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "app.db"
    AppStore(path, load_key(tmp_path / "k")).close()
    newer = sqlite3.connect(path)
    newer.execute("PRAGMA user_version = 9")  # written by a later release
    newer.commit()
    newer.close()
    store = AppStore(path, load_key(tmp_path / "k"))  # a rollback starts
    assert store.schema_version() == 9  # noqa: PLR2004


def test_a_bad_encryption_key_stops_start_up(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a valid Fernet key"):
        AppStore(tmp_path / "app.db", b"not-a-key")


def test_incomplete_mail_settings_turn_mail_off(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "smtp.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "host": "smtp.gmail.com",
                "from": "x",
                "public_url": "u",
                "user": "",
                "password": "",
            }
        )
    )
    with pytest.raises(ConfigError, match=r"lacks \['user', 'password'\]"):
        load_smtp(path)
    with caplog.at_level(logging.WARNING):
        assert serve_mailer(tmp_path) is None
    assert (
        "e-mail is off" in caplog.text
        and "password" not in caplog.text.split("lacks")[0]
    )
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_smtp(path)
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "host": "relay",
                "from": "x",
                "public_url": "u",
                "auth": False,
            }
        )
    )
    relay = load_smtp(path)
    assert relay is not None and relay.user == ""
    assert "secret" not in repr(load_smtp_with_password(tmp_path))


def load_smtp_with_password(tmp_path: Path) -> object:
    path = tmp_path / "smtp2.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "host": "h",
                "from": "f",
                "public_url": "u",
                "user": "me",
                "password": "secret",
            }
        )
    )
    return load_smtp(path)


class FakeSmtp:
    log: list[str] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.log.append(f"connect {host}:{port}")

    def __enter__(self) -> "FakeSmtp":
        return self

    def __exit__(self, *_: object) -> None:
        self.log.append("quit")

    def starttls(self) -> None:
        self.log.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.log.append(f"login {user}")

    def send_message(self, message: object) -> None:
        self.log.append(f"send {message['To']}")  # type: ignore[index]


def test_smtp_mail_uses_tls_and_signs_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askiisc.web.mailer.smtplib.SMTP", FakeSmtp)
    FakeSmtp.log.clear()
    mail = SmtpMailer(
        "smtp.example", 587, "Ask <a@x>", "u", user="a@x", password="pw"
    )
    mail.send("b@iisc.example", "s", "body")
    assert FakeSmtp.log == [
        "connect smtp.example:587",
        "starttls",
        "login a@x",
        "send b@iisc.example",
        "quit",
    ]
    assert "pw" not in repr(mail)
