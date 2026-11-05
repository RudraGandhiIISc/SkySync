"""The web app: accounts and sessions, plus the chat and admin routers.

Everyone signs in with an id and a password. Real people (ids that are
e-mail addresses) get in only while an admin has them on the allowlist and
in the directory, with an active account and a password. They create the
account themselves, confirmed by an e-mailed link (web/registration.py);
nobody outside the allowlist can register. Forgotten passwords are reset
through single-use links (web/password_links.py).
Locust's test identities (ids without "@", created from the terminal by
`askiisc.web.cli seed-locust`) sign in only while `ASKIISC_TEST_LOGINS`
is on. Department, role and programme always come from the directory.

Every failed sign-in gets the same answer, whatever the reason, and takes
as long whether or not the account exists; repeated failures are slowed
down (web/throttle.py). A sign-in always starts a new session and ends any
session the browser presented. Sessions are random cookies stored hashed;
each request re-checks the account and the allowlist, then gets a
short-lived internal token for the gateway and the access service.

Every state-changing call must carry the header `X-AskIISc: 1` (browsers
cannot add it cross-site without CORS, which is never enabled) and, when
the browser sends an Origin header, come from this site. Together with
SameSite=Strict cookies this stops CSRF.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, Gauge
from pydantic import BaseModel, ConfigDict, Field

from askiisc.access.directory import PROGRAMS, Directory, Person
from askiisc.access.policy import INSTITUTE
from askiisc.common.clock import Clock
from askiisc.common.errors import AuthError, ConfigError
from askiisc.common.tokens import TokenIssuer
from askiisc.common.types import UserId
from askiisc.gateway.app import GatewayDeps, Origin
from askiisc.gateway.auth import Caller, tenant_of
from askiisc.gateway.early import EarlyAdmission
from askiisc.observability.timing import ask_kind
from askiisc.web.mailer import Mailer
from askiisc.web.passwords import (
    MAX_PASSWORD_CHARS,
    burn,
    check_strength,
    hash_password,
    needs_rehash,
    verify_password,
)
from askiisc.web.pending import PendingRegistrations
from askiisc.web.people import PeopleTree
from askiisc.web.requestlog import RequestLog
from askiisc.web.store import Account, AppStore
from askiisc.web.throttle import LoginThrottle

STATIC = Path(__file__).parent / "static"
COOKIE = "askiisc_session"
CSRF_HEADER = "x-askiisc"
# The pages load nothing from other origins and have no inline scripts, so
# the policy allows only this origin (inline style attributes excepted).
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
        # Grafana (reached through the same SSH tunnel) on the Grafana tab.
        "frame-src http://localhost:* http://127.0.0.1:*; "
        "form-action 'self'; object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
SESSION_TTL_S = 12 * 3600.0
SIGN_IN_FAILED = "wrong e-mail or password, or no active access"
INTERNAL_TOKEN_TTL_S = 600.0
INTERNAL_TOKEN_REUSE_S = 300.0

T = TypeVar("T")


class WebMetrics:
    """Web app counters on the gateway's registry (no user ids)."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.logins = Counter(
            "askiisc_logins", "Login attempts.", ["result"], registry=registry
        )
        self.chat_events = Counter(
            "askiisc_chat_events",
            "Chats created, renamed, deleted; messages sent.",
            ["event"],
            registry=registry,
        )
        self.ratings = Counter(
            "askiisc_answer_ratings",
            "Thumbs up and down.",
            ["rating"],
            registry=registry,
        )
        self.sessions = Gauge(
            "askiisc_sessions", "Live login sessions.", registry=registry
        )
        self.accounts = Gauge(
            "askiisc_accounts",
            "Accounts by status and kind.",
            ["status", "kind"],
            registry=registry,
        )


class InternalTokens:
    """Short-lived tokens for the gateway and access service, per user.

    One token is reused for a few minutes so the access client's filter
    cache (keyed by token) keeps working.
    """

    def __init__(self, issuer: TokenIssuer, clock: Clock) -> None:
        self._issuer = issuer
        self._clock = clock
        self._cache: dict[str, tuple[float, str]] = {}

    def for_user(self, user_id: str) -> str:
        """A valid token for the user."""
        now = self._clock.now()
        hit = self._cache.get(user_id)
        if hit is not None and now - hit[0] < INTERNAL_TOKEN_REUSE_S:
            return hit[1]
        token = self._issuer.issue(user_id, now + INTERNAL_TOKEN_TTL_S)
        self._cache[user_id] = (now, token)
        return token


@dataclass
class WebDeps:
    """Everything the web app is built from (all injected).

    Attributes:
        gateway: The gateway's dependencies (answers come from there).
        store: Accounts, sessions and chats.
        requests: The admin request log.
        directory: The IISc directory.
        tokens: Internal token source.
        clock: Time source.
        docs: Document id to {"title", "url", "kind"} for citations.
        chunk_text: A chunk's text by id (for links to the cited passage).
        access_http: HTTP client whose base URL is the access service
            (policy administration).
        system: Live system facts for the admin page (workers, index).
        revise: Publishes a new version of a document from text.
        find_docs: Documents whose id or title match a query (admin search).
        publish: Publishes a new version from file bytes
            (doc_id, data, content type, title).
        history: A document's versions (version, valid_from, valid_to,
            chunks), oldest first.
        models_dir: Per-model result folders (`results/models`).
        active_model: The ASKIISC_MODEL key this job serves.
        serve_job: The serving job's Slurm id (for switch instructions).
        chaos: Fault injections for load tests, by name.
        loadtests: Summaries of finished Locust runs.
        measurements: Summaries of other measurements (recall probes,
            graded answers, throughput, cost), for the scorecard.
        people: The editable users tree (None: read-only directory).
        mailer: Sends registration and reset links (None: no e-mail).
        registrations: Registrations waiting for their e-mailed link.
        email_domains: The e-mail domains people may be added with.
        throttle: Slows down repeated failed sign-ins.
        test_logins: Allow password sign-in for test identities.
        secure_cookies: Send cookies only over HTTPS.
    """

    gateway: GatewayDeps
    store: AppStore
    requests: RequestLog
    directory: Directory
    tokens: InternalTokens
    clock: Clock
    docs: Callable[[str], dict[str, str]] = lambda doc_id: {"title": doc_id}
    chunk_text: Callable[[str], str | None] = lambda chunk_id: None
    access_http: Any = None
    system: Callable[[], dict[str, Any]] = dict
    revise: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None
    publish: (
        Callable[[str, bytes, str, str], Awaitable[dict[str, Any]]] | None
    ) = None
    history: Callable[[str], list[dict[str, Any]]] | None = None
    find_docs: Callable[[str, int, str], dict[str, Any]] = lambda q, n, s: {
        "documents": [],
        "total": 0,
        "sites": [],
    }
    models_dir: Path | None = None
    active_model: str = ""
    serve_job: str = ""
    chaos: dict[str, Callable[[], Awaitable[str]]] = field(default_factory=dict)
    loadtests: Callable[[], list[dict[str, Any]]] = list
    measurements: Callable[[], list[dict[str, Any]]] = list
    people: PeopleTree | None = None
    mailer: Mailer | None = None
    registrations: PendingRegistrations | None = None
    email_domains: tuple[str, ...] = ("iisc.ac.in",)
    throttle: LoginThrottle | None = None
    test_logins: bool = True
    secure_cookies: bool = False
    metrics: WebMetrics | None = None

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = WebMetrics(self.gateway.metrics.registry)
        if self.throttle is None:
            self.throttle = LoginThrottle(self.clock)
        if self.registrations is None:
            self.registrations = PendingRegistrations(self.clock)

    @property
    def m(self) -> WebMetrics:
        """The metrics (always set after construction)."""
        if self.metrics is None:  # pragma: no cover - set in __post_init__
            raise ConfigError("web metrics not set")
        return self.metrics


@dataclass(frozen=True)
class Me:
    """The logged-in user."""

    user_id: str
    person: Person
    account: Account
    token: str

    @property
    def is_admin(self) -> bool:
        """Directory role is admin."""
        return self.person.role == "admin"

    @property
    def is_institute_admin(self) -> bool:
        """An admin of the institute folder (created from the terminal)."""
        return self.is_admin and self.person.department == INSTITUTE

    def scope(self) -> set[str] | None:
        """Departments this admin may see: every admin sees all (None).

        What an admin may change is narrower (people_admin.may_edit, and
        the access service for document policies).

        Raises:
            HTTPException: 403 if the user is not an admin.
        """
        if not self.is_admin:
            raise HTTPException(status_code=403, detail="admins only")
        return None


async def run(fn: Callable[..., T], *args: Any) -> T:
    """Runs a blocking store call off the event loop."""
    return await asyncio.to_thread(fn, *args)


def error(status: int, message: str) -> JSONResponse:
    """A JSON error response."""
    return JSONResponse({"error": message}, status_code=status)


class Credentials(BaseModel):
    """Sign-in form: an e-mail address (or a test id) and a password."""

    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)


class PasswordChange(BaseModel):
    """Change-password form."""

    model_config = ConfigDict(extra="forbid")
    old: str = Field(max_length=MAX_PASSWORD_CHARS)
    new: str = Field(max_length=MAX_PASSWORD_CHARS)


class Confirm(BaseModel):
    """Re-entered password for a destructive action."""

    model_config = ConfigDict(extra="forbid")
    password: str = Field(max_length=MAX_PASSWORD_CHARS)


def is_real(user_id: str) -> bool:
    """Whether an id names a real person (an e-mail address)."""
    return "@" in user_id


def normalise(user_id: str) -> str:
    """One spelling of an id: trimmed, lower-case."""
    return user_id.strip().lower()


def find_person(directory: Directory, user_id: str) -> Person | None:
    """Looks a user up as typed, then lower-cased (emails)."""
    typed = user_id.strip()
    return directory.get(UserId(typed)) or directory.get(UserId(typed.lower()))


def same_site(request: Request) -> bool:
    """Whether a state-changing request may come from this browser page.

    It needs the X-AskIISc header, and an Origin header, when the browser
    sends one, naming this host.
    """
    if request.headers.get(CSRF_HEADER) != "1":
        return False
    origin = request.headers.get("origin")
    if origin is None:
        return True
    return urlparse(origin).netloc == request.headers.get("host", "")


def session_dependency(web: WebDeps) -> Callable[[Request], Awaitable[Me]]:
    """Builds the FastAPI dependency that resolves the session cookie."""

    async def current(request: Request) -> Me:
        if request.method not in {"GET", "HEAD"} and not same_site(request):
            raise HTTPException(status_code=403, detail="cross-site request")
        cookie = request.cookies.get(COOKIE)
        now = web.clock.now()
        user = (
            None
            if not cookie
            else await run(web.store.session_user, cookie, now)
        )
        if user is None:
            raise HTTPException(status_code=401, detail="please log in")
        account = await run(web.store.account, user)
        person = web.directory.get(UserId(user))
        if account is None or account.status != "active" or person is None:
            raise HTTPException(status_code=401, detail="please log in")
        if not await run(may_sign_in, web, user):
            raise HTTPException(status_code=401, detail="please log in")
        return Me(user, person, account, web.tokens.for_user(user))

    return current


def session_caller(
    web: WebDeps,
) -> Callable[[Any], Awaitable[tuple[Caller, Origin] | None]]:
    """Names a question request's user and tenant from its cookie (FR-6).

    Used by EarlyAdmission: one session lookup, no account checks (the
    route still makes them). None for a cross-site request, no session or
    a user not in the directory; the route then refuses it.
    """

    async def identify(scope: Any) -> tuple[Caller, Origin] | None:
        request = Request(scope)
        cookie = request.cookies.get(COOKIE)
        if not cookie or not same_site(request):
            return None
        user = await run(web.store.session_user, cookie, web.clock.now())
        person = web.directory.get(UserId(user)) if user else None
        if user is None or person is None:
            return None
        caller = Caller(UserId(user), tenant_of(person), token="")  # nosec B106 - no token: the route verifies one
        source = ask_kind(request.url.path) or "chat"
        return caller, Origin(source, not is_real(user))

    return identify


def may_sign_in(web: WebDeps, user_id: str) -> bool:
    """Whether a user may hold a session now.

    Real people need an active allowlist entry (a revocation applies at
    the next request); test identities need test sign-ins switched on.
    """
    if is_real(user_id):
        allowed = web.store.allowed(user_id)
        return allowed is not None and allowed.status == "active"
    return web.test_logins


class SecurityHeaders:
    """ASGI middleware adding SECURITY_HEADERS to every HTTP response.

    It also adds `Cache-Control: no-cache` where the response sets none, so
    browsers revalidate pages and scripts (a cheap 304) instead of running
    an old script against a new page after an update.

    Plain ASGI (not BaseHTTPMiddleware), so streamed answers and client
    disconnects pass through untouched.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self._headers = [
            (k.lower().encode(), v.encode())
            for k, v in SECURITY_HEADERS.items()
        ]

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Passes the request on; adds the headers to the response start."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def with_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(k.lower() == b"cache-control" for k, _ in headers):
                    headers.append((b"cache-control", b"no-cache"))
                message["headers"] = [*headers, *self._headers]
            await send(message)

        await self._app(scope, receive, with_headers)


def mount_web(app: FastAPI, web: WebDeps) -> None:
    """Adds the web pages and /api routes to the gateway app."""
    from askiisc.web.admin import admin_router  # noqa: PLC0415
    from askiisc.web.chat import chat_router  # noqa: PLC0415

    # Inside SecurityHeaders, so an early 429 carries the same headers.
    app.add_middleware(
        EarlyAdmission, deps=web.gateway, identify=session_caller(web)
    )
    app.add_middleware(SecurityHeaders)
    current = session_dependency(web)
    app.include_router(auth_router(web, current))
    app.include_router(chat_router(web, current))
    app.include_router(admin_router(web, current))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page() -> FileResponse:
        return FileResponse(STATIC / "admin.html")

    @app.get("/admin/results", include_in_schema=False)
    async def results_page() -> FileResponse:
        return FileResponse(STATIC / "results.html")

    @app.get("/admin/models", include_in_schema=False)
    async def models_page() -> FileResponse:
        return FileResponse(STATIC / "models.html")


def _set_cookie(web: WebDeps, response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE,
        token,
        max_age=int(SESSION_TTL_S),
        httponly=True,
        samesite="strict",
        secure=web.secure_cookies,
        path="/",
    )


def auth_router(
    web: WebDeps, current: Callable[[Request], Awaitable[Me]]
) -> APIRouter:
    """Sign-in, sign-out, password links and the account."""
    from askiisc.web.password_links import link_routes  # noqa: PLC0415
    from askiisc.web.registration import register_routes  # noqa: PLC0415

    router = APIRouter(prefix="/api")

    def csrf(request: Request) -> None:
        if not same_site(request):
            raise HTTPException(status_code=403, detail="cross-site request")

    link_routes(web, router, csrf)
    register_routes(web, router, csrf)

    @router.get("/auth/options")
    async def options() -> dict[str, Any]:
        return {
            "test_logins": web.test_logins,
            "reset_by_email": web.mailer is not None,
        }

    @router.post("/auth/login", dependencies=[Depends(csrf)])
    async def login(body: Credentials, request: Request) -> Response:
        return await _login(web, body, request.cookies.get(COOKIE))

    @router.post("/auth/logout", dependencies=[Depends(csrf)])
    async def logout(request: Request) -> Response:
        cookie = request.cookies.get(COOKIE)
        if cookie:
            await run(web.store.end_session, cookie)
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    @router.get("/me")
    async def me(user: Me = Depends(current)) -> dict[str, Any]:  # noqa: B008
        return {
            "user_id": user.user_id,
            "department": user.person.department,
            "role": user.person.role,
            "program": user.person.program,
            "program_name": PROGRAMS.get(user.person.program, ""),
            "is_admin": user.is_admin,
            "institute_admin": user.is_institute_admin,
            "synthetic": user.account.synthetic,
        }

    @router.post("/me/password")
    async def change_password(
        body: PasswordChange,
        user: Me = Depends(current),  # noqa: B008
    ) -> Response:
        stored = await run(web.store.password_hash, user.user_id) or ""
        if not stored or not await run(verify_password, body.old, stored):
            return error(403, "current password is wrong")
        try:
            check_strength(body.new, user.user_id)
        except AuthError as err:
            return error(400, str(err))
        hashed = await run(hash_password, body.new)
        await run(web.store.set_password, user.user_id, hashed)
        response = JSONResponse({"ok": True, "logged_out": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    @router.delete("/me")
    async def delete_me(
        body: Confirm,
        user: Me = Depends(current),  # noqa: B008
    ) -> Response:
        stored = await run(web.store.password_hash, user.user_id) or ""
        if not stored or not await run(verify_password, body.password, stored):
            return error(403, "password is wrong")
        await run(web.store.delete_account, user.user_id)
        response = JSONResponse({"deleted": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    return router


async def _login(web: WebDeps, body: Credentials, old: str | None) -> Response:
    """Password sign-in. Every refusal looks and takes the same."""
    throttle = web.throttle
    if throttle is None:  # pragma: no cover - set in __post_init__
        raise ConfigError("no sign-in throttle")
    key = normalise(body.user_id)
    wait = throttle.wait_s(key)
    if wait > 0:
        web.m.logins.labels("throttled").inc()
        return JSONResponse(
            {"error": "too many attempts; try again later"},
            status_code=429,
            headers={"Retry-After": str(int(wait) + 1)},
        )
    person = find_person(web.directory, body.user_id)
    user_id = person.user_id if person else key
    account = await run(web.store.account, user_id)
    found = None
    if account is not None:
        found = await run(web.store.password_hash, user_id)
    stored = found or ""
    if stored:
        ok = await run(verify_password, body.password, stored)
    else:
        await run(burn, body.password)
        ok = False
    allowed = (
        ok
        and person is not None
        and account is not None
        and account.status == "active"
        and await run(may_sign_in, web, user_id)
    )
    if not allowed:
        throttle.failed(key)
        web.m.logins.labels("refused").inc()
        return error(401, SIGN_IN_FAILED)
    throttle.succeeded(key)
    if needs_rehash(stored):
        rehashed = await run(hash_password, body.password)
        await run(web.store.replace_hash, user_id, rehashed)
    now = web.clock.now()
    await run(web.store.login_succeeded, user_id, now)
    if old:
        await run(web.store.end_session, old)  # no session fixation
    token = await run(web.store.start_session, user_id, now, SESSION_TTL_S)
    web.m.logins.labels("ok").inc()
    response = JSONResponse({"user_id": user_id})
    _set_cookie(web, response, token)
    return response
