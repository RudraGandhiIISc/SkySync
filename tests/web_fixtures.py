"""The web app on top of the fake gateway stack (tests/gateway_fixtures)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from askiisc.access.directory import Directory
from askiisc.access.service import create_app as create_access
from askiisc.gateway.app import create_app
from askiisc.gateway.auth import Authenticator
from askiisc.web.app import InternalTokens, WebDeps, mount_web
from askiisc.web.passwords import hash_password
from askiisc.web.people import PeopleTree
from askiisc.web.requestlog import RequestLog
from askiisc.web.store import AppStore, load_key
from tests.access_fixtures import MATRIX
from tests.gateway_fixtures import Stack, build

PASSWORD = "Correct-horse-9"
HDR = {"X-AskIISc": "1"}
EXTRA_PEOPLE = [
    {"user_id": "u_cds_admin", "department": "cds", "role": "admin"},
    {
        "user_id": "real.person@iisc.example",
        "department": "cds",
        "role": "student",
    },
]
_HASH = hash_password(PASSWORD)  # scrypt is slow: hash once per session


@dataclass
class Web:
    stack: Stack
    deps: WebDeps
    client: httpx.AsyncClient
    access_http: httpx.AsyncClient

    def browser(self) -> httpx.AsyncClient:
        """Another browser (own cookie jar) on the same app."""
        return httpx.AsyncClient(
            transport=self.client._transport,  # noqa: SLF001
            base_url="http://testserver",
        )

    def account(self, user: str, status: str = "active", **kw: Any) -> None:
        """Creates an account directly in the store."""
        self.deps.store.create_account(
            user, _HASH, status, self.stack.w.clock.now(), **kw
        )

    async def institute_admin(self, root: Path) -> None:
        """Gives the app an editable users tree and logs u_admin in."""
        for name, rows in {
            "institute/admins.json": [{"user_id": "u_admin"}],
            "cds/admins.json": [{"user_id": "u_cds_admin"}],
        }.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(json.dumps(rows))
        self.deps.directory.load(root)
        self.deps.people = PeopleTree(root, self.deps.directory)
        self.account("u_admin")
        assert (await self.login("u_admin")).status_code == 200

    async def login(
        self, user: str, client: httpx.AsyncClient | None = None
    ) -> httpx.Response:
        """Logs a browser in."""
        return await (client or self.client).post(
            "/api/auth/login",
            json={"user_id": user, "password": PASSWORD},
            headers=HDR,
        )


def sse(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in response.text.split("\n\n"):
        if block.strip():
            lines = dict(ln.split(": ", 1) for ln in block.splitlines())
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def build_web(tmp_path: Path, **stack_args: Any) -> Web:
    stack = build(**stack_args)
    w = stack.w
    directory = Directory.from_records([*MATRIX["directory"], *EXTRA_PEOPLE])
    stack.deps.auth = Authenticator(w.issuer.verifier(w.clock), directory)
    requests = RequestLog(
        tmp_path / "requests.db",
        role_of=lambda u: p.role if (p := directory.get(u)) else "",
        now=w.clock.now(),
    )
    stack.deps.on_request = requests.record
    access_app = create_access(
        directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )
    access_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=access_app), base_url="http://access"
    )
    revisions: list[tuple[str, str]] = []

    async def revise(doc_id: str, text: str) -> dict[str, Any]:
        revisions.append((doc_id, text))
        return {"doc_id": doc_id, "version": 2, "chunks": 1}

    async def kill() -> str:
        return "killed vllm-1"

    deps = WebDeps(
        gateway=stack.deps,
        store=AppStore(tmp_path / "app.db", load_key(tmp_path / "chat.key")),
        requests=requests,
        directory=directory,
        tokens=InternalTokens(w.issuer, w.clock),
        clock=w.clock,
        docs=lambda d: {"title": f"Doc {d}", "url": f"https://x/{d}"},
        access_http=access_http,
        system=lambda: {"model": "fake-model"},
        revise=revise,
        chaos={"kill_worker": kill},
        loadtests=lambda: [{"name": "run1"}],
        email_domains=("iisc.example",),
    )
    app = create_app(stack.deps)
    mount_web(app, deps)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    web = Web(stack, deps, client, access_http)
    web.revisions = revisions  # type: ignore[attr-defined]
    return web
