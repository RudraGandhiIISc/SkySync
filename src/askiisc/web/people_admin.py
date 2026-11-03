"""People administration: the admin dashboard's People tab.

Who may change whom:
- institute admins (created only from the terminal, web/cli.py) manage
  every department's people, including department admins;
- department admins (several per department) manage the students,
  faculty and staff of their own department, but never an admin;
- nobody changes the institute folder or their own entry here;
- only real people, named by their e-mail address, are added or changed
  here. Locust's test users (ids without "@") get their synthetic accounts
  only from the terminal (`python -m askiisc.web.cli seed-locust`).
Every admin may view everyone. Each change is in the admin action log.

Adding a person also puts their e-mail address on the sign-in allowlist
and gives them an account without a password, in a database transaction
that commits only after the users tree is written: one action authorises a
person, and a failed tree write leaves nothing behind. Nothing is sent:
the person creates their own account on the sign-in page, confirmed by a
link e-mailed to that address (web/registration.py). Only addresses of
the allowed e-mail domains (ASKIISC_EMAIL_DOMAINS) can be added; people reset
a forgotten password themselves (web/password_links.py). Revoking access
("suspend") signs them out and keeps the person, their account and
their chats; removing a person also deletes the account and chats. A
change of role or department ends the person's sessions, so new rights
apply from a fresh sign-in.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from askiisc.access.directory import DEPARTMENT_PROGRAMS, KINDS, PROGRAMS
from askiisc.access.policy import INSTITUTE
from askiisc.common.errors import ConfigError, NotFound
from askiisc.common.types import UserId
from askiisc.web.app import Me, WebDeps, error, run
from askiisc.web.people import STEM_OF, Entry, PeopleTree, make_entry, parse_csv
from askiisc.web.store import Access

MAX_CSV_CHARS = 2_000_000
ACCESS_ACTIONS: dict[str, Access] = {"revoke": "revoked", "restore": "active"}
NOT_REAL = (
    "only real people can be added here: use their e-mail address "
    "(Locust test users come from `askiisc.web.cli seed-locust`)"
)


class PersonForm(BaseModel):
    """Add or edit one person."""

    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=200)
    department: str = Field(min_length=1, max_length=40)
    role: str = Field(min_length=1, max_length=20)
    program: str = Field(default="", max_length=40)
    groups: list[str] = Field(default_factory=list, max_length=50)
    kind: str = Field(default="", max_length=40)
    joined: int | None = None


class BulkForm(BaseModel):
    """Bulk add from CSV; `apply` false only checks and previews."""

    model_config = ConfigDict(extra="forbid")
    csv: str = Field(min_length=1, max_length=MAX_CSV_CHARS)
    apply: bool = False


def real_person(user_id: str) -> bool:
    """Whether an id names a real person (an e-mail address)."""
    return "@" in user_id


def domain_error(user_id: str, domains: tuple[str, ...]) -> str | None:
    """Why an address may not be added (its domain), or None."""
    if user_id.rsplit("@", 1)[-1] in domains:
        return None
    allowed = ", ".join(f"@{d}" for d in domains)
    return f"{user_id}: only {allowed} addresses can be added"


def may_edit(me: Me, department: str, role: str) -> bool:
    """Whether this admin may add, change or remove such a person."""
    if not me.is_admin or department == INSTITUTE:
        return False
    if role == "admin":
        return me.is_institute_admin
    return me.is_institute_admin or department == me.person.department


def people_routes(  # noqa: PLR0915 - one route table
    web: WebDeps,
    router: APIRouter,
    user_dep: Any,
    log: Callable[[Me, str, str], Awaitable[None]],
) -> None:
    """Adds /api/admin/people routes to the admin router."""

    def tree() -> PeopleTree | None:
        return web.people

    @router.get("/people")
    async def people(me: Me = user_dep) -> Any:
        me.scope()
        t = tree()
        if t is None:
            return error(503, "the users tree is not configured")
        entries = await run(t.entries)
        accounts = {a.user_id: a for a in await run(web.store.accounts)}
        allowed = await run(web.store.allowed_users)
        return {
            "people": [
                _row(me, e, accounts.get(e.user_id), allowed.get(e.user_id))
                for e in entries
            ],
            "programs": PROGRAMS,
            "department_programs": DEPARTMENT_PROGRAMS,
            "kinds": KINDS,
            "roles": sorted(STEM_OF),
            "departments": sorted(
                {e.department for e in entries} - {INSTITUTE}
            ),
            "manages": "all"
            if me.is_institute_admin
            else [me.person.department],
            "institute_admin": me.is_institute_admin,
        }

    @router.post("/people", status_code=201)
    async def add(body: PersonForm, me: Me = user_dep) -> Any:
        me.scope()
        return await _change(web, me, log, None, body)

    @router.put("/people/{user_id}")
    async def edit(user_id: str, body: PersonForm, me: Me = user_dep) -> Any:
        me.scope()
        return await _change(web, me, log, user_id, body)

    @router.delete("/people/{user_id}")
    async def remove(user_id: str, me: Me = user_dep) -> Any:
        me.scope()
        t = tree()
        if t is None:
            return error(503, "the users tree is not configured")
        old = await run(t.find, user_id)
        if old is None:
            return error(404, "no such person")
        if (
            user_id == me.user_id
            or not real_person(user_id)
            or not may_edit(me, old.department, old.role)
        ):
            return error(403, "you may not remove this person")
        try:
            await run(t.remove, user_id)
        except ConfigError as err:
            return error(422, str(err))
        now = web.clock.now()
        await run(web.store.set_access, user_id, "revoked", me.user_id, now)
        await run(web.store.delete_account, user_id)
        await log(me, "person_remove", user_id)
        return {"removed": user_id}

    @router.post("/people/{user_id}/{action}")
    async def access(user_id: str, action: str, me: Me = user_dep) -> Any:
        me.scope()
        status: Access | None = ACCESS_ACTIONS.get(action)
        if status is None:
            return error(404, f"unknown action {action}")
        person = web.directory.get(UserId(user_id))
        if person is None or not real_person(user_id):
            return error(404, "no such person")
        if user_id == me.user_id or not may_edit(
            me, person.department, person.role
        ):
            return error(403, "you may not change this person's access")
        now = web.clock.now()
        if status == "active" and await run(web.store.allowed, user_id) is None:
            await run(web.store.allow_existing, [user_id], me.user_id, now)
        else:
            await run(web.store.set_access, user_id, status, me.user_id, now)
        await log(me, f"access_{action}", user_id)
        return {"user_id": user_id, "access": status}

    @router.post("/people/bulk")
    async def bulk(body: BulkForm, me: Me = user_dep) -> Any:
        me.scope()
        t = tree()
        if t is None:
            return error(503, "the users tree is not configured")
        default = "" if me.is_institute_admin else me.person.department
        try:
            entries = parse_csv(body.csv, default)
        except ConfigError as err:
            return error(422, str(err))
        for n, entry in enumerate(entries, start=1):
            if not real_person(entry.user_id):
                return error(422, f"row {n}: {NOT_REAL}")
            if why := domain_error(entry.user_id, web.email_domains):
                return error(422, f"row {n}: {why}")
            if not may_edit(me, entry.department, entry.role):
                return error(403, f"row {n}: you may not add {entry.user_id}")
        if body.apply:
            try:
                await run(_add_allowed, web, t, entries, me.user_id)
            except ConfigError as err:
                return error(422, str(err))
            await log(me, "person_bulk_add", f"{len(entries)} people")
        return {
            "applied": body.apply,
            "count": len(entries),
            "people": [
                e.__dict__ | {"groups": list(e.groups)} for e in entries
            ],
        }


async def _change(
    web: WebDeps,
    me: Me,
    log: Callable[[Me, str, str], Awaitable[None]],
    user_id: str | None,
    body: PersonForm,
) -> Any:
    """Adds (user_id None) or replaces one person."""
    t = web.people
    if t is None:
        return error(503, "the users tree is not configured")
    try:
        entry = make_entry(
            body.user_id,
            body.department,
            body.role,
            body.program,
            body.groups,
            body.kind,
            body.joined,
        )
    except ConfigError as err:
        return error(422, str(err))
    if not real_person(entry.user_id):
        return error(422, NOT_REAL)
    old = None if user_id is None else await run(t.find, user_id)
    if user_id is not None and old is None:
        return error(404, "no such person")
    if old is None and (why := domain_error(entry.user_id, web.email_domains)):
        return error(422, why)
    if old is not None and entry.user_id != old.user_id:
        return error(422, "the user id cannot change; remove and add instead")
    allowed = may_edit(me, entry.department, entry.role) and (
        old is None or may_edit(me, old.department, old.role)
    )
    if not allowed or entry.user_id == me.user_id:
        return error(403, "you may not make this change")
    try:
        if old is None:
            await run(_add_allowed, web, t, [entry], me.user_id)
        else:
            await run(t.replace, old.user_id, entry)
    except (ConfigError, NotFound) as err:
        return error(422, str(err))
    await log(me, "person_add" if old is None else "person_edit", entry.user_id)
    out: dict[str, Any] = {
        "person": entry.__dict__ | {"groups": list(entry.groups)}
    }
    if old is not None and (old.department, old.role) != (
        entry.department,
        entry.role,
    ):
        await run(web.store.end_sessions, entry.user_id)  # rights changed
    return out


def _add_allowed(
    web: WebDeps, tree: PeopleTree, entries: list[Entry], by: str
) -> None:
    """Adds people to the tree and the allowlist, or to neither.

    Raises:
        ConfigError: If the tree rejects them (nothing is allowed then).
    """
    emails = [e.user_id for e in entries]
    with web.store.allowing(emails, by, web.clock.now()):
        tree.add(entries)


def _access(entry: Entry, account: Any, allowed: Any) -> str:
    """How a person signs in now, for the People tab."""
    if not real_person(entry.user_id):
        return "test account"
    if allowed is None:
        return "not allowed"
    if allowed.status != "active":
        return "revoked"
    if account is None or not account.has_password:
        return "not registered"
    return "active"


def _row(
    me: Me, entry: Entry, account: Any, allowed: Any = None
) -> dict[str, Any]:
    return {
        "user_id": entry.user_id,
        "department": entry.department,
        "role": entry.role,
        "program": entry.program,
        "kind": entry.kind,
        "joined": entry.joined,
        "groups": list(entry.groups),
        "account": "not registered" if account is None else account.status,
        "synthetic": not real_person(entry.user_id),  # a test identity
        "registered": None if account is None else account.created,
        "access": _access(entry, account, allowed),
        "last_login": None if account is None else account.last_login,
        "can_manage": entry.user_id != me.user_id
        and real_person(entry.user_id)
        and may_edit(me, entry.department, entry.role),
    }
