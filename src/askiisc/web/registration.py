"""Creating an account: an allowlisted person registers themselves.

1. POST /api/auth/register {user_id, password, confirm}: the two
   passwords must match and be strong enough; then the address must be on
   the allowlist (active), in the directory and without a password yet.
   A single-use confirmation link (valid CONFIRM_TTL_S) is e-mailed to it.
2. POST /api/auth/register/confirm {token}, from the page the link opens:
   sets the chosen password; the person then signs in.

The link proves the person reads that mailbox, so nobody can claim a
colleague's allowlisted address. Unlike sign-in, these answers say
whether an address may register (chosen for clarity, 2026-10-27); refused
addresses count as failures in the sign-in throttle, which slows anyone
trying many.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from askiisc.common.errors import AuthError, ConfigError
from askiisc.web.app import (
    WebDeps,
    error,
    find_person,
    is_real,
    may_sign_in,
    normalise,
    run,
)
from askiisc.web.password_links import (
    NOT_SET_UP,
    SEND_FAILED,
    deliver,
    throttled,
)
from askiisc.web.passwords import (
    MAX_PASSWORD_CHARS,
    check_strength,
    hash_password,
)
from askiisc.web.pending import PendingRegistrations
from askiisc.web.store import AccountExists

MISMATCH = "The two passwords do not match."
NOT_ADDED = (
    "You are not added as an IISc user. Ask your department admin to add you."
)
HAS_ACCOUNT = (
    "This e-mail already has an account. Sign in, or use Forgot password."
)
BAD_CONFIRM = (
    "This confirmation link is invalid, used or expired; create your "
    "account again."
)
TOO_MANY = (
    "Too many confirmation e-mails for this address; try again in an hour."
)


class Register(BaseModel):
    """POST /api/auth/register body."""

    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)
    confirm: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)


class ConfirmRegistration(BaseModel):
    """POST /api/auth/register/confirm body."""

    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=20, max_length=100)


def confirm_path(token: str) -> str:
    """Where a confirmation link opens (the token is in the fragment)."""
    return f"/#confirm={token}"


def register_routes(
    web: WebDeps, router: APIRouter, csrf: Callable[..., None]
) -> None:
    """Adds /api/auth/register and /api/auth/register/confirm."""

    @router.post("/auth/register", dependencies=[Depends(csrf)])
    async def register(body: Register) -> Response:
        return await _register(web, body)

    @router.post("/auth/register/confirm", dependencies=[Depends(csrf)])
    async def confirm(body: ConfirmRegistration) -> Response:
        return await _confirm(web, body.token)


async def _register(web: WebDeps, body: Register) -> Response:
    """Checks a registration and e-mails its confirmation link."""
    if body.password != body.confirm:
        return error(400, MISMATCH)
    person = find_person(web.directory, body.user_id)
    user_id = person.user_id if person else normalise(body.user_id)
    try:
        check_strength(body.password, user_id)
    except AuthError as err:
        return error(400, str(err))
    key = f"register:{user_id}"
    if wait := throttled(web, key):
        return wait
    if not await _may_register(web, user_id, person is not None):
        _failed(web, key)
        return error(403, NOT_ADDED)
    account = await run(web.store.account, user_id)
    if account is not None and account.has_password:
        return error(409, HAS_ACCOUNT)
    if web.mailer is None:
        return error(503, NOT_SET_UP)
    hashed = await run(hash_password, body.password)
    token = _pending(web).add(user_id, hashed)
    if token is None:
        return error(429, TOO_MANY)
    url = web.mailer.public_url + confirm_path(token)
    if not await deliver(web, user_id, *_message(url)):
        return error(502, SEND_FAILED)
    return JSONResponse(
        {
            "message": f"We sent a link to {user_id}. Open it within 10 "
            "minutes to finish creating your account."
        },
        status_code=202,
    )


async def _confirm(web: WebDeps, token: str) -> Response:
    """Creates the account of a confirmed registration."""
    pending = _pending(web).take(token)
    if pending is None:
        return error(400, BAD_CONFIRM)
    user_id = pending.user_id
    person = find_person(web.directory, user_id)
    if not await _may_register(web, user_id, person is not None):
        return error(403, NOT_ADDED)
    account = await run(web.store.account, user_id)
    if account is not None and account.has_password:
        return error(409, HAS_ACCOUNT)
    now = web.clock.now()
    try:
        if account is None:
            await run(
                web.store.create_account,
                user_id,
                pending.password_hash,
                "active",
                now,
            )
        else:
            await run(web.store.use_link, user_id, pending.password_hash)
    except AccountExists:
        return error(409, HAS_ACCOUNT)
    await run(web.store.log_action, user_id, "registered", user_id, now)
    return JSONResponse({"ok": True, "user_id": user_id})


async def _may_register(web: WebDeps, user_id: str, in_directory: bool) -> bool:
    """A real person in the directory, allowed and not disabled."""
    if not (in_directory and is_real(user_id)):
        return False
    if not await run(may_sign_in, web, user_id):
        return False
    account = await run(web.store.account, user_id)
    return account is None or account.status == "active"


def _pending(web: WebDeps) -> PendingRegistrations:
    if web.registrations is None:  # pragma: no cover - set in __post_init__
        raise ConfigError("no pending registrations")
    return web.registrations


def _failed(web: WebDeps, key: str) -> None:
    if web.throttle is not None:
        web.throttle.failed(key)


def _message(url: str) -> tuple[str, str]:
    return "Ask IISc: confirm your account", (
        "Someone (we hope you) created an Ask IISc account with this "
        "address.\n\n"
        f"Open this link to finish (works once, for 10 minutes):\n{url}\n\n"
        "If this was not you, ignore this e-mail; no account is created.\n"
    )
