"""Single-use links that set a password: resets.

People create their own account (web/registration.py). A forgotten
password gets a reset link from "Forgot password?" on the sign-in page
(valid RESET_TTL_S); while e-mail is not working, the terminal prints one
instead (`askiisc.web.cli password-link`, valid TERMINAL_LINK_TTL_S).
Opening the link lets the person choose a new password.

- Tokens are 32 random bytes; only their SHA-256 is stored, a new link
  voids the person's earlier ones, and using one ends every session.
- Links are e-mailed to the address on the person's own record, never to
  an address typed into a form, and are never logged.
- "Forgot password?" says at once when an address has no account (chosen
  for clarity, 2026-10-27); such addresses count as failures in the
  sign-in throttle. At most MAX_LINKS_PER_HOUR links an hour.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from askiisc.common.errors import AuthError
from askiisc.web.app import (
    WebDeps,
    error,
    find_person,
    is_real,
    may_sign_in,
    normalise,
    run,
)
from askiisc.web.passwords import (
    MAX_PASSWORD_CHARS,
    check_strength,
    hash_password,
)

_LOG = logging.getLogger(__name__)
_HOUR_S = 3600.0
TERMINAL_LINK_TTL_S = _HOUR_S
RESET_TTL_S = 10 * 60.0
MAX_LINKS_PER_HOUR = 3
NOT_SET_UP = "E-mail is not set up yet; ask your department admin for a link."
NO_ACCOUNT = (
    "This e-mail has no Ask IISc account. Check it, or create an account."
)
SEND_FAILED = "The e-mail could not be sent; try again in a few minutes."
BAD_LINK = "this link is invalid, used or expired; ask for a new one"


class Forgot(BaseModel):
    """POST /api/auth/forgot body."""

    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=200)


class SetPassword(BaseModel):
    """POST /api/auth/set-password body."""

    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=20, max_length=100)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_CHARS)


def link_path(token: str) -> str:
    """Where a link opens (the token is in the fragment: never sent)."""
    return f"/#set-password={token}"


def _lasts(ttl_s: float) -> str:
    if ttl_s >= _HOUR_S:
        hours = round(ttl_s / _HOUR_S)
        return "1 hour" if hours == 1 else f"{hours} hours"
    return f"{round(ttl_s / 60)} minutes"


def _message(url: str, ttl_s: float) -> tuple[str, str]:
    return "Ask IISc: set a new password", (
        "Someone asked to reset the password of your Ask IISc account.\n\n"
        f"Choose a new password here (works once, for {_lasts(ttl_s)}):\n"
        f"{url}\n\n"
        "If this was not you, ignore this e-mail; your password is "
        "unchanged.\n"
    )


async def send_link(
    web: WebDeps, user_id: str, token: str, ttl_s: float
) -> bool:
    """E-mails a link to the user's own address; False if not sent."""
    if web.mailer is None:
        return False
    url = web.mailer.public_url + link_path(token)
    return await deliver(web, user_id, *_message(url, ttl_s))


async def deliver(web: WebDeps, user_id: str, subject: str, body: str) -> bool:
    """E-mails a real person; False if not sent."""
    mailer = web.mailer
    if mailer is None or not is_real(user_id):
        return False
    try:
        await asyncio.to_thread(mailer.send, user_id, subject, body)
    except OSError as err:
        _LOG.warning("e-mail failed: %s", type(err).__name__)
        web.m.logins.labels("mail_failed").inc()
        return False
    return True


def throttled(web: WebDeps, key: str) -> Response | None:
    """A 429 answer while this key must wait, else None."""
    wait = web.throttle.wait_s(key) if web.throttle is not None else 0.0
    if wait <= 0:
        return None
    return JSONResponse(
        {"error": "too many attempts; try again later"},
        status_code=429,
        headers={"Retry-After": str(int(wait) + 1)},
    )


async def _forgot(web: WebDeps, typed: str) -> Response:
    """E-mails a reset link to an account that may sign in."""
    if web.mailer is None:
        return error(503, NOT_SET_UP)
    person = find_person(web.directory, typed)
    user_id = person.user_id if person else normalise(typed)
    key = f"forgot:{user_id}"
    if wait := throttled(web, key):
        return wait
    account = await run(web.store.account, user_id)
    if (
        person is None
        or account is None
        or not account.has_password
        or account.status != "active"
        or not is_real(user_id)
        or not await run(may_sign_in, web, user_id)
    ):
        if web.throttle is not None:
            web.throttle.failed(key)
        return error(404, NO_ACCOUNT)
    now = web.clock.now()
    if (
        await run(web.store.actions_since, user_id, "reset_sent", now - _HOUR_S)
        >= MAX_LINKS_PER_HOUR
    ):
        return error(429, "Too many reset e-mails; try again in an hour.")
    token = await run(web.store.create_link, user_id, now, RESET_TTL_S)
    if not await send_link(web, user_id, token, RESET_TTL_S):
        return error(502, SEND_FAILED)
    await run(web.store.log_action, user_id, "reset_sent", user_id, now)
    return JSONResponse(
        {
            "message": f"We sent a reset link to {user_id}. It works once, "
            "for 10 minutes."
        }
    )


def link_routes(
    web: WebDeps, router: APIRouter, csrf: Callable[..., None]
) -> None:
    """Adds /api/auth/forgot and /api/auth/set-password."""

    @router.post("/auth/forgot", dependencies=[Depends(csrf)])
    async def forgot(body: Forgot) -> Response:
        return await _forgot(web, body.user_id)

    @router.post("/auth/set-password", dependencies=[Depends(csrf)])
    async def set_password(body: SetPassword) -> Response:
        if wait := throttled(web, "set-password"):
            return wait
        user_id = await run(web.store.link_user, body.token, web.clock.now())
        if user_id is None:
            if web.throttle is not None:
                web.throttle.failed("set-password")
            return error(400, BAD_LINK)
        try:
            check_strength(body.password, user_id)
        except AuthError as err:
            return error(400, str(err))
        hashed = await run(hash_password, body.password)
        await run(web.store.use_link, user_id, hashed)
        await run(
            web.store.log_action,
            user_id,
            "password_set",
            user_id,
            web.clock.now(),
        )
        return JSONResponse({"ok": True, "user_id": user_id})
