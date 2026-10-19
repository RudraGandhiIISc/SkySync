"""The standalone access-control service (FR-4).

The caller passes the user's verified SSO token; the user id comes only
from that token, never from the request body. Policy changes require a
caller whose directory role is "admin".
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from askiisc.access.directory import Directory
from askiisc.access.policy import DocPolicy, PolicyEngine, PolicyStore
from askiisc.common.clock import Clock
from askiisc.common.errors import AuthError
from askiisc.common.tokens import TokenVerifier
from askiisc.common.types import DocId, UserId

ADMIN_ROLE = "admin"


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthorizeRequest(_Body):
    """POST /v1/authorize body."""

    doc_id: DocId
    at: float | None = None


class FilterRequest(_Body):
    """POST /v1/filter body."""

    at: float | None = None


class PolicyBody(_Body):
    """PUT /v1/policy/{doc_id} body (DocPolicy without the id)."""

    visibility: Literal["public", "restricted"] = "restricted"
    groups: frozenset[str] = frozenset()
    departments: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    valid_from: float | None = None
    valid_until: float | None = None
    allow_users: frozenset[UserId] = frozenset()
    deny_users: frozenset[UserId] = frozenset()


def create_app(
    directory: Directory,
    policies: PolicyStore,
    verifier: TokenVerifier,
    clock: Clock,
) -> FastAPI:
    """Builds the access service.

    Args:
        directory: The user directory.
        policies: The policy store.
        verifier: SSO token verifier.
        clock: Time source for decisions.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="ask-iisc-access")
    engine = PolicyEngine(directory, policies)

    def caller(authorization: str = Header(default="")) -> UserId:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing token")
        try:
            return verifier.verify(token)
        except AuthError as err:
            raise HTTPException(status_code=401, detail="bad token") from err

    def admin(user: UserId = Depends(caller)) -> UserId:  # noqa: B008
        person = directory.get(user)
        if person is None or person.role != ADMIN_ROLE:
            raise HTTPException(status_code=403, detail="admin only")
        return user

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/authorize")
    async def authorize(
        body: AuthorizeRequest,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, object]:
        started = time.perf_counter()
        at = clock.now() if body.at is None else body.at
        decision = engine.authorize(user, body.doc_id, at)
        return {
            "allowed": decision.allowed,
            "policy_version": decision.policy_version,
            "elapsed_ms": (time.perf_counter() - started) * 1e3,
        }

    @app.post("/v1/filter")
    async def filter_docs(
        body: FilterRequest,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, object]:
        started = time.perf_counter()
        at = clock.now() if body.at is None else body.at
        result = engine.filter(user, at)
        return {
            "doc_ids": sorted(result.doc_ids),
            "policy_version": result.policy_version,
            "elapsed_ms": (time.perf_counter() - started) * 1e3,
        }

    @app.put("/v1/policy/{doc_id}")
    async def put_policy(
        doc_id: DocId,
        body: PolicyBody,
        _: UserId = Depends(admin),  # noqa: B008
    ) -> dict[str, int]:
        policy = DocPolicy(doc_id=doc_id, **body.model_dump())
        return {"policy_version": policies.put(policy)}

    @app.delete("/v1/policy/{doc_id}")
    async def revoke(
        doc_id: DocId,
        _: UserId = Depends(admin),  # noqa: B008
    ) -> dict[str, int]:
        return {"policy_version": policies.delete(doc_id)}

    return app
