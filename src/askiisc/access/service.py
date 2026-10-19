"""The standalone access-control service (FR-4).

The caller passes the user's verified SSO token; the user id comes only
from that token, never from the request body. Policy changes require an
admin of the document's owning department, or an institute admin.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from askiisc.access.directory import Directory
from askiisc.access.policy import (
    INSTITUTE,
    DocPolicy,
    PolicyEngine,
    PolicyStore,
    may_manage,
)
from askiisc.common.clock import Clock
from askiisc.common.errors import AuthError
from askiisc.common.tokens import TokenVerifier
from askiisc.common.types import DocId, UserId


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

    owner: str = INSTITUTE
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

    def require_manager(user: UserId, owners: set[str]) -> None:
        person = directory.get(user)
        if not all(may_manage(person, owner) for owner in owners):
            raise HTTPException(
                status_code=403, detail="admin of the owning department only"
            )

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

    def current_owner(doc_id: DocId) -> set[str]:
        existing = policies.snapshot()[1].get(doc_id)
        return set() if existing is None else {existing.owner}

    @app.put("/v1/policy/{doc_id}")
    async def put_policy(
        doc_id: DocId,
        body: PolicyBody,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, int]:
        # Both the current and the new owner's admin rights are needed, so
        # nobody can take over another department's document.
        require_manager(user, current_owner(doc_id) | {body.owner})
        policy = DocPolicy(doc_id=doc_id, **body.model_dump())
        return {"policy_version": policies.put(policy)}

    @app.delete("/v1/policy/{doc_id}")
    async def revoke(
        doc_id: DocId,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, int]:
        require_manager(user, current_owner(doc_id) or {INSTITUTE})
        return {"policy_version": policies.delete(doc_id)}

    return app
