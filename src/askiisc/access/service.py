"""The standalone access-control service (FR-4).

The caller passes the user's verified SSO token; the user id comes only
from that token, never from the request body. Policy changes require an
admin of the document's owning department, or an institute admin.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict

from askiisc.access.directory import Directory
from askiisc.access.policy import (
    INSTITUTE,
    DocPolicy,
    PolicyEngine,
    PolicyStore,
    may_change_owners,
    may_manage_policy,
)
from askiisc.common.clock import Clock
from askiisc.common.errors import AuthError
from askiisc.common.requestid import log_requests
from askiisc.common.tokens import TokenVerifier
from askiisc.common.types import DocId, UserId


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthorizeRequest(_Body):
    """POST /v1/authorize body."""

    doc_id: DocId
    at: float | None = None


class FilterRequest(_Body):
    """POST /v1/filter body.

    With `compact`, the plain public documents are not listed: the reply
    says `public: true` and lists only the others (`GET /v1/public` lists
    the plain public set for the same policy version).
    """

    at: float | None = None
    compact: bool = False


class PolicyBody(_Body):
    """PUT /v1/policy/{doc_id} body (DocPolicy without the id)."""

    owner: str = INSTITUTE
    co_owners: frozenset[str] = frozenset()
    visibility: Literal["public", "restricted"] = "restricted"
    groups: frozenset[str] = frozenset()
    departments: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    valid_from: float | None = None
    valid_until: float | None = None
    allow_users: frozenset[UserId] = frozenset()
    deny_users: frozenset[UserId] = frozenset()


REQUEST_LOG = logging.getLogger("askiisc.access.requests")
ACCESS_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5)


class AccessMetrics:
    """Access service metrics (no user ids, no document ids)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.seconds = Histogram(
            "askiisc_access_seconds",
            "Time to decide, by call.",
            ["call"],
            buckets=ACCESS_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "askiisc_access_decisions",
            "Authorize decisions.",
            ["allowed"],
            registry=self.registry,
        )
        self.policy_changes = Counter(
            "askiisc_access_policy_changes",
            "Policy puts and revocations.",
            ["kind"],
            registry=self.registry,
        )


PolicyListener = Callable[[str, DocPolicy | DocId], None]
"""Called after a change: ("put", policy) or ("delete", doc_id)."""


def create_app(  # noqa: PLR0915 - one route table, read top to bottom
    directory: Directory,
    policies: PolicyStore,
    verifier: TokenVerifier,
    clock: Clock,
    on_change: PolicyListener | None = None,
) -> FastAPI:
    """Builds the access service.

    Args:
        directory: The user directory.
        policies: The policy store.
        verifier: SSO token verifier.
        clock: Time source for decisions.
        on_change: Told about every accepted policy change (persistence).

    Returns:
        A FastAPI application.
    """
    app = FastAPI(title="ask-iisc-access")
    log_requests(app, "access", REQUEST_LOG)
    engine = PolicyEngine(directory, policies)
    metrics = AccessMetrics()

    def caller(authorization: str = Header(default="")) -> UserId:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing token")
        try:
            return verifier.verify(token)
        except AuthError as err:
            raise HTTPException(status_code=401, detail="bad token") from err

    def require(allowed: bool) -> None:
        if not allowed:
            raise HTTPException(
                status_code=403, detail="admin of an owning department only"
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(
            generate_latest(metrics.registry), media_type="text/plain"
        )

    @app.post("/v1/authorize")
    async def authorize(
        body: AuthorizeRequest,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, object]:
        started = time.perf_counter()
        at = clock.now() if body.at is None else body.at
        decision = engine.authorize(user, body.doc_id, at)
        elapsed = time.perf_counter() - started
        metrics.seconds.labels("authorize").observe(elapsed)
        metrics.decisions.labels(str(decision.allowed).lower()).inc()
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
        if body.compact:
            compact = engine.filter_compact(user, at)
            reply: dict[str, object] = {
                "public": compact.public,
                "doc_ids": sorted(compact.extra),
                "policy_version": compact.policy_version,
            }
        else:
            result = engine.filter(user, at)
            reply = {
                "doc_ids": sorted(result.doc_ids),
                "policy_version": result.policy_version,
            }
        elapsed = time.perf_counter() - started
        metrics.seconds.labels("filter").observe(elapsed)
        return {**reply, "elapsed_ms": elapsed * 1e3}

    @app.get("/v1/public")
    async def public_docs(
        _: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, object]:
        result = engine.public()
        return {
            "doc_ids": sorted(result.doc_ids),
            "policy_version": result.policy_version,
        }

    @app.get("/v1/policy/{doc_id}")
    async def get_policy(
        doc_id: DocId,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, object]:
        existing = policies.snapshot()[1].get(doc_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="unknown document")
        require(may_manage_policy(directory.get(user), existing.owners))
        return existing.model_dump(mode="json")

    def current_owners(doc_id: DocId) -> frozenset[str] | None:
        existing = policies.snapshot()[1].get(doc_id)
        return None if existing is None else existing.owners

    @app.put("/v1/policy/{doc_id}")
    async def put_policy(
        doc_id: DocId,
        body: PolicyBody,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, int]:
        # Owners may be added, but only departments the admin manages may
        # be removed, so nobody takes over another department's document.
        policy = DocPolicy(doc_id=doc_id, **body.model_dump())
        require(
            may_change_owners(
                directory.get(user), current_owners(doc_id), policy.owners
            )
        )
        version = policies.put(policy)
        metrics.policy_changes.labels("put").inc()
        if on_change is not None:
            on_change("put", policy)
        return {"policy_version": version}

    @app.delete("/v1/policy/{doc_id}")
    async def revoke(
        doc_id: DocId,
        user: UserId = Depends(caller),  # noqa: B008
    ) -> dict[str, int]:
        owners = current_owners(doc_id) or frozenset({INSTITUTE})
        require(may_manage_policy(directory.get(user), owners))
        version = policies.delete(doc_id)
        metrics.policy_changes.labels("delete").inc()
        if on_change is not None:
            on_change("delete", doc_id)
        return {"policy_version": version}

    return app
