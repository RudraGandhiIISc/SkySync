"""Admin routes: usage, the request log, accounts, policies, load tests.

Every admin sees every department's users and requests; what they may
change is narrower (people_admin.may_edit for people, the
access service for document policies). Admins never see question, answer or
chat text: the request log has none, and no route reads another user's
chats. Viewing the request log, exporting it, and every change are
recorded in the admin action log.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from askiisc.access.policy import (
    INSTITUTE,
    DocPolicy,
    PolicyEngine,
    PolicyStore,
    may_manage,
)
from askiisc.common.types import DocId, UserId
from askiisc.web.app import Me, WebDeps, error, run
from askiisc.web.documents import document_routes, unknown_owners
from askiisc.web.models import comparison
from askiisc.web.models import labels as model_labels
from askiisc.web.people_admin import people_routes, real_person
from askiisc.web.requestlog import COLUMNS
from askiisc.web.scorecard import scorecard

MAX_WINDOW_S = 30 * 86400.0
# Every document id there is (crawled ids run to ~110 characters); anything
# else, such as "..", never reaches the access service URL.
DOC_REF = re.compile(r"[a-z0-9][a-z0-9._-]{0,199}")
MAX_CSV_ROWS = 100_000
MAX_PAGE_ROWS = 1000
ORIGINS = {"real": False, "locust": True}


class Revision(BaseModel):
    """POST /api/admin/documents/{doc_id}/revise body."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2_000_000)


def _window(seconds: float) -> float:
    return max(60.0, min(seconds, MAX_WINDOW_S))


def admin_router(  # noqa: PLR0915 - one route table
    web: WebDeps, current: Callable[[Request], Awaitable[Me]]
) -> APIRouter:
    """The /api/admin routes."""
    router = APIRouter(prefix="/api/admin")
    user_dep = Depends(current)

    def log(me: Me, action: str, target: str = "") -> Awaitable[None]:
        return run(
            web.store.log_action, me.user_id, action, target, web.clock.now()
        )

    @router.get("/overview")
    async def overview(
        window_s: float = 3600, origin: str = "", me: Me = user_dep
    ) -> Any:
        scope = me.scope()
        now = web.clock.now()
        since = now - _window(window_s)
        synthetic = _origin(origin)
        summary = await run(
            lambda: web.requests.summary(scope, since, now, **synthetic)
        )
        return {
            "window_s": _window(window_s),
            "scope": "all" if scope is None else sorted(scope),
            "users": await run(_user_counts, web, scope, synthetic),
            "requests": summary,
            "ratings": await run(
                web.store.ratings, _ids(web, scope, synthetic)
            ),
            "system": web.system(),
        }

    @router.get("/series")
    async def series(
        window_s: float = 3600,
        step_s: float = 60,
        origin: str = "",
        me: Me = user_dep,
    ) -> Any:
        scope = me.scope()
        window = _window(window_s)
        step = max(5.0, min(step_s, window / 10))
        since = web.clock.now() - window
        synthetic = _origin(origin)
        return {
            "step_s": step,
            "points": await run(
                lambda: web.requests.series(scope, since, step, **synthetic)
            ),
        }

    @router.get("/requests")
    async def requests(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        me: Me = user_dep,
    ) -> Any:
        scope = me.scope()
        filters = _filters(request)
        await log(me, "view_requests", _describe(filters))
        size, start = _page(limit, offset)
        rows = await run(
            lambda: web.requests.rows(
                scope, size, None, offset=start, **filters
            )
        )
        total = await run(lambda: web.requests.count(scope, **filters))
        return {"rows": rows, "total": total}

    @router.get("/requests.csv")
    async def requests_csv(request: Request, me: Me = user_dep) -> Response:
        scope = me.scope()
        filters = _filters(request)
        await log(me, "export_requests", _describe(filters))
        rows = await run(
            lambda: web.requests.rows(scope, MAX_CSV_ROWS, None, **filters)
        )
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "documents": " ".join(row["documents"])})
        return Response(
            out.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="requests.csv"'
            },
        )

    @router.get("/users")
    async def users(
        window_s: float = 86400, origin: str = "", me: Me = user_dep
    ) -> Any:
        scope = me.scope()
        since = web.clock.now() - _window(window_s)
        synthetic = _origin(origin)
        stats = {
            r["user_id"]: r
            for r in await run(
                lambda: web.requests.per_user(scope, since, **synthetic)
            )
        }
        out = []
        for account in await run(web.store.accounts):
            person = web.directory.get(UserId(account.user_id))
            if person is None or not _counted(
                web, account.user_id, scope, synthetic
            ):
                continue
            usage = stats.get(account.user_id, {})
            out.append(
                {
                    "user_id": account.user_id,
                    "department": person.department,
                    "role": person.role,
                    "status": account.status,
                    "synthetic": _is_locust(account.user_id),
                    "program": person.program,
                    "created": account.created,
                    "last_login": account.last_login,
                    "requests": usage.get("requests", 0),
                    "errors": usage.get("errors", 0),
                    "rejected_429": usage.get("rejected_429", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "last_request": usage.get("last_request"),
                }
            )
        return {"users": out}

    @router.get("/actions")
    async def actions(
        limit: int = 20, offset: int = 0, me: Me = user_dep
    ) -> Any:
        me.scope()
        size, start = _page(limit, offset)
        return {
            "actions": await run(web.store.actions, size, start),
            "total": await run(web.store.action_count),
        }

    @router.get("/results")
    async def results(me: Me = user_dep) -> Any:
        me.scope()
        rows = await run(scorecard, [*web.loadtests(), *web.measurements()])
        return {"requirements": [asdict(r) for r in rows]}

    @router.get("/models")
    async def models(me: Me = user_dep) -> Any:
        me.scope()
        rows = await run(comparison, web.models_dir, web.active_model)
        job = web.serve_job or "<job>"
        return {
            "labels": model_labels(),
            "models": rows,
            "active": web.active_model,
            "switch": (
                f"scancel {job} && sbatch "
                "--export=ALL,ASKIISC_MODEL=<key> slurm/serve.sh"
            ),
        }

    _policy_routes(web, router, user_dep, log)
    people_routes(web, router, user_dep, log)
    return router


def _policy_routes(
    web: WebDeps,
    router: APIRouter,
    user_dep: Any,
    log: Callable[[Me, str, str], Awaitable[None]],
) -> None:
    """Policy administration and document revisions (via the access API).

    The access service itself checks that the admin manages the
    document's owning department.
    """

    async def proxy(
        me: Me, method: str, doc_id: str, body: Any = None
    ) -> Response:
        if web.access_http is None:
            return error(503, "policy administration is not configured")
        me.scope()
        if not DOC_REF.fullmatch(doc_id):
            return error(404, "no such document")
        response = await web.access_http.request(
            method,
            f"/v1/policy/{doc_id}",
            json=body,
            headers={"Authorization": f"Bearer {me.token}"},
            timeout=5.0,
        )
        return Response(
            response.content,
            status_code=response.status_code,
            media_type="application/json",
        )

    @router.post("/policies/preview")
    async def preview_policy(request: Request, me: Me = user_dep) -> Any:
        me.scope()
        try:
            draft = DocPolicy(**await request.json())
        except (ValidationError, TypeError, ValueError) as err:
            return error(422, f"invalid policy: {err}")
        return await run(policy_matches, web, draft)

    @router.get("/policies/{doc_id}")
    async def get_policy(doc_id: str, me: Me = user_dep) -> Response:
        return await proxy(me, "GET", doc_id)

    @router.put("/policies/{doc_id}")
    async def put_policy(
        doc_id: str, request: Request, me: Me = user_dep
    ) -> Response:
        body = await request.json()
        if isinstance(body, dict):  # malformed bodies: the service says
            co = body.get("co_owners")
            owners = [body.get("owner", INSTITUTE)]
            owners += co if isinstance(co, list) else []
            if unknown := unknown_owners(web, map(str, owners)):
                return error(400, f"no such department: {', '.join(unknown)}")
        response = await proxy(me, "PUT", doc_id, body)
        if response.status_code == 200:  # noqa: PLR2004
            await log(me, "policy_put", doc_id)
        return response

    @router.delete("/policies/{doc_id}")
    async def delete_policy(doc_id: str, me: Me = user_dep) -> Response:
        response = await proxy(me, "DELETE", doc_id)
        if response.status_code == 200:  # noqa: PLR2004
            await log(me, "policy_delete", doc_id)
        return response

    @router.post("/documents/{doc_id}/revise")
    async def revise(doc_id: str, body: Revision, me: Me = user_dep) -> Any:
        if web.revise is None:
            return error(503, "document revisions are not configured")
        policy = await proxy(me, "GET", doc_id)
        if policy.status_code != 200:  # noqa: PLR2004
            return policy
        result = await web.revise(doc_id, body.text)
        await log(me, "document_revise", doc_id)
        return result

    document_routes(web, router, user_dep, proxy, log)

    @router.post("/chaos/{name}")
    async def chaos(name: str, me: Me = user_dep) -> Any:
        if not may_manage(me.person, INSTITUTE) or name not in web.chaos:
            return error(404, "no such fault injection")
        await log(me, "chaos", name)
        return {"result": await web.chaos[name]()}


def policy_matches(web: WebDeps, draft: DocPolicy) -> dict[str, Any]:
    """How many directory people a draft policy lets read the document now.

    Uses the access service's own engine, so the count is what the policy
    will do (FR-3).
    """
    engine = PolicyEngine(web.directory, PolicyStore([draft]))
    now = web.clock.now()
    by_department: dict[str, int] = {}
    people = web.directory.people()
    for person in people:
        if engine.authorize(person.user_id, DocId(draft.doc_id), now).allowed:
            dept = person.department
            by_department[dept] = by_department.get(dept, 0) + 1
    return {
        "matches": sum(by_department.values()),
        "of": len(people),
        "by_department": dict(sorted(by_department.items())),
    }


def _in_scope(scope: set[str] | None, department: str) -> bool:
    return scope is None or department in scope


def _origin(origin: str) -> dict[str, bool]:
    """The request-log filter for "real", "locust" or both ("")."""
    return {"synthetic": ORIGINS[origin]} if origin in ORIGINS else {}


def _origin_matches(synthetic: dict[str, bool], is_synthetic: bool) -> bool:
    return not synthetic or synthetic["synthetic"] == is_synthetic


def _page(limit: int, offset: int) -> tuple[int, int]:
    """A page size (1 to MAX_PAGE_ROWS) and a start row (from 0)."""
    return max(1, min(limit, MAX_PAGE_ROWS)), max(0, offset)


def _ids(
    web: WebDeps,
    scope: set[str] | None,
    synthetic: dict[str, bool] | None = None,
) -> set[str] | None:
    synthetic = synthetic or {}
    if scope is None and not synthetic:
        return None
    return {
        a.user_id
        for a in web.store.accounts()
        if _counted(web, a.user_id, scope, synthetic)
    }


def _user_counts(
    web: WebDeps,
    scope: set[str] | None,
    synthetic: dict[str, bool] | None = None,
) -> dict[str, int]:
    counts = {"accounts": 0, "active": 0, "pending": 0, "disabled": 0}
    counts["locust"] = 0
    for account in web.store.accounts():
        if not _counted(web, account.user_id, scope, synthetic):
            continue
        counts["accounts"] += 1
        counts[account.status] += 1
        counts["locust"] += _is_locust(account.user_id)
    counts["live_sessions"] = sum(
        _counted(web, user_id, scope, synthetic)
        for user_id in web.store.session_users(web.clock.now())
    )
    return counts


def _is_locust(user_id: str) -> bool:
    """Test identities (no "@") are Locust's; e-mail ids are real people."""
    return not real_person(user_id)


def _counted(
    web: WebDeps,
    user_id: str,
    scope: set[str] | None,
    synthetic: dict[str, bool] | None,
) -> bool:
    """Whether a user is in the admin's departments and the Users filter."""
    person = web.directory.get(UserId(user_id))
    return (
        person is not None
        and _in_scope(scope, person.department)
        and _origin_matches(synthetic or {}, _is_locust(user_id))
    )


def _filters(request: Request) -> dict[str, Any]:
    params = request.query_params
    filters: dict[str, Any] = {
        key: params[key]
        for key in ("user_id", "department", "source", "decision")
        if params.get(key)
    }
    if params.get("status", "").isdigit():
        filters["status"] = int(params["status"])
    if params.get("synthetic") in {"true", "false"}:
        filters["synthetic"] = params["synthetic"] == "true"
    return filters


def _describe(filters: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(filters.items()))
