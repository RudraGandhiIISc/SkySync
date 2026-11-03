"""Documents for admins: find, upload, new versions and version history.

An admin uploads a PDF, HTML or text file as a new document, under an id
no other document has, with access rules in the usual policy form and one
or more owning departments (department admins must include their own;
institute admins choose any). The policy is written through the access
service (which checks ownership again), then the file goes through the
ingest pipeline: extracted, chunked, embedded and added to the index and
Milvus. If ingesting fails, the new policy is removed again.

An existing id is never overwritten by an upload. A new version of a
document is published explicitly ("Upload new version", or new text from
web/admin.py); the old version is retired, never deleted.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from askiisc.access.policy import INSTITUTE, may_change_owners
from askiisc.common.errors import AskIIScError
from askiisc.web.app import Me, WebDeps, error

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_OWNERS = 10
MAX_FOUND = 50
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}
DOC_ID = r"^[a-z0-9][a-z0-9._-]{2,99}$"
RULES = ("groups", "departments", "roles")
ID_TAKEN = "this document id is already used; choose another id"


class NewFile(BaseModel):
    """A file as base64 (a new version of an existing document)."""

    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=200)
    content_b64: str = Field(min_length=1, max_length=MAX_UPLOAD_BYTES * 2)


class Upload(NewFile):
    """POST /api/admin/documents/upload body: a new document."""

    doc_id: str = Field(pattern=DOC_ID)
    owners: list[str] = Field(default_factory=list, max_length=MAX_OWNERS)
    visibility: Literal["public", "restricted"] = "restricted"
    groups: list[str] = Field(default_factory=list, max_length=50)
    departments: list[str] = Field(default_factory=list, max_length=50)
    roles: list[str] = Field(default_factory=list, max_length=50)


def known_departments(web: WebDeps) -> set[str]:
    """Departments that may own documents: the directory's, and institute."""
    return {p.department for p in web.directory.people()} | {INSTITUTE}


def unknown_owners(web: WebDeps, owners: Iterable[str]) -> list[str]:
    """Owners that are not departments (nobody could ever manage them)."""
    known = known_departments(web)
    return sorted({o for o in owners if o not in known})


def owners_for(me: Me, requested: list[str]) -> list[str] | None:
    """The owning departments, own first; None if this admin may not.

    No choice means the admin's own department. Department admins must
    include their own; institute admins may name any departments.
    """
    own = me.person.department
    owners = list(
        dict.fromkeys(o.strip().lower() for o in requested if o.strip())
    )
    owners = owners or [own]
    if own in owners:
        owners = [own, *(o for o in owners if o != own)]
    if not may_change_owners(me.person, None, frozenset(owners)):
        return None
    return owners


def decode_file(body: NewFile) -> tuple[bytes, str] | Response:
    """The file bytes and content type, or an error response."""
    suffix = PurePosixPath(body.filename.lower()).suffix
    content_type = CONTENT_TYPES.get(suffix)
    if content_type is None:
        return error(415, "only .pdf, .html, .htm and .txt files")
    try:
        data = base64.b64decode(body.content_b64, validate=True)
    except (binascii.Error, ValueError):
        return error(400, "the file is not valid base64")
    if not data:
        return error(400, "the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        return error(413, f"files are limited to {MAX_UPLOAD_BYTES >> 20} MB")
    if content_type == "application/pdf" and not data.startswith(b"%PDF"):
        return error(415, "this is not a PDF file")
    return data, content_type


def find_documents(
    catalogue: Mapping[str, Mapping[str, str]],
    query: str,
    limit: int,
    site: str = "",
) -> dict[str, Any]:
    """Documents whose id or title has every word of the query.

    Ids that start with the query come first, then by id. Also returns how
    many match in all, and every site with its number of documents (for
    the site filter).

    Returns:
        {"documents": [{doc_id, title, url, site, kind, edition}],
        "total": int, "sites": [[site, count], ...]}.
    """
    words = query.lower().split()
    found = []
    sites: dict[str, int] = {}
    for doc_id, info in catalogue.items():
        where = info.get("site", "")
        sites[where] = sites.get(where, 0) + 1
        text = f"{doc_id} {info.get('title', '')}".lower()
        if (not site or where == site) and all(w in text for w in words):
            found.append(
                {
                    "doc_id": doc_id,
                    "title": info.get("title", ""),
                    "url": info.get("url", ""),
                    "site": where,
                    "kind": info.get("kind", ""),
                    "edition": info.get("edition", ""),
                }
            )
    first = query.strip().lower()
    found.sort(key=lambda d: (not d["doc_id"].startswith(first), d["doc_id"]))
    return {
        "documents": found[: max(1, min(limit, MAX_FOUND))],
        "total": len(found),
        "sites": sorted(
            ([s, n] for s, n in sites.items() if s), key=lambda x: (-x[1], x[0])
        ),
    }


Access = Callable[[Me, str, str, Any], Awaitable[Response]]
Log = Callable[[Me, str, str], Awaitable[None]]


def document_routes(  # noqa: PLR0915 - one route table
    web: WebDeps, router: APIRouter, user_dep: Any, access: Access, log: Log
) -> None:
    """Adds document search, upload, new-version and history routes."""

    @router.get("/documents")
    async def documents(
        q: str = "", limit: int = 20, site: str = "", me: Me = user_dep
    ) -> Any:
        me.scope()
        return web.find_docs(q, limit, site)

    @router.get("/documents/check")
    async def check(doc_id: str, me: Me = user_dep) -> Response:
        """Whether an id is taken (by any document, whoever owns it)."""
        me.scope()
        policy = await access(me, "GET", doc_id, None)
        if policy.status_code not in (200, 403, 404):
            return policy
        exists = policy.status_code != 404  # noqa: PLR2004
        title = web.docs(doc_id).get("title", "") if exists else ""
        return JSONResponse(
            {"doc_id": doc_id, "exists": exists, "title": title}
        )

    @router.post("/documents/upload")
    async def upload(body: Upload, me: Me = user_dep) -> Response:
        me.scope()
        if web.publish is None:
            return error(503, "uploads are not configured")
        owners = owners_for(me, body.owners)
        if owners is None:
            return error(403, "include your own department as an owner")
        if unknown := unknown_owners(web, owners):
            return error(400, f"no such department: {', '.join(unknown)}")
        if body.visibility == "restricted" and not any(
            getattr(body, rule) for rule in RULES
        ):
            return error(400, "a restricted document needs at least one rule")
        decoded = decode_file(body)
        if isinstance(decoded, Response):
            return decoded
        return await _publish_new(
            web.publish, me, body, owners, decoded, access, log
        )

    @router.post("/documents/{doc_id}/version")
    async def new_version(
        doc_id: str, body: NewFile, me: Me = user_dep
    ) -> Response:
        me.scope()
        if web.publish is None:
            return error(503, "uploads are not configured")
        policy = await access(me, "GET", doc_id, None)
        if policy.status_code != 200:  # noqa: PLR2004
            return policy  # unknown, or another department's document
        decoded = decode_file(body)
        if isinstance(decoded, Response):
            return decoded
        data, content_type = decoded
        try:
            result = await web.publish(
                doc_id, data, content_type, body.filename
            )
        except (AskIIScError, ValueError) as err:
            return error(422, f"the file could not be ingested: {err}")
        await log(me, "document_version", f"{doc_id} v{result['version']}")
        return JSONResponse(result)

    @router.get("/documents/{doc_id}/versions")
    async def versions(doc_id: str, me: Me = user_dep) -> Response:
        me.scope()
        if web.history is None:
            return error(503, "version history is not configured")
        policy = await access(me, "GET", doc_id, None)
        if policy.status_code != 200:  # noqa: PLR2004
            return policy
        history = web.history(doc_id)
        if not history:
            return error(404, "no versions of this document")
        return JSONResponse({"doc_id": doc_id, "versions": history})


Publish = Callable[[str, bytes, str, str], Awaitable[dict[str, Any]]]


async def _publish_new(  # noqa: PLR0913 - one upload's parts
    publish: Publish,
    me: Me,
    body: Upload,
    owners: list[str],
    decoded: tuple[bytes, str],
    access: Access,
    log: Log,
) -> Response:
    data, content_type = decoded
    previous = await access(me, "GET", body.doc_id, None)
    if previous.status_code in (200, 403):  # noqa: PLR2004
        return error(409, ID_TAKEN)  # never overwrite another document
    if previous.status_code != 404:  # noqa: PLR2004
        return previous
    policy = {
        "owner": owners[0],
        "co_owners": owners[1:],
        "visibility": body.visibility,
        **{rule: getattr(body, rule) for rule in RULES},
    }
    put = await access(me, "PUT", body.doc_id, policy)
    if put.status_code != 200:  # noqa: PLR2004
        return put
    try:
        result = await publish(body.doc_id, data, content_type, body.filename)
    except (AskIIScError, ValueError) as err:
        await access(me, "DELETE", body.doc_id, None)  # leave nothing behind
        return error(422, f"the file could not be ingested: {err}")
    await log(me, "document_upload", f"{body.doc_id} v{result['version']}")
    return JSONResponse({**result, "owners": owners})
