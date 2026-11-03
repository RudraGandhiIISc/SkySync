"""Document upload and version history on the admin API."""

import base64
from pathlib import Path
from typing import Any

import httpx
import pytest

from askiisc.common.errors import ExtractionError
from askiisc.common.types import DocId
from askiisc.web.documents import ID_TAKEN, MAX_UPLOAD_BYTES, find_documents
from tests.web_fixtures import HDR, Web, build_web


class Publisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, str, str]] = []
        self.fail = False
        self.versions: dict[str, list[dict[str, Any]]] = {}

    async def publish(
        self, doc_id: str, data: bytes, content_type: str, title: str
    ) -> dict[str, Any]:
        if self.fail:
            raise ExtractionError("no text")
        self.calls.append((doc_id, data, content_type, title))
        history = self.versions.setdefault(doc_id, [])
        if history:
            history[-1]["valid_to"] = 2000.0
        history.append(
            {"version": len(history) + 1, "valid_from": 1000.0,
             "valid_to": None, "chunks": 1}
        )  # fmt: skip
        return {"doc_id": doc_id, "version": len(history), "chunks": 1,
                "visible_from": 1005.0, "retired": 0}  # fmt: skip

    def history(self, doc_id: str) -> list[dict[str, Any]]:
        return self.versions.get(doc_id, [])


@pytest.fixture
def web(tmp_path: Path) -> tuple[Web, Publisher]:
    w = build_web(tmp_path)
    pub = Publisher()
    w.deps.publish = pub.publish
    w.deps.history = pub.history
    return w, pub


async def _as(web: Web, user: str) -> httpx.AsyncClient:
    client = web.browser()
    web.account(user)
    await web.login(user, client)
    return client


def _body(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "doc_id": "cds-lab-rules",
        "filename": "rules.txt",
        "content_b64": base64.b64encode(b"Lab rules: badge needed.").decode(),
        "departments": ["cds"],
    }
    body.update(kw)
    return body


async def test_department_admin_uploads_for_own_department(
    web: tuple[Web, Publisher],
) -> None:
    w, pub = web
    cds = await _as(w, "u_cds_admin")
    r = await cds.post("/api/admin/documents/upload", json=_body(), headers=HDR)
    assert r.status_code == 200, r.text
    assert r.json()["owners"] == ["cds"] and r.json()["version"] == 1
    assert pub.calls == [
        (
            "cds-lab-rules",
            b"Lab rules: badge needed.",
            "text/plain",
            "rules.txt",
        )
    ]
    policy = w.stack.w.policies.snapshot()[1][DocId("cds-lab-rules")]
    assert policy.owner == "cds" and policy.departments == {"cds"}
    # The same id again is refused: an upload never replaces a document.
    again = _body(
        filename="rules.html",
        content_b64=base64.b64encode(b"<p>v2</p>").decode(),
    )
    r = await cds.post("/api/admin/documents/upload", json=again, headers=HDR)
    assert r.status_code == 409 and r.json()["error"] == ID_TAKEN
    assert len(pub.calls) == 1
    # A new version is published explicitly.
    new = {"filename": "rules.html", "content_b64": again["content_b64"]}
    r = await cds.post(
        "/api/admin/documents/cds-lab-rules/version", json=new, headers=HDR
    )
    assert r.json()["version"] == 2 and pub.calls[-1][2] == "text/html"
    versions = (
        await cds.get("/api/admin/documents/cds-lab-rules/versions")
    ).json()
    assert [v["version"] for v in versions["versions"]] == [1, 2]
    assert versions["versions"][0]["valid_to"] == 2000.0
    actions = {
        (a["action"], a["target"])
        for a in w.deps.store.actions()
        if a["action"].startswith("document_")
    }
    assert actions == {
        ("document_upload", "cds-lab-rules v1"),
        ("document_version", "cds-lab-rules v2"),
    }
    other = _body(owners=["ece"], doc_id="ece-doc")
    r = await cds.post("/api/admin/documents/upload", json=other, headers=HDR)
    assert r.status_code == 403  # must include their own department


async def test_documents_can_have_several_owners(
    web: tuple[Web, Publisher],
) -> None:
    w, _ = web
    cds = await _as(w, "u_cds_admin")
    body = _body(doc_id="joint-seminar", owners=["ece", "CDS"])
    r = await cds.post("/api/admin/documents/upload", json=body, headers=HDR)
    assert r.status_code == 200 and r.json()["owners"] == ["cds", "ece"]
    policy = w.stack.w.policies.snapshot()[1][DocId("joint-seminar")]
    assert (policy.owner, policy.co_owners) == ("cds", {"ece"})
    assert policy.owners == {"cds", "ece"}
    assert (
        await cds.get("/api/admin/policies/joint-seminar")
    ).status_code == 200


async def test_owners_must_be_real_departments(
    web: tuple[Web, Publisher],
) -> None:
    w, pub = web
    inst = await _as(w, "u_admin")
    body = _body(doc_id="mech-notes", owners=["mech"], visibility="public")
    r = await inst.post("/api/admin/documents/upload", json=body, headers=HDR)
    assert r.status_code == 400 and "mech" in r.json()["error"]
    assert not pub.calls  # nothing ingested, no policy written
    assert DocId("mech-notes") not in w.stack.w.policies.snapshot()[1]
    r = await inst.put(
        "/api/admin/policies/d02",
        json={"owner": "cds", "co_owners": ["mech"], "visibility": "public"},
        headers=HDR,
    )
    assert r.status_code == 400 and "mech" in r.json()["error"]


async def test_institute_admin_chooses_the_owner(
    web: tuple[Web, Publisher],
) -> None:
    w, _ = web
    inst = await _as(w, "u_admin")
    body = _body(
        doc_id="ece-handbook",
        owners=["ece"],
        departments=[],
        visibility="public",
    )
    r = await inst.post("/api/admin/documents/upload", json=body, headers=HDR)
    assert r.status_code == 200 and r.json()["owners"] == ["ece"]
    # Nobody reuses the id, and other departments cannot add versions.
    cds = await _as(w, "u_cds_admin")
    r = await cds.post(
        "/api/admin/documents/upload",
        json=_body(doc_id="ece-handbook"),
        headers=HDR,
    )
    assert r.status_code == 409
    r = await cds.get("/api/admin/documents/ece-handbook/versions")
    assert r.status_code == 403
    new = {"filename": "x.txt", "content_b64": _body()["content_b64"]}
    r = await cds.post(
        "/api/admin/documents/ece-handbook/version", json=new, headers=HDR
    )
    assert r.status_code == 403
    r = await cds.post(
        "/api/admin/documents/no-such-doc/version", json=new, headers=HDR
    )
    assert r.status_code == 404


async def test_ids_are_checked_while_typing(
    web: tuple[Web, Publisher],
) -> None:
    w, _ = web
    inst = await _as(w, "u_admin")
    body = _body(doc_id="ece-handbook", owners=["ece"])
    assert (
        await inst.post("/api/admin/documents/upload", json=body, headers=HDR)
    ).status_code == 200
    cds = await _as(w, "u_cds_admin")
    for doc_id, exists in (("ece-handbook", True), ("fresh-id", False)):
        r = await cds.get(f"/api/admin/documents/check?doc_id={doc_id}")
        assert r.status_code == 200 and r.json()["exists"] is exists, doc_id
    assert r.json()["title"] == ""


def test_documents_are_found_by_id_or_title() -> None:
    catalogue = {
        "csa-lab-rules": {"title": "CSA lab rules", "url": "u1", "site": "csa"},
        "cds-thesis": {
            "title": "Thesis guidelines",
            "url": "u2",
            "site": "cds",
        },
        "rules-csa": {"title": "Older rules", "site": "csa", "kind": "pdf"},
    }
    found = find_documents(catalogue, "csa rules", 10)
    assert [d["doc_id"] for d in found["documents"]] == [
        "csa-lab-rules",
        "rules-csa",
    ]
    assert found["total"] == 2 and found["sites"] == [["csa", 2], ["cds", 1]]
    assert found["documents"][1]["kind"] == "pdf"
    thesis = find_documents(catalogue, "thesis", 10)["documents"]
    assert thesis[0]["url"] == "u2"
    everything = find_documents(catalogue, "", 1)
    assert len(everything["documents"]) == 1 and everything["total"] == 3
    only_cds = find_documents(catalogue, "", 10, site="cds")
    assert [d["doc_id"] for d in only_cds["documents"]] == ["cds-thesis"]


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"filename": "notes.docx"}, 415),
        ({"filename": "fake.pdf"}, 415),
        ({"content_b64": "@@not base64@@"}, 400),
        ({"content_b64": ""}, 422),
        ({"doc_id": "Bad Id!"}, 422),
        ({"departments": []}, 400),
    ],
)
async def test_bad_uploads_are_rejected(
    web: tuple[Web, Publisher], change: dict[str, Any], status: int
) -> None:
    w, pub = web
    cds = await _as(w, "u_cds_admin")
    r = await cds.post(
        "/api/admin/documents/upload", json=_body(**change), headers=HDR
    )
    assert r.status_code == status
    assert pub.calls == []


async def test_size_limit_and_empty_file(web: tuple[Web, Publisher]) -> None:
    w, _ = web
    cds = await _as(w, "u_cds_admin")
    big = base64.b64encode(b"x" * (MAX_UPLOAD_BYTES + 1)).decode()
    r = await cds.post(
        "/api/admin/documents/upload", json=_body(content_b64=big), headers=HDR
    )
    assert r.status_code == 413
    r = await cds.post(
        "/api/admin/documents/upload",
        json=_body(content_b64="===="),
        headers=HDR,
    )
    assert r.status_code == 400


async def test_a_failed_upload_leaves_nothing_behind(
    web: tuple[Web, Publisher],
) -> None:
    w, pub = web
    cds = await _as(w, "u_cds_admin")
    pub.fail = True
    r = await cds.post(
        "/api/admin/documents/upload", json=_body(doc_id="cds-new"), headers=HDR
    )
    assert r.status_code == 422 and "could not be ingested" in r.json()["error"]
    assert DocId("cds-new") not in w.stack.w.policies.snapshot()[1]
    pub.fail = False
    assert (
        await cds.post("/api/admin/documents/upload", json=_body(), headers=HDR)
    ).status_code == 200
    pub.fail = True
    new = {"filename": "x.txt", "content_b64": _body()["content_b64"]}
    r = await cds.post(
        "/api/admin/documents/cds-lab-rules/version", json=new, headers=HDR
    )
    assert r.status_code == 422
    assert DocId("cds-lab-rules") in w.stack.w.policies.snapshot()[1]


async def test_only_admins_and_configured(web: tuple[Web, Publisher]) -> None:
    w, _ = web
    student = await _as(w, "u_cds_stu")
    r = await student.post(
        "/api/admin/documents/upload", json=_body(), headers=HDR
    )
    assert r.status_code == 403
    cds = await _as(w, "u_cds_admin")
    assert (
        await cds.get("/api/admin/documents/zz/versions")
    ).status_code == 404
    w.deps.publish = None
    w.deps.history = None
    r = await cds.post("/api/admin/documents/upload", json=_body(), headers=HDR)
    assert r.status_code == 503
    assert (
        await cds.get("/api/admin/documents/d02/versions")
    ).status_code == 503
