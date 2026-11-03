"""People administration: the users tree, its live reload, and who may
change whom (institute admins, department admins, users)."""

import json
import logging
from pathlib import Path

import httpx
import pytest
from deploy.serve import people_tree

from askiisc.access.directory import (
    Directory,
    DirectoryWatcher,
    build_people,
    parse_tree,
    tree_stamp,
)
from askiisc.common.errors import ConfigError, NotFound
from askiisc.common.types import UserId
from askiisc.web.cli import institute_admin
from askiisc.web.people import PeopleTree, make_entry, parse_csv
from askiisc.web.store import AppStore, load_key
from tests.web_fixtures import HDR, Web, build_web

TREE = {
    "institute/admins.json": [{"user_id": "u_admin"}],
    "cds/admins.json": [{"user_id": "u_cds_admin"}],
    "cds/students.json": [
        {
            "user_id": "u_cds_stu",
            "program": "mtech-cds",
            "joined": 2025,
            "groups": ["g1"],
        }
    ],
    "ece/admins.json": [{"user_id": "u_ece_admin"}],
    "ece/students.json": [{"user_id": "u_ece_stu"}],
}


def _tree(root: Path) -> Path:
    for name, records in TREE.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(records))
    return root


# ------------------------------------------------------- directory tree


def test_programmes_become_groups(tmp_path: Path) -> None:
    people = parse_tree(_tree(tmp_path))
    stu = people[UserId("u_cds_stu")]
    assert (stu.program, stu.joined) == ("mtech-cds", 2025)
    assert stu.groups == {
        "g1",
        "cds-students",
        "mtech-cds",
        "cds-mtech-cds",
        "cds-mtech-cds-2025",
        "cds-course-students",
    }
    with pytest.raises(ConfigError, match="programme 'z' is not valid"):
        build_people({("cds", "students"): [{"user_id": "x", "program": "z"}]})
    with pytest.raises(ConfigError, match="unknown role file"):
        build_people({("cds", "visitors"): []})
    (tmp_path / "cds" / "staff.json").write_text("{}")
    with pytest.raises(ConfigError, match="not a list"):
        parse_tree(tmp_path)


def test_watcher_reloads_and_keeps_the_last_good_tree(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = _tree(tmp_path)
    directory = Directory()
    directory.load(root)
    watcher = DirectoryWatcher(directory, root)
    assert not watcher.check()
    students = root / "ece" / "students.json"
    students.write_text(
        json.dumps([{"user_id": "u_ece_stu"}, {"user_id": "n"}])
    )
    assert watcher.check() and directory.get(UserId("n")) is not None
    caplog.set_level(logging.WARNING)
    students.write_text("not json")
    assert not watcher.check()
    assert directory.get(UserId("n")) is not None  # last good copy stays
    assert "not reloaded" in caplog.text
    assert not DirectoryWatcher(directory, tmp_path / "none").check()


def test_stamp_skips_a_file_removed_while_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path)
    real = Path.stat

    def stat(self: Path, **kw: object) -> object:
        if self.name == "students.json" and self.parent.name == "ece":
            raise FileNotFoundError(self)
        return real(self, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", stat)
    names = [name for name, _, _ in tree_stamp(root)]
    assert "ece/students.json" not in names and "cds/admins.json" in names


# ------------------------------------------------------------ the editor


def test_types_years_and_research_degrees_become_groups() -> None:
    people = build_people(
        {
            ("csa", "admins"): [{"user_id": "a"}],
            ("csa", "students"): [
                {"user_id": "p", "program": "phd", "joined": "2023"},
                {"user_id": "n", "program": "mtech-research"},  # no year
            ],
            ("csa", "faculty"): [
                {"user_id": "f", "kind": "faculty", "joined": 2010},
                {"user_id": "v", "kind": "visiting"},
            ],
            ("csa", "staff"): [{"user_id": "s", "kind": "research"}],
        }
    )
    groups = {str(u): p.groups for u, p in people.items()}
    assert groups["p"] >= {"phd", "csa-phd", "csa-phd-2023"}
    assert "csa-research-students" in groups["p"]
    assert groups["n"] == {
        "csa-students",
        "mtech-research",
        "csa-mtech-research",
        "csa-research-students",
    }
    assert groups["f"] == {"csa-faculty"}  # the plain type adds nothing
    assert people[UserId("f")].joined == 2010  # noqa: PLR2004
    assert groups["v"] == {"csa-faculty", "csa-visiting-faculty"}
    assert groups["s"] == {"csa-staff", "csa-research-staff"}
    for record, message in [
        ({"user_id": "x", "program": "phd"}, "programme 'phd' is not valid"),
        ({"user_id": "x", "kind": "visiting"}, "type 'visiting' is not valid"),
        ({"user_id": "x", "joined": "soon"}, "joining year 'soon'"),
        ({"user_id": "x", "joined": 1800}, "joining year 1800"),
    ]:
        with pytest.raises(ConfigError, match=message):
            build_people({("csa", "staff"): [record]})


def test_add_replace_remove_all_or_nothing(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    directory = Directory()
    tree = PeopleTree(root, directory)
    new = make_entry(
        " New.Person@IISc.example ", "CDS", "student", "phd", joined=2025
    )
    tree.add([new])
    assert new.user_id == "new.person@iisc.example"
    assert directory.get(UserId(new.user_id)) is not None  # reloaded
    before = (root / "cds" / "students.json").read_text()
    with pytest.raises(ConfigError, match="duplicate"):
        tree.add([make_entry("x1", "cds", "staff"), new])
    assert (root / "cds" / "students.json").read_text() == before
    assert not (root / "cds" / "staff.json").exists()  # nothing written
    with pytest.raises(ConfigError, match="without an admin"):
        tree.add([make_entry("x2", "physics", "staff")])
    tree.replace(new.user_id, make_entry(new.user_id, "ece", "faculty"))
    found = tree.find(new.user_id)
    assert found is not None and (found.department, found.role) == (
        "ece",
        "faculty",
    )
    assert oct((root / "ece" / "faculty.json").stat().st_mode)[-3:] == "600"
    with pytest.raises(ConfigError, match="without an admin"):
        tree.remove("u_ece_admin")  # ece still has people
    tree.remove(new.user_id)
    assert not (root / "ece" / "faculty.json").exists()
    with pytest.raises(NotFound):
        tree.remove("ghost")
    tree.remove("u_ece_stu")
    tree.remove("u_ece_admin")
    assert not (root / "ece").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("bad id!", "cds", "student"), "valid user id"),
        (("ok1", "C D S", "student"), "valid department"),
        (("ok1", "cds", "visitor"), "unknown role"),
        (("ok1", "cds", "student", "mba"), "choose the programme"),
        (("ok1", "csa", "student", "mtech-cds", (), "", 2025), "M.Tech CSE"),
        (("ok1", "cds", "student", "phd"), "needs a joining year"),
        (("ok1", "cds", "faculty", "phd"), "only students"),
        (("ok1", "cds", "staff", "", (), "visiting"), "unknown type"),
        (("ok1", "cds", "staff", "", (), "", 1800), "valid joining year"),
        (("ok1", "cds", "staff", "", ["Bad Group"]), "groups"),
    ],
)
def test_entries_are_checked(args: tuple, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        make_entry(*args)


def test_bulk_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = parse_csv(
        "user_id,role,program,joined,kind,groups\n"
        "a@iisc.example,student,phd,2025,,lab1;lab2\n"
        "b@iisc.example,staff,,,technical,\n",
        "cds",
    )
    assert [
        (r.user_id, r.department, r.role, r.program, r.joined, r.kind)
        for r in rows
    ] == [
        ("a@iisc.example", "cds", "student", "phd", 2025, ""),
        ("b@iisc.example", "cds", "staff", "", None, "technical"),
    ]
    assert rows[0].groups == ("lab1", "lab2")
    for text, message in [
        ("name,role\nx,student", "header with user_id"),
        ("user_id,colour\nx,red", "unknown columns"),
        ("user_id\n", "no rows"),
        ("user_id,role\nok1,staff\nok2,boss", "row 2: unknown role"),
        ("user_id,role,joined\nok1,staff,next", "row 1: 'next' is not"),
    ]:
        with pytest.raises(ConfigError, match=message):
            parse_csv(text, "cds")
    monkeypatch.setattr("askiisc.web.people.MAX_BULK_ROWS", 1)
    with pytest.raises(ConfigError, match="at most 1 rows"):
        parse_csv("user_id,role\nab,staff\ncd,staff\n", "cds")


def test_institute_admins_only_from_the_terminal_command(
    tmp_path: Path,
) -> None:
    tree = PeopleTree(_tree(tmp_path))
    store = AppStore(tmp_path / "app.db", load_key(tmp_path / "k"))
    assert institute_admin(tree, store, "Boss@IISc.example", True, 1.0) == (
        "boss@iisc.example"
    )
    found = tree.find("boss@iisc.example")
    assert found is not None and found.department == "institute"
    assert store.allowed("boss@iisc.example").status == "active"  # type: ignore[union-attr]
    with pytest.raises(NotFound):
        institute_admin(tree, store, "u_cds_admin", False, 2.0)
    institute_admin(tree, store, "boss@iisc.example", False, 3.0)
    assert tree.find("boss@iisc.example") is None
    assert store.allowed("boss@iisc.example").status == "revoked"  # type: ignore[union-attr]


def test_serve_edits_only_a_real_tree(tmp_path: Path) -> None:
    directory = Directory()
    assert people_tree("", directory) is None
    assert people_tree(str(tmp_path / "users.json"), directory) is None
    tree = people_tree(str(_tree(tmp_path)), directory)
    assert tree is not None and tree.root == tmp_path


# --------------------------------------------------------- dashboard API


# Real people (e-mail ids) beside the test ids: only they are managed on
# the dashboard; test ids belong to Locust's synthetic accounts.
REAL_STUDENT = "cds.stu@iisc.example"
REAL_ECE_ADMIN = "ece.admin@iisc.example"


@pytest.fixture
def web(tmp_path: Path) -> Web:
    web = build_web(tmp_path)
    root = _tree(tmp_path / "users")
    students = json.loads((root / "cds/students.json").read_text())
    students.append(
        {"user_id": REAL_STUDENT, "program": "mtech-cds", "joined": 2024}
    )
    (root / "cds/students.json").write_text(json.dumps(students))
    (root / "ece/admins.json").write_text(
        json.dumps([{"user_id": REAL_ECE_ADMIN}])
    )
    web.deps.directory.load(root)
    web.deps.people = PeopleTree(root, web.deps.directory)
    return web


async def _as(web: Web, user: str) -> httpx.AsyncClient:
    browser = web.browser()
    web.account(user)
    assert (await web.login(user, browser)).status_code == 200
    return browser


def _person(
    name: str, department: str = "cds", role: str = "student"
) -> dict[str, object]:
    """A form for a real person (an e-mail id) unless `name` has an @."""
    student = role == "student"
    return {
        "user_id": name if "@" in name else f"{name}@iisc.example",
        "department": department,
        "role": role,
        "program": "phd" if student else "",
        "joined": 2025 if student else None,
        "groups": [],
    }


async def test_users_see_only_the_chat_app(web: Web) -> None:
    stu = await _as(web, "u_cds_stu")
    assert (await stu.get("/api/admin/people")).status_code == 403
    r = await stu.post("/api/admin/people", json=_person("x9"), headers=HDR)
    assert r.status_code == 403


async def test_department_admins_view_all_and_manage_their_own(
    web: Web,
) -> None:
    cds = await _as(web, "u_cds_admin")
    body = (await cds.get("/api/admin/people")).json()
    rows = {p["user_id"]: p for p in body["people"]}
    assert set(rows) == {
        "u_admin",
        "u_cds_admin",
        "u_cds_stu",
        REAL_STUDENT,
        REAL_ECE_ADMIN,
        "u_ece_stu",
    }
    assert body["manages"] == ["cds"] and not body["institute_admin"]
    assert body["departments"] == ["cds", "ece"]
    assert "phd" in body["programs"]
    manage = {u for u, p in rows.items() if p["can_manage"]}
    assert manage == {REAL_STUDENT}  # not the test id u_cds_stu
    assert rows["u_cds_admin"]["account"] == "active"
    assert rows["u_cds_admin"]["registered"] is not None
    assert rows["u_ece_stu"]["account"] == "not registered"
    assert rows["u_ece_stu"]["registered"] is None
    r = await cds.post("/api/admin/people", json=_person("new1"), headers=HDR)
    assert r.status_code == 201
    test_id = {**_person("x"), "user_id": "new0"}
    r = await cds.post("/api/admin/people", json=test_id, headers=HDR)
    assert r.status_code == 422 and "real people" in r.json()["error"]
    for form in (
        _person("new2", "ece"),
        _person("new3", role="admin"),
        _person("new4", "institute", "admin"),
    ):
        r = await cds.post("/api/admin/people", json=form, headers=HDR)
        assert r.status_code == 403, form
    for target in ("u_ece_stu", REAL_ECE_ADMIN, "u_admin", "u_cds_stu"):
        r = await cds.delete(f"/api/admin/people/{target}", headers=HDR)
        assert r.status_code == 403, target
    r = await cds.put(
        f"/api/admin/people/{REAL_STUDENT}",
        json=_person(REAL_STUDENT, "ece"),
        headers=HDR,
    )
    assert r.status_code == 403  # cannot move people out of the department
    actions = (await cds.get("/api/admin/actions")).json()["actions"]
    assert [(a["action"], a["target"]) for a in actions] == [
        ("person_add", "new1@iisc.example")
    ]


async def test_adding_a_person_allows_them_in_one_action(web: Web) -> None:
    cds = await _as(web, "u_cds_admin")
    form = _person("New.Student@IISc.example")
    new = "new.student@iisc.example"
    assert (
        await cds.post("/api/admin/people", json=form, headers=HDR)
    ).status_code == 201
    assert web.deps.directory.get(UserId(new)) is not None
    entry = web.deps.store.allowed(new)
    assert entry is not None and entry.status == "active"
    assert entry.added_by == "u_cds_admin"
    rows = {
        p["user_id"]: p
        for p in (await cds.get("/api/admin/people")).json()["people"]
    }
    assert rows[new]["access"] == "not registered"  # no password yet
    account = web.deps.store.account(new)
    assert account is not None and not account.has_password
    # A refused tree write allows nobody (the allowlist row is rolled back).
    again = _person("other.student@iisc.example")
    web.deps.people.add = _refuse  # type: ignore[method-assign, union-attr]
    r = await cds.post("/api/admin/people", json=again, headers=HDR)
    assert r.status_code == 422
    assert web.deps.store.allowed("other.student@iisc.example") is None
    r = await cds.put(
        f"/api/admin/people/{new}",
        json={**form, "user_id": new, "program": "mtech-research"},
        headers=HDR,
    )
    person = r.json()["person"]
    assert r.status_code == 200 and person["program"] == "mtech-research"
    rows = {
        p["user_id"]: p
        for p in (await cds.get("/api/admin/people")).json()["people"]
    }
    assert (rows[new]["program"], rows[new]["joined"]) == (
        "mtech-research",
        2025,
    )
    r = await cds.delete(f"/api/admin/people/{new}", headers=HDR)
    assert r.status_code == 200
    assert web.deps.store.account(new) is None
    assert web.deps.store.allowed(new).status == "revoked"  # type: ignore[union-attr]


def _refuse(entries: object) -> None:
    raise ConfigError("the users tree refused it")


async def test_revoking_access_keeps_the_account_and_chats(web: Web) -> None:
    cds = await _as(web, "u_cds_admin")
    web.account(REAL_STUDENT)
    web.deps.store.allow_existing([REAL_STUDENT], "u_cds_admin", 1.0)
    chat = web.deps.store.create_chat(REAL_STUDENT, "Fees", 1.0)
    for action, status, access in (
        ("revoke", 200, "revoked"),
        ("restore", 200, "active"),
        ("explode", 404, None),
    ):
        r = await cds.post(
            f"/api/admin/people/{REAL_STUDENT}/{action}", headers=HDR
        )
        assert r.status_code == status, action
        if access:
            rows = {
                p["user_id"]: p
                for p in (await cds.get("/api/admin/people")).json()["people"]
            }
            assert rows[REAL_STUDENT]["access"] == access
    assert web.deps.store.account(REAL_STUDENT) is not None
    assert [c.chat_id for c in web.deps.store.chats(REAL_STUDENT)] == [
        chat.chat_id
    ]
    for target in (REAL_ECE_ADMIN, "u_cds_stu", "ghost@iisc.example"):
        r = await cds.post(f"/api/admin/people/{target}/revoke", headers=HDR)
        assert r.status_code in {403, 404}, target
    actions = [
        a["action"]
        for a in (await cds.get("/api/admin/actions")).json()["actions"]
    ]
    assert {"access_restore", "access_revoke"} <= set(actions)


async def test_institute_admins_manage_department_admins(web: Web) -> None:
    inst = await _as(web, "u_admin")
    body = (await inst.get("/api/admin/people")).json()
    assert body["institute_admin"] and body["manages"] == "all"
    r = await inst.post(
        "/api/admin/people", json=_person("ece2", "ece", "admin"), headers=HDR
    )
    assert r.status_code == 201
    r = await inst.put(
        "/api/admin/people/ece2@iisc.example",
        json=_person("ece2", "ece", "faculty"),
        headers=HDR,
    )
    assert r.status_code == 200  # admin access revoked
    r = await inst.post(
        "/api/admin/people",
        json=_person("boss2", "institute", "admin"),
        headers=HDR,
    )
    assert r.status_code == 403  # institute admins: terminal only
    assert (
        await inst.delete("/api/admin/people/u_admin", headers=HDR)
    ).status_code == 403  # not themselves
    r = await inst.post(
        "/api/admin/people", json=_person(REAL_STUDENT), headers=HDR
    )
    assert r.status_code == 422 and "duplicate" in r.json()["error"]
    r = await inst.put(
        f"/api/admin/people/{REAL_STUDENT}", json=_person("other"), headers=HDR
    )
    assert r.status_code == 422  # the id cannot change
    for method in ("put", "delete"):
        kw = {"json": _person("ghost")} if method == "put" else {}
        r = await getattr(inst, method)(
            "/api/admin/people/ghost", headers=HDR, **kw
        )
        assert r.status_code == 404
    r = await inst.post(
        "/api/admin/people",
        json={**_person("bad"), "program": "mba"},
        headers=HDR,
    )
    assert r.status_code == 422
    r = await inst.delete(f"/api/admin/people/{REAL_ECE_ADMIN}", headers=HDR)
    assert r.status_code == 422  # ece would have no admin left


async def test_bulk_add_previews_then_applies(web: Web) -> None:
    cds = await _as(web, "u_cds_admin")
    text = (
        "user_id,role,program,joined\nb1@iisc.example,student,phd,2025\n"
        "b2@iisc.example,staff,,\n"
    )
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": text}, headers=HDR
    )
    assert r.json()["applied"] is False and r.json()["count"] == 2
    assert web.deps.directory.get(UserId("b2@iisc.example")) is None
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": text, "apply": True}, headers=HDR
    )
    assert r.json()["applied"] is True
    assert web.deps.directory.get(UserId("b2@iisc.example")) is not None
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": text, "apply": True}, headers=HDR
    )
    assert r.status_code == 422  # already there; nothing added twice
    other = "user_id,department,role\nb3@iisc.example,ece,staff\n"
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": other}, headers=HDR
    )
    assert r.status_code == 403 and "row 1" in r.json()["error"]
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": "nope"}, headers=HDR
    )
    assert r.status_code == 422
    test_ids = "user_id,role\nb4@iisc.example,staff\nlocust9,staff\n"
    r = await cds.post(
        "/api/admin/people/bulk", json={"csv": test_ids}, headers=HDR
    )
    assert r.status_code == 422 and "row 2" in r.json()["error"]


async def test_without_a_users_tree_people_is_read_only(web: Web) -> None:
    web.deps.people = None
    inst = await _as(web, "u_admin")
    assert (await inst.get("/api/admin/people")).status_code == 503
    for method, path, kw in (
        ("post", "/api/admin/people", {"json": _person("z1")}),
        ("put", "/api/admin/people/u_cds_stu", {"json": _person("u_cds_stu")}),
        ("delete", "/api/admin/people/u_cds_stu", {}),
        ("post", "/api/admin/people/bulk", {"json": {"csv": "user_id\nz"}}),
    ):
        r = await getattr(inst, method)(path, headers=HDR, **kw)
        assert r.status_code == 503, path


async def test_test_ids_never_count_as_real_people(web: Web) -> None:
    # Regression: admin_csa (no account yet) was listed under "Real".
    inst = await _as(web, "u_admin")
    rows = {
        p["user_id"]: p
        for p in (await inst.get("/api/admin/people")).json()["people"]
    }
    assert rows["u_ece_stu"]["account"] == "not registered"
    assert rows["u_ece_stu"]["synthetic"] is True
    assert rows[REAL_STUDENT]["synthetic"] is False
