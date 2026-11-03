"""The user directory: departments, roles and groups from an export (A4).

The export is JSON (a list of objects) or CSV with the columns user_id,
department, role, groups; CSV groups are separated by semicolons.

It can also be a folder tree, one file per department and role:

    users/<department>/<students|faculty|staff|admins>.json

where department and role come from the path, and every user also gets
the group "<department>-<file stem>" (for example "cds-faculty"), so a
policy can name an intersection such as "CDS faculty" with one group.

Only students have a programme (`"program": "phd"`, one of PROGRAMS) and
usually a joining year (`"joined": 2025`). A student then also gets the
groups "phd", "<department>-phd", "<department>-phd-<year>" and
"<department>-research-students" (PhD, M.Tech (Research)) or
"<department>-course-students" (M.Tech). Faculty and staff may have a
type (`"kind"`, one of KINDS for their role; for example visiting
faculty, technical staff) and an optional joining year; a type other than
the plain one adds the group "<department>-<kind>-<file stem>" (for
example "csa-visiting-faculty", "cds-research-staff").

Admins edit the tree from the dashboard (web/people.py) and the terminal
(web/cli.py); `DirectoryWatcher` reloads it when its files change, and
keeps the last good copy if an edit leaves it unreadable.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from askiisc.common.errors import ConfigError
from askiisc.common.types import UserId

_LOG = logging.getLogger(__name__)

PROGRAMS = {
    "phd": "PhD",
    "mtech-research": "M.Tech (Research)",
    "mtech-cse": "M.Tech CSE",
    "mtech-cds": "M.Tech (CDS)",
}
"""Student programme key (also an access group) to its display name."""
RESEARCH_PROGRAMS = frozenset({"phd", "mtech-research"})
"""Degrees by research; the others are course degrees."""
DEPARTMENT_PROGRAMS = {
    "csa": ("phd", "mtech-research", "mtech-cse"),
    "cds": ("phd", "mtech-research", "mtech-cds"),
}
"""The programmes each department offers (from its website, 2026-10)."""
KINDS = {
    "faculty": {"faculty": "Faculty", "visiting": "Visiting / guest faculty"},
    "staff": {
        "administrative": "Administrative",
        "technical": "Technical",
        "research": "Research staff (postdoc, project)",
    },
}
"""The types of faculty and staff, by role (the first is the plain one)."""
FIRST_YEAR = 1950
LAST_YEAR = 2100


@dataclass(frozen=True)
class Person:
    """One directory entry."""

    user_id: UserId
    department: str
    role: str
    groups: frozenset[str]
    program: str = ""
    kind: str = ""
    joined: int | None = None


def _person(record: Mapping[str, object]) -> Person:
    groups = record.get("groups", ())
    if isinstance(groups, str):
        groups = [g for g in groups.split(";") if g]
    if not isinstance(groups, Iterable):
        raise ConfigError(f"bad groups for {record.get('user_id')}")
    user_id = str(record["user_id"]).strip()
    if not user_id:
        raise ConfigError("empty user_id in directory export")
    role = str(record.get("role", "")).strip()
    program = str(record.get("program", "")).strip()
    if program and (program not in PROGRAMS or role != "student"):
        raise ConfigError(f"{user_id}: programme {program!r} is not valid")
    kind = str(record.get("kind", "")).strip()
    if kind and kind not in KINDS.get(role, {}):
        raise ConfigError(f"{user_id}: type {kind!r} is not valid")
    return Person(
        user_id=UserId(user_id),
        department=str(record.get("department", "")).strip(),
        role=role,
        groups=frozenset(str(g).strip() for g in groups),
        program=program,
        kind=kind,
        joined=_year(user_id, record.get("joined")),
    )


def _year(user_id: str, value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(str(value))
    except ValueError:
        year = 0
    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise ConfigError(f"{user_id}: joining year {value!r} is not valid")
    return year


ROLE_FILES = {
    "students": "student",
    "faculty": "faculty",
    "staff": "staff",
    "admins": "admin",
}
ROLE_STEMS = {role: stem for stem, role in ROLE_FILES.items()}


def _tree_person(
    record: Mapping[str, object], department: str, stem: str
) -> Person:
    role = ROLE_FILES[stem]
    for key, value in (("department", department), ("role", role)):
        if key in record and record[key] != value:
            raise ConfigError(
                f"{record.get('user_id')}: {key} {record[key]!r} contradicts "
                f"its folder ({value!r})"
            )
    person = _person({**record, "department": department, "role": role})
    groups = person.groups | {f"{department}-{stem}"} | _derived(person)
    return replace(person, groups=frozenset(groups))


def _derived(person: Person) -> set[str]:
    """The groups a person's programme, type and year add (see above)."""
    dept, program = person.department, person.program
    if program:
        degree = "research" if program in RESEARCH_PROGRAMS else "course"
        out = {program, f"{dept}-{program}", f"{dept}-{degree}-students"}
        if person.joined is not None:
            out.add(f"{dept}-{program}-{person.joined}")
        return out
    plain = next(iter(KINDS.get(person.role, {})), "")
    if person.kind and person.kind != plain:
        return {f"{dept}-{person.kind}-{ROLE_STEMS[person.role]}"}
    return set()


def read_tree(root: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    """The raw records of a users tree, by (department, file stem).

    Raises:
        ConfigError: On an unknown role file or malformed JSON.
    """
    files: dict[tuple[str, str], list[dict[str, object]]] = {}
    for path in sorted(root.glob("*/*.json")):
        if path.stem not in ROLE_FILES:
            raise ConfigError(f"unknown role file {path.name}")
        try:
            records = json.loads(path.read_text())
        except ValueError as err:
            raise ConfigError(f"malformed {path}: {err}") from err
        if not isinstance(records, list):
            raise ConfigError(f"malformed {path}: not a list")
        files[(path.parent.name, path.stem)] = records
    return files


def parse_tree(root: Path) -> dict[UserId, Person]:
    """Reads a users/<department>/<role>.json folder tree.

    Args:
        root: The users folder.

    Returns:
        People by user id.

    Raises:
        ConfigError: On an unknown role file, a record that contradicts its
            folder, a duplicate user id, malformed JSON, or a department
            with no admin (every department needs its own admin).
    """
    return build_people(read_tree(root))


def build_people(
    files: Mapping[tuple[str, str], Iterable[Mapping[str, object]]],
) -> dict[UserId, Person]:
    """Validates tree records (see parse_tree) and returns the people.

    Raises:
        ConfigError: As for parse_tree.
    """
    people: dict[UserId, Person] = {}
    for (department, stem), records in sorted(files.items()):
        if stem not in ROLE_FILES:
            raise ConfigError(f"unknown role file {stem}.json")
        try:
            batch = [_tree_person(r, department, stem) for r in records]
        except (KeyError, TypeError, AttributeError) as err:
            raise ConfigError(f"malformed {department}/{stem}: {err}") from err
        for person in batch:
            if person.user_id in people:
                raise ConfigError(f"duplicate user {person.user_id}")
            people[person.user_id] = person
    departments = {p.department for p in people.values()}
    with_admin = {p.department for p in people.values() if p.role == "admin"}
    if missing := sorted(departments - with_admin):
        raise ConfigError(f"departments without an admin: {missing}")
    return people


def parse_export(text: str, fmt: str) -> dict[UserId, Person]:
    """Parses a directory export.

    Args:
        text: File contents.
        fmt: "json" or "csv".

    Returns:
        People by user id.

    Raises:
        ConfigError: On an unknown format or a malformed record.
    """
    if fmt == "json":
        records = json.loads(text)
    elif fmt == "csv":
        records = list(csv.DictReader(io.StringIO(text)))
    else:
        raise ConfigError(f"unknown directory format {fmt}")
    try:
        people = [_person(r) for r in records]
    except (KeyError, TypeError, AttributeError) as err:
        raise ConfigError(f"malformed directory record: {err}") from err
    return {p.user_id: p for p in people}


class Directory:
    """A reloadable, thread-safe view of the directory export."""

    def __init__(self, people: Mapping[UserId, Person] | None = None) -> None:
        self._people = dict(people or {})
        self._lock = threading.Lock()

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> Directory:
        """Builds a directory from in-memory records."""
        return cls({p.user_id: p for p in map(_person, records)})

    def load(self, path: Path) -> int:
        """Replaces the contents from an export file or folder tree.

        Args:
            path: A .json or .csv export, or a users folder tree.

        Returns:
            The number of people loaded.
        """
        if path.is_dir():
            people = parse_tree(path)
        else:
            fmt = path.suffix.lstrip(".").lower()
            people = parse_export(path.read_text(), fmt)
        with self._lock:
            self._people = people
        return len(people)

    def get(self, user_id: UserId) -> Person | None:
        """Returns a person, or None for an unknown user."""
        with self._lock:
            return self._people.get(user_id)

    def people(self) -> list[Person]:
        """Everyone, sorted by user id."""
        with self._lock:
            return sorted(self._people.values(), key=lambda p: p.user_id)


def tree_stamp(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Name, modification time and size of every file in a users tree."""
    out = []
    for path in sorted(root.glob("*/*.json")):
        try:
            info = path.stat()
        except OSError:  # removed while listing
            continue
        out.append(
            (str(path.relative_to(root)), info.st_mtime_ns, info.st_size)
        )
    return tuple(out)


class DirectoryWatcher:
    """Reloads a directory when its users tree changes on disk.

    A tree that fails to parse is logged and skipped: the last good copy
    stays in use, so a bad edit never opens access (unknown users stay
    unknown) and the next good edit is picked up.
    """

    def __init__(self, directory: Directory, root: Path) -> None:
        self._directory = directory
        self._root = root
        self._stamp = tree_stamp(root) if root.is_dir() else ()

    def check(self) -> bool:
        """Reloads if the tree changed; returns whether it reloaded."""
        if not self._root.is_dir():
            return False
        stamp = tree_stamp(self._root)
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        try:
            count = self._directory.load(self._root)
        except ConfigError as err:
            _LOG.warning("users tree not reloaded: %s", err)
            return False
        _LOG.info("users tree reloaded: %d people", count)
        return True

    def run_forever(self, interval_s: float) -> None:  # pragma: no cover
        """Checks every interval_s seconds (a daemon thread's target)."""
        import time  # noqa: PLC0415

        while True:
            time.sleep(interval_s)
            self.check()
