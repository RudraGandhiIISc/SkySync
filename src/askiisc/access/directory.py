"""The user directory: departments, roles and groups from an export (A4).

The export is JSON (a list of objects) or CSV with the columns user_id,
department, role, groups; CSV groups are separated by semicolons.

It can also be a folder tree, one file per department and role:

    users/<department>/<students|faculty|staff|admins>.json

where department and role come from the path, and every user also gets
the group "<department>-<file stem>" (for example "cds-faculty"), so a
policy can name an intersection such as "CDS faculty" with one group.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from askiisc.common.errors import ConfigError
from askiisc.common.types import UserId


@dataclass(frozen=True)
class Person:
    """One directory entry."""

    user_id: UserId
    department: str
    role: str
    groups: frozenset[str]


def _person(record: Mapping[str, object]) -> Person:
    groups = record.get("groups", ())
    if isinstance(groups, str):
        groups = [g for g in groups.split(";") if g]
    if not isinstance(groups, Iterable):
        raise ConfigError(f"bad groups for {record.get('user_id')}")
    user_id = str(record["user_id"]).strip()
    if not user_id:
        raise ConfigError("empty user_id in directory export")
    return Person(
        user_id=UserId(user_id),
        department=str(record.get("department", "")).strip(),
        role=str(record.get("role", "")).strip(),
        groups=frozenset(str(g).strip() for g in groups),
    )


ROLE_FILES = {
    "students": "student",
    "faculty": "faculty",
    "staff": "staff",
    "admins": "admin",
}


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
    groups = person.groups | {f"{department}-{stem}"}
    return Person(person.user_id, department, role, groups)


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
    people: dict[UserId, Person] = {}
    for path in sorted(root.glob("*/*.json")):
        if path.stem not in ROLE_FILES:
            raise ConfigError(f"unknown role file {path.name}")
        try:
            records = json.loads(path.read_text())
            batch = [
                _tree_person(r, path.parent.name, path.stem) for r in records
            ]
        except (ValueError, KeyError, TypeError, AttributeError) as err:
            raise ConfigError(f"malformed {path}: {err}") from err
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
