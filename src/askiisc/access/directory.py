"""The user directory: departments, roles and groups from an export (A4).

The export is JSON (a list of objects) or CSV with the columns user_id,
department, role, groups; CSV groups are separated by semicolons.
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
        """Replaces the contents from an export file.

        Args:
            path: A .json or .csv export.

        Returns:
            The number of people loaded.
        """
        fmt = path.suffix.lstrip(".").lower()
        people = parse_export(path.read_text(), fmt)
        with self._lock:
            self._people = people
        return len(people)

    def get(self, user_id: UserId) -> Person | None:
        """Returns a person, or None for an unknown user."""
        with self._lock:
            return self._people.get(user_id)
