"""The users tree, edited by admins (dashboard and terminal).

The tree is `users/<department>/<students|faculty|staff|admins>.json`
(access/directory.py). Every edit reads the whole tree under a file lock,
applies the change, validates the result exactly as the access service
will read it (`build_people`), and only then rewrites the changed files
(each atomically, via a temporary file and rename). A rejected edit
changes nothing. The web app's directory is reloaded at once; the access
service picks the change up within seconds (DirectoryWatcher).

Permissions are checked by the caller (web/people_admin.py, web/cli.py).
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from askiisc.access.directory import (
    DEPARTMENT_PROGRAMS,
    FIRST_YEAR,
    KINDS,
    LAST_YEAR,
    PROGRAMS,
    ROLE_FILES,
    Directory,
    build_people,
    read_tree,
)
from askiisc.common.errors import ConfigError, NotFound

STEM_OF = {role: stem for stem, role in ROLE_FILES.items()}
"""Role ("student") to its file stem ("students")."""
MAX_GROUPS = 20
MAX_BULK_ROWS = 5000
_ID = re.compile(r"^[a-z0-9][a-z0-9._+-]{0,99}(@[a-z0-9.-]{3,100})?$")
_NAME = re.compile(r"^[a-z][a-z0-9-]{1,30}$")
_GROUP = re.compile(r"^[a-z0-9][a-z0-9._-]{0,60}$")
CSV_COLUMNS = (
    "user_id",
    "department",
    "role",
    "program",
    "joined",
    "kind",
    "groups",
)


@dataclass(frozen=True)
class Entry:
    """One person as an admin enters them.

    Attributes:
        user_id: IISc e-mail (lower-cased) or a plain test id.
        department: Folder name, for example "cds".
        role: student, faculty, staff or admin.
        program: A PROGRAMS key (students only), or "".
        groups: Extra access groups (the derived ones are added on read).
        kind: A KINDS type for faculty and staff, or "".
        joined: The joining year (required for students), or None.
    """

    user_id: str
    department: str
    role: str
    program: str = ""
    groups: tuple[str, ...] = field(default_factory=tuple)
    kind: str = ""
    joined: int | None = None

    def record(self) -> dict[str, object]:
        """The JSON record stored in the department's role file."""
        out: dict[str, object] = {"user_id": self.user_id}
        if self.program:
            out["program"] = self.program
        if self.kind:
            out["kind"] = self.kind
        if self.joined is not None:
            out["joined"] = self.joined
        out["groups"] = list(self.groups)
        return out


def make_entry(  # noqa: PLR0913 - one argument per form field
    user_id: str,
    department: str,
    role: str,
    program: str = "",
    groups: Sequence[str] = (),
    kind: str = "",
    joined: int | None = None,
) -> Entry:
    """Normalises and checks one entry.

    A student needs a programme their department offers and a joining
    year; faculty and staff may have a type and a joining year.

    Raises:
        ConfigError: On a malformed id, department, role, programme, type,
            year or group.
    """
    uid = user_id.strip().lower()
    dept = department.strip().lower()
    clean = tuple(sorted({g.strip().lower() for g in groups if g.strip()}))
    if not _ID.match(uid):
        raise ConfigError(f"{user_id!r} is not a valid user id or e-mail")
    if not dept:
        raise ConfigError("the department is missing")
    if not _NAME.match(dept):
        raise ConfigError(f"{department!r} is not a valid department")
    if role not in STEM_OF:
        raise ConfigError(f"unknown role {role!r}")
    _check_details(dept, role, program, kind, joined)
    if len(clean) > MAX_GROUPS or not all(_GROUP.match(g) for g in clean):
        raise ConfigError("groups: up to 20 of a-z, 0-9, '.', '_', '-'")
    return Entry(uid, dept, role, program, clean, kind, joined)


def _check_details(
    dept: str, role: str, program: str, kind: str, joined: int | None
) -> None:
    """Checks programme, type and joining year for a role.

    Only students have a programme (one their department offers) and need
    a joining year; a type is only for faculty and staff.

    Raises:
        ConfigError: Saying what is wrong.
    """
    if role == "student":
        offered = DEPARTMENT_PROGRAMS.get(dept, tuple(PROGRAMS))
        if program not in offered:
            names = ", ".join(PROGRAMS[p] for p in offered)
            raise ConfigError(f"choose the programme: {names}")
        if joined is None:
            raise ConfigError("a student needs a joining year")
    elif program:
        raise ConfigError("only students have a programme")
    if kind and kind not in KINDS.get(role, {}):
        raise ConfigError(f"unknown type {kind!r} for {role}")
    if joined is not None and not FIRST_YEAR <= joined <= LAST_YEAR:
        raise ConfigError(f"{joined} is not a valid joining year")


def parse_csv(text: str, default_department: str = "") -> list[Entry]:
    """Bulk-add rows: user_id, department, role, program, joined, kind, groups.

    Groups are separated by ";".

    The header row is required; department may be left out (or blank)
    when `default_department` is given.

    Raises:
        ConfigError: Naming the first bad row (1 is the first data row).
    """
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames or "user_id" not in reader.fieldnames:
        raise ConfigError("the first line must be a header with user_id")
    unknown = set(reader.fieldnames) - set(CSV_COLUMNS)
    if unknown:
        raise ConfigError(f"unknown columns: {sorted(unknown)}")
    out = []
    for n, row in enumerate(reader, start=1):
        if n > MAX_BULK_ROWS:
            raise ConfigError(f"at most {MAX_BULK_ROWS} rows at a time")
        try:
            out.append(
                make_entry(
                    row.get("user_id") or "",
                    row.get("department") or default_department,
                    (row.get("role") or "student").strip().lower(),
                    (row.get("program") or "").strip().lower(),
                    (row.get("groups") or "").split(";"),
                    (row.get("kind") or "").strip().lower(),
                    _csv_year(row.get("joined")),
                )
            )
        except ConfigError as err:
            raise ConfigError(f"row {n}: {err}") from err
    if not out:
        raise ConfigError("no rows")
    return out


def _csv_year(value: str | None) -> int | None:
    """A CSV joining year, or None when blank.

    Raises:
        ConfigError: If it is not a whole number.
    """
    text = (value or "").strip()
    if not text:
        return None
    if not text.isdigit():
        raise ConfigError(f"{text!r} is not a valid joining year")
    return int(text)


Files = dict[tuple[str, str], list[dict[str, object]]]


class PeopleTree:
    """Reads and edits one users tree."""

    def __init__(self, root: Path, directory: Directory | None = None) -> None:
        """Opens a tree.

        Args:
            root: The users folder.
            directory: Reloaded after every edit (the web app's copy).
        """
        self.root = root
        self._directory = directory

    def entries(self) -> list[Entry]:
        """Everyone as stored (without the derived groups), by user id."""
        out = [
            Entry(
                str(r["user_id"]),
                dept,
                ROLE_FILES[stem],
                str(r.get("program", "")),
                tuple(r.get("groups", ())),  # type: ignore[arg-type]
                str(r.get("kind", "")),
                _stored_year(r.get("joined")),
            )
            for (dept, stem), records in read_tree(self.root).items()
            for r in records
        ]
        return sorted(out, key=lambda e: e.user_id)

    def find(self, user_id: str) -> Entry | None:
        """One stored entry, or None."""
        return next((e for e in self.entries() if e.user_id == user_id), None)

    def add(self, entries: Sequence[Entry]) -> None:
        """Adds people, all or none.

        Raises:
            ConfigError: If an id is taken (or repeated), or the result is
                invalid (for example a department without an admin).
        """

        def change(files: Files) -> None:
            for entry in entries:
                files.setdefault(_key(entry), []).append(entry.record())

        self._edit(change)

    def replace(self, user_id: str, entry: Entry) -> None:
        """Replaces a person's entry (department, role, programme, groups).

        Raises:
            NotFound: If the user is not in the tree.
            ConfigError: If the result is invalid.
        """

        def change(files: Files) -> None:
            _drop(files, user_id)
            files.setdefault(_key(entry), []).append(entry.record())

        self._edit(change)

    def remove(self, user_id: str) -> None:
        """Removes a person.

        Raises:
            NotFound: If the user is not in the tree.
            ConfigError: If the result is invalid (a department's last
                admin while it still has people).
        """
        self._edit(lambda files: _drop(files, user_id))

    def _edit(self, change: Callable[[Files], None]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with _locked(self.root / ".lock"):
            before = read_tree(self.root)
            after = {k: list(v) for k, v in before.items()}
            change(after)
            build_people(after)  # raises before anything is written
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    _write(self.root, key, after.get(key, []))
        if self._directory is not None:
            self._directory.load(self.root)


def _stored_year(value: object) -> int | None:
    return int(str(value)) if value not in (None, "") else None


def _key(entry: Entry) -> tuple[str, str]:
    return (entry.department, STEM_OF[entry.role])


def _drop(files: Files, user_id: str) -> None:
    for records in files.values():
        for record in records:
            if record.get("user_id") == user_id:
                records.remove(record)
                return
    raise NotFound(user_id)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write(
    root: Path, key: tuple[str, str], records: list[dict[str, object]]
) -> None:
    """Rewrites one role file atomically (an empty one is removed)."""
    folder = root / key[0]
    path = folder / f"{key[1]}.json"
    if not records:
        path.unlink(missing_ok=True)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
        return
    folder.mkdir(mode=0o700, exist_ok=True)
    body = "[\n" + ",\n".join(f"  {json.dumps(r)}" for r in records) + "\n]\n"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(body)
    tmp.chmod(0o600)
    os.replace(tmp, path)
