"""The access oracle: a deliberately naive reference checker.

It reads the same JSON shapes as the access service (directory records and
document policies) but shares no code with `askiisc.access`. Plain loops,
no caching, no indexes: independence is the point.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Record = Mapping[str, Any]


def _find(records: Iterable[Record], key: str, value: str) -> Record | None:
    for record in records:
        if record.get(key) == value:
            return record
    return None


def oracle_allows(
    directory: Sequence[Record],
    policies: Sequence[Record],
    user_id: str,
    doc_id: str,
    at: float,
) -> bool:
    """Decides access the slow, obvious way.

    Args:
        directory: Directory records (user_id, department, role, groups).
        policies: Policy records (doc_id and optional rule lists).
        user_id: The user.
        doc_id: The document.
        at: Decision time.

    Returns:
        True only if some rule allows and nothing denies.
    """
    person = _find(directory, "user_id", user_id)
    policy = _find(policies, "doc_id", doc_id)
    if person is None or policy is None:
        return False
    if user_id in list(policy.get("deny_users", [])):
        return False
    start = policy.get("valid_from")
    end = policy.get("valid_until")
    if start is not None and at < start:
        return False
    if end is not None and at >= end:
        return False
    if user_id in list(policy.get("allow_users", [])):
        return True
    for group in person.get("groups", []):
        if group in list(policy.get("groups", [])):
            return True
    department = person.get("department", "")
    if department and department in list(policy.get("departments", [])):
        return True
    role = person.get("role", "")
    return bool(role) and role in list(policy.get("roles", []))


@dataclass(frozen=True)
class LogEntry:
    """One logged retrieval.

    Attributes:
        user_id: Who asked.
        at: When.
        retrieved: (chunk_id, doc_id) pairs the system returned.
        denied_docs: Documents the system refused for this user.
    """

    user_id: str
    at: float
    retrieved: tuple[tuple[str, str], ...] = ()
    denied_docs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Violation:
    """A disagreement between the system and the oracle."""

    kind: str  # "false_allow" or "false_deny"
    user_id: str
    doc_id: str
    chunk_id: str | None
    at: float


def check_run(
    log: Iterable[LogEntry],
    directory: Sequence[Record],
    policies: Sequence[Record],
) -> list[Violation]:
    """Replays a run log against the oracle.

    Args:
        log: What the system returned and refused.
        directory: Directory records.
        policies: Policy records in force during the run.

    Returns:
        Every false allow (returned but forbidden) and false deny (refused
        but allowed).
    """
    violations = []
    for entry in log:
        for chunk_id, doc_id in entry.retrieved:
            if not oracle_allows(
                directory, policies, entry.user_id, doc_id, entry.at
            ):
                violations.append(
                    Violation(
                        "false_allow", entry.user_id, doc_id, chunk_id, entry.at
                    )
                )
        for doc_id in entry.denied_docs:
            if oracle_allows(
                directory, policies, entry.user_id, doc_id, entry.at
            ):
                violations.append(
                    Violation(
                        "false_deny", entry.user_id, doc_id, None, entry.at
                    )
                )
    return violations
