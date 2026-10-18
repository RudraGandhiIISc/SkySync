"""The access policy engine (FR-3, NFR-10). Pure logic, no I/O.

Rules per document: allowed groups, departments and roles; an optional
validity window [valid_from, valid_until); per-user allow and deny lists.
Deny overrides allow. Unknown users, unknown documents, malformed policies
and any error deny (fail closed).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from askiisc.access.directory import Directory, Person
from askiisc.common.types import DocId, UserId

_LOG = logging.getLogger(__name__)


class DocPolicy(BaseModel):
    """Who may read one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: DocId
    groups: frozenset[str] = frozenset()
    departments: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    valid_from: float | None = None
    valid_until: float | None = None
    allow_users: frozenset[UserId] = frozenset()
    deny_users: frozenset[UserId] = frozenset()


@dataclass(frozen=True)
class Decision:
    """An access decision and the policy version it was made under."""

    allowed: bool
    policy_version: int
    reason: str


@dataclass(frozen=True)
class AccessFilter:
    """Backend-neutral search filter: the documents a user may read."""

    doc_ids: frozenset[DocId]
    policy_version: int


class PolicyStore:
    """Thread-safe policies with a version number bumped on every change."""

    def __init__(self, policies: list[DocPolicy] | None = None) -> None:
        self._policies: dict[DocId, DocPolicy] = {
            p.doc_id: p for p in policies or []
        }
        self._version = 1
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        """The current policy version."""
        with self._lock:
            return self._version

    def put(self, policy: DocPolicy) -> int:
        """Adds or replaces a document's policy; returns the new version."""
        with self._lock:
            self._policies[policy.doc_id] = policy
            self._version += 1
            return self._version

    def delete(self, doc_id: DocId) -> int:
        """Removes a policy (revokes all access); returns the new version."""
        with self._lock:
            self._policies.pop(doc_id, None)
            self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, dict[DocId, DocPolicy]]:
        """Returns the version and a copy of every policy, consistently."""
        with self._lock:
            return self._version, dict(self._policies)


def _in_window(policy: DocPolicy, at: float) -> bool:
    if policy.valid_from is not None and at < policy.valid_from:
        return False
    return policy.valid_until is None or at < policy.valid_until


def evaluate(person: Person, policy: DocPolicy, at: float) -> tuple[bool, str]:
    """Applies one policy to one person.

    Args:
        person: The directory entry.
        policy: The document policy.
        at: Decision time.

    Returns:
        The decision and a short reason.
    """
    if person.user_id in policy.deny_users:
        return False, "user denied"
    if not _in_window(policy, at):
        return False, "outside validity window"
    if person.user_id in policy.allow_users:
        return True, "user allowed"
    if person.groups & policy.groups:
        return True, "group"
    if person.department and person.department in policy.departments:
        return True, "department"
    if person.role and person.role in policy.roles:
        return True, "role"
    return False, "no rule matches"


class PolicyEngine:
    """Answers authorize and filter questions from a directory and policies."""

    def __init__(self, directory: Directory, policies: PolicyStore) -> None:
        self._directory = directory
        self._policies = policies

    def authorize(self, user_id: UserId, doc_id: DocId, at: float) -> Decision:
        """Decides whether a user may read a document.

        Never raises: any error is a deny.
        """
        version, policies = self._policies.snapshot()
        try:
            person = self._directory.get(user_id)
            policy = policies.get(doc_id)
            if person is None:
                return Decision(False, version, "unknown user")
            if policy is None:
                return Decision(False, version, "unknown document")
            allowed, reason = evaluate(person, policy, at)
            return Decision(allowed, version, reason)
        except Exception as err:  # fail closed on anything
            _LOG.warning("authorize failed closed: %s", type(err).__name__)
            return Decision(False, version, "error")

    def filter(self, user_id: UserId, at: float) -> AccessFilter:
        """Lists every document the user may read at `at`.

        Never raises: on any error the filter is empty.
        """
        version, policies = self._policies.snapshot()
        try:
            person = self._directory.get(user_id)
            if person is None:
                return AccessFilter(frozenset(), version)
            allowed = frozenset(
                doc_id
                for doc_id, policy in policies.items()
                if evaluate(person, policy, at)[0]
            )
            return AccessFilter(allowed, version)
        except Exception as err:  # fail closed on anything
            _LOG.warning("filter failed closed: %s", type(err).__name__)
            return AccessFilter(frozenset(), version)
