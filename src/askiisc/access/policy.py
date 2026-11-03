"""The access policy engine (FR-3, NFR-10). Pure logic, no I/O.

Visibility: "public" documents are readable by every user in the IISc
directory (they are on the open web anyway); "restricted" documents (the
default) need a matching rule. Only directory users can reach the service
at all, so "public" never means anonymous.

Rules per document: allowed groups, departments and roles; an optional
validity window [valid_from, valid_until); per-user allow and deny lists.
Deny overrides allow. Unknown users, unknown documents, malformed policies
and any error deny (fail closed).

Who may change a policy: a document has an owner department and optional
co-owners; admins of any of them, and institute admins, manage it
(`may_manage_policy`). A change may add owners but may remove only
departments the admin manages (`may_change_owners`), so nobody takes a
document away from another department.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict

from askiisc.access.directory import Directory, Person
from askiisc.common.types import DocId, UserId

_LOG = logging.getLogger(__name__)


INSTITUTE = "institute"
"""Owner of institute-wide documents; its admins may manage any policy."""


def may_manage(person: Person | None, owner: str) -> bool:
    """Whether a person may change policies of documents owned by `owner`.

    Admins manage their own department's documents; institute admins
    manage every document.
    """
    if person is None or person.role != "admin":
        return False
    return person.department in {owner, INSTITUTE}


def may_manage_policy(person: Person | None, owners: frozenset[str]) -> bool:
    """Whether a person manages a document owned by any of `owners`."""
    return any(may_manage(person, owner) for owner in owners)


def may_change_owners(
    person: Person | None,
    before: frozenset[str] | None,
    after: frozenset[str],
) -> bool:
    """Whether a person may set a document's owners from `before` to `after`.

    Args:
        person: The admin making the change.
        before: The current owners, or None for a new document.
        after: The owners after the change (never empty).

    Returns:
        For a new document, whether they manage one of the new owners; for
        an existing one, whether they manage it now and every owner they
        remove.
    """
    if not after:
        return False
    if before is None:
        return may_manage_policy(person, after)
    return may_manage_policy(person, before) and all(
        may_manage(person, owner) for owner in before - after
    )


class DocPolicy(BaseModel):
    """Who may read one document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: DocId
    owner: str = INSTITUTE
    co_owners: frozenset[str] = frozenset()
    visibility: Literal["public", "restricted"] = "restricted"
    groups: frozenset[str] = frozenset()
    departments: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    valid_from: float | None = None
    valid_until: float | None = None
    allow_users: frozenset[UserId] = frozenset()
    deny_users: frozenset[UserId] = frozenset()

    @property
    def owners(self) -> frozenset[str]:
        """The owner and the co-owners: departments that manage it."""
        return self.co_owners | {self.owner}


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


@dataclass(frozen=True)
class CompactFilter:
    """A filter sent as "every plain public document, plus these".

    The plain public set (`PolicyEngine.public`) is the same for every
    directory user and changes only with the policy version, so clients
    fetch it once per version instead of with every request.

    Attributes:
        public: Whether the plain public set is included (False for an
            unknown user).
        extra: Readable documents outside the plain public set.
        policy_version: The version both parts were computed at.
    """

    public: bool
    extra: frozenset[DocId]
    policy_version: int


class PolicyStore:
    """Thread-safe policies with a version number bumped on every change.

    Copy-on-write: a change builds a new mapping, so a snapshot is O(1)
    (the real corpus has ~25k policies and every request takes a snapshot).
    """

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
            self._policies = {**self._policies, policy.doc_id: policy}
            self._version += 1
            return self._version

    def delete(self, doc_id: DocId) -> int:
        """Removes a policy (revokes all access); returns the new version."""
        with self._lock:
            self._policies = {
                d: p for d, p in self._policies.items() if d != doc_id
            }
            self._version += 1
            return self._version

    def snapshot(self) -> tuple[int, Mapping[DocId, DocPolicy]]:
        """Returns the version and a read-only view of every policy."""
        with self._lock:
            return self._version, MappingProxyType(self._policies)


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
    if policy.visibility == "public":
        return True, "public"
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
        self._split: (
            tuple[int, frozenset[DocId], tuple[tuple[DocId, DocPolicy], ...]]
            | None
        ) = None

    def _plain_and_special(
        self, version: int, policies: Mapping[DocId, DocPolicy]
    ) -> tuple[frozenset[DocId], tuple[tuple[DocId, DocPolicy], ...]]:
        """Splits policies once per version: plain public and the rest.

        Plain public documents (no deny list, no window) are readable by
        every known user; the rest are evaluated per user.
        """
        cached = self._split
        if cached is not None and cached[0] == version:
            return cached[1], cached[2]
        plain = frozenset(
            d
            for d, p in policies.items()
            if p.visibility == "public"
            and not p.deny_users
            and p.valid_from is None
            and p.valid_until is None
        )
        special = tuple((d, p) for d, p in policies.items() if d not in plain)
        self._split = (version, plain, special)
        return plain, special

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

    def public(self) -> AccessFilter:
        """The plain public documents: readable by every directory user."""
        version, policies = self._policies.snapshot()
        return AccessFilter(
            self._plain_and_special(version, policies)[0], version
        )

    def filter_compact(self, user_id: UserId, at: float) -> CompactFilter:
        """Like `filter`, without listing the plain public documents.

        Never raises: on any error nothing is readable.
        """
        version, policies = self._policies.snapshot()
        try:
            person = self._directory.get(user_id)
            if person is None:
                return CompactFilter(False, frozenset(), version)
            _, special = self._plain_and_special(version, policies)
            extra = frozenset(
                doc_id
                for doc_id, policy in special
                if evaluate(person, policy, at)[0]
            )
            return CompactFilter(True, extra, version)
        except Exception as err:  # fail closed on anything
            _LOG.warning("filter failed closed: %s", type(err).__name__)
            return CompactFilter(False, frozenset(), version)

    def filter(self, user_id: UserId, at: float) -> AccessFilter:
        """Lists every document the user may read at `at`.

        Never raises: on any error the filter is empty.
        """
        version, policies = self._policies.snapshot()
        try:
            person = self._directory.get(user_id)
            if person is None:
                return AccessFilter(frozenset(), version)
            plain, special = self._plain_and_special(version, policies)
            allowed = plain | {
                doc_id
                for doc_id, policy in special
                if evaluate(person, policy, at)[0]
            }
            return AccessFilter(allowed, version)
        except Exception as err:  # fail closed on anything
            _LOG.warning("filter failed closed: %s", type(err).__name__)
            return AccessFilter(frozenset(), version)
