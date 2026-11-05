"""What replaced a superseded document version (FR-5).

A version stops being current in one of two ways: a revision of the same
document (version 2 replaces version 1), or a newer edition, which is a
document of its own in the same family (the 2026 handbook replaces the
2025 one; `ingest/editions.py`). `Lineage` names the replacement for a
citation's `superseded_by`: the next version of the same document if
there is one, else the current version of the family's next edition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from askiisc.common.errors import NotFound
from askiisc.common.types import DocId, DocRef, Version
from askiisc.ingest.versions import VersionStore


def next_editions(rows: Iterable[Mapping[str, Any]]) -> dict[DocId, DocId]:
    """Maps each edition to the one that follows it in its family.

    Args:
        rows: Catalogue rows with "doc_id", "family" and "valid_from".

    Returns:
        Edition to next edition, for every edition but a family's newest.
    """
    families: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        family, start = row.get("family"), row.get("valid_from")
        if family and start is not None:
            families[str(family)].append((float(start), str(row["doc_id"])))
    following: dict[DocId, DocId] = {}
    for editions in families.values():
        editions.sort()
        for (_, older), (_, newer) in zip(editions, editions[1:], strict=False):
            following[DocId(older)] = DocId(newer)
    return following


class Lineage:
    """Resolves the version that replaced a document version."""

    def __init__(
        self, versions: VersionStore, following: Mapping[DocId, DocId]
    ) -> None:
        """Creates the resolver.

        Args:
            versions: Every document's versions.
            following: Edition to next edition (`next_editions`).
        """
        self._versions = versions
        self._following = following

    def __call__(self, doc_id: DocId, version: Version) -> DocRef | None:
        """The replacement of `doc_id` version `version`, if any.

        Returns:
            The next version of the document, else the current version of
            its next edition, else None.
        """
        later = [
            r.version
            for r in self._versions.history(doc_id)
            if r.version > version
        ]
        if later:
            return DocRef(doc_id=doc_id, version=min(later))
        newer = self._following.get(doc_id)
        if newer is None:
            return None
        try:
            current = self._versions.current(newer)
        except NotFound:
            return None
        return DocRef(doc_id=newer, version=current.version)
