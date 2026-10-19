"""The on-disk corpus: layout, access policies and a reconciler source.

corpus/
  public/<department>/<category>/<doc_id>.{html,pdf,txt}
      crawled public pages; visibility "public" (every IISc user)
  restricted/<department>/<doc_id>.{txt,html,pdf}
  restricted/<department>/<doc_id>.policy.json
      internal documents; visibility "restricted" plus at least one
      rule (groups, departments, roles or allow_users)
  manifest.jsonl     one line per public document (written by the crawl)
  policies.json      every document's DocPolicy (written by build)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from askiisc.access.policy import DocPolicy
from askiisc.common.errors import ConfigError, NotFound
from askiisc.common.types import DocId
from askiisc.ingest.reconciler import SourceDocument

CONTENT_TYPES = {
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
}
RULE_FIELDS = ("groups", "departments", "roles", "allow_users")


@dataclass(frozen=True)
class CorpusDoc:
    """One document on disk and its policy."""

    path: Path
    policy: DocPolicy
    etag: str | None = None

    @property
    def content_type(self) -> str:
        """MIME type from the file suffix."""
        return CONTENT_TYPES[self.path.suffix]


def _public(root: Path) -> list[CorpusDoc]:
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return []
    docs = []
    for line in manifest.read_text().splitlines():
        entry = json.loads(line)
        policy = DocPolicy(
            doc_id=DocId(entry["doc_id"]),
            owner=entry.get("department", "institute"),
            visibility="public",
        )
        docs.append(CorpusDoc(root / entry["path"], policy, entry.get("etag")))
    return docs


def _restricted(root: Path) -> list[CorpusDoc]:
    docs = []
    for rules_path in sorted((root / "restricted").glob("*/*.policy.json")):
        doc_id = rules_path.name.removesuffix(".policy.json")
        rules = json.loads(rules_path.read_text())
        if not any(rules.get(f) for f in RULE_FIELDS):
            raise ConfigError(f"{doc_id}: restricted document needs a rule")
        if {"visibility", "doc_id", "owner"} & set(rules):
            raise ConfigError(
                f"{doc_id}: id, owner and visibility come from the path"
            )
        body = [
            p
            for p in rules_path.parent.glob(f"{doc_id}.*")
            if p.suffix in CONTENT_TYPES
        ]
        if len(body) != 1:
            raise ConfigError(f"{doc_id}: expected one document file")
        policy = DocPolicy(
            doc_id=DocId(doc_id),
            owner=rules_path.parent.name,
            visibility="restricted",
            **rules,
        )
        docs.append(CorpusDoc(body[0], policy))
    return docs


def load_corpus(root: Path) -> list[CorpusDoc]:
    """Reads every document and its policy.

    Raises:
        ConfigError: On a duplicate id, a restricted document with no rule,
            or a malformed policy file.
    """
    try:
        docs = _public(root) + _restricted(root)
    except (ValueError, KeyError, TypeError) as err:
        raise ConfigError(f"malformed corpus: {err}") from err
    ids = [d.policy.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate document ids in corpus")
    return docs


def write_policies(root: Path, docs: Sequence[CorpusDoc]) -> Path:
    """Writes policies.json for the access service."""
    path = root / "policies.json"
    rows = [
        d.policy.model_dump(mode="json", exclude_defaults=True)
        | {"visibility": d.policy.visibility}
        for d in docs
    ]
    path.write_text(json.dumps(rows, indent=1, sort_keys=True))
    return path


class CorpusSource:
    """A reconciler Source over the on-disk corpus."""

    def __init__(self, docs: Sequence[CorpusDoc]) -> None:
        self._docs = {d.policy.doc_id: d for d in docs}

    async def list_documents(self) -> Sequence[SourceDocument]:
        """Lists every document with its ETag (crawled ones only)."""
        return [
            SourceDocument(
                doc_id=d.policy.doc_id, content_type=d.content_type, etag=d.etag
            )
            for d in self._docs.values()
        ]

    async def fetch(self, doc_id: DocId) -> bytes:
        """Reads a document's bytes.

        Raises:
            NotFound: If the id is unknown or the file is gone.
        """
        doc = self._docs.get(doc_id)
        if doc is None or not doc.path.is_file():
            raise NotFound(doc_id)
        return doc.path.read_bytes()


def main() -> None:  # pragma: no cover
    """Builds policies.json for ASKIISC_CORPUS_DIR and prints counts."""
    import os  # noqa: PLC0415

    root = Path(os.environ["ASKIISC_CORPUS_DIR"])
    docs = load_corpus(root)
    write_policies(root, docs)
    public = sum(d.policy.visibility == "public" for d in docs)
    print(
        json.dumps(
            {
                "documents": len(docs),
                "public": public,
                "restricted": len(docs) - public,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
