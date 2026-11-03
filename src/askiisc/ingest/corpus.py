"""The on-disk corpus: layout, access policies and a reconciler source.

corpus/
  public/<department>/<category>/<doc_id>.{html,pdf,txt}
      crawled public pages; visibility "public" (every IISc user)
  restricted/<department>/<doc_id>.{txt,html,pdf}
  restricted/<department>/<doc_id>.policy.json
      internal documents; visibility "restricted" plus at least one
      rule (groups, departments, roles or allow_users)
  manifest.jsonl     one line per public document (written by the crawl)
  archive/, archive.jsonl
                     past editions from the Internet Archive (ingest/archive)
  boilerplate.json   each site's template lines, removed from its HTML
                     pages before chunking (ingest/boilerplate)
  catalog.jsonl      the public documents after duplicates were removed and
                     editions found, with validity windows (ingest/catalog);
                     read instead of the manifests when present
  policies.json      every document's DocPolicy (written by build)
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from askiisc.access.policy import DocPolicy
from askiisc.common.errors import ConfigError, NotFound
from askiisc.common.types import DocId
from askiisc.ingest import boilerplate
from askiisc.ingest.crawl import excluded
from askiisc.ingest.extract import DOCX, HTML, PDF, PPTX, TEXT, XLSX
from askiisc.ingest.reconciler import SourceDocument

CONTENT_TYPES = {
    ".html": HTML,
    ".pdf": PDF,
    ".txt": TEXT,
    ".docx": DOCX,
    ".pptx": PPTX,
    ".xlsx": XLSX,
}
RULE_FIELDS = ("groups", "departments", "roles", "allow_users")


@dataclass(frozen=True)
class CorpusDoc:
    """One document on disk and its policy."""

    path: Path
    policy: DocPolicy
    etag: str | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    context: str = ""
    template: boilerplate.Template = frozenset()

    @property
    def content_type(self) -> str:
        """MIME type from the file suffix."""
        return CONTENT_TYPES[self.path.suffix]


def public_rows(root: Path) -> list[dict[str, Any]]:
    """The public documents: catalog.jsonl if built, else manifest.jsonl."""
    for name in ("catalog.jsonl", "manifest.jsonl"):
        path = root / name
        if path.exists():
            return [json.loads(ln) for ln in path.open() if ln.strip()]
    return []


def context_line(row: dict[str, Any]) -> str:
    """Title, edition and site of a document, for its embeddings."""
    site = urlparse(str(row.get("original_url") or row["url"])).netloc
    parts = [str(row.get("title") or ""), site]
    if row.get("edition") and int(row.get("editions") or 1) > 1:
        parts.insert(1, f"edition {row['edition']}")
    return " | ".join(p for p in parts if p)


def _public(root: Path) -> list[CorpusDoc]:
    templates = boilerplate.load(root / "boilerplate.json")
    docs = []
    for entry in public_rows(root):
        original = entry.get("original_url") or entry.get("url", "")
        if excluded(original):
            continue  # saved before the address was excluded
        policy = DocPolicy(
            doc_id=DocId(entry["doc_id"]),
            owner=entry.get("department", "institute"),
            visibility="public",
        )
        docs.append(
            CorpusDoc(
                root / entry["path"],
                policy,
                entry.get("etag"),
                entry.get("valid_from"),
                entry.get("valid_to"),
                context_line(entry) if "family" in entry else "",
                templates.get(boilerplate.group_of(entry), frozenset()),
            )
        )
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
                doc_id=d.policy.doc_id,
                content_type=d.content_type,
                etag=d.etag,
                valid_from=d.valid_from,
                valid_to=d.valid_to,
                context=d.context,
                template=d.template,
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


_TITLE = re.compile(rb"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TITLE_SCAN_BYTES = 65_536
MAX_TITLE_CHARS = 160


def html_title(data: bytes) -> str:
    """The <title> of an HTML page (first 64 KB), cleaned, or empty."""
    found = _TITLE.search(data[:TITLE_SCAN_BYTES])
    if not found:
        return ""
    text = html.unescape(found.group(1).decode("utf-8", "replace"))
    return " ".join(text.split())[:MAX_TITLE_CHARS]


def write_titles(root: Path) -> int:
    """Writes titles.json: catalogue titles, else saved HTML page titles.

    Returns:
        The number of titles found.
    """
    titles: dict[str, str] = {}
    for entry in public_rows(root):
        path = root / entry["path"]
        if "family" in entry and entry.get("title"):  # catalogue: any kind
            titles[entry["doc_id"]] = entry["title"]
        elif entry.get("content_type") == "text/html" and path.exists():
            title = entry.get("title") or html_title(path.read_bytes())
            if title:
                titles[entry["doc_id"]] = title
    (root / "titles.json").write_text(json.dumps(titles, indent=0))
    return len(titles)


def read_titles(root: Path) -> dict[str, str]:
    """titles.json, or empty if it is missing or broken."""
    try:
        data = json.loads((root / "titles.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:  # pragma: no cover
    """Builds policies.json and titles.json for ASKIISC_CORPUS_DIR."""
    import os  # noqa: PLC0415

    root = Path(os.environ["ASKIISC_CORPUS_DIR"])
    docs = load_corpus(root)
    write_policies(root, docs)
    titled = write_titles(root)
    public = sum(d.policy.visibility == "public" for d in docs)
    print(
        json.dumps(
            {
                "documents": len(docs),
                "public": public,
                "restricted": len(docs) - public,
                "titles": titled,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
