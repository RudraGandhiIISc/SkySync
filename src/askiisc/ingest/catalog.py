"""The catalogue: duplicates removed, editions found, windows assigned.

Runs after the crawl (and the archive fetch), before ingestion:

  manifest.jsonl + archive.jsonl
      -> learn each site's template (menus, footers) from its HTML pages
         and remove it (ingest/boilerplate); boilerplate.json
      -> extract text once per document (process pool)
      -> drop exact and near duplicates (MinHash >= NEAR_DUPLICATE), keeping
         the other addresses as aliases of the kept copy
      -> group editions into families (same family key and similar text)
      -> order each family's editions and set valid_from / valid_to
      -> catalog.jsonl (what ingestion reads), duplicates.jsonl,
         catalog_report.json

Different editions are never merged as duplicates, however similar: the
2025 and 2026 handbooks differ in a few rules, and those rules are what a
question about one year needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from askiisc.common.errors import AskIIScError
from askiisc.ingest import boilerplate, minhash
from askiisc.ingest.archive import NOISE_WORDS
from askiisc.ingest.crawl import (
    PRIORITY_CATEGORIES,
    categorise,
    department_of,
    excluded,
)
from askiisc.ingest.editions import (
    Edition,
    Window,
    assign_windows,
    detect_edition,
    family_key,
)
from askiisc.ingest.extract import HTML, extract_text

LOG = logging.getLogger(__name__)
# Near copies: this similar and the same address pattern (re-uploads,
# http/www spellings, a file posted on several sites), or nearly identical
# wherever they are. Two tenders that differ in one word are two documents.
NEAR_DUPLICATE = 0.9
NEAR_DUPLICATE_ANYWHERE = 0.98
# Only academic documents have editions; a series of reports, tenders,
# talks or results is a set of separate documents, none replacing another.
SERIES_WORDS = (
    *NOISE_WORDS,
    "tender",
    "corrigendum",
    "quotation",
    "rfq",
    "nit-",
    "newsletter",
    "minutes",
    "result",
    "shortlist",
    "selected",
    "merit",
    "report",
)
# Editions of one document share most of their wording; unrelated files
# that happen to share a generic name do not.
FAMILY_MIN_SIMILARITY = 0.2
# A key shared by more documents than this is too generic to be one
# document's editions (pairwise comparison would also be slow).
MAX_FAMILY_CANDIDATES = 300
TITLE_CHARS = 160
HEAD_KEEP_CHARS = 4000
INSTITUTE_HOSTS = ("iisc.ac.in", "www.iisc.ac.in")
EXPLICIT_SOURCES = frozenset({"url", "title", "upload", "text"})


@dataclass
class Item:
    """One saved document while the catalogue is built."""

    entry: dict[str, object]
    text_hash: str = ""
    sig: minhash.Signature | None = None
    head: str = ""
    chars: int = 0
    title: str = ""
    edition: Edition = field(default_factory=lambda: Edition(None))
    aliases: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        """The document id."""
        return str(self.entry["doc_id"])

    @property
    def url(self) -> str:
        """The address a citation links to."""
        return str(self.entry["url"])

    @property
    def original(self) -> str:
        """The address on the IISc site (archive copies: the archived one)."""
        return str(self.entry.get("original_url") or self.entry["url"])

    @property
    def archived(self) -> bool:
        """Whether this is an archive snapshot."""
        return self.entry.get("snapshot") is not None

    @property
    def start(self) -> float:
        """Edition start; undated documents count as of their fetch."""
        if self.edition.start is not None:
            return self.edition.start
        return float(str(self.entry.get("fetched_at") or 0.0))


def load_entries(root: Path) -> list[dict[str, object]]:
    """Live and archived manifest entries (out-of-scope addresses dropped)."""
    rows: list[dict[str, object]] = []
    for name in ("manifest.jsonl", "archive.jsonl"):
        path = root / name
        if path.exists():
            rows.extend(json.loads(ln) for ln in path.open() if ln.strip())
    return [
        r for r in rows if not excluded(str(r.get("original_url") or r["url"]))
    ]


def page_shingles(args: tuple[str, str]) -> list[int] | None:
    """Template shingles of one HTML file (None: not HTML or no text)."""
    path, kind = args
    if kind != HTML:
        return None
    try:
        pages = extract_text(Path(path).read_bytes(), kind)
    except (AskIIScError, OSError, ValueError):
        return None
    return boilerplate.shingles("\n".join(p.text for p in pages))


def learn_templates(
    root: Path, entries: Sequence[dict[str, object]], workers: int
) -> dict[str, boilerplate.Template]:
    """Each design group's template, from its HTML pages (process pool)."""
    jobs = [
        (str(root / str(e["path"])), str(e["content_type"])) for e in entries
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        found = pool.map(page_shingles, jobs, chunksize=16)
        return boilerplate.learn(
            boilerplate.by_group(
                (e, s)
                for e, s in zip(entries, found, strict=True)
                if s is not None
            )
        )


def analyse(
    path: Path, kind: str, template: boilerplate.Template = frozenset()
) -> tuple[str, minhash.Signature, str, int] | None:
    """Text hash, signature, opening text and length of one file.

    Args:
        path: The saved file.
        kind: Its content type.
        template: Its site's template lines, removed first (HTML only).

    Returns:
        None when the file has no text (or only template text).
    """
    try:
        pages = extract_text(path.read_bytes(), kind)
    except (AskIIScError, OSError, ValueError):
        return None
    text = "\n".join(p.text for p in pages)
    if kind == HTML:
        text = boilerplate.strip(text, template)
    norm = minhash.normalise_text(text)
    if not norm:
        return None
    digest = hashlib.sha256(norm.encode()).hexdigest()
    return digest, minhash.signature(text), text[:HEAD_KEEP_CHARS], len(text)


def _analyse_entry(
    args: tuple[str, str, boilerplate.Template],
) -> tuple[str, minhash.Signature, str, int] | None:
    return analyse(Path(args[0]), args[1], args[2])


def _guess_title(item: Item) -> str:
    title = str(item.entry.get("title") or "")
    if title:
        return " ".join(title.split())[:TITLE_CHARS]
    for line in item.head.splitlines():
        words = line.split()
        if len(words) >= 2 and any(c.isalpha() for c in line):  # noqa: PLR2004
            return " ".join(words)[:TITLE_CHARS]
    name = urlparse(item.original).path.rstrip("/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0].replace("-", " ").replace("_", " ")


def analyse_all(
    root: Path,
    entries: Sequence[dict[str, object]],
    workers: int,
    templates: Mapping[str, boilerplate.Template] | None = None,
) -> list[Item]:
    """Extracts and fingerprints every entry in parallel.

    Args:
        root: The corpus folder.
        entries: Manifest entries.
        workers: Extraction processes.
        templates: Site templates by design group, removed before
            fingerprinting.

    Returns:
        Items with text (entries without text are left out).
    """
    templates = templates or {}
    jobs = [
        (
            str(root / str(e["path"])),
            str(e["content_type"]),
            templates.get(boilerplate.group_of(e), frozenset()),
        )
        for e in entries
    ]
    items: list[Item] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_analyse_entry, jobs, chunksize=16)
        for entry, result in zip(entries, results, strict=True):
            if result is None:
                continue
            digest, sig, head, chars = result
            item = Item(entry, digest, sig, head, chars)
            item.title = _guess_title(item)
            snapshot = entry.get("snapshot")
            item.edition = detect_edition(
                item.original,
                item.title,
                head,
                None if snapshot is None else float(str(snapshot)),
            )
            items.append(item)
    return items


def _explicit(item: Item) -> bool:
    return item.edition.source in EXPLICIT_SOURCES


def same_edition(a: Item, b: Item) -> bool:
    """Whether two documents can be copies of one edition."""
    if not (_explicit(a) and _explicit(b)):
        return True
    return (a.edition.start, a.edition.minor) == (
        b.edition.start,
        b.edition.minor,
    )


def keep_order(item: Item) -> tuple[object, ...]:
    """Sort key: the copy to keep comes first.

    Live before archived, newest edition, the institute site, a page over
    a file, then the shortest address.
    """
    host = urlparse(item.original).netloc.lower()
    return (
        item.archived,
        -item.start,
        -item.edition.minor,
        host not in INSTITUTE_HOSTS,
        item.entry.get("content_type") != HTML,
        len(item.original),
        item.original,
    )


class _Union:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())


def _sig(item: Item) -> minhash.Signature:
    if item.sig is None:
        raise AskIIScError(f"{item.doc_id}: no signature")
    return item.sig


def _key(item: Item) -> str:
    return family_key(item.original, _is_file(item))


def duplicate_groups(items: Sequence[Item]) -> list[list[int]]:
    """Groups of copies: same text, or near-same text of one edition."""
    union = _Union(len(items))
    keys = [_key(it) for it in items]
    by_hash: dict[str, int] = {}
    for i, item in enumerate(items):
        if item.text_hash in by_hash:
            union.join(i, by_hash[item.text_hash])
        else:
            by_hash[item.text_hash] = i
    sigs = [_sig(it) for it in items]
    for i, j in minhash.candidate_pairs(sigs):
        if union.find(i) == union.find(j) or not same_edition(
            items[i], items[j]
        ):
            continue
        sim = minhash.similarity(sigs[i], sigs[j])
        if sim >= NEAR_DUPLICATE_ANYWHERE or (
            sim >= NEAR_DUPLICATE and keys[i] == keys[j]
        ):
            union.join(i, j)
    return union.groups()


def remove_duplicates(
    items: Sequence[Item],
) -> tuple[list[Item], list[dict[str, str]]]:
    """Keeps one copy per duplicate group; the rest become its aliases.

    Returns:
        The kept items, and one record per dropped copy.
    """
    kept: list[Item] = []
    dropped: list[dict[str, str]] = []
    for group in duplicate_groups(items):
        ordered = sorted((items[i] for i in group), key=keep_order)
        head = ordered[0]
        for other in ordered[1:]:
            head.aliases.append(other.url)
            head.aliases.extend(other.aliases)
            dropped.append(
                {"doc_id": other.doc_id, "url": other.url, "kept": head.doc_id}
            )
        kept.append(head)
    return kept, dropped


def _is_file(item: Item) -> bool:
    return item.entry.get("content_type") != HTML


def versionable(item: Item) -> bool:
    """Whether a document is the kind that has yearly editions."""
    topic = categorise(item.original, item.title)
    text = f"{urlparse(item.original).path} {item.title}".lower()
    return topic in PRIORITY_CATEGORIES and not any(
        word in text for word in SERIES_WORDS
    )


def families(items: Sequence[Item]) -> list[list[Item]]:
    """Editions of one document: same family key and similar text."""
    by_key: dict[str, list[Item]] = defaultdict(list)
    out: list[list[Item]] = []
    for item in items:
        if versionable(item):
            by_key[_key(item)].append(item)
        else:
            out.append([item])
    for members in by_key.values():
        if len(members) > MAX_FAMILY_CANDIDATES:
            out.extend([m] for m in members)
            continue
        union = _Union(len(members))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = minhash.similarity(_sig(members[i]), _sig(members[j]))
                if sim >= FAMILY_MIN_SIMILARITY:
                    union.join(i, j)
        out.extend([members[i] for i in g] for g in union.groups())
    return out


def _family_id(members: Sequence[Item]) -> str:
    newest = sorted(members, key=keep_order)[0]
    return newest.doc_id


def catalogue_rows(groups: Iterable[Sequence[Item]]) -> list[dict[str, object]]:
    """One catalogue row per kept document, with its validity window."""
    rows: list[dict[str, object]] = []
    for members in groups:
        windows = assign_windows([(m.start, m.edition.minor) for m in members])
        family = _family_id(members)
        for item, window in zip(members, windows, strict=True):
            rows.append(_row(item, family, len(members), window))
    rows.sort(key=lambda r: str(r["doc_id"]))
    return rows


def _row(
    item: Item, family: str, editions: int, window: Window
) -> dict[str, object]:
    entry = item.entry
    host = urlparse(item.original).netloc.lower()
    return {
        "doc_id": item.doc_id,
        "path": entry["path"],
        "url": item.url,
        "original_url": item.original,
        "snapshot": entry.get("snapshot"),
        "aliases": sorted(set(item.aliases)),
        "title": item.title,
        "department": entry.get("department") or department_of(host),
        "category": entry.get("category", "general"),
        "content_type": entry["content_type"],
        "etag": entry.get("etag"),
        "sha256": entry.get("sha256"),
        "chars": item.chars,
        "family": family,
        "editions": editions,
        "edition": item.edition.label,
        "edition_source": item.edition.source,
        "edition_start": item.edition.start,
        "valid_from": window.valid_from,
        "valid_to": window.valid_to,
        "visibility": "public",
    }


def report(
    entries: int,
    items: Sequence[Item],
    rows: Sequence[dict[str, object]],
    dropped: Sequence[dict[str, str]],
) -> dict[str, object]:
    """Counts for catalog_report.json."""
    sizes = Counter(int(str(r["editions"])) for r in rows)
    family_count = len({r["family"] for r in rows})
    return {
        "entries": entries,
        "with_text": len(items),
        "duplicates_dropped": len(dropped),
        "documents": len(rows),
        "families": family_count,
        "families_with_several_editions": len(
            {r["family"] for r in rows if int(str(r["editions"])) > 1}
        ),
        "documents_in_multi_edition_families": sum(
            n for k, n in sizes.items() if k > 1
        ),
        "superseded": sum(r["valid_to"] is not None for r in rows),
        "edition_sources": dict(
            Counter(str(r["edition_source"] or "none") for r in rows)
        ),
        "archived": sum(r["snapshot"] is not None for r in rows),
    }


def build(
    root: Path, workers: int, out: Path | None = None
) -> dict[str, object]:
    """Builds catalog.jsonl, duplicates.jsonl and catalog_report.json.

    Args:
        root: The corpus folder.
        workers: Extraction processes.
        out: Where to write (default: the corpus folder).

    Returns:
        The report.
    """
    entries = load_entries(root)
    templates = learn_templates(root, entries, workers)
    LOG.info("templates for %d design groups", len(templates))
    items = analyse_all(root, entries, workers, templates)
    LOG.info("%d of %d documents have text", len(items), len(entries))
    kept, dropped = remove_duplicates(items)
    rows = catalogue_rows(families(kept))
    out = out or root
    out.mkdir(parents=True, exist_ok=True)
    boilerplate.save(out / "boilerplate.json", templates)
    _write_jsonl(out / "catalog.jsonl", rows)
    _write_jsonl(out / "duplicates.jsonl", dropped)
    summary = report(len(entries), items, rows, dropped)
    (out / "catalog_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, default=_json_default) + "\n")
    tmp.replace(path)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:  # pragma: no cover
    """Entry point: ASKIISC_CORPUS_DIR, workers from SLURM_CPUS_PER_TASK.

    ASKIISC_CATALOG_OUT writes elsewhere (a dry run on a live corpus).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = Path(os.environ["ASKIISC_CORPUS_DIR"])
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    out = os.environ.get("ASKIISC_CATALOG_OUT")
    summary = build(root, workers, Path(out) if out else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
