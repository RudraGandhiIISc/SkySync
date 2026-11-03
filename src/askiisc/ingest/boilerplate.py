"""Site templates: menus, sidebars and footers repeated across a site.

Tag and class rules (ingest/extract) remove most chrome, but many IISc
themes render their menus as plain lists inside the content region, so
every page of www.physics.iisc.ac.in starts with the same eighty menu
lines. Those lines drown the page's own text in search, and they change
when a site is redesigned, which makes two editions look different where
nothing real changed.

A template is learnt per group of pages that share one design: the live
pages of a host, or the archive snapshots of a host from one year (old
designs). A window of SHINGLE_LINES consecutive lines that appears on at
least MIN_PAGES pages and on MIN_SHARE of the group is template text, and
every line inside such a window is removed. Three-line windows keep the
real content that repeats single lines ("Professor", "Credits: 3:0"),
because the lines around them differ from page to page.

Only HTML pages are learnt from and cleaned; PDFs and Office files have no
site menus. Hashes are stable across processes (blake2b), so a template
learnt in one job can clean pages in another.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

SHINGLE_LINES = 3
MIN_PAGES = 4
MIN_SHARE = 0.02
_SPACE = re.compile(r"\s+")

Template = frozenset[int]


def group_of(entry: Mapping[str, object]) -> str:
    """The design group of a manifest or catalogue entry.

    Returns:
        "host" for a live page, "host@year" for an archive snapshot.
    """
    url = entry.get("original_url") or entry.get("url") or ""
    host = urlparse(str(url)).netloc
    snapshot = entry.get("snapshot")
    if snapshot is None:
        return host
    year = datetime.fromtimestamp(float(str(snapshot)), UTC).year
    return f"{host}@{year}"


def _norm(line: str) -> str:
    return _SPACE.sub(" ", line).strip().lower()


def _hash(lines: Iterable[str]) -> int:
    digest = hashlib.blake2b("\n".join(lines).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def shingles(text: str) -> list[int]:
    """Hashes of every window of SHINGLE_LINES non-empty lines, in order."""
    lines = [n for n in map(_norm, text.splitlines()) if n]
    return [
        _hash(lines[i : i + SHINGLE_LINES])
        for i in range(len(lines) - SHINGLE_LINES + 1)
    ]


def learn(pages: Mapping[str, Iterable[Iterable[int]]]) -> dict[str, Template]:
    """Finds each group's template from its pages' shingles.

    Args:
        pages: For each group, the shingle hashes of each of its pages.

    Returns:
        For each group with template text, the template shingle hashes.
    """
    templates: dict[str, Template] = {}
    for group, members in pages.items():
        counts: Counter[int] = Counter()
        total = 0
        for page in members:
            counts.update(set(page))
            total += 1
        need = max(MIN_PAGES, MIN_SHARE * total)
        found = frozenset(h for h, n in counts.items() if n >= need)
        if found:
            templates[group] = found
    return templates


def strip(text: str, template: Template) -> str:
    """Removes every line covered by a template window.

    Args:
        text: Extracted page text, one block per line.
        template: The group's template hashes (empty: text unchanged).

    Returns:
        The text without template lines; blank lines are dropped.
    """
    if not template:
        return text
    kept = [(line, _norm(line)) for line in text.splitlines()]
    kept = [(line, n) for line, n in kept if n]
    drop = [False] * len(kept)
    norms = [n for _, n in kept]
    for i in range(len(kept) - SHINGLE_LINES + 1):
        if _hash(norms[i : i + SHINGLE_LINES]) in template:
            drop[i : i + SHINGLE_LINES] = [True] * SHINGLE_LINES
    return "\n".join(
        line for (line, _), d in zip(kept, drop, strict=True) if not d
    )


def save(path: Path, templates: Mapping[str, Template]) -> None:
    """Writes templates as JSON (group -> sorted hashes)."""
    rows = {g: sorted(t) for g, t in sorted(templates.items())}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)


def load(path: Path) -> dict[str, Template]:
    """Reads templates written by save (none if the file is missing)."""
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {g: frozenset(int(h) for h in hs) for g, hs in rows.items()}


def by_group(
    entries: Iterable[tuple[Mapping[str, object], list[int]]],
) -> dict[str, list[list[int]]]:
    """Groups pages' shingles by design group."""
    grouped: dict[str, list[list[int]]] = defaultdict(list)
    for entry, page in entries:
        grouped[group_of(entry)].append(page)
    return grouped
