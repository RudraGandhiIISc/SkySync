"""Splits pages into overlapping chunks on sentence boundaries.

Table rows (lines with `|` or tab separators) are not split unless they are
longer than MAX_CHUNK_CHARS. No chunk is longer than MAX_CHUNK_CHARS, so very
long unbroken text (tables, URLs, encoded blobs) is cut by characters; the
index stores text up to 65,535 characters. A chunk never crosses a page, so
every chunk keeps its page number for citations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from askiisc.common.targets import CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from askiisc.common.types import ChunkDraft, Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")

Span = tuple[int, int]
MAX_CHUNK_CHARS = 8000


class TokenCounter(Protocol):
    """Counts tokens in a piece of text."""

    def count(self, text: str) -> int:
        """Returns the number of tokens in `text`."""
        ...


class WhitespaceCounter:
    """Counts whitespace-separated words, a cheap stand-in for a tokenizer."""

    def count(self, text: str) -> int:
        """Returns the number of whitespace-separated words."""
        return len(text.split())


def chunk_pages(
    pages: Sequence[Page],
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    counter: TokenCounter | None = None,
) -> list[ChunkDraft]:
    """Chunks every page.

    Args:
        pages: Extracted pages.
        target_tokens: Maximum tokens per chunk, unless a single table row is
            longer (rows are never split).
        overlap: Tokens repeated from the end of one chunk at the start of
            the next chunk on the same page.
        counter: Token counter; whitespace words by default.

    Returns:
        Chunk drafts in page order.

    Raises:
        ValueError: If the sizes are inconsistent.
    """
    if target_tokens < 1 or not 0 <= overlap < target_tokens:
        raise ValueError("need target_tokens >= 1 and 0 <= overlap < target")
    count = counter or WhitespaceCounter()
    drafts: list[ChunkDraft] = []
    for page in pages:
        units = _units(page.text, target_tokens, count)
        for start, end in _pack(
            units, page.text, target_tokens, overlap, count
        ):
            drafts.append(
                ChunkDraft(
                    text=page.text[start:end],
                    page=page.number,
                    start=start,
                    end=end,
                )
            )
    return drafts


def _is_table_row(line: str) -> bool:
    return "|" in line or "\t" in line


def _units(text: str, target: int, count: TokenCounter) -> list[Span]:
    """Splits a page into atomic spans: table rows, sentences or words."""
    units: list[Span] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        start, end = offset, offset + len(line.rstrip("\n"))
        offset += len(line)
        if not text[start:end].strip():
            continue
        if _is_table_row(text[start:end]):
            units.extend(_cap((start, end)))
            continue
        for s_start, s_end in _sentences(text, start, end):
            if count.count(text[s_start:s_end]) <= target:
                units.extend(_cap((s_start, s_end)))
            else:
                for word in _words(text, s_start, s_end):
                    units.extend(_cap(word))
    return units


def _cap(span: Span) -> list[Span]:
    """Cuts a span longer than MAX_CHUNK_CHARS into character windows."""
    start, end = span
    return [
        (i, min(i + MAX_CHUNK_CHARS, end))
        for i in range(start, end, MAX_CHUNK_CHARS)
    ]


def _sentences(text: str, start: int, end: int) -> list[Span]:
    spans: list[Span] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return [s for s in spans if text[s[0] : s[1]].strip()]


def _words(text: str, start: int, end: int) -> list[Span]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text, start, end)]


def _pack(
    units: list[Span],
    text: str,
    target: int,
    overlap: int,
    count: TokenCounter,
) -> list[Span]:
    """Greedily packs units into chunks, with overlap between neighbours."""
    sizes = [count.count(text[s:e]) for s, e in units]
    chunks: list[Span] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (
            j == i
            or (
                total + sizes[j] <= target
                and units[j][1] - units[i][0] <= MAX_CHUNK_CHARS
            )
        ):
            total += sizes[j]
            j += 1
        chunks.append((units[i][0], units[j - 1][1]))
        if j == len(units):
            break
        k, back = j, 0
        while k - 1 > i and back + sizes[k - 1] <= overlap:
            back += sizes[k - 1]
            k -= 1
        i = k
    return chunks
