"""BM25 keyword scores over the rows of the dense index (NFR-7).

Vector search misses questions that hinge on exact words (a form number,
a course code, a hostel name). This index scores every row by BM25 so the
dense index can fuse both rankings. Row i here is row i of DenseIndex; the
permission and validity masks are applied by the caller, inside the search.

The postings are one CSR structure (numpy only): for term t, rows and term
frequencies live at offsets[t]:offsets[t + 1]. Rows added after a build go
to a small per-term overflow, so uploads need no rebuild.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

TOKEN = re.compile(r"[a-z0-9]+")
_STOP = """
a an the of to in for on and or is are was were be by with at as from that
this it its what which who whom how when where why do does did can i my me
we our you your there their they them will shall should would may might
must have has had not no any all each per about into than then so if such
also
"""
STOPWORDS = frozenset(_STOP.split())
K1 = 1.2
B = 0.75

Ints = npt.NDArray[np.int32]
Floats = npt.NDArray[np.float32]


def terms(text: str) -> list[str]:
    """Lower-cased word and number tokens without stop words."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class KeywordIndex:
    """BM25 scores for every row, built once and appended to after."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._offsets: Ints = np.zeros(1, np.int64)
        self._rows: Ints = np.zeros(0, np.int32)
        self._tf: Floats = np.zeros(0, np.float32)
        self._extra: dict[int, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._length_array: Floats | None = None
        self._total = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def _term(self, word: str) -> int:
        return self._vocab.setdefault(word, len(self._vocab))

    def build(self, texts: Sequence[str]) -> None:
        """Replaces the contents with one row per text."""
        self._vocab, self._extra, self._lengths = {}, {}, []
        self._length_array = None
        term_ids: list[int] = []
        rows: list[int] = []
        tfs: list[int] = []
        for row, text in enumerate(texts):
            words = terms(text)
            self._lengths.append(len(words))
            for word, count in Counter(words).items():
                term_ids.append(self._term(word))
                rows.append(row)
                tfs.append(count)
        order = np.argsort(np.asarray(term_ids, np.int32), kind="stable")
        self._rows = np.asarray(rows, np.int32)[order]
        self._tf = np.asarray(tfs, np.float32)[order]
        counts = np.bincount(
            np.asarray(term_ids, np.int64), minlength=len(self._vocab)
        )
        self._offsets = np.concatenate(([0], np.cumsum(counts)))
        self._total = sum(self._lengths)

    def put(self, row: int, text: str, old: str | None = None) -> None:
        """Adds row `row` (the next row), or replaces it when `old` is given.

        Raises:
            ValueError: If a new row is not the next one.
        """
        if old is None:
            if row != len(self._lengths):
                raise ValueError("rows must be added in order")
            self._lengths.append(0)
        else:
            self._drop(row, old)
        words = terms(text)
        self._lengths[row] = len(words)
        self._length_array = None
        self._total += len(words)
        for word, count in Counter(words).items():
            self._extra.setdefault(self._term(word), []).append((row, count))

    def _drop(self, row: int, old: str) -> None:
        for word in set(terms(old)):
            t = self._vocab.get(word)
            if t is None:
                continue
            if t + 1 < len(self._offsets):
                lo, hi = int(self._offsets[t]), int(self._offsets[t + 1])
                self._tf[lo:hi][self._rows[lo:hi] == row] = 0.0
            if t in self._extra:
                self._extra[t] = [p for p in self._extra[t] if p[0] != row]
        self._total -= self._lengths[row]

    def _postings(self, t: int) -> tuple[Ints, Floats]:
        rows, tfs = self._rows[:0], self._tf[:0]
        if t + 1 < len(self._offsets):
            lo, hi = int(self._offsets[t]), int(self._offsets[t + 1])
            rows, tfs = self._rows[lo:hi], self._tf[lo:hi]
        extra = self._extra.get(t)
        if extra:
            rows = np.concatenate(
                (rows, np.asarray([r for r, _ in extra], np.int32))
            )
            tfs = np.concatenate(
                (tfs, np.asarray([c for _, c in extra], np.float32))
            )
        return rows, tfs

    def scores(self, query: str, n: int) -> Floats:
        """BM25 score of each of the first n rows for the query."""
        out = np.zeros(n, np.float32)
        rows_total = len(self._lengths)
        if not rows_total or not n:
            return out
        if self._length_array is None or len(self._length_array) != rows_total:
            self._length_array = np.asarray(self._lengths, np.float32)
        lengths = self._length_array[:n]
        avg = max(self._total / rows_total, 1.0)
        norm = K1 * (1 - B + B * lengths / avg)
        for word in set(terms(query)):
            t = self._vocab.get(word)
            if t is None:
                continue
            rows, tfs = self._postings(t)
            keep = (rows < n) & (tfs > 0)
            rows, tfs = rows[keep], tfs[keep]
            df = len(rows)
            if not df:
                continue
            idf = math.log(1 + (rows_total - df + 0.5) / (df + 0.5))
            out[rows] += idf * tfs * (K1 + 1) / (tfs + norm[rows])
        return out

    def save(self, folder: Path) -> None:
        """Writes keyword.npz and keyword_vocab.txt into folder."""
        self._fold()
        np.savez(
            folder / "keyword.npz",
            offsets=self._offsets,
            rows=self._rows,
            tf=self._tf,
            lengths=np.asarray(self._lengths, np.int32),
        )
        words = sorted(self._vocab, key=self._vocab.__getitem__)
        (folder / "keyword_vocab.txt").write_text("\n".join(words))

    def _fold(self) -> None:
        """Merges the overflow into the CSR arrays."""
        if not self._extra and len(self._offsets) == len(self._vocab) + 1:
            return
        parts = [self._postings(t) for t in range(len(self._vocab))]
        self._extra = {}
        self._rows = np.concatenate([r for r, _ in parts] or [self._rows[:0]])
        self._tf = np.concatenate([f for _, f in parts] or [self._tf[:0]])
        counts = [len(r) for r, _ in parts]
        self._offsets = np.concatenate(([0], np.cumsum(counts))).astype(
            np.int64
        )

    @staticmethod
    def exists(folder: Path) -> bool:
        """Whether a saved index is in folder."""
        return (folder / "keyword.npz").exists()

    def load(self, folder: Path) -> None:
        """Replaces the contents with a saved index."""
        data: Any = np.load(folder / "keyword.npz")
        self._offsets = data["offsets"].astype(np.int64)
        self._rows = data["rows"].astype(np.int32)
        self._tf = data["tf"].astype(np.float32)
        self._lengths = data["lengths"].astype(np.int64).tolist()
        self._total = sum(self._lengths)
        self._length_array = None
        text = (folder / "keyword_vocab.txt").read_text()
        self._vocab = {w: i for i, w in enumerate(text.split("\n")) if w}
        self._extra = {}
