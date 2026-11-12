"""MinHash signatures for near-duplicate and similar-text detection.

A document's text becomes a set of word 5-grams; its signature is the
minimum of NUM_PERM hash permutations over that set. The fraction of equal
positions in two signatures estimates the Jaccard similarity of the sets.
Locality-sensitive hashing (bands of rows) finds candidate pairs without
comparing every pair.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

NUM_PERM = 128
SHINGLE_WORDS = 5
# Long PDFs: the first this many shingles are enough to compare.
MAX_SHINGLES = 200_000
LSH_BANDS = 16
_SEED = 20261026
_WORD = re.compile(r"[a-z0-9]+")
_MASK = np.uint64((1 << 61) - 1)

Signature = npt.NDArray[np.uint64]


def _permutations() -> tuple[Signature, Signature]:
    rng = np.random.default_rng(_SEED)
    a = rng.integers(1, 1 << 61, NUM_PERM, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 1 << 61, NUM_PERM, dtype=np.uint64)
    return a, b


_A, _B = _permutations()


def normalise_text(text: str) -> str:
    """Lower-case words joined by single spaces (for exact comparisons)."""
    return " ".join(_WORD.findall(text.lower()))


def shingle_hashes(text: str) -> npt.NDArray[np.uint64]:
    """64-bit hashes of the word 5-grams of a text."""
    words = _WORD.findall(text.lower())
    count = max(1, len(words) - SHINGLE_WORDS + 1)
    grams = (
        " ".join(words[i : i + SHINGLE_WORDS])
        for i in range(min(count, MAX_SHINGLES))
    )
    values = [
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest())
        for g in grams
    ]
    return np.unique(np.asarray(values, dtype=np.uint64))


def signature(text: str) -> Signature:
    """The MinHash signature of a text (NUM_PERM unsigned integers)."""
    hashes = shingle_hashes(text)
    with np.errstate(over="ignore"):
        mixed = (hashes[:, None] * _A[None, :] + _B[None, :]) & _MASK
    return np.asarray(mixed.min(axis=0), dtype=np.uint64)


def similarity(a: Signature, b: Signature) -> float:
    """Estimated Jaccard similarity of the two texts."""
    return float(np.mean(a == b))


def candidate_pairs(
    signatures: Sequence[Signature], bands: int = LSH_BANDS
) -> set[tuple[int, int]]:
    """Index pairs that share at least one LSH band (i < j).

    With 16 bands of 8 rows, pairs at similarity 0.9 are found with
    probability 1 - (1 - 0.9**8)**16, about 0.99988, and pairs below 0.5
    rarely.
    """
    rows = NUM_PERM // bands
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, sig in enumerate(signatures):
            buckets[sig[band * rows : (band + 1) * rows].tobytes()].append(i)
        for members in buckets.values():
            pairs.update(_all_pairs(members))
    return pairs


def _all_pairs(members: Iterable[int]) -> Iterable[tuple[int, int]]:
    ordered = sorted(members)
    for x, i in enumerate(ordered):
        for j in ordered[x + 1 :]:
            yield i, j
