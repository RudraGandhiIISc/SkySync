"""BM25 keyword index (NFR-7 hybrid search)."""

import math
from pathlib import Path

import numpy as np
import pytest

from askiisc.retrieval.keyword import KeywordIndex, terms

TEXTS = [
    "Hostel fees are paid each term through the fee portal.",
    "The library is open until midnight in the exam week.",
    "Form PG-12 is needed for a change of guide.",
    "Hostel allotment for new students happens in August.",
]


def _reference(texts: list[str], query: str) -> list[float]:
    """BM25 written out the slow way."""
    docs = [terms(t) for t in texts]
    avg = sum(map(len, docs)) / len(docs)
    out = []
    for d in docs:
        s = 0.0
        for q in set(terms(query)):
            df = sum(q in x for x in docs)
            if not df or q not in d:
                continue
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = d.count(q)
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / avg))
        out.append(s)
    return out


def test_terms_drop_stop_words_and_case() -> None:
    assert terms("What is the PG-12 form?") == ["pg", "12", "form"]


@pytest.mark.parametrize("query", ["hostel fee", "form pg-12", "nothing here"])
def test_scores_match_the_formula(query: str) -> None:
    index = KeywordIndex()
    index.build(TEXTS)
    got = index.scores(query, len(TEXTS))
    assert got.tolist() == pytest.approx(_reference(TEXTS, query), rel=1e-5)


def test_appended_and_replaced_rows_score_like_a_rebuild(
    tmp_path: Path,
) -> None:
    index = KeywordIndex()
    index.build(TEXTS[:2])
    index.put(2, "an old text about hostel")
    index.put(2, TEXTS[2], old="an old text about hostel")
    index.put(3, TEXTS[3])
    fresh = KeywordIndex()
    fresh.build(TEXTS)
    for query in ("hostel", "pg-12 guide", "old text"):
        want = fresh.scores(query, 4)
        assert index.scores(query, 4) == pytest.approx(want, rel=1e-5)
    index.save(tmp_path)
    loaded = KeywordIndex()
    assert KeywordIndex.exists(tmp_path)
    loaded.load(tmp_path)
    assert len(loaded) == 4
    want = fresh.scores("hostel fee", 4)
    assert loaded.scores("hostel fee", 4) == pytest.approx(want, rel=1e-5)
    with pytest.raises(ValueError, match="in order"):
        index.put(9, "x")


def test_empty_and_short_views() -> None:
    index = KeywordIndex()
    assert not index.scores("hostel", 3).any()
    index.build(TEXTS)
    assert index.scores("hostel", 2).shape == (2,)
    assert np.count_nonzero(index.scores("hostel", 4)) == 2
    empty = KeywordIndex()
    empty.build([])
    assert len(empty) == 0
