"""NFR-7 adjudication: the review sheet and the final scores."""

import csv
import json
from pathlib import Path

import pytest
from eval.grader.adjudicate import (
    agreement,
    kappa,
    main,
    pick,
    score,
    sheet_rows,
    verdict,
    write_sheet,
)
from eval.versioned.grade import final


def row(i: int, rubric: bool, judge: bool | None) -> dict[str, object]:
    graded: dict[str, object] = {
        "id": f"q{i}", "type": "general", "abstained": False,
        "rubric_correct": rubric, "judge_correct": judge,
        "cites_support": True, "right_edition": None, "edition_ok": None,
        "retrieved": True, "wrong_cited": False, "any_stale_citation": False,
    }  # fmt: skip
    # As in a real graded.json: the grader's headline verdicts.
    graded["correct"] = final(graded)
    graded["strict_correct"] = final(graded, strict=True)
    return graded


GRADED = [row(0, True, False), row(1, False, True)] + [
    row(i, True, True) for i in range(2, 100)
]


def test_pick_takes_every_disagreement_and_a_seeded_sample() -> None:
    picked = pick(GRADED, 60, seed=252)
    assert [s for s, _ in picked[:2]] == ["disagreement", "disagreement"]
    audit = [r["id"] for s, r in picked if s == "audit"]
    assert len(audit) == 60  # noqa: PLR2004
    assert audit == [r["id"] for s, r in pick(GRADED, 60, 252) if s == "audit"]
    assert audit != [r["id"] for s, r in pick(GRADED, 60, 7) if s == "audit"]
    assert len(pick(GRADED[:3], 60, 1)) == 3  # noqa: PLR2004


def test_the_sheet_leaves_the_human_columns_empty(tmp_path: Path) -> None:
    answers = {"q0": {"question": "Q?", "reference_answer": "R", "answer": "A"}}
    rows = sheet_rows(pick(GRADED, 2, 1), answers, lambda item: "chunk text")
    assert rows[0]["question"] == "Q?" and rows[0]["cited_text"] == "chunk text"
    assert rows[1]["question"] == "" and rows[1]["cited_text"] == ""
    assert all(r["human_correct"] == r["human_citations"] == "" for r in rows)
    write_sheet(rows, tmp_path / "review.csv")
    with (tmp_path / "review.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4  # noqa: PLR2004


def test_verdict_cells() -> None:
    assert verdict(" Yes ") is True and verdict("no") is False
    assert verdict("") is None
    with pytest.raises(ValueError):
        verdict("maybe")


def test_kappa_and_agreement() -> None:
    assert kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert kappa([], []) is None
    assert kappa([True, True], [True, True]) is None  # no variation
    rows = [
        {"a": True, "b": True},
        {"a": False, "b": True},
        {"a": None, "b": 1},
    ]
    assert agreement(rows, "a", "b") == {
        "items": 2,
        "percent": 50.0,
        "kappa": 0.0,
    }
    assert agreement([], "a", "b")["percent"] is None


def test_human_verdicts_replace_the_judge_and_rescore() -> None:
    review = [
        {"id": "q0", "human_correct": "yes", "human_citations": ""},
        {"id": "q1", "human_correct": "no", "human_citations": "no"},
        {"id": "q2", "human_correct": "", "human_citations": ""},
    ]
    result = score(GRADED, review)
    assert result["before"]["accuracy"] == 0.99  # noqa: PLR2004 - q0 judge no
    assert result["after"]["accuracy"] == 0.99  # noqa: PLR2004 - q0 yes, q1 no
    assert result["after"]["citation_correctness"] == 0.99  # noqa: PLR2004
    assert result["human_verdicts"] == 2  # noqa: PLR2004
    assert result["agreement"]["judge_human"]["items"] == 2  # noqa: PLR2004
    assert result["agreement"]["judge_human"]["percent"] == 0.0


def test_score_command(tmp_path: Path) -> None:
    graded = tmp_path / "graded.json"
    graded.write_text(json.dumps(GRADED))
    sheet = tmp_path / "review.csv"
    write_sheet(
        [{"id": "q0", "human_correct": "yes", "human_citations": ""}], sheet
    )
    assert main(["score", str(graded), str(sheet), str(tmp_path / "o")]) == 0
    saved = json.loads((tmp_path / "o" / "adjudicated.json").read_text())
    assert saved["human_verdicts"] == 1
