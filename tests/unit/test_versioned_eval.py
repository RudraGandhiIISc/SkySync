import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from eval.versioned import generate as gen
from eval.versioned import grade

AUG_2025 = datetime(2025, 8, 1, tzinfo=UTC).timestamp()
AUG_2026 = datetime(2026, 8, 1, tzinfo=UTC).timestamp()
NOW = datetime(2026, 10, 26, tzinfo=UTC).timestamp()


def row(doc_id: str, start: float, end: float | None, **extra: object) -> dict:
    return {
        "doc_id": doc_id,
        "family": "ds",
        "editions": 2,
        "valid_from": start,
        "valid_to": end,
        "edition": extra.pop("edition", ""),
        "edition_start": extra.pop("edition_start", None),
        **extra,
    }


def test_pairs_and_academic_years() -> None:
    catalog = {
        "old": row("old", 0.0, AUG_2026, edition_start=AUG_2025),
        "new": row("new", AUG_2026, None, edition_start=AUG_2026),
        "solo": {**row("solo", 0.0, None), "editions": 1, "family": "solo"},
    }
    pairs = gen.edition_pairs(catalog)
    assert [(n["doc_id"], o["doc_id"]) for n, o in pairs] == [("new", "old")]
    assert gen.academic_year(catalog["old"], NOW) == 2025  # noqa: PLR2004
    assert gen.academic_year(catalog["new"], NOW) == 2026  # noqa: PLR2004
    # A January 2021 offering belongs to 2020-21 and is seen at its end.
    jan = datetime(2021, 1, 1, tzinfo=UTC).timestamp()
    feb = datetime(2022, 1, 1, tzinfo=UTC).timestamp()
    assert gen.academic_year(row("j", jan, feb, edition_start=jan), NOW) == (
        2020  # noqa: PLR2004
    )
    # Replaced within its own year before either probe: no year question.
    short = row("s", jan, jan + 86400, edition_start=jan)
    assert gen.academic_year(short, NOW) is None
    assert gen.academic_year(row("u", 0.0, None), NOW) is None
    assert gen.ay_label(2025) == "2025-26"


def test_alignment_keeps_changed_chunks() -> None:
    base = (
        "Students must earn 64 credits including core courses in data systems. "
        * 4
    )
    new = [
        gen.Piece("n#0", base.replace("64", "72")),
        gen.Piece("n#1", "short"),
    ]
    old = [
        gen.Piece("o#0", base),
        gen.Piece("o#1", "Hostel rules and mess timings. " * 10),
    ]
    pairs = gen.align(new, old)
    assert [(a.chunk_id, b.chunk_id) for a, b, _ in pairs] == [("n#0", "o#0")]
    assert gen.align([gen.Piece("n", base)], [gen.Piece("o", base)]) == []
    assert gen.jaccard(set(), set()) == 0.0


def test_changed_fact_validation() -> None:
    new = gen.Piece(
        "n", "The M.Tech CDS programme requires 72 credits in total."
    )
    old = gen.Piece(
        "o", "The M.Tech CDS programme requires 64 credits in total."
    )
    good = {
        "question": "How many credits does the M.Tech CDS programme require?",
        "new_value": "72 credits",
        "old_value": "64 credits",
    }
    assert gen.valid_change(good, new, old, new.text, old.text)
    with_year = {
        **good,
        "question": good["question"].replace("require", "require in 2025"),
    }
    assert not gen.valid_change(with_year, new, old, new.text, old.text)
    same = {**good, "old_value": "72 credits"}
    assert not gen.valid_change(same, new, old, new.text, old.text)
    # The "new" value also appears in the old edition: not a change.
    assert not gen.valid_change(
        good, new, old, new.text, old.text + " 72 credits"
    )
    assert not gen.valid_change({"question": None}, new, old, "", "")
    titled = {
        **good,
        "new_value": "Dr. P. K. Yalavarthy",
        "old_value": "P K Yalavarthy",
    }
    assert gen.change_problem(titled, new, old, "", "") == "same value"
    vague = {**good, "question": "How many credits does the course need?"}
    row = {"title": "M.Tech CDS curriculum", "original_url": "https://x/mtech/"}
    assert gen.change_problem(vague, new, old, new.text, old.text, row) == (
        "subject not named"
    )


def test_one_change_yields_four_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKIISC_LLM_MODEL", "qwen3-8b")
    newer = row(
        "new",
        AUG_2026,
        None,
        edition="Aug 2026",
        edition_start=AUG_2026,
        title="DS252: Introduction to Cloud Computing",
        original_url="https://cds.iisc.ac.in/courses/ds252/",
    )
    older = row(
        "old", 0.0, AUG_2026, edition="Aug 2025", edition_start=AUG_2025
    )
    found = {
        "question": "How many credits does DS252 carry?",
        "topic": "the DS252 credits",
        "new_value": "4:0",
        "old_value": "3:1",
        "new_chunk": "new@v1#0",
        "old_chunk": "old@v1#2",
    }
    items = {i["type"]: i for i in gen.change_items(found, newer, older, NOW)}
    assert set(items) == {"current", "year", "year-current", "change"}
    assert items["current"]["question"] == found["question"]
    assert items["current"]["expected_docs"] == ["new"]
    assert items["year"]["question"].startswith(
        "For the 2025-26 academic year, how many"
    )
    assert items["year"]["expected_docs"] == ["old"] and items["year"][
        "key_facts"
    ] == ["3:1"]
    assert items["year-current"]["question"].startswith("In 2026-27, how")
    assert items["change"]["question"] == (
        "How did the DS252 credits change between 2025-26 and 2026-27?"
    )
    assert items["change"]["key_facts"] == ["3:1", "4:0"]
    retired = row("new", AUG_2026, NOW - 1, edition_start=AUG_2026)
    assert {
        i["type"] for i in gen.change_items(found, retired, older, NOW)
    } == {"year", "change"}


def test_pick_fills_quotas_with_general(tmp_path: Path) -> None:
    heldout = tmp_path / "h.jsonl"
    heldout.write_text(
        json.dumps({"category": "unanswerable", "question": "IIT fee?"}) + "\n"
    )
    unanswerable = gen.unanswerable_items(heldout)
    assert unanswerable[0]["reviewed"] and not unanswerable[-1]["reviewed"]
    items = (
        [{"type": "current"}] * 5
        + [{"type": "year"}] * 3
        + [{"type": "year-current"}] * 2
        + [{"type": "change"}]
        + [{"type": "general"}] * 500
        + unanswerable
    )
    chosen = gen.pick([dict(i) for i in items], random.Random(1))
    assert len(chosen) == sum(gen.QUOTAS.values())
    types = [c["type"] for c in chosen]
    assert types.count("current") == 5 and types.count("year") == 5  # noqa: PLR2004
    assert chosen[0]["id"] == "v001" and len({c["id"] for c in chosen}) == 400  # noqa: PLR2004


def answer(kind: str, cited: list[str], text: str, **extra: object) -> dict:
    return {
        "id": extra.pop("id", "x"),
        "type": kind,
        "answer": text,
        "abstained": text == "I don't know based on the provided documents.",
        "citations": [
            {"doc_id": d, "chunk_id": f"{d}@v1#0", "stale": d == "old"}
            for d in cited
        ],
        "retrieved_docs": extra.pop("retrieved", cited),
        "key_facts": extra.pop("facts", ["72 credits"]),
        "expected_docs": extra.pop("expected", ["new"]),
        "wrong_docs": extra.pop("wrong", ["old"]),
        **extra,
    }


def test_edition_checks() -> None:
    assert grade.right_edition(answer("current", ["new"], "72 credits"))
    assert not grade.right_edition(
        answer("current", ["new", "old"], "72 credits")
    )
    assert not grade.right_edition(answer("year", ["x"], "72"))
    change = answer(
        "change",
        ["old", "new"],
        "64 then 72",
        expected=["old", "new"],
        wrong=[],
    )
    assert grade.right_edition(change)
    assert not grade.right_edition(
        {**change, "citations": change["citations"][1:]}
    )
    assert (
        grade.right_edition(answer("general", ["a"], "x", expected=[])) is None
    )


def test_grading_without_a_judge() -> None:
    refusal = "I don't know based on the provided documents."
    items = [
        answer("current", ["new"], "It needs 72 credits.", id="a"),
        answer(
            "current", ["old"], "It needs 72 credits.", id="b"
        ),  # wrong edition
        answer("year", [], refusal, id="c", expected=["old"], wrong=["new"]),
        answer(
            "general",
            ["g"],
            "Rule: 72 credits.",
            id="d",
            expected=["g"],
            wrong=[],
        ),
        answer(
            "unanswerable", [], refusal, id="e", facts=[], expected=[], wrong=[]
        ),
        answer(
            "unanswerable",
            ["g"],
            "IIT fee is 5",
            id="f",
            facts=[],
            expected=[],
            wrong=[],
        ),
    ]
    rows = {r["id"]: r for r in grade.grade_all(items, None, None)}  # type: ignore[arg-type]
    assert rows["a"]["correct"] and not rows["b"]["correct"]
    assert rows["b"]["rubric_correct"] and not rows["b"]["right_edition"]
    assert not rows["c"]["correct"] and rows["c"]["abstained"]
    assert rows["d"]["correct"] and rows["d"]["right_edition"] is None
    assert rows["e"]["correct"] and not rows["f"]["correct"]
    assert rows["b"]["any_stale_citation"]
    summary = grade.summarise(list(rows.values()))
    assert summary["items"] == 6  # noqa: PLR2004
    assert summary["accuracy"] == 0.5  # noqa: PLR2004
    assert summary["edition_correctness"] == round(1 / 3, 4)
    assert summary["by_type"]["current"]["correct"] == 0.5  # noqa: PLR2004
    judged = {**rows["a"], "judge_correct": False}
    assert not grade.final(judged)


def test_questions_must_name_their_subject() -> None:
    ds = {
        "title": "DS252: Introduction to Cloud Computing – CDS",
        "original_url": "https://cds.iisc.ac.in/courses/ds252-aug25/",
    }
    assert gen.anchored("How many credits does DS252 carry?", ds)
    assert gen.anchored("When do cloud computing lectures start?", ds)
    assert gen.anchored("What is the DS 252 project weight?", ds)
    assert not gen.anchored(
        "What is the first day of lectures for the course?", ds
    )
    assert not gen.anchored("Which cycle is mentioned in the CDS document?", ds)
    words = gen.subject_words(ds)
    assert {"ds252", "cloud", "computing"} <= words
    assert "introduction" in words and "course" not in words


def test_judge_runs_in_threads_with_sqlite_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: ChunkTexts (SQLite) was read from the judge's threads.
    import sqlite3

    import httpx
    from eval.grader.rubric import ChunkTexts

    db = sqlite3.connect(tmp_path / "versions.db")
    db.execute("CREATE TABLE versions (doc_id, version, drafts)")
    db.execute(
        "INSERT INTO versions VALUES ('new', 1, ?)",
        (json.dumps([{"text": "The programme needs 72 credits."}]),),
    )
    db.commit()
    db.close()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][0]["content"])
        content = '{"correct": true, "citations_support": true}'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    monkeypatch.setenv("ASKIISC_LLM_URL", "http://judge")
    monkeypatch.setenv("ASKIISC_LLM_MODEL", "granite")
    items = [
        answer(
            "current",
            ["new"],
            f"It needs 72 credits ({i}).",
            id=str(i),
            question="How many credits does the M.Tech CDS need?",
        )
        for i in range(20)
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        rows = grade.grade_all(items, ChunkTexts(tmp_path), http)
    assert all(r["judge_correct"] and r["cites_support"] for r in rows)
    assert len(seen) == 20 and "needs 72 credits" in seen[0]  # noqa: PLR2004


def test_changed_spans_show_real_replacements() -> None:
    old = "DS252 has 3:1 credits. Lectures by Dr. Simmhan on Tuesday."
    new = "DS252 has 4:0 credits. Lectures by Simmhan on Tuesday."
    spans = gen.changed_spans(old, new)
    assert spans == [("DS252 has", "3:1", "4:0")]
    assert gen.changed_spans(old, old) == []


def test_change_question_falls_back_when_topic_is_vague(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKIISC_LLM_MODEL", "qwen3-8b")
    newer = row(
        "new",
        AUG_2026,
        None,
        title="DS252 Cloud Computing",
        original_url="https://cds.iisc.ac.in/courses/ds252/",
    )
    older = row("old", 0.0, AUG_2026, edition_start=AUG_2025)
    found = {
        "question": "When does DS252 start?",
        "topic": "the first class date",
        "new_value": "Aug 4",
        "old_value": "Aug 6",
        "new_chunk": "n",
        "old_chunk": "o",
    }
    change = next(
        i
        for i in gen.change_items(found, newer, older, NOW)
        if i["type"] == "change"
    )
    assert change["question"] == (
        "Between 2025-26 and 2026-27, how did this change: "
        "when does DS252 start?"
    )
    assert gen.plausible_answer("Aug 4, 2026")
    assert not gen.plausible_answer("Date of Joining : ..........")
    assert not gen.plausible_answer("856 1707 130 474 446")


def test_edition_ok_accepts_another_supporting_page_but_no_wrong_edition() -> (
    None
):
    base = {"type": "current", "right_edition": False, "wrong_cited": False}
    assert grade.edition_ok({**base, "cites_support": True})
    assert not grade.edition_ok({**base, "cites_support": False})
    assert not grade.edition_ok({**base, "cites_support": None})
    wrong = {**base, "cites_support": True, "wrong_cited": True}
    assert not grade.edition_ok(wrong)
    assert grade.edition_ok({**base, "right_edition": True})
    assert grade.edition_ok({**base, "right_edition": None}) is None
    row = {
        **base,
        "cites_support": True,
        "abstained": False,
        "judge_correct": True,
        "rubric_correct": True,
        "edition_ok": True,
    }
    assert grade.final(row) and not grade.final(row, strict=True)


def test_a_grade_is_recorded_for_the_scorecard(tmp_path: Path) -> None:
    from deploy.serve import load_test_runs

    from askiisc.web.scorecard import scorecard

    summary = {
        "accuracy": 0.845,
        "citation_correctness": 0.977,
        "strict_accuracy": 0.7175,
        "edition_correctness": 0.957,
        "judge_model": "granite-3.3-8b-instruct",
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    measurements = tmp_path / "measurements"
    assert grade.main(["record", str(path), str(measurements)]) == 0
    saved = json.loads(next(measurements.glob("*/summary.json")).read_text())
    assert saved["workload"] == "quality-judge"
    assert saved["checks"]["judge_accuracy"] == 0.845  # noqa: PLR2004
    assert "edition-aware set" in saved["kind"]
    row = next(
        r for r in scorecard(load_test_runs(measurements)) if r.id == "NFR-7"
    )
    assert row.verdict == "pass"


def test_scorecard_record_carries_the_fr1_refusal_rate(tmp_path: Path) -> None:
    from eval.versioned.grade import record

    summary = {
        "accuracy": 0.845,
        "citation_correctness": 0.9771,
        "strict_accuracy": 0.7175,
        "edition_correctness": 0.9568,
        "by_type": {"unanswerable": {"items": 40, "abstained": 0.775}},
        "judge_model": "granite-3.3-8b-instruct",
    }
    written = json.loads(record(summary, tmp_path, "grade.json").read_text())
    assert written["requirements"] == "FR-1, NFR-7"
    assert written["checks"]["abstain_unanswerable"] == 0.775
    del summary["by_type"]
    again = json.loads(record(summary, tmp_path, "grade.json").read_text())
    assert "abstain_unanswerable" not in again["checks"]
