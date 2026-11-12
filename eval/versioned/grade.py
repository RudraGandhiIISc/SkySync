"""Grades an edition-aware run: accuracy, citations and editions.

Per item (answers.json from eval.versioned.answer):
  rubric_correct   every key fact appears in the answer (lenient match:
                   80% of its words), and the answer is not a refusal;
                   unanswerable items are correct only when refused
  judge_correct    the LLM judge (a different model from the one tested)
                   says the answer states the reference fact(s)
  cites_support    the judge says the cited chunks contain the fact
  right_edition    strict: current/year: a cited document is the expected
                   one and none is a wrong edition; change: both editions
                   are cited
  edition_ok       right_edition, or the judge says the cited chunks
                   support the answer and no wrong edition is cited (the
                   same fact is often on another valid page: the course
                   list and the course's own page)
  retrieved        an expected document was among the retrieved chunks
  correct          judge_correct (rubric_correct when the judge failed)
                   and, for edition items, edition_ok
  strict_correct   the same with right_edition (reported alongside)

Usage (inside slurm/llm_job.sh with MODEL=granite-3.3-8b-instruct):
    python -m eval.versioned.grade ANSWERS.json STATE_DIR OUT_DIR [MEASUREMENTS]
MEASUREMENTS (results/measurements) gets a quality-judge summary for the
admin scorecard (NFR-7). An earlier grade is recorded without grading again:
    python -m eval.versioned.grade record OUT_DIR/summary.json MEASUREMENTS
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from eval.grader.judge import PROMPT, judge_one
from eval.grader.rubric import ChunkTexts, lenient_match
from eval.measurements import write_measurement

EDITION_TYPES = frozenset({"current", "year", "change"})
WORKERS = 16
CITED_CHARS = 1500


def right_edition(item: dict[str, Any]) -> bool | None:
    """Whether the cited documents are the edition the question needs."""
    expected = set(item.get("expected_docs") or [])
    if not expected:
        return None
    cited = {c["doc_id"] for c in item.get("citations") or []}
    if item["type"] == "change":
        return expected <= cited
    wrong = set(item.get("wrong_docs") or [])
    return bool(cited & expected) and not cited & wrong


def rubric(item: dict[str, Any]) -> dict[str, Any]:
    """The deterministic checks of one item."""
    abstained = bool(item.get("abstained"))
    facts = item.get("key_facts") or []
    if item["type"] == "unanswerable":
        return {
            "rubric_correct": abstained,
            "right_edition": None,
            "retrieved": None,
        }
    has_facts = bool(facts) and all(
        lenient_match(f, str(item.get("answer") or "")) for f in facts
    )
    expected = set(item.get("expected_docs") or [])
    return {
        "rubric_correct": has_facts and not abstained,
        "right_edition": (
            right_edition(item) if item["type"] in EDITION_TYPES else None
        ),
        "retrieved": bool(expected & set(item.get("retrieved_docs") or [])),
    }


def cited_text(chunks: ChunkTexts, item: dict[str, Any]) -> str:
    """The text of the chunks an answer cites (SQLite: caller's thread)."""
    return (
        "\n---\n".join(
            chunks.text(c["chunk_id"])[:CITED_CHARS]
            for c in item.get("citations") or []
        )
        or "(no citations)"
    )


def judge(
    http: httpx.Client, item: dict[str, Any], cited: str
) -> dict[str, bool] | None:
    """The LLM judge's verdict on one answered item."""
    reference = " ; ".join(item.get("key_facts") or [])
    if item["type"] == "change":
        reference = (
            f"earlier: {item['key_facts'][0]}; later: {item['key_facts'][1]}"
        )
    return judge_one(
        http,
        os.environ["ASKIISC_LLM_URL"],
        os.environ["ASKIISC_LLM_MODEL"],
        PROMPT.format(
            question=item["question"],
            reference=reference,
            answer=str(item.get("answer") or "")[:CITED_CHARS],
            cited=cited,
        ),
    )


def wrong_cited(item: dict[str, Any]) -> bool:
    """Whether the answer cites an edition the question rules out."""
    cited = {c["doc_id"] for c in item.get("citations") or []}
    return bool(cited & set(item.get("wrong_docs") or []))


def edition_ok(row: dict[str, Any]) -> bool | None:
    """The exact edition, or supported citations and no wrong edition."""
    if row["right_edition"] is None:
        return None
    return bool(
        row["right_edition"]
        or (row.get("cites_support") and not row["wrong_cited"])
    )


def final(row: dict[str, Any], strict: bool = False) -> bool:
    """The headline verdict of one graded item.

    Args:
        row: The graded item.
        strict: Require the exact expected edition (right_edition).
    """
    if row["type"] == "unanswerable":
        return bool(row["rubric_correct"])
    if row["abstained"]:
        return False
    verdict = row.get("judge_correct")
    correct = row["rubric_correct"] if verdict is None else verdict
    if row["type"] in EDITION_TYPES:
        key = "right_edition" if strict else "edition_ok"
        return bool(correct and row[key])
    return bool(correct)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(map(bool, values)) / len(values), 4) if values else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall and per-type rates."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    keys = (
        "correct",
        "strict_correct",
        "rubric_correct",
        "judge_correct",
        "cites_support",
        "right_edition",
        "edition_ok",
        "retrieved",
        "abstained",
        "any_stale_citation",
    )
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    return {
        "items": len(rows),
        "accuracy": _rate(rows, "correct"),
        "strict_accuracy": _rate(rows, "strict_correct"),
        "answerable_accuracy": _rate(answerable, "correct"),
        "citation_correctness": _rate(answerable, "cites_support"),
        "edition_correctness": _rate(
            [r for r in rows if r["type"] in EDITION_TYPES], "edition_ok"
        ),
        "exact_edition": _rate(
            [r for r in rows if r["type"] in EDITION_TYPES], "right_edition"
        ),
        "false_refusals": _rate(answerable, "abstained"),
        "by_type": {
            t: {"items": len(rs), **{k: _rate(rs, k) for k in keys}}
            for t, rs in sorted(by_type.items())
        },
    }


def grade_all(
    items: list[dict[str, Any]], chunks: ChunkTexts, http: httpx.Client | None
) -> list[dict[str, Any]]:
    """Rubric for every item; the judge for answered answerable items."""
    rows = []
    for item in items:
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "abstained": bool(item.get("abstained")),
                "any_stale_citation": any(
                    c.get("stale") for c in item.get("citations") or []
                ),
                "wrong_cited": wrong_cited(item),
                **rubric(item),
            }
        )
    if http is not None:
        todo = [
            (row, item, cited_text(chunks, item))
            for row, item in zip(rows, items, strict=True)
            if item["type"] != "unanswerable" and not item.get("abstained")
        ]
        with ThreadPoolExecutor(WORKERS) as pool:
            verdicts = pool.map(lambda x: judge(http, x[1], x[2]), todo)
            for (row, _, _), verdict in zip(todo, verdicts, strict=True):
                row["judge_correct"] = verdict["correct"] if verdict else None
                row["cites_support"] = (
                    verdict["citations_support"] if verdict else None
                )
    for row in rows:
        row["edition_ok"] = edition_ok(row)
        row["correct"] = final(row)
        row["strict_correct"] = final(row, strict=True)
    return rows


SCORECARD_SET = "edition-aware set"


def record(summary: dict[str, Any], measurements: Path, source: str) -> Path:
    """Writes the FR-1 and NFR-7 numbers of a grade for the scorecard.

    Args:
        summary: A summary.json written by this grader.
        measurements: The measurements folder (results/measurements).
        source: Where the summary came from, for people.

    Returns:
        The measurement file.
    """
    checks = {
        "judge_accuracy": summary["accuracy"],
        "judge_citation_support": summary["citation_correctness"],
        "strict_accuracy": summary["strict_accuracy"],
        "edition_correctness": summary["edition_correctness"],
    }
    unanswerable = summary.get("by_type", {}).get("unanswerable", {})
    if unanswerable.get("items"):
        checks["abstain_unanswerable"] = unanswerable["abstained"]
    return write_measurement(
        measurements,
        "quality-judge",
        "FR-1, NFR-7",
        checks,
        f"Grader ({SCORECARD_SET}, {summary.get('judge_model', 'none')})",
        time.time(),
        answers=source,
    )


def main(argv: list[str]) -> int:
    """Writes graded.json and summary.json; prints the summary."""
    if argv[:1] == ["record"]:
        path = Path(argv[1])
        print(record(json.loads(path.read_text()), Path(argv[2]), str(path)))
        return 0
    answers, state, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(answers.read_text())
    chunks = ChunkTexts(state)
    use_judge = "ASKIISC_LLM_URL" in os.environ
    with httpx.Client() as http:
        rows = grade_all(items, chunks, http if use_judge else None)
    summary = summarise(rows)
    summary["judge_model"] = os.environ.get("ASKIISC_LLM_MODEL", "none")
    (out / "graded.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if len(argv) > 3:  # noqa: PLR2004
        record(summary, Path(argv[3]), str(answers))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
