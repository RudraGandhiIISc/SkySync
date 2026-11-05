r"""NFR-7 adjudication: a review sheet for people, then the final scores.

`sheet` writes `review.csv` for one graded run: every item where the
rubric and the judge disagree, plus a seeded random audit sample of the
others (60 by default). Each row has the question, the reference answer,
the answer, the text of the chunks it cites, both verdicts, and empty
`human_correct` and `human_citations` columns. People fill those in
(yes / no, or blank to leave an item as graded); this tool never does.

`score` reads the completed sheet back. A human verdict replaces the
judge's content verdict, a human citation verdict replaces
`cites_support`, and every item is scored again with the grader's own
rules (`eval/versioned/grade.py`). It reports accuracy and citation
correctness before and after, and agreement between rubric, judge and
human (percent agreement and Cohen's kappa).

Run (inside Slurm):
    python -m eval.grader.adjudicate sheet <answers.json> <graded.json> \\
        <state dir> <out dir> [--audit 60] [--seed 252]
    python -m eval.grader.adjudicate score <graded.json> <review.csv> <out>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from eval.versioned.grade import final, summarise

AUDIT_ITEMS = 60
SEED = 252
COLUMNS = (
    "id",
    "sample",
    "type",
    "question",
    "reference",
    "answer",
    "cited_text",
    "rubric_correct",
    "judge_correct",
    "human_correct",
    "human_citations",
    "notes",
)
YES = frozenset({"yes", "y", "true", "1"})
NO = frozenset({"no", "n", "false", "0"})


def disagrees(row: dict[str, Any]) -> bool:
    """Whether rubric and judge both gave a verdict and differ."""
    judge = row.get("judge_correct")
    return judge is not None and bool(judge) != bool(row["rubric_correct"])


def pick(
    graded: list[dict[str, Any]], audit: int, seed: int
) -> list[tuple[str, dict[str, Any]]]:
    """The items to review: all disagreements, then a seeded sample.

    Args:
        graded: The grader's rows.
        audit: How many agreeing items to sample.
        seed: The sample's random seed (recorded on the sheet).

    Returns:
        (sample name, row) pairs, disagreements first.
    """
    split = [r for r in graded if disagrees(r)]
    rest = [r for r in graded if not disagrees(r)]
    sample = random.Random(seed).sample(rest, min(audit, len(rest)))
    return [("disagreement", r) for r in split] + [("audit", r) for r in sample]


def sheet_rows(
    picked: list[tuple[str, dict[str, Any]]],
    answers: dict[str, dict[str, Any]],
    cited: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    """One review row per picked item, human columns empty."""
    rows = []
    for sample, graded in picked:
        item = answers.get(graded["id"], {})
        rows.append(
            {
                "id": graded["id"],
                "sample": sample,
                "type": graded.get("type", ""),
                "question": item.get("question", ""),
                "reference": item.get("reference_answer", ""),
                "answer": item.get("answer", ""),
                "cited_text": cited(item) if item else "",
                "rubric_correct": graded["rubric_correct"],
                "judge_correct": graded.get("judge_correct"),
                "human_correct": "",
                "human_citations": "",
                "notes": "",
            }
        )
    return rows


def write_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes the review sheet as CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def verdict(text: str) -> bool | None:
    """A sheet cell as yes / no / not given.

    Raises:
        ValueError: For anything else, so a typo is never read as a verdict.
    """
    value = text.strip().lower()
    if not value:
        return None
    if value in YES:
        return True
    if value in NO:
        return False
    raise ValueError(f"not a verdict: {text!r} (use yes or no)")


def kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa of two raters' yes / no verdicts (None if undefined)."""
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def agreement(
    rows: list[dict[str, Any]], first: str, second: str
) -> dict[str, Any]:
    """Percent agreement and kappa of two verdict columns."""
    pairs = [
        (bool(r[first]), bool(r[second]))
        for r in rows
        if r.get(first) is not None and r.get(second) is not None
    ]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    same = sum(x == y for x, y in pairs)
    return {
        "items": len(pairs),
        "percent": round(100 * same / len(pairs), 2) if pairs else None,
        "kappa": kappa(a, b),
    }


def score(
    graded: list[dict[str, Any]], review: list[dict[str, str]]
) -> dict[str, Any]:
    """Final scores with the human verdicts applied.

    Args:
        graded: The grader's rows.
        review: The completed sheet's rows.

    Returns:
        Before and after summaries, how many verdicts were applied, and
        the agreement between rubric, judge and human.
    """
    human = {r["id"]: r for r in review}
    rows = []
    for row in graded:
        new = dict(row)
        sheet = human.get(row["id"])
        new["human_correct"] = None
        if sheet is not None:
            new["human_correct"] = verdict(sheet.get("human_correct", ""))
            cites = verdict(sheet.get("human_citations", ""))
            if new["human_correct"] is not None:
                new["judge_correct"] = new["human_correct"]
            if cites is not None:
                new["cites_support"] = cites
        new["correct"] = final(new)
        new["strict_correct"] = final(new, strict=True)
        rows.append(new)
    judged = [
        {**old, "human": new["human_correct"]}
        for old, new in zip(graded, rows, strict=True)
    ]
    return {
        "before": summarise(graded),
        "after": summarise(rows),
        "human_verdicts": sum(r["human_correct"] is not None for r in rows),
        "agreement": {
            "rubric_judge": agreement(
                graded, "rubric_correct", "judge_correct"
            ),
            "rubric_human": agreement(judged, "rubric_correct", "human"),
            "judge_human": agreement(judged, "judge_correct", "human"),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """`sheet` or `score` (see the module doc)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet")
    for name in ("answers", "graded", "state", "out"):
        s.add_argument(name, type=Path)
    s.add_argument("--audit", type=int, default=AUDIT_ITEMS)
    s.add_argument("--seed", type=int, default=SEED)
    c = sub.add_parser("score")
    for name in ("graded", "review", "out"):
        c.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    graded = json.loads(args.graded.read_text())
    if args.command == "sheet":
        return make_sheet(args, graded)
    with args.review.open(newline="", encoding="utf-8") as f:
        review = list(csv.DictReader(f))
    result = score(graded, review)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adjudicated.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("human_verdicts", "agreement")}))  # noqa: T201
    return 0


def make_sheet(args: argparse.Namespace, graded: list[dict[str, Any]]) -> int:
    """Writes review.csv and sheet.json (what was sampled and how)."""
    from eval.grader.rubric import ChunkTexts  # noqa: PLC0415 - needs state
    from eval.versioned.grade import cited_text  # noqa: PLC0415

    answers = {a["id"]: a for a in json.loads(args.answers.read_text())}
    chunks = ChunkTexts(args.state)
    picked = pick(graded, args.audit, args.seed)
    rows = sheet_rows(picked, answers, lambda item: cited_text(chunks, item))
    args.out.mkdir(parents=True, exist_ok=True)
    write_sheet(rows, args.out / "review.csv")
    info = {
        "graded": str(args.graded),
        "answers": str(args.answers),
        "seed": args.seed,
        "disagreements": sum(p[0] == "disagreement" for p in picked),
        "audit": sum(p[0] == "audit" for p in picked),
    }
    (args.out / "sheet.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
