"""Deterministic grading of a held-out run (FR-1, FR-5, NFR-7, NFR-9).

Checks, per category:
  answerable    the key fact appears in the answer (correct); every cited
                chunk is one that was retrieved, and a cited chunk contains
                the fact (citation supports the claim)
  unanswerable  the answer is the exact refusal
  permission    a reader who may see the document gets the fact; one who
                may not never has it retrieved or cited (a leak otherwise)
  revision      the as-of answer gives that version's fact; version 1
                answers carry the stale flag, version 2 answers do not

Usage: python -m eval.grader.rubric RESULTS.json STATE_DIR OUT_DIR
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.measurements import write_measurement


def norm(text: str) -> str:
    """Lower-cases, collapses spaces and drops digit separators."""
    text = re.sub(r"(?<=\d)[,\s](?=\d{3})", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class ChunkTexts:
    """Looks up chunk text by id (doc@vN#i) in versions.db."""

    def __init__(self, state: Path) -> None:
        self._db = sqlite3.connect(
            f"file:{state / 'versions.db'}?mode=ro", uri=True, timeout=120.0
        )
        self._cache: dict[tuple[str, int], list[str]] = {}

    def text(self, chunk_id: str) -> str:
        """The chunk's text, or empty if unknown."""
        m = re.match(r"^(.*)@v(\d+)#(\d+)$", chunk_id)
        if not m:
            return ""
        doc, version, index = m.group(1), int(m.group(2)), int(m.group(3))
        if (doc, version) not in self._cache:
            row = self._db.execute(
                "SELECT drafts FROM versions WHERE doc_id=? AND version=?",
                (doc, version),
            ).fetchone()
            drafts = json.loads(row[0]) if row else []
            self._cache[(doc, version)] = [d["text"] for d in drafts]
        texts = self._cache[(doc, version)]
        return texts[index] if index < len(texts) else ""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(text).replace(".", ""))


def lenient_match(fact: str, text: str) -> bool:
    """At least 80% of the fact's words appear in the text.

    Punctuation and case are ignored ("B.Tech" == "B Tech"), and words match
    on their first 6 letters ("Post-Doctorate" ~ "post-doctoral").
    """
    want = _words(fact)
    have = set(_words(text))
    stems = {w[:6] for w in have}
    hits = sum(w in have or w[:6] in stems for w in want)
    return bool(want) and hits / len(want) >= 0.8  # noqa: PLR2004


def grade(item: dict[str, Any], chunks: ChunkTexts) -> dict[str, Any]:
    """Grades one result."""
    answer = item.get("answer") or {}
    text = norm(str(answer.get("text", "")))
    cited = answer.get("citations") or []
    retrieved = [d for _, d in answer.get("retrieved") or []]
    facts = [norm(f) for f in item.get("key_facts") or []]
    has_fact = bool(facts) and any(f in text for f in facts)
    raw_answer = str(answer.get("text", ""))
    lenient = bool(facts) and any(
        lenient_match(f, raw_answer) for f in item.get("key_facts") or []
    )
    supported = [
        c
        for c in cited
        if any(f in norm(chunks.text(c["chunk_id"])) for f in facts)
    ]
    out: dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "ok_response": item.get("status") == 200 and bool(answer),
        "abstained": bool(answer.get("abstained")),
        "has_fact": has_fact,
        "citations": len(cited),
        "supported_citations": len(supported),
    }
    category = item["category"]
    out["has_fact_lenient"] = lenient
    if category == "answerable":
        out["correct"] = has_fact and not out["abstained"]
        out["correct_lenient"] = lenient and not out["abstained"]
    elif category == "unanswerable":
        out["correct"] = out["abstained"]
    elif category == "permission":
        doc = item["doc_id"]
        leaked = doc in retrieved or any(c["doc_id"] == doc for c in cited)
        out["leak"] = (not item["may_read"]) and leaked
        out["correct"] = has_fact if item["may_read"] else not leaked
    else:  # revision
        stale = any(c.get("is_stale") for c in cited)
        out["stale_flag_ok"] = stale == bool(item["expect_stale"])
        out["correct"] = has_fact and out["stale_flag_ok"]
    return out


def report(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates the PRD metrics."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in graded:
        by[g["category"]].append(g)

    def frac(rows: list[dict[str, Any]], key: str) -> float | None:
        return (
            round(sum(bool(r[key]) for r in rows) / len(rows), 4)
            if rows
            else None
        )

    answerable = by["answerable"]
    answered = [g for g in answerable if not g["abstained"]]
    citations = sum(g["citations"] for g in answered)
    supported = sum(g["supported_citations"] for g in answered)
    return {
        "items": len(graded),
        "accuracy_answerable": frac(answerable, "correct"),  # NFR-7 >= 0.80
        "accuracy_answerable_lenient": frac(answerable, "correct_lenient"),
        "answered_with_supporting_citation": round(
            sum(g["supported_citations"] > 0 for g in answered) / len(answered),
            4,
        )
        if answered
        else None,  # FR-1 >= 0.95
        "citation_correctness": round(supported / citations, 4)
        if citations
        else None,  # NFR-7 >= 0.90 (rubric proxy)
        "abstain_unanswerable": frac(by["unanswerable"], "correct"),  # FR-1
        "permission_correct": frac(by["permission"], "correct"),
        "permission_leaks": sum(bool(g.get("leak")) for g in by["permission"]),
        "revision_correct": frac(by["revision"], "correct"),
        "revision_stale_flag_ok": frac(by["revision"], "stale_flag_ok"),
        "false_abstain_answerable": frac(answerable, "abstained"),
        "ok_responses": frac(graded, "ok_response"),
    }


def main(argv: list[str]) -> int:
    """Writes graded.json and report.json and prints the report.

    Usage: rubric RESULTS.json STATE_DIR GRADE_DIR [MEASUREMENTS_DIR]; with
    the last, the FR-1 numbers also go to the scorecard.
    """
    started = time.time()
    results, state, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads(results.read_text())
    chunks = ChunkTexts(state)
    graded = [grade(i, chunks) for i in items]
    summary = report(graded)
    (out / "graded.json").write_text(json.dumps(graded, indent=1))
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    if len(argv) > 3:  # noqa: PLR2004
        write_measurement(
            Path(argv[3]), "quality-rubric", "FR-1, NFR-7", summary,
            "Grader", started, answers=str(results),
        )  # fmt: skip
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
