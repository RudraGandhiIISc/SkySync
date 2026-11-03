"""LLM judge for answerable held-out items (NFR-7).

A fixed model that differs from the model under test (Granite-3.3-8B for
Qwen3-4B) rates, at temperature 0 with a strict JSON reply, whether the
answer is correct against the reference and whether the cited chunks
support it. Items where judge and rubric disagree go to disagreements.csv
for a person to settle.

Usage (inside slurm/llm_job.sh with MODEL=granite-3.3-8b-instruct):
    python -m eval.grader.judge RESULTS.json STATE_DIR GRADE_DIR
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from eval.grader.rubric import ChunkTexts
from eval.measurements import write_measurement

PROMPT = """You grade answers of a document question-answering system.
Question: {question}
Reference answer: {reference}
System answer: {answer}
Text of the chunks the system cited:
<<<
{cited}
>>>
Reply with JSON only, no prose:
{{"correct": true or false, "citations_support": true or false}}
"correct" is true if the system answer states the reference answer's fact
(wording may differ). "citations_support" is true if the cited text
contains that fact."""


def judge_one(
    http: httpx.Client, url: str, model: str, prompt: str
) -> dict[str, bool] | None:
    """Asks the judge, retrying once on malformed JSON."""
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 60,
        "messages": [{"role": "user", "content": prompt}],
    }
    for _ in range(2):
        try:
            r = http.post(url + "/v1/chat/completions", json=body, timeout=120)
            text = r.json()["choices"][0]["message"]["content"]
            data = json.loads(text[text.index("{") : text.rindex("}") + 1])
            return {
                "correct": bool(data["correct"]),
                "citations_support": bool(data["citations_support"]),
            }
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            continue
    return None


def main(argv: list[str]) -> int:
    """Judges every answered answerable item; writes judge.json.

    A fourth argument (the measurements folder) also sends the NFR-7
    numbers to the scorecard.
    """
    started = time.time()
    results, state, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    url, model = os.environ["ASKIISC_LLM_URL"], os.environ["ASKIISC_LLM_MODEL"]
    items = [
        i
        for i in json.loads(results.read_text())
        if i["category"] == "answerable"
    ]
    graded = {g["id"]: g for g in json.loads((out / "graded.json").read_text())}
    chunks = ChunkTexts(state)
    rows: list[dict[str, Any]] = []
    with httpx.Client() as http:
        for item in items:
            answer = item.get("answer") or {}
            if answer.get("abstained") or not answer:
                rows.append({"id": item["id"], "judged": False})
                continue
            cited = (
                "\n---\n".join(
                    chunks.text(c["chunk_id"])[:1500]
                    for c in answer.get("citations", [])
                )
                or "(no citations)"
            )
            verdict = judge_one(
                http,
                url,
                model,
                PROMPT.format(
                    question=item["question"],
                    reference=item["reference_answer"],
                    answer=answer["text"][:1500],
                    cited=cited,
                ),
            )
            rows.append(
                {
                    "id": item["id"],
                    "judged": verdict is not None,
                    **(verdict or {}),
                }
            )
    judged = [r for r in rows if r["judged"]]
    agree = [r for r in judged if r["correct"] == graded[r["id"]]["correct"]]
    summary = {
        "judge_model": model,
        "answerable": len(items),
        "judged": len(judged),
        "judge_accuracy": round(
            sum(r["correct"] for r in judged) / len(items), 4
        )
        if items
        else None,
        "judge_citation_support": round(
            sum(r["citations_support"] for r in judged) / len(judged), 4
        )
        if judged
        else None,
        "agreement_with_rubric": round(len(agree) / len(judged), 4)
        if judged
        else None,
    }
    (out / "judge.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1)
    )
    with (out / "disagreements.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "question",
                "reference",
                "answer",
                "rubric",
                "judge",
                "human_verdict",
            ]
        )
        by_id = {i["id"]: i for i in items}
        for r in judged:
            if r["correct"] != graded[r["id"]]["correct"]:
                i = by_id[r["id"]]
                writer.writerow(
                    [
                        r["id"],
                        i["question"],
                        i["reference_answer"],
                        (i.get("answer") or {}).get("text", "")[:500],
                        graded[r["id"]]["correct"],
                        r["correct"],
                        "",
                    ]
                )
    if len(argv) > 3:  # noqa: PLR2004
        write_measurement(
            Path(argv[3]), "quality-judge", "NFR-7", summary, "Grader",
            started, answers=str(results),
        )  # fmt: skip
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
