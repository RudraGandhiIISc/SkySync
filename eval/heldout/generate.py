"""Drafts the answerable part of the held-out set from the real index.

For each sampled in-scope chunk, an LLM proposes one question and a short
answer copied from the chunk. An item is kept only if the answer appears
verbatim (after normalising case, spaces and digit separators) in the
chunk and not in the question, so every reference answer is grounded in a
real document. Items are marked reviewed=false: a person must check them
before they count as a held-out set (README Phase 5.3).

Usage (inside slurm/llm_job.sh, which sets ASKIISC_LLM_URL/_MODEL):
    python -m eval.heldout.generate --state DIR --corpus DIR --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import httpx

OTHER = "other"

QUOTAS = {  # (department, category) -> questions; 150 answerable (v2)
    ("institute", "admissions"): 25,
    ("institute", "academic-policy"): 25,
    ("institute", "course-catalogue"): 10,
    ("institute", "handbooks"): 15,
    ("cds", None): 15,
    ("csa", None): 15,
    (OTHER, None): 45,  # every other department, at most 3 each
}
PER_OTHER_DEPARTMENT = 3
SKIP_URL_WORDS = (
    "shortlist",
    "result",
    "selected",
    "candidat",
    "/people",
    "/faculty/",
    "news",
    "event",
    "seminar",
    "talk",
    "award",
    "tender",
    "/students/",
    "alumni",
    "obituar",
    "merit",
)
PROMPT = """Text from an IISc web page or document:
<<<
{text}
>>>
Write ONE question that a student, applicant or staff member might ask about
IISc rules, programmes, admissions, fees, dates, eligibility, courses or
procedures, and that this text answers. Never ask about a named individual,
and never about research findings, and give the answer as a SHORT phrase
(1 to 8 words)
copied exactly from the text. The question must make sense on its own
(name the programme, department or document it is about) and must not
contain the answer. Reply with JSON only:
{{"question": "...", "answer": "..."}}
If the text has no clear, specific fact, reply {{"question": null}}."""


def norm(text: str) -> str:
    """Lower-cases, collapses spaces and drops digit separators."""
    text = re.sub(r"(?<=\d)[,\s](?=\d{3})", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def grounded(question: str, answer: str, source: str) -> bool:
    """Whether a proposed item is usable."""
    words = answer.split()
    if not 1 <= len(words) <= 12 or not 4 <= len(question.split()) <= 40:
        return False
    return norm(answer) in norm(source) and norm(answer) not in norm(question)


def candidates(
    state: Path, corpus: Path
) -> dict[tuple[str, str | None], list[dict[str, str]]]:
    """In-scope chunks of public documents, grouped by quota bucket."""
    meta = {}
    for line in (corpus / "manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        meta[e["doc_id"]] = e
    db = sqlite3.connect(
        f"file:{state / 'versions.db'}?mode=ro", uri=True, timeout=120.0
    )
    rows = db.execute(
        "SELECT doc_id, version, drafts FROM versions WHERE valid_to IS NULL"
    )
    out: dict[tuple[str, str | None], list[dict[str, str]]] = {
        k: [] for k in QUOTAS
    }
    for doc_id, version, drafts in rows:
        e = meta.get(doc_id)
        if e is None:
            continue
        if any(w in e["url"].lower() for w in SKIP_URL_WORDS):
            continue  # lists of people, results, news: not policy text
        if e["category"] in ("general", "department"):
            continue
        key = (e["department"], e["category"])
        if key not in QUOTAS:
            key = (e["department"], None)
        if key not in QUOTAS:
            key = (OTHER, None)
        for i, d in enumerate(json.loads(drafts)):
            text = d["text"]
            letters = sum(c.isalpha() for c in text) / max(1, len(text))
            if 300 <= len(text) <= 1500 and letters > 0.6:
                out[key].append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}@v{version}#{i}",
                        "text": text,
                        "url": e["url"],
                        "category": e["category"],
                        "department": e["department"],
                    }
                )
    return out


def propose(http: httpx.Client, url: str, model: str, text: str) -> dict | None:
    """Asks the LLM for one question and answer; None if unusable."""
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
    }
    try:
        r = http.post(url + "/v1/chat/completions", json=body, timeout=60)
        content = r.json()["choices"][0]["message"]["content"]
        item = json.loads(content[content.index("{") : content.rindex("}") + 1])
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
    if not item.get("question") or not item.get("answer"):
        return None
    return {
        "question": str(item["question"]).strip(),
        "answer": str(item["answer"]).strip(),
    }


def main() -> int:
    """Writes the answerable items as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=252)
    args = parser.parse_args()
    url, model = os.environ["ASKIISC_LLM_URL"], os.environ["ASKIISC_LLM_MODEL"]
    rng = random.Random(args.seed)
    pools = candidates(args.state, args.corpus)
    items, tried = [], Counter()
    users = {"cds": "stu_cds", "csa": "stu_csa"}
    with httpx.Client() as http:
        for key, quota in QUOTAS.items():
            pool = pools[key]
            rng.shuffle(pool)
            per_doc: Counter[str] = Counter()
            per_dept: Counter[str] = Counter()
            got = 0
            for c in pool:
                if got >= quota:
                    break
                if per_doc[c["doc_id"]] >= 2:
                    continue
                if key[0] == OTHER and (
                    per_dept[c["department"]] >= PER_OTHER_DEPARTMENT
                ):
                    continue
                tried[key] += 1
                p = propose(http, url, model, c["text"])
                if p is None or not grounded(
                    p["question"], p["answer"], c["text"]
                ):
                    continue
                per_doc[c["doc_id"]] += 1
                per_dept[c["department"]] += 1
                got += 1
                items.append(
                    {
                        "id": f"a{len(items) + 1:03d}",
                        "category": "answerable",
                        "topic": f"{c['department']}/{c['category']}",
                        "user": users.get(c["department"], "stu_csa"),
                        "question": p["question"],
                        "reference_answer": p["answer"],
                        "key_facts": [p["answer"]],
                        "doc_id": c["doc_id"],
                        "chunk_id": c["chunk_id"],
                        "source_url": c["url"],
                        "reviewed": False,
                        "generated_by": f"{model}; answer verbatim in source",
                    }
                )
            print(
                f"{key}: {got}/{quota} kept from {tried[key]} tried "
                f"(pool {len(pool)})"
            )
    args.out.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"{len(items)} answerable items -> {args.out}")
    return 0 if len(items) >= 140 else 1  # noqa: PLR2004


if __name__ == "__main__":
    sys.exit(main())
