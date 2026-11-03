"""Assembles the held-out set into eval/datasets/heldout/questions.jsonl.

It combines answerable, unanswerable, permission and revision items.

- answerable (150): eval/datasets/heldout/answerable.jsonl (generate.py)
- unanswerable (30): written by hand; nothing in an IISc corpus answers them
- permission (20): 10 restricted test documents, each asked by one user who
  may read it (must answer) and one who may not (must not cite it)
- revision (20): 10 test circulars, each asked as of version 1 (old fact,
  stale citation) and as of version 2 (new fact)

Usage: python -m eval.heldout.build --revisions FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HELDOUT = Path("eval/datasets/heldout")

UNANSWERABLE = [
    "What is the hostel fee at IIT Bombay?",
    "Who is the current Prime Minister of Japan?",
    "What is the cut-off rank for MBBS at AIIMS Delhi?",
    "How do I cook masala dosa?",
    "Who won the 2022 FIFA World Cup?",
    "What is the capital of Australia?",
    "What is the admission fee at Stanford University?",
    "How many moons does Jupiter have?",
    "What is the price of gold in Mumbai today?",
    "Which team won the 2023 Cricket World Cup?",
    "What is the syllabus of the IIT JEE Advanced exam?",
    "Who wrote the novel War and Peace?",
    "What is the weather forecast for Bangalore tomorrow?",
    "What is the boiling point of mercury?",
    "How do I renew an Indian passport?",
    "What are the opening hours of the Taj Mahal?",
    "What is the MBA fee at IIM Ahmedabad?",
    "Who is the vice-chancellor of the University of Oxford?",
    "What is the GST rate on restaurant food?",
    "How do I apply for a US student visa?",
    "What is the population of Tokyo?",
    "Which company makes the iPhone?",
    "What is the speed of light in water?",
    "What is the train fare from Bangalore to Chennai?",
    "Who painted the Mona Lisa?",
    "What is the hostel curfew at IIT Madras?",
    "How many players are there in a rugby union team?",
    "What is the minimum balance for an SBI savings account?",
    "What is the tallest mountain in Africa?",
    "What is the placement record of NIT Trichy?",
]

PERMISSION = [  # doc, question, fact, allowed user, denied user
    (
        "cds-faculty-meeting-minutes",
        "To when was the CDS PhD comprehensive examination moved?",
        "second week of January",
        "fac_cds",
        "stu_cds",
    ),
    (
        "cds-phd-comprehensive-panel",
        "How many faculty members examine a candidate on the CDS PhD "
        "comprehensive examination panel?",
        "three",
        "fac_cds",
        "stu_cds_phd",
    ),
    (
        "cds-hpc-cluster-access",
        "Whose approval do CDS GPU cluster jobs longer than 48 hours need?",
        "advisor",
        "stu_cds",
        "stu_csa",
    ),
    (
        "cds-student-council-notes",
        "On what date will the CDS student council hold the department "
        "research symposium?",
        "14 November",
        "stu_cds_phd",
        "fac_cds",
    ),
    (
        "cds-office-procurement",
        "How many quotations does a CDS office purchase above 50,000 rupees "
        "need?",
        "three",
        "staff_cds",
        "stu_cds",
    ),
    (
        "csa-faculty-search-shortlist",
        "What do shortlisted candidates of the CSA faculty search committee "
        "give?",
        "seminar",
        "fac_csa",
        "fac_csa2",
    ),
    (
        "csa-internal-seminar-calendar",
        "On which day and time are CSA internal seminars held?",
        "Thursday",
        "stu_csa",
        "stu_cds",
    ),
    (
        "csa-ta-allocation",
        "In CSA, each core course gets one teaching assistant per how many "
        "students?",
        "30",
        "staff_csa",
        "stu_csa",
    ),
    (
        "csa-exam-moderation",
        "Who reviews a CSA question paper before the examination?",
        "second faculty member",
        "fac_csa2",
        "stu_csa_phd",
    ),
    (
        "csa-lab-server-room",
        "Who holds keys to the CSA server room?",
        "staff",
        "staff_csa",
        "stu_csa",
    ),
]


def build(revisions: Path) -> list[dict[str, object]]:
    """Returns every held-out item."""
    items = [
        json.loads(ln)
        for ln in (HELDOUT / "answerable.jsonl").read_text().splitlines()
    ]
    for i, q in enumerate(UNANSWERABLE, 1):
        items.append(
            {
                "id": f"u{i:03d}",
                "category": "unanswerable",
                "user": "stu_cds",
                "question": q,
                "key_facts": [],
                "doc_id": None,
                "reviewed": True,
            }
        )
    n = 0
    for doc, q, fact, allowed, denied in PERMISSION:
        for user, may in ((allowed, True), (denied, False)):
            n += 1
            items.append(
                {
                    "id": f"p{n:03d}",
                    "category": "permission",
                    "user": user,
                    "question": q,
                    "doc_id": doc,
                    "may_read": may,
                    "key_facts": [fact] if may else [],
                    "reviewed": True,
                }
            )
    n = 0
    for rev in json.loads(revisions.read_text()):
        for version in (1, 2):
            n += 1
            items.append(
                {
                    "id": f"r{n:03d}",
                    "category": "revision",
                    "user": "stu_cds",
                    "question": rev["question"],
                    "doc_id": rev["doc_id"],
                    "as_of_version": version,
                    "key_facts": [rev[f"v{version}"]],
                    "expect_stale": version == 1,
                    "reviewed": True,
                }
            )
    return items


def main() -> None:
    """Writes questions.jsonl and prints the counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revisions", type=Path, required=True)
    args = parser.parse_args()
    items = build(args.revisions)
    out = HELDOUT / "questions.jsonl"
    out.write_text("".join(json.dumps(i) + "\n" for i in items))
    counts: dict[str, int] = {}
    for i in items:
        counts[str(i["category"])] = counts.get(str(i["category"]), 0) + 1
    print(f"{len(items)} items -> {out}: {counts}")


if __name__ == "__main__":
    main()
