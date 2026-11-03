"""Builds the edition-aware question set (400 items) from the real index.

Kinds of item (QUOTAS):
  current       a fact that changed between two editions, asked without a
                year: the answer must come from the newest edition
  year          the same kind of fact asked for a named academic year: the
                answer must come from the edition in force that year
  change        "how did X change between year A and year B": both facts
  general       a fact from a document with one edition (as before)
  unanswerable  facts no IISc document holds: the system must refuse

Changed facts: for each pair of consecutive editions of a family
(catalog.jsonl), the new edition's chunks are aligned with their most
similar old chunk; similar but not identical pairs go to an LLM (Qwen3-8B,
thinking off), which names one fact that differs and copies both values.
An item is kept only if the new value appears verbatim in the new chunk
and nowhere in the old edition, and the old value in the old chunk and
nowhere in the new edition. Every item is reviewed=false until a person
checks it.

Usage (inside slurm/llm_job.sh with MODEL=qwen3-8b):
    python -m eval.versioned.generate --state DIR --corpus DIR --out FILE
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from askiisc.ingest.catalog import SERIES_WORDS
from askiisc.ingest.crawl import PRIORITY_CATEGORIES, categorise
from eval.heldout.generate import SKIP_URL_WORDS, grounded, norm

QUOTAS = {
    "current": 120,
    "year": 100,
    "change": 40,
    "general": 100,
    "unanswerable": 40,
}
# Of the "year" items, this many name the current year (the rest a past one).
CURRENT_YEAR_ITEMS = 20
MAX_PER_FAMILY = 3
# Aligned chunk pairs offered to the LLM per pair of editions.
MAX_TRIES = 8
TITLES = frozenset({"dr", "prof", "mr", "ms", "mrs", "shri", "smt"})
MIN_ALIGN = 0.25
MAX_ALIGN = 0.97
MIN_CHUNK_CHARS = 200
PROMPT_CHUNK_CHARS = 2500
WORKERS = 16
ACADEMIC_START_MONTH = 8
ACADEMIC_PROBE_MONTH = 10
ACADEMIC_END_MONTH = 7
EARLIEST_YEAR = 2010
USER = "stu_cds"
_WORD = re.compile(r"[a-z0-9]+")
_YEAR = re.compile(r"(?<!\d)20\d\d(?!\d)")

CHANGE_PROMPT = """Two editions of the same IISc document: {title}
NEWER EDITION ({new_label}):
<<<
{new}
>>>
These phrases differ between the older edition ({old_label}) and the newer
one (the words just before each change are shown for context):
{changes}
Pick ONE difference that is a real fact (a number, date, fee, credit count,
course, rule, eligibility condition, person in charge or name of a
programme component), not a typo or formatting change. Write a question a
student or applicant would ask whose answer is that fact. The question must
make sense to someone who has not seen the document, so it must contain
some of these words: {subject}. It must NOT mention any year, edition or
version, and must not contain either value. new_value and old_value are
the newer and older phrases, copied EXACTLY (you may add a few neighbouring
words from the text so each reads as an answer). The topic names what
changed and whose (for example "the DS252 assignment weightage"). Reply
with JSON only:
{{"question": "...", "topic": "...", "new_value": "...", "old_value": "..."}}
If none of the differences is a real fact, reply {{"question": null}}."""

GENERAL_PROMPT = """Text from the IISc page or document "{title}":
<<<
{text}
>>>
Write ONE question that a student, applicant or staff member might ask about
IISc rules, programmes, admissions, fees, dates, eligibility, courses or
procedures, and that this text answers. The question must make sense to
someone who has not seen the text, so it must contain some of these words:
{subject} (never say "the document", "the course" or "mentioned").
Never ask about a named individual or research findings. Give
the answer as a SHORT phrase (1 to 8 words) copied exactly from the text,
not contained in the question. Reply with JSON only:
{{"question": "...", "answer": "..."}}
If the text has no clear, specific fact, reply {{"question": null}}."""
VERIFY_PROMPT = """Text:
<<<
{text}
>>>
Question: {question}
Proposed answer: {answer}
Is the question specific and sensible for a student or applicant, and
does the text clearly answer it with exactly the proposed answer (not a
neighbouring value, a table fragment or a form field)? The question must
also have ONE correct answer across all of IISc: it names the department,
programme, course, scheme or event it is about. "What is the stipend for
fellowships?" or "When is the application deadline?" could have many
answers, so they are not ok. Reply with JSON only: {{"ok": true}} or
{{"ok": false}}"""
_JUNK = re.compile(r"\.{4,}|_{4,}|-{4,}")
MAX_ANSWER_NUMBERS = 4
# Words too common in titles to tie a question to its document.
GENERIC_WORDS = frozenset(
    [
        "about",
        "after",
        "also",
        "and",
        "are",
        "before",
        "being",
        "between",
        "brochure",
        "centre",
        "course",
        "courses",
        "department",
        "details",
        "document",
        "during",
        "each",
        "first",
        "from",
        "have",
        "iisc",
        "india",
        "indian",
        "information",
        "institute",
        "into",
        "must",
        "page",
        "programme",
        "programmes",
        "program",
        "programs",
        "science",
        "should",
        "student",
        "students",
        "their",
        "there",
        "these",
        "this",
        "through",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "year",
        "years",
        "bangalore",
        "bengaluru",
        "home",
        "welcome",
    ]
)
_CODE = re.compile(r"\b([a-z]{1,3})\s?-?(\d{3})\b")
_VAGUE = re.compile(
    r"\b(?:the|this) (?:document|text|page|pdf|file|brochure|notice)\b"
    r"|\bmentioned\b|\bthe course\?"
)


def subject_words(row: dict[str, Any]) -> set[str]:
    """Distinctive words of a document's title and address."""
    path = str(row.get("original_url") or row.get("url") or "").lower()
    tail = " ".join(path.strip("/").split("/")[-2:])
    text = f"{row.get('title') or ''} {tail}".lower()
    found = {w for w in _WORD.findall(text) if len(w) >= 4}  # noqa: PLR2004
    codes = {a + b for a, b in _CODE.findall(text)}
    return (found - GENERIC_WORDS) | codes


def anchored(question: str, row: dict[str, Any]) -> bool:
    """Whether a question names what it is about (title word or code)."""
    q = question.lower()
    if _VAGUE.search(q):
        return False
    mine = set(_WORD.findall(q)) | {a + b for a, b in _CODE.findall(q)}
    return bool(mine & subject_words(row))


EXTRA_UNANSWERABLE = (
    "What is the M.Tech fee at IIT Madras for 2026?",
    "When is the JEE Advanced 2027 exam?",
    "What is the hostel curfew at IIT Delhi?",
    "Who won the 2026 FIFA World Cup?",
    "What is the placement average of IIM Bangalore?",
    "How many credits is the cloud computing course at NIT Trichy?",
    "What is the CAT 2026 cut-off for IIM Ahmedabad?",
    "What is the PhD stipend at Stanford University?",
    "When does Bangalore University announce its BSc results?",
    "What is the syllabus of CS101 at IIT Kanpur?",
)


@dataclass(frozen=True)
class Piece:
    """One stored chunk."""

    chunk_id: str
    text: str


def words(text: str) -> set[str]:
    """Word 3-grams of a text."""
    w = _WORD.findall(text.lower())
    return {" ".join(w[i : i + 3]) for i in range(max(1, len(w) - 2))}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets (0 when both are empty)."""
    return len(a & b) / len(a | b) if a | b else 0.0


def load_catalog(corpus: Path) -> dict[str, dict[str, Any]]:
    """catalog.jsonl rows by document id."""
    rows = (json.loads(ln) for ln in (corpus / "catalog.jsonl").open())
    return {r["doc_id"]: r for r in rows}


def load_chunks(state: Path, doc_ids: Iterable[str]) -> dict[str, list[Piece]]:
    """The newest version's chunks of each document."""
    wanted = set(doc_ids)
    db = sqlite3.connect(f"file:{state / 'versions.db'}?mode=ro", uri=True)
    out: dict[str, tuple[int, list[Piece]]] = {}
    for doc_id, version, drafts in db.execute(
        "SELECT doc_id, version, drafts FROM versions"
    ):
        if doc_id not in wanted or out.get(doc_id, (0, []))[0] > version:
            continue
        pieces = [
            Piece(f"{doc_id}@v{version}#{i}", d["text"])
            for i, d in enumerate(json.loads(drafts))
        ]
        out[doc_id] = (version, pieces)
    return {d: p for d, (_, p) in out.items()}


def edition_pairs(
    catalog: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(newer, older) consecutive editions of every family, newest first."""
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog.values():
        if int(row.get("editions") or 1) > 1:
            families[row["family"]].append(row)
    pairs = []
    for members in families.values():
        members.sort(key=lambda r: r["valid_from"])
        for older, newer in zip(members, members[1:], strict=False):
            if newer["valid_from"] > older["valid_from"]:
                pairs.append((newer, older))
    pairs.sort(
        key=lambda p: (p[0]["valid_to"] is not None, -p[0]["valid_from"])
    )
    return pairs


MAX_CHANGE_WORDS = 12
CONTEXT_WORDS = 10
MAX_SPANS = 6


def changed_spans(old: str, new: str) -> list[tuple[str, str, str]]:
    """Short phrases replaced between two texts, with preceding context.

    Returns:
        (context, old phrase, new phrase) for each replacement of at most
        MAX_CHANGE_WORDS words whose wording really differs.
    """
    a, b = old.split(), new.split()
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or max(i2 - i1, j2 - j1) > MAX_CHANGE_WORDS:
            continue
        before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if canonical_value(before) == canonical_value(after):
            continue
        context = " ".join(b[max(0, j1 - CONTEXT_WORDS) : j1])
        spans.append((context, before, after))
    return spans[:MAX_SPANS]


def align(
    new: Sequence[Piece], old: Sequence[Piece]
) -> list[tuple[Piece, Piece, float]]:
    """Each long new chunk with its most similar old chunk, if changed."""
    old_sets = [
        (p, words(p.text)) for p in old if len(p.text) >= MIN_CHUNK_CHARS
    ]
    out = []
    for piece in new:
        if len(piece.text) < MIN_CHUNK_CHARS or not old_sets:
            continue
        mine = words(piece.text)
        best, sim = max(
            ((p, jaccard(mine, s)) for p, s in old_sets), key=lambda x: x[1]
        )
        if MIN_ALIGN <= sim < MAX_ALIGN:
            out.append((piece, best, sim))
    return out


def academic_year(row: dict[str, Any], now: float) -> int | None:
    """The academic year an edition belongs to, if a search sees it then.

    The year that contains the edition's start (a January 2021 offering
    is 2020-21). A question naming that year is searched at the year's
    start and end (retrieval/temporal.py); the edition must be in force at
    one of them, or no question names its year.
    """
    start = row.get("edition_start") or row["valid_from"]
    if not start:
        return None
    when = datetime.fromtimestamp(start, UTC)
    year = when.year if when.month >= ACADEMIC_START_MONTH else when.year - 1
    end = row["valid_to"] if row["valid_to"] is not None else float("inf")
    probes = [
        datetime(year, ACADEMIC_PROBE_MONTH, 1, tzinfo=UTC).timestamp(),
        datetime(year + 1, ACADEMIC_END_MONTH, 1, tzinfo=UTC).timestamp(),
    ]
    seen = any(row["valid_from"] <= min(p, now) < end for p in probes)
    return year if year >= EARLIEST_YEAR and seen else None


def ay_label(year: int) -> str:
    """'2025-26' for the academic year starting in 2025."""
    return f"{year}-{(year + 1) % 100:02d}"


def chat(http: httpx.Client, prompt: str, max_tokens: int = 300) -> dict | None:
    """One JSON reply from the generator model (thinking off)."""
    body = {
        "model": os.environ["ASKIISC_LLM_MODEL"],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = http.post(
            os.environ["ASKIISC_LLM_URL"] + "/v1/chat/completions",
            json=body,
            timeout=120,
        )
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content[content.index("{") : content.rindex("}") + 1])
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None
    return data if isinstance(data, dict) else None


def canonical_value(value: str) -> str:
    """A value's words without punctuation or titles ("Dr.", "Prof.")."""
    return " ".join(w for w in _WORD.findall(value.lower()) if w not in TITLES)


def change_problem(
    item: dict[str, Any],
    new: Piece,
    old: Piece,
    new_doc: str,
    old_doc: str,
    row: dict[str, Any] | None = None,
) -> str | None:
    """Why a proposed change is unusable, or None if it is good.

    Both values must be verbatim in their own edition only, really differ,
    and the question must name its subject and no year.
    """
    q = str(item.get("question") or "")
    nv, ov = str(item.get("new_value") or ""), str(item.get("old_value") or "")
    if not q or not nv or not ov:
        return "no change proposed"
    if canonical_value(nv) == canonical_value(ov):
        return "same value"
    if _YEAR.search(q) or re.search(r"\b(edition|version)\b", q.lower()):
        return "year in question"
    if not (grounded(q, nv, new.text) and grounded(q, ov, old.text)):
        return "value not verbatim"
    if norm(nv) in norm(old_doc) or norm(ov) in norm(new_doc):
        return "value in both editions"
    if row is not None and not anchored(q, row):
        return "subject not named"
    if not (plausible_answer(nv) and plausible_answer(ov)):
        return "not a fact"
    return None


def plausible_answer(answer: str) -> bool:
    """Not form filler ("....") and not a row of table numbers."""
    numbers = re.findall(r"\d+", answer)
    return not _JUNK.search(answer) and len(numbers) <= MAX_ANSWER_NUMBERS


def verified(http: httpx.Client, question: str, answer: str, text: str) -> bool:
    """A second look by the model: does the text answer exactly this?"""
    reply = chat(
        http,
        VERIFY_PROMPT.format(
            text=text[:PROMPT_CHUNK_CHARS], question=question, answer=answer
        ),
        20,
    )
    return bool(reply and reply.get("ok") is True)


def valid_change(
    item: dict[str, Any],
    new: Piece,
    old: Piece,
    new_doc: str,
    old_doc: str,
    row: dict[str, Any] | None = None,
) -> bool:
    """Whether a proposed change is usable (see change_problem)."""
    return change_problem(item, new, old, new_doc, old_doc, row) is None


def lower_first(question: str) -> str:
    """The question with its first letter lower-cased (for a prefix)."""
    return question[:1].lower() + question[1:] if question else question


def change_items(
    found: dict[str, Any],
    newer: dict[str, Any],
    older: dict[str, Any],
    now: float,
) -> list[dict[str, Any]]:
    """The current, year and change items one changed fact yields."""
    q = found["question"].strip()
    base = {
        "user": USER,
        "family": newer["family"],
        "reviewed": False,
        "generated_by": f"{os.environ['ASKIISC_LLM_MODEL']}; values verbatim",
    }
    new_docs, old_docs = [newer["doc_id"]], [older["doc_id"]]
    new_fact = {
        "key_facts": [found["new_value"]],
        "reference_answer": found["new_value"],
    }
    old_fact = {
        "key_facts": [found["old_value"]],
        "reference_answer": found["old_value"],
    }
    items = []
    current = newer["valid_to"] is None
    new_year, old_year = academic_year(newer, now), academic_year(older, now)
    if current:
        items.append(
            {
                **base,
                **new_fact,
                "type": "current",
                "question": q,
                "expected_docs": new_docs,
                "wrong_docs": old_docs,
                "edition": newer["edition"],
                "chunk_id": found["new_chunk"],
            }
        )
    if old_year is not None:
        items.append(
            {
                **base,
                **old_fact,
                "type": "year",
                "question": (
                    f"For the {ay_label(old_year)} academic year, "
                    f"{lower_first(q)}"
                ),
                "expected_docs": old_docs,
                "wrong_docs": new_docs,
                "edition": older["edition"],
                "year": old_year,
                "chunk_id": found["old_chunk"],
            }
        )
    if current and new_year is not None:
        items.append(
            {
                **base,
                **new_fact,
                "type": "year-current",
                "question": f"In {ay_label(new_year)}, {lower_first(q)}",
                "expected_docs": new_docs,
                "wrong_docs": old_docs,
                "edition": newer["edition"],
                "year": new_year,
                "chunk_id": found["new_chunk"],
            }
        )
    if old_year is not None and new_year is not None and old_year != new_year:
        span = f"{ay_label(old_year)} and {ay_label(new_year)}"
        topic = lower_first(str(found.get("topic") or "").strip()).rstrip("?.")
        question = (
            f"How did {topic} change between {span}?"
            if topic and anchored(topic, newer)
            else f"Between {span}, how did this change: {lower_first(q)}"
        )
        items.append(
            {
                **base,
                "type": "change",
                "question": question,
                "key_facts": [found["old_value"], found["new_value"]],
                "reference_answer": (
                    f"{found['old_value']} -> {found['new_value']}"
                ),
                "expected_docs": old_docs + new_docs,
                "wrong_docs": [],
                "edition": f"{older['edition']} -> {newer['edition']}",
                "chunk_id": found["new_chunk"],
            }
        )
    return items


def find_change(
    http: httpx.Client,
    newer: dict[str, Any],
    older: dict[str, Any],
    chunks: dict[str, list[Piece]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """The first aligned chunk pair the LLM finds a valid change in.

    Returns:
        The change (or None) and why each rejected proposal was rejected.
    """
    new, old = chunks.get(newer["doc_id"]), chunks.get(older["doc_id"])
    if not new or not old:
        return None, ["no chunks"]
    new_doc = " ".join(p.text for p in new)
    old_doc = " ".join(p.text for p in old)
    pairs = [
        (piece, match, spans)
        for piece, match, _ in sorted(align(new, old), key=lambda x: -x[2])
        if (spans := changed_spans(match.text, piece.text))
    ]
    if not pairs:
        return None, ["no changed phrases"]
    problems: list[str] = []
    subject = ", ".join(sorted(subject_words(newer))[:12])
    for piece, match, spans in pairs[:MAX_TRIES]:
        changes = "\n".join(
            f'- after "...{ctx}": older "{before}" -> newer "{after}"'
            for ctx, before, after in spans
        )
        found = chat(
            http,
            CHANGE_PROMPT.format(
                title=newer.get("title") or "",
                old_label=older["edition"] or "older",
                new_label=newer["edition"] or "newer",
                new=piece.text[:PROMPT_CHUNK_CHARS],
                changes=changes,
                subject=subject,
            ),
        )
        problem = (
            change_problem(found, piece, match, new_doc, old_doc, newer)
            if found
            else "no reply"
        )
        if (
            problem is None
            and found
            and not (
                verified(
                    http, found["question"], found["new_value"], piece.text
                )
                and verified(
                    http, found["question"], found["old_value"], match.text
                )
            )
        ):
            problem = "failed check"
        if problem is None and found:
            change = {
                **found,
                "new_chunk": piece.chunk_id,
                "old_chunk": match.chunk_id,
            }
            return change, problems
        problems.append(problem or "")
    return None, problems


def versioned_items(
    catalog: dict[str, dict[str, Any]],
    state: Path,
    now: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Items from facts that changed between editions."""
    pairs = edition_pairs(catalog)
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: p[0]["valid_to"] is not None)  # current first
    chunks = load_chunks(state, {r["doc_id"] for p in pairs for r in p})
    per_family: Counter[str] = Counter()
    jobs = []
    for newer, older in pairs:
        if per_family[newer["family"]] < MAX_PER_FAMILY:
            per_family[newer["family"]] += 1
            jobs.append((newer, older))
    items: list[dict[str, Any]] = []
    with httpx.Client() as http, ThreadPoolExecutor(WORKERS) as pool:
        found = pool.map(lambda p: find_change(http, p[0], p[1], chunks), jobs)
        problems: Counter[str] = Counter()
        for (newer, older), (change, why) in zip(jobs, found, strict=True):
            problems.update(why)
            if change:
                items.extend(change_items(change, newer, older, now))
    print(f"edition pairs tried {len(jobs)}, items {len(items)}")
    print(f"rejected proposals: {dict(problems)}")
    return items


def general_items(
    catalog: dict[str, dict[str, Any]], state: Path, rng: random.Random, n: int
) -> list[dict[str, Any]]:
    """Facts from single-edition documents, spread over departments."""
    docs = [
        r
        for r in catalog.values()
        if int(r.get("editions") or 1) == 1
        and r.get("snapshot") is None
        and not any(w in r["url"].lower() for w in SKIP_URL_WORDS)
        and categorise(r["original_url"], r.get("title") or "")
        in PRIORITY_CATEGORIES
        and not any(
            w in f"{r['original_url']} {r.get('title') or ''}".lower()
            for w in SERIES_WORDS
        )
    ]
    by_dept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in docs:
        by_dept[r["department"]].append(r)
    order = []
    for rows in by_dept.values():
        rng.shuffle(rows)
    while any(by_dept.values()) and len(order) < n * 4:
        for rows in by_dept.values():
            if rows:
                order.append(rows.pop())
    chunks = load_chunks(state, {r["doc_id"] for r in order})
    picks = []
    for r in order:
        good = [
            p for p in chunks.get(r["doc_id"], []) if 300 <= len(p.text) <= 1500
        ]
        if good:
            picks.append((r, rng.choice(good)))
    items: list[dict[str, Any]] = []
    with httpx.Client() as http, ThreadPoolExecutor(WORKERS) as pool:
        replies = pool.map(
            lambda x: chat(
                http,
                GENERAL_PROMPT.format(
                    title=x[0].get("title"),
                    text=x[1].text,
                    subject=", ".join(sorted(subject_words(x[0]))[:12]),
                ),
                200,
            ),
            picks,
        )
        proposed = []
        for (row, piece), reply in zip(picks, replies, strict=True):
            if not reply or not reply.get("question"):
                continue
            q = str(reply["question"]).strip()
            a = str(reply.get("answer") or "").strip()
            if (
                a
                and grounded(q, a, piece.text)
                and anchored(q, row)
                and plausible_answer(a)
            ):
                proposed.append((row, piece, q, a))
        checks = pool.map(
            lambda x: verified(http, x[2], x[3], x[1].text), proposed
        )
        for (row, piece, q, a), ok in zip(proposed, checks, strict=True):
            if ok and len(items) < n:
                items.append(
                    {
                        "type": "general",
                        "user": USER,
                        "question": q,
                        "reference_answer": a,
                        "key_facts": [a],
                        "expected_docs": [row["doc_id"]],
                        "wrong_docs": [],
                        "chunk_id": piece.chunk_id,
                        "family": row["family"],
                        "edition": row.get("edition") or "",
                        "reviewed": False,
                        "generated_by": (
                            f"{os.environ['ASKIISC_LLM_MODEL']}; "
                            "answer verbatim"
                        ),
                    }
                )
    return items


def unanswerable_items(heldout: Path) -> list[dict[str, Any]]:
    """The 30 reviewed refusals of held-out v2 plus 10 more."""
    rows = [json.loads(ln) for ln in heldout.open()]
    old = [r["question"] for r in rows if r.get("category") == "unanswerable"]
    return [
        {
            "type": "unanswerable",
            "user": USER,
            "question": q,
            "key_facts": [],
            "reference_answer": "",
            "expected_docs": [],
            "wrong_docs": [],
            "reviewed": q in old,
            "generated_by": "hand-written",
        }
        for q in [*old, *EXTRA_UNANSWERABLE]
    ]


def pick(
    items: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Fills each quota; spare general items cover any shortfall."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_type[item["type"]].append(item)
    for rows in by_type.values():
        rng.shuffle(rows)
    chosen = by_type["current"][: QUOTAS["current"]]
    past = by_type["year"][: QUOTAS["year"] - CURRENT_YEAR_ITEMS]
    now_year = by_type["year-current"][: QUOTAS["year"] - len(past)]
    for row in now_year:
        row["type"] = "year"
    chosen += past + now_year + by_type["change"][: QUOTAS["change"]]
    chosen += by_type["unanswerable"][: QUOTAS["unanswerable"]]
    short = sum(QUOTAS.values()) - len(chosen)
    chosen += by_type["general"][:short]
    for i, item in enumerate(chosen, 1):
        item["id"] = f"v{i:03d}"
    return chosen


def main() -> int:
    """Writes the question set and prints the counts per kind."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--heldout",
        type=Path,
        default=Path("eval/datasets/heldout/questions.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=252)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    now = datetime.now(UTC).timestamp()
    catalog = load_catalog(args.corpus)
    items = versioned_items(catalog, args.state, now, rng)
    found = Counter(i["type"] for i in items)
    needed = (
        sum(QUOTAS.values())
        - min(found["current"], QUOTAS["current"])
        - min(found["year"] + found["year-current"], QUOTAS["year"])
        - min(found["change"], QUOTAS["change"])
        - QUOTAS["unanswerable"]
    )
    items += general_items(catalog, args.state, rng, needed + 20)
    items += unanswerable_items(args.heldout)
    chosen = pick(items, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(i) + "\n" for i in chosen))
    counts = Counter(i["type"] for i in chosen)
    print(json.dumps({"total": len(chosen), "by_type": counts, "found": found}))
    return 0 if len(chosen) == sum(QUOTAS.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
