"""FR-5 / E5 check of a `freshness` run: demotion, stale flags, as-of.

Reads only raw files:

- Locust's `events.jsonl`: each publish (document, version, code) and
  each answer the readers got (time sent, codes quoted, citations);
- the serving job's `audit.jsonl` (documents used per request, stale flag);
- the live version store (`versions.db`).

It reports, per revision, the time from its publish to the first answer
sent afterwards that quotes the new code (time to demotion of the old
chunk); the answers that used a superseded version without the stale
flag (from the audit store and from the citations; target 0, including
requests sent inside the reconciliation window); and, at each of the
revisions' checkpoints, whether the version store's as-of lookup gives the
version published then.

Run (inside Slurm):
    python -m eval.freshness_check <locust run> <serve run> <versions.db> <out>
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askiisc.common.types import DocId
from askiisc.ingest.versions import SQLiteVersionStore, VersionStore
from eval.fr6_report import read_lines, spread


@dataclass(frozen=True)
class Publish:
    """One published revision."""

    time: float
    doc: str
    version: int
    code: str


def publishes(events: list[dict[str, Any]]) -> list[Publish]:
    """The run's publishes, in time order."""
    rows = [
        Publish(float(e["time"]), str(e["doc"]), int(e["version"]), e["code"])
        for e in events
        if e.get("kind") == "publish" and e.get("version") is not None
    ]
    return sorted(rows, key=lambda p: p.time)


def superseded_at(pubs: list[Publish]) -> dict[tuple[str, int], float]:
    """When each published version was replaced by the next publish."""
    out: dict[tuple[str, int], float] = {}
    last: dict[str, Publish] = {}
    for pub in pubs:
        if pub.doc in last:
            older = last[pub.doc]
            out[(older.doc, older.version)] = pub.time
        last[pub.doc] = pub
    return out


def replaced_in(
    versions: VersionStore, docs: set[str]
) -> dict[tuple[str, int], float]:
    """When the version store retired each version of the given documents.

    The server makes a revision current when it commits, a few seconds
    after the publish call returns, so these times (not the publish
    events') decide whether an answer used a superseded version.
    """
    out: dict[tuple[str, int], float] = {}
    for doc in docs:
        for record in versions.history(DocId(doc)):
            if record.valid_to is not None:
                out[(doc, int(record.version))] = float(record.valid_to)
    return out


def demotion_s(
    pubs: list[Publish], answers: list[dict[str, Any]]
) -> list[float]:
    """Seconds from each publish to the first later answer quoting it."""
    out = []
    for pub in pubs:
        later = [
            float(a["sent"])
            for a in answers
            if a["doc"] == pub.doc
            and float(a["sent"]) >= pub.time
            and pub.code in a.get("codes", [])
        ]
        if later:
            out.append(min(later) - pub.time)
    return out


def unflagged_cited(
    answers: list[dict[str, Any]], replaced: dict[tuple[str, int], float]
) -> int:
    """Answers citing a version replaced before they were sent, unflagged."""
    bad = 0
    for answer in answers:
        sent = float(answer["sent"])
        for c in answer.get("citations", []):
            ended = replaced.get((str(c["doc_id"]), int(c["version"])))
            if ended is not None and ended <= sent and not c.get("is_stale"):
                bad += 1
                break
    return bad


def unflagged_used(
    audit: list[dict[str, Any]], replaced: dict[tuple[str, int], float]
) -> tuple[int, int]:
    """Audit records that used a replaced version: (all, without the flag)."""
    used = unflagged = 0
    for record in audit:
        start = float(record["time"]) - float(record.get("latency_ms", 0)) / 1e3
        hit = any(
            (ended := replaced.get((d["doc_id"], int(d["version"]))))
            is not None
            and ended <= start
            for d in record.get("documents", [])
        )
        if hit and record.get("decision") in {"answered", "abstained"}:
            used += 1
            unflagged += not record.get("stale", False)
    return used, unflagged


AS_OF_LAST_S = 60.0
"""Checkpoint after a document's last revision: this long after it."""


def as_of_checks(
    pubs: list[Publish], versions: VersionStore
) -> list[dict[str, Any]]:
    """The version store's as-of answer at each revision's checkpoint.

    The checkpoint is half-way between a revision's publish and the next
    publish of the same document (AS_OF_LAST_S after the last one), so it
    falls after the commit and before the next revision.
    """
    out = []
    for i, pub in enumerate(pubs):
        later = [p.time for p in pubs[i + 1 :] if p.doc == pub.doc]
        end = later[0] if later else pub.time + 2 * AS_OF_LAST_S
        record = versions.as_of(DocId(pub.doc), (pub.time + end) / 2)
        got = None if record is None else int(record.version)
        out.append({"doc": pub.doc, "time": pub.time, "expected": pub.version,
                    "got": got, "ok": got == pub.version})  # fmt: skip
    return out


def check(locust: Path, serve: Path, versions: VersionStore) -> dict[str, Any]:
    """The FR-5 numbers of one run."""
    events = read_lines(locust / "events.jsonl")
    pubs = publishes(events)
    answers = [e for e in events if e.get("kind") == "answer"]
    replaced = replaced_in(versions, {p.doc for p in pubs})
    audit = read_lines(serve / "audit.jsonl")
    used, unflagged = unflagged_used(audit, replaced)
    checkpoints = as_of_checks(pubs, versions)
    return {
        "revisions": len(pubs),
        "documents": len({p.doc for p in pubs}),
        "answers": len(answers),
        "demotion_s": spread(demotion_s(pubs, answers)),
        "unflagged_cited": unflagged_cited(answers, replaced),
        "audit_used_superseded": used,
        "audit_unflagged": unflagged,
        "as_of_ok": sum(c["ok"] for c in checkpoints),
        "as_of_checkpoints": checkpoints,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Writes freshness.json and prints the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("locust", "serve", "versions", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    result = check(args.locust, args.serve, SQLiteVersionStore(args.versions))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "freshness.json").write_text(json.dumps(result, indent=1))
    headline = {k: v for k, v in result.items() if k != "as_of_checkpoints"}
    print(json.dumps(headline))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
