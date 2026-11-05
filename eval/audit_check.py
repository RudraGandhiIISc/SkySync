"""FR-12 check of one serving run's audit store against what clients saw.

Given the run folder (audit.jsonl, requests.log) and the load generator's
client log, it checks:

1. completeness: every request the client sent (by the `X-Request-Id` it
   got back) and every request in the gateway's request log has exactly
   one audit record;
2. reproduction: each answer's citations (doc_id, version, chunk_id) are
   among the documents and chunks its record names;
3. privacy: no question string the load test can send, and no user id,
   appears anywhere in the audit store.

It also reports p50 and p95 of every latency stage, among them `other_ms`
(time no named stage explains: p95 above 50 ms means a stage is untimed),
the refusal reasons, and the client's statuses (for error shares). Usage:

    python -m eval.audit_check RUN_DIR CLIENT_LOG OUT_DIR \
        [--questions FILE.jsonl ...] [--users FILE.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from askiisc.gateway.audit import AuditRecord, LatencyBreakdown
from askiisc.gateway.cache import normalise
from eval.locust.client_log import ClientRecord, read_client_log
from eval.locust.questions import FOLLOW_UPS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    REPO / "eval" / "datasets" / "heldout" / "questions.jsonl",
)
# Strings shorter than this are too generic to search for (they would match
# hashes or field names by chance).
MIN_SEARCH_CHARS = 12
MIN_USER_ID_CHARS = 4


@dataclass
class AuditReport:
    """The result of the three checks."""

    client_requests: int = 0
    gateway_requests: int = 0
    audit_records: int = 0
    missing_client: list[str] = field(default_factory=list)
    missing_gateway: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    client_without_id: int = 0
    answers_checked: int = 0
    citations_checked: int = 0
    unreproduced: list[str] = field(default_factory=list)
    strings_searched: int = 0
    raw_text_hits: int = 0
    user_id_hits: int = 0
    other_ms_p50: float = 0.0
    other_ms_p95: float = 0.0
    stages_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    by_decision: dict[str, int] = field(default_factory=dict)
    by_reason: dict[str, int] = field(default_factory=dict)
    client_status: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when every FR-12 check holds."""
        return not (
            self.missing_client
            or self.missing_gateway
            or self.duplicated
            or self.client_without_id
            or self.unreproduced
            or self.raw_text_hits
            or self.user_id_hits
        )


def read_audit(path: Path) -> list[AuditRecord]:
    """Reads audit.jsonl (every line must be a valid record)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AuditRecord.model_validate_json(ln) for ln in lines if ln]


def read_gateway_ids(path: Path) -> list[str]:
    """Request ids from the gateway's request log (requests.log)."""
    if not path.exists():
        return []
    return [
        json.loads(ln)["request_id"]
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln
    ]


def question_strings(files: Iterable[Path]) -> list[str]:
    """Every question text the load test can send."""
    out = list(FOLLOW_UPS)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                out.append(str(json.loads(line)["question"]))
    return out


def completeness(
    report: AuditReport,
    records: Sequence[AuditRecord],
    client: Sequence[ClientRecord],
    gateway_ids: Sequence[str],
) -> None:
    """Fills the completeness fields of `report`."""
    counts = Counter(r.request_id for r in records)
    report.audit_records = len(records)
    report.duplicated = sorted(k for k, n in counts.items() if n > 1)
    ids = [c.request_id for c in client if c.request_id]
    report.client_requests = len(client)
    report.client_without_id = len(client) - len(ids)
    report.missing_client = sorted(set(ids) - set(counts))
    report.gateway_requests = len(gateway_ids)
    report.missing_gateway = sorted(set(gateway_ids) - set(counts))
    report.by_decision = dict(Counter(r.decision for r in records))
    report.by_reason = dict(Counter(r.reason or "none" for r in records))
    report.client_status = dict(Counter(str(c.status) for c in client))


def reproduction(
    report: AuditReport,
    records: Sequence[AuditRecord],
    client: Sequence[ClientRecord],
) -> None:
    """Checks each answer's citations against its record."""
    by_id = {r.request_id: r for r in records}
    for row in client:
        record = by_id.get(row.request_id or "")
        if record is None or row.status != 200 or not row.citations:  # noqa: PLR2004
            continue
        report.answers_checked += 1
        versions = {(d.doc_id, d.version) for d in record.documents}
        chunks = set(record.chunk_ids)
        for doc_id, version, chunk_id in row.citations:
            report.citations_checked += 1
            if (doc_id, version) not in versions or chunk_id not in chunks:
                report.unreproduced.append(record.request_id)
                break


def privacy(
    report: AuditReport,
    raw: str,
    questions: Sequence[str],
    user_ids: Sequence[str],
) -> None:
    """Searches the raw audit store for question text and user ids."""
    haystack = raw.lower()
    searched = sorted(
        {q.strip() for q in questions if len(q.strip()) >= MIN_SEARCH_CHARS}
    )
    report.strings_searched = len(searched)
    report.raw_text_hits = sum(
        1 for q in searched if q.lower() in haystack or normalise(q) in haystack
    )
    report.user_id_hits = sum(
        1 for u in user_ids if len(u) >= MIN_USER_ID_CHARS and u in raw
    )


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def other_time(report: AuditReport, records: Sequence[AuditRecord]) -> None:
    """p50 and p95 of each stage over records that went past admission."""
    served = [r for r in records if r.decision in ("answered", "abstained")]
    if not served:
        return
    for name in LatencyBreakdown.model_fields:
        values = [getattr(r.latency, name) for r in served]
        report.stages_ms[name] = {
            "p50": statistics.median(values),
            "p95": _pct(values, 0.95),
        }
    report.other_ms_p50 = report.stages_ms["other_ms"]["p50"]
    report.other_ms_p95 = report.stages_ms["other_ms"]["p95"]


def check_run(
    run_dir: Path,
    client_log: Path,
    questions: Sequence[Path] = DEFAULT_QUESTIONS,
    user_ids: Sequence[str] = (),
) -> AuditReport:
    """Runs every FR-12 check on one serving run.

    Args:
        run_dir: The serving run folder (audit.jsonl, requests.log).
        client_log: The load generator's client_requests.jsonl.
        questions: Question sets the load test drew from.
        user_ids: Identities that must never appear in the audit store.

    Returns:
        The report.
    """
    audit_path = run_dir / "audit.jsonl"
    records = read_audit(audit_path)
    client = read_client_log(client_log)
    report = AuditReport()
    completeness(
        report, records, client, read_gateway_ids(run_dir / "requests.log")
    )
    reproduction(report, records, client)
    privacy(
        report,
        audit_path.read_text(encoding="utf-8"),
        question_strings(questions),
        user_ids,
    )
    other_time(report, records)
    return report


def summary(
    report: AuditReport, run_dir: Path, client_log: Path
) -> dict[str, Any]:
    """The summary.json content: counts, the verdict and the sources."""
    data = asdict(report)
    for key in (
        "missing_client",
        "missing_gateway",
        "duplicated",
        "unreproduced",
    ):
        data[f"{key}_count"] = len(data[key])
        data[key] = data[key][:20]  # a sample; the counts are complete
    return {
        "requirement": "FR-12",
        "passed": report.passed,
        "sources": {
            "audit": str(run_dir / "audit.jsonl"),
            "requests": str(run_dir / "requests.log"),
            "client": str(client_log),
        },
        **data,
    }


def _users(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    return [u["user_id"] for u in data.get("users", [])]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exits 1 if any check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("client_log", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--questions", type=Path, nargs="*", default=list(DEFAULT_QUESTIONS)
    )
    parser.add_argument("--users", type=Path, default=None)
    args = parser.parse_args(argv)
    report = check_run(
        args.run_dir, args.client_log, args.questions, _users(args.users)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = summary(report, args.run_dir, args.client_log)
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {k: v for k, v in out.items() if not isinstance(v, list)}, indent=1
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
