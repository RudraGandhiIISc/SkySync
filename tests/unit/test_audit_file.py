"""The audit file stays open (regression, FR-6 profile run 2117).

Run 2117 profiled the event loop at 3x load: 74% of its time was spent in
FileAuditSink.write, which opened, flushed and closed audit.jsonl on the
NFS drive for every record (a close waits for the server). Shed requests
then waited seconds for their 429.
"""

from pathlib import Path
from typing import Any

from askiisc.gateway.audit import AuditRecord, FileAuditSink


def record(n: int) -> AuditRecord:
    return AuditRecord(
        request_id=f"r{n}",
        time=float(n),
        as_of=float(n),
        user_hash="u",
        query_hash="q",
        tenant="cds",
        policy_version=1,
        model="m",
        decision="rejected",
    )


def test_the_file_is_opened_once_for_many_records(tmp_path: Path) -> None:
    opened: list[Path] = []

    def opener(path: Path) -> Any:
        opened.append(path)
        return path.open("a", encoding="utf-8")

    sink = FileAuditSink(tmp_path / "audit.jsonl", opener=opener)
    for n in range(100):
        sink.write(record(n))
    assert len(opened) == 1
    # Readable at once, in order, without closing (same process).
    assert [r.request_id for r in sink.read()] == [f"r{n}" for n in range(100)]
    sink.close()
    sink.close()  # closing twice is fine
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 100  # noqa: PLR2004


def test_a_closed_sink_reopens_on_the_next_write(tmp_path: Path) -> None:
    sink = FileAuditSink(tmp_path / "audit.jsonl")
    sink.write(record(1))
    sink.close()
    sink.write(record(2))
    assert [r.request_id for r in sink.read()] == ["r1", "r2"]
