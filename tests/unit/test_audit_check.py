"""eval/audit_check.py on synthetic run folders (FR-12)."""

import json
from pathlib import Path

import pytest
from eval.audit_check import check_run, main
from eval.locust.client_log import ClientLog, ClientRecord, read_client_log

from askiisc.gateway.audit import (
    AuditRecord,
    DocVersion,
    LatencyBreakdown,
    salted_hash,
)

QUESTION = "What are the hostel fee rules for 2025?"


def _record(
    rid: str, decision: str = "answered", other: float = 3.0
) -> AuditRecord:
    return AuditRecord(
        request_id=rid,
        time=1.0,
        as_of=1.0,
        user_hash=salted_hash("s", "stu_csa"),
        query_hash=salted_hash("s", QUESTION.lower()),
        tenant="csa",
        policy_version=4,
        documents=(DocVersion(doc_id="d1", version=2),),
        chunk_ids=("d1@v2#0", "d1@v2#1"),
        model="qwen3-8b",
        decision=decision,  # type: ignore[arg-type]
        latency_ms=100.0,
        latency=LatencyBreakdown(embed_ms=97.0 - other, other_ms=other),
    )


def _run(
    tmp: Path,
    records: list[AuditRecord],
    client: list[ClientRecord],
    gateway: list[str] | None = None,
    extra: str = "",
) -> tuple[Path, Path, Path]:
    run = tmp / "run"
    run.mkdir()
    lines = "".join(r.model_dump_json() + "\n" for r in records)
    (run / "audit.jsonl").write_text(lines + extra)
    ids = gateway if gateway is not None else [r.request_id for r in records]
    (run / "requests.log").write_text(
        "".join(json.dumps({"request_id": i}) + "\n" for i in ids)
    )
    log = ClientLog()
    for row in client:
        log.add(row)
    log.write(tmp / "client.jsonl")
    questions = tmp / "q.jsonl"
    questions.write_text(json.dumps({"id": "a1", "question": QUESTION}) + "\n")
    return run, tmp / "client.jsonl", questions


def _answer(
    rid: str, cited: tuple[str, int, str] = ("d1", 2, "d1@v2#1")
) -> ClientRecord:
    return ClientRecord(1.0, rid, 200, "a1", (cited,), False, 1.2)


def test_a_clean_run_passes(tmp_path: Path) -> None:
    records = [_record("r1"), _record("r2", "rejected")]
    client = [_answer("r1"), ClientRecord(2.0, "r2", 429, "a1")]
    run, log, q = _run(tmp_path, records, client)
    report = check_run(run, log, [q], ["stu_csa"])
    assert report.passed
    assert (report.client_requests, report.audit_records) == (2, 2)
    assert report.citations_checked == 1 and report.strings_searched == 5  # noqa: PLR2004
    assert report.other_ms_p50 == pytest.approx(3.0)
    assert report.by_decision == {"answered": 1, "rejected": 1}


def test_missing_records_are_found_from_both_sides(tmp_path: Path) -> None:
    client = [_answer("r1"), _answer("r9")]
    run, log, q = _run(tmp_path, [_record("r1")], client, ["r1", "r8"])
    report = check_run(run, log, [q])
    assert report.missing_client == ["r9"] and report.missing_gateway == ["r8"]
    assert not report.passed


def test_duplicates_and_missing_ids_fail(tmp_path: Path) -> None:
    client = [_answer("r1"), ClientRecord(2.0, None, 500, None)]
    run, log, q = _run(tmp_path, [_record("r1"), _record("r1")], client)
    report = check_run(run, log, [q])
    assert report.duplicated == ["r1"] and report.client_without_id == 1
    assert not report.passed


def test_a_citation_outside_the_record_is_unreproduced(tmp_path: Path) -> None:
    wrong_version = _answer("r1", ("d1", 1, "d1@v2#1"))
    wrong_chunk = _answer("r2", ("d1", 2, "d1@v2#7"))
    run, log, q = _run(
        tmp_path, [_record("r1"), _record("r2")], [wrong_version, wrong_chunk]
    )
    assert check_run(run, log, [q]).unreproduced == ["r1", "r2"]


def test_planted_question_text_and_user_id_are_found(tmp_path: Path) -> None:
    # A leak would sit in a string field of an otherwise valid record.
    leaky = _record("r2").model_copy(update={"tenant": QUESTION.upper()})
    named = _record("r3").model_copy(update={"model": "stu_csa"})
    run, log, q = _run(tmp_path, [_record("r1"), leaky, named], [_answer("r1")])
    report = check_run(run, log, [q], ["stu_csa"])
    assert report.raw_text_hits == 1 and report.user_id_hits == 1
    assert not report.passed


def test_cli_writes_summary_and_exit_code(tmp_path: Path) -> None:
    run, log, q = _run(tmp_path, [_record("r1")], [_answer("r1")])
    out = tmp_path / "out"
    assert main([str(run), str(log), str(out), "--questions", str(q)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passed"] and summary["requirement"] == "FR-12"
    assert summary["missing_client_count"] == 0
    assert summary["sources"]["audit"].endswith("audit.jsonl")


def test_client_log_round_trip_holds_no_text(tmp_path: Path) -> None:
    log = ClientLog()
    log.add(_answer("r1"))
    log.write(tmp_path / "c.jsonl")
    raw = (tmp_path / "c.jsonl").read_text()
    assert "hostel" not in raw.lower()
    assert read_client_log(tmp_path / "c.jsonl") == [_answer("r1")]
    (tmp_path / "bad.jsonl").write_text('{"x": 1}\n')
    with pytest.raises(ValueError, match="not a client record"):
        read_client_log(tmp_path / "bad.jsonl")


def test_question_ids_cover_dataset_and_follow_ups() -> None:
    from eval.locust.questions import FOLLOW_UPS, question_ids

    ids = question_ids([{"id": "a1", "question": QUESTION}])
    assert ids[QUESTION] == "a1"
    assert ids[FOLLOW_UPS[0]] == "follow-up-1"
    log = ClientLog()
    log.add(_answer("r1"))
    assert len(log) == 1
    log.clear()
    assert len(log) == 0


def test_reasons_statuses_and_stage_percentiles(tmp_path: Path) -> None:
    refused = _record("r2", "denied").model_copy(
        update={"reason": "access_check_timeout"}
    )
    client = [_answer("r1"), ClientRecord(2.0, "r2", 503, "a1")]
    run, log, q = _run(tmp_path, [_record("r1"), refused], client)
    report = check_run(run, log, [q])
    assert report.passed
    assert report.by_reason == {"none": 1, "access_check_timeout": 1}
    assert report.client_status == {"200": 1, "503": 1}
    assert report.stages_ms["embed_ms"]["p95"] == pytest.approx(94.0)
    assert report.stages_ms["start_ms"] == {"p50": 0.0, "p95": 0.0}
