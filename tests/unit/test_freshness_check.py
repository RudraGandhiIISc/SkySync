"""FR-5 / E5: the freshness check reads a run's raw files."""

import json
from pathlib import Path

from eval.freshness_check import (
    check,
    demotion_s,
    main,
    publishes,
    superseded_at,
    unflagged_cited,
    unflagged_used,
)

from askiisc.common.types import ChunkDraft, DocId
from askiisc.ingest.versions import InMemoryVersionStore

DRAFT = (ChunkDraft(text="code", page=1, start=0, end=4),)


def pub(t: float, doc: str, v: int | None, code: str) -> dict[str, object]:
    return {"kind": "publish", "time": t, "doc": doc, "version": v,
            "code": code}  # fmt: skip


def ans(sent: float, code: str, version: int, stale: bool) -> dict[str, object]:
    cited = {"doc_id": "f-01", "version": version, "is_stale": stale}
    return {"kind": "answer", "doc": "f-01", "sent": sent, "codes": [code],
            "citations": [cited]}  # fmt: skip


EVENTS = [
    pub(100.0, "f-01", 1, "F1-1111"),
    pub(120.0, "f-02", 1, "F2-2222"),
    pub(140.0, "f-01", 2, "F3-3333"),
    pub(150.0, "f-01", None, "x"),
    ans(101.0, "F1-1111", 1, False),
    ans(141.0, "F1-1111", 1, False),
    ans(142.0, "F1-1111", 1, True),
    ans(143.5, "F3-3333", 2, False),
]


def store() -> InMemoryVersionStore:
    versions = InMemoryVersionStore()
    versions.publish(DocId("f-01"), DRAFT, 100.0, "a")
    versions.publish(DocId("f-02"), DRAFT, 120.0, "b")
    versions.publish(DocId("f-01"), DRAFT, 140.0, "c")
    return versions


def test_publishes_and_their_replacements() -> None:
    pubs = publishes(EVENTS)
    assert [(p.doc, p.version) for p in pubs] == [
        ("f-01", 1),
        ("f-02", 1),
        ("f-01", 2),
    ]
    assert superseded_at(pubs) == {("f-01", 1): 140.0}


def test_demotion_and_unflagged_answers() -> None:
    pubs = publishes(EVENTS)
    answers = [e for e in EVENTS if e["kind"] == "answer"]
    assert demotion_s(pubs, answers) == [1.0, 3.5]
    # The 141.0 answer cites version 1 after version 2 was out, unflagged.
    assert unflagged_cited(answers, superseded_at(pubs)) == 1


def test_audit_counts_use_of_a_replaced_version() -> None:
    replaced = {("f-01", 1): 140.0}
    used = [{"doc_id": "f-01", "version": 1}]

    def rec(t: float, stale: bool) -> dict[str, object]:
        return {"time": t, "latency_ms": 500, "decision": "answered",
                "documents": used, "stale": stale}  # fmt: skip

    audit = [
        rec(141.0, False),
        rec(142.0, True),
        rec(139.0, False),
        {"time": 150.0, "decision": "rejected", "documents": []},
    ]
    assert unflagged_used(audit, replaced) == (2, 1)


def test_check_and_main_write_the_report(tmp_path: Path) -> None:
    locust, serve = tmp_path / "locust", tmp_path / "serve"
    locust.mkdir()
    serve.mkdir()
    (locust / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in EVENTS)
    )
    result = check(locust, serve, store())
    assert (result["revisions"], result["documents"]) == (3, 2)
    assert result["as_of_ok"] == 3  # noqa: PLR2004
    assert result["audit_used_superseded"] == 0
    db = tmp_path / "v.db"
    from askiisc.ingest.versions import SQLiteVersionStore

    sql = SQLiteVersionStore(db)
    sql.publish(DocId("f-01"), DRAFT, 100.0, "a")
    out = tmp_path / "out"
    assert main([str(locust), str(serve), str(db), str(out)]) == 0
    saved = json.loads((out / "freshness.json").read_text())
    assert saved["as_of_ok"] == 1  # only f-01 v1 exists in this store
