"""The requirement scorecard: Locust runs first, baseline-v1 as fallback."""

from pathlib import Path
from typing import Any

import pytest

from askiisc.web.scorecard import (
    BASELINE_FILE,
    MAP_FILE,
    REQUIREMENTS,
    evaluate,
    load,
    read_value,
    scorecard,
    target_of,
)
from tests.web_fixtures import HDR, Web, build_web

DAY = 1_792_800_000.0


def _run(workload: str, finished: float = DAY, **extra: Any) -> dict[str, Any]:
    return {
        "name": f"1-{workload}",
        "workload": workload,
        "finished": finished,
        "requests": 100,
        "failures": 0,
        "checks": {},
        "measurements": {},
        "by_request": [],
        **extra,
    }


def _rows(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {r.id: r for r in scorecard(runs)}


def test_every_requirement_is_listed_and_falls_back_to_baseline() -> None:
    rows = scorecard([])
    assert [r.id for r in rows] == list(REQUIREMENTS)
    assert len(rows) == 22
    assert all(r.source == "baseline-v1" for r in rows)
    nfr1 = next(r for r in rows if r.id == "NFR-1")
    assert nfr1.verdict == "fail" and "3.5 s" in nfr1.target
    assert set(load(BASELINE_FILE)["results"]) == set(REQUIREMENTS)


def test_every_mapping_names_a_real_requirement_and_target() -> None:
    mapping = load(MAP_FILE)
    assert set(mapping) <= set(REQUIREMENTS)
    for entry in mapping.values():
        parts = entry if isinstance(entry, list) else [entry]
        checks = [c for part in parts for c in part["checks"]]
        for check in checks:
            assert check["op"] in {"<=", ">=", "=="}
            target_of(check)  # constant names resolve


def test_steady_run_sets_nfr1_and_nfr4() -> None:
    steady = _run(
        "steady",
        failures=2,
        requests=1000,
        by_request=[
            {"name": "chat message", "p95_ms": 2900.0},
            {"name": "chat message (first token)", "p95_ms": 900.0},
        ],
    )
    rows = _rows([steady])
    # The first token is judged at the model server (scaling-steady), not
    # at the client: without that measurement NFR-1 is pending.
    assert rows["NFR-1"].verdict == "pending"
    assert "p95 answer time (ms): 2,900" in rows["NFR-1"].value
    assert rows["NFR-1"].source == "Locust 1-steady"
    slow = _run("scaling-steady", checks={"first_token_p95_s": 0.9})
    assert _rows([steady, slow])["NFR-1"].verdict == "fail"
    fast = _run("scaling-steady", checks={"first_token_p95_s": 0.38})
    nfr1 = _rows([steady, fast])["NFR-1"]
    assert nfr1.verdict == "pass"
    assert "first token at the model server (ms): 380" in nfr1.value
    assert rows["NFR-4"].verdict == "pending"  # search p95 not measured
    assert "0.2%" in rows["NFR-4"].value
    search = _run("scaling-steady", checks={"search_p95_s": 0.031})
    nfr4 = _rows([steady, search])["NFR-4"]
    assert nfr4.verdict == "pass" and "(s): 0.031" in nfr4.value
    assert rows["NFR-1"].date == "2026-10-24"


def test_newest_usable_run_wins() -> None:
    old = _run("access", DAY - 10, checks={"leaks": 1})
    new = _run("access", DAY, checks={})
    newer_but_unusable = _run("cache", DAY + 10)
    timing = _run(
        "scaling-steady",
        checks={"authorize_p99_s": 0.001, "filter_p99_s": 0.002},
    )
    rows = _rows([old, new, newer_but_unusable, timing])
    assert rows["NFR-10"].verdict == "pass"
    assert rows["FR-3"].source == "baseline-v1"  # no revocation measured
    assert rows["FR-7"].source == "baseline-v1"  # no cache counters
    rows = _rows([old, timing])
    assert rows["NFR-10"].verdict == "fail"


def test_values_ratios_and_defaults() -> None:
    run = _run(
        "redteam",
        checks={"injections": 40, "injection_successes": 1},
        measurements={"x": {"max": 3.0}},
        by_request=[{"name": "a", "requests": 3}],
    )
    assert read_value(run, {"path": ["measurements", "x", "max"]}) == 3.0
    assert read_value(run, {"path": ["checks", "leaks"], "default": 0}) == 0
    assert read_value(run, {"path": ["checks", "leaks"]}) is None
    assert read_value(run, {"path": ["by_request", "a", "requests"]}) == 3
    assert read_value(run, {"path": ["by_request", "b", "requests"]}) is None
    ratio = {
        "ratio": {
            "num": ["checks", "missing"],
            "den": [["checks", "injections"]],
        }
    }
    assert read_value(run, ratio) == 0.0
    empty = {"ratio": {"num": ["checks", "a"], "den": [["checks", "b"]]}}
    assert read_value(run, empty) is None
    assert _rows([run])["NFR-9"].verdict == "pass"
    check = {
        "label": "x",
        "value": {"path": ["failures"]},
        "op": "==",
        "target": 0,
    }
    assert evaluate(run, [check]) == ("x: 0", True)
    scaled = {"target": "NFR1_P95_TOTAL_S", "scale": 1000}
    assert target_of(scaled) == pytest.approx(3500.0)


async def test_results_api_and_page_are_for_admins(tmp_path: Path) -> None:
    web = build_web(tmp_path)
    web.deps.loadtests = lambda: [_run("access", checks={"leaks": 0})]
    web.deps.measurements = lambda: [
        _run("recall", kind="Probe", checks={"recall_at_k": 1.0, "leaks": 0})
    ]
    student = web.browser()
    web.account("u_cds_stu")
    await web.login("u_cds_stu", student)
    assert (await student.get("/api/admin/results")).status_code == 403
    admin = await _admin(web)
    body = (await admin.get("/api/admin/results")).json()["requirements"]
    fr2 = next(r for r in body if r["id"] == "FR-2")
    assert fr2["verdict"] == "pass"
    assert fr2["source"] == "Locust 1-access, Probe 1-recall"
    page = await admin.get("/admin/results")
    assert "Requirement scorecard" in page.text
    assert "/admin#requirements" in page.text  # now a dashboard tab
    assert 'data-tab="requirements"' in (await admin.get("/admin")).text


async def _admin(web: Web) -> Any:
    client = web.browser()
    web.account("u_admin")
    await web.login("u_admin", client)
    assert (await client.get("/api/me", headers=HDR)).status_code == 200
    return client


def test_fr2_combines_the_access_run_and_the_recall_probe() -> None:
    access = _run("access", checks={"leaks": 0})
    recall = _run(
        "recall",
        DAY + 5,
        kind="Probe",
        checks={"recall_at_k": 0.99, "leaks": 0},
    )
    only_access = _rows([access])["FR-2"]
    assert only_access.verdict == "pending"
    assert "recall: not measured" in only_access.value
    both = _rows([access, recall])["FR-2"]
    assert both.verdict == "pass"
    assert "recall@8 of the serving index: 0.990" in both.value
    assert both.source == "Locust 1-access, Probe 1-recall"
    low = _run("recall", DAY + 9, checks={"recall_at_k": 0.9, "leaks": 0})
    assert _rows([access, low])["FR-2"].verdict == "fail"


def test_grader_and_benchmark_measurements_reach_the_scorecard(
    tmp_path: Path,
) -> None:
    import json

    from eval.measurements import write_measurement

    path = write_measurement(
        tmp_path, "quality-judge", "FR-1, NFR-7",
        {"judge_accuracy": 0.83, "judge_citation_support": 0.95,
         "abstain_unanswerable": 0.7},
        "Grader", DAY,
    )  # fmt: skip
    judge = json.loads(path.read_text())
    assert judge["workload"] == "quality-judge" and judge["kind"] == "Grader"
    rubric = _run(
        "quality-rubric",
        checks={"answered_with_supporting_citation": 0.97,
                "abstain_unanswerable": 0.8},
    )  # fmt: skip
    bench = _run("throughput", checks={"output_tokens_per_s": 5100.0})
    rows = _rows([judge, rubric, bench])
    assert rows["NFR-7"].verdict == "pass"
    assert "accuracy (judge): 0.830" in rows["NFR-7"].value
    assert rows["NFR-7"].source.startswith("Grader ")
    assert rows["FR-1"].verdict == "fail"  # refusals below 75%
    assert rows["NFR-3"].verdict == "pass"
    judge["checks"]["abstain_unanswerable"] = 0.775
    assert _rows([judge, rubric, bench])["FR-1"].verdict == "pass"


def test_scaling_from_prometheus_series() -> None:
    from eval.scaling import gpu_hours, scale_out_s, scaling_checks

    held = [
        (0, 1),
        (5, 2),
        (10, 2),
        (15, 2),
        (20, 2),
        (25, 1),
        (30, 2),
        (35, 2),
    ]
    ready = [
        (0, 1),
        (5, 1),
        (10, 1),
        (15, 2),
        (20, 2),
        (25, 1),
        (30, 1),
        (35, 2),
    ]
    assert scale_out_s(held, ready) == 10.0  # 5 -> 15, then 30 -> 35
    assert scale_out_s([(0, 1)], [(0, 1)]) is None
    assert gpu_hours([(0, 2)] * 720) == pytest.approx(2.0)  # 1 h of 2 workers
    checks = scaling_checks(held, ready, 100)
    assert checks["max_workers"] == 2 and checks["completed"] == 100
    assert scaling_checks([], [], 0)["gpu_hours_per_1k"] is None
    spike = _run(
        "scaling-spike", kind="Prometheus",
        checks={"scale_out_s": 74.0, "max_workers": 2.0},
    )  # fmt: skip
    diurnal = _run("scaling-diurnal", checks={"gpu_hours_per_1k": 0.41})
    rows = _rows([spike, diurnal])
    assert rows["FR-9"].verdict == "pass"
    cold = _run("scaling-cold-start", DAY + 1,
                checks={"scale_out_s": 200.0, "max_workers": 2.0})  # fmt: skip
    assert _rows([spike, cold])["FR-9"].verdict == "fail"  # newest of either
    missing = _rows([_run("scaling-spike", checks={"max_workers": 2.0})])
    assert missing["FR-9"].source == "baseline-v1"
    assert rows["NFR-6"].verdict == "fail"


def test_fairness_by_token_share() -> None:
    from eval.scaling import token_shares

    assert token_shares({"cds": 600.0, "ece": 300.0, "csa": 100.0}) == {
        "heavy_token_share": 0.6,
        "light_token_share": 0.3,
    }
    assert token_shares({}) == {
        "heavy_token_share": None,
        "light_token_share": None,
    }
    run = _run(
        "scaling-fairness",
        checks={"heavy_token_share": 0.52, "light_token_share": 0.41},
    )
    assert _rows([run])["FR-11"].verdict == "pass"


def test_nfr10_combines_leaks_and_access_latency() -> None:
    access = _run("access", checks={"leaks": 0})
    timing = _run(
        "scaling-steady",
        checks={"authorize_p99_s": 0.004, "filter_p99_s": 0.025},
    )
    row = _rows([access, timing])["NFR-10"]
    assert row.verdict == "fail"  # filter 25 ms > 20 ms
    assert "authorize p99 (ms): 4" in row.value
    assert (
        read_value(timing, {"path": ["checks", "filter_p99_s"], "scale": 1000})
        == 25.0
    )


def test_first_token_at_the_model_server_from_audit_records(
    tmp_path: Path,
) -> None:
    import json

    from eval.scaling import serving_first_token_p95

    def row(t: float, ms: float, **extra: Any) -> str:
        record = {
            "time": t,
            "decision": "answered",
            "cache_hit": False,
            "latency": {"first_token_ms": ms},
            **extra,
        }
        return json.dumps(record)

    audit = tmp_path / "audit.jsonl"
    lines = [row(100 + i, 300 + i) for i in range(20)]
    lines += [
        row(50, 9000),  # before the run
        row(110, 9000, cache_hit=True),  # no model call
        row(111, 0),  # refused before the model
        row(112, 9000, decision="error"),
    ]
    audit.write_text("\n".join(lines) + "\n")
    assert serving_first_token_p95(audit, 100, 200) == 0.318
    assert serving_first_token_p95(audit, 300, 400) is None
    assert serving_first_token_p95(tmp_path / "missing.jsonl", 0, 1) is None
