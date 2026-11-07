# ruff: noqa: E501 - log lines and table rows read best unwrapped
"""The v3 results page is built from raw files, listing missing runs."""

import json
from pathlib import Path

import pytest
import tools.v3_results as v3


def test_an_empty_results_folder_lists_every_run_as_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "DATA", tmp_path / "results")
    monkeypatch.setattr(v3, "OUT", tmp_path / "docs")
    assert v3.main(["--fr6", "1", "--v3", "2"]) == 0
    page = (tmp_path / "docs" / "RESULTS.md").read_text()
    assert page.count("not run") >= 6  # noqa: PLR2004
    assert "—" not in page


def test_fr6_reports_become_a_verdict_and_a_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "DATA", tmp_path)
    monkeypatch.setattr(v3, "OUT", tmp_path / "docs")
    spread = {"n": 5, "p50": 1.0, "p95": 2.0, "p99": 20.0, "max": 30.0}
    for arm, goodput in (("on", 3.0), ("off", 1.0)):
        for k in (1, 2, 3):
            folder = tmp_path / "fr6" / "9" / arm / f"rep{k}" / "report"
            folder.mkdir(parents=True)
            multiples = [
                {"multiple": m, "offered_per_s": 5.0 * m,
                 "goodput_per_s": goodput * m, "node_429_ms": spread,
                 "client_429_ms": spread, "accepted_ms": {"p95": 9000.0},
                 "loop_lag_ms": {"max": 4.0}}
                for m in (1, 2, 3)
            ]  # fmt: skip
            report = {"multiples": multiples, "verdicts": {
                "shed_p99_ms_at_3x": 20.0, "shed_within_limit": True,
                "accepted_rise": -0.1, "accepted_within_limit": True}}  # fmt: skip
            (folder / "fr6.json").write_text(json.dumps(report))
    text = "\n".join(v3.fr6_section("9", []))
    assert "| 429 time at the node, p99 at 3x | at most 50 ms | 20.0 ms" in text
    assert "pass" in text
    svg = (tmp_path / "docs" / "FR-6" / "goodput.svg").read_text()
    assert svg.count("<polyline") == 2  # noqa: PLR2004


def test_hit_rates_read_only_busy_log_lines(tmp_path: Path) -> None:
    log = tmp_path / "vllm-1.log"
    log.write_text(
        "Running: 3 reqs, Waiting: 0 reqs, GPU KV cache usage: 1%, Prefix cache hit rate: 61.5%\n"
        "Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0%, Prefix cache hit rate: 10.0%\n"
    )
    assert v3.hit_rates(log) == [61.5]
    assert v3.hit_rates(tmp_path / "missing.log") == []


def test_a_cold_run_without_any_worker_is_not_a_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "DATA", tmp_path)
    for model, workers in (
        ("qwen3-4b-instruct-2507", []),  # every card full: none started
        (
            "qwen3-8b",
            [{"requested_at": 100.0, "acquired_at": 100.5, "ready_at": 148.0}],
        ),  # fmt: skip
    ):
        base = tmp_path / "v3" / "7" / f"cold-{model}"
        (base / "rep1" / "7-cold20").mkdir(parents=True)
        (base / "serve").mkdir()
        (base / "serve" / "workers.json").write_text(json.dumps(workers))
        answered = bool(workers)
        events = [
            {"kind": "cold_start", "answered": answered,
             "ttft_s": 50.0 if answered else None}
            for _ in range(20)
        ]  # fmt: skip
        (base / "rep1" / "7-cold20" / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )
    rows = [r for r in v3.cold_section("7") if r.startswith("| qwen3")]
    assert rows[0].endswith("| not measured: no GPU worker started |")
    assert rows[1].endswith("| pass |")
