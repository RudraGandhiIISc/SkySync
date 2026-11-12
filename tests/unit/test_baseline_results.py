# ruff: noqa: E501 - log lines and table rows read best unwrapped
"""The red-team baseline evidence page is built from raw files only."""

import json
from pathlib import Path

import pytest
import tools.baseline_results as br
import tools.v3_results as v3


def _vllm_log(path: Path, weights: float, gib: float, tokens: int) -> None:
    path.mkdir(parents=True)
    (path / "vllm.log").write_text(
        f"INFO Model loading took {weights} GiB and 2.3 seconds\n"
        f"INFO Available KV cache memory: {gib} GiB\n"
        f"INFO GPU KV cache size: {tokens:,} tokens\n"
    )


def test_kv_rows_use_the_worst_case_ratio(tmp_path: Path) -> None:
    _vllm_log(tmp_path / "p", 15.27, 2.74, 19_984)
    _vllm_log(tmp_path / "c1", 5.71, 9.05, 65_920)
    _vllm_log(tmp_path / "c2", 5.71, 12.30, 89_584)
    rows = br.kv_rows(
        tmp_path, {"Qwen3-8B": ["p"], "Qwen3-8B-AWQ": ["c1", "c2"]}
    )
    assert (
        rows[0]
        == "| Qwen3-8B-AWQ against Qwen3-8B | 5.71 / 15.27 | 65,920, 89,584 / 19,984 | 3.30x | pass |"
    )
    assert rows[1].endswith("| not measured |")  # no Granite runs given


def test_token_shares_count_generated_tokens_in_the_window(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit.jsonl"
    rows = [
        {"time": 10, "tenant": "csa", "completion_tokens": 100},
        {"time": 11, "tenant": "cds", "completion_tokens": 100},
        {
            "time": 12,
            "tenant": "csa",
            "completion_tokens": 900,
            "cache_hit": True,
        },
        {"time": 99, "tenant": "csa", "completion_tokens": 900},
    ]
    audit.write_text("\n".join(json.dumps(r) for r in rows))
    assert br.token_shares(audit, 0, 50) == {"csa": 0.5, "cds": 0.5}
    assert br.token_shares(audit, 200, 300) == {}


def test_restore_time_is_death_to_the_next_ready_worker() -> None:
    workers = [
        {"requested_at": 0.0, "ready_at": 40.0, "released_at": 1800.0},
        {"requested_at": 1801.0, "ready_at": 1852.5, "released_at": None},
    ]
    assert br.restore_s(workers) == 52.5
    assert br.restore_s(workers[:1]) is None


def test_an_empty_results_folder_lists_every_run_as_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "DATA", tmp_path / "results")
    monkeypatch.setattr(br, "OUT", tmp_path / "docs")
    assert br.main(["--v3", "1", "--kv", "2", "--new", "3"]) == 0
    page = (tmp_path / "docs" / "README.md").read_text()
    assert page.count("not run") >= 6  # noqa: PLR2004
    assert (
        "(code at 74401a5, job 1)" in page and "(release code, job 3)" in page
    )


def test_fr5_after_the_fixes_needs_zero_unflagged_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v3, "DATA", tmp_path)
    report = tmp_path / "v3" / "5" / "fresh5" / "report"
    report.mkdir(parents=True)
    result = {
        "revisions": 60, "documents": 20, "answers": 3000,
        "demotion_s": {"n": 60, "p50": 11.0, "p95": 38.0, "max": 120.0},
        "unflagged_cited": 0, "audit_used_superseded": 40,
        "audit_unflagged": 0, "as_of_ok": 60,
        "as_of_checkpoints": [{"ok": True}] * 60,
    }  # fmt: skip
    (report / "freshness.json").write_text(json.dumps(result))
    assert "Verdict: pass." in "\n".join(br.fresh5_section("5"))
    result["audit_unflagged"] = 1
    (report / "freshness.json").write_text(json.dumps(result))
    assert "Verdict: fail." in "\n".join(br.fresh5_section("5"))
