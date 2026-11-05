"""The FR-6 report reads a run's raw files into verdicts."""

import json
from pathlib import Path

import pytest
from eval.fr6_report import main, percentile, report, spread, windows

T0 = 1_000_000.0


def test_w2_windows_are_the_three_steady_parts() -> None:
    spans = windows("w2", T0, 1.0)
    assert [(w.multiple, w.start - T0, w.end - T0) for w in spans] == [
        (1, 60, 660),
        (2, 720, 1320),
        (3, 1380, 1980),
    ]
    over = windows("overload", T0, 0.1)
    assert [(w.start - T0, w.end - T0) for w in over] == [
        (0, 30),
        (30, 60),
        (60, 90),
    ]
    with pytest.raises(ValueError):
        windows("steady", T0, 1.0)


def test_percentiles_use_the_nearest_rank() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([3, 1, 2, 4], 0.5) == 2  # noqa: PLR2004
    assert percentile(list(range(1, 101)), 0.99) == 99  # noqa: PLR2004
    assert spread([1.0])["max"] == 1.0


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run_files(tmp_path: Path, shed_ms: float, slow: float) -> tuple[Path, Path]:
    serve, locust = tmp_path / "serve", tmp_path / "locust"
    serve.mkdir()
    locust.mkdir()
    (locust / "summary.json").write_text(
        json.dumps({"workload": "w2", "started": T0, "duration_scale": 1.0})
    )
    node, client, lags = [], [], []
    for k, (a, _) in enumerate(((60, 660), (720, 1320), (1380, 1980)), 1):
        t = T0 + a + 10
        took = 2.0 * (slow if k == 3 else 1)  # noqa: PLR2004
        for i in range(10):
            client.append({"time": t + i + 2.0, "status": 200, "seconds": took})
        if k > 1:
            for i in range(20):
                node.append({"time": t + i, "status": 429, "done_ms": shed_ms})
                client.append({"time": t + i + 0.1, "status": 429,
                               "seconds": 0.1})  # fmt: skip
        lags.append({"time": t, "mean_ms": 1.0, "max_ms": 4.0})
    write(serve / "responses.jsonl", node)
    write(serve / "loop_lag.jsonl", lags)
    write(locust / "client_requests.jsonl", client)
    return serve, locust


def test_a_fast_shed_and_flat_latency_pass(tmp_path: Path) -> None:
    serve, locust = run_files(tmp_path, shed_ms=3.0, slow=1.1)
    out = report(serve, locust)
    by = {r["multiple"]: r for r in out["multiples"]}
    assert by[1]["node_429_ms"]["n"] == 0
    assert by[3]["node_429_ms"]["p99"] == 3.0  # noqa: PLR2004
    assert by[3]["client_429_ms"]["p50"] == 100.0  # noqa: PLR2004
    assert by[3]["client_status"] == {"200": 10, "429": 20}
    assert by[1]["goodput_per_s"] == pytest.approx(10 / 600, abs=1e-3)
    assert by[2]["loop_lag_ms"]["max"] == 4.0  # noqa: PLR2004
    assert out["verdicts"]["shed_within_limit"] is True
    assert out["verdicts"]["accepted_rise"] == pytest.approx(0.1)
    assert out["verdicts"]["accepted_within_limit"] is True


def test_a_slow_shed_or_a_latency_rise_fails(tmp_path: Path) -> None:
    serve, locust = run_files(tmp_path, shed_ms=80.0, slow=1.5)
    verdict = report(serve, locust)["verdicts"]
    assert verdict["shed_within_limit"] is False
    assert verdict["accepted_within_limit"] is False


def test_main_writes_the_report(tmp_path: Path) -> None:
    serve, locust = run_files(tmp_path, shed_ms=3.0, slow=1.0)
    assert main([str(serve), str(locust), str(tmp_path / "out")]) == 0
    saved = json.loads((tmp_path / "out" / "fr6.json").read_text())
    assert saved["workload"] == "w2"


def test_missing_multiples_give_no_verdict(tmp_path: Path) -> None:
    from eval.fr6_report import verdicts

    assert verdicts([])["shed_within_limit"] is None
    assert (
        verdicts(
            [
                {"multiple": 1, "accepted_ms": {"p95": 0.0}},
                {
                    "multiple": 3,
                    "accepted_ms": {"p95": 1.0},
                    "node_429_ms": {"n": 0, "p99": 0.0},
                },
            ]  # fmt: skip
        )["accepted_rise"]
        is None
    )
