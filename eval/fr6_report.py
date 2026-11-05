"""FR-6 report for one load run: time to 429, goodput, accepted latency.

Reads only raw files of a run:

- the serving job's `responses.jsonl` (one line per question request,
  milliseconds at the serving node) and `loop_lag.jsonl`;
- Locust's `client_requests.jsonl` and `summary.json`.

For each load multiple (W2: the 10-minute steady part at 1x, 2x and 3x;
`overload`: each 5-minute stage) it reports time to 429 at the node and at
the client (p50, p95, p99, max), offered load and goodput (requests started
and answers completed per second), accepted-request p95 and event-loop
lag. Verdicts (PRD FR-6, judged at the serving node, 2026-11-05):
time-to-429 p99 at 3x at most 50 ms, and accepted p95 at 3x at most 20%
above 1x.

Run (inside Slurm):
    python -m eval.fr6_report <serve run dir> <locust run dir> <out dir>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from askiisc.common.targets import (
    FR6_MAX_P95_RISE_FRACTION,
    FR6_REJECT_WITHIN_MS,
)
from eval.locust.workloads import plan

SHED = 429
OK = 200


@dataclass(frozen=True)
class Window:
    """One load multiple's measured interval (epoch seconds)."""

    multiple: int
    start: float
    end: float

    def holds(self, t: float) -> bool:
        """Whether a time falls inside."""
        return self.start <= t < self.end


def windows(workload: str, started: float, scale: float) -> list[Window]:
    """The measured intervals of a run.

    Args:
        workload: "w2" or "overload".
        started: Epoch seconds the run started.
        scale: The run's duration scale.

    Returns:
        One window per multiple.

    Raises:
        ValueError: For a workload without load multiples.
    """
    stages = plan(workload, scale=scale).stages
    edges, t = [], started
    for stage in stages:
        edges.append((t, t + stage.seconds, stage))
        t += stage.seconds
    if workload == "w2":
        steady = [e for i, e in enumerate(edges) if i % 2 == 1 and i < 6]  # noqa: PLR2004
    elif workload == "overload":
        steady = edges
    else:
        raise ValueError(f"{workload} has no load multiples")
    return [Window(k, a, b) for k, (a, b, _) in enumerate(steady, 1)]


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (0 for no values)."""
    if not values:
        return 0.0
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q * len(ranked)) - 1)]


def spread(values: Sequence[float]) -> dict[str, float]:
    """n, p50, p95, p99 and max, rounded to 0.01."""
    return {
        "n": len(values),
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "max": round(max(values, default=0.0), 2),
    }


def read_lines(path: Path) -> list[dict[str, Any]]:
    """JSON lines of a file ([] if it is missing)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def started_at(row: dict[str, Any]) -> float:
    """A client row's start time (it is logged when the request ends)."""
    return float(row["time"]) - float(row.get("seconds", 0.0))


def measure(
    window: Window,
    node: Iterable[dict[str, Any]],
    client: Iterable[dict[str, Any]],
    lags: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """The FR-6 numbers of one window."""
    node_in = [r for r in node if window.holds(float(r["time"]))]
    client_in = [r for r in client if window.holds(started_at(r))]
    seconds = window.end - window.start
    statuses: dict[str, int] = {}
    for r in client_in:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    accepted = [1000 * r["seconds"] for r in client_in if r["status"] == OK]
    lag = [r for r in lags if window.holds(float(r["time"]))]
    return {
        "multiple": window.multiple,
        "seconds": round(seconds, 1),
        "offered_per_s": round(len(client_in) / seconds, 3),
        "goodput_per_s": round(len(accepted) / seconds, 3),
        "client_status": statuses,
        "node_429_ms": spread(
            [r["done_ms"] for r in node_in if r["status"] == SHED]
        ),
        "client_429_ms": spread(
            [1000 * r["seconds"] for r in client_in if r["status"] == SHED]
        ),
        "accepted_ms": spread(accepted),
        "loop_lag_ms": {
            "mean": round(
                sum(r["mean_ms"] for r in lag) / len(lag) if lag else 0.0, 3
            ),
            "p95_of_window_max": round(
                percentile([r["max_ms"] for r in lag], 0.95), 2
            ),
            "max": round(max((r["max_ms"] for r in lag), default=0.0), 2),
        },
    }


def verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """FR-6 verdicts from the 1x and 3x rows."""
    by = {r["multiple"]: r for r in rows}
    top, base = by.get(3), by.get(1)
    if top is None or base is None:
        return {"shed_within_limit": None, "accepted_rise": None}
    shed = top["node_429_ms"]
    rise = (
        top["accepted_ms"]["p95"] / base["accepted_ms"]["p95"] - 1
        if base["accepted_ms"]["p95"]
        else None
    )
    return {
        "shed_p99_ms_at_3x": shed["p99"],
        "shed_within_limit": shed["n"] > 0
        and shed["p99"] <= FR6_REJECT_WITHIN_MS,
        "accepted_rise": None if rise is None else round(rise, 4),
        "accepted_within_limit": rise is not None
        and rise <= FR6_MAX_P95_RISE_FRACTION,
    }


def report(serve: Path, locust: Path) -> dict[str, Any]:
    """The FR-6 report of one run.

    Args:
        serve: The serving job's run folder.
        locust: The Locust run folder.

    Returns:
        Windows, numbers per window and verdicts.
    """
    summary = json.loads((locust / "summary.json").read_text())
    spans = windows(
        summary["workload"],
        float(summary["started"]),
        float(summary["duration_scale"]),
    )
    node = read_lines(serve / "responses.jsonl")
    client = read_lines(locust / "client_requests.jsonl")
    lags = read_lines(serve / "loop_lag.jsonl")
    rows = [measure(w, node, client, lags) for w in spans]
    return {
        "serve": str(serve),
        "locust": str(locust),
        "workload": summary["workload"],
        "windows": [asdict(w) for w in spans],
        "multiples": rows,
        "verdicts": verdicts(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Writes fr6.json for one run and prints its verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serve", type=Path)
    parser.add_argument("locust", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    result = report(args.serve, args.locust)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fr6.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["verdicts"]))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
