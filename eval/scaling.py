"""Scaling and cost of one Locust run, from Prometheus (FR-9, NFR-6).

After a run, the worker counts the autoscaler exported
(`askiisc_workers{state}`) over the run's time window give:

- max_workers: the most GPU workers held at once (FR-9: never above 2);
- scale_out_s: the longest time from a second worker being requested to
  both being ready (FR-9: within 180 s);
- gpu_hours: worker-hours held (starting, loading or ready), and per 1,000
  answered questions (NFR-6: at most 0.30, start-up and idle included);
- search_p95_s: the p95 of the index search over the run (NFR-4: 150 ms);
- authorize_p99_s, filter_p99_s: the access service's decision time, as
  it measures it (NFR-10: 10 ms and 20 ms);
- heavy_token_share, light_token_share: the two busiest departments' share
  of the output tokens generated in the run (FR-11: at most 55% and at
  least 20% when one asks 10 times as much as the other).

slurm/loadtest.sh runs it after the steady, spike, cold-start and diurnal
runs:
    python -m eval.scaling SUMMARY.json [--results DIR]
It writes measurement `scaling-<workload>` for the scorecard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from eval.measurements import write_measurement

STEP_S = 5.0
HELD = 'sum(askiisc_workers{state=~"starting|loading|ready"})'
READY = 'sum(askiisc_workers{state="ready"})'
SEARCH_P95 = (
    "histogram_quantile(0.95, sum by (le) "
    '(increase(askiisc_stage_seconds_bucket{{stage="search"}}[{window}s])))'
)
Series = list[tuple[float, float]]
ACCESS_P99 = (
    "histogram_quantile(0.99, sum by (le, call) "
    "(increase(askiisc_access_seconds_bucket[{window}s])))"
)
TOKENS = "sum by (tenant) (increase(askiisc_output_tokens_total[{window}s]))"
MEASURES = {
    "fairness": "FR-11",
    "steady": "NFR-4",
    "spike": "FR-9",
    "cold-start": "FR-10",
    "diurnal": "NFR-6",
}


def scale_out_s(held: Series, ready: Series, size: int = 2) -> float | None:
    """Longest wait from `size` workers requested to `size` ready.

    Only real scale-outs count: the held count must rise to `size` from
    below it inside the window. None if that never happened.
    """
    ready_at = dict(ready)
    waits: list[float] = []
    start: float | None = None
    below = False
    for t, n in held:
        if n < size:
            below, start = True, None
            continue
        if below and start is None:
            start = t
        if start is not None and ready_at.get(t, 0.0) >= size:
            waits.append(t - start)
            start, below = None, False
    return max(waits) if waits else None


def gpu_hours(held: Series, step_s: float = STEP_S) -> float:
    """Worker-hours held over the samples."""
    return sum(n for _, n in held) * step_s / 3600.0


def scaling_checks(
    held: Series, ready: Series, completed: int
) -> dict[str, Any]:
    """The FR-9 and NFR-6 numbers of one run."""
    hours = gpu_hours(held)
    return {
        "max_workers": max((n for _, n in held), default=0.0),
        "scale_out_s": scale_out_s(held, ready),
        "gpu_hours": round(hours, 4),
        "completed": completed,
        "gpu_hours_per_1k": round(1000 * hours / completed, 4)
        if completed
        else None,
    }


def token_shares(by_tenant: dict[str, float]) -> dict[str, float | None]:
    """The largest and second-largest tenant shares of the tokens."""
    total = sum(by_tenant.values())
    ranked = sorted(by_tenant.values(), reverse=True) + [0.0, 0.0]
    if not total:
        return {"heavy_token_share": None, "light_token_share": None}
    return {
        "heavy_token_share": round(ranked[0] / total, 4),
        "light_token_share": round(ranked[1] / total, 4),
    }


def _by_label(
    http: httpx.Client, url: str, query: str, at: float, label: str
) -> dict[str, float]:  # pragma: no cover - needs Prometheus
    r = http.get(url + "/api/v1/query", params={"query": query, "time": at})
    r.raise_for_status()
    out = {}
    for x in r.json()["data"]["result"]:
        value = float(x["value"][1])
        if not math.isnan(value):  # NaN when nothing was measured
            out[x["metric"].get(label, "")] = value
    return out


def _by_tenant(
    http: httpx.Client, url: str, query: str, at: float
) -> dict[str, float]:  # pragma: no cover - needs Prometheus
    return _by_label(http, url, query, at, "tenant")


def _at(
    http: httpx.Client, url: str, query: str, at: float
) -> float | None:  # pragma: no cover - needs Prometheus
    r = http.get(url + "/api/v1/query", params={"query": query, "time": at})
    r.raise_for_status()
    result = r.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else None


def _range(
    http: httpx.Client, url: str, query: str, start: float, end: float
) -> Series:  # pragma: no cover - needs Prometheus
    r = http.get(
        url + "/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": STEP_S},
    )
    r.raise_for_status()
    result: Sequence[dict[str, Any]] = r.json()["data"]["result"]
    return (
        [(float(t), float(v)) for t, v in result[0]["values"]] if result else []
    )


def main() -> None:  # pragma: no cover - needs a serving job
    """Reads Prometheus for the run in SUMMARY and writes the measurement."""
    data = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--results", type=Path, default=data / "results" / "measurements"
    )
    args = parser.parse_args()
    run = json.loads(args.summary.read_text())
    current = json.loads((data / "run" / "current.json").read_text())
    url = f"http://127.0.0.1:{current['ports']['prometheus']}"
    password = (data / "private" / "app" / "prometheus_password").read_text()
    started = time.time()
    with httpx.Client(
        auth=("prometheus", password.strip()), timeout=30
    ) as http:
        held = _range(http, url, HELD, run["started"], run["finished"])
        ready = _range(http, url, READY, run["started"], run["finished"])
        window = max(60, int(run["finished"] - run["started"]))
        search = _at(
            http, url, SEARCH_P95.format(window=window), run["finished"]
        )
        tokens = _by_tenant(
            http, url, TOKENS.format(window=window), run["finished"]
        )
        access = _by_label(
            http, url, ACCESS_P99.format(window=window), run["finished"],
            "call",
        )  # fmt: skip
    completed = int(run.get("checks", {}).get("answers", 0))  # questions
    checks = {
        **scaling_checks(held, ready, completed),
        "search_p95_s": search,
        **token_shares(tokens),
        "authorize_p99_s": access.get("authorize"),
        "filter_p99_s": access.get("filter"),
    }
    path = write_measurement(
        args.results, f"scaling-{run['workload']}",
        MEASURES.get(run["workload"], "FR-9"), checks,
        "Prometheus", started, run=run["name"],
    )  # fmt: skip
    print(path.read_text())


if __name__ == "__main__":  # pragma: no cover
    main()
