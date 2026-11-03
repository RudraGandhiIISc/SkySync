"""Measurement summaries the admin scorecard reads (results/measurements).

Locust runs write summary.json themselves; the other measurements (the FR-2
recall probe, the graders, the throughput benchmark) write one through
`write_measurement`, in the same shape: workload, finished time, checks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_measurement(
    results: Path,
    workload: str,
    requirements: str,
    checks: Mapping[str, Any],
    kind: str,
    started: float,
    **extra: Any,
) -> Path:
    """Writes <results>/<date>-<workload>/summary.json.

    Args:
        results: The measurements folder.
        workload: The name scorecard_map.yaml refers to.
        requirements: The FR/NFR ids measured, for people.
        checks: The numbers the scorecard reads.
        kind: Shown as the source ("Probe", "Grader", "Benchmark").
        started: When the measurement started (epoch seconds).
        **extra: More fields for people (never user ids or text).

    Returns:
        The summary file.
    """
    out = results / time.strftime(f"%Y%m%d-%H%M%S-{workload}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": out.name,
        "workload": workload,
        "kind": kind,
        "requirements": requirements,
        "started": started,
        "finished": time.time(),
        "checks": dict(checks),
        **extra,
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=1))
    return path
