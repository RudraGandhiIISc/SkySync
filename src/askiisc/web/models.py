"""Model comparison (FR-8, Phase 13): the six models side by side.

Each model's numbers live in `<results>/models/<key>/metrics.json`, written
after its runs (see docs/STATUS.md, Phase 13) with:

    python -m askiisc.web.models record <key> --accuracy 0.80 ...

A model without the file shows as pending. The active model is the one the
serving job was started with (ASKIISC_MODEL); switching means restarting
the job, never a hot swap.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from askiisc.common import targets as t
from askiisc.llm.models import load_models

METRICS = {
    "accuracy": ("Accuracy", "share", ">=", t.NFR7_MIN_ACCURACY),
    "citation_correctness": (
        "Citation correctness",
        "share",
        ">=",
        t.NFR7_MIN_CITATION_CORRECTNESS,
    ),
    "p95_latency_s": ("p95 latency (s)", "seconds", "<=", t.NFR1_P95_TOTAL_S),
    "tokens_per_s": ("Output tokens/s per worker", "number", ">=", None),
    "cold_start_s": ("Cold start (s)", "seconds", "<=", None),
    "gpu_hours_per_1k": (
        "GPU-hours per 1,000 queries",
        "number",
        "<=",
        t.NFR6_MAX_GPU_HOURS_PER_1K,
    ),
}
"""Field: (label, kind, better direction, PRD target or None)."""

TOKENS_TARGET = {
    "4B": t.NFR3_TOKENS_PER_S_4B,
    "8B": t.NFR3_TOKENS_PER_S_8B,
    "8B-AWQ": t.NFR3_TOKENS_PER_S_8B_4BIT,
}
COLD_TARGET = {
    "4B": t.NFR2_COLD_TTFT_SMALL_S,
    "8B": t.NFR2_COLD_TTFT_8B_S,
    "8B-AWQ": t.NFR2_COLD_TTFT_SMALL_S,
}


def target_for(field: str, size_class: str) -> float | None:
    """The PRD target of a metric for a model of this size."""
    if field == "tokens_per_s":
        return TOKENS_TARGET[size_class]
    if field == "cold_start_s":
        return COLD_TARGET[size_class]
    return METRICS[field][3]


def meets(field: str, value: float, target: float | None) -> bool | None:
    """Whether a value meets its target (None if there is no target)."""
    if target is None:
        return None
    better = METRICS[field][2]
    return value >= target if better == ">=" else value <= target


def read_metrics(folder: Path) -> dict[str, Any] | None:
    """A model's metrics.json, or None if absent or unreadable."""
    try:
        data = json.loads((folder / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def comparison(models_dir: Path | None, active: str) -> list[dict[str, Any]]:
    """One row per registered model, in registry order.

    Args:
        models_dir: `<results>/models`, or None (everything pending).
        active: The serving job's model key.

    Returns:
        Rows with key, family, size, active flag, status, measured date,
        source and, per metric, value, target and whether it is met.
    """
    rows = []
    for key, spec in load_models().items():
        data = read_metrics(models_dir / key) if models_dir else None
        metrics = {}
        for field in METRICS:
            value = None if data is None else data.get(field)
            target = target_for(field, spec.size_class)
            metrics[field] = {
                "value": value,
                "target": target,
                "met": None if value is None else meets(field, value, target),
            }
        rows.append(
            {
                "key": key,
                "family": spec.family,
                "size_class": spec.size_class,
                "gated": spec.gated,
                "active": key == active,
                "status": "pending" if data is None else "measured",
                "measured": None if data is None else data.get("measured"),
                "source": None if data is None else data.get("source"),
                "metrics": metrics,
            }
        )
    return rows


def labels() -> dict[str, str]:
    """Column labels for the page."""
    return {field: spec[0] for field, spec in METRICS.items()}


def record(folder: Path, values: dict[str, float], source: str) -> Path:
    """Writes a model's metrics.json (fields not given are left out)."""
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        k: v for k, v in values.items() if k in METRICS and v is not None
    }
    data["measured"] = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d")
    data["source"] = source
    path = folder / "metrics.json"
    path.write_text(json.dumps(data, indent=1))
    return path


def main() -> None:  # pragma: no cover - thin command-line wrapper
    """`record <key> --accuracy ... --source ...`."""
    import os  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["record"])
    parser.add_argument("key", choices=sorted(load_models()))
    for field in METRICS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=float)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    data = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
    values = {field: getattr(args, field) for field in METRICS}
    path = record(data / "results" / "models" / args.key, values, args.source)
    print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
