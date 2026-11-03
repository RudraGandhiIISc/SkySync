"""The requirement scorecard: FR-1 to FR-12 and NFR-1 to NFR-10.

Each requirement shows its target (from common/targets.py), the latest
measured value, a verdict, the date and the source. Values come from the
newest Locust run of the workload that measures the requirement
(scorecard_map.yaml says which numbers in summary.json to read and how to
compare them); without a usable run, the baseline-v1 result is shown.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from askiisc.common import targets as t

HERE = Path(__file__).parent
MAP_FILE = HERE / "scorecard_map.yaml"
BASELINE_FILE = HERE / "baseline_v1.yaml"

OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
}

REQUIREMENTS: dict[str, tuple[str, str]] = {
    "FR-1": (
        "Cited answers, honest refusals",
        f"≥ {t.FR1_MIN_CITED_ANSWER_FRACTION:.0%} cited correctly, "
        f"≥ {t.FR1_MIN_ABSTAIN_FRACTION:.0%} refusals",
    ),
    "FR-2": (
        "Filtered retrieval",
        f"{t.FR2_MAX_FORBIDDEN_CHUNKS} forbidden chunks; "
        f"recall@{t.RETRIEVAL_TOP_K} ≥ {t.FR2_MIN_RECALL_AT_K}",
    ),
    "FR-3": (
        "Per-user access, revocation",
        f"revoked within {t.FR3_REVOCATION_SECONDS:.0f} s; 0 leaks",
    ),
    "FR-4": (
        "Separate access service, fails closed",
        "nothing answered unchecked",
    ),
    "FR-5": ("Versions, stale content", "answers use the current version"),
    "FR-6": (
        "Load limits, 429",
        f"429 within {t.FR6_REJECT_WITHIN_MS:.0f} ms "
        f"at {t.FR6_OVERLOAD_FACTOR:.0f}x load",
    ),
    "FR-7": (
        "Batching, prefix and answer cache",
        f"{t.FR7_MIN_THROUGHPUT_GAIN:.0f}x throughput; "
        f"hit rate ≥ {t.FR7_MIN_CACHE_HIT_RATE:.0%}",
    ),
    "FR-8": (
        "Six models, one API",
        f"all {t.FR8_MODEL_COUNT} models; config-only switch",
    ),
    "FR-9": (
        "Scale out, at most 2 workers",
        f"needed size within {t.FR9_SCALE_OUT_SECONDS:.0f} s; "
        f"never above {t.MAX_GPU_WORKERS}",
    ),
    "FR-10": (
        "Scale to zero, hold the first request",
        f"0 workers {t.FR10_RELEASE_AFTER_IDLE_SECONDS:.0f} s after the "
        "idle timeout; held requests complete",
    ),
    "FR-11": (
        "Fairness between departments",
        f"heavy ≤ {t.FR11_MAX_HEAVY_SHARE:.0%}, light "
        f"≥ {t.FR11_MIN_LIGHT_SHARE:.0%} at {t.FR11_LOAD_RATIO:.0f}:1",
    ),
    "FR-12": ("Audit record per request", "every request; no raw text"),
    "NFR-1": (
        "Warm latency",
        f"p95 ≤ {t.NFR1_P95_TOTAL_S} s; p95 first token "
        f"≤ {t.NFR1_P95_TTFT_S} s",
    ),
    "NFR-2": (
        "Cold-start latency",
        f"first token ≤ {t.NFR2_COLD_TTFT_SMALL_S:.0f} s (4B)",
    ),
    "NFR-3": (
        "Throughput per worker",
        f"≥ {t.NFR3_TOKENS_PER_S_4B:.0f} output tokens/s (4B)",
    ),
    "NFR-4": (
        "Scale and elasticity",
        f"≤ {t.NFR4_MAX_ERROR_FRACTION:.0%} errors at "
        f"{t.NFR4_USERS_PER_WORKER} users; search p95 "
        f"≤ {t.NFR4_SEARCH_P95_S * 1000:.0f} ms",
    ),
    "NFR-5": (
        "Availability",
        f"≥ {t.NFR5_MIN_SUCCESS_FRACTION:.0%} success; restore "
        f"≤ {t.NFR5_RESTORE_SECONDS:.0f} s",
    ),
    "NFR-6": (
        "Cost",
        f"≤ {t.NFR6_MAX_GPU_HOURS_PER_1K} GPU-h per 1,000 queries",
    ),
    "NFR-7": (
        "Answer quality",
        f"accuracy ≥ {t.NFR7_MIN_ACCURACY:.0%}; citations "
        f"≥ {t.NFR7_MIN_CITATION_CORRECTNESS:.0%}",
    ),
    "NFR-8": (
        "Freshness",
        f"≥ {t.NFR8_MIN_CURRENT_FRACTION:.0%} current within "
        f"{t.FR5_STALE_DEMOTION_SECONDS / 60:.0f} min",
    ),
    "NFR-9": (
        "Security and privacy",
        f"{t.NFR9_MAX_LEAKS} leaks; injection success "
        f"≤ {t.NFR9_MAX_INJECTION_SUCCESS:.0%}",
    ),
    "NFR-10": (
        "Access correctness and speed",
        f"{t.NFR10_MAX_FALSE_ALLOWS} false allows; authorize "
        f"≤ {t.NFR10_AUTHORIZE_MS:.0f} ms, filter ≤ {t.NFR10_FILTER_MS:.0f} ms",
    ),
}


@dataclass(frozen=True)
class Row:
    """One requirement on the scorecard."""

    id: str
    name: str
    target: str
    value: str
    verdict: str
    date: str
    source: str


def _get(summary: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = summary
    for key in path:
        if key == "by_request" and node is summary:
            node = {r["name"]: r for r in summary.get("by_request", [])}
        elif isinstance(node, Mapping) and key in node:
            node = node[key]
        else:
            return None
    return node


def read_value(
    summary: Mapping[str, Any], spec: Mapping[str, Any]
) -> float | None:
    """One number from a summary, as a scorecard_map.yaml value spec says.

    Returns:
        The number, or None if the summary does not have it.
    """
    if "ratio" in spec:
        ratio = spec["ratio"]
        num = _get(summary, ratio["num"])
        den = [_get(summary, p) for p in ratio["den"]]
        if num is None:
            num = 0 if any(d is not None for d in den) else None
        total = sum(float(d) for d in den if d is not None)
        if num is None or total == 0:
            return None
        return float(num) / total
    value = _get(summary, spec["path"])
    if value is None:
        value = spec.get("default")
    scale = float(spec.get("scale", 1))
    return None if value is None else float(value) * scale


def target_of(check: Mapping[str, Any]) -> float:
    """The threshold: a targets.py constant (times scale) or a number."""
    raw = check["target"]
    base = float(getattr(t, raw)) if isinstance(raw, str) else float(raw)
    return base * float(check.get("scale", 1))


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"


def evaluate(
    summary: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]
) -> tuple[str, bool] | None:
    """Applies a requirement's checks to one run.

    Returns:
        The value text and whether every check passed, or None if the run
        lacks a value any check needs.
    """
    parts, passed = [], True
    for check in checks:
        value = read_value(summary, check["value"])
        if value is None:
            return None
        ok = OPS[check["op"]](value, target_of(check))
        passed = passed and ok
        if "ratio" in check["value"]:
            shown = f"{value:.1%}"
        elif "digits" in check:
            shown = f"{value:.{int(check['digits'])}f}"
        else:
            shown = _fmt(value)
        parts.append(f"{check['label']}: {shown}")
    return "; ".join(parts), passed


def _date(epoch: float) -> str:
    import datetime as dt  # noqa: PLC0415

    return dt.datetime.fromtimestamp(epoch, tz=dt.UTC).strftime("%Y-%m-%d")


def scorecard(
    runs: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any] | None = None,
    baseline: Mapping[str, Any] | None = None,
) -> list[Row]:
    """Every requirement with its latest result.

    Args:
        runs: Locust run summaries (any order).
        mapping: scorecard_map.yaml contents (default: the packaged file).
        baseline: baseline_v1.yaml contents (default: the packaged file).

    Returns:
        One row per requirement, FR-1 to FR-12 then NFR-1 to NFR-10.
    """
    mapping = mapping if mapping is not None else load(MAP_FILE)
    baseline = baseline if baseline is not None else load(BASELINE_FILE)
    newest = sorted(runs, key=lambda r: r.get("finished", 0), reverse=True)
    rows = []
    for rid, (name, target) in REQUIREMENTS.items():
        row = _from_runs(rid, name, target, newest, mapping.get(rid))
        if row is None:
            base = baseline["results"].get(rid, {})
            row = Row(
                rid,
                name,
                target,
                base.get("value", "not measured"),
                base.get("verdict", "pending"),
                str(baseline.get("date", "")),
                "baseline-v1",
            )
        rows.append(row)
    return rows


def _part(
    spec: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], str, bool] | None:
    """The newest run of the spec's workload(s) that has every value."""
    names = spec["workload"]
    names = {names} if isinstance(names, str) else set(names)
    for run in runs:
        if run.get("workload") not in names:
            continue
        result = evaluate(run, spec["checks"])
        if result is not None:
            return run, result[0], result[1]
    return None


def _from_runs(
    rid: str,
    name: str,
    target: str,
    runs: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> Row | None:
    """A row from the newest runs; a list spec combines several workloads.

    With several parts the verdict is fail if any part fails, pass if every
    part was measured and passed, and pending otherwise.
    """
    if spec is None:
        return None
    specs = [spec] if isinstance(spec, Mapping) else list(spec)
    parts = [_part(s, runs) for s in specs]
    found = [p for p in parts if p is not None]
    if not found:
        return None
    values = [v for _, v, _ in found]
    values += [
        f"{_names(s['workload'])}: not measured"
        for s, p in zip(specs, parts, strict=True)
        if p is None
    ]
    if not all(ok for _, _, ok in found):
        verdict = "fail"
    else:
        verdict = "pass" if len(found) == len(specs) else "pending"
    finished = max(float(r.get("finished", 0)) for r, _, _ in found)
    sources = ", ".join(_source(r) for r, _, _ in found)
    return Row(
        rid, name, target, "; ".join(values), verdict, _date(finished), sources
    )


def _names(workload: str | Sequence[str]) -> str:
    return workload if isinstance(workload, str) else " or ".join(workload)


def _source(run: Mapping[str, Any]) -> str:
    kind = run.get("kind", "Locust")
    return f"{kind} {run.get('name', '')}"


def load(path: Path) -> dict[str, Any]:
    """Reads a YAML file."""
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}
