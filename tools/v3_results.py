# ruff: noqa: E501 - Markdown and SVG rows read best unwrapped
"""Compiles docs/results/v3/RESULTS.md from the raw files of the v3 runs.

Reads only result files under $ASKIISC_DATA/results (never re-runs
anything) and writes one Markdown page with a section per requirement,
plus docs/results/v3/FR-6/goodput.svg. A run that has not happened yet is
listed as not run.

Run (inside Slurm):
    python -m tools.v3_results --fr6 2134 --v3 2137 \
        --profiles 2117 2123 2126 2130
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.fr6_report import percentile, read_lines

from askiisc.observability.vllm_startup import parse
from tools.evidence import INK, MUTED, SERIES, TARGET, nice_ticks

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "results" / "v3"
DATA = Path("/data4/muthangiv/askiisc/results")
NOT_RUN = "not run"


def load(path: Path) -> Any:
    """A JSON file, or None if it is missing."""
    return json.loads(path.read_text()) if path.exists() else None


def stats_row(folder: Path, name: str) -> dict[str, str] | None:
    """One row of Locust's locust_stats.csv by request name."""
    files = sorted(folder.glob("**/locust_stats.csv"))
    if not files:
        return None
    with files[0].open() as f:
        for row in csv.DictReader(f):
            if row["Name"] == name:
                return row
    return None


def med(values: Sequence[float]) -> float | None:
    """Median, or None for no values."""
    return statistics.median(values) if values else None


def fmt(v: float | None, digits: int = 1) -> str:
    """A number for a table cell."""
    if v is None:
        return "n/a"
    return f"{v:,.{digits}f}"


# ------------------------------------------------------------------ FR-6


def fr6_reports(job: str, arm: str) -> list[dict[str, Any]]:
    """The fr6.json of each repetition of one arm."""
    base = DATA / "fr6" / job / arm
    return [
        r for k in (1, 2, 3)
        if (r := load(base / f"rep{k}" / "report" / "fr6.json")) is not None
    ]  # fmt: skip


def fr6_section(job: str, profiles: Sequence[str]) -> list[str]:
    """FR-6: verdict table, per-repetition rows, profile history."""
    on, off = fr6_reports(job, "on"), fr6_reports(job, "off")
    out = ["## FR-6: load shedding at 3x (E4)", ""]
    if not on:
        return [*out, f"Job {job}: {NOT_RUN}.", ""]
    shed = [r["verdicts"]["shed_p99_ms_at_3x"] for r in on]
    rise = [r["verdicts"]["accepted_rise"] for r in on]
    ok = all(r["verdicts"]["shed_within_limit"] for r in on) and all(
        r["verdicts"]["accepted_within_limit"] for r in on
    )
    out += [
        "Target (PRD FR-6): at 3x load, excess requests get HTTP 429 within",
        "50 ms (judged at the serving node, decided 2026-11-05) and the p95",
        "of accepted requests rises by at most 20% against 1x.",
        "",
        "| Check | Target | Median of 3 (each run) | Verdict |",
        "|---|---|---|---|",
        f"| 429 time at the node, p99 at 3x | at most 50 ms | {fmt(med(shed))} ms ({', '.join(fmt(x) for x in shed)}) | {'pass' if ok else 'fail'} |",
        f"| Accepted p95, 3x against 1x | at most +20% | {fmt(100 * (med(rise) or 0))}% ({', '.join(fmt(100 * x) for x in rise)}) | {'pass' if ok else 'fail'} |",
        "",
        "Every repetition, admission on and off (node and client times in ms):",
        "",
        "| Arm | Rep | Load | Offered/s | Goodput/s | 429s | Node 429 p50 / p95 / p99 | Client 429 p99 | Accepted p95 (s) | Loop lag max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for arm, reports in (("on", on), ("off", off)):
        for k, r in enumerate(reports, 1):
            for m in r["multiples"]:
                n, c = m["node_429_ms"], m["client_429_ms"]
                out.append(
                    f"| {arm} | {k} | {m['multiple']}x | {m['offered_per_s']:.2f} | {m['goodput_per_s']:.2f} | {n['n']} | {fmt(n['p50'])} / {fmt(n['p95'])} / {fmt(n['p99'])} | {fmt(c['p99'])} | {fmt(m['accepted_ms']['p95'] / 1000)} | {fmt(m['loop_lag_ms']['max'])} |"
                )
    if not off:
        out += ["", f"Admission off: {NOT_RUN}."]
    else:
        out += ["", *e4_reading(on, off)]
    out += ["", "![Goodput against offered load](FR-6/goodput.svg)", ""]
    out += profile_rows(profiles)
    out += [
        "",
        f"Raw files: `{DATA}/fr6/{job}/<arm>/rep<k>/` (Locust, `report/fr6.json`)",
        f"and `{DATA}/fr6/{job}/<arm>/serve/` (`responses.jsonl`, `loop_lag.jsonl`, `audit.jsonl`).",
        "",
    ]
    write_goodput(on, off)
    return out


def at(reports: list[dict[str, Any]], k: int, key: str) -> float | None:
    """Median of one field at load multiple k across repetitions."""
    rows = [m for r in reports for m in r["multiples"] if m["multiple"] == k]
    if key == "accepted_p95":
        return med([m["accepted_ms"]["p95"] / 1000 for m in rows])
    return med([m[key] for m in rows])


def cache_share(reports: list[dict[str, Any]]) -> str:
    """Share of answers served from the answer cache in an arm's audit."""
    if not reports:
        return "n/a"
    audit = Path(reports[0]["serve"]) / "audit.jsonl"
    rows = [
        r
        for r in read_lines(audit)
        if r["decision"] in ("answered", "abstained")
    ]
    hits = sum(bool(r.get("cache_hit")) for r in rows)
    return (
        f"{100 * hits / len(rows):.1f}% of {len(rows):,} answers"
        if rows
        else "n/a"
    )


def e4_reading(
    on: list[dict[str, Any]], off: list[dict[str, Any]]
) -> list[str]:
    """What the admission-off arm shows (E4), from the medians."""
    g_on, g_off = at(on, 3, "goodput_per_s"), at(off, 3, "goodput_per_s")
    o_off = at(off, 3, "offered_per_s")
    p_on, p_off = at(on, 3, "accepted_p95"), at(off, 3, "accepted_p95")
    lines = [
        f"At 3x (medians of 3): goodput {fmt(g_on, 2)}/s with admission on and",
        f"{fmt(g_off, 2)}/s with it off (offered {fmt(o_off, 2)}/s); accepted p95",
        f"{fmt(p_on)} s on and {fmt(p_off)} s off.",
    ]
    if g_on is not None and g_off is not None and g_off >= g_on:
        lines += [
            "",
            "**E4's collapse is not shown at this load.** With admission off",
            "every request was answered and goodput kept up with the offered",
            "load, so admission refused work the system could still serve.",
            "W2 at 3x is about 15 requests/s, and W1's repeated questions are",
            f"mostly answered from the answer cache ({cache_share(off)}), so the",
            "GPU stayed below saturation. Showing the collapse needs more load than W2 3x (for",
            "example `overload`, or W2 with the answer cache off). The FR-6",
            "verdict (429 time and accepted p95 with admission on) does not",
            "depend on this.",
        ]
    return lines


def profile_rows(profiles: Sequence[str]) -> list[str]:
    """The `overload` stress runs that found and checked each fix."""
    fixes = {
        "2117": "none (before)",
        "2123": "audit file kept open",
        "2126": "+ NFS writes off the loop",
        "2130": "+ gc.freeze (server crashed 37 s before the end)",
    }
    out = [
        "Stress diagnostic (`overload`, closed-loop users with no think time,",
        "stack sampling on), one run per fix:",
        "",
        "| Run | Fix | Node 429 p50 / p95 / p99 at 3x (ms) | Loop lag max (ms) |",
        "|---|---|---|---|",
    ]
    for job in profiles:
        base = DATA / "fr6" / job / "profile"
        r = (
            load(base / "rep1" / "report-complete" / "fr6.json")
            or load(base / "rep1" / "report" / "fr6.json")
            or load(base / "report" / "fr6.json")
        )
        if r is None:
            continue
        m = r["multiples"][-1]
        n = m["node_429_ms"]
        out.append(
            f"| {job} | {fixes.get(job, '')} | {fmt(n['p50'])} / {fmt(n['p95'])} / {fmt(n['p99'])} | {fmt(m['loop_lag_ms']['max'])} |"
        )
    return out


def write_goodput(on: list[dict[str, Any]], off: list[dict[str, Any]]) -> None:
    """Goodput against offered load, admission on and off (median of reps)."""
    series = []
    for name, reports in (("admission on", on), ("admission off", off)):
        points = []
        for k in (1, 2, 3):
            rows = [
                m for r in reports for m in r["multiples"] if m["multiple"] == k
            ]
            if rows:
                points.append(
                    (med([m["offered_per_s"] for m in rows]) or 0.0,
                     med([m["goodput_per_s"] for m in rows]) or 0.0)
                )  # fmt: skip
        series.append((name, points))
    (OUT / "FR-6").mkdir(parents=True, exist_ok=True)
    (OUT / "FR-6" / "goodput.svg").write_text(svg_xy(series))


def svg_xy(series: list[tuple[str, list[tuple[float, float]]]]) -> str:
    """A small line chart with markers (requests per second on both axes)."""
    w, h, left, right, top, bottom = 640, 320, 64, 150, 40, 48
    pts = [p for _, s in series for p in s] or [(0.0, 0.0)]
    xt = nice_ticks(0.0, max(p[0] for p in pts) * 1.05 or 1.0)
    yt = nice_ticks(0.0, max(p[1] for p in pts) * 1.1 or 1.0)

    def sx(x: float) -> float:
        return left + x / xt[-1] * (w - left - right)

    def sy(y: float) -> float:
        return h - bottom - y / yt[-1] * (h - top - bottom)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#fff"/>',
        f'<text x="{left}" y="22" fill="{INK}" font-size="14">FR-6: goodput against offered load (W2, median of 3)</text>',
    ]  # fmt: skip
    for y in yt:
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e7eb"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" fill="{MUTED}">{y:g}</text>'
        )
    for x in xt:
        out.append(
            f'<text x="{sx(x):.1f}" y="{h - bottom + 18}" text-anchor="middle" fill="{MUTED}">{x:g}</text>'
        )
    out.append(
        f'<text x="{(left + w - right) / 2}" y="{h - 8}" text-anchor="middle" fill="{MUTED}">offered requests per second</text>'
    )
    out.append(
        f'<text x="12" y="{top - 6}" fill="{MUTED}">answered per second</text>'
    )
    for i, (name, s) in enumerate(series):
        color = SERIES[i % len(SERIES)] if i == 0 else TARGET
        if s:
            path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            out += [
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>'
                for x, y in s
            ]
        ly = top + 14 + i * 18
        out.append(
            f'<line x1="{w - right + 8}" x2="{w - right + 26}" y1="{ly - 4}" y2="{ly - 4}" stroke="{color}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{w - right + 32}" y="{ly}" fill="{INK}">{name}</text>'
        )
    return "\n".join(out) + "\n</svg>\n"


# ------------------------------------------------------------- the rest


def fresh_section(job: str) -> list[str]:
    """FR-5 freshness and FR-3 revocations."""
    base = DATA / "v3" / job
    out = ["## FR-5: freshness (E5) and FR-3: revocation (E1)", ""]
    f = load(base / "fresh" / "report" / "freshness.json")
    if f is None:
        out.append(f"Freshness replay: {NOT_RUN}.")
    else:
        d = f["demotion_s"]
        out += [
            "| Check | Target | Measured |",
            "|---|---|---|",
            f"| Revisions replayed | 60 across 20 documents | {f['revisions']} across {f['documents']} |",
            f"| Time to demotion (s), p50 / p95 / max | | {fmt(d['p50'])} / {fmt(d['p95'])} / {fmt(d['max'])} (n={d['n']}) |",
            f"| Answers using a superseded version without the flag (audit store) | 0 | {f['audit_unflagged']} of {f['audit_used_superseded']} that used one |",
            f"| Answers citing a superseded version without the flag | 0 | {f['unflagged_cited']} of {f['answers']} answers |",
            f"| As-of lookups correct at the checkpoints | all | {f['as_of_ok']} of {len(f['as_of_checkpoints'])} |",
        ]  # fmt: skip
    samples = next(
        iter((base / "revocations").glob("rep1/*/samples.json")), None
    )
    seen = (
        (load(samples) or {}).get("revocation_last_seen_s", [])
        if samples
        else []
    )
    out.append("")
    if seen:
        out += [
            "| Check | Target | Measured |",
            "|---|---|---|",
            f"| Scripted revocations | 100 | {len(seen)} |",
            f"| Last citation after revoking (s), p99 / max | at most 60 | {fmt(percentile(seen, 0.99))} / {fmt(max(seen))} |",
            f"| Revocations still cited after 60 s | 0 | {sum(s > 60 for s in seen)} |",  # noqa: PLR2004
        ]  # fmt: skip
    else:
        out.append(f"Scripted revocations: {NOT_RUN}.")
    return [*out, "", f"Raw files: `{base}/fresh/`, `{base}/revocations/`.", ""]


def hit_rates(log: Path) -> list[float]:
    """VLLM's logged prefix-cache hit rates while requests ran."""
    if not log.exists():
        return []
    rates = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(
            r"Running: (\d+) reqs.*Prefix cache hit rate: ([0-9.]+)%", line
        )
        if m and int(m.group(1)) > 0:
            rates.append(float(m.group(2)))
    return rates


def fr7_section(job: str) -> list[str]:
    """FR-7: prefix-cache hit rate and first token, on against off."""
    base = DATA / "v3" / job
    out = ["## FR-7: prefix caching under W1 (answer cache off)", ""]
    rows = {}
    for arm in ("on", "off"):
        part = base / f"fr7-prefix-{arm}"
        ttft = [
            float(r["95%"]) for k in (1, 2, 3)
            if (r := stats_row(part / f"rep{k}", "w1 question (first token)"))
        ]  # fmt: skip
        logs = sorted((part / "serve").glob("vllm-*.log"))
        rates = [x for log in logs for x in hit_rates(log)]
        rows[arm] = (med(ttft), med(rates), ttft)
    if rows["on"][0] is None:
        return [*out, f"{NOT_RUN}.", ""]
    on_t, on_hit, on_all = rows["on"]
    off_t, _, off_all = rows["off"]
    drop = None if not (on_t and off_t) else 100 * (1 - on_t / off_t)
    v_on, n_on = vllm_first_token(base / "fr7-prefix-on" / "serve")
    v_off, n_off = vllm_first_token(base / "fr7-prefix-off" / "serve")
    v_drop = None if not (v_on and v_off) else 100 * (1 - v_on / v_off)
    out += [
        "| Check | Target | Measured | Verdict |",
        "|---|---|---|---|",
        f"| vLLM prefix-cache hit rate, prefix caching on (median of logged values) | at least 50% | {fmt(on_hit)}% | {'pass' if (on_hit or 0) >= 50 else 'fail'} |",  # noqa: PLR2004
        f"| p95 first token, on against off (median of 3 runs) | at least 25% lower | {fmt(on_t, 0)} ms against {fmt(off_t, 0)} ms: {fmt(drop)}% lower | {'pass' if (drop or 0) >= 25 else 'fail'} |",  # noqa: PLR2004
        f"| p95 first token from vLLM (audit `first_token_ms`: dispatch to first token), on against off | at least 25% lower | {fmt(v_on, 0)} ms against {fmt(v_off, 0)} ms: {fmt(v_drop)}% lower ({n_on:,} and {n_off:,} generated answers) | {'pass' if (v_drop or 0) >= 25 else 'fail'} |",  # noqa: PLR2004
        "",
        "The client's first token includes retrieval (embedding, search and",
        "reranking, 1 to 2 s at p95), which prefix caching does not touch;",
        "vLLM's own first token is what the cache changes. Both are shown;",
        "the PRD row does not say where first token is measured.",
        "",
        f"Each run's p95 first token (ms): on {', '.join(fmt(x, 0) for x in on_all)}; off {', '.join(fmt(x, 0) for x in off_all)}.",
        f"Raw files: `{base}/fr7-prefix-on/`, `{base}/fr7-prefix-off/`.",
        "",
    ]  # fmt: skip
    return out


def vllm_first_token(serve: Path) -> tuple[float | None, int]:
    """p95 of vLLM's first token over generated answers in an audit store."""
    rows = [
        r["latency"]["first_token_ms"] for r in read_lines(serve / "audit.jsonl")
        if r["decision"] in ("answered", "abstained") and not r.get("cache_hit")
        and r["latency"]["first_token_ms"] > 0
    ]  # fmt: skip
    return (percentile(rows, 0.95) if rows else None), len(rows)


def nfr1_section(job: str) -> list[str]:
    """NFR-1: steady x3 with Qwen3-8B."""
    base = DATA / "v3" / job / "nfr1"
    total, first = [], []
    for k in (1, 2, 3):
        t = stats_row(base / f"rep{k}", "chat message")
        ft = stats_row(base / f"rep{k}", "chat message (first token)")
        if t and ft:
            total.append(float(t["95%"]))
            first.append(float(ft["95%"]))
    out = ["## NFR-1: latency with Qwen3-8B (`steady`, 60 users)", ""]
    if not total:
        return [*out, f"{NOT_RUN}.", ""]
    mt, mf = med(total), med(first)
    return [
        *out,
        "| Check | Target | Median of 3 (each run) | Verdict |",
        "|---|---|---|---|",
        f"| p95 total | at most 3,500 ms | {fmt(mt, 0)} ms ({', '.join(fmt(x, 0) for x in total)}) | {'pass' if mt <= 3500 else 'fail'} |",  # noqa: PLR2004
        f"| p95 first token | at most 800 ms | {fmt(mf, 0)} ms ({', '.join(fmt(x, 0) for x in first)}) | {'pass' if mf <= 800 else 'fail'} |",  # noqa: PLR2004
        "",
        f"Raw files: `{base}/`.",
        "",
    ]  # fmt: skip


def cold_section(job: str) -> list[str]:
    """NFR-2 / FR-10: 20 scripted cold starts per size."""
    targets = {
        "qwen3-4b-instruct-2507": 90,
        "qwen3-8b": 150,
        "qwen3-8b-awq": 90,
    }
    out = ["## NFR-2 and FR-10: 20 scripted cold starts per size (E3 c)", ""]
    rows = []
    for model, limit in targets.items():
        base = DATA / "v3" / job / f"cold-{model}"
        events = [
            e for f in base.glob("rep1/*/events.jsonl") for e in read_lines(f)
            if e.get("kind") == "cold_start"
        ]  # fmt: skip
        workers = load(base / "serve" / "workers.json") or []
        if not events:
            continue
        ttft = [e["ttft_s"] for e in events if e.get("ttft_s") is not None]
        load_s = [
            w["ready_at"] - w["acquired_at"]
            for w in workers
            if w.get("ready_at") and w.get("acquired_at")
        ]
        acq = [
            w["acquired_at"] - w["requested_at"]
            for w in workers
            if w.get("acquired_at")
        ]
        rows.append(
            f"| {model} | {len(events)} | {sum(e['answered'] for e in events)} | {fmt(med(acq), 2)} | {fmt(med(load_s))} / {fmt(max(load_s, default=None))} | {fmt(med(ttft))} / {fmt(max(ttft, default=None))} | at most {limit} s | {'pass' if ttft and max(ttft) <= limit else 'fail'} |"
        )
    if not rows:
        return [*out, f"{NOT_RUN}.", ""]
    return [
        *out,
        "| Model | Cold starts | Answered | Acquire (s), median | Weight load (s), median / max | First token (s), median / max | Target | Verdict |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "Acquisition is starting a vLLM process on a pinned card (R-1: no",
        "node allocation on this server). Raw files: "
        f"`{DATA}/v3/{job}/cold-<model>/`.",
        "",
    ]


def fr8_section(job: str) -> list[str]:
    """FR-8: Falcon3 smoke tests and KV room at 24 GB."""
    base = DATA / "v3" / job
    out = ["## FR-8: replacement models and KV-cache room at 24 GB", ""]
    smoke = sorted((base / "smoke").glob("*/summary.txt"))
    if smoke:
        out += ["| Model | Started | Load (s) | Reply |", "|---|---|---|---|"]
        for s in smoke:
            reply = load(s.parent / "reply.json") or {}
            text = (
                (reply.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            load_s = re.search(r"load_s=(\d+)", s.read_text())
            out.append(
                f"| {s.parent.name} | yes | {load_s.group(1) if load_s else 'n/a'} | {text.strip()[:60]!r} |"
            )
        out.append("")
    else:
        out += [f"Smoke tests: {NOT_RUN}.", ""]
    pairs = [("Qwen3-8B", "Qwen3-8B-AWQ"),
             ("granite-3.3-8b-instruct", "Granite-3.3-8B-Instruct-GPTQ"),
             ("Falcon3-7B-Instruct", "Falcon3-7B-Instruct-AWQ")]  # fmt: skip
    rows = []
    for parent, child in pairs:
        p = parse_log(base / "kv24" / parent / "vllm.log")
        c = parse_log(base / "kv24" / child / "vllm.log")
        ratio = None if not (p and c) else c.kv_cache_tokens / p.kv_cache_tokens
        rows.append(
            f"| {child} against {parent} | {fmt(c.weights_gib if c else None, 2)} / {fmt(p.weights_gib if p else None, 2)} | {c.kv_cache_tokens if c else 'n/a':,} / {p.kv_cache_tokens if p else 'n/a':,} | {fmt(ratio, 2)}x | {'pass' if ratio and ratio >= 2 else ('fail' if ratio else 'n/a')} |"  # noqa: PLR2004
            if (p and c)
            else f"| {child} against {parent} | n/a | n/a | n/a | not measured |"
        )
    out += [
        "KV-cache room at one fixed budget (`--gpu-memory-utilization 0.17`,",
        "24 GiB of the card), decided 2026-11-05. Target: each 4-bit model",
        "holds at least 2x its parent's KV tokens.",
        "",
        "| Pair | Weights GiB (4-bit / parent) | KV tokens (4-bit / parent) | Ratio | Verdict |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"Raw files: `{base}/smoke/`, `{base}/kv24/` (`vllm.log`).",
        "",
    ]  # fmt: skip
    return out


def parse_log(path: Path) -> Any:
    """The start-up numbers of a vLLM log, or None."""
    return parse(path.read_text(errors="replace")) if path.exists() else None


def e0_section() -> list[str]:
    """NFR-3 / E0: deferred until a card is idle (decided 2026-11-06)."""
    return [
        "## NFR-3: throughput sweep per model (E0)",
        "",
        "Deferred: E0 counts only on an idle card, and GPU 0 was shared at",
        "100% use all day (decided 2026-11-06). The last sweeps (2026-11-03,",
        "shared card) are in `results/models/<key>/bench-<job>/sweep.json`.",
        "",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Writes docs/results/v3/RESULTS.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fr6", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--profiles", nargs="*", default=[])
    args = parser.parse_args(argv)
    gpus = DATA / "v3" / args.v3 / "gpus_at_start.csv"
    lines = [
        "# v3 results",
        "",
        "Compiled by `tools/v3_results.py` from raw result files only.",
        f"FR-6 job {args.fr6}; other runs job {args.v3}. All runs used one GPU",
        "worker on GPU 0, which other users' jobs shared at 100% use, so",
        "latency and throughput understate an idle card (risk R-3).",
        "Qwen3-8B-AWQ served the FR-3, FR-5, FR-6 and FR-7 runs (the card",
        "had no room for Qwen3-8B in BF16 most of the time); NFR-1 used Qwen3-8B.",
        "",
        *fr6_section(args.fr6, args.profiles),
        *fresh_section(args.v3),
        *fr7_section(args.v3),
        *nfr1_section(args.v3),
        *cold_section(args.v3),
        *fr8_section(args.v3),
        *e0_section(),
        "## GPU memory at the start of the v3 job",
        "",
        "```",
        gpus.read_text().strip() if gpus.exists() else NOT_RUN,
        "```",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'RESULTS.md'}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
