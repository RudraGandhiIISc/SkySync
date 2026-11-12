# ruff: noqa: E501 - Markdown rows read best unwrapped
"""Compiles docs/results/rt-baseline/README.md: the runs behind rt-baseline.

Reads only result files under $ASKIISC_DATA/results (never re-runs
anything). Three Slurm jobs of slurm/v3_stack.sh made them:

- V3 (`kv24 fresh nfr1 cold`): FR-5 and FR-3, NFR-1, NFR-2 and FR-10, and
  the Granite pair of the FR-8 KV-room check;
- KV (`kvq e0sel`): the Qwen pair of the KV check on a quiet card, and the
  E0 sweep of Qwen3-8B and Qwen3-8B-AWQ (NFR-3, FR-7);
- NEW (`faults fair2 nfr1 soak`), on the release code: service keys and
  the watchdog (NFR-9, FR-4, NFR-5), FR-11 with both departments
  backlogged, NFR-1 again and the NFR-5 soak.

Run (inside Slurm):
    python -m tools.baseline_results --v3 2234 --kv 2238 --new 2240
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.fr6_report import read_lines

import tools.v3_results as v3

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "results" / "rt-baseline"
FR8_PAIRS = (
    ("Qwen3-8B", "Qwen3-8B-AWQ"),
    ("granite-3.3-8b-instruct", "Granite-3.3-8B-Instruct-GPTQ"),
)
NFR3_TARGETS = {"qwen3-8b": 250.0, "qwen3-8b-awq": 450.0}
FR11_MAX_HEAVY = 0.55
FR11_MIN_LIGHT = 0.20
NFR5_MIN_SUCCESS = 0.99
NFR5_RESTORE_S = 180.0


def kv_tokens(log: Path) -> tuple[float, int] | None:
    """Weights (GiB) and KV-cache tokens of a vLLM start-up log."""
    start = v3.parse_log(log)
    return None if start is None else (start.weights_gib, start.kv_cache_tokens)


def kv_rows(base: Path, folders: dict[str, list[str]]) -> list[str]:
    """One row per 4-bit model: its KV tokens against its parent's.

    Args:
        base: The results folder that holds the run folders.
        folders: For each model name, its run folders (repeats allowed).

    Returns:
        Markdown table rows.
    """
    rows = []
    for parent, child in FR8_PAIRS:
        p = [kv_tokens(base / f / "vllm.log") for f in folders.get(parent, [])]
        c = [kv_tokens(base / f / "vllm.log") for f in folders.get(child, [])]
        p_ok, c_ok = [x for x in p if x], [x for x in c if x]
        if not (p_ok and c_ok):
            rows.append(
                f"| {child} against {parent} | n/a | n/a | n/a | not measured |"
            )
            continue
        low = min(t for _, t in c_ok) / max(t for _, t in p_ok)
        rows.append(
            f"| {child} against {parent} | {c_ok[0][0]:.2f} / {p_ok[0][0]:.2f} | {_tokens(c_ok)} / {_tokens(p_ok)} | {low:.2f}x | {'pass' if low >= 2 else 'fail'} |"  # noqa: PLR2004
        )
    return rows


def _tokens(runs: Sequence[tuple[float, int]]) -> str:
    return ", ".join(f"{t:,}" for _, t in runs)


def kv_section(v3_job: str, kv_job: str) -> list[str]:
    """FR-8: KV-cache room of each 4-bit model at one 24 GiB budget."""
    base = v3.DATA / "v3"
    folders = {
        "granite-3.3-8b-instruct": [f"{v3_job}/kv24/granite-3.3-8b-instruct"],
        "Granite-3.3-8B-Instruct-GPTQ": [
            f"{v3_job}/kv24/Granite-3.3-8B-Instruct-GPTQ"
        ],
        "Qwen3-8B": [f"{kv_job}/kvq/Qwen3-8B-{k}" for k in (1, 2)],
        "Qwen3-8B-AWQ": [f"{kv_job}/kvq/Qwen3-8B-AWQ-{k}" for k in (1, 2)],
    }
    return [
        "## FR-8: KV-cache room at one 24 GiB budget",
        "",
        "Target: at the same GPU memory budget (`--gpu-memory-utilization",
        "0.17`, 24 GiB), each 4-bit model holds at least 2x the KV tokens of",
        "its 8B parent. The ratio below is the worst case: the 4-bit model's",
        "smallest run against the parent's largest.",
        "",
        "| Pair | Weights GiB (4-bit / parent) | KV tokens (4-bit / parent) | Worst ratio | Verdict |",
        "|---|---|---|---|---|",
        *kv_rows(base, folders),
        "",
        f"The Qwen pair of job {v3_job} is not used: both ran on a card where",
        "another user's job grew while vLLM measured its memory, and vLLM",
        "counts that growth against the KV budget (Qwen3-8B-AWQ got 2.53 GiB",
        f"there, 9.05 and 12.30 GiB on a quiet card in job {kv_job}).",
        "",
        f"Raw files: `{base}/{v3_job}/kv24/`, `{base}/{kv_job}/kvq/` (`vllm.log`).",
        "",
    ]


def sweep(path: Path) -> dict[int, dict[str, Any]]:
    """An E0 sweep (fr8_bench.sh) by concurrency."""
    rows = v3.load(path) or []
    return {int(r["concurrency"]): r for r in rows}


def e0_section(kv_job: str) -> list[str]:
    """NFR-3 and FR-7: throughput of the selected models on one card."""
    out = [
        "## NFR-3 and FR-7: throughput on one card (E0)",
        "",
        "vLLM alone, closed loop with the gateway's prompts. NFR-3 reads",
        "concurrency 64; FR-7 compares it with one request at a time.",
        "",
        "| Model | Tokens/s at 1 | Tokens/s at 64 | NFR-3 target | NFR-3 | FR-7 gain (64 against 1) | FR-7 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = 0
    for key, target in NFR3_TARGETS.items():
        points = sweep(
            v3.DATA / "models" / key / f"bench-{kv_job}" / "sweep.json"
        )
        one, many = points.get(1), points.get(64)
        if not (one and many):
            out.append(
                f"| {key} | n/a | n/a | {target:.0f} | {v3.NOT_RUN} | n/a | {v3.NOT_RUN} |"
            )
            continue
        rows += 1
        a, b = one["output_tokens_per_s"], many["output_tokens_per_s"]
        gain = b / a if a else None
        out.append(
            f"| {key} | {a:,.0f} | {b:,.0f} | at least {target:.0f} | {'pass' if b >= target else 'fail'} | {v3.fmt(gain, 1)}x | {'pass' if gain and gain >= 2 else 'fail'} |"  # noqa: PLR2004
        )
    gpus = v3.DATA / "v3" / kv_job / "gpus_at_start.csv"
    return [
        *out,
        "",
        "Cards at the start of the job (another user's job can still start",
        "on the same card during the sweep):",
        "",
        "```",
        gpus.read_text().strip() if gpus.exists() else v3.NOT_RUN,
        "```",
        "",
        f"Raw files: `{v3.DATA}/models/<model>/bench-{kv_job}/sweep.json`.",
        "",
    ]


def faults_section(new_job: str) -> list[str]:
    """NFR-9, FR-4, NFR-5: keys and the watchdog on the live stack."""
    report = v3.load(
        v3.DATA / "v3" / new_job / "faults" / "report" / "fault_check.json"
    )
    out = ["## Service keys and the watchdog (live stack, release code)", ""]
    if report is None:
        return [*out, f"{v3.NOT_RUN}.", ""]
    rows = [
        "| Check | Expected | Measured |",
        "|---|---|---|",
        f"| A question through the chat app | answered | {'answered' if report['answered_before'] else 'not answered'} |",
    ]
    for url, codes in report["refusals"]["workers"].items():
        rows.append(
            f"| vLLM worker {url}: /v1 and /invocations without the key; /health | 401, 401; 200 | {codes['chat_without_key']}, {codes['invocations_without_key']}; {codes['health']} |"
        )
    e = report["refusals"]["embed"]
    rows.append(
        f"| Embedding server: /embed and /rerank without the key; /health | 401, 401; 200 | {e['embed_without_key']}, {e['rerank_without_key']}; {e['health']} |"
    )
    for name, f in report["faults"].items():
        if "error" in f:
            rows.append(
                f"| Kill the {name} service | restarted | {f['error']} |"
            )
            continue
        rows.append(
            f"| Kill the {name} service (SIGKILL): new process, healthy, question answered | restarted | {v3.fmt(f['new_process_s'])} s, {v3.fmt(f['healthy_s'])} s, {v3.fmt(f['answered_s'])} s |"
        )
    return [
        *out,
        *rows,
        "",
        f"Verdict: {'pass' if report.get('passed') else 'fail'}. Raw file: `{v3.DATA}/v3/{new_job}/faults/report/fault_check.json`.",
        "",
    ]


def token_shares(audit: Path, start: float, end: float) -> dict[str, float]:
    """Each department's share of the generated tokens in a time window."""
    tokens: dict[str, float] = defaultdict(float)
    for r in read_lines(audit):
        if start <= float(r["time"]) <= end and not r.get("cache_hit"):
            tokens[r["tenant"]] += float(r.get("completion_tokens") or 0)
    total = sum(tokens.values())
    return {t: n / total for t, n in tokens.items()} if total else {}


def fairness_section(new_job: str) -> list[str]:
    """FR-11 with both departments backlogged."""
    base = v3.DATA / "v3" / new_job / "fairness-backlogged"
    run = next(iter(sorted(base.glob("rep1/*/summary.json"))), None)
    out = ["## FR-11: fairness with both departments backlogged", ""]
    summary = v3.load(run) if run else None
    shares = (
        token_shares(
            base / "serve" / "audit.jsonl",
            summary["started"],
            summary["finished"],
        )
        if summary
        else {}
    )
    if len(shares) < 2:  # noqa: PLR2004
        return [*out, f"{v3.NOT_RUN}.", ""]
    heavy, light = sorted(shares.values(), reverse=True)[:2]
    ok = heavy <= FR11_MAX_HEAVY and light >= FR11_MIN_LIGHT
    return [
        *out,
        "`fairness` workload: 40 users of the busy department and 4 of the",
        "quiet one, all asking back to back (10:1); 6 LLM slots and the",
        "answer cache off, so both departments always have more requests",
        "waiting than their fair share.",
        "",
        "| Check | Target | Measured | Verdict |",
        "|---|---|---|---|",
        f"| Busy department's share of tokens | at most 55% | {100 * heavy:.1f}% | {'pass' if heavy <= FR11_MAX_HEAVY else 'fail'} |",
        f"| Quiet department's share of tokens | at least 20% | {100 * light:.1f}% | {'pass' if light >= FR11_MIN_LIGHT else 'fail'} |",
        "",
        f"Verdict: {'pass' if ok else 'fail'}. Raw files: `{base}/` (audit records and the Locust run).",
        "",
    ]


def soak_section(new_job: str) -> list[str]:
    """NFR-5: one hour at half load, one worker killed at 30 minutes."""
    base = v3.DATA / "v3" / new_job / "soak"
    run = next(iter(sorted(base.glob("rep1/*/summary.json"))), None)
    out = ["## NFR-5: 1 hour at half load, a worker killed at 30 minutes", ""]
    summary = v3.load(run) if run else None
    if summary is None:
        return [*out, f"{v3.NOT_RUN}.", ""]
    success = 1 - summary["failures"] / max(1, summary["requests"])
    restore = restore_s(v3.load(base / "serve" / "workers.json") or [])
    killed = summary.get("checks", {}).get("chaos_kill_worker_ok", 0)
    return [
        *out,
        "| Check | Target | Measured | Verdict |",
        "|---|---|---|---|",
        f"| Non-error responses | at least 99% | {100 * success:.2f}% of {summary['requests']:,} | {'pass' if success >= NFR5_MIN_SUCCESS else 'fail'} |",
        f"| Worker killed | once, at 30 minutes | {killed} | |",
        f"| Capacity restored after the kill | within 180 s | {v3.fmt(restore)} s | {'pass' if restore is not None and restore <= NFR5_RESTORE_S else 'fail'} |",
        "",
        f"Raw files: `{base}/`.",
        "",
    ]


def restore_s(workers: Sequence[dict[str, Any]]) -> float | None:
    """Seconds from the first worker's death to the next worker ready."""
    rows = sorted(workers, key=lambda w: w.get("requested_at") or 0.0)
    for gone, nxt in zip(rows, rows[1:], strict=False):
        if gone.get("released_at") and nxt.get("ready_at"):
            return round(float(nxt["ready_at"]) - float(gone["released_at"]), 1)
    return None


def titled(lines: list[str], job: str, what: str) -> list[str]:
    """A section with the job and code version added to its title."""
    return [f"{lines[0]} ({what}, job {job})", *lines[1:]] if lines else lines


def main(argv: Sequence[str] | None = None) -> int:
    """Writes docs/results/rt-baseline/README.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--kv", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args(argv)
    lines = [
        "# Evidence for the red-team baseline",
        "",
        "Compiled by `tools/baseline_results.py` from raw result files only.",
        f"Jobs {args.v3} and {args.kv} ran the code of `main` at 74401a5;",
        f"job {args.new} ran the release code (service keys, watchdog, full-GPU",
        "handling). Every run used one GPU worker on a card that other users'",
        "jobs could also use (risk R-3). Qwen3-8B-AWQ served the FR-5 and FR-3",
        "runs; Qwen3-8B served the rest.",
        "",
        *v3.fresh_section(args.v3),
        *titled(v3.nfr1_section(args.v3), args.v3, "code at 74401a5"),
        *titled(v3.nfr1_section(args.new), args.new, "release code"),
        *v3.cold_section(args.v3),
        *kv_section(args.v3, args.kv),
        *e0_section(args.kv),
        *faults_section(args.new),
        *fairness_section(args.new),
        *soak_section(args.new),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'README.md'}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
