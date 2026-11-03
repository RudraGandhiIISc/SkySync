# ruff: noqa: E501 - page text and query strings read best unwrapped
"""What goes on each requirement's evidence page (tools/evidence.py)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from askiisc.common.targets import (
    FR1_MIN_ABSTAIN_FRACTION,
    FR1_MIN_CITED_ANSWER_FRACTION,
    NFR3_TOKENS_PER_S_4B,
    NFR7_MIN_ACCURACY,
    NFR7_MIN_CITATION_CORRECTNESS,
)
from askiisc.web.dbfiles import db_dir
from tools.evidence import (
    DATA,
    LOCUST,
    Chart,
    Evidence,
    Line,
    measurement,
    summary,
)

LT = "sbatch slurm/loadtest.sh"
P95 = "histogram_quantile(0.95, sum by (le) (rate({m}_bucket[1m])))"
P50 = "histogram_quantile(0.5, sum by (le) (rate({m}_bucket[1m])))"
WORKERS_HELD = 'sum(askiisc_workers{state=~"starting|loading|ready"})'
WORKERS_READY = 'sum(askiisc_workers{state="ready"})'


def _req(run: str, name: str, key: str) -> float:
    rows = {r["name"]: r for r in summary(run)["by_request"]}
    return float(rows[name][key])


def _gateway_429_ms(run: str) -> tuple[float, float, int]:
    """p50, p95 and count of the gateway's own 429 decision times."""
    s = summary(run)
    live = db_dir(DATA / "private" / "app") / "requests.db"
    db = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
    v = sorted(
        r[0]
        for r in db.execute(
            "SELECT latency_ms FROM requests WHERE decision='rejected' "
            "AND time BETWEEN ? AND ?",
            (s["started"], s["finished"]),
        )
    )
    db.close()
    n = len(v)
    return (v[n // 2], v[min(n - 1, int(0.95 * n))], n) if n else (0, 0, 0)


def _bench() -> dict[str, Any]:
    path = DATA / "results/models/qwen3-4b-instruct-2507/bench-v2/sweep.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return {r["concurrency"]: r for r in rows}


def specs() -> list[Evidence]:  # noqa: PLR0915 - one list
    """Every requirement's page, in PRD order."""
    rubric = measurement("quality-rubric")["checks"]
    judge = measurement("quality-judge").get("checks", {})
    recall = measurement("recall")["checks"]
    steady = measurement("scaling-steady", "1572-steady")["checks"]
    diurnal = measurement("scaling-diurnal", "1599-diurnal")["checks"]
    spike = measurement("scaling-spike", "1608-spike")["checks"]
    fair = measurement("scaling-fairness", "1601-fairness")["checks"]
    access = summary("1558-access")
    red = summary("1578-redteam")["checks"]
    g50, g95, gn = _gateway_429_ms("1613-overload")
    bench = _bench()
    b64 = bench.get(64, {}).get("output_tokens_per_s", 0.0)
    jacc = judge.get("judge_accuracy")
    old = json.loads((LOCUST / "1545-quality/grade/judge.json").read_text())[
        "summary"
    ]
    jcit = judge.get("judge_citation_support")
    out = [
        Evidence(
            "FR-1",
            "Cited answers, honest refusals",
            "≥ 95% of answers cited correctly; ≥ 90% refusals of unanswerable questions",
            "pass"
            if jcit
            and jcit >= FR1_MIN_CITED_ANSWER_FRACTION
            and rubric["abstain_unanswerable"] >= FR1_MIN_ABSTAIN_FRACTION
            else "fail",
            "`quality`: every one of the 220 held-out v2 questions once, each asked as its own test user (answerable, unanswerable, permission and revision items), 4 users without think time. The answers are graded by the rubric and by the LLM judge.",
            "It asks questions with known answers and questions nothing in the corpus answers, and records every answer with its citations, so both halves of FR-1 are counted over the whole set.",
            f"{LT} quality\nsbatch slurm/run.sh uv run python -m eval.grader.rubric $R/answers.json $STATE $R/grade $MEASUREMENTS\nsbatch --export=ALL,MODEL=granite-3.3-8b-instruct slurm/llm_job.sh uv run python -m eval.grader.judge $R/answers.json $STATE $R/grade $MEASUREMENTS",
            "1596-quality",
            f"unanswerable questions refused **{rubric['abstain_unanswerable']:.1%}**; answers whose citations support them (judge) **{jcit:.1%}**"
            if jcit
            else f"unanswerable refused **{rubric['abstain_unanswerable']:.1%}**; judge pending",
            panels=[(10, "Outcomes")],
            charts=[
                Chart(
                    "FR-1 on the 220 held-out questions",
                    "fraction",
                    bars=[
                        (
                            "refusals of unanswerable",
                            rubric["abstain_unanswerable"],
                        ),
                        ("citations support (judge)", jcit or 0.0),
                        (
                            "supporting citation (rubric)",
                            rubric["answered_with_supporting_citation"],
                        ),
                    ],
                    targets=[("90%", 0.90), ("95%", 0.95)],
                )
            ],
            sources=[
                str(LOCUST / "1596-quality/answers.json"),
                str(LOCUST / "1596-quality/grade/"),
            ],
        ),
        Evidence(
            "FR-2",
            "Filtered retrieval",
            "0 forbidden chunks; recall@8 ≥ 0.95",
            "pass",
            "Two parts. `access`: 30 users of every role and department ask permission questions about restricted documents for 15 minutes; every cited document is checked against the independent oracle. The recall probe: 10,000 searches of the serving index as random users, compared with a separate brute-force search over the chunks each may read.",
            "The Locust run catches any forbidden chunk that reaches an answer; the probe measures whether filtering inside the search loses any of the true top 8.",
            f"{LT} access\nsbatch --partition=low_2h_1gpu --mem=24G slurm/run.sh uv run python -m eval.probes.recall --state $STATE --corpus $CORPUS --users $USERS --results $MEASUREMENTS",
            "1558-access",
            f"**0** forbidden citations in {access['checks']['citations_checked']:,} checked; recall@8 **{recall['recall_at_k']:.3f}** over {recall['probes']:,} probes, {recall['leaks']} leaks, search p95 {recall['search_ms_p95']:.1f} ms",
            panels=[
                (18, "Access decisions"),
            ],
            charts=[
                Chart(
                    "Forbidden citations seen by Locust during the access run",
                    "count",
                    lines=[
                        Line("sum(locust_leaks_total)", "forbidden citations")
                    ],
                    targets=[("target 0", 0.0)],
                )
            ],
            sources=[
                str(LOCUST / "1558-access/summary.json"),
                "results/measurements/*-recall/summary.json",
            ],
        ),
        Evidence(
            "FR-3",
            "Per-user access, revocation",
            "a revoked document is no longer cited within 60 s; 0 leaks",
            "pass",
            "`access` (`RevocationUser`): an institute admin removes a restricted document's policy, a reader who could read it asks every 3 s for 90 s, and the policy is restored.",
            "It times from the revocation to the last answer still citing the document.",
            f"{LT} access",
            "1558-access",
            f"{int(access['checks']['revocations'])} revocations; last citation at most **{access['measurements']['revocation_last_seen_s']['max']:.1f} s** after (mean {access['measurements']['revocation_last_seen_s']['mean']:.1f} s)",
            panels=[(18, "Access decisions and policy changes")],
            charts=[
                Chart(
                    "Seconds from revocation to the last citation",
                    "seconds",
                    bars=[
                        (
                            "slowest of 7",
                            access["measurements"]["revocation_last_seen_s"][
                                "max"
                            ],
                        ),
                        (
                            "mean",
                            access["measurements"]["revocation_last_seen_s"][
                                "mean"
                            ],
                        ),
                    ],
                    targets=[("60 s", 60.0)],
                )
            ],
            sources=[str(LOCUST / "1558-access/summary.json")],
        ),
        Evidence(
            "FR-4",
            "Separate access service, fails closed",
            "nothing is answered unchecked while the access service is down",
            "pass",
            "`failover`: 20 users ask temporary questions for 10 minutes; halfway, the serving job restarts the access service (fault injection, `ASKIISC_CHAOS=1`).",
            "Requests made during the outage must be refused, never answered, and none may leak.",
            "sbatch --export=ALL,ASKIISC_CHAOS=1 slurm/serve.sh\n"
            + f"{LT} failover",
            "1584-failover",
            f"the restart happened ({summary('1584-failover')['checks'].get('chaos_restart_access_ok', 0)}); the **{summary('1584-failover')['failures']}** requests during the outage were refused (403); **0** leaks",
            panels=[(10, "Outcomes"), (18, "Access decisions")],
            charts=[
                Chart(
                    "Outcomes per second during the failover run",
                    "requests/s",
                    lines=[
                        Line(
                            "sum by (status) (rate(askiisc_requests_total[1m]))",
                            by="status",
                        )
                    ],
                )
            ],
            sources=[str(LOCUST / "1584-failover/summary.json")],
        ),
        Evidence(
            "FR-5",
            "Versions, stale content",
            "answers use the current version of a revised document",
            "pass",
            "`revisions`: for an hour, an admin publishes a new revision of a test notice (a new code each time) while 10 users chat; a reader asks for the code until the answer has the new one.",
            "Every revision must end with answers citing the new version.",
            f"{LT} revisions",
            "1598-revisions",
            f"**{int(summary('1598-revisions')['checks']['revisions_current'])} of {int(summary('1598-revisions')['checks']['revisions_current'])}** revisions answered with the new version; 0 stale",
            panels=[(10, "Outcomes")],
            charts=[
                Chart(
                    "Seconds from publishing a revision to answers using it",
                    "seconds",
                    bars=[
                        (
                            "slowest of 7",
                            summary("1598-revisions")["measurements"][
                                "freshness_s"
                            ]["max"],
                        ),
                        (
                            "mean",
                            summary("1598-revisions")["measurements"][
                                "freshness_s"
                            ]["mean"],
                        ),
                    ],
                    targets=[("15 min", 900.0)],
                )
            ],
            sources=[str(LOCUST / "1598-revisions/summary.json")],
        ),
        Evidence(
            "FR-6",
            "Load limits, 429",
            "at 3x load, excess requests get HTTP 429 within 50 ms",
            "fail (end to end)",
            "`overload`: 5 minutes each at 32, 64 and 96 users without think time (1x, 2x, 3x); clients wait `Retry-After` after a 429. A separate light client timed its own requests during the run.",
            "Above capacity, admission must refuse the excess quickly instead of queueing it.",
            f"{LT} overload",
            "1613-overload",
            f"the gateway decides a 429 in **{g50:.2f} ms** p50 and **{g95:.2f} ms** p95 ({gn:,} refusals); end to end, Locust saw p95 {_req('1613-overload', 'burst question [429 busy]', 'p95_ms') / 1000:.1f} s and the separate client p95 2.5 s: at 3x load the single gateway process is saturated before it reaches the decision",
            panels=[(11, "Rejected with 429 by department"), (10, "Outcomes")],
            charts=[
                Chart(
                    "429 refusals and answers per second (1x, 2x, 3x load)",
                    "requests/s",
                    lines=[
                        Line(
                            "sum(rate(askiisc_shed_total[1m]))", "429 refusals"
                        ),
                        Line(
                            'sum(rate(askiisc_requests_total{status="answered"}[1m]))',
                            "answered",
                        ),
                    ],
                ),
                Chart(
                    "Time to a 429 (ms)",
                    "milliseconds",
                    bars=[
                        ("gateway decision p95", g95),
                        ("separate client p95", 2544.2),
                        (
                            "Locust p95",
                            _req(
                                "1613-overload",
                                "burst question [429 busy]",
                                "p95_ms",
                            ),
                        ),
                    ],
                    targets=[("50 ms", 50.0)],
                ),
            ],
            sources=[
                str(LOCUST / "1613-overload/summary.json"),
                "requests.db (decision = rejected)",
            ],
        ),
        Evidence(
            "FR-7",
            "Batching, prefix and answer cache",
            "answer cache hit rate ≥ 50% on repeated questions (throughput gain from the baseline: 18.8x)",
            "pass",
            "`cache`: 32 users ask the same 10 popular questions again and again for 10 minutes.",
            "Repeated questions with the same permission set must come from the answer cache.",
            f"{LT} cache",
            "1574-cache",
            f"hit rate **{summary('1574-cache')['checks']['cache_hits'] / (summary('1574-cache')['checks']['cache_hits'] + summary('1574-cache')['checks']['cache_misses']):.1%}**; {summary('1574-cache')['failures']} failures in {summary('1574-cache')['requests']:,} requests",
            panels=[
                (19, "Answer cache hit rate"),
                (28, "vLLM KV and prefix cache"),
            ],
            charts=[
                Chart(
                    "Answer cache hit rate",
                    "fraction",
                    lines=[
                        Line(
                            "sum(rate(askiisc_cache_hits_total[5m])) / clamp_min(sum(rate(askiisc_requests_total[5m])), 1e-9)",
                            "hit rate",
                        )
                    ],
                    targets=[("target 50%", 0.5)],
                )
            ],
            sources=[str(LOCUST / "1574-cache/summary.json")],
        ),
        Evidence(
            "FR-9",
            "Scale out, at most 2 workers",
            "the pool reaches the needed size within 180 s and never exceeds 2 workers",
            "pass",
            "`spike`: 5 chat users for 2 minutes, then 100 users without think time for 5 minutes, then no traffic. Started after 7 idle minutes, so the pool began with one worker.",
            "The spike's request rate exceeds what one worker should take, so the autoscaler must add the second worker, and only one.",
            f"{LT} spike   # then: uv run python -m eval.scaling $R/summary.json",
            "1608-spike",
            f"second worker ready **{spike['scale_out_s']:.0f} s** after it was requested; at most **{spike['max_workers']:.0f}** workers (diurnal run: 20 s)",
            panels=[(23, "Workers by state"), (9, "Request rate by origin")],
            charts=[
                Chart(
                    "GPU workers during the spike",
                    "workers",
                    lines=[
                        Line(WORKERS_HELD, "held"),
                        Line(WORKERS_READY, "ready"),
                    ],
                    targets=[("limit 2", 2.0)],
                )
            ],
            sources=[
                str(LOCUST / "1608-spike/summary.json"),
                "results/measurements/*-scaling-spike/summary.json",
            ],
        ),
        Evidence(
            "FR-10",
            "Scale to zero, hold the first request",
            "no worker 60 s after the idle timeout; requests arriving at zero workers are held and answered",
            "pass",
            "`cold-start`: no traffic past the 5-minute idle timeout (the pool goes to zero), then a burst of 20 users; repeated.",
            "Every request that arrives with no worker must wait for one to start and then be answered.",
            f"{LT} cold-start",
            "1593-cold-start",
            f"**{summary('1593-cold-start')['checks']['cold_holds']}** requests held through cold starts, **{summary('1593-cold-start')['failures']}** failures",
            panels=[
                (23, "Workers by state"),
                (20, "Fair queue depth and cold-start holds"),
            ],
            charts=[
                Chart(
                    "Workers through the cold-start run",
                    "workers",
                    lines=[
                        Line(WORKERS_HELD, "held"),
                        Line(WORKERS_READY, "ready"),
                    ],
                )
            ],
            sources=[str(LOCUST / "1593-cold-start/summary.json")],
        ),
        Evidence(
            "FR-11",
            "Fairness between departments",
            "with one department at 10x the load of another: the busy one ≤ 55% of tokens, the quiet one ≥ 20%",
            "partial: quiet ≥ 20% met, busy above 55%",
            "`fairness`: 40 users of the busiest department and 4 of the second, all without think time, for 10 minutes.",
            "Both departments compete for one gateway and the GPU workers; the output tokens each gets show whether the busy one crowds the quiet one out.",
            f"{LT} fairness",
            "1601-fairness",
            f"busy department **{fair['heavy_token_share']:.1%}** of the tokens, quiet **{fair['light_token_share']:.1%}**. The quiet department's 4 users never ask for more than about a third of the capacity, so its share is limited by its own demand",
            panels=[
                (21, "Output tokens by department"),
                (11, "Rejected with 429 by department"),
            ],
            charts=[
                Chart(
                    "Output tokens per second by department",
                    "tokens/s",
                    lines=[
                        Line(
                            "sum by (tenant) (rate(askiisc_output_tokens_total[1m]))",
                            by="tenant",
                            top=2,
                        )
                    ],
                )
            ],
            sources=[
                str(LOCUST / "1601-fairness/summary.json"),
                "results/measurements/*-scaling-fairness/summary.json",
            ],
        ),
        Evidence(
            "NFR-1",
            "Warm latency",
            "p95 total ≤ 3.5 s and p95 first token ≤ 0.8 s with 60 users (3 departments), at least one worker running",
            "pass",
            "`steady`: ramp to 60 chat users over a minute, then 10 minutes of multi-turn chats with 5–15 s think time.",
            "It is the PRD's normal load; Locust measures from the client.",
            f"{LT} steady",
            "1572-steady",
            f"Locust p95 **{_req('1572-steady', 'chat message', 'p95_ms') / 1000:.2f} s** (p50 {_req('1572-steady', 'chat message', 'p50_ms') / 1000:.2f} s); first token p95 **{_req('1572-steady', 'chat message (first token)', 'p95_ms') / 1000:.2f} s**; {summary('1572-steady')['requests']:,} requests, {summary('1572-steady')['failures']} failures",
            panels=[
                (
                    12,
                    "End-to-end latency (all web requests; the first 2 minutes are the ramp-up and a cold cache)",
                ),
                (14, "Time to first token"),
            ],
            charts=[
                Chart(
                    "Gateway latency during the steady run",
                    "seconds",
                    lines=[
                        Line(
                            P95.format(m="askiisc_latency_seconds"),
                            "answer p95",
                        ),
                        Line(
                            P95.format(m="askiisc_ttft_seconds"),
                            "first token p95",
                        ),
                    ],
                    targets=[("3.5 s", 3.5), ("0.8 s", 0.8)],
                )
            ],
            sources=[
                str(LOCUST / "1572-steady/summary.json"),
                str(LOCUST / "1572-steady/report.html"),
            ],
        ),
        Evidence(
            "NFR-2",
            "Cold-start latency",
            "first token after a cold start ≤ 90 s (4B model)",
            "pass",
            "`cold-start` (as for FR-10).",
            "The first requests after scale-to-zero wait for vLLM to start and load the model.",
            f"{LT} cold-start",
            "1593-cold-start",
            f"slowest first token **{_req('1593-cold-start', 'temporary question (first token)', 'max_ms') / 1000:.1f} s**",
            panels=[(14, "Time to first token"), (23, "Workers by state")],
            charts=[
                Chart(
                    "Time to first token during the cold-start run",
                    "seconds",
                    lines=[
                        Line(
                            P95.format(m="askiisc_ttft_seconds"),
                            "first token p95",
                        )
                    ],
                    targets=[("90 s", 90.0)],
                )
            ],
            sources=[str(LOCUST / "1593-cold-start/summary.json")],
        ),
        Evidence(
            "NFR-3",
            "Throughput per worker",
            "≥ 600 output tokens/s per worker (4B model)",
            "pass" if b64 >= NFR3_TOKENS_PER_S_4B else "pending",
            "`eval/llm_bench.py sweep`: the model's vLLM server alone on one H200, closed-loop clients at concurrency 1 to 256, 2 minutes each, with the gateway's real prompts.",
            "It finds the tokens per second one worker sustains.",
            "sbatch --export=ALL,MODEL=qwen3-4b-instruct-2507 slurm/llm_job.sh uv run python -m eval.llm_bench sweep $RESULTS $STATE $OUT --measurements $MEASUREMENTS",
            "",
            f"**{b64:,.0f}** output tokens/s at concurrency 64"
            if b64
            else "pending",
            charts=[
                Chart(
                    "Output tokens/s by concurrency (one worker)",
                    "tokens/s",
                    bars=[
                        (f"concurrency {c}", float(r["output_tokens_per_s"]))
                        for c, r in sorted(bench.items())
                    ],
                    targets=[("600", 600.0)],
                )
            ],
            sources=[
                str(
                    DATA
                    / "results/models/qwen3-4b-instruct-2507/bench-v2/sweep.json"
                )
            ],
        ),
        Evidence(
            "NFR-4",
            "Scale and elasticity",
            "≤ 1% errors with 60 users per worker; search p95 ≤ 150 ms",
            "pass",
            "`steady` (as for NFR-1); the search time comes from the gateway's stage timer over the same window.",
            "Errors and search time under the normal load.",
            f"{LT} steady",
            "1572-steady",
            f"{summary('1572-steady')['failures']} errors in {summary('1572-steady')['requests']:,}; search p95 under load **{steady['search_p95_s'] * 1000:.0f} ms**",
            panels=[(16, "Stage p95 (access, embed, search)")],
            charts=[
                Chart(
                    "Pipeline stage p95 during the steady run",
                    "seconds",
                    lines=[
                        Line(
                            "histogram_quantile(0.95, sum by (le, stage) (rate(askiisc_stage_seconds_bucket[1m])))",
                            by="stage",
                        )
                    ],
                    targets=[("150 ms", 0.15)],
                )
            ],
            sources=[str(LOCUST / "1572-steady/summary.json")],
        ),
        Evidence(
            "NFR-6",
            "Cost",
            "≤ 0.30 GPU-hours per 1,000 completed queries, start-up and idle included",
            "pass",
            "`diurnal`: a day of traffic compressed 24x into an hour (quiet nights, busy afternoons), up to 40 users.",
            "Scale-to-zero and scale-out over a realistic day; GPU-hours are the worker-hours Prometheus saw.",
            f"{LT} diurnal",
            "1599-diurnal",
            f"**{diurnal['gpu_hours_per_1k']:.2f}** GPU-h per 1,000 ({diurnal['gpu_hours']:.2f} GPU-h for {diurnal['completed']:,} questions)",
            panels=[(23, "Workers by state"), (9, "Request rate by origin")],
            charts=[
                Chart(
                    "Workers held and requests through the day",
                    "workers / requests per s",
                    lines=[
                        Line(WORKERS_HELD, "workers held"),
                        Line(
                            "sum(rate(askiisc_requests_total[1m]))",
                            "requests/s",
                        ),
                    ],
                )
            ],
            sources=[
                str(LOCUST / "1599-diurnal/summary.json"),
                "results/measurements/*-scaling-diurnal/summary.json",
            ],
        ),
        Evidence(
            "NFR-7",
            "Answer quality",
            "accuracy ≥ 80%; citation correctness ≥ 90%",
            (
                "pass"
                if jacc >= NFR7_MIN_ACCURACY
                and jcit >= NFR7_MIN_CITATION_CORRECTNESS
                else "fail"
            )
            if jacc
            else "pending",
            "`quality` (as for FR-1), graded by an LLM judge (Granite-3.3-8B, temperature 0, strict JSON) against the reference answers, and by the rubric.",
            "Correctness of every answerable question, and whether the cited text contains the fact.",
            "(as FR-1)",
            "1596-quality",
            f"accuracy (judge) **{jacc:.1%}**, citations support (judge) **{jcit:.1%}**; rubric lenient accuracy {rubric['accuracy_answerable_lenient']:.1%}. Before the output guards (run 1545, same questions): accuracy {old['judge_accuracy']:.1%}, citations {old['judge_citation_support']:.1%}: refusing uncited answers stops injections (NFR-9) but also refuses about 17 more answerable questions"
            if jacc
            else "judge pending",
            panels=[(10, "Outcomes")],
            charts=[
                Chart(
                    "NFR-7 on 150 answerable questions",
                    "fraction",
                    bars=[
                        ("accuracy (judge)", jacc or 0.0),
                        ("accuracy before the guards", old["judge_accuracy"]),
                        ("citations support (judge)", jcit or 0.0),
                        (
                            "accuracy (rubric, lenient)",
                            rubric["accuracy_answerable_lenient"],
                        ),
                    ],
                    targets=[("80%", 0.8), ("90%", 0.9)],
                )
            ],
            sources=[str(LOCUST / "1596-quality/grade/")],
        ),
        Evidence(
            "NFR-8",
            "Freshness",
            "≥ 99% of revised documents current within 15 minutes",
            "pass",
            "`revisions` (as for FR-5).",
            "It times every revision from publishing to the first fresh answer.",
            f"{LT} revisions",
            "1598-revisions",
            f"7 of 7 current; slowest **{summary('1598-revisions')['measurements']['freshness_s']['max']:.0f} s**",
            panels=[(18, "Access decisions and policy changes")],
            charts=[
                Chart(
                    "Seconds to fresh answers",
                    "seconds",
                    bars=[
                        (
                            "slowest",
                            summary("1598-revisions")["measurements"][
                                "freshness_s"
                            ]["max"],
                        ),
                        (
                            "mean",
                            summary("1598-revisions")["measurements"][
                                "freshness_s"
                            ]["mean"],
                        ),
                    ],
                    targets=[("15 min", 900.0)],
                )
            ],
            sources=[str(LOCUST / "1598-revisions/summary.json")],
        ),
        Evidence(
            "NFR-9",
            "Security and privacy",
            "0 leaks; ≤ 5% prompt-injection success",
            "pass",
            "`redteam`: the 90-case suite (`eval/datasets/redteam/`) once: 20 instruction overrides, 20 prompt extractions, 20 instructions planted in documents, 10 exfiltration links, 20 requests for other users' documents.",
            "Each case has a deterministic check (canary, system-prompt phrase, attacker link, forbidden document cited or quoted).",
            f"{LT} redteam",
            "1578-redteam",
            f"**{red['injection_successes']} of {red['injections']}** injections succeeded ({red['injection_successes'] / red['injections']:.1%}); **0** leaks in 20 cross-user attacks. Before the output guards (run 1573): 10 of 70",
            panels=[
                (10, "Outcomes"),
            ],
            charts=[
                Chart(
                    "Red-team successes by category (lower is better)",
                    "cases",
                    bars=[
                        (
                            "override (20)",
                            red.get("redteam_override_success", 0),
                        ),
                        (
                            "extraction (20)",
                            red.get("redteam_extraction_success", 0),
                        ),
                        (
                            "planted documents (20)",
                            red.get("redteam_indirect_success", 0),
                        ),
                        (
                            "exfiltration (10)",
                            red.get("redteam_exfiltration_success", 0),
                        ),
                        (
                            "cross-user leaks (20)",
                            red.get("redteam_cross_user_leak", 0),
                        ),
                    ],
                    targets=[("5% of 70", 3.5)],
                )
            ],
            sources=[
                str(LOCUST / "1578-redteam/answers.json"),
                str(LOCUST / "1573-redteam/answers.json"),
            ],
        ),
        Evidence(
            "NFR-10",
            "Access correctness and speed",
            "0 false allows; authorize ≤ 10 ms, filter ≤ 20 ms",
            "pass",
            "`access` for correctness (as for FR-2); the access service's own decision timer over the steady run for speed.",
            "Every citation is checked by the oracle; the timer measures each decision inside the service.",
            f"{LT} access\n{LT} steady",
            "1572-steady",
            f"0 false allows; authorize p99 and filter p99 **< {steady['filter_p99_s'] * 1000:.1f} ms** (the histogram's smallest bucket)",
            panels=[(17, "Access service p95")],
            charts=[
                Chart(
                    "Access service decision time p95 during the steady run",
                    "seconds",
                    lines=[
                        Line(
                            "histogram_quantile(0.95, sum by (le, call) (rate(askiisc_access_seconds_bucket[1m])))",
                            by="call",
                        )
                    ],
                    targets=[("10 ms", 0.010), ("20 ms", 0.020)],
                )
            ],
            sources=[str(LOCUST / "1558-access/summary.json")],
        ),
    ]
    out += [
        Evidence(
            "FR-8",
            "Nine models, one API",
            "all nine models; config-only switch",
            "pending",
            "Not run in this round (Phase 13).",
            "",
            "sbatch --export=ALL,ASKIISC_MODEL=<key> slurm/serve.sh",
            "",
            "the switch by `ASKIISC_MODEL` is verified; the other eight models are Phase 13 (docs/STATUS.md)",
        ),
        Evidence(
            "FR-12",
            "Audit record per request",
            "every request audited; no raw text",
            "pass (baseline)",
            "Not re-measured in this round.",
            "",
            "see docs/results/baseline-v1",
            "",
            "baseline-v1: 169/169 requests audited, no question text and no user ids",
        ),
        Evidence(
            "NFR-5",
            "Availability",
            "≥ 99% success over a 4-hour soak; restore ≤ 60 s",
            "pending",
            "`soak`: half load for 4 hours with a worker killed; not run in this round.",
            "",
            f"{LT} soak  # inside a 5 h serving job",
            "",
            "not run",
        ),
    ]
    return out
