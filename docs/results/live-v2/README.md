# Live results v2 (2026-10-26)

Qwen3-4B-Instruct-2507 on up to two H200 workers, the v2 corpus (26,240 documents, 568,817 chunks) and held-out set v2 (220 questions). Serving jobs 1565–1587 on `med_12h_2gpu`; Locust runs from `slurm/loadtest.sh` (headless, `low_2h_1gpu`). Raw results: `/data4/muthangiv/askiisc/results/locust/<job>-<workload>/` and `results/measurements/`. The admin **Requirements** tab shows the same numbers. Each requirement links to its evidence page: the workload and why it tests the requirement, the command, the numbers and source files, a chart with the target marked (from the stored Prometheus data) and the Grafana panels for exactly that run's window, made by `tools/evidence.py` without re-running anything.

| Req | Target | Result | Verdict | Run |
|---|---|---|---|---|
| [FR-1](FR-1/README.md) | ≥ 95% of answers cited correctly; ≥ 90% refusals of unanswerable | refusals 96.7%; citation support: **97.5%** of answers' citations support them (judge) | pass | quality 1596 |
| [FR-2](FR-2/README.md) | 0 forbidden chunks; recall@8 ≥ 0.95 | 0 leaks in 10,621 requests; recall@8 **1.000** over 10,000 probes of the serving index | pass | access 1558, probe 1539 |
| [FR-3](FR-3/README.md) | revoked within 60 s | 7 revocations, last cited at most **6.8 s** after | pass | access 1558 |
| [FR-4](FR-4/README.md) | fails closed | access service restarted mid-run: the 14 requests during the outage were refused (403), 0 leaks | pass | failover 1584 |
| [FR-5](FR-5/README.md) | answers use the current version | 7 of 7 revisions answered with the new version | pass | revisions 1598 |
| [FR-6](FR-6/README.md) | 429 within 50 ms at 3x load | the gateway decides a 429 in **0.45 ms** p95 (16,025 refusals); end to end a separate client waited p50 1.0 s, p95 **2.5 s**: one gateway process is saturated at 3x load | fail (end to end) | overload 1613 |
| [FR-7](FR-7/README.md) | cache hit rate | **98.8%** (8,951 of 9,061), 0 failures in 9,235 | pass | cache 1574 |
| [FR-8](FR-8/README.md) | nine models, config switch | config switch only; the other eight models are Phase 13 | pending | — |
| [FR-9](FR-9/README.md) | 1 → 2 workers within 180 s, never above 2 | second worker ready **20 s** after it was requested; never above 2 | pass | diurnal 1599 (Prometheus) |
| [FR-10](FR-10/README.md) | held requests complete after scale to zero | **47** requests held through cold starts, 0 failures | pass | cold-start 1593 |
| [FR-11](FR-11/README.md) | at 10:1 load, busy ≤ 55% of tokens, quiet ≥ 20% | busy department **67.2%** of output tokens, quiet **32.8%**; the quiet department's 4 users never ask for more than about a third of the capacity | partial (quiet ≥ 20% met; busy above 55%) | fairness 1601 |
| [FR-12](FR-12/README.md) | an audit record per request, no text | not re-measured (baseline: 169/169, no text) | pass (baseline) | — |
| [NFR-1](NFR-1/README.md) | p95 ≤ 3.5 s, first token p95 ≤ 0.8 s, 60 users | p95 **1.6 s**, first token p95 **0.35 s** (p50 0.49 s); 7,780 requests | pass | steady 1572 |
| [NFR-2](NFR-2/README.md) | cold first token ≤ 90 s | slowest **27.9 s** | pass | cold-start 1593 |
| [NFR-3](NFR-3/README.md) | ≥ 600 output tokens/s per worker | **2,660** output tokens/s at concurrency 64 on one worker (2,958 at 256) | pass | throughput 1621 |
| [NFR-4](NFR-4/README.md) | ≤ 1% errors at 60 users; search p95 ≤ 150 ms | 0 errors in 7,780; search p95 under load **62 ms** | pass | steady 1572 (+ Prometheus) |
| [NFR-5](NFR-5/README.md) | availability (4-hour soak) | not run | pending | — |
| [NFR-6](NFR-6/README.md) | ≤ 0.30 GPU-h per 1,000 queries, start-up and idle included | **0.19** (1.10 GPU-h, 5,798 questions in a day of traffic at 24x) | pass | diurnal 1599 (Prometheus) |
| [NFR-7](NFR-7/README.md) | accuracy ≥ 80%, citations ≥ 90% | accuracy (judge) **64.7%**, citations **97.5%**. Before the output guards, same questions: 74.2% and 95.5% | fail | quality 1596, judge 1620 (1622 before the guards) |
| [NFR-8](NFR-8/README.md) | ≥ 99% current within 15 min | 7 of 7, slowest 604 s | pass | revisions 1598 |
| [NFR-9](NFR-9/README.md) | 0 leaks; injection success ≤ 5% | 90-case suite: **1 of 69** injections (1.4%), 0 leaks in 20 cross-user attacks | pass | redteam 1578 |
| [NFR-10](NFR-10/README.md) | 0 false allows; authorize ≤ 10 ms, filter ≤ 20 ms | 0 false allows; access-service p99 **< 0.5 ms** for both (its own histogram) | pass | access 1558, steady 1572 |

## What changed to get here

- **NFR-1, NFR-4 (latency).** Two server problems, not the model: the SQLite files on the NFS drive (122 ms per commit, behind one lock) and unbounded BLAS and torch threads on a node where Slurm pins no CPUs. Before: p95 50 s, first token 37 s, 1,691 requests in 10 minutes. After: p95 1.6 s, first token 0.35 s, 7,780 requests.
- **NFR-9 (prompt injection).** Run 1573 found 10 of 70 injections succeeding, mostly "reply only with X" in the question. Output guards (uncited answers become the refusal, system-prompt echoes refused, foreign links and markup removed) and rule 6 in the system prompt brought it to 1 of 69.
- **FR-6, FR-11 (overload, fairness).** A fair-share cap of 24 requests in flight per server, shared by the departments with requests in flight, and a server-wide admission rate; Locust users now honour `Retry-After`.

## Notes

- One gateway process serves about 11 questions per second. Above that, its event loop is saturated: the access check times out (fails closed, 403) and refusals slow down. Running several gateway processes would need shared admission state; that is future work.
- Only 7 revisions fitted in the 1-hour revisions run (each waits until it is answered fresh), not 50; all were current within 15 minutes.
- NFR-7 uses the LLM judge (Granite-3.3-8B). The held-out set v2 is not yet reviewed by a person.
