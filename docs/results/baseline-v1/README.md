# baseline-v1: single-model results (Qwen3-4B-Instruct-2507)

> **Update 2026-10-21:** the search bottleneck behind the NFR-1, NFR-4 and NFR-10 failures is fixed (search p95 840 ms → 18.5 ms). Those rows are re-measured with Locust next; see [NEXT_STEPS.md](../../../NEXT_STEPS.md) section 1. The numbers below are unchanged from the baseline run.

Every functional (FR-1 to FR-12) and non-functional (NFR-1 to NFR-10) requirement of `PRD.md`, measured on the CSA automation server on 2026-10-19/20, with one model.

Each folder has a README with:
- the requirement;
- how the experiment was run (with the command);
- the results;
- where the raw data lives on `/data4`.

## Setup

| Item | Value |
|---|---|
| Model | Qwen3-4B-Instruct-2507, BF16, vLLM 0.11.0, prefix caching on, 8,192-token context window |
| Hardware | 1 H200 NVL per worker (GPU 2; the second worker for FR-9 ran on GPU 1, which another user's job was also using) |
| Corpus | 24,889 public IISc documents from 291 sites, plus 22 restricted test documents: 539,351 chunks, BGE-base embeddings, Milvus Lite |
| Users | the users tree (`/data4/.../private/users`): 3 real users and test users, an admin per department |
| Questions | held-out set v1, 168 items (`eval/datasets/heldout/`); the 98 answerable ones are not yet reviewed by a person |

## Scorecard

| ID | Requirement | Target | Result | Verdict |
|---|---|---|---|---|
| [FR-1](functional/FR-01/) | Cited answers, honest refusals | at least 95% cited correctly, at least 90% refusals | refusals 90.0%; citations support the answer 96.2% (judge) | pass |
| [FR-2](functional/FR-02/) | Filtered retrieval | 0 forbidden chunks in 10,000 probes; recall@8 at least 0.95 | 0 forbidden: 2,000 real-index and 10,000 test-index probes; recall exact (FLAT) | pass |
| [FR-3](functional/FR-03/) | Per-user access, revocation within 60 s | match the reference checker; 0 leaks after 60 s | 220/220 matrix; 10/10 live readers; 0/10 leaks; 0 leaks after 10 s | pass |
| [FR-4](functional/FR-04/) | Separate access service, fails closed | conformance; nothing answered when it is killed | both implementations conform; all denied when killed | pass (run it as its own process) |
| [FR-5](functional/FR-05/) | Versions, stale content | as-of answers right at all checkpoints; stale flagged | 60/60 simulated; live 18/20 answers, stale flag 19/20 | pass |
| [FR-6](functional/FR-06/) | Load limits, 429 | 429 within 50 ms at 3x load | unit and e2e only | deferred |
| [FR-7](functional/FR-07/) | Continuous batching, prefix cache | 2x throughput, p99 -30%, hit rate at least 50%, TTFT -25% | 18.8x, -90%, 99.7%, -84% | pass |
| [FR-8](functional/FR-08/) | Nine models, one API | all nine via one client; config-only switch | config switch verified; a second model (Granite judge) served unchanged | partial (Phase 13) |
| [FR-9](functional/FR-09/) | Scale out to at most 2 | needed size within 180 s; never more than 2 | 2 workers in 74 s; max 2 (mock index) | pass, with access timeouts noted |
| [FR-10](functional/FR-10/) | Scale to zero, hold the first request | 0 workers within timeout + 60 s; 20/20 held requests complete | 20/20 held and answered, warming shown; zero about 11 s past the timeout | pass |
| [FR-11](functional/FR-11/) | Fairness between tenants | heavy at most 55%, light at least 20% at 10:1 | 50% / 50% (simulation) | pass (simulated) |
| [FR-12](functional/FR-12/) | Audit record per request | every request; no raw text | 169/169; 0 questions, 0 user ids | pass |
| [NFR-1](non-functional/NFR-01/) | Warm latency | p95 at most 3.5 s, p95 TTFT at most 0.8 s | p95 292 s (Milvus Lite bottleneck) | **fail** |
| [NFR-2](non-functional/NFR-02/) | Cold-start latency | at most 90 s (4B) | p50 25.6 s, max 26.1 s (warm cache; first-ever start 130 s) | pass |
| [NFR-3](non-functional/NFR-03/) | Throughput per worker | at least 600 output tokens/s (4B) | 5,195 tokens/s at the rated concurrency of 64 | pass |
| [NFR-4](non-functional/NFR-04/) | Scale and elasticity | 1 to 2 within 180 s; 60 users at most 1% errors; search p95 at most 150 ms | 74 s; 10.5% errors; search about 1.1 s | **partial fail** |
| [NFR-5](non-functional/NFR-05/) | Availability | at least 99%; restore within 180 s | simulation only | deferred |
| [NFR-6](non-functional/NFR-06/) | Cost | at most 0.30 GPU-h per 1k queries | 0.12 (burst), floor 0.005; 1.25 when the index idles the GPU | partial (24 h replay deferred) |
| [NFR-7](non-functional/NFR-07/) | Answer quality | accuracy at least 80%, citations at least 90% | accuracy 70.4% (judge), citations 96.2% | **fail** (accuracy), pending human adjudication |
| [NFR-8](non-functional/NFR-08/) | Freshness | at least 99% current within 15 min | 10/10 revisions detected and ingested | partial (long run deferred) |
| [NFR-9](non-functional/NFR-09/) | Security and privacy | 0 leaks; at most 5% injection; no raw text | 0 leaks; no raw text; red-team suite not built | partial |
| [NFR-10](non-functional/NFR-10/) | Access correctness and speed | 0 false allows; authorize at most 10 ms, filter at most 20 ms at 500/s | 0 false allows; engine filter 1.7 ms; HTTP authorize p99 318 ms, filter 34/s | **partial fail** (filter payload) |

## The two main findings

1. **The search index is the bottleneck, not the GPU.**
   - One H200 serves about 58 requests/s within the latency target (NFR-3).
   - The full system on the real index manages about 0.3–1 requests/s, because Milvus Lite searches 539k chunks by brute force, one search at a time. Each search carries a 24,892-id permission list (822 ms per search).
   - Fix: Milvus standalone with HNSW, plus a compact permission filter. That should move NFR-1, NFR-4 and NFR-6 to passing.
2. **Accuracy is limited by retrieval.**
   - When the model answers, it is right 88% of the time (judge).
   - But the question's source document is in the top 8 for only 46% of questions, and 20% of answerable questions were refused.
   - Fix: hybrid keyword and vector search, collapsing near-duplicate notices, and reranking.

## Deviations from the PRD in this round

| Deviation | Why |
|---|---|
| Milvus Lite instead of Milvus 2.4 standalone | No containers or root on this server; the standalone spike is still pending |
| Throughput and batching measured against vLLM directly | E0 is defined on one node; the index stage caps the full system |
| 8,192-token context instead of 4,096 | Dense chunks overflowed 4,096 tokens (job 1360) |
| Load generator in `deploy/experiments.py` instead of Locust | Lighter to run inside one Slurm job with the stack |
| FR-6, NFR-5, NFR-6 (24 h replay) and NFR-8 (long run) deferred | Agreed on 2026-10-19 to fit the overnight GPU budget |
| One repetition per experiment, not 3 | Same reason |
