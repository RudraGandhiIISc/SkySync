# NFR-1: warm latency

**Requirement (PRD).** p95 total time at most 3.5 s and p95 time-to-first-token at most 800 ms. Workload: normal load W1 (60 users, 3+ tenants, think time Exp(10 s)) with at least one worker running. Measured client-side over a 10-minute steady period.

**Verdict: fail.** The cause is the search index on this server, not the model.

## How it was run

**Run (job 1393, 2026-10-20 00:00):** the full stack on the real index (24,889 documents, 539,351 chunks). One Qwen3-4B worker ran on GPU 2.

**Workload** (`deploy/experiments.py`, `load`):
- 60 closed-loop users over 4 tenants (cds, csa, ece, accounts), think time Exp(10 s).
- Questions drawn with Zipf(1.1) popularity from the 98 answerable held-out questions.
- A 60 s ramp, then 600 s measured. The answer cache was on, as W1 specifies.
- Tenant limits were raised, so that FR-6 shedding would not mix in.

```bash
sbatch --export=ALL,ASKIISC_NO_INGEST=1,ASKIISC_TENANT_RATE_PER_S=1000,ASKIISC_TENANT_BURST=1000,\
ASKIISC_EXPERIMENT=load,ASKIISC_EXP_USERS=60,ASKIISC_EXP_THINK_S=10,ASKIISC_EXP_RAMP_S=60,ASKIISC_EXP_STEADY_S=600 \
  slurm/run_stack.sh --corpus /data4/muthangiv/askiisc/corpus --state-dir /data4/muthangiv/askiisc/state
```

## Results

| Metric | Target | Result |
|---|---|---|
| p95 total | at most 3.5 s | **292 s** |
| p95 time to first token | at most 800 ms | **292 s** |
| Completed requests per second | about 4–6 offered | **0.32** |
| Errors | | 34 (HTTP 403) |
| vLLM output tokens per second | | 29.5 (the GPU was mostly idle) |
| Prefix-cache hit rate | | 44% |

## Diagnosis (`tools/diag_latency.py`, job after 1393)

| Stage | p50 |
|---|---|
| Access policy filter (after the fix below) | 1.7 ms |
| **Milvus Lite search, 539k chunks, 24,892-id permission list in the expression** | **822 ms** |
| Milvus Lite search, validity filter only | 214 ms |

- **Search caps throughput.** Milvus Lite searches by brute force (FLAT index only) and serves one search at a time, so the whole system tops out near 1 request per second. W1 offers about 6, so the queue grows for the entire run, and every latency above is mostly queueing.
- **Where the 403s come from:** the access client's 50 ms timeout, on a gateway event loop saturated by the queue. Requests then fail closed, as designed.
- **What NFR-3 shows:** the GPU node alone (Qwen3-4B on vLLM) handles far more; see `../NFR-03/`.

**Fixed before this run:** the first W1 run (job 1384) had 97 403 errors, because the policy filter evaluated all 24,892 policies per request and the load generator issued a new token per request. Now plain public documents are precomputed once per policy version, and clients reuse their token.

**Next fixes, not done:**
- **Milvus standalone with an HNSW index** (the PRD design), or an in-process approximate index with filtering during search.
- **A compact permission filter:** "public, plus these restricted ids", instead of a 24,892-id list per query.

Raw data: `/data4/muthangiv/askiisc/results/stack/1393/` (`summary.json`, `records.json`), first run `/data4/muthangiv/askiisc/results/stack/1384/`.
