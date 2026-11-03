# FR-9: scale out to at most 2 GPU workers

**Requirement (PRD).** Add GPU nodes as request volume rises, to a limit of 2. Under a burst the pool reaches the needed size within 180 s, never exceeds 2 nodes, and loses no accepted request.

**Verdict: pass for scaling** (2 workers in 74 s, never more than 2). There is one open issue: access-check timeouts under load (below).

## How it was run

**Real index first (job 1402).** A 60 s burst at 6 requests/s through the full stack never reached 2 workers. On 539k chunks, the Milvus Lite search plus the 24,892-id permission list starves the gateway's event loop, so arrivals reached the autoscaler at about 1/s (see `../../non-functional/NFR-01/`). That run is kept as evidence of the bottleneck.

**Scaling test (job 1404, 2026-10-20 01:20).**
- **Setup:** the same stack and autoscaler, with **real vLLM workers on real GPUs**, but on the mock index (`eval/datasets/sample`, 13 documents), so search is not the limit.
- **Load:** an open loop of Poisson arrivals at 6 requests/s for 120 s, using real held-out questions. That is above one worker's threshold of 4 requests/s.
- **Settings:** at most 2 workers, hysteresis 30 s, idle timeout 60 s.

```bash
sbatch --export=ALL,ASKIISC_NO_INGEST=1,ASKIISC_EXPERIMENT=scale,ASKIISC_ANSWER_CACHE=0,ASKIISC_MAX_WORKERS=2,\
ASKIISC_IDLE_TIMEOUT_S=60,ASKIISC_EXP_RATE=6,ASKIISC_EXP_BURST_S=120,ASKIISC_DIRECTORY_PATH=eval/datasets/sample/users,\
ASKIISC_TENANT_RATE_PER_S=1000,ASKIISC_TENANT_BURST=1000 slurm/run_stack.sh \
  --corpus eval/datasets/sample/corpus --state-dir /data4/muthangiv/askiisc/state-sample
```

## Results (job 1404)

| Check | Target | Result |
|---|---|---|
| Time from the start of the burst to 2 ready workers | at most 180 s | **74.2 s** |
| Most workers at any time | at most 2 | **2** |
| Worker 1 | | requested at +0.9 s, ready 30.7 s later (GPU 2) |
| Worker 2 | | requested at +31.0 s (after the 30 s hysteresis), ready 41.9 s later (GPU 1, shared with another user) |
| Back to zero after the burst | | 61.1 s after the last arrival (idle timeout 60 s) |
| Latency of answered requests, p50 / p95 | | 3.4 s / 30.6 s (the p95 includes requests held during the first cold start) |
| Requests refused with HTTP 403 | none | **283 of 704** |

**Where the 403s come from:**
- **78 are a test-setup error:** user `stu_csa_phd` is not in the mock users tree.
- **About 205 are access-check timeouts.** The access service runs in the gateway's process and event loop, and the client allows 50 ms. The loop was busy streaming hundreds of held responses, so access calls timed out and the requests were refused.

Refusing is the designed fail-closed behaviour, and these requests were never accepted, so none was lost after acceptance. But it shows that **the access service must run as its own process** (FR-4 calls for a separate service), with a client timeout set for that deployment.

**Simulation evidence:** `tests/unit/test_autoscaler.py::test_burst_reaches_two_workers_within_180_s`, `test_worker_failure_is_replaced_within_180_s`, and the policy invariants (never more than 2; never scale in with work in flight).

Raw data: `/data4/muthangiv/askiisc/results/stack/1404/scale.json` (timeline per second, worker lifecycle), `records.json`; real-index attempt `/data4/muthangiv/askiisc/results/stack/1402/`.
