# NFR-10: access correctness and speed

**Requirement (PRD).** 0 false allows, at most 1% false denies, revocation within 60 s, and decisions in at most 10 ms (authorize) and 20 ms (filter), at 500 access decisions per second.

**Verdict: correctness passes; engine speed passes; the HTTP service at 500/s fails** (the filter response is too large).

## Results

| Check | Target | Result | Where |
|---|---|---|---|
| False allows | 0 | **0** in 2,000 real-index probes and 10,000 test-index probes; 0 in the 220-case matrix | FR-2, FR-3 |
| False denies (live, readers allowed) | at most 1% | **0/10** in the permission questions | FR-3 |
| Revocation | within 60 s | no leak 10 s after a revocation (cache TTL 10 s) | FR-3 |
| Policy engine filter, real policies (24,911 documents) | at most 20 ms | **p50 1.7 ms, p99 2.3 ms** | `/data4/muthangiv/askiisc/results/probes/1403//probes.json` |
| HTTP `/v1/authorize` at 500/s, client-side | at most 10 ms | achieved 470/s; **p50 49 ms, p99 318 ms**; 0 errors | `/data4/muthangiv/askiisc/results/probes/1403//latency.json` |
| HTTP `/v1/filter` at 500/s, client-side | at most 20 ms | **achieved only 34/s; p50 7.2 s** | same |

## How the HTTP numbers were measured

`eval/probes/real.py` starts the access service with uvicorn in a thread and drives it from an asyncio client in the same Python process, at a fixed 500 requests/s for 60 s per endpoint.

**Two effects inflate the numbers:**
1. **Shared interpreter.** Client and server share one interpreter, which inflates authorize times; it is a pessimistic setup.
2. **Response size.** Every `filter` response lists all of the user's readable document ids: 24,892, about 700 KB of JSON. That, not the decision (2 ms), is what fails.

**Fix:** a compact filter, "every public document, plus these restricted ids, minus these denied ones". It would shrink responses to a few hundred bytes and also remove the Milvus expression bottleneck (NFR-1). Then rerun with the service as its own process.
