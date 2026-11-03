# NFR-4: scale and elasticity

**Requirement (PRD).**
- Scale from 1 to 2 nodes within 180 s, and never above 2.
- Back to 0 within the idle timeout + 60 s.
- One node handles 60 concurrent users with at most 1% errors.
- Growing from 60 to 6,000 users, and from 40k to 1M chunks, raises warm p95 by at most 20%, with search p95 at most 150 ms.

**Verdict: elasticity passes; the per-node user target and search latency fail; growth not run.**

| Part | Target | Result | Where |
|---|---|---|---|
| 1 to 2 workers | within 180 s, at most 2 | **74 s, max 2** | `../../functional/FR-09/` |
| Back to 0 | within idle timeout + 60 s | 61 s after the last arrival (timeout 60 s); after cold starts, about 31 s after the last arrival (timeout 20 s) | FR-09, FR-10 |
| 60 users on one node, at most 1% errors | at most 1% | **fail:** 34 errors in 323 (10.5%), p95 292 s on the real index | `../NFR-01/` |
| Search p95 | at most 150 ms | **fail:** about 1,087 ms (Milvus Lite FLAT, 539k chunks, full permission list) | `../NFR-01/` diagnosis |
| Growth to 6,000 users and 1M chunks (W4, E6) | | not run (deferred; the index already fails at 539k) | |

**What it takes to pass:** Milvus standalone with an HNSW index, plus a compact permission filter. Then rerun NFR-1 and E6. The GPU node itself is not the limit: one worker sustains about 58 requests/s at p95 3.0 s (`../NFR-03/`).
