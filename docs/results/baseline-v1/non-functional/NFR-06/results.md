# NFR-6 results (this round)

GPU-hours = total held time of every worker, from its request to its release, including acquisition, model load, serving and idle (`observability/cost.py`). Target: at most 0.30 GPU-hours per 1,000 completed queries.

| Run | Workers | GPU-hours | Completed queries | GPU-h per 1,000 queries |
|---|---|---|---|---|
| FR-9 burst, 6 req/s, mock index (job 1404) | 2 | 0.050 | 421 | **0.119** (pass) |
| Held-out, sequential, real index (job 1376) | 1 | 0.085 | 169 | 0.50 (one user at a time: not a load test) |
| NFR-1 W1, real index (job 1393) | 2 (sequential: a restart) | 0.360 | 289 | **1.25** (fail: the Milvus Lite bottleneck kept the GPU idle, at 29 tokens/s) |
| Steady-state floor implied by NFR-3 (rated 57.6 req/s on one worker) | 1 | | | **0.0048** |

**Reading these numbers:**
- **Cost is set by how busy the worker is.** At the rated load one H200 costs about 0.005 GPU-hours per 1,000 queries, 60x under the target.
- **On the real index the search bottleneck idles the GPU,** so held time is wasted and the figure rises to 1.25.
- **The 24-hour replay is deferred.** It is the run that charges overnight idle time and cold starts against each idle timeout.

Raw data: `workers.json` and `summary.json` / `scale.json` in `/data4/muthangiv/askiisc/results/stack/<job>/`.
