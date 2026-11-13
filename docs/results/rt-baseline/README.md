# Evidence for the red-team baseline

Compiled by `tools/baseline_results.py` from raw result files only.
Jobs 2234 and 2238 ran the code of `main` at 74401a5;
the later jobs ran the release code (service keys, watchdog,
full-GPU handling). Every run used one GPU worker, except NFR-1 of
the release code (up to two, as the serving job does), on cards that
other users' jobs could also use. Qwen3-8B-AWQ served the FR-5 and
FR-3 runs; Qwen3-8B served the rest.

## FR-5: freshness (E5) and FR-3: revocation (E1) (code at 74401a5, before the FR-5 fixes, job 2234)

| Check | Target | Measured |
|---|---|---|
| Revisions replayed | 60 across 20 documents | 60 across 20 |
| Time to demotion (s), p50 / p95 / max | | 11.0 / 37.7 / 178.4 (n=60) |
| Answers using a superseded version without the flag (audit store) | 0 | 2903 of 2924 that used one |
| Answers citing a superseded version without the flag | 0 | 1 of 3118 answers |
| As-of lookups correct at the checkpoints | all | 60 of 60 |

| Check | Target | Measured |
|---|---|---|
| Scripted revocations | 100 | 77 |
| Last citation after revoking (s), p99 / max | at most 60 | 8.9 / 8.9 |
| Revocations still cited after 60 s | 0 | 0 |

Raw files: `/data4/muthangiv/askiisc/results/v3/2234/fresh/`, `/data4/muthangiv/askiisc/results/v3/2234/revocations/`.

## FR-5: freshness replay on the release code

| Check | Target | Measured |
|---|---|---|
| Revisions replayed | 60 across 20 documents | 60 across 20 |
| Time to demotion (s), p50 / p95 / max | at most 900 | 7.6 / 34.3 / 63.9 (n=60) |
| Answers using a superseded version without the flag (audit store) | 0 | 0 of 0 that used one |
| Answers citing a superseded version without the flag | 0 | 0 of 2992 answers |
| As-of lookups correct at the checkpoints | all | 60 of 60 |

Verdict: pass. Raw files: `/data4/muthangiv/askiisc/results/v3/2259/fresh5/`.

## NFR-1 and NFR-4: Qwen3-8B, `steady`, 60 users (code at 74401a5, job 2234)

| Check | Target | Median of 3 (each run) | Verdict |
|---|---|---|---|
| p95 total (client) | at most 4,000 ms | 6,400 ms (6,500, 6,400, 6,400) | fail |
| p95 first token at the model server | at most 800 ms | 985 ms (944, 1,028, 985) | fail |
| p95 first token at the client (reported) | | 2,600 ms (2,500, 2,600, 2,600) | |
| Errors (NFR-4) | at most 1% | 0.05% (0.05, 0.02, 0.13) | pass |
| Questions turned away with 429 (admission; reported) | | 14.23% (14.41, 14.23, 13.52) | |

Raw files: `/data4/muthangiv/askiisc/results/v3/2234/nfr1/`.

## NFR-1 and NFR-4: Qwen3-8B, `steady`, 60 users (release code, job 2314)

| Check | Target | Median of 3 (each run) | Verdict |
|---|---|---|---|
| p95 total (client) | at most 4,000 ms | 3,900 ms (4,900, 3,900, 3,900) | pass |
| p95 first token at the model server | at most 800 ms | 457 ms (480, 457, 436) | pass |
| p95 first token at the client (reported) | | 2,000 ms (2,400, 2,000, 1,900) | |
| Errors (NFR-4) | at most 1% | 0.03% (0.03, 0.03, 0.01) | pass |
| Questions turned away with 429 (admission; reported) | | 2.52% (6.57, 2.20, 2.52) | |

Raw files: `/data4/muthangiv/askiisc/results/v3/2314/nfr1/`.

## NFR-2 and FR-10: 20 scripted cold starts per size (E3 c) (code at 74401a5, job 2234)

| Model | Cold starts | Answered | Acquire (s), median | Weight load (s), median / max | First token (s), median / max | Target | Verdict |
|---|---|---|---|---|---|---|---|
| qwen3-4b-instruct-2507 | 20 | 20 | 0.00 | 24.1 / 56.1 | 25.3 / 57.4 | at most 90 s | pass |
| qwen3-8b | 20 | 9 | 0.00 | 40.1 / 42.2 | 41.1 / 43.5 | at most 150 s | pass |
| qwen3-8b-awq | 20 | 19 | 0.00 | 28.1 / 28.9 | 29.8 / 31.4 | at most 90 s | pass |

Acquisition is starting a vLLM process on a pinned card (R-1: no
node allocation on this server). Raw files: `/data4/muthangiv/askiisc/results/v3/2234/cold-<model>/`.

## NFR-2 and FR-10: 20 scripted cold starts per size (E3 c) (release code, job 2259)

| Model | Cold starts | Answered | Acquire (s), median | Weight load (s), median / max | First token (s), median / max | Target | Verdict |
|---|---|---|---|---|---|---|---|
| qwen3-8b | 20 | 20 | 0.00 | 26.1 / 26.2 | 27.1 / 27.9 | at most 150 s | pass |

Acquisition is starting a vLLM process on a pinned card (R-1: no
node allocation on this server). Raw files: `/data4/muthangiv/askiisc/results/v3/2259/cold-<model>/`.

## FR-8: KV-cache room at one 24 GiB budget

Target: at the same GPU memory budget (`--gpu-memory-utilization
0.17`, 24 GiB), each 4-bit model holds at least 2x the KV tokens of
its 8B parent. The ratio below is the worst case: the 4-bit model's
smallest run against the parent's largest.

| Pair | Weights GiB (4-bit / parent) | KV tokens (4-bit / parent) | Worst ratio | Verdict |
|---|---|---|---|---|
| Qwen3-8B-AWQ against Qwen3-8B | 5.71 / 15.27 | 65,920, 89,584 / 19,984, 19,984 | 3.30x | pass |
| Granite-3.3-8B-Instruct-GPTQ against granite-3.3-8b-instruct | 4.32 / 15.25 | 114,144 / 35,088 | 3.25x | pass |

The Qwen pair of job 2234 is not used: both ran on a card where
another user's job grew while vLLM measured its memory, and vLLM
counts that growth against the KV budget (Qwen3-8B-AWQ got 2.53 GiB
there, 9.05 and 12.30 GiB on a quiet card in job 2238).

Raw files: `/data4/muthangiv/askiisc/results/v3/2234/kv24/`, `/data4/muthangiv/askiisc/results/v3/2238/kvq/` (`vllm.log`).

## NFR-3 and FR-7: throughput on one card (E0)

vLLM alone, closed loop with the gateway's prompts. NFR-3 reads
concurrency 64; FR-7 compares it with one request at a time.

| Model | Tokens/s at 1 | Tokens/s at 64 | NFR-3 target | NFR-3 | FR-7 gain (64 against 1) | FR-7 |
|---|---|---|---|---|---|---|
| qwen3-8b | 144 | 1,549 | at least 250 | pass | 10.8x | pass |
| qwen3-8b-awq | 30 | 881 | at least 450 | pass | 29.1x | pass |

Cards at the start of the job (another user's job can still start
on the same card during the sweep):

```
index, utilization.gpu [%], memory.free [MiB], memory.total [MiB]
0, 100 %, 1329 MiB, 143771 MiB
1, 100 %, 88783 MiB, 143771 MiB
2, 0 %, 138718 MiB, 143771 MiB
```

Raw files: `/data4/muthangiv/askiisc/results/models/<model>/bench-2238/sweep.json`.

## Service keys and the watchdog (live stack, release code)

| Check | Expected | Measured |
|---|---|---|
| A question through the chat app | answered | answered |
| vLLM worker http://127.0.0.1:35557: /v1 and /invocations without the key; /health | 401, 401; 200 | 401, 401; 200 |
| Embedding server: /embed and /rerank without the key; /health | 401, 401; 200 | 401, 401; 200 |
| Kill the embed service (SIGKILL): new process, healthy, question answered | restarted | 2.6 s, 4.6 s, 0.0 s |
| Kill the access service (SIGKILL): new process, healthy, question answered | restarted | 1.6 s, 1.0 s, 0.1 s |

Verdict: pass. Raw file: `/data4/muthangiv/askiisc/results/v3/2274/faults/report/fault_check.json`.

## FR-11: fairness with both departments backlogged

`fairness` workload: 80 users of the busy department and 8 of
the quiet one, all asking back to back (10:1); 6 LLM slots and the
answer cache off, so both departments always have more requests
waiting than their fair share.

| Check | Target | Measured | Verdict |
|---|---|---|---|
| Busy department's share of tokens | at most 55% | 53.7% | pass |
| Quiet department's share of tokens | at least 20% | 46.3% | pass |

Verdict: pass. Raw files: `/data4/muthangiv/askiisc/results/v3/2314/fairness-backlogged/` (audit records and the Locust run).

## NFR-5: 1 hour at half load, a worker killed at 30 minutes

| Check | Target | Measured | Verdict |
|---|---|---|---|
| Non-error responses | at least 99% | 99.99% of 18,892 | pass |
| Worker killed | once, at 30 minutes | 1 | |
| Capacity restored after the kill | within 180 s | 26.1 s | pass |

Raw files: `/data4/muthangiv/askiisc/results/v3/2314/soak/`.
