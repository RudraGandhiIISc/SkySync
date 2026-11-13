# FR-8: six models, one API (2026-11-03)

Each model was served by itself with `ASKIISC_MODEL=<key>` and nothing else
changed (FR-8: a configuration switch, no code change). These are the
numbers on the admin **Models** tab; each model's
`/data4/muthangiv/askiisc/results/models/<key>/metrics.json` names its
sources.

| Model | Size | Accuracy | Citations | p95, vLLM alone at concurrency 64 (s) | Output tokens/s at 64 | Cold first token (s) | GPU-h per 1,000 |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 4B | 0.70 | 0.96 | 3.0 | 5,195 | 25.6 | 0.12 |
| Phi-4-mini-instruct | 4B | 0.41 | 0.92 | 19.9 | 958 | 50.1 | 0.21 |
| Qwen3-8B | 8B | 0.81 | 0.97 | 40.0 | 540 | 57.5 | 0.21 |
| Granite-3.3-8B-Instruct | 8B | 0.67 | 0.97 | 52.5 | 162 | 45.1 | 0.24 |
| Qwen3-8B-AWQ | 4-bit 8B | **0.82** | 0.97 | 51.1 | 447 | 51.3 | 0.31 |
| Granite-3.3-8B-Instruct-GPTQ | 4-bit 8B | 0.64 | 0.96 | 58.5 | 129 | 57.5 | 0.29 |

Which targets apply to this table:

- NFR-7 (accuracy at least 0.80, citations at least 0.90) judges the
  selected model, Qwen3-8B; the others are reported.
- NFR-2: cold first token at most 90 s (4B, 4-bit 8B) or 150 s (8B).
- The p95 column is not NFR-1. It is vLLM alone at concurrency 64; NFR-1
  is measured through the whole system with 60 users
  ([STATUS](../../STATUS.md)).
- NFR-3 (tokens/s at least 600 for 4B, 250 for 8B, 450 for 4-bit 8B) and
  NFR-6 (at most 0.30 GPU-hours per 1,000 queries) judge the Qwen3 model of
  each size on a card no other job uses. The shared-card numbers here are
  reported only; the quiet-card ones are below.

## How each number was measured

| Column | Source |
|---|---|
| Accuracy, citations | the Granite judge on the `quality` run (the held-out questions through the whole system) |
| p95, tokens/s | `slurm/fr8_bench.sh`: vLLM alone, gateway-identical prompts, closed loop at concurrency 64 |
| Cold first token | the slowest first token of the `cold-start` run (a question held while a worker starts) |
| GPU-hours per 1,000 | the `diurnal` run (a day compressed into an hour, scale to zero included) |

Qwen3-4B's row is the baseline-v1 round (2026-10-20/24), measured on an
idle card.

## Read with care

- **Shared GPU.** The five new runs shared GPU 0 with another user's job
  running at 100% (the other two cards were full). Their p95 and tokens/s
  are far below what an idle H200 gives and cannot be compared with
  Qwen3-4B's row. Accuracy and citations do not depend on it; GPU-hours
  per 1,000 do, since a slower card answers fewer questions in the hours
  it is held.
- **Locust sign-ins.** 14 of 60 Locust sign-ins per run failed (a stale
  credentials file, fixed afterwards), so those load tests ran with about
  a quarter fewer users than planned.
- **Granite 4-bit.** IBM publishes no 4-bit Granite-3.3-8B. The first
  community build (drawais, AWQ) produced gibberish on long RAG prompts
  under vLLM 0.11 (every quality answer became a refusal), so a GPTQ build
  (TevunahAi) replaced it.

## On a quiet card (2026-11-12)

Job 2238 measured the two Qwen3 8B models again on a card no other job
was using (`slurm/fr8_bench.sh`, the same prompts), and their KV-cache room
at one fixed budget of 24 GiB (`--gpu-memory-utilization 0.17`):

| Model | Tokens/s at concurrency 1 | Tokens/s at 64 | NFR-3 target | KV tokens at 24 GiB |
|---|---|---|---|---|
| Qwen3-8B | 144 | 1,549 | 250 | 19,984 (twice) |
| Qwen3-8B-AWQ | 30 | 881 | 450 | 65,920 and 89,584 |

Both meet NFR-3, and batching serves 10.8x and 29x the tokens of one
request at a time (FR-7). The 4-bit model holds at least 3.3x its parent's
KV tokens (FR-8); the Granite pair, measured in job 2234, 3.25x (114,144
against 35,088). The 4-bit Qwen is the slower of the two for a single
request on vLLM 0.11, so it is the cheaper choice for memory, not for one
user's latency. Raw files: `/data4/muthangiv/askiisc/results/models/<key>/bench-2238/`,
`/data4/muthangiv/askiisc/results/v3/2238/kvq/`.

## Reading

Only Qwen3-8B and Qwen3-8B-AWQ meet the accuracy target. The 4-bit Qwen
matches its parent in accuracy with about a third of the weight memory
(5.7 against 15.3 GiB), so it leaves more KV cache for concurrent users.

## Commands

```bash
sbatch slurm/fr8_bench.sh <key> ...                 # throughput and start-up
sbatch slurm/fr8_stack.sh <key> ...                 # whole system, one model after another
python -m askiisc.web.models record <key> --accuracy ... --source ...   # inside slurm/run.sh
```
