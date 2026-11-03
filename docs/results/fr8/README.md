# FR-8: six models, one API (2026-11-03)

Each model was served by itself with `ASKIISC_MODEL=<key>` and nothing else
changed (FR-8: a configuration switch, no code change). These are the
numbers on the admin **Models** tab; each model's
`/data4/muthangiv/askiisc/results/models/<key>/metrics.json` names its
sources.

| Model | Size | Accuracy | Citations | p95 at 64 (s) | Output tokens/s | Cold first token (s) | GPU-h per 1,000 |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 4B | 0.70 | 0.96 | 3.0 | 5,195 | 25.6 | 0.12 |
| Phi-4-mini-instruct | 4B | 0.41 | 0.92 | 19.9 | 958 | 50.1 | 0.21 |
| Qwen3-8B | 8B | 0.81 | 0.97 | 40.0 | 540 | 57.5 | 0.21 |
| Granite-3.3-8B-Instruct | 8B | 0.67 | 0.97 | 52.5 | 162 | 45.1 | 0.24 |
| Qwen3-8B-AWQ | 4-bit 8B | **0.82** | 0.97 | 51.1 | 447 | 51.3 | 0.31 |
| Granite-3.3-8B-Instruct-GPTQ | 4-bit 8B | 0.64 | 0.96 | 58.5 | 129 | 57.5 | 0.29 |

Targets: accuracy at least 0.80 and citations at least 0.90 (NFR-7); p95
at most 3.5 s (NFR-1); tokens/s at least 600 (4B), 250 (8B), 450 (4-bit
8B) (NFR-3); cold first token at most 90 s (4B, 4-bit 8B) or 150 s (8B)
(NFR-2); at most 0.30 GPU-hours per 1,000 queries (NFR-6).

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
  Qwen3-4B's row. Accuracy, citations and GPU-hours do not depend on it.
- **Locust sign-ins.** 14 of 60 Locust sign-ins per run failed (a stale
  credentials file, fixed afterwards), so those load tests ran with about
  a quarter fewer users than planned.
- **Granite 4-bit.** IBM publishes no 4-bit Granite-3.3-8B. The first
  community build (drawais, AWQ) produced gibberish on long RAG prompts
  under vLLM 0.11 (every quality answer became a refusal), so a GPTQ build
  (TevunahAi) replaced it.

## Reading

Only Qwen3-8B and Qwen3-8B-AWQ meet the accuracy target. The 4-bit Qwen
matches its parent in accuracy with about half the weight memory, so it
leaves more KV cache for concurrent users.

## Commands

```bash
sbatch slurm/fr8_bench.sh <key> ...                 # throughput and start-up
sbatch slurm/fr8_stack.sh <key> ...                 # whole system, one model after another
python -m askiisc.web.models record <key> --accuracy ... --source ...   # inside slurm/run.sh
```
