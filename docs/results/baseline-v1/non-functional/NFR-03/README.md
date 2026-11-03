# NFR-3: throughput per GPU worker (E0 saturation sweep)

**Requirement (PRD).** Per node at least 600 output tokens/s for 4B models (250 for 8B, 450 for 4-bit 8B, provisional), at load that fully uses one node without breaking NFR-1. Measured by the vLLM token counter; E0 sweeps concurrency 1 to 256 to find the knee and the rated concurrency.

**Verdict: pass.** 5,195 output tokens/s at the rated concurrency of 64, where p95 is 3.03 s, within NFR-1's 3.5 s.

## How it was run

**Run (job 1396, 2026-10-20 00:25):** Qwen3-4B-Instruct-2507, BF16, vLLM 0.11, prefix caching on, an 8,192-token context window. One H200 (GPU 2), with 0.90 of its memory.

**Prompts** (`eval/llm_bench.py`): exactly what the gateway sends. That is the fixed system prompt, the real chunks the held-out run retrieved for each question (same context budget) and the question. That gives 167 prompts of about 1.4–3k tokens, with at most 256 output tokens, temperature 0.

**Why the model is measured alone:** they go straight to vLLM, because on this server the Milvus Lite search stage caps the full system near 1 request/s (`../NFR-01/`). E0 is defined as a measurement of the GPU node.

**Load:** closed loop, no think time, 90 s per step after a 20 s warm-up. Tokens/s come from the vLLM counter `vllm:generation_tokens_total`, which matches the client count exactly.

```bash
sbatch --export=ALL,GPU_EXCLUDE=0,1 slurm/llm_job.sh uv run python -m eval.llm_bench sweep \
  /data4/muthangiv/askiisc/results/stack/1376/heldout_results.json /data4/muthangiv/askiisc/state \
  /data4/muthangiv/askiisc/results/bench/nfr3 --seconds 90
```

## Results

| Concurrency | Requests/s | Output tokens/s | p95 total (s) | p95 first token (s) |
|---|---|---|---|---|
| 1 | 2.05 | 202 | 1.18 | 0.081 |
| 2 | 4.30 | 413 | 1.20 | 0.040 |
| 4 | 8.24 | 775 | 1.26 | 0.047 |
| 8 | 14.98 | 1,360 | 1.42 | 0.062 |
| 16 | 25.57 | 2,296 | 1.68 | 0.071 |
| 32 | 41.08 | 3,726 | 2.08 | 0.074 |
| **64 (rated)** | **57.62** | **5,195** | **3.03** | **0.102** |
| 128 | 67.95 | 6,116 | 5.16 | 0.133 |
| 256 | 70.03 | 6,267 | 10.11 | 0.324 |

- **Knee:** throughput grows almost linearly up to 64, then flattens (+18% from 64 to 128, +2% from 128 to 256), while p95 roughly doubles at each step.
- **Rated concurrency:** 64, the highest step with p95 within 3.5 s.

**Caveat.** The 167 prompts are reused in a loop, so the prefix-cache hit rate is 99.7% after the first pass. Real traffic shares only the system prompt and popular chunks. FR-7's run without prefix caching bounds the effect (`../../functional/FR-07/`).

Raw data: `/data4/muthangiv/askiisc/results/bench/nfr3/sweep.json`, vLLM log `results/llm/1396/vllm.log`.
