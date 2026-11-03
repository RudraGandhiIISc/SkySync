# FR-7: continuous batching and prefix caching

**Requirement (PRD).** Use continuous batching and prefix-cache reuse so GPUs serve more users. Compared with static batching and no caching:
- throughput is at least 2x higher and p99 latency at least 30% lower;
- the cache hit rate is at least 50%, and p95 time-to-first-token is at least 25% lower.

**Verdict: pass on all four criteria**, with the two caveats below.

## How it was run

**Runs:** jobs 1397, 1398 and 1399, 2026-10-20 00:28 to 00:44. Each used the same vLLM server setup (Qwen3-4B, BF16, one H200 on GPU 2) and the same 167 gateway-identical RAG prompts as NFR-3 (`eval/llm_bench.py`).

**Load:** 32 closed-loop clients, no think time, 180 s each after a warm-up. Only one vLLM flag changes between runs:

| Variant | vLLM flags |
|---|---|
| default | continuous batching, `--enable-prefix-caching` |
| no prefix cache | `--no-enable-prefix-caching` |
| no batching | `--max-num-seqs 1` (one sequence at a time) |

```bash
sbatch --export=ALL,GPU_EXCLUDE=0,1,VLLM_EXTRA_ARGS=--no-enable-prefix-caching slurm/llm_job.sh \
  uv run python -m eval.llm_bench fixed HELDOUT_RESULTS STATE OUT --concurrency 32 --seconds 180 --label no_prefix_cache
```

## Results

| Variant | Requests/s | Output tokens/s | p95 total (s) | p99 total (s) | p95 first token (s) | Prefix-cache hit rate |
|---|---|---|---|---|---|---|
| **default** | **41.59** | **3,747** | 2.07 | **2.14** | **0.077** | 99.7% |
| no prefix cache | 14.67 | 1,316 | 6.25 | 7.27 | 0.496 | (off) |
| no batching | 2.21 | 220 | 19.69 | 20.63 | 19.34 | 70.5% |

| Criterion | Target | Result |
|---|---|---|
| Throughput, batching vs one-at-a-time | at least 2x | **18.8x** (41.59 vs 2.21 req/s) |
| p99, batching vs one-at-a-time | at least 30% lower | **90% lower** (2.14 vs 20.63 s) |
| Prefix-cache hit rate | at least 50% | **99.7%** |
| p95 first token, cache vs no cache | at least 25% lower | **84% lower** (0.077 vs 0.496 s) |

**Caveats.**
1. **No true static batching.** vLLM has no static-batching mode, so the baseline is batch size 1 (`--max-num-seqs 1`). That is a stronger contrast than static batching would give.
2. **Inflated hit rate.** The 167 prompts are reused in a loop, so the hit rate is near 100%; real traffic shares the system prompt and popular chunks only. Even so, caching alone gave 2.8x throughput.

Raw data: `/data4/muthangiv/askiisc/results/bench/fr7/fixed_{default,no_prefix_cache,no_batching}.json`.
