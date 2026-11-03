# NFR-2: cold-start latency

**Requirement (PRD).** Time-to-first-token at most 90 s for 4B and 4-bit 8B models, and 150 s for 8B models, for the first request after all GPUs were released. Measured over 20 scripted cold starts per model size, with node start and model load reported separately.

**Verdict: pass for 4B** (Qwen3-4B). 8B and 4-bit 8B are Phase 13.

## How it was run

Same run as FR-10 (job 1400): 20 cold starts. Each time-to-first-token is client-side, from sending the question to the first token, including the gateway hold. The split comes from the worker lifecycle: `requested_at`, `acquired_at`, `ready_at`.

## Results (20 cold starts, Qwen3-4B, BF16, H200)

| Measure | Result |
|---|---|
| Time to first token, p50 / p95 / max | **25.6 / 26.0 / 26.1 s** (target at most 90 s) |
| Acquisition (pick a free GPU, start the process) | under 0.1 s. There is no node to boot, because Slurm reserves no GPU (`docs/SERVER.md`) |
| Model load until vLLM is healthy | **24.1 s** every time |
| The rest (retrieval and first-token generation) | about 1.5 s |

**Conditions matter.**
- **Warm caches:** these starts had the weights in the OS page cache and the torch.compile cache in `~/.cache/vllm`.
- **The first-ever start was slower:** on this server it took **130 s** (job 1290), with 70 s of that reading weights from NFS (`/data4`).
- **Cold-cache start:** after a reboot or cache eviction, expect around 100–130 s unless the active model is copied to local disk (`NEXT_STEPS.md`).

Raw data: `/data4/muthangiv/askiisc/results/stack/1400/coldstarts.json`; first start `/data4/muthangiv/askiisc/results/smoke/1290/`.
