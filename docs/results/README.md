# Results archive

Measured evidence, one folder per round. Each page records a run as it
was: its commands and links refer to the scripts and documents of that
date. Several have since been removed or merged: `slurm/run_stack.sh` and
`deploy/experiments.py` (replaced by `slurm/serve.sh` and Locust),
`eval/retrieval_bench.py` (replaced by `eval/versioned/retrieval.py`), and
`PROGRESS.md`, `NEXT_STEPS.md`, `docs/OVERVIEW.md`, `docs/SERVER.md`
(merged into [../STATUS.md](../STATUS.md),
[../ARCHITECTURE.md](../ARCHITECTURE.md) and
[../OPERATIONS.md](../OPERATIONS.md)).

| Round | Date | What |
|---|---|---|
| [baseline-v1/](baseline-v1/README.md) | 2026-10-20 | the first single-model round (Qwen3-4B), one folder per requirement |
| [retrieval-v2/](retrieval-v2/README.md) | 2026-10-26 | vector-only against hybrid search on the held-out set |
| [live-v2/](live-v2/README.md) | 2026-10-26 | every requirement measured on the serving job with Locust, with charts and Grafana panels |
| [fr8/](fr8/README.md) | 2026-11-03 | the six models side by side (FR-8) |
| [v3/](v3/FR-12/README.md) | 2026-11-05 | MVP close-out, one folder per requirement (FR-12 so far) |

The edition-aware evaluation of 2026-10-27 (Qwen3-8B, 84.5%) is in
[../STATUS.md](../STATUS.md) section 3; its raw files are in
`/data4/muthangiv/askiisc/results/versioned/v4/`.
