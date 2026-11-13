# v3: MVP close-out (2026-11-05 onward)

One folder per requirement once it is measured. This page says what is
done, what is built and waiting for a GPU run, and the exact commands.

## State on 2026-11-06

| Phase | Requirement | Code | Measured |
|---|---|---|---|
| 1 | FR-12 audit | done (branch `phase-14/fr12-audit`) | **pass**, run 2099/2100 ([FR-12](FR-12/README.md)) |
| 2 | FR-6 admission | done (`phase-15/fr6-admission`) | not yet: needs the serving card |
| 3 | FR-5 freshness, FR-3 revocations | done (`phase-16/fr5-freshness`) | not yet |
| 4 | FR-8 models, KV room | log parser done (`phase-17/fr8-kv`); Falcon3 weights fetched (job 2108) | not yet: no card had room for a smoke test |
| 5, 7 | FR-7, E4 switches | done (`phase-18/e4-switches`) | not yet |
| 9 | NFR-7 adjudication | done (`phase-19/nfr7-adjudication`) | review sheet made; waiting for people |

The branches are stacked in this order; each builds on the one before.

## What blocks the runs

Every run below needs the serving ports and the live databases, which the
standing serving job holds. It has to be stopped for a run and started
again after it (same settings: `ASKIISC_MODEL=qwen3-8b-awq`,
`ASKIISC_MAX_WORKERS=1`, `ASKIISC_RENEW=1`, `ASKIISC_RENEW_TIME=24:00:00`,
partition `low_unl_1gpu`, 24 h). A second stack next to it would share the
chat database and its backups, so it is not an option.

On 2026-11-06 the cards were also full: GPU 0 had about 21 GB free
(another user's jobs at 100%), GPUs 1 and 2 had none
(`nvidia-smi`, 02:00). Qwen3-8B in BF16 needs about 16 GB for weights
alone, so the FR-6 runs may have to use Qwen3-8B-AWQ; the evidence page
will say which model ran and how much memory was free.

## Commands, in order

`results/` below is `/data4/muthangiv/askiisc/results/`.

FR-6 profile (about 0.8 GPU-h):

```bash
sbatch --partition=low_4h_2gpu --time=00:50:00 \
  --export=ALL,ASKIISC_MAX_WORKERS=1,ASKIISC_MODEL=qwen3-8b-awq,ASKIISC_PROFILE=1 slurm/serve.sh
sbatch --time=00:20:00 slurm/loadtest.sh overload --duration-scale 0.4
```

FR-6 measurement, once with `ASKIISC_ADMISSION=on` and once with `off`
(about 2.2 GPU-h each; three W2 repetitions per arm):

```bash
S=$(sbatch --parsable --partition=low_4h_2gpu --time=02:30:00 \
  --export=ALL,ASKIISC_MAX_WORKERS=1,ASKIISC_MODEL=qwen3-8b-awq,ASKIISC_ADMISSION=on slurm/serve.sh)
L1=$(sbatch --parsable --time=00:45:00 slurm/loadtest.sh w2)
L2=$(sbatch --parsable --dependency=afterany:$L1 --time=00:45:00 slurm/loadtest.sh w2)
sbatch --dependency=afterany:$L2 --time=00:45:00 slurm/loadtest.sh w2
# per repetition:
python -m eval.fr6_report results/serve/$S results/locust/<job>-w2 results/measurements/fr6-<job>
```

FR-5 and FR-3 (one serving job, about 2.5 GPU-h):

```bash
sbatch --time=00:40:00 slurm/loadtest.sh freshness
sbatch --time=01:10:00 slurm/loadtest.sh revocations
python -m eval.freshness_check results/locust/<job>-freshness results/serve/<serve job> \
  /data4/muthangiv/askiisc/state-v4/versions.db results/measurements/fr5-<job>
```

FR-8 smoke tests for the replacement family (needs a card with 25% free):

```bash
sbatch --export=ALL,MODEL=tiiuae/Falcon3-3B-Instruct slurm/vllm_smoke.sh
sbatch --export=ALL,MODEL=tiiuae/Falcon3-7B-Instruct slurm/vllm_smoke.sh
sbatch --export=ALL,MODEL=tiiuae/Falcon3-7B-Instruct-AWQ slurm/vllm_smoke.sh
```

The three models join `llm/models.yaml`, with `FR8_MODEL_COUNT` and the
registry test set to 9 in the same commit, only once all three answer
correctly (decided 2026-11-05). Superseded on 2026-11-12: FR-8 asks for
six models, two families per size class.

## Found without a GPU run

- **The in-flight cap that binds is `max_in_flight` (24), not
  `llm_concurrency` (64).** In run 2099 vLLM reported room for 26.07
  requests of 16,384 tokens (427,184 KV tokens), but never ran more than
  18 at once, with at most 17.5% of its KV cache in use
  (`/data4/muthangiv/askiisc/results/serve/2099/vllm-1.log`). Admission
  stops at 24 per gateway process, so the card was never full.
- **Admission already ran before the access call, and embedding and
  reranking already ran in the embed server process**, not on the gateway
  loop (`deploy/serve.py`). What ran before admission for a chat question
  was the session's account checks, the chat history, its access check
  (the source of the 30 HTTP 503s in run 2090) and a token round trip.
  `gateway/early.py` now admits before all of that.

## NFR-7 review sheet

`/data4/muthangiv/askiisc/results/adjudication/qwen3-8b-p2/review.csv`
(job 2115): the 111 items where the rubric and the Granite judge disagree
on the 400-question Qwen3-8B run, plus 60 items sampled with seed 252, 171
rows. The `human_correct` and `human_citations` columns are empty. Once
people fill them in:

```bash
python -m eval.grader.adjudicate score \
  /data4/muthangiv/askiisc/results/versioned/v4/qwen3-8b-p2/grade/graded.json \
  /data4/muthangiv/askiisc/results/adjudication/qwen3-8b-p2/review.csv \
  /data4/muthangiv/askiisc/results/adjudication/qwen3-8b-p2
```
