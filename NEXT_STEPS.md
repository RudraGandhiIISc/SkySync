# Next steps

What to do next, in order, after the overnight build of 2026-10-18. What
exists now is in [PROGRESS.md](PROGRESS.md); the full procedure is the build
guide in [README.md](README.md); server rules are in
[docs/SERVER.md](docs/SERVER.md). Commands run from
`/data2/home/muthangiv/AskIISc/ask-iisc`.

## 1. First, five minutes of checking

```bash
squeue -u $USER                                 # nothing should be left running
tail -3 logs/askiisc_gate_<last id>.out         # last gate: expect "gate ... passed"
git log --oneline | head                        # the overnight commits
python3 -m json.tool /data4/muthangiv/askiisc/results/stack/1293/results.json | less
```

If the last gate did not pass, rerun it (section 3) and read the first
`FAILED` line.

## 2. Decisions and messages only you can send

1. **Email the admin** (akshaynath@iisc.ac.in) the five questions in
   `docs/SERVER.md` section 6. The two that matter most: may a job run
   localhost HTTP services, and may short CPU jobs use `low_2h_1gpu` when `dev`
   is blocked.
2. **Team review** of the deviations in PROGRESS.md section 6, mainly MinIO
   being replaced by files and Milvus Lite being used until the standalone
   spike is done. Record what the team decides in `docs/PRD.md`.
3. **Corpus:** decide which real IISc documents to use (public pages only,
   robots.txt, at most 1 request per second), and who will check the held-out
   answers by hand.
4. **For the nine-model stage:** accept the Gemma-3 and Llama-3.1 licences on
   HuggingFace, then put a read token in `~/.cache/huggingface/token` (mode 600).
5. **Which model is the LLM judge** (Phase 9; it must differ from the models
   under test).

## 3. Everyday commands

```bash
# Gate (CPU). If dev is blocked by another user's GPU job and a GPU is free,
# add --partition=low_2h_1gpu (approved for short CPU jobs).
sbatch --wait slurm/gate.sh all                       # fmt-check lint type test audit
sbatch --wait --time=00:45:00 slurm/gate.sh test-int test-e2e test-sec

sbatch slurm/vllm_smoke.sh                            # one vLLM worker, one reply
sbatch slurm/run_stack.sh                             # whole stack, real model, 8 questions
sbatch --export=ALL,ASKIISC_MODEL=qwen3-8b slurm/run_stack.sh   # another model
sbatch slurm/fetch_models.sh <hf/repo-id> ...         # weights to /data4 (offline runs)
```

Logs are in `logs/<job>_<id>.out`. Stack results are in
`/data4/muthangiv/askiisc/results/stack/<id>/`.

## 4. Finish the remaining M1 items (about one day)

1. **Real corpus.** Write `ingest/scrape.py` plus `slurm/ingest.sh` (a CPU job)
   to fetch public IISc pages, recording ETag and Last-Modified. Add a
   `FileSource` (Reconciler `Source`) for a directory of documents.
2. **Reconciler in the stack.** Start `Reconciler.run_forever` inside
   `deploy/stack.py` against that source, and bump the gateway's cache
   `generation` on every ingested version.
3. **Held-out question set** (Phase 5.3): `eval/datasets/questions.jsonl`, at
   least 150 items, with every reference answer checked by a named person.
4. **Prometheus and Grafana**: add release tarballs to
   `slurm/fetch_services.sh`, `deploy/local/prometheus.yml` and a dashboard.
5. **Milvus standalone spike**: unpack the official 2.4 `.deb` with
   `dpkg-deb -x` into scratch, run it with embedded etcd and local storage
   inside a job, and point `ASKIISC_MILVUS_URI` at it. If it cannot run
   without root, raise the Qdrant fallback with the team. HNSW recall (FR-2)
   is only meaningful once this works.
6. **Cold start (NFR-2, 4B target 90 s).** The first start took 130 s, 70 s of
   it reading weights from NFS. Copy the active model's snapshot to
   `/data2/scratch/muthangiv/askiisc/models/` (about 8 GB; watch the 50 GB
   quota), point `HF_HOME` or the model path there for runs, and re-measure.
   The torch.compile cache in `~/.cache/vllm` already cuts later starts.

## 5. Then follow the build guide

The README prompts for Phases 9 to 13 still apply, with the server mapping in
`docs/SERVER.md` (Slurm instead of Kubernetes, ProcessPool instead of a node
group, FileObjectStore instead of MinIO). In short:

- **Phase 9, evaluation tooling:** Locust W1 to W4 (`eval/locust/`), rubric
  grader and LLM judge, 10,000 gateway probes and 100 revocations replayed
  through the oracle, 90 attack cases. Run Locust inside the stack job, against
  `127.0.0.1`.
- **Phase 10, hardening:** TLS 1.3 on the gateway (`uvicorn --ssl-*`, self-signed
  certificate in scratch), question-length and body limits, `docs/security.md`.
- **Phase 11, goldens and performance thresholds.**
- **Phase 12, baseline-v1 with Qwen3-4B:** E0 to E6 as chained Slurm jobs with
  `--dependency` (at most 2 running and 2 queued GPU jobs), with every quiet
  period under 30 minutes. Ask for approval of the GPU hours before submitting.
  Write the scorecard to `docs/results/baseline-v1/`.
- **Phase 13, the other eight models:** fetch the weights, quantise the three
  AWQ models (llm-compressor in the serve venv, a GPU job), and run
  `ASKIISC_MODEL=<key> sbatch slurm/run_stack.sh` per model to check each one
  before the full E0 to E2 runs.

## 6. Things to watch

- `dev` jobs wait while **any** user's GPU job runs. Check `squeue` and
  `nvidia-smi` before assuming a job is stuck.
- GPUs 1 and 2 were at 100% use by other users on 2026-10-18. `pick_gpu` and
  `ProcessPool` choose the least busy card and size vLLM's memory from what is
  free; they refuse to start below 25% free.
- Keep `/data2` under 50 GB. Weights, run state and results go to `/data4`.
- No service may outlive its job. `run_stack.sh` kills its children on exit;
  check with `ps -u $USER` after an interrupted run.
