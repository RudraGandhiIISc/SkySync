# CLAUDE.md: Ask IISc

Multi-tenant retrieval-augmented question answering over versioned IISc documents, on a shared departmental GPU cluster of **at most 2 GPU nodes**. Team SkySync, DS252 Cloud Computing. The full requirements are in `PRD.md`; how it works is in `docs/ARCHITECTURE.md` and the current state in `docs/STATUS.md`. Read the PRD row for any FR or NFR before changing code that serves it.

Everything is developed, tested and run on the CSA `automation` server (Slurm, one node with 3x H200 NVL, no Docker, no Kubernetes, no sudo). How each PRD component runs there is in `docs/OPERATIONS.md`; read it before touching `deploy/`, `slurm/`, `autoscaler/pool.py` or anything that starts a process. A PRD "GPU node" is a **GPU worker** here: one `vllm serve` process pinned to one H200.

## Commands

Nothing but editing, `git`, `sbatch`, `squeue` and `scancel` runs on the login node. Every `make` target runs inside Slurm through `slurm/gate.sh`, submitted from the repo root:

| Task | Command |
|---|---|
| Install (both venvs) | `sbatch slurm/setup_env.sh` |
| Format / check format | `sbatch --wait slurm/gate.sh fmt` / `... gate.sh fmt-check` |
| Lint | `sbatch --wait slurm/gate.sh lint` (ruff) |
| Types | `sbatch --wait slurm/gate.sh type` (mypy --strict) |
| Unit tests | `sbatch --wait slurm/gate.sh test` |
| Integration tests | `sbatch --wait slurm/gate.sh test-int` (real Redis binary, real HTTP services, Milvus Lite; started by `tests/integration/conftest.py`) |
| End-to-end tests | `sbatch --wait slurm/gate.sh test-e2e` (full stack with the fake LLM) |
| Performance tests | `sbatch --wait --time=02:00:00 slurm/gate.sh test-perf` |
| Security tests | `sbatch --wait slurm/gate.sh test-sec` (bandit, pip-audit, gitleaks, Garak, access probes, socket audit) |
| Regression tests | `sbatch --wait slurm/gate.sh test-regress` |
| Everything | `sbatch --wait slurm/gate.sh all` |
| Model weights | `sbatch slurm/fetch_models.sh <hf repo id>` |
| vLLM on one GPU | `sbatch slurm/vllm_smoke.sh` |
| Serve (chat, admin, monitoring) | `sbatch slurm/serve.sh` (run folder `/data4/muthangiv/askiisc/results/serve/<job>/`) |
| Corpus and index | `slurm/crawl.sh`, `archive.sh`, `catalog.sh`, `ingest.sh` (docs/OPERATIONS.md section 4) |
| Service binaries | `sbatch slurm/fetch_services.sh` (Redis, gitleaks) |

`dev` jobs wait while any user's GPU-partition job runs; when that happens and a GPU is free, submit short CPU jobs with `--partition=low_2h_1gpu` (approved by the account owner). Output lands in `logs/<job>_<id>.out`. Run the `all` gate before reporting any task as done, and report the real output from that log. Never say tests pass without running them. `dev` jobs wait while any GPU-partition job runs (see `docs/OPERATIONS.md`); if a gate is pending, say so instead of running it outside Slurm.

## Layout

```
src/askiisc/
  common/         config, types, errors, clock, logging, targets (NFR constants)
  ingest/         crawl, archive, boilerplate, catalog, editions, extract, chunk,
                  corpus, versions, reconciler, pipeline
  access/         policy, directory, service (FastAPI), client
  retrieval/      embedder, embed_server, rerank, store (Milvus), dense (exact in-memory
                  search), keyword (BM25), temporal (time scope), search
  llm/            client (OpenAI-compatible), prompts, rag, fake_server
  gateway/        app, auth, limits, drr, cache, audit, output_filter
  autoscaler/     policy, controller, pool (ProcessPool, FakeWorkerPool, SlurmPool)
  observability/  metrics, tracing, cost, gpu (NVML sampler, stands in for DCGM)
  web/            chat app and admin dashboard: accounts, chats, request log, static pages
eval/             datasets, locust, versioned, heldout, grader, oracle, probes
deploy/           serve.py (serving job), ingest.py (batch ingest), stack.py (shared
                  helpers), logs.py, grafana/ (dashboards, alerts)
slurm/            lib.sh (live corpus and index), gate.sh, run.sh, setup_env.sh,
                  fetch_models.sh, fetch_services.sh, crawl.sh, archive.sh, catalog.sh,
                  ingest.sh, serve.sh, loadtest.sh, llm_job.sh, vllm_smoke.sh, ui_preview.sh
docs/             all documentation (README.md links each file)
tests/            unit, integration, e2e, perf, security, regression, conformance
```

## Code rules

- Python 3.11. Google Python Style Guide: 80 columns, double quotes, Google-style docstrings (Args, Returns, Raises) on every public function, full type hints, no wildcard imports, no mutable default arguments.
- Functions about 40 lines at most, one responsibility each. Small modules.
- No global state. Inject `Clock`, `Random`, HTTP clients and database handles. Tests must never call real time or `sleep`.
- Configuration comes from one typed settings object (environment variables). No magic numbers: every PRD threshold is a named constant in `common/targets.py`.
- Raise typed exceptions from `common/errors.py`. Never swallow exceptions.
- Do not add a dependency without saying why and asking first.

## Safety rules (never break these)

1. **Access control fails closed.** Any error, timeout or unreachable access service means deny. There is no fallback that allows.
2. **Filter inside the search.** Permission and version filters go in the Milvus search expression. Never post-filter results.
3. **User identity comes only from the verified session** (password sign-in, or the gateway's signed token). Departments and roles come from the directory, never from the client.
4. **Never log or store raw query text, answer text or secrets.** The audit log stores queries and users only as salted SHA-256 hashes; the request log names the user (admins see who uses the service) but holds no query or answer text.
5. **Retrieved chunks are untrusted data.** The system prompt must say so, and the prompt prefix stays identical across requests so vLLM prefix caching works. Per-tenant separation uses the cache salt parameter, not prompt text.
6. **Superseded document versions are never deleted.** They are retired by setting `valid_to`, and any answer that cites one carries a stale flag.
7. **Never exceed 2 GPU workers.** The autoscaler's ceiling is a config value defaulting to 2, which is also the server's per-user GPU limit.
8. **Every service binds to `127.0.0.1`** and dies with its Slurm job. Internal services take a free port; the four people reach through tunnels have fixed, overridable ports (web 18000, Grafana 13000, Prometheus 19090, Locust 18089). Every user of the server can reach a loopback port, so the vLLM workers and the embedding server require the job's key (`common/servicekey.py`). No service is left running between jobs.

## Testing rules

- Write the test first or together with the code. A change without a test is not done.
- Tests are hermetic: no internet, no GPU. Use the fake LLM server (`llm/fake_server.py`), Milvus Lite, in-memory fakes and `FakeClock`. Unit tests never start a service binary; integration tests start the real binaries (redis-server from `slurm/fetch_services.sh`) on random ports in `tests/integration/conftest.py`.
- Coverage: at least 90% overall; 100% on `access/`, `gateway/limits.py`, `gateway/drr.py` and `autoscaler/policy.py`.
- Every bug fix adds a regression test that failed before the fix.
- Do not edit a test, threshold or golden file to make a failure pass. If a target cannot be met, stop and report it.
- The access oracle (`eval/oracle/`) is a deliberately simple, independent reference. Do not share code between it and `access/policy.py`.

## Key targets (full table in the PRD)

| Item | Target |
|---|---|
| FR-6 | At 3x load, excess requests get HTTP 429 within 50 ms |
| FR-9 / FR-10 | Pool reaches needed size in 180 s, never above 2 nodes; no active node 60 s after idle timeout; held requests complete, or end at once with "GPUs full" and a retry time when no card has room |
| FR-11 | With both tenants backlogged, one at 10x load gets at most 55% of tokens; the other at least 20% or all it asks for |
| NFR-1 | Warm p95 total at most 4.0 s (client); p95 first token at most 800 ms (model server) |
| NFR-2 | Cold-start first token at most 90 s (4B, 4-bit 8B), 150 s (8B) |
| NFR-6 | At most 0.30 GPU-hours per 1,000 completed queries, including start-up and idle |
| NFR-7 | Accuracy at least 80%, citation correctness at least 90% |
| NFR-9 / NFR-10 | 0 leaks; at most 5% prompt-injection success; 0 false allows; authorize at most 10 ms, filter at most 20 ms |

## Git rules

- `main` stays green. Work on branches named `phase-N/short-topic`.
- Conventional Commits: `feat(access): add policy store`, `test(gateway): cover 429 path`, `fix(ingest): handle missing last-modified`. Name the FR or NFR in the commit body.
- One logical change per commit and one phase per pull request. Tag finished phases (`phase-4-done`); tag the single-model baseline `baseline-v1`.
- Pre-commit hooks (ruff, mypy, gitleaks) must pass. Never use `--no-verify`.

## Server rules (from `docs/OPERATIONS.md`)

- Pin GPUs yourself: Slurm reserves none (`--gres` is rejected). Use `pick_gpu` in `slurm/lib.sh` or the same logic in `ProcessPool`, and size `--gpu-memory-utilization` from the card's free memory.
- Always set `--time` and `--mem`. CPU-only work goes to `dev`, GPU work to a GPU partition; CPU services a GPU run needs start inside that GPU job.
- Quiet periods inside a run stay under 30 minutes (the idle-GPU and idle-CPU monitors kill jobs).
- Storage: the repo on `/data4/muthangiv/AskIISc/ask-iisc`; the venvs and the live SQLite databases on `/data2/scratch/muthangiv/askiisc` (local disk, 50 GB shared quota; SQLite on the NFS drive is about 30x slower); weights, run state and raw results on `/data4/muthangiv` (`HF_HOME=/data4/muthangiv/huggingface`).
- Runs are offline (`HF_HUB_OFFLINE=1`). Download weights beforehand with `slurm/fetch_models.sh`.
- Ask before submitting any job longer than a smoke test or a gate.

## How to work in this repo

1. For any non-trivial task, first list the files you will change, the function signatures, the tests you will write, and anything ambiguous. Wait for approval.
2. Tests first, then code, then `make all`, then commit.
3. If a requirement is unclear or two documents disagree (for example GPU memory, or the node limit), ask. Do not invent behaviour.
4. Stay in scope: one phase at a time, no refactors of unrelated code.
5. The serving job defaults to `qwen3-8b` (`slurm/serve.sh`; the only model that meets NFR-7); the baseline variant was `Qwen3-4B-Instruct-2507`. The active model is chosen only by the `ASKIISC_MODEL` environment variable; switching models must never need a code change (FR-8).
