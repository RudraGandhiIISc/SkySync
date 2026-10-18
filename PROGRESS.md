# Progress report: overnight build, 2026-10-18

What was built on the CSA `automation` server in the session of 2026-10-18
(about 22:50 to 23:45 IST), how it was checked, and what is not done. Every
number below comes from a Slurm log in `logs/` or a results file under
`/data4/muthangiv/askiisc/results/`. The next actions are in
[NEXT_STEPS.md](NEXT_STEPS.md).

## 1. Headline

- **Milestone M1 is met on the fake stack, and one real run passed.** Questions go
  through gateway → access service → filtered Milvus search → LLM → output
  filter → audit, and come back as a streamed, cited answer.
- **The real Qwen3-4B run passed 8 of 8 smoke questions** (job 1293), with a cold
  start held and answered, not failed.
- **The gate is green**: format, lint (ruff), types (mypy --strict), 587 unit and
  conformance tests at **98.6% coverage** (100% on access, limits, DRR and the
  scaling policy), 5 integration tests, 6 security tests, gitleaks with no leaks
  found, pip-audit with no known vulnerabilities, and the end-to-end suite
  over real HTTP (jobs 1299 and 1300).
- **Nothing ran on the login node.** Every install, download and test ran as a
  Slurm job, all inside your folders, with no sudo.

## 2. Phase status (README build guide)

| Phase | Status | What exists |
|---|---|---|
| 0 Toolchain | Done | `pyproject.toml` (Python 3.11, uv), `Makefile`, `.pre-commit-config.yaml` (local hooks, run in a job), optional CI, `slurm/gate.sh` |
| 1 Types, fake LLM | Done | `common/` (types, errors, clock, targets, config, tokens), `llm/fake_server.py` (OpenAI SSE, vLLM metric names, prefix-cache simulation, warm-up delay) |
| 2 Ingestion (FR-5) | Done | extract (PDF, HTML, text), chunk (sentence boundaries, table rows kept, overlap), versions (nothing deleted, atomic switch), reconciler (ETag or content hash), pipeline, object store |
| 3 Access (FR-3, FR-4) | Done | policy engine (deny wins, windows, fails closed), directory export, FastAPI service, HTTP client (50 ms timeout, one retry on connect error, fails closed), conformance suite, independent oracle |
| 4 Retrieval (FR-2) | Done | embedder (HTTP plus fake), Milvus index (HNSW params, filter inside the search), exact in-memory index, retriever, CPU embedding server |
| 5 RAG (FR-1, FR-8) | Done except the question set | fixed-prefix prompt with untrusted-data chunk tags, citation parsing (bogus ones dropped, stale ones flagged), streaming client with cache salt, 9-model registry `llm/models.yaml`. Held-out question set: **not built** (needs the real corpus and a human reviewer) |
| 6 Gateway (FR-6, FR-11, FR-12) | Done | JWT auth, per-tenant token bucket and bounded queue (429 with Retry-After), deficit round robin, permission-keyed cache (memory and Redis), output filter, salted-hash audit, SSE API with warming hold, Prometheus metrics |
| 7 Autoscaler (FR-9, FR-10) | Done (process pool) | pure policy, controller, `ProcessPool` (vLLM on the least busy H200, memory share from free memory, process-group kill), `FakeWorkerPool`. `SlurmPool` not built (needs admin permission) |
| 8 Observability (NFR-6) | Mostly done | metrics, tracing (refuses text attributes), cost model (a, l, b, d; per 1,000 queries and per 1,000 correct), NVML sampler. Prometheus and Grafana binaries and dashboards: **not yet** |
| 9 to 13 | Not started | Locust workloads, grader and judge, probe and red-team suites, hardening, goldens, baseline experiments, nine-model sweep |

## 3. Requirements with evidence so far

These are checks on the fake stack or in simulation, plus one small real run.
They are not the PRD's measured results; those are Phase 12.

| Req | Evidence | Where |
|---|---|---|
| FR-1 | Citations must name a retrieved chunk; refusal without chunks skips the LLM. Real run: 5/5 answerable cited the right doc, 3/3 unanswerable or forbidden refused | `tests/unit/test_rag.py`, job 1293 |
| FR-2 | **10,000 probes, 0 forbidden chunks** (oracle replay); recall@8 ≥ 0.95 against exact search. Milvus Lite is FLAT, so this is not yet an HNSW recall figure | `tests/integration/test_fr2_probes.py` |
| FR-3 | 200-case matrix agrees for the engine, the oracle and both API implementations; no leak 10 s after a revocation (cache TTL), under the 60 s target | `tests/unit/test_access_*.py`, `tests/conformance/` |
| FR-4 | Access service killed mid-run: every later call denied; gateway answers 403 | `tests/integration/test_services.py`, `tests/unit/test_gateway.py` |
| FR-5 | 60 revisions across 20 documents: as-of answers right at all 60 checkpoints; no moment with both or neither version visible | `tests/unit/test_versions_reconciler.py` |
| FR-6 | Bucket never exceeds rate × time + burst (property test); 429 with Retry-After; admission is synchronous (no await on the LLM) | `tests/unit/test_limits_drr.py`, `test_gateway.py` |
| FR-8 | Switching `ASKIISC_MODEL` changes only the request's model field; nine models in three classes | `tests/unit/test_rag.py` |
| FR-9 / FR-10 | Simulated: 2 workers ready within 180 s of a burst, never more than 2; zero workers 60 s after the idle timeout; 20 scripted cold starts all completed (hold ≤ 37 s simulated); failed worker replaced within 180 s | `tests/unit/test_autoscaler.py` |
| FR-11 | Simulated 10:1 overload: heavy tenant ≤ 55% of tokens, light ≥ 20%; a tenant alone gets 100% | `tests/unit/test_limits_drr.py` |
| FR-12 / NFR-9 | Audit has hashes only (grep for question text and user ids finds nothing); replay with the record's as-of time gives the same chunks; no question text in any log; all listeners on 127.0.0.1 | `test_gateway.py`, `tests/security/`, job 1293 `audit.jsonl` |

## 4. Real GPU runs

| Job | What | Result |
|---|---|---|
| 1290 | `slurm/vllm_smoke.sh`: Qwen3-4B-Instruct-2507, BF16, GPU 0, memory share 0.90 | Healthy after **130 s**: weights 70 s (read from NFS `/data4`), torch.compile 16 s, engine warm-up 31 s. Over the NFR-2 target of 90 s (see NEXT_STEPS) |
| 1293 | `slurm/run_stack.sh`: whole stack, ProcessPool with 1 worker, 8 smoke questions | **8/8 HTTP 200, all answers correct.** q1 held through a cold start of 26 s (compile cache and page cache warm by then), with a `warming` event; the rest took 0.1 to 0.3 s each, one user at a time. ECE student asking about the CDS thesis and student asking about staff travel were refused (filtered inside the search). The planted "IGNORE ALL PREVIOUS INSTRUCTIONS" document still gave a normal cited answer. Worker released at the end; no processes left |

Results: `/data4/muthangiv/askiisc/results/stack/1293/` (`results.json`,
`audit.jsonl`, `workers.json`, `vllm-1.log`, `embed.log`).

The end-to-end suite on the fake stack (`tests/e2e/test_stack_fake.py`) runs
every component as a real HTTP server, and passes (job 1300). It covers a
cold-start hold, filtered retrieval, a 429 burst, a 401, scale to zero, a 403
once the access service is killed, and an audit log free of question text.

## 5. Bugs the tests found (all fixed, each with a regression test)

1. `Controller.close()` released the workers but the still-running control
   loop started a new one. `close()` now stops the loop.
2. **Scale-to-zero flapping:** after an idle scale-in, the 30 s arrival window
   still showed demand, so a worker restarted at once. From zero workers, only
   an arrival after the last action now triggers a cold start.
3. The vLLM worker's process type did not match the pool protocol (mypy), and
   one test patched `asyncio.wait_for` globally and hung. The grace period is
   now a config value and the test uses a small one.

## 6. Deviations from the PRD and the build guide

| Planned | Done instead | Why |
|---|---|---|
| Kubernetes, Docker Compose | Slurm jobs; `deploy/stack.py` supervisor; test fixtures start services | No Docker access, no Kubernetes, no sudo |
| MinIO for originals | `FileObjectStore` on `/data4` (same interface) | MinIO stopped publishing community binaries (HTTP 410); only the licensed AIStor build remains |
| Milvus 2.4 standalone | Milvus Lite (embedded, FLAT index) | Standalone without root is still a spike (Phase 12) |
| TEI embedding server | `retrieval/embed_server.py` (sentence-transformers, CPU, same `/embed` contract) | TEI needs Docker or a Rust build |
| DCGM exporter | `observability/gpu.py` (NVML at 1 Hz) | DCGM needs root |
| Access service as its own deployment | Its own HTTP server on its own port, started inside the stack job | One job holds the whole stack; it is still reached only over HTTP |
| NetworkPolicy, packet capture | Loopback-only binding plus a source check | No root |
| CPU work only in `dev` | Short CPU jobs ran on `low_2h_1gpu` while GPU 0 was idle | `dev` was blocked by another user's 24-hour GPU job; you approved this |
| Answer-time output filter strips spans | It re-checks every cited document and replaces the whole answer with a refusal if any check fails | Streamed text comes only from chunks already allowed, so only a mid-request revocation can trip it |

## 7. Jobs submitted

| Job | Partition | Purpose | Outcome |
|---|---|---|---|
| 1285 | low_2h_1gpu | `setup_env.sh`: both venvs | done (vLLM 0.11.0, torch 2.8.0+cu128, Python 3.11.16) |
| 1286, 1287, 1289, 1291, 1292, 1296 to 1300 | low_2h_1gpu | gate runs while fixing code | final runs green (see section 1) |
| 1288 | low_2h_1gpu | `fetch_models.sh`: Qwen3-4B, BGE-base | done |
| 1290 | low_2h_1gpu | vLLM smoke test | done |
| 1293 | low_2h_1gpu | first whole-stack run | done, 8/8 |
| 1294, 1295 | low_2h_1gpu | `fetch_services.sh` | 1294 failed (MinIO 410); 1295 done (Redis 7.2.5, gitleaks 8.18.4) |

Disk: `/data2` is at about 29 of 50 GB (serve venv 9.3 GB, project venv, uv
cache). `/data4/muthangiv/huggingface` holds 37 GB, including older models
from other projects.

## 8. Not done

- Held-out question set (needs the real corpus and a named human reviewer).
- A real IISc corpus. The sample corpus is synthetic and labelled as such.
- The reconciler loop inside the stack job; Milvus standalone; Prometheus and
  Grafana; `SlurmPool`.
- Phases 9 to 13: Locust workloads, grader and LLM judge, red-team suite,
  hardening pass, goldens and performance thresholds, baseline experiments
  E0 to E6, the other eight models.
- Admin questions (`docs/SERVER.md` section 6) were not sent.
