# Developer guide

What each part of the Ask IISc codebase does, how the parts fit together,
and how to test and extend them. Install and run: [README.md](README.md).
Requirements (FR-n, NFR-n): [PRD.md](PRD.md). Deeper design notes:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. The big picture

```
browser --SSH tunnel--> web app (/, /admin)          src/askiisc/web
                           | internal token
                           v
                        gateway                       src/askiisc/gateway
                           |-- access service (own process)   src/askiisc/access
                           |-- time scope, embed, hybrid search, rerank
                           |                          src/askiisc/retrieval
                           |-- vLLM worker (GPU) answer with citations
                           |                          src/askiisc/llm
                           |-- output filter + audit log
                           v
                     streamed answer (SSE) + cited sources

autoscaler   0..2 vLLM workers on free GPUs, scale to zero   src/askiisc/autoscaler
ingest       crawl -> archive -> catalogue -> chunk/embed/store, every version kept
                                                             src/askiisc/ingest
monitoring   Prometheus metrics, Grafana, Loki, NVML GPU     src/askiisc/observability
jobs         Slurm scripts that start all of the above       slurm/, deploy/
evaluation   Locust workloads, graders, oracle, probes       eval/
```

One question goes: web app → gateway `POST /v1/ask` → admission (FR-6
limits, DRR fair queue) → access service returns the documents the user may
read → retrieval searches only those documents and only the edition valid
at the question's date → reranker keeps the best 12 → vLLM answers with
`[chunk]` citations or "I don't know" → output filter re-checks each cited
document → audit record (hashes only) → streamed to the browser.

## 2. Source packages (`src/askiisc/`)

### `common/` — shared foundations
| File | Purpose |
|---|---|
| `types.py` | Immutable domain types (users, documents, chunks, answers) used by every package |
| `errors.py` | Typed exceptions; code raises these, never generic errors |
| `config.py` | The single typed settings object, read from `ASKIISC_*` environment variables |
| `targets.py` | Every numeric target from the PRD as a named constant (no magic numbers) |
| `clock.py` | Injected time source; `FakeClock` makes tests deterministic |
| `tokens.py` | Verifies institute SSO tokens (JWT, RS256) |
| `requestid.py` | Request id that ties one question's log lines together |

### `ingest/` — getting documents into the index (FR-5)
| File | Purpose |
|---|---|
| `crawl.py` | Polite, parallel, resumable crawler for public IISc sites |
| `archive.py` | Fetches past editions of academic pages from the Internet Archive |
| `boilerplate.py` | Detects site templates (menus, sidebars, footers) and strips them |
| `minhash.py` | MinHash signatures for near-duplicate detection |
| `catalog.py` | Builds the catalogue: duplicates removed, editions found, validity windows assigned |
| `editions.py` | Decides which documents are yearly editions of one another |
| `extract.py` | PDF, HTML, text and Office documents → per-page text |
| `chunk.py` | Overlapping chunks on sentence boundaries |
| `corpus.py` | On-disk corpus layout, per-document access policies, reconciler source |
| `versions.py` | Version lineage in `versions.db`; old versions retired via `valid_to`, never deleted |
| `lineage.py` | What replaced a superseded version (for stale-answer flags) |
| `reconciler.py` | Detects changed source documents and ingests the new versions |
| `pipeline.py` | `Ingestor`: one document version end to end (extract → chunk → embed → store) |
| `objects.py` | Original-document storage (in memory for tests, a directory standing in for MinIO on the server) |

### `access/` — standalone access-control service (FR-3, FR-4, NFR-10)
| File | Purpose |
|---|---|
| `directory.py` | User directory: departments, roles, groups from an export |
| `policy.py` | Policy engine, pure logic: may user U read document D? |
| `service.py` | FastAPI app exposing the policy engine |
| `server.py` | Runs the service as its own process |
| `client.py` | Client used by the gateway; every failure raises `AccessDenied` (fails closed) |

### `retrieval/` — permission- and version-filtered search (FR-2)
| File | Purpose |
|---|---|
| `embedder.py` | Embeddings from an HTTP embedding server, or a deterministic fake |
| `embed_server.py` | TEI-compatible embedding/rerank server for bge-m3 (TEI itself needs Docker) |
| `store.py` | Chunk index: Milvus HNSW (persistent) and an exact in-memory index |
| `dense.py` | In-memory exact index over every chunk, persisted in Milvus |
| `keyword.py` | BM25 scores over the same rows (hybrid search) |
| `temporal.py` | Which date(s) a question is about, so the right edition is searched |
| `rerank.py` | Cross-encoder reranking (bge-reranker-v2-m3) |
| `search.py` | `Retriever`: filters by allowed documents and valid edition *inside* the search |

### `llm/` — answering (FR-1, FR-8)
| File | Purpose |
|---|---|
| `prompts.py` | The RAG prompt; fixed prefix so vLLM prefix caching works |
| `rag.py` | `RagAnswerer`: grounded answers with citations, honest refusals, output guards |
| `client.py` | Async client for an OpenAI-compatible server (vLLM or the fake) |
| `models.py` + `models.yaml` | Model registry; switching models is a config change |
| `fake_server.py` | Deterministic stand-in for vLLM, used by all tests |

### `gateway/` — the request path (FR-6, FR-11, FR-12)
| File | Purpose |
|---|---|
| `app.py` | `POST /v1/ask`, streams a cited answer as Server-Sent Events |
| `auth.py` | Login check and tenant lookup |
| `early.py` | Fast shed path before the route runs (429 within 50 ms under overload) |
| `limits.py` | Per-tenant load limits and admission control |
| `drr.py` | Deficit Round Robin across tenant queues (fair sharing) |
| `cache.py` | Answer cache whose key includes permissions and document versions |
| `output_filter.py` | Re-checks every cited document before the answer leaves |
| `audit.py` | Append-only, privacy-preserving audit records (salted hashes, no text) |

### `autoscaler/` — GPU workers (FR-9, FR-10)
| File | Purpose |
|---|---|
| `policy.py` | Scaling decision as a pure function (queue, workers, idle time → target size, max 2) |
| `controller.py` | Loop that applies the policy, drains idle workers, replaces crashed ones |
| `pool.py` | Worker pools: `ProcessPool` (one `vllm serve` per free GPU), `FakeWorkerPool` (tests) |

### `observability/` — metrics, logs, cost (NFR-6)
| File | Purpose |
|---|---|
| `metrics.py` | Prometheus metrics, one registry per component |
| `tracing.py` | OpenTelemetry spans: ask → access → search → llm → filter |
| `stages.py`, `timing.py` | Per-request stage timings and serving-node latency/lag |
| `gpu.py` | NVML GPU sampling at 1 Hz (stands in for DCGM, which needs root) |
| `cost.py` | Cost model: GPU-hours per 1,000 queries incl. start-up and idle |
| `vllm_stats.py`, `vllm_startup.py` | vLLM queue/KV-cache stats; model memory from start-up logs |
| `authproxy.py` | Basic-auth front door for Loki on a Unix socket |

### `web/` — chat app and admin dashboard
| File | Purpose |
|---|---|
| `app.py` | Accounts and sessions; mounts chat and admin routers |
| `chat.py` | Chats, streamed messages, temporary chats, ratings, export |
| `registration.py`, `pending.py`, `password_links.py`, `passwords.py`, `throttle.py`, `mailer.py` | Sign-up for allowlisted people, e-mail confirmation, password reset, Argon2id hashing, login throttling |
| `store.py`, `dbfiles.py` | SQLite store for accounts, sessions, chats; where the files live |
| `admin.py` | Admin routes: usage, request log, accounts, policies, load tests |
| `people.py`, `people_admin.py` | The users tree and its admin tab |
| `documents.py` | Admin document upload, new versions, version history |
| `requestlog.py` | One row per request, never any text |
| `scorecard.py` | FR-1..FR-12 / NFR-1..NFR-10 scorecard from the latest measurements |
| `models.py` | Model comparison page |
| `cli.py` | Account administration from the command line |
| `static/` | HTML, CSS, JS of both UIs |

## 3. Outside `src/`

| Folder | Purpose |
|---|---|
| `deploy/serve.py` | The serving job: starts embedding server, access service, index, gateway, web app, autoscaler, Prometheus, Grafana, Loki/Alloy, Locust UI, all on `127.0.0.1` |
| `deploy/ingest.py` | Batch ingest of the corpus into the persistent index |
| `deploy/stack.py` | Helpers shared by the jobs (ports, process start/stop, health checks) |
| `deploy/logs.py` | Loki (log store) and Alloy (log shipper) config |
| `deploy/grafana/` | Dashboards (`askiisc.json`, `requests.json`) and alert rules |
| `slurm/` | Job scripts: `setup_env`, `fetch_services`, `fetch_models`, `crawl`, `archive`, `catalog`, `ingest`, `serve`, `loadtest`, `gate`, `run`, experiment stacks (`fr6_`, `fr8_`, `v3_stack.sh`); `lib.sh` holds shared paths and GPU picking |
| `eval/locust/` | Load-test workloads (one per requirement) and the client-side request log |
| `eval/versioned/`, `eval/heldout/` | Edition-aware and held-out question sets: generate, answer, grade |
| `eval/grader/` | Deterministic rubric, LLM judge, human adjudication sheet (NFR-7) |
| `eval/oracle/` | Deliberately naive access oracle, independent of `access/policy.py` |
| `eval/probes/` | Access checks and recall on the real index (FR-2, FR-3, NFR-10) |
| `eval/redteam.py` | 90-case prompt-injection / leak suite (NFR-9) |
| `eval/scaling.py`, `eval/llm_bench.py` | Scaling/cost from Prometheus; vLLM throughput alone (NFR-3) |
| `eval/datasets/` | Sample users and mock documents, held-out and red-team sets |
| `tools/` | Coverage gate, UI preview on fakes, evidence pages, v3 results compiler |
| `docs/results/` | Measured evidence per requirement |
| `tests/` | See section 4 |

## 4. Testing

Tests are hermetic: no internet, no GPU. They use the fake vLLM
(`llm/fake_server.py`), Milvus Lite, in-memory fakes and `FakeClock`.

| Target | What runs |
|---|---|
| `make all` | format check, ruff, mypy, unit tests with coverage, dependency audit |
| `make test-int` | integration: real Redis, HTTP services, Milvus Lite |
| `make test-e2e` | whole stack with the fake model |
| `make test-sec` | security: access leaks, injection, auth |
| `make test-regress` | golden-file regressions |
| `make test-perf` | latency budgets |

On the cluster run them inside Slurm, never on the login node:
`sbatch --wait slurm/gate.sh all test-int test-e2e test-sec test-regress`.

Coverage: at least 90% overall, 100% on `access/`, `gateway/limits.py`,
`gateway/drr.py`, `autoscaler/policy.py`. `tests/conformance/` runs the
same cases against any access-control implementation.

## 5. Rules that keep it safe

1. Access control fails closed: any error or timeout means deny.
2. Permission and version filters go inside the search, never after it.
3. User identity comes only from the verified session; department and role
   only from the directory.
4. Never log raw query or answer text; users and queries are salted hashes.
5. Retrieved chunks are untrusted data; the prompt prefix stays identical.
6. Superseded versions are never deleted, only retired (`valid_to`); answers
   citing them carry a stale flag.
7. Never more than 2 GPU workers.
8. Every service binds to `127.0.0.1` and dies with its Slurm job.

## 6. Conventions and extending

- Python 3.11, Google style (80 columns, type hints, docstrings), ruff +
  mypy; small single-purpose functions; no global state (inject clocks,
  clients, databases); config only via `common/config.py`.
- Conventional Commits naming the FR/NFR (`feat(gateway): ... (FR-6)`);
  pre-commit runs ruff, mypy and gitleaks.
- **New model:** add an entry to `llm/models.yaml`, fetch weights with
  `slurm/fetch_models.sh`, serve with `ASKIISC_MODEL=<key> sbatch slurm/serve.sh`.
- **New document source:** produce documents in the corpus layout
  (`ingest/corpus.py`) with a `.policy.json`, then rerun `slurm/ingest.sh`;
  only new or changed documents are ingested.
- **New access rule:** change `access/policy.py`, add matching cases to
  `tests/access_fixtures.py`, and check the oracle still agrees.
- **New load workload:** add it to `eval/locust/workloads.py`; it appears in
  the Locust UI and `slurm/loadtest.sh <name>`.
