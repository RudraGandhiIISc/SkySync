# Architecture

How Ask IISc works: the parts, the path of a question, the path of a
document, the access model, and where each piece of code lives. Running it
is in [OPERATIONS.md](OPERATIONS.md); the requirements are in
[PRD.md](../PRD.md).

## 1. What it does

Ask IISc answers questions about IISc documents (admissions, academic
rules, curricula, course pages, handbooks, department pages):

- Every answer cites the passages it came from, or says exactly
  "I don't know based on the provided documents."
- Each user only ever gets text they may read.
- Yearly documents are kept in every edition, so "this year" and
  "in 2025-26" are answered from the right one; nothing is ever deleted.
- GPU workers start when questions arrive and stop when the system is idle.

```
browser --SSH tunnel--> web app (chat /, admin /admin): accounts, sessions, chats
                          | internal token
                          v
                        gateway --> access service (own process): which documents may this user read?
                          |  --> time scope: which date(s) the question is about
                          |  --> embedding server: question -> vector (bge-m3)
                          |  --> in-memory index: hybrid search per date, filtered INSIDE the
                          |        search by (allowed documents, edition valid at that date)
                          |  --> reranker (bge-reranker-v2-m3): best 12 of 80 candidates
                          |  --> vLLM worker (GPU): answer with [chunk] citations
                          |  --> output guards and filter: re-check every cited document
                          v  --> audit log (hashes) + admin request log (user, no text)
                  streamed answer + cited sources (edition, link to the passage)
autoscaler: 0-2 vLLM workers on free H200s, scale to zero when idle, replace crashed ones
corpus:     crawl -> Internet Archive editions -> catalogue (templates, duplicates, editions)
            -> ingest (chunk, embed, Milvus + versions.db); every version kept
monitoring: Prometheus -> Grafana (2 dashboards, 6 alert rules); Loki for per-request logs
load:       Locust drives one workload per requirement with synthetic users
```

Everything runs on the CSA `automation` server in Slurm jobs, with no
containers and no root: one H200 per vLLM worker, every service on
`127.0.0.1`. Other users of the server can reach a loopback port, so the
vLLM workers and the embedding server answer only calls carrying the
serving job's key (`common/servicekey.py`), and the access service checks
a signed token on every decision.

## 2. How a question is answered

**Chat app.** `POST /api/chats/{id}/messages` (or `/api/temporary`), with
the session cookie and the `X-AskIISc: 1` header:

- `web/app.py` checks the session, the account and the directory entry,
  and issues a short-lived internal token.
- `web/chat.py` loads the chat's visible messages as history and calls the
  gateway's `ask`. Once the answer streams, it stores the question; when
  the answer ends, it adds titles, editions and passage links to the
  citations and saves it (text encrypted).

**Gateway** (`gateway/app.py`; `POST /v1/ask` is the same path):

1. **Authenticate** (`gateway/auth.py`): RS256 token, user id from `sub`
   only; tenant = the user's department from the directory. 401 / 403.
2. **Size check:** over 512 question tokens is 413.
3. **Admission** (`gateway/limits.py`): per-department token bucket, queue
   bound and a fair share of 24 requests in flight; otherwise 429 with
   `Retry-After`, decided at once.
4. **Permission set** (`access/client.py`): the documents the user may
   read, cached 10 s per token. Any failure or timeout is 403 (fail
   closed).
5. **Answer cache** (`gateway/cache.py`): keyed by the permission set,
   policy version, corpus generation and question. Follow-ups skip it.
6. **Time scope** (`retrieval/temporal.py`): no time words, "this year" or
   "current" mean now; "2025", "2025-26" mean that academic year (searched
   at its start and end); "January 2025" that month; "last year" a year
   ago; "what changed" also the year before. The time phrase is taken out
   of the search text, since the date filter already applies it.
7. **Retrieve** (`retrieval/search.py`, `retrieval/dense.py`), once per
   date: embed the question (bge-m3, query instruction); score every chunk
   by vector and by BM25 keywords, both under the masks for "allowed
   document" and "edition valid at this date" (applied before ranking);
   fuse by reciprocal rank (keyword weight 2) to 80 candidates; rerank
   with bge-reranker-v2-m3. Results of several dates are merged.
8. **Prompt** (`llm/prompts.py`): the best 12 chunks within 16,000
   characters (each at most 2,000), wrapped in `<chunk>` tags with the
   title, edition and a short id; a fixed system prompt first (prefix
   cache); the last 6 chat turns. No chunks: refuse without the model.
9. **Hold for a worker** if none is ready (`warming` event), then the
   **fair queue** (`gateway/drr.py`, deficit round robin by department).
10. **Generate** (`llm/client.py`): vLLM, OpenAI-compatible streaming,
    `cache_salt` per department, 16,384-token context.
11. **Parse and guard** (`llm/rag.py`, `gateway/output_filter.py`): keep
    citations of retrieved chunks only; mark superseded editions
    `is_stale`; an answer that cites nothing, repeats the system prompt,
    or whose content words are mostly absent from the chunks it cites
    (under 30%: outside knowledge) becomes the refusal; links not in a cited chunk and HTML are removed;
    every cited document is re-authorised.
12. **Record**: an audit line of hashes (no text) with the ready workers,
    a cold-start flag, a refusal reason and milliseconds per stage, named
    by the response's `X-Request-Id`; a request-log row (user, outcome,
    timings, documents; no text); Prometheus metrics.

## 3. How a document gets into the index

```
slurm/crawl.sh    -> corpus/public/**, manifest.jsonl          (live pages)
slurm/archive.sh  -> corpus/archive/**, archive.jsonl          (past editions)
slurm/catalog.sh  -> catalog.jsonl, duplicates.jsonl, boilerplate.json,
                     policies.json, titles.json                (what to index)
slurm/ingest.sh   -> state/versions.db, milvus.db, objects/, dense/
```

- **Crawl** (`ingest/crawl.py`): every public `*.iisc.ac.in` host except
  portals, mail and the research repositories; HTML, PDF and Office files;
  robots.txt, one request per second per host; sitemaps and
  certificate-transparency host lists; resumable.
- **Past editions** (`ingest/archive.py`): for academic pages, the
  Wayback Machine snapshot of each academic year since 2022 whose text
  differs from the live copy.
- **Catalogue** (`ingest/catalog.py`):
  - learns each site's template (menus, footers repeated on at least 2% of
    its pages, `ingest/boilerplate.py`) and removes it;
  - drops exact and near duplicates (MinHash similarity 0.9 within a site,
    0.98 anywhere), keeping their addresses as aliases;
  - groups editions into families (`ingest/editions.py`: the address with
    dates removed, similar text) and reads each edition's date from the
    address, title, upload folder, opening text or snapshot time;
  - gives each edition a validity window: the newest is current, each
    older one ends when the next starts. Different editions are never
    merged as duplicates.
- **Ingest** (`deploy/ingest.py`, `ingest/reconciler.py`,
  `ingest/pipeline.py`): only new or changed documents (ETag or content
  hash). Each version is extracted (`ingest/extract.py`), stripped of its
  site template, chunked (`ingest/chunk.py`: about 190 tokens, 30 overlap,
  table rows whole), given a first line "title | edition | site", embedded
  and written to Milvus; only then recorded in `versions.db`. A new
  version retires the previous one by setting `valid_to`; old versions and
  their chunks are kept (safety rule 6) and stay searchable "as of" their
  own time.

The serving job holds the whole index in memory (`retrieval/dense.py`:
574,672 vectors of 1,024 dimensions, with a BM25 index over the same rows),
loaded from a snapshot in `state/dense/` or rebuilt from Milvus when the
stamp of `versions.db` changed. It stops at start-up if the embedding
model's vector size does not match the index.

## 4. Access control

| Piece | Meaning |
|---|---|
| Person (`access/directory.py`) | user id, department, role, groups; for students a programme and joining year, for faculty and staff an optional type and joining year; from the users tree `users/<dept>/<students\|faculty\|staff\|admins>.json`; never from the client |
| Departments and programmes | CSA: PhD, M.Tech (Research), M.Tech CSE. CDS: PhD, M.Tech (Research), M.Tech (CDS) (`DEPARTMENT_PROGRAMS`, from the departments' websites, 2026-10). Faculty types: faculty, visiting / guest faculty. Staff types: administrative, technical, research staff (postdoc, project) |
| Groups added automatically | `<dept>-<role file>` (`cds-students`); for a student `<programme>`, `<dept>-<programme>`, `<dept>-<programme>-<year>`, and `<dept>-research-students` (PhD, M.Tech (Research)) or `<dept>-course-students` (M.Tech), e.g. `phd`, `cds-phd`, `cds-phd-2023`; for a non-plain type `<dept>-<type>-<role file>` (`csa-visiting-faculty`, `cds-research-staff`) |
| DocPolicy (`access/policy.py`) | owner and co-owners (admins of any of them, and institute admins, manage it; owners can be added, but only removed by their own admins), visibility (public / restricted), groups, departments, roles, allow and deny lists, validity dates |

Decision order: deny list, then validity dates, then public, then allow
list, then a shared group, department or role; otherwise deny. Unknown
users, unknown documents and any error deny. `eval/oracle/` re-implements
the rules independently and replays decisions.

- The permission set is applied inside the search as a document mask, so a
  forbidden chunk is never ranked; cited documents are checked again before
  an answer leaves, and a chat answer citing a document the user can no
  longer read is hidden when the chat is opened.
- Policy changes (`GET/PUT/DELETE /v1/policy/{doc}`) need an admin of the
  owning department or an institute admin; they apply within seconds
  (60 s required) and are kept in `private/app/policy_overrides.jsonl`.
- Admins: institute admins (added only from the terminal) manage everyone;
  department admins manage their own department's students, faculty and
  staff; every admin can view everything. Users see only the chat app.

## 5. The web app

Plain HTML, CSS and JavaScript (`web/static/`), no build step and no
external files; strict Content-Security-Policy.

- **Chat** (`index.html`, `chat.js`): sign in with e-mail and password,
  "Forgot password?", and a page that sets a password from a single-use
  link; chats grouped by date,
  search, rename, delete, export as Markdown with numbered sources,
  temporary chats (never stored), streamed answers with numbered citations
  and source cards (title, site, page, edition, superseded flag, a link
  that opens the cited passage), copy, ratings, regenerate, stop, settings
  (change password, delete chats, delete account),
  light and dark.
- **Admin** (`admin.html`, `admin.js`, `policy_editor.js`): Overview
  (latency against target, requests, errors, cache, GPU workers, index
  size, alerts), People (add, bulk CSV, edit, suspend or restore
  access, remove; sign-in state, registered
  and last sign-in times; the automatic groups shown), Request
  log (filters, CSV, a Grafana logs link per request), Documents (search;
  for the chosen document who can read it, as a form with a live count, its
  versions and a new version from a file or text; upload of a new document
  under an id not in use, with one or more owning departments), Requirements
  (scorecard), Models. A **Users** switch (real, Locust, both) filters
  the Overview and People; long tables come in pages of 20, 50 or 100
  rows with previous and next arrows.
- **Real and test people:** the dashboard adds and changes only real
  people, named by their e-mail address. Locust's test users (ids without
  "@") get synthetic accounts only from the terminal
  (`askiisc.web.cli seed-locust`) and cannot be changed on the dashboard.
- **Sign-in** (`web/app.py`, `web/password_links.py`, `web/passwords.py`,
  `web/throttle.py`, `web/registration.py`): e-mail and password. A person
  needs an active allowlist entry (`allowed_users`, written by admins;
  never inferred from the domain), a directory record, an active account
  and a password. Every refusal gets the same answer and costs one Argon2
  check. Passwords are Argon2id (argon2-cffi); old scrypt hashes are
  replaced at the next sign-in. Allowlisted people create their own
  account: the password is chosen on the form and takes effect when the
  e-mailed confirmation link (10 minutes, kept in memory,
  `web/pending.py`) is opened; adding a person sends nothing.
  "Forgot password?" reset links last 10 minutes; admins send none.
  Only a link's SHA-256 is stored (`password_resets`), a new link
  voids older ones, and using one ends every session. Links are e-mailed
  over SMTP (`web/mailer.py`, `private/app/smtp.json`) to the person's own
  address, or shown once to the admin who asked. Only addresses of the
  allowed domains (`ASKIISC_EMAIL_DOMAINS`) can be added. Failed sign-ins
  are
  slowed per typed address (5 free in 15 minutes, then 30 s doubling to
  15 minutes) and across all addresses (60 a minute). Every request
  re-checks the account and the allowlist, so a suspension applies at
  once. Test identities (ids without "@") sign in only while
  `ASKIISC_TEST_LOGINS` is on.
- Sessions: random cookie (HttpOnly, SameSite=Strict, `Secure` with
  `ASKIISC_SECURE_COOKIES`) stored hashed, 12 h, replaced at every sign-in
  and ended by a password change or a change of rights; state-changing
  calls need the `X-AskIISc` header and a same-site `Origin` (CSRF);
  chat text encrypted with Fernet; live databases on local disk with
  backups on `/data4` (`web/dbfiles.py`).

## 6. Autoscaling and observability

- `autoscaler/policy.py` decides scale out / in / hold (never above 2
  workers, never scale in with work in flight, 30 s hysteresis);
  `controller.py` runs it every second and reaps crashed workers;
  `pool.py` starts `vllm serve` on the least busy H200, sizing its memory
  from the card's free memory (a 4-bit model starts in about 21 GB free,
  a BF16 8B model needs about 39 GB). When every card is too full for a
  worker (other users' jobs), a held question ends at once with a "GPUs
  full" message and a retry time instead of waiting out the 150 s hold
  (FR-10); the next question tries to start a worker again.
- `deploy/watchdog.py` restarts the access service or the embedding server
  if either dies, on the same port and with the same key (backoff 1 s,
  doubling to 60 s if it keeps dying).
- Prometheus scrapes the gateway, access service, vLLM workers and Locust;
  Grafana has the "Ask IISc" dashboard (37 panels) and "Ask IISc requests"
  (Loki logs by request id), plus six alert rules
  (`deploy/grafana/`). Every request id is passed to the access service,
  embedding server and vLLM; no question or answer text is logged.

## 7. Code map

| Folder | What is in it |
|---|---|
| `src/askiisc/common/` | types, typed errors, clock (`FakeClock` for tests), settings from `ASKIISC_*`, every PRD number in `targets.py`, tokens, request ids |
| `src/askiisc/ingest/` | `crawl`, `archive`, `boilerplate`, `catalog`, `editions`, `minhash`, `extract`, `chunk`, `corpus` (corpus folder to documents and policies), `reconciler`, `pipeline` (`Ingestor`), `versions` (`versions.db`), `objects` |
| `src/askiisc/access/` | `directory`, `policy`, `service` (FastAPI), `server` (own process), `client` |
| `src/askiisc/retrieval/` | `embedder`, `embed_server` (bge-m3 and the reranker), `rerank`, `store` (Milvus), `dense` (in-memory exact search), `keyword` (BM25), `temporal` (time scope), `search` (`Retriever`) |
| `src/askiisc/llm/` | `prompts`, `rag` (`RagAnswerer`, parsing, guards), `client`, `models.py` + `models.yaml` (the six models; Granite-8B is also the judge), `fake_server` (tests) |
| `src/askiisc/gateway/` | `app` (request path), `auth`, `limits`, `drr`, `cache`, `output_filter`, `audit` |
| `src/askiisc/autoscaler/` | `policy`, `controller`, `pool` |
| `src/askiisc/observability/` | `metrics`, `gpu` (NVML), `cost`, `tracing`, `authproxy` (Loki login) |
| `src/askiisc/web/` | `app` (accounts), `chat`, `admin`, `people`, `people_admin`, `documents`, `requestlog`, `scorecard` (+ `scorecard_map.yaml`, `baseline_v1.yaml`), `models`, `reset`, `mailer`, `store`, `dbfiles`, `passwords`, `cli`, `static/` |
| `deploy/` | `serve.py` (the serving job), `ingest.py` (batch ingest), `stack.py` (helpers both use), `logs.py` (Loki, Alloy), `grafana/` (dashboards, alerts) |
| `eval/` | `locust/` (16 workloads), `versioned/` (edition-aware questions: generate, answer, grade, retrieval-only check), `heldout/`, `grader/` (rubric, judge), `oracle/`, `probes/` (FR-2 recall, NFR-10 latency), `redteam.py`, `llm_bench.py` (NFR-3), `scaling.py`, `measurements.py` (scorecard inputs), `datasets/` |
| `slurm/` | job scripts (OPERATIONS.md section 3) |
| `tools/` | `check_coverage.py` (gate), `ui_preview.py` + `ui_shots.js` (screenshots on fakes), `evidence.py` + `evidence_specs.py` + `grafana_shots.js` (the `docs/results/live-v2` pages) |
| `tests/` | unit, conformance (any access implementation), integration (real Redis, HTTP services, Milvus Lite), e2e (whole stack with the fake model), security, regression (goldens), perf |

## 8. Data on the server

| Place | What |
|---|---|
| `/data4/muthangiv/askiisc/corpus-v3/` | the live corpus: `public/`, `archive/`, `restricted/` (22 synthetic test documents with `.policy.json`), `manifest.jsonl`, `archive.jsonl`, `catalog.jsonl`, `policies.json`, `titles.json` |
| `/data4/muthangiv/askiisc/state-v4/` | the live index: `versions.db`, `milvus.db`, `objects/`, `dense/` |
| `/data4/muthangiv/askiisc/corpus/`, `state/` | the previous corpus and index (BGE-base, 768 dimensions), kept, not served |
| `/data4/muthangiv/askiisc/private/` | the users tree and app secrets (mode 700): accounts backup, keys, passwords, `locust_accounts.json`, `policy_overrides.jsonl` |
| `/data2/scratch/muthangiv/askiisc/` | both Python environments, service binaries, the live `app.db` and `requests.db` |
| `/data4/muthangiv/huggingface/` | model weights |
| `/data4/muthangiv/askiisc/results/` | raw output of every job: `serve/<job>/`, `locust/<job>-<workload>/`, `measurements/`, `versioned/` |

Formats worth knowing: `catalog.jsonl` has one row per document with
`doc_id, path, url, original_url, snapshot, aliases, title, department,
family, editions, edition, edition_start, valid_from, valid_to`;
`versions.db` has `versions(doc_id, version, valid_from, valid_to,
fingerprint, acl, drafts)`; chunk ids are `doc@vN#i`.
