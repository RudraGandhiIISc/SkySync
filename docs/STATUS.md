# Status

Where Ask IISc stands on 2026-11-06, what was measured, how it differs from
the PRD, and what is left. Every number comes from a Slurm log or a results
file under `/data4/muthangiv/askiisc/results/`.

## 1. Summary

- **Serving:** `sbatch slurm/serve.sh` runs the whole system in one GPU
  job: chat app, admin dashboard, Prometheus, Grafana, Loki and the Locust
  UI, reached through an SSH tunnel. Default model Qwen3-8B, up to 2 GPU
  workers.
- **Corpus and index:** the cleaned corpus `corpus-v3` and its index
  `state-v4` (bge-m3): 30,740 public documents (duplicates and site
  templates removed, past editions kept with validity windows) plus 22
  restricted test documents; 574,672 chunks.
- **Answer quality:** Qwen3-8B answers 84.5% of 400 edition-aware
  questions correctly, citations 97.7% correct (NFR-7: 80% and 90%).
- **Quality gate:** green on 2026-11-06 (job 2154: integration, e2e,
  security, regression; job 2156 on the last fix: format, lint, mypy
  --strict, 1,137 unit and conformance tests, coverage above 97%).
- **v3 runs (2026-11-06):** results in
  [results/v3/RESULTS.md](results/v3/RESULTS.md), built from the raw
  files by `tools/v3_results.py`. All on GPU 0 shared with other users'
  jobs at 100% (R-3), one worker, Qwen3-8B-AWQ unless noted.
- **Live check** (serving job 1794, 2026-10-27): every chat and admin
  action works on the real system; "this year" and "in Aug 2025" questions
  cite the right DS252 edition and mark the old one superseded; a CDS
  student never gets a CSA-only document while CSA faculty do; the Locust
  `access` workload checked 2,562 citations with 0 leaks, and a revoked
  document stopped being cited after 7.4 s (60 s required).

## 2. Requirements

Measured in the live-v2 round (2026-10-26, Qwen3-4B, 8 chunks:
[results/live-v2/](results/live-v2/README.md)) unless noted; the admin
**Requirements** tab always shows the newest measurement.

| Req | State | Notes |
|---|---|---|
| FR-1 cited answers, refusals | pass | |
| FR-2 no forbidden chunk retrieved | pass | recall@8 1.000, 0 leaks in 10,000 probes; 0 leaks in the 2026-10-27 live run |
| FR-3 revocation within 60 s | pass (7 revocations) | 3.1 s (live-v2), 7.4 s (2026-10-27). E1's 100 scripted revocations: the 2026-11-06 run (job 2143) is invalid (a test bug ended it after 23 cycles); re-run queued (job 2156) |
| FR-4 access service failover | pass | denies while down, never answers unchecked |
| FR-5 versions and staleness | **open** | citations name the version that replaced them (`superseded_by`) and answers carry `staleness_flag`. Replay of 60 revisions over 20 documents (job 2143): as-of correct at 60 of 60 checkpoints, demotion p50 17 s, p95 44 s, max 99 s; but 266 answers from the answer cache used a superseded version without the flag (target 0). Fixed (cache generation bumped again when a revision becomes visible); re-run queued (job 2156) |
| FR-6 429 within 50 ms at 3x load | **pass** (2026-11-06) | W2 at 3x, median of 3 (job 2134): 429 p99 24.3 ms at the serving node, accepted p95 4.1% lower than at 1x. Client 429 p99 72 to 90 ms. With admission off nothing collapsed at this load (82% answer-cache hits), so E4's collapse is not shown ([results/v3](results/v3/RESULTS.md)) |
| FR-7 batching and prefix cache | pass at vLLM, **open** at the client | W1, answer cache off, 3 runs each (job 2143): prefix-cache hit rate 65.9% (target 50%); p95 first token 38.9% lower at vLLM (935 against 1,529 ms) but 7.4% lower at the client (2.5 against 2.7 s; retrieval before the model dominates). Where first token is measured is not decided |
| FR-8 six models | pass | six models (section 4) through one client, switched by `ASKIISC_MODEL` only; all six measured 2026-11-03 ([results/fr8](results/fr8/README.md)) |
| FR-9, FR-10 scaling, scale to zero | pass | 2 workers in 74 s, never above 2, back to 0 61 s after idle |
| FR-11 fairness | **open** | quiet department 33% (≥ 20% met); busy department 67% (≤ 55% not met) |
| FR-12 audit | pass (2026-11-05) | v3: 2,633 of 2,633 requests audited, 1,940 of 1,940 citations reproduced, 0 question strings in the store; latency breakdown per stage ([results/v3/FR-12](results/v3/FR-12/README.md), run 2099/2100, shared GPU) |
| NFR-1 latency | **open** with 8B (3 runs running, job 2143) | Qwen3-8B, 12 chunks, `steady` 60 users (job 1818, 2026-10-27): p95 total 4.0 s, first token 1.7 s; embedding (p95 0.64 s) and reranking (0.88 s) before the model, vLLM's own first token 0.38 s |
| NFR-2 cold start | pass | Qwen3-8B: worst first token 34.7 s (job 1819) |
| NFR-3 throughput | pass | 5,195 tokens/s at concurrency 64 (4B) |
| NFR-4 search at scale | pass | |
| NFR-5 4-hour soak with a node killed | **open** | not run |
| NFR-6 GPU-hours per 1,000 queries | pass | Qwen3-8B `diurnal`: 0.181 (job 1820) |
| NFR-7 accuracy, citations | **pass** (2026-10-27) | 84.5% and 97.7% on the edition-aware set with Qwen3-8B; the held-out v2 set scored 64.7% with Qwen3-4B on the old corpus. Review sheet for people: 111 rubric/judge disagreements plus a 60-item sample (job 2115, [results/v3](results/v3/README.md)) |
| NFR-8 freshness | pass | |
| NFR-9 security | pass | 0 leaks; 1 of 69 injections succeeded (≤ 5%) |
| NFR-10 access latency | pass | |

## 3. Answer quality (edition-aware set, 2026-10-27)

| Model | Accuracy | Strict edition | Citations | Edition correct | False refusals |
|---|---|---|---|---|---|
| **Qwen3-8B** (current prompt) | **84.5%** | 71.8% | 97.7% | 95.7% | 3.1% |
| Qwen3-4B-Instruct-2507 | 79.8% | 71.0% | 98.5% | 86.5% | 10.0% |

Qwen3-8B by type: change 90%, year 84%, current 80%, general 86%,
unanswerable 77.5%. Method and set: [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md).

What moved it from 63% (same pipeline on the uncleaned corpus):

| Change | Effect |
|---|---|
| Site templates removed (23% of HTML text was menus and footers) | chunks hold the page's own text; menu changes no longer look like new editions |
| About 10,900 generated Lean/Mathlib API pages taken out of scope | a third of the old index was code reference |
| Duplicates removed, editions grouped with validity windows (231 families with several editions) | the right edition is found for dated questions |
| "title \| edition \| site" line at the top of every chunk | search, reranker and model see which document and year a passage is from |
| 80 reranked candidates, 12 chunks and 16,000 characters in the prompt; the time phrase kept out of the search text | key facts reach the prompt for 86% of items (81% before) |
| Prompt: no guessing from related facts; cite each edition compared | change questions 85% → 90% |

## 4. Deviations from the PRD

| PRD | Here | Why |
|---|---|---|
| Kubernetes, Docker Compose, MinIO, TEI, DCGM | Slurm jobs, `deploy/serve.py`, files, sentence-transformers, NVML | no containers or root on this server |
| Milvus HNSW search; filter in the Milvus expression | Milvus Lite stores the chunks; search is exact, in memory, with permission and validity masks applied inside the search before ranking | Milvus Lite took 840 ms per search; exact search takes about 20 ms with perfect recall, and the safety guarantee is the same |
| 8 chunks, 4,096-token context | 12 chunks (16,000 characters), 16,384 tokens; FR-2 recall@8 still measured at 8 | more of the right passages reach the model; costs latency and GPU time, so NFR-1 and NFR-6 are measured again |
| Qwen3-4B as the first variant | Qwen3-8B by default | only the 8B model meets NFR-7 |
| Never store question or answer text | chat history stored encrypted (Fernet), readable only by its owner; request log and audit hold no text | chat history was asked for |
| Institute SSO | E-mail and password (Argon2id) for allowlisted people, chosen at self-registration confirmed by an e-mailed link; single-use reset links | registering an app in IISc's Microsoft tenant was not possible; allowlist managed by admins (docs/OPERATIONS.md section 5) |
| Nine models (FR-8, NFR-6) | Six: Qwen3-4B, Phi-4-mini (4B); Qwen3-8B, Granite-3.3-8B (8B); Qwen3-8B-AWQ, Granite-3.3-8B-GPTQ (4-bit, published checkpoints) | Gemma-3 and Llama-3.1 need an accepted licence; Llama-3.1-8B-AWQ dropped with its parent; two models per size class (2026-11-02). Granite's 4-bit build is GPTQ (TevunahAi): the drawais AWQ build produced gibberish on long RAG prompts under vLLM 0.11 (2026-11-03) |
| Teammates on the network | services on 127.0.0.1, SSH tunnel per person | server rule |

## 5. Known limits

- **Grounding is checked by words, not meaning.** An answer whose content
  words are almost all missing from the chunks it cites (under 30%, as for
  "Paris" citing a seminar page) becomes the refusal. A stricter check was
  tuned out: on the 400 graded answers it would refuse more correct answers
  than wrong ones, because wrong answers to unanswerable questions are
  mostly built from related passage text.
- **Tokens stream before the guards run;** a refused answer is replaced
  when the stream ends.
- **The question sets are drafts** (`reviewed: false`) until a person
  checks them.
- **Revision test items** need the test circulars' v1-to-v2 history,
  which only the old index has.
- **Query-string pages** (`page.php?id=N`) are not crawled: the crawler
  drops query strings, so one page per script is kept.
- **No TLS on the gateway** (reached only through the SSH tunnel); no
  packet capture (no root).

## 6. What is left (2026-11-06)

**Running and queued.** Commands and parts: `slurm/v3_stack.sh`.

| Job | What | Expected end |
|---|---|---|
| 2143 (running) | NFR-1 (`steady` x3, Qwen3-8B); 20 scripted cold starts each for Qwen3-4B, Qwen3-8B, Qwen3-8B-AWQ (NFR-2, FR-10); Falcon3 3B, 7B, 7B-AWQ smoke tests and the KV comparison at 24 GB (FR-8); E2, one W1 run per model | about 17:00 |
| 2156 (queued) | FR-5 freshness replay again (checks the cache fix) and FR-3's 100 scripted revocations | about 18:30 |
| 2124 (queued) | the standing serving job again (Qwen3-8B-AWQ, 1 worker, renews itself) | after 2156 |

**Not scheduled.**

1. **E3 idle-timeout sweep** (T_idle 0, 60, 300, 900 s and none, W3 at
   24x, 3 models): about 15 GPU-hours. Left for later on 2026-11-06.
2. **E0 throughput sweep and the NFR-3 baseline:** needs an idle card;
   GPU 0 was shared all day. Explain 5,195 against 2,660 tokens/s then.
3. **FR-8 nine models:** add Falcon3 to `llm/models.yaml` (with the count
   and registry test set to 9 in the same commit) once its smoke tests
   pass; Gemma-3 and Llama-3.1 still need accepted licences.
4. **FR-8 and NFR-7 for new models:** quality runs, and an outside judge
   for the Granite rows (a proposal is owed before this).
5. **NFR-6 three ways** (rated load, goodput, W3 replay) and per 1,000
   correct answers, for every model; needs E2 and the sweep.
6. **FR-11:** the busy department still gets 67% (at most 55%). Run
   `fairness` with the fair queue on and off; a token quota per
   department only after approval.
7. **E4 collapse:** W2 at 3x did not saturate the GPU (answer cache).
   Showing the collapse needs W2 with the answer cache off, or more load.
8. **NFR-4 chat-app 503s:** 1.2% in run 2090 (limit 1%); not seen in the
   2026-11-06 runs, to be checked in a full report.
9. **NFR-5:** the 4-hour soak with a node killed (about 4 GPU-hours).
10. **Decisions for people:** where FR-7's first token is measured
    (client or vLLM); fill in the NFR-7 review sheet
    (`/data4/muthangiv/askiisc/results/adjudication/qwen3-8b-p2/review.csv`,
    171 rows); review the question sets; real department admins; server
    accounts for teammates.
11. **Close-out:** evidence page per requirement, traceability table,
    `make figures`, the full gate, merge the phase branches
    (`phase-14` to `phase-19`) in order; the release push needs the GitHub
    address and a deploy key.

## 7. Bugs found and fixed

Each has a regression test unless noted.

| Bug | Found by |
|---|---|
| Two links redirecting to one page saved it twice; a pypdf error stopped the crawl | real crawl |
| A 105k-character chunk broke the Milvus upsert; a job killed mid-document marked it done with no chunks; lone surrogates broke embedding | real ingest |
| `[chunk:a, chunk:b]` citations dropped; prompts over the context limit | real answers |
| The policy filter evaluated all policies per request; Milvus Lite took 840 ms per search | load test (fixed by the in-memory index) |
| Scale-to-zero flapping; a crashed vLLM worker stayed in the routing list | tests, soak review |
| SQLite on the NFS drive (122 ms per commit) capped the app at about 5 requests/s; numpy and torch started a thread per core | load test |
| Locust ignored `Retry-After`; its quality run ended with questions in flight (checked live only) | Locust runs |
| Superseded editions were ingested again on every run | catalogue rebuild |
| Regenerate deleted the question and answer before asking again, so a refusal lost both | review, 2026-10-27 |
| Passage links highlighted the context line, which is not on the page | live check, 2026-10-27 |
| Job scripts defaulted to the old index (768 dimensions) while the code embeds with bge-m3 (1,024): every question failed | serving job, 2026-10-27; the serving job now stops at start-up on a mismatch |
| The new corpus lacked the 22 restricted test documents | review, 2026-10-27 |
| The crawl chain started the ingest without rebuilding the catalogue | review, 2026-10-27 (script, checked by hand) |
| Renaming a chat: Enter cancelled it (Cancel was the form's first button), and Esc after a rename renamed again | user report, 2026-10-27 |
| An off-topic question got a general-knowledge answer citing an unrelated page | live check, 2026-10-27; grounding guard, tuned on the 400 graded answers so that none of them changes |
| A chat failed with HTTP 500 from its 12th question, or after any question over 6,000 characters: every earlier turn was sent, over the request's 20-turn limit (only the last 6 reach the prompt) | review, 2026-10-27 |
| The serving job's autoscaler never saw the gateway's load, so it scaled in a worker with an answer streaming; the request hung 30 s and failed (FR-10) | `diurnal` load test, job 1820 |
| 14 of 60 Locust sign-ins got 401: the credentials file still named test accounts of the removed ECE and accounts departments (fixed by `seed-locust`; the FR-8 runs carry these failures) | FR-8 runs, 2026-11-03 (data, no code change) |
| The Granite-3.3-8B AWQ build declared `compressed-tensors`, not `awq`, and its newer config was refused by vLLM 0.11; once loaded, it produced gibberish on long RAG prompts (replaced by a GPTQ build) | FR-8 runs, 2026-11-03 |
| Admin policy routes passed the document id unchecked into the access service URL (a ".." id climbed out of `/v1/policy/`, with the admin's own token) | review, 2026-11-03 |
| An institute admin could make a document owned by a department that does not exist, which no department admin could then manage | review, 2026-11-03 |
| The audit file was opened and closed on the NFS drive for every record; the event loop spent 74% of its time there and a 429 at 3x took up to 16 s | FR-6 profile, job 2117, 2026-11-06 |
| `workers.json` and the Prometheus targets were written on the event loop once a second (stalls up to 613 ms); full garbage collections walked the whole index | FR-6 profiles, jobs 2123 and 2126 |
| Answers cached in the 5 s before a revision became visible kept the old version and were served without the stale flag (278 answers, up to 249 s after the old version was retired) | FR-5 replay, job 2143 |
| The scripted revocation user also ran its base class's uncoordinated cycle; when the run ended, 5 restricted test policies stayed deleted (restored by hand through the admin API; the test code now always restores) | FR-3 run, job 2143 |
| The freshness check took replacement times from Locust's publish events, about 4 s before the server commits (0 of 60 as-of checks passed; 60 of 60 with the store's times) | FR-5 replay, job 2143 (test tool) |
| New Locust user classes were not registered for the load shape, so W2 crashed at once | FR-6, job 2126 (test tool) |

## 8. Evidence

- [results/v3/RESULTS.md](results/v3/RESULTS.md): the v3 runs
  (2026-11-06), compiled from raw files; [results/v3/](results/v3/README.md)
  has the run commands.

- [results/fr8/](results/fr8/README.md): the six models side by side
  (2026-11-03).
- [results/live-v2/](results/live-v2/README.md): one page per requirement
  (2026-10-26): workload, command, numbers, a chart with the target, and
  the Grafana panels of that run.
- [results/baseline-v1/](results/baseline-v1/README.md): the first
  single-model round (2026-10-20).
- [results/retrieval-v2/](results/retrieval-v2/README.md): vector-only
  against hybrid search on the held-out set (2026-10-26).
