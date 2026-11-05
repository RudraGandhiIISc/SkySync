# Status

Where Ask IISc stands on 2026-11-05, what was measured, how it differs from
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
- **Quality gate:** green on 2026-11-05 (job 2098): format, lint, mypy --strict, 1,038
  unit and conformance tests (coverage above 97%), integration, e2e,
  security, regression.
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
| FR-3 revocation within 60 s | pass | 3.1 s (live-v2), 7.4 s (2026-10-27) |
| FR-4 access service failover | pass | denies while down, never answers unchecked |
| FR-5 versions and staleness | pass | |
| FR-6 429 within 50 ms at 3x load | **open** | the gateway decides in under 0.5 ms, but one gateway process saturates at 3x load, so clients wait about 1 s (p50) |
| FR-7 batching and prefix cache | pass | |
| FR-8 six models | pass | six models (section 4) through one client, switched by `ASKIISC_MODEL` only; all six measured 2026-11-03 ([results/fr8](results/fr8/README.md)) |
| FR-9, FR-10 scaling, scale to zero | pass | 2 workers in 74 s, never above 2, back to 0 61 s after idle |
| FR-11 fairness | **open** | quiet department 33% (≥ 20% met); busy department 67% (≤ 55% not met) |
| FR-12 audit | pass (2026-11-05) | v3: 2,633 of 2,633 requests audited, 1,940 of 1,940 citations reproduced, 0 question strings in the store; latency breakdown per stage ([results/v3/FR-12](results/v3/FR-12/README.md), run 2099/2100, shared GPU) |
| NFR-1 latency | **open** with 8B | Qwen3-8B, 12 chunks, `steady` 60 users (job 1818, 2026-10-27): p95 total 4.0 s, first token 1.7 s; embedding (p95 0.64 s) and reranking (0.88 s) before the model, vLLM's own first token 0.38 s |
| NFR-2 cold start | pass | Qwen3-8B: worst first token 34.7 s (job 1819) |
| NFR-3 throughput | pass | 5,195 tokens/s at concurrency 64 (4B) |
| NFR-4 search at scale | pass | |
| NFR-5 4-hour soak with a node killed | **open** | not run |
| NFR-6 GPU-hours per 1,000 queries | pass | Qwen3-8B `diurnal`: 0.181 (job 1820) |
| NFR-7 accuracy, citations | **pass** (2026-10-27) | 84.5% and 97.7% on the edition-aware set with Qwen3-8B; the held-out v2 set scored 64.7% with Qwen3-4B on the old corpus |
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

## 6. Next steps

0. **Chat-app 503s under load (open, NFR-4).** In run 2090 (`steady`, one
   worker, shared GPU) 30 of 2,493 client requests (1.2%) got HTTP 503:
   the chat app's access check for the chat history timed out and it
   refused, failing closed. They are now audited
   (`reason=access_check_timeout`), but 1.2% is above NFR-4's 1% error
   limit; to be measured again in Phase 8. Fail-closed behaviour is not
   changed. Source:
   `/data4/muthangiv/askiisc/results/locust/2090-steady/client_requests.jsonl`.
1. **Re-measure NFR-1 and NFR-6** with Qwen3-8B: `sbatch slurm/serve.sh`,
   then `sbatch slurm/loadtest.sh steady` and `diurnal`.
2. **FR-6:** several gateway processes behind one port, with admission
   and the fair queue in shared state (Redis), so refusals do not wait
   behind busy streams.
3. **FR-11:** a workload where the quiet department asks for at least half
   the capacity, or a token-level quota per department.
4. **NFR-5:** `sbatch --partition=med_12h_2gpu --time=05:00:00
   --export=ALL,ASKIISC_CHAOS=1 slurm/serve.sh`, then the `soak` workload
   (about 4 GPU-hours).
5. **FR-8 speed on an idle card:** the 2026-11-03 runs shared GPU 0 with
   another user's job at 100%, so their p95 and tokens/s understate the
   models; repeat `slurm/fr8_bench.sh` for the five models when a card is
   free (about 20 GPU-minutes each).
6. **People:** review the question sets; settle the judge's disagreements;
   real department admins for CSA and CDS (the directory now holds only
   CSA, CDS and the institute); server accounts for teammates (SSH tunnel).
   E-mail for registration and password resets goes through a Gmail
   account with an app password (`private/app/smtp.json`); reset e-mails
   were tested live on 2026-11-02.
7. **Release:** run the whole gate, commit (the working repository is
   committed once, at deployment), and push the code-only copy once the
   GitHub address and a deploy key exist. Real users, accounts, chats and
   the corpus stay on `/data4` and are never committed.

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

## 8. Evidence

- [results/fr8/](results/fr8/README.md): the six models side by side
  (2026-11-03).
- [results/live-v2/](results/live-v2/README.md): one page per requirement
  (2026-10-26): workload, command, numbers, a chart with the target, and
  the Grafana panels of that run.
- [results/baseline-v1/](results/baseline-v1/README.md): the first
  single-model round (2026-10-20).
- [results/retrieval-v2/](results/retrieval-v2/README.md): vector-only
  against hybrid search on the held-out set (2026-10-26).
