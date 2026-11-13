# Status

Where Ask IISc stands on 2026-11-13, what was measured, how it differs from
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
- **Red-team baseline (2026-11-12 and 13):** the vLLM workers and the
  embedding server need the serving job's key; a watchdog restarts the
  access and embedding servers; a full card ends a held question at once
  with "GPUs full"; two FR-5 bugs and an FR-11 admission bug were found by
  the runs and fixed. Results: [results/rt-baseline/](results/rt-baseline/README.md).
- **Quality gate:** green on 2026-11-13 (job 2321: format, lint, mypy
  --strict, 1,186 unit and conformance tests, coverage 97.4%,
  integration, e2e, security with no known vulnerability in the lock
  file, regression, pre-commit hooks).
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
[results/live-v2/](results/live-v2/README.md)) unless noted; the runs of
2026-11-12 and 13 are in [results/rt-baseline/](results/rt-baseline/README.md).
The admin **Requirements** tab always shows the newest measurement.

| Req | State | Notes |
|---|---|---|
| FR-1 cited answers, refusals | pass | Edition-aware set, Qwen3-8B (2026-10-27): 97.7% of citations support their answer (target 95%); 31 of 40 unanswerable questions refused, 77.5% (target 75%). On held-out v2 (job 2047) 20 of 30, partly because the larger corpus now answers some of its questions ([DATA_AND_EVALUATION](DATA_AND_EVALUATION.md) section 3) |
| FR-2 no forbidden chunk retrieved | pass | recall@8 1.000, 0 leaks in 10,000 probes; 0 leaks in the 2026-10-27 live run |
| FR-3 revocation within 60 s | pass | 77 scripted revocations (job 2234, Qwen3-8B-AWQ): the last citation after revoking came at most 8.9 s later, none after 60 s. The run stopped at its 400-attempt limit: in 323 attempts the reader was not cited the document before the revoke, so there was nothing to revoke. Earlier: at most 6.8 s (live-v2, 7 revocations), 7.4 s (2026-10-27) |
| FR-4 access service failover | pass | Denies while down, never answers unchecked (live-v2). On the release code (job 2274) the watchdog restarted a killed access service in 1.6 s and a question was answered 2.7 s after the kill; a killed embedding server was back in 2.6 s and answering at 7.2 s |
| FR-5 versions and staleness | pass | Replay of 60 revisions over 20 documents on the release code (job 2259): 0 of 2,992 answers used a superseded version without the stale flag (audit store and citations); demotion p50 7.6 s, p95 34.3 s, max 63.9 s (15 minutes allowed); as-of correct at 60 of 60 checkpoints. The run before the fixes (job 2234) found the two bugs in section 7 |
| FR-6 429 within 50 ms at 3x load | **pass** (2026-11-06) | W2 at 3x, median of 3 (job 2134): 429 p99 24.3 ms at the serving node, accepted p95 4.1% lower than at 1x. Client 429 p99 72 to 90 ms. With admission off nothing collapsed at this load (82% answer-cache hits), so E4's collapse is not shown ([results/v3](results/v3/RESULTS.md)) |
| FR-7 batching and prefix cache | pass | Continuous batching on a quiet card (job 2238): 1,549 output tokens/s at concurrency 64 against 144 at 1 for Qwen3-8B (10.8x), 881 against 30 for Qwen3-8B-AWQ (29x). Prefix cache, W1 with the answer cache off (job 2143): hit rate 65.9%; p95 first token at the model server 38.9% lower (935 against 1,529 ms); at the client 7.4% lower, since retrieval comes first |
| FR-8 six models | pass | Six models through one client, switched by `ASKIISC_MODEL` only; all six measured 2026-11-03 ([results/fr8](results/fr8/README.md)). KV room at one 24 GiB budget: Qwen3-8B-AWQ 65,920 and 89,584 tokens against Qwen3-8B's 19,984 (at least 3.3x, job 2238); Granite-3.3-8B-GPTQ 114,144 against 35,088 (3.25x, job 2234) |
| FR-9, FR-10 scaling, scale to zero | pass | 2 workers in 74 s, never above 2, back to 0 61 s after idle (live-v2). 20 scripted cold starts each: Qwen3-8B on the release code 20 of 20 answered (job 2259); Qwen3-4B 20 of 20 and Qwen3-8B-AWQ 19 of 20 (job 2234), the one miss a worker killed by CUDA out-of-memory when another job filled the card during its start-up. Since 2026-11-12 a full card ends a held question at once with "GPUs full" and a retry time, out-of-memory deaths included |
| FR-11 fairness | pass | Both departments backlogged (job 2314: 80 busy and 8 quiet users asking back to back, 6 LLM slots, answer cache off): busy department 53.7% of tokens (target at most 55%), quiet 46.3% (at least 20%). Job 2259 found an admission bug that held the quiet department to 11% (section 7); with 4 quiet users (job 2274, busy 61.4%) the quiet department did not keep more requests waiting than its share, so the run did not test FR-11 |
| FR-12 audit | pass (2026-11-05) | v3: 2,633 of 2,633 requests audited, 1,940 of 1,940 citations reproduced, 0 question strings in the store; latency breakdown per stage ([results/v3/FR-12](results/v3/FR-12/README.md), run 2099/2100, shared GPU) |
| NFR-1 latency | pass | Qwen3-8B, `steady` 60 users, up to 2 workers, cards at 0% use when the job started (job 2314, median of 3): p95 total 3.9 s at the client (4.9, 3.9, 3.9; target 4.0 s); p95 first token 457 ms at the model server (target 800 ms), 2.0 s at the client, where embedding and reranking come first |
| NFR-2 cold start | pass | Slowest first token of 20 scripted cold starts: Qwen3-4B 57.4 s (target 90 s), Qwen3-8B-AWQ 31.4 s (90 s), Qwen3-8B 27.9 s (150 s); medians 25.3, 29.8 and 27.2 s (jobs 2234, 2259) |
| NFR-3 throughput | pass | On a quiet card at concurrency 64 (job 2238): Qwen3-8B 1,549 output tokens/s (target 250), Qwen3-8B-AWQ 881 (450); Qwen3-4B 5,195 (600; baseline-v1 on an idle card). Shared-card numbers of the other models are on the FR-8 page |
| NFR-4 scale | pass | 2 workers in 74 s, never above 2, back to 0 within the idle timeout + 60 s; search p95 62 ms under load (live-v2). Growth to 6,000 users and 1M chunks is the stretch run E6 (PRD section 8). 60 users: 0.03% errors (job 2314, median of 3); 2.5% of questions got the per-department 429 (FR-6), which counts as shed, not as an error (PRD section 7) |
| NFR-5 availability | pass | 1 hour at half load with Qwen3-8B (job 2314): 1 of 18,892 requests failed (99.99% success; target 99%); a worker killed at 30 minutes was replaced and ready 26.1 s later (target 180 s). 162 questions got the per-department 429, counted as shed |
| NFR-6 GPU-hours per 1,000 queries | pass | Qwen3-8B `diurnal`: 0.181 (job 1820); all six models reported on the FR-8 page |
| NFR-7 accuracy, citations | **pass** (2026-10-27) | 84.5% and 97.7% on the edition-aware set with Qwen3-8B; the held-out v2 set scored 64.7% with Qwen3-4B on the old corpus. Review sheet for people: 111 rubric/judge disagreements plus a 60-item sample (job 2115, [results/v3](results/v3/README.md)) |
| NFR-8 freshness | pass | |
| NFR-9 security | pass | 0 leaks; 1 of 69 injections succeeded (≤ 5%). On the release code (job 2274) a vLLM worker answers 401 to `/v1/chat/completions` and `/invocations` without the serving job's key, and the embedding server 401 to `/embed` and `/rerank`. The project's lock has no known advisory; the GPU-side ones, and why none can be reached, are in [SECURITY.md](../SECURITY.md) |
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

## 6. What is left (2026-11-13)

The runs that the cancelled jobs of 2026-11-06 left were done on
2026-11-12 and 13 (jobs 2234, 2238, 2259, 2274 and 2314, [results/rt-baseline](results/rt-baseline/README.md)).
Still open, none of them a requirement of the PRD:

1. **E3 idle-timeout sweep** (T_idle 0, 60, 300, 900 s and none, W3 at
   24x, 3 models): about 15 GPU-hours.
2. **E0 for the other four models on a quiet card**, and why Qwen3-4B's
   throughput at concurrency 64 was 5,195 tokens/s in baseline-v1 and
   2,660 in live-v2.
3. **E2 for every model** and **NFR-6 three ways** (rated load, goodput,
   W3 replay; per 1,000 correct answers).
4. **E4 collapse:** W2 at 3x did not saturate the GPU (82% answer-cache
   hits). Showing the collapse needs W2 with the answer cache off, or more
   load.
5. **FR-3, all 100 revocations:** the run stops after 400 attempts, and in
   323 of them the reader was not cited the document before the revoke.
   A permission item list with questions the reader is reliably cited for
   would complete the 100.
6. **vLLM upgrade** to a release without the 93 advisories of 0.11.0
   (0.32 or later), with torch and transformers, and every measurement
   repeated ([SECURITY.md](../SECURITY.md)).
7. **Decisions for people:** fill in the NFR-7 review sheet
   (`/data4/muthangiv/askiisc/results/adjudication/qwen3-8b-p2/review.csv`,
   171 rows); review the question sets; real department admins; server
   accounts for teammates.

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
| An answer whose retrieval began before a revision became visible was cached after the generation moved, and served unflagged to later askers (171 answers, up to 107 s after the old version ended); such an answer is no longer cached | FR-5 replay, job 2234 |
| A job killed between recording a revision and retiring the old chunks (job 2163) left the superseded version searchable for six days (2,903 unflagged answers in one replay); the serving job now retires at start-up every version the version store has ended | FR-5 replay, job 2234 |
| The vLLM workers and the embedding server answered any user of the server on their loopback ports, bypassing sign-in and admission; vLLM's own `--api-key` would still leave `/invocations` open. Both now need the serving job's key on every path but health and counters | review, 2026-11-12 |
| A killed access service or embedding server was never restarted, so every question was denied or failed until the job restarted; a watchdog now restarts it on the same port and key | review, 2026-11-12 |
| With every card full, a held question waited the whole 150 s hold and then failed with a bare error; it now ends at once with a "GPUs full" message and a retry time | cold-start run, job 2143 |
| Commands run outside a job fell back to the author's data folder | review, 2026-11-12 |
| Admission's wait estimate, which assumes first come first served, refused a department below its fair share; with 6 LLM slots the quiet department got 11% of the tokens. A department below its share now skips that check while others are active (at the default 64 slots the estimate stays 0) | FR-11 run, job 2259 |
| A worker that died with "CUDA out of memory" because another job filled the card during its start-up was treated as a broken worker, so the held question waited out the hold; it now counts as full GPUs | cold-start run, job 2234 |
| The FR-8 KV check of the Qwen pair ran on a card where another user's job grew during vLLM's memory profiling, which vLLM counts against the KV budget (the 4-bit model seemed to hold fewer tokens than its parent); repeated on a quiet card | FR-8 run, job 2234 (test setup) |
| New Locust user classes were not registered for the load shape, so W2 crashed at once | FR-6, job 2126 (test tool) |

## 8. Evidence

- [results/rt-baseline/](results/rt-baseline/README.md): the runs behind
  the red-team baseline (2026-11-12 and 13), compiled from raw files.
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
