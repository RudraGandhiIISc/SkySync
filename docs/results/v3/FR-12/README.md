# FR-12: per-request audit record (v3, 2026-11-05)

**Requirement (PRD).** Every request has a record whose document versions
and policy version reproduce the answer, and a search of the audit store
for raw query text finds nothing.

**Verdict: pass** on run 2099/2100. Latencies on this page come from a
GPU shared with another user's job (see "Read with care").

## What changed for this round

Each audit record now also holds:

- `active_workers`: ready GPU workers when the request arrived;
- `cold_start`: true if the request was held while a worker started;
- `latency`: milliseconds per stage (below), with `other_ms` the total
  minus the named stages, never below 0;
- `reason`: a fixed code for a refusal (`invalid_token`,
  `not_in_directory`, `question_too_long`, `rate_limited`,
  `access_check_timeout`, `access_check_failed`).

Every response carries `X-Request-Id`, the id of its audit record, and
Locust writes one line per request it sends (`client_requests.jsonl`:
request id, status, a question id, citations; no text).
`eval/audit_check.py` checks a run against that log.

## Runs

| Item | Value |
|---|---|
| Serving job | 2099, `low_4h_2gpu`, Qwen3-8B, `ASKIISC_MAX_WORKERS=1`, commit 5a382c7 |
| Load | job 2100, workload `steady`: 60 users, 60 s ramp, 10 min steady (no drain stage) |
| GPU | GPU 0; 85,828 of 143,771 MiB free at worker start; utilisation 100%; `gpu_shared: true` by method `processes` (another user's process on the card) |
| Worker held | 666 s (`workers.json`), 0.19 GPU-h |

```bash
sbatch --partition=low_4h_2gpu --time=01:00:00 --export=ALL,ASKIISC_MAX_WORKERS=1 slurm/serve.sh
sbatch --time=00:20:00 slurm/loadtest.sh steady
sbatch --partition=low_2h_1gpu --time=00:10:00 slurm/run.sh uv run python -m eval.audit_check \
  /data4/muthangiv/askiisc/results/serve/2099 \
  /data4/muthangiv/askiisc/results/locust/2100-steady/client_requests.jsonl \
  /data4/muthangiv/askiisc/results/measurements/2099-audit \
  --users /data4/muthangiv/askiisc/private/app/locust_accounts.json
```

## Results

| Check | Target | Result |
|---|---|---|
| Gateway requests with exactly one audit record | all | **2,633 of 2,633**, 0 duplicates |
| Client requests (by `X-Request-Id`) with a record | all | **2,616 of 2,616**; 0 without an id |
| Answer citations found in their record (doc_id, version, chunk_id) | all | **1,940 of 1,940** citations, 1,114 answers |
| Question strings found in the audit store | 0 | **0** of 202 strings searched |
| Test user ids found in the audit store | 0 | **0** |
| `other_ms` p95 (completeness of the breakdown) | at most 50 ms | **20.6 ms** (p50 1.0 ms) |

Decisions: 1,114 answered, 481 abstained, 1,021 rejected (all
`rate_limited`), 17 error. The 17 errors are the 17 gateway records with
no client line (2,633 against 2,616): streams still open when Locust
stopped. In run 2089/2090 the same 17 were checked by time: all were
written 0.2 to 0.8 s after Locust's last request. The `steady` workload
has no drain stage; the PRD-shaped W1 (Phase 5) adds the 60 s drain.

**Latency breakdown** (ms, the 1,595 answered and abstained requests):

| Stage | p50 | p95 |
|---|---|---|
| auth | 0.4 | 0.5 |
| access (gateway and retriever) | 6.2 | 41.8 |
| start (stream waits for the event loop) | 17.5 | 59.1 |
| embed | 114.1 | 909.0 |
| search | 53.7 | 165.7 |
| rerank | 321.9 | 1,405.8 |
| prompt | 0.2 | 0.4 |
| cold hold | 0.0 | 0.0 |
| fair queue | 0.0 | 0.0 |
| first token (dispatch to first token) | 482.8 | 1,885.5 |
| generation | 3,459.7 | 12,480.0 |
| finish (parse, answer cache) | 0.3 | 20.2 |
| output filter | 4.7 | 43.7 |
| deliver (chat app saves the answer) | 20.8 | 116.3 |
| other | 1.0 | 20.6 |

`start_ms` (p95 59.1 ms) is time a ready stream waits for the single
gateway event loop. It is evidence for the saturated loop behind FR-6 and
is carried into Phase 2; nothing was optimised in this phase.

## The first run (2089/2090) and what it found

The same check on run 2089/2090 failed twice
(`results/measurements/2089-audit/summary.json`):

| Finding | Cause | Fix |
|---|---|---|
| 30 of 2,493 client requests (1.2%) had no audit record and no request id | the chat app checks access for a chat's history before calling the gateway; under load that check timed out and the app answered 503 itself (fail closed) | the refusal still fails closed and is now audited through the gateway, with `reason=access_check_timeout`, and the response carries `X-Request-Id` (regression test `test_a_chat_refused_before_the_gateway_is_audited`) |
| `other_ms` p95 1,117 ms | four stages were not timed | added `start`, `prompt`, `finish` and `deliver` |

Run 2099/2100 had no access timeouts (0 HTTP 503), so the audited refusal
path is shown by the regression test, not live. The 1.2% of 503s in run
2090 is above NFR-4's 1% error limit; it is an open item for Phase 8
(STATUS.md).

## Read with care

- **Shared card, one worker.** Another user's job used GPU 0 at 100%
  throughout, and GPUs 1 and 2 had no room, so the run used one worker.
  At 60 users that worker was over capacity: 1,021 of 2,616 client
  requests (39%) got 429. The stage latencies above are not idle-card
  numbers and are not an NFR-1 result.
- **Workload.** `steady` is the repo's existing workload (uniform 5 to
  15 s think time, questions drawn uniformly), not the PRD's W1.

## Source files

- `/data4/muthangiv/askiisc/results/measurements/2099-audit/summary.json`
- `/data4/muthangiv/askiisc/results/serve/2099/audit.jsonl`, `requests.log`, `workers.json`
- `/data4/muthangiv/askiisc/results/locust/2100-steady/client_requests.jsonl`, `summary.json`
- First run: `/data4/muthangiv/askiisc/results/measurements/2089-audit/summary.json`,
  `results/serve/2089/`, `results/locust/2090-steady/`
