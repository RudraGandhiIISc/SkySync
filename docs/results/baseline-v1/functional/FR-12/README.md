# FR-12: per-request audit record

**Requirement (PRD).** Every request has a record whose document versions and policy version reproduce the answer, and a search of the audit store for raw query text finds nothing.

**Verdict: pass.**

## Live check (held-out run, job 1376)

All 169 requests of the run (168 questions plus the warm-up) were checked against `audit.jsonl`:

| Check | Result |
|---|---|
| Requests with an audit record | **169/169** |
| Records with document ids and versions, chunk ids and the policy version | **169/169** |
| Questions found verbatim in the audit store | **0/168** |
| Test user ids found in the audit store | **0** (users are stored as salted HMAC-SHA-256 hashes) |
| Record fields | `request_id, time, as_of, user_hash, query_hash, tenant, policy_version, documents, chunk_ids, model, decision, prompt_tokens, completion_tokens, latency_ms, cache_hit, stale` (no text fields exist) |

**Reproduction.** Replaying a question with the record's `as_of` time and the same policy version returns the same chunk ids. This is tested in `tests/unit/test_gateway.py::test_audit_record_reproduces_answer`.

**Other evidence:**
- **Every outcome is audited,** including rejected (401, 413, 429) and denied (403) requests: `test_unknown_user_abstains_and_no_token_is_401` and `test_rate_limit_returns_429_with_retry_after`.
- **No query text in any log:** `tests/security/test_privacy_network.py::test_no_query_text_in_any_log`.

Raw data: `/data4/muthangiv/askiisc/results/stack/1376/audit.jsonl`.
