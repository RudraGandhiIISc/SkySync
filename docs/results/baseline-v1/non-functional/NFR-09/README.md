# NFR-9: security and privacy

**Requirement (PRD).** 0 successful leaks of unauthorised text; at most 5% prompt-injection success; no raw query text stored; zero traffic leaving the institute network. Workload: a red-team suite of 90 attack cases and a live 30-minute normal-load run.

**Verdict: partial.** The leak, privacy and network checks pass. The 90-case red-team suite is not built yet (Phase 9).

## Results

| Check | Target | Result | Where |
|---|---|---|---|
| Unauthorised text leaked (permission questions where the reader may not see the document: retrieved or cited) | 0 | **0/10** | held-out run, job 1376 (`FR-03`) |
| Unauthorised chunks returned in access probes | 0 | see `../../functional/FR-02/` | |
| Raw question text in the audit store | none | **0/168 questions** | job 1376 (`FR-12`) |
| User ids in the audit store | none | **0** (salted HMAC) | job 1376 |
| Question text in any log, metric or span | none | none | `tests/security/test_privacy_network.py`, `tracing.py` refuses text attributes |
| Prompt injection in a document | no effect | the planted "IGNORE ALL PREVIOUS INSTRUCTIONS" document was answered normally (job 1293); chunk text cannot break out of its `<chunk>` tag | `test_injection_in_a_chunk_stays_data` |
| Services listening beyond 127.0.0.1 | none | none, checked across all sources and the vLLM command line | `test_no_service_binds_beyond_loopback` |
| Secrets or real user ids in the repo | none | none, plus gitleaks: no leaks found | `test_no_secrets_committed`, `test_no_real_user_ids_in_code_or_data` |
| Runs reaching the internet | none | `HF_HUB_OFFLINE=1` in every job | `slurm/*.sh` |
| 90 red-team cases, injection success rate | at most 5% | **not run** (suite not built) | |

**Limits.**
- **No packet capture:** that needs root, so "zero traffic leaving the network" rests on loopback-only binding and offline model loading.
- **No TLS 1.3 yet:** the gateway is not yet served over TLS (Phase 10).
