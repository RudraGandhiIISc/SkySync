# Security controls (NFR-9, Phase 10)

Each control, where it lives, and the test that holds it. Tests run in the gate (`sbatch --wait slurm/gate.sh all test-sec`); the red-team suite and the recall probe run against the real stack. NFR-9 targets: 0 leaks of unauthorised text, at most 5% prompt-injection success, no raw query text stored, no traffic leaving the institute network.

## Access to documents

| Control | Where | Test |
|---|---|---|
| Access decisions in a separate service; any error, timeout or outage denies (fails closed) | `access/service.py`, `access/client.py` | `test_access_service.py::test_any_failure_denies`, `::test_unreachable_service_denies_after_one_retry`, `test_services.py::test_killing_the_access_service_denies_everything`, `test_gateway.py::test_access_service_down_denies` |
| User identity only from the verified token, never from the request body | `access/service.py` | `test_access_service.py::test_user_id_never_comes_from_the_body`, `::test_bad_tokens_get_401`, `::test_expired_token_gets_401` |
| Permission and validity filters inside the search, never after it | `retrieval/dense.py` | `test_dense.py`; FR-2 recall probe (`eval/probes/recall.py`): 10,000 searches of the serving index, every result replayed through the independent oracle |
| Cited documents re-checked before an answer leaves; a revocation during generation withholds the answer | `gateway/output_filter.py` | `test_gateway.py::test_revocation_during_generation_is_withheld`, `test_web.py::test_answers_citing_revoked_documents_are_hidden` |
| Revocation effective within 60 s | `access/client.py` cache lifetime | `test_access_service.py::test_revocation_takes_effect_within_sixty_seconds`; Locust `access` (`RevocationUser`) |
| Answer cache keyed by the permission set, so users with different rights never share an answer | `gateway/cache.py` | `test_gateway.py::test_cache_hit_and_isolation_between_permission_sets` |
| Department admins change only their own department's documents and non-admin people | `access/service.py`, `web/people_admin.py` | `test_access_service.py::test_department_admins_manage_only_their_documents`, `test_people.py` |
| Documents are owned only by real departments; an admin route never forwards a string that is not a document id (no ".." into the access service URL) | `web/documents.py` `unknown_owners`, `web/admin.py` `DOC_REF` | `test_documents.py::test_owners_must_be_real_departments`, `test_web.py::test_policies_revisions_and_chaos_through_the_admin_api` |

## Prompt injection

| Control | Where | Test |
|---|---|---|
| Retrieved chunks are wrapped in `<chunk>` tags, tag characters escaped, and marked as untrusted data in a fixed system prompt | `llm/prompts.py` | `test_privacy_network.py::test_injection_in_a_chunk_stays_data` |
| Only citations of chunks retrieved for the request survive | `llm/rag.py` | `test_rag.py` |
| Output guards: an answer citing no retrieved chunk becomes the refusal (this stops "reply only with X" injections in the question); an answer repeating six words in a row of the system prompt becomes the refusal; so does an answer whose content words are mostly absent from its cited chunks (outside knowledge); links not found in a cited chunk and HTML tags are removed | `llm/rag.py` `parse_answer`, `scrub`, `echoes_system_prompt`, `grounded_share` | `test_rag.py::test_output_guards`, `::test_answers_from_outside_knowledge_become_the_refusal`; red-team run 1573 (before): 10 of 70 injections succeeded; after the guards: 1 of 69 |
| The question is untrusted too: rule 6 of the system prompt says to ignore requests in it to change the rules, role or output format, repeat words, add links or use outside knowledge | `llm/prompts.py` | golden `tests/regression/goldens/prompts.json` |
| The chat page escapes model text and adds only basic formatting (bold, lists, code) and its own source links: no images, no model-written links, no HTML | `static/chat.js` | red-team `exfiltration` cases |
| 90-case red-team suite: instruction override, prompt extraction, instructions hidden in planted documents, exfiltration links, other users' documents | `eval/redteam.py`, `eval/datasets/redteam/` | `test_redteam.py` (schema, verdicts); live: `sbatch slurm/loadtest.sh redteam` → scorecard NFR-9 |

## Accounts and sessions

| Control | Where | Test |
|---|---|---|
| Only allowlisted people can create an account, and only by opening a single-use link (10 minutes) e-mailed to that address; the password is chosen on the form and kept as an Argon2id hash in memory until then | `web/registration.py`, `web/pending.py` | `test_auth.py::test_registration_is_confirmed_by_an_e_mailed_link`, `::test_only_allowlisted_people_can_register`, `::test_confirmation_links_expire_and_are_limited`, `::test_access_is_checked_again_when_confirming` |
| A real person needs an active allowlist entry, a directory record, an active account and a password; never admitted by e-mail domain | `web/app.py` `_login`, `may_sign_in` | `test_auth.py::test_an_approved_active_person_signs_in`, `::test_every_refusal_looks_the_same` |
| One generic answer for every refusal (wrong password, unknown address, not allowed, suspended, disabled, no password yet); unknown addresses cost one Argon2 check too | `web/app.py`, `web/passwords.py` `burn` | `test_auth.py::test_every_refusal_looks_the_same`, `::test_unknown_ids_cost_a_verification_too` |
| Passwords hashed with Argon2id (argon2-cffi defaults, RFC 9106); 8 to 256 characters, not the e-mail address; old scrypt hashes replaced at the next sign-in | `web/passwords.py` | `test_web.py::test_passwords`, `test_auth.py::test_old_scrypt_passwords_work_once_then_become_argon2` |
| Guessing slowed per typed address (5 failures in 15 min, then 30 s doubling to 15 min, HTTP 429 with `Retry-After`) and across all addresses (60 failures a minute) | `web/throttle.py` | `test_web.py::test_repeated_wrong_passwords_are_slowed_down`, `test_auth.py::test_spraying_many_addresses_slows_everyone` |
| Reset links (10 minutes; admins send none): 32 random bytes, only the SHA-256 stored, single use, a new link voids older ones, using one ends every session; sent only to the person's own address; never logged | `web/password_links.py`, `web/store.py` | `test_auth.py::test_forgot_password_mails_a_ten_minute_link`, `::test_links_expire_and_newer_ones_void_older`, `::test_adding_sends_nothing_and_the_person_registers` |
| Only addresses of the allowed domains (`ASKIISC_EMAIL_DOMAINS`, default `iisc.ac.in`) can be added; adding sends nothing, and an account exists only once its address has received and opened a confirmation link | `web/people_admin.py` `domain_error`, `web/registration.py` | `test_auth.py::test_only_allowed_domains_can_be_added`, `::test_adding_sends_nothing_and_the_person_registers` |
| "Forgot password?" sends only to an account that may sign in; at most 3 an hour per person; refused addresses are slowed like sign-in | `web/password_links.py` | `test_auth.py::test_forgot_password_mails_a_ten_minute_link`, `::test_forgot_password_limits_and_mail_failures`, `::test_forgot_without_mail_says_to_ask_the_admin` |
| Session cookie random, stored hashed, HttpOnly, SameSite=Strict (`Secure` with `ASKIISC_SECURE_COOKIES`), 12 h; replaced at every sign-in (no fixation); ended by sign-out, password change, suspension or a change of department or role; every request re-checks the account and the allowlist | `web/app.py`, `web/store.py` | `test_auth.py::test_signing_in_replaces_any_presented_session`, `::test_logout_ends_the_session_on_the_server`, `::test_rights_changes_and_suspension_end_sessions` |
| Every state-changing call needs the `X-AskIISc` header and a same-site `Origin` (CSRF) | `web/app.py` `same_site` | `test_auth.py::test_state_changes_need_the_header_and_this_origin` |
| Chats, admin pages and the API need a session; nobody reads, renames, rates or exports another person's chat; nobody makes themselves an admin | `web/chat.py`, `web/admin.py`, `web/people_admin.py` | `test_auth.py::test_anonymous_requests_are_refused`, `::test_nobody_reads_another_persons_chats`, `::test_users_cannot_make_themselves_admins` |
| Test identities (ids without "@") only while `ASKIISC_TEST_LOGINS` is on | `web/app.py` `may_sign_in` | `test_auth.py::test_test_accounts_need_test_logins` |
| The SMTP app password only in a mode-600 file, never in git, environment, logs or `repr`; an incomplete file turns mail off and names only the missing fields | `web/mailer.py`, `deploy/serve.py` | `test_auth.py::test_incomplete_mail_settings_turn_mail_off`, `::test_smtp_mail_uses_tls_and_signs_in`; gitleaks in the gate |
| Content-Security-Policy (this origin only, no inline scripts), no framing, nosniff, no referrer | `web/app.py` `SecurityHeaders` | `test_web.py::test_pages_are_served` |
| Prometheus, Grafana, Loki and Locust each need a password | `deploy/serve.py`, `observability/authproxy.py` | `test_web.py::test_prometheus_basic_auth_file`, `test_request_logs.py::test_auth_proxy` |

## Privacy

| Control | Where | Test |
|---|---|---|
| No question or answer text in any log, metric or audit record; users and queries stored as salted SHA-256 hashes | `gateway/audit.py`, `common/requestid.py` | `test_privacy_network.py::test_no_query_text_in_any_log`, `test_gateway.py::test_audit_never_stores_raw_query`, `::test_metrics_endpoint_has_no_query_text`, `test_request_logs.py::test_vllm_logs_request_ids_without_prompts` |
| Stored chat text encrypted (Fernet key, mode 600) | `web/store.py` | `test_web.py::test_store_encrypts_chat_text` |
| Temporary chats never stored | `web/chat.py` | `test_web.py::test_temporary_chats_are_never_stored` |
| Real users only in the private users tree on `/data4`, never in the repository | `slurm/lib.sh` | `test_privacy_network.py::test_no_real_user_ids_in_code_or_data` |

## Processes, files and network

| Control | Where | Test |
|---|---|---|
| Every service binds to 127.0.0.1 and dies with its Slurm job | `deploy/serve.py`, `autoscaler/pool.py` | `test_privacy_network.py::test_no_service_binds_beyond_loopback`, `::test_vllm_workers_listen_on_loopback` |
| Jobs run with umask 077; secrets in mode-600 files under `/data4/<user>/askiisc/private` (folder mode 700) | `slurm/lib.sh`, `deploy/serve.py` | `test_request_logs.py::test_certificates_are_made_once_and_private` |
| Models load offline (`HF_HUB_OFFLINE=1`); nothing is fetched while serving | `deploy/stack.py`, `slurm/*.sh` | reviewed; weights fetched beforehand by `slurm/fetch_models.sh` |
| Request limits: question at most 512 tokens (413), body fields bounded, per-user rate limit (429) | `gateway/app.py`, `gateway/limits.py` | `test_gateway.py::test_question_too_long_is_413`, `::test_rate_limit_returns_429_with_retry_after` |
| Static analysis, secret scan and dependency audit | `Makefile` (`test-sec`, `audit`) | bandit, gitleaks, pip-audit in the gate; Dependabot (`.github/dependabot.yml`) once the repository is on GitHub |

## Known limits

- **No TLS on the gateway.** It listens on 127.0.0.1 and is reached only through an SSH tunnel, which encrypts the traffic. If it is ever exposed otherwise, put it behind TLS 1.3 and set `secure_cookies`.
- **No packet capture.** Proving that no traffic leaves the network needs root. The evidence instead is the loopback-only socket audit plus offline model loading.
- **A mistyped address on the allowlist is not detected.** It does no harm (nobody can confirm it), and the real person is told they are not added; the People tab keeps showing it as *Not registered yet*. A mistyped address that belongs to someone else at IISc would let that person register.
- **Create account and Forgot password tell whether an address is allowed.** Chosen for clarity; refused addresses are slowed by the sign-in limits (6 free tries per address, 60 refusals a minute in all), so listing the allowlist is slow but possible.
- **Passwords only, no second factor.** Sign-in rests on the password and the single-use links; the links are as safe as the person's mailbox.
- **E-mail goes through a personal Gmail account** (an app password in `private/app/smtp.json`). Google's sending limits apply, and the sender shown is that account.
- **Sign-in limits are in memory.** A restart of the serving job clears them.
- **Test accounts have passwords.** Only ids without "@" (created from the terminal), only while `ASKIISC_TEST_LOGINS` is on; switch it off for serving jobs without load tests.
- **The red-team verdicts are string checks** (canaries, system-prompt phrases, attacker URLs, phrases of forbidden documents). They are strict, since an answer that only quotes the attack counts as a success; a person reviews any successes in `answers.json`.
- **Grounding is checked by words, not meaning.** An answer whose content words are almost all missing from its cited chunks (under 30%) becomes the refusal; a plausible wrong answer built from the cited text passes. No restricted text is involved either way: citations are always documents the user may read.
- **Tokens stream before the guards run.** The guards judge the final answer; a refused answer is replaced in the chat when the stream ends, but its tokens were shown while it streamed. Cited content is never affected, since forbidden chunks are never retrieved.
