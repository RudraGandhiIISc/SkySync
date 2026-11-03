# FR-4: access control as a separate, reusable service

**Requirement (PRD).** Provide access control as a separate and reusable service. It passes its conformance tests, and with the service killed nothing is answered without an access check.

**Verdict: pass.** One deployment finding: run it as its own process.

| Check | Result | Where |
|---|---|---|
| A conformance suite that any `AccessAPI` implementation must pass (matrix, filter, as-of time, revocation, bad tokens) | passes for both the in-process and the HTTP implementation | `tests/conformance/test_access_api.py` |
| Service killed mid-run: every later `filter` and `authorize` call is denied (20 tries each) | all denied | `tests/integration/test_services.py::test_killing_the_access_service_denies_everything` (real HTTP server) |
| Gateway with the access service down: HTTP 403, audit decision `denied` | yes | `tests/unit/test_gateway.py::test_access_service_down_denies`; e2e: `tests/e2e/test_stack_fake.py` |
| Access service down mid-request: error event, nothing answered | yes | `test_mid_stream_access_failure_is_error_event` |
| The same service is reused by the gateway, the output filter, the ingest checks and the probes | yes | `src/askiisc/access/` |

**Deployment finding (FR-9 and NFR-1 runs).** In `deploy/stack.py` the access service runs in the gateway's process and event loop. Under heavy load the 50 ms client timeout expired, and requests were refused (fail-closed: correct, but too many refusals). Running the service as its own process, with the timeout set for that deployment, is the fix (`NEXT_STEPS.md`).
