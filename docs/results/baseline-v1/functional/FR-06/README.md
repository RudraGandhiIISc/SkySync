# FR-6: load limits and HTTP 429

**Requirement (PRD).** Limit load per tenant and reject requests that would wait too long. At 3x load, excess requests get HTTP 429 within 50 ms, and the p95 latency of accepted requests rises by at most 20%.

**Status: deferred** (agreed on 2026-10-19). The live 1x/2x/3x overload runs (E4, about 3.6 GPU-hours with repetitions) were not run in this round.

## Evidence so far (no GPU)

| Check | Result | Where |
|---|---|---|
| The token bucket never admits more than rate x time + burst | property test, 50 random cases | `tests/unit/test_limits_drr.py::test_bucket_never_exceeds_rate_times_time_plus_burst` |
| Over the rate, over the queue bound, or over the wait estimate, the request is rejected with Retry-After | unit tests | `test_admission_*` |
| Admission is synchronous, so a 429 never waits on the LLM | `Admission.admit` is not a coroutine | `test_rejection_does_not_await` |
| The gateway returns 429 with `Retry-After`; other tenants are unaffected | full gateway on the fake LLM | `tests/unit/test_gateway.py::test_rate_limit_returns_429_with_retry_after` |
| A 429 burst over real HTTP | end-to-end on the fake stack | `tests/e2e/test_stack_fake.py` |

**Not measured yet:** the 50 ms rejection time and the 20% p95 bound at 3x real load.

**To run it:** add a `load` run at 3x the W1 rate with the default limits (no `ASKIISC_TENANT_*` overrides), and one with limits off, then compare the p95 of accepted requests.
