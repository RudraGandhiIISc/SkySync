# NFR-5: availability

**Requirement (PRD).** At least 99% non-error responses; capacity restored within 180 s after a node failure. Workload: a 4-hour test at half normal load, with one node killed at the 2-hour mark.

**Status: deferred** (agreed on 2026-10-19). The 4-hour soak (E5) was not run. It would also need two chained 2-hour jobs, because of the partition limits.

## Evidence so far

| Check | Result | Where |
|---|---|---|
| A worker killed under traffic is replaced within 180 s (simulated: 5 s acquire + 30 s load) | passes; replaced well inside 180 s | `tests/unit/test_autoscaler.py::test_worker_failure_is_replaced_within_180_s` |
| A failed start is counted and retried | passes | `test_failed_start_is_counted_and_retried` |
| Error rate in the live runs of this round | see NFR-1 (W1) and FR-9 (scale) | `../NFR-01/`, `../../functional/FR-09/` |
