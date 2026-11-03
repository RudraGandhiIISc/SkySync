# NFR-6: cost (GPU-hours per 1,000 queries)

**Requirement (PRD).** At most 0.30 GPU-hours per 1,000 completed queries, including start-up and idle time. Workload: normal load, plus a 24-hour replay of campus traffic, for 5 idle timeouts.

**Status: partly measured.** The 24-hour replay and its idle-timeout sweep (about 5 GPU-hours) were deferred (agreed on 2026-10-19). The cost model is built and tested, and it is applied below to the live runs of this round.

## Method

`src/askiisc/observability/cost.py`: for each worker, a (acquire) + l (load) + b (serving) + d (idle but held) seconds are charged, because the department bills occupancy, not use. G = sum / 3600, and C = 1000 x G / completed queries. The worker lifecycle comes from `workers.json` of each run.

## Results

See the table in this folder's `results.md`, filled from the FR-9 scale run and the NFR-1 W1 run.

## Unit evidence

`tests/unit/test_pool_observability.py`: a hand-computed example, a night of pure idle time (charged in full), and a property test that cost is never negative and grows with idle time.
