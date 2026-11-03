# FR-10: scale to zero and hold the first request

**Requirement (PRD).** Release idle nodes after a timeout, and hold the first request after zero instead of failing it. 60 s after the idle timeout no node is active or charged, and in 20 scripted cold starts every held request completes with a visible warming indication.

**Verdict: pass.**

## How it was run

**Run (job 1400, 2026-10-20 00:45):** the full stack on the real index, with the autoscaler driving a ProcessPool (real vLLM processes, at most 1 worker). Idle timeout 20 s, hysteresis 30 s.

**Experiment** (`deploy/experiments.py`, `coldstarts`): 20 cycles. Each cycle waits until the pool has 0 workers, then sends one question through the gateway and records the stream.

```bash
sbatch --export=ALL,ASKIISC_NO_INGEST=1,ASKIISC_EXPERIMENT=coldstarts,ASKIISC_ANSWER_CACHE=0,\
ASKIISC_IDLE_TIMEOUT_S=20,ASKIISC_EXP_COLD_STARTS=20 slurm/run_stack.sh \
  --corpus /data4/muthangiv/askiisc/corpus --state-dir /data4/muthangiv/askiisc/state
```

## Results

| Check | Target | Result |
|---|---|---|
| Held requests that completed | 20/20 | **20/20** (HTTP 200, a cited or refused answer) |
| Requests that showed the `warming` event | all | **20/20** |
| Workers before each request | 0 | 0, so the pool went back to zero 19 times between cycles |
| From the answer to zero workers | within the idle timeout + 60 s | **about 6 s after the answer** (p50 6.0 s, max 6.1 s). That is 31–32 s after the last arrival, so 11 s past the 20 s timeout. The 30 s hysteresis sets this floor |
| GPU charged while at zero | none | no vLLM process alive between cycles (process-group kill) |

**Unit and simulation evidence:**
- `tests/unit/test_autoscaler.py::test_twenty_scripted_cold_starts_all_complete` and `test_scale_to_zero_after_idle_timeout`.
- The flapping regression: a stale arrival-rate window must not restart a worker after scale-in.

Raw data: `/data4/muthangiv/askiisc/results/stack/1400/coldstarts.json`, `workers.json`.
