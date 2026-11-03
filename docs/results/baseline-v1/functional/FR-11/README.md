# FR-11: fairness between tenants

**Requirement (PRD).** When one tenant sends 10x the load of another, it gets at most 55% of tokens and the other at least 20%; with the other tenant idle, it runs at full speed.

**Verdict: pass in simulation. Not yet measured on the GPU** (that is the E4 fairness run, deferred together with FR-6).

## Method

The gateway hands out LLM slots by deficit round robin over tenant queues (`src/askiisc/gateway/drr.py`), with each request's cost set to its estimated tokens. The simulation (`tests/unit/test_limits_drr.py::_served_shares`) feeds one LLM slot at 5 requests per second of capacity, with 200-token requests, for 600 simulated seconds.

Rerun: `PYTHONPATH=. uv run python tools/fr11_shares.py` (inside `slurm/run.sh`).

## Results

| Scenario | Target | Token share |
|---|---|---|
| Heavy tenant 30 req/s vs light 3 req/s (10:1, both over their fair share) | heavy at most 55%, light at least 20% | heavy **50.0%**, light **50.0%** |
| Three tenants 40 : 4 : 4 req/s | no tenant over 55% | **33.4% / 33.3% / 33.3%** |
| Heavy tenant alone | full speed | **100%** |

**Also tested:** no backlogged tenant waits more than one round (property test, 60 random cases); cancelled waiters never hold a slot; the slot count follows the number of ready workers (`test_dispatcher_resize_follows_workers`).

Raw data: `/data4/muthangiv/askiisc/results/evidence/fr11.txt`.
