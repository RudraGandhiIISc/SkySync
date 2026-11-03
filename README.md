# Ask IISc

Question answering over IISc documents, with citations, per-user access
control, every edition of yearly documents kept, and GPU workers that scale
with demand. DS252 Cloud Computing, team SkySync.

It runs on the CSA `automation` server through Slurm (3x H200, no
containers, no root). The corpus, index, users, accounts and model weights
live in `/data4/muthangiv/askiisc` on the server, never in this repository.

## Start

```bash
sbatch slurm/serve.sh          # chat app, admin dashboard, Grafana, Prometheus, Locust; Qwen3-8B
ssh -N -L 18000:localhost:18000 -L 13000:localhost:13000 \
       -L 19090:localhost:19090 -L 18089:localhost:18089 <you>@automation.csa.iisc.ac.in
```

Then open http://localhost:18000 (chat) and http://localhost:18000/admin.
Putting it into service (setup, secrets, release checklist, update,
rollback, backups): [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). The corpus
pipeline and everything else: [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Check

```bash
sbatch --wait --partition=low_2h_1gpu slurm/gate.sh all test-int test-e2e test-sec test-regress
```

## Documentation

| Document | For |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it works: the request path, corpus and editions, access control, web app, code map |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | deploying, updating, rolling back and backing up the serving job |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | setup, building the corpus and index, serving, load tests, the quality gate, monitoring, troubleshooting |
| [docs/DATA_AND_EVALUATION.md](docs/DATA_AND_EVALUATION.md) | data sets, question sets, how answers are graded |
| [docs/STATUS.md](docs/STATUS.md) | results, requirement status, deviations from the PRD, next steps |
| [docs/DEMO.md](docs/DEMO.md) | a scripted live demo |
| [docs/security.md](docs/security.md) | security controls and the tests that hold them |
| [docs/PRD.md](docs/PRD.md) | the requirements |
| [docs/results/](docs/results/) | measured evidence per requirement |
| [CLAUDE.md](CLAUDE.md) | rules for working in this repository |

## Layout

```
src/askiisc/  common, ingest, access, retrieval, llm, gateway, autoscaler, observability, web
deploy/       serve.py (serving job), ingest.py (batch ingest), stack.py (shared helpers), grafana/
eval/         locust/, versioned/, heldout/, grader/, oracle/, probes/, datasets/
slurm/        job scripts
tests/        unit, conformance, integration, e2e, security, regression, perf
tools/        coverage check, UI screenshots, evidence pages
```
