# Ask IISc

Question answering over IISc documents, with citations, per-user access
control, every edition of yearly documents kept, and GPU workers that scale
with demand. DS252 Cloud Computing, team SkySync.

It runs on the CSA `automation` server through Slurm (3x H200, no
containers, no root). The corpus, index, users, accounts and model weights
live in `/data4/muthangiv/askiisc` on the server, never in this repository.

## Try it without a GPU (5 minutes)

```bash
uv sync --locked
uv run python -m tools.ui_preview 18000
```

Open http://127.0.0.1:18000 and sign in as `u_cds_stu` (CDS student),
`u_csa_stu` (CSA student, sees different documents), `u_cds_admin`
(department admin) or `u_admin` (institute admin), password
`Correct-horse-9`. This runs the real web app,
gateway API and access control on a simulated model, sample documents and
test users, so you can try the UI, the API and who-sees-what without the
cluster. Answers come from the simulated model, not a real LLM.
`Correct-horse-9` is the password of these built-in test users of the
preview only (`tests/web_fixtures.py`, in memory); it opens nothing on a
real deployment, where people choose their own passwords and load-test
accounts get a random one.

## Team SkySync

| Member | E-mail |
|---|---|
| Rudra Kumar Gandhi | rudragandhi@iisc.ac.in |
| Muthangi Vighnesh | muthangiv@iisc.ac.in |
| Mohammad Haris | harism@iisc.ac.in |

Requirements: [PRD.md](PRD.md). Developer guide (what each part of the
code does): [dev.md](dev.md). Security policy, and how to report a finding
privately: [SECURITY.md](SECURITY.md).

## Install

Needs a Linux Slurm cluster with NVIDIA GPUs (built for one node with
3x H200; no Docker, Kubernetes or root), `git`, and
[`uv`](https://docs.astral.sh/uv/) on the PATH. `uv` installs Python 3.11
(app) and 3.12 (vLLM). Run every `sbatch` from the repository root; paths
are set in `slurm/lib.sh` (`ASKIISC_DATA`, `ASKIISC_SCRATCH`, `HF_HOME`).

```bash
git clone https://github.com/RudraGandhiIISc/SkySync.git ask-iisc && cd ask-iisc
sbatch slurm/setup_env.sh        # project venv (3.11) and serve venv (vLLM, 3.12)
sbatch slurm/fetch_services.sh   # gitleaks, Prometheus, Grafana, Loki, Alloy; Redis for tests
sbatch slurm/fetch_models.sh Qwen/Qwen3-8B BAAI/bge-m3 BAAI/bge-reranker-v2-m3
# corpus and index (hours): crawl, past editions, catalogue, embed + store
sbatch slurm/crawl.sh && sbatch slurm/archive.sh && sbatch slurm/catalog.sh
sbatch --partition=low_2h_1gpu --export=ALL,EMBED_DEVICE=cuda slurm/ingest.sh
# accounts: load-test users and the first institute admin
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli seed-locust
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli add-institute-admin <e-mail>
```

Secrets (Grafana, Prometheus, Locust, Loki passwords; SMTP) are created on
first start under `$ASKIISC_DATA/private/app/`; see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#2-before-the-first-deployment).

Without a GPU you can still install and run the unit tests (fake LLM,
Milvus Lite, no network):

```bash
uv sync --locked && make all     # format check, lint, mypy, unit tests, audit
```

`make test-int` also needs a `redis-server` binary on the PATH
(`sbatch slurm/fetch_services.sh` builds one, or `apt install redis-server`).

## Use

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
| [PRD.md](PRD.md) | the requirements (original plan) |
| [dev.md](dev.md) | developer guide: what each part of the code does, testing, conventions |
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
