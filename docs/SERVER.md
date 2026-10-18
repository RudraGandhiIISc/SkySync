# Running Ask IISc on the CSA automation server

Every phase of Ask IISc, from development to the nine-model sweep, runs on the CSA
`automation.csa.iisc.ac.in` server. This file records what the server is, how each component
of the PRD runs on it without root or containers, and the rules every script must follow.
The site policy is in `~/CLUSTER_POLICIES.md`; where the server behaves differently from
that document, this file records what was observed.

## 1. The server (checked 2026-10-18)

| Item | Value |
|---|---|
| Node | One node, `AutomationServer`: AMD EPYC 9555 (64 cores, 256 threads), 1 TB RAM, Ubuntu 24.04, Slurm 23.11.4 |
| GPUs | 3x NVIDIA H200 NVL, 143,771 MiB each (the PRD's "144 GB"). Shared with other users: on 2026-10-18 GPU 1 and GPU 2 each had 53 GB in use at 100% utilisation |
| Containers | None. `docker` exists but its socket is permission-denied; no Apptainer, Podman or sudo |
| Python | System 3.12.3; `uv` in `~/.local/bin` (it can download 3.11) |
| Internet | HuggingFace reachable; use it only for downloads, never while experiments run |
| GPU allocation | `Gres=(null)`: `--gres=gpu:N` is rejected and Slurm reserves no GPU. Every job sees all three cards; scripts pin one with `CUDA_VISIBLE_DEVICES` (`slurm/lib.sh`, `pick_gpu`) |
| Accounting | `sacct` is disabled; log job times from inside the job |

### Partitions

| Partition | Time limit | Use for |
|---|---|---|
| `dev` (default) | unlimited | CPU-only work: `make` gates, installs, ingestion, building service binaries |
| `low_2h_1gpu` | 2 h | vLLM smoke tests, short single-worker runs |
| `low_unl_1gpu` | 5 days | not used (long single-GPU holds hurt everyone) |
| `low_4h_2gpu`, `med_2h_2gpu`, `med_4h_2gpu` | 2 to 4 h | experiment runs (E0 to E4, E6) |
| `med_12h_2gpu`, `med_24h_2gpu` | 12 h, 24 h | E5 (4-hour soak), long model sweeps |
| `all` | 2 h | not used |

Two scheduling facts matter:

0. While `dev` is blocked and a GPU is free, short CPU-only jobs (gates, installs,
   downloads) were run on `low_2h_1gpu` with the account owner's approval
   (`sbatch --partition=low_2h_1gpu ...`). This is what the policy's own
   Example 2 does; admin question 3 asks whether it is allowed in general.
1. `dev` has `OverSubscribe=NO`, the GPU partitions `OverSubscribe=FORCE:4`. A `dev` job waits
   while **any** user's GPU-partition job runs, and the reverse. The pending reason can show as
   `None` or the misleading `InvalidAccount`; `scontrol show job <id>` gives the planned start.
   So CPU preparation that a GPU run needs (starting Milvus, MinIO, Redis, the gateway) happens
   inside the GPU job, never in a separate `dev` job.
2. Limits: 2 running + 2 queued on GPU partitions, 2 running + 4 queued on `dev`. Long chains
   of runs use jobs that submit their successor (`sbatch --dependency=afterany:$SLURM_JOB_ID`).

### Rules every job must follow

- Nothing runs on the login node except editing, `git`, `sbatch`, `squeue`, `scancel`.
  This includes `pytest`, `make all`, `uv sync` and model downloads: use `slurm/gate.sh`,
  `slurm/setup_env.sh` and `slurm/fetch_models.sh`.
- Always set `--time` and `--mem`. At most 2 GPUs per job, and 2 GPU workers in total (FR-9).
- A GPU job below 10% utilisation for 30 minutes may be killed, and a CPU job with negligible
  CPU for 30 minutes too. Every quiet period inside a run (scaled to zero, W3 gaps, idle timeout
  sweeps) must therefore stay well under 30 minutes of wall-clock time. W3 replayed 24x faster
  keeps a 3-hour campus gap to 7.5 minutes.
- No service outlives its job. A run starts the stack, runs, writes results and stops
  everything; the job's `trap` kills every child process on exit.

## 2. Storage

The enforced quota is **50 GB soft / 60 GB hard for `/data2` home and scratch together** (not the
1 TB the policy doc states). `/data4/muthangiv` is an NFS mount with no quota.

| Path | Holds |
|---|---|
| `/data2/home/muthangiv/AskIISc/ask-iisc` | the git repository (code, docs, small fixtures, `docs/results/` summaries) |
| `/data2/scratch/muthangiv/askiisc/` | `.venv` (project), `serve-venv` (vLLM and embeddings), `bin/` (service binaries) |
| `/data4/muthangiv/huggingface` | `HF_HOME`: all model weights (the nine models are about 90 GB) |
| `/data4/muthangiv/askiisc/` | `data/` (corpus, chunks), `state/` (Milvus, MinIO, Redis data per run), `results/` (raw run logs), `models/` (AWQ checkpoints) |

Raw logs stay on `/data4`; only the summary files a phase asks for are copied into
`docs/results/`. Check `quota -v -s` before any large download.

## 3. How each PRD component runs here

The PRD architecture is unchanged; only how it is started changes. Kubernetes, Docker
Compose, DCGM and TEI cannot run without root or containers, so each has a user-space
equivalent.

| PRD component | On this server |
|---|---|
| GPU node (FR-9, FR-10) | A **GPU worker**: one `vllm serve` process pinned to one H200 with `CUDA_VISIBLE_DEVICES`, started and stopped by the autoscaler inside the run's Slurm job. At most 2 workers. |
| Node pool / Kubernetes | `autoscaler/pool.py`: a `WorkerPool` interface with `ProcessPool` (real), `FakeWorkerPool` (tests) and, only if the admin allows it, `SlurmPool` (one `sbatch` job per worker) |
| Deployments, CronJob | `deploy/stack.py` (run by `slurm/run_stack.sh`): starts the services in order on free localhost ports, waits for health checks and stops them in reverse. The reconciler loop is not wired into it yet |
| vLLM | `serve-venv`, `vllm serve`; `--gpu-memory-utilization` is computed from the card's **free** memory at start (`pick_gpu`), not fixed at 0.90 |
| Milvus 2.4 | Unit tests: Milvus Lite (`pymilvus`). Integration tests and runs: Milvus standalone unpacked from the official `.deb` with `dpkg-deb -x` (no root), embedded etcd and local storage. Week-1 spike; fallback below. |
| MinIO | **Not available**: MinIO stopped publishing community binaries (HTTP 410 since 2026; only the licensed AIStor build remains). Originals are stored as files by `ingest/objects.py` `FileObjectStore` under the run directory on `/data4`, behind the same `ObjectStore` interface |
| Redis | 7.2.5 (last BSD-licensed line), built from source by `slurm/fetch_services.sh` into `bin/` |
| Embedding server (TEI) | TEI needs Docker or a Rust build, so `retrieval/embed_server.py` serves `BAAI/bge-base-en-v1.5` with sentence-transformers on CPU behind the same `/embed` route. The project code only talks to it over HTTP. |
| Prometheus, Grafana | official release tarballs in `bin/`, config in `deploy/local/` |
| DCGM exporter | not available; `observability/gpu.py` samples NVML (`pynvml`) at 1 Hz and exports Prometheus gauges for the pinned cards |
| Docker Compose integration tests | `tests/integration/conftest.py` starts the real binaries (Redis) and real HTTP servers on random ports inside the test job |
| NetworkPolicy (NFR-9) | every service binds to `127.0.0.1`; a socket-audit test fails if any port listens elsewhere. Demos use an SSH tunnel. No packet capture (needs root): egress is checked with `HF_HUB_OFFLINE=1` and the socket audit, and the report states this limit |
| TLS 1.3 | `uvicorn --ssl-*` with a self-signed certificate kept in scratch (mode 600) |
| SSO and LDAP | a local JWKS file and the fake directory export (A4) |
| GitHub Actions CI | optional mirror for lint, types and unit tests. The gates that count run on this server through `slurm/gate.sh` |

**Fallback if Milvus standalone will not run without root:** Qdrant ships as a single static
binary with HNSW and in-search payload filtering, which the earlier plan used. Switching
changes only `retrieval/store.py` and the PRD stack table, so it needs a team decision, not a
redesign.

### Cost metering (NFR-6)

Because Slurm reserves no GPU, the job does not hold a card; a worker does. GPU-hours are
therefore charged from each worker's lifetime: `a` (choosing a card plus the optional
`ASKIISC_ACQUIRE_DELAY_S` that emulates node acquisition, reported separately), `l` (weight
load until healthy), `b` (serving) and `d` (resident but idle). The control plane's CPU time
is not charged, which matches the PRD's billing of GPU occupancy.

## 4. Run pattern

```
login node                       Slurm job (GPU partition, --time, --mem)
-----------                      ---------------------------------------------------------
sbatch slurm/run_stack.sh  ───▶  trap cleanup EXIT
                                 deploy/stack.py up: MinIO, Redis, Milvus, embed server,
                                   access service, gateway, Prometheus  (all 127.0.0.1)
                                 autoscaler: ProcessPool, at most 2 vLLM workers
                                 Locust or the evaluation tool runs for a fixed time
                                 results -> /data4/muthangiv/askiisc/results/<run>/
                                 deploy/stack.py down
```

Development and the gate:

```bash
sbatch --wait slurm/gate.sh all        # make all inside a dev job; output in logs/
sbatch --wait slurm/gate.sh test-int   # integration tests against the real binaries
```

Interactive debugging (30 minutes idle limit, exit when done):

```bash
srun --partition=dev --cpus-per-task=8 --mem=16G --time=00:30:00 --pty bash
```

Demo from a laptop on the campus VPN: start `slurm/run_stack.sh` with `DEMO=1`, read the
gateway port from the log, then `ssh -L 8443:127.0.0.1:<port> muthangiv@10.192.27.32`.

## 5. Week-1 spikes

| Spike | Command | Done when |
|---|---|---|
| Project environment | `sbatch slurm/setup_env.sh` | the log ends with `setup done` |
| vLLM on one H200 | `sbatch slurm/vllm_smoke.sh` | one reply printed, with load time and the card used |
| Model weights | `sbatch slurm/fetch_models.sh Qwen/Qwen3-4B-Instruct-2507` | weights in `/data4/muthangiv/huggingface` |
| Service binaries | `sbatch slurm/fetch_services.sh` (Redis, gitleaks); Milvus standalone, Prometheus and Grafana still to do | each answers its health check inside a job |

Gemma-3 and Llama-3.1 are gated on HuggingFace: accept their licences and put an access token
in `~/.cache/huggingface/token` (mode 600) before fetching them.

## 6. Open questions for the admin (Mr. Akshay Nath, akshaynath@iisc.ac.in)

1. May a Slurm job run HTTP services on localhost ports (vLLM, Milvus, gateway) for the
   duration of the job, or does the monitor treat them as unauthorised background services?
2. GPUs are not reserved by Slurm (`Gres=(null)`). Is pinning a free card with
   `CUDA_VISIBLE_DEVICES` the intended practice, and can we book an exclusive window for the
   benchmark runs?
3. `dev` and GPU-partition jobs block each other, so a `dev` gate can wait up to 24 hours behind
   another user's GPU job. May short CPU-only test jobs use `low_2h_1gpu`, as Example 2 in the
   policy does?
4. Confirm the high-partition window ("8 PM to 5 PM" reads like a typo).
5. May the autoscaler submit and cancel its own GPU jobs (`SlurmPool`), within the 2-GPU limit?

## 7. Status

- [x] `sinfo`, GPU, Docker and quota facts recorded (2026-10-18)
- [ ] admin replies recorded
- [x] `slurm/setup_env.sh` succeeded (job 1285: Python 3.11.16 project venv; vLLM 0.11.0, torch 2.8.0+cu128)
- [x] `slurm/vllm_smoke.sh` succeeded (job 1290: Qwen3-4B on GPU 0, healthy after 130 s; 70 s of it is reading weights from NFS)
- [x] Redis 7.2.5 and gitleaks 8.18.4 installed (job 1295); MinIO unavailable (see section 3)
- [ ] Milvus standalone without root (Milvus Lite is used until then), Prometheus, Grafana
- [x] First end-to-end run of the whole stack with Qwen3-4B (job 1293, `slurm/run_stack.sh`)
