# Operations

How to set up, build, run, test and monitor Ask IISc on the CSA
`automation` server. How the parts work is in
[ARCHITECTURE.md](ARCHITECTURE.md); a demo script is in [DEMO.md](DEMO.md).
Every command runs from `/data4/muthangiv/AskIISc/ask-iisc`.

## 1. The server

| Item | Value |
|---|---|
| Node | One node, `AutomationServer`: 256 threads, 1 TB RAM, Ubuntu 24.04, Slurm 23.11 |
| GPUs | 3x H200 NVL (143,771 MiB each), shared with other users |
| GPU allocation | Slurm reserves none (`--gres` is rejected). Scripts pin a card themselves (`pick_gpu` in `slurm/lib.sh`) and size vLLM's memory from the card's free memory |
| Containers, root | None: everything runs as user-space processes inside Slurm jobs |
| Storage | `/data2` home and scratch together: 50 GB soft / 60 GB hard (venvs, live SQLite databases). `/data4/muthangiv`: no quota (this repository, data, weights, results) |

**Rules every job follows.**

- Nothing runs on the login node except editing, `git`, `sbatch`,
  `squeue` and `scancel`. Tests, installs and downloads run in Slurm.
- Always set `--time` and `--mem`; at most 2 GPUs and 2 GPU workers.
- A GPU job below 10% use, or a CPU job with negligible CPU, for 30 minutes
  may be killed; quiet periods inside a run stay well under that.
- Every service binds to `127.0.0.1` and dies with its job. People connect
  through an SSH tunnel.
- `dev` and GPU-partition jobs block each other (the pending reason may
  show as `InvalidAccount`). While a GPU job runs, short CPU jobs go to
  `low_2h_1gpu` (approved by the account owner).
- Limits: 2 running + 2 queued on GPU partitions, 2 + 4 on `dev`.

| Partition | Limit | Use |
|---|---|---|
| `dev` (default) | none | CPU work: gates, installs, crawl, catalogue |
| `low_2h_1gpu` | 2 h | short GPU jobs (ingest, evaluation), CPU jobs while `dev` is blocked |
| `low_4h_2gpu` | 4 h | the serving job (default) |
| `med_12h_2gpu`, `med_24h_2gpu` | 12, 24 h | long serving sessions, the 4-hour soak |

## 2. First-time setup

```bash
sbatch slurm/setup_env.sh                  # project venv (3.11) and serve venv (vLLM, 3.12)
sbatch slurm/fetch_services.sh             # Redis, gitleaks, Prometheus, Grafana, Loki, Alloy
sbatch slurm/fetch_models.sh Qwen/Qwen3-8B ibm-granite/granite-3.3-8b-instruct BAAI/bge-m3 BAAI/bge-reranker-v2-m3
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli seed-locust   # load-test accounts
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli add-institute-admin <e-mail>
```

Runs are offline (`HF_HUB_OFFLINE=1`): fetch weights before using a model.
Users live only in the private tree
`/data4/muthangiv/askiisc/private/users/<dept>/<students|faculty|staff|admins>.json`;
admins edit it on the admin **People** tab.

## 3. Job scripts (`slurm/`)

| Script | Does |
|---|---|
| `lib.sh` | Sourced by all: paths, `umask 077`, the live corpus and index (`ASKIISC_CORPUS_DIR`, `ASKIISC_STATE_DIR`), `pick_gpu`, `mem_fraction`, `free_port` |
| `gate.sh [targets]` | `make` targets in a job (the quality gate) |
| `run.sh <command>` | any command in a job with the project environment |
| `setup_env.sh`, `fetch_models.sh`, `fetch_services.sh` | one-time setup |
| `crawl.sh` | resumable crawl into the live corpus; resubmits itself every 2 h; then the catalogue and the GPU ingest |
| `archive.sh [manifest]` | past editions from the Internet Archive into `archive/` (resumable, chained) |
| `catalog.sh` | templates, duplicates, editions and validity windows: `catalog.jsonl`, then `policies.json` and `titles.json` (CPU, about 10 min) |
| `ingest.sh` | chunk, embed and store new or changed documents in the live index; `EMBED_DEVICE=cuda` embeds on a GPU; resubmits itself at its deadline |
| `serve.sh` | the serving job (section 5) |
| `loadtest.sh <workload> [args]` | a headless Locust run against the serving job |
| `llm_job.sh <command>` | one vLLM server for `MODEL`, then a command against it (question generation, judge, evaluation) |
| `vllm_smoke.sh` | one vLLM worker, one reply |
| `ui_preview.sh` | screenshots of every page on fakes, failing on page errors (no GPU) |

## 4. Building the corpus and the index

The live corpus is `/data4/muthangiv/askiisc/corpus-v3` and its index
`state-v4` (set in `slurm/lib.sh`; the older `corpus` and `state` are
kept but not served). To refresh them:

```bash
sbatch --export=ALL,INGEST_AFTER=0 slurm/crawl.sh   # new and changed pages (hours, chained)
sbatch slurm/archive.sh                             # past editions of new academic pages
sbatch slurm/catalog.sh                             # duplicates, editions, windows
sbatch --partition=low_2h_1gpu --export=ALL,EMBED_DEVICE=cuda slurm/ingest.sh
```

Without `INGEST_AFTER=0`, the crawl submits the catalogue and the ingest
itself, skipping the archive step. The ingest only processes documents
whose ETag or content changed; a changed page becomes a new version and the
old one is retired, never deleted. A rebuild from scratch (new embedding
model) took 1.7 h on one H200 for 30,740 documents. `catalog_report.json`
and the ingest log give the counts.

Restricted test documents live in `corpus-v3/restricted/<owner>/` with a
`.policy.json` each; `python -m askiisc.ingest.corpus` (run by
`catalog.sh`) adds them to `policies.json`.

## 5. Serving

```bash
sbatch slurm/serve.sh                     # Qwen3-8B, up to 2 GPU workers, 4 h
sbatch --export=ALL,ASKIISC_MODEL=qwen3-4b-instruct-2507 slurm/serve.sh   # another model
sbatch --partition=med_12h_2gpu --time=12:00:00 slurm/serve.sh            # longer
sbatch --export=ALL,ASKIISC_RENEW=1 slurm/serve.sh    # resubmits itself before its limit
sbatch --export=ALL,ASKIISC_CHAOS=1 slurm/serve.sh    # fault injection for load tests
```

Experiment switches (all default on; `off` only for the named arm):

| Variable | Off means | Used by |
|---|---|---|
| `ASKIISC_ADMISSION` | every request admitted, queues without bound | E4 (FR-6) |
| `ASKIISC_FAIR_QUEUE` | one first-come queue for LLM slots | E4 (FR-11) |
| `ASKIISC_ANSWER_CACHE` | every request goes to vLLM | FR-7 prefix-cache run |
| `ASKIISC_PROFILE` (default off) | `1` samples the event loop's stacks to `profile.folded` | FR-6 diagnosis |

The log (`logs/askiisc_serve_<job>.out`) prints `serving on
http://127.0.0.1:18000` and the tunnel command. Start-up takes about a
minute with an index snapshot, about 10 minutes when the index changed
since the last snapshot. From a laptop:

```bash
ssh -N -L 18000:localhost:18000 -L 13000:localhost:13000 \
       -L 19090:localhost:19090 -L 18089:localhost:18089 <you>@automation.csa.iisc.ac.in
```

| Address | What | Login |
|---|---|---|
| http://localhost:18000 | chat app | IISc e-mail and password (allowlisted people); test accounts |
| http://localhost:18000/admin | admin dashboard | admins only |
| http://localhost:13000 | Grafana | `admin`, `private/app/grafana_admin_password` |
| http://localhost:19090 | Prometheus | `prometheus`, `private/app/prometheus_password` |
| http://localhost:18089 | Locust UI | `admin`, `private/app/locust_ui_password` |

Passwords are created on the first start under
`/data4/muthangiv/askiisc/private/app/` and never shown on screen.
`scancel <job>` stops everything; accounts, chats and the request log are
backed up to `private/app/` every 10 minutes and at shutdown.

**Settings** (environment variables, `ASKIISC_*`; `common/config.py`
reads them, except the corpus and index folders, which the job scripts
pass): `MODEL`, `MAX_WORKERS` (2), `IDLE_TIMEOUT_S` (300),
`TENANT_RATE_PER_S` (20 in `serve.sh`), `TENANT_BURST` (40),
`MAX_IN_FLIGHT` (24), `ACCESS_TIMEOUT_S` (0.25 in `serve.sh`),
`EMBED_MODEL`, `RERANK_MODEL`, `DIRECTORY_PATH`; `CORPUS_DIR`,
`STATE_DIR` (`slurm/lib.sh`). Alert notifications need
`private/app/alerting.json`.

### Sign-in: e-mail and password, an allowlist you control

A person signs in with their IISc e-mail and their own password, and only
while **all** of these hold:

- their address is on the allowlist (`allowed_users` in `app.db`), active.
  The e-mail domain alone never admits anyone;
- they are in the users directory, which gives department, role and
  programme;
- their account is active (not disabled) and has a password.

Every refused sign-in (wrong password, unknown address, not allowed,
revoked, disabled, no password yet) gets the same answer, "wrong e-mail or
password, or no active access", and takes as long.

**Create account** (on the sign-in page; needs e-mail, below). The person
enters their IISc e-mail and a password twice. The page says at once if the
two passwords differ, if the password is shorter than 8 characters (at
most 256) or is their e-mail address, if the address is not on the
allowlist ("You are not added as an IISc user"), or if it already has an
account. Otherwise a confirmation link (works once, 10 minutes) is
e-mailed to the address; opening it creates the account, and the person
signs in. The link proves they read that mailbox, so nobody can claim a
colleague's address. Waiting registrations are kept in memory: after a
restart of the serving job, register again.

**Forgot password?** says at once when an address has no account;
otherwise it e-mails a reset link that works once, for 10 minutes (at most
3 an hour per person). These two forms tell whether an address is on the
allowlist (chosen for clarity); addresses they refuse count as failures in
the sign-in limits below, so trying many is slowed.

**Add person** on the People tab (or bulk add) does both the directory and
the allowlist in one action and creates an account without a password.
The form asks only what fits the role: a student needs a programme their
department offers (CSA: PhD, M.Tech (Research), M.Tech CSE; CDS: PhD,
M.Tech (Research), M.Tech (CDS)) and a joining year; faculty choose a type
(faculty, visiting / guest faculty) and staff a type (administrative,
technical, research staff), each with an optional joining year. The access
groups these add are listed in [ARCHITECTURE.md](ARCHITECTURE.md) section 4
and shown in the form. Bulk add takes the CSV columns `user_id`,
`department`, `role`, `program`, `joined`, `kind`, `groups`.
Only addresses of the allowed domains can be added
(`ASKIISC_EMAIL_DOMAINS`, comma-separated, default `iisc.ac.in`), so a
typo in the domain is refused at once. Nothing is e-mailed: the person
then uses **Create account** (above). A mistyped address does no harm:
nobody can confirm it, and the real person is told they are not added, so
they ask their admin to fix it.
**Suspend access** signs the person out at once and keeps their account
and chats; **Restore access** undoes it; **Remove** also deletes the
account and chats. Changing a person's department or role signs them out.
The People tab shows each person as *Not registered yet*, *Password set*,
*Access suspended*, *Not allowed* or *Test account*.

**Admins send no links.** People create their account and reset their
password themselves. A new reset link voids any earlier one, and setting a
password through a link signs the person out everywhere. Links go only to
the address on the person's own record and are never logged. Each person
has one control on the People tab, **Suspend access** / **Restore access**
(the terminal's `disable` and `enable` remain for test accounts).

**E-mail (needed for Create account and Forgot password).**
`private/app/smtp.json`, mode 600, for example a
Gmail account with an app password (Google account → Security → 2-Step
Verification → App passwords). Create it in an interactive terminal so the
password is never on screen or in shell history:

```bash
cd /data4/muthangiv/askiisc/private/app && umask 077
python3 - <<'PY'
import getpass, json
user = input("Gmail address: ").strip()
json.dump({"enabled": True, "host": "smtp.gmail.com", "port": 587,
           "starttls": True, "user": user,
           "password": getpass.getpass("App password: ").replace(" ", ""),
           "from": f"Ask IISc <{user}>",
           "public_url": "http://localhost:18000"},
          open("smtp.json", "w"))
PY
```

The serving job reads the file at start; its log says "e-mail is off" (naming
any missing fields, never values) if the file is incomplete. `public_url`
is the address people open the chat app at; links point there.

**Sign-in limits.** After 5 failures for one address within 15 minutes,
each further try waits 30 s, doubling up to 15 minutes (HTTP 429 with
`Retry-After`); a success clears it. Over 60 failures a minute across all
addresses slow every sign-in (password spraying). Limits are kept per
typed address, since everyone arrives through the SSH tunnel from one
address, and in memory, so a restart clears them.

**Sessions.** A random cookie (stored hashed), HttpOnly, SameSite=Strict,
12 hours; a new one at every sign-in. Every request re-checks the account
and the allowlist. State-changing calls need the `X-AskIISc` header and,
when the browser sends one, an `Origin` of this site (CSRF).
`ASKIISC_SECURE_COOKIES=true` adds `Secure`; set it only if the app is
served over HTTPS (not through the plain SSH tunnel).

**Passwords** are hashed with Argon2id. The scrypt hashes made before
October 2026 still work and are replaced with Argon2id at the person's next
sign-in. No database change was needed: links use the existing
`password_resets` table; `signin_flows`, `ms_oid` and `ms_tid` (from the
Microsoft sign-in tried in October 2026) are unused and kept. The serving
job restores the newest backup first and backs `app.db` up to
`private/app/` every 10 minutes and at shutdown. To keep a copy before an
upgrade:

```bash
cp -p /data4/muthangiv/askiisc/private/app/app.db \
      /data4/muthangiv/askiisc/private/app/app.db.$(date +%F)
```

**Test accounts.** Locust's test identities (ids without "@") have local
passwords (`askiisc.web.cli seed-locust`) and need no allowlist entry. They
work only while `ASKIISC_TEST_LOGINS` is on (the default); set
`ASKIISC_TEST_LOGINS=false` for a serving job that real people use without
load tests, which also ends their sessions.

**From the terminal** (people already in the directory). The first
institute admin comes from here; they then use Create account. Without
e-mail, `password-link` prints a single-use link (1 hour) that sets their
password instead:

```bash
srun --partition=low_2h_1gpu --time=00:10:00 --pty uv run python -m askiisc.web.cli add-institute-admin <e-mail>
srun --partition=low_2h_1gpu --time=00:10:00 --pty uv run python -m askiisc.web.cli password-link <e-mail>
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli remove-institute-admin <e-mail>
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli allow <e-mail>     # or: revoke
```

`srun --pty` shows the link in your terminal only, not in a log file.
Adding an institute admin also allows them; removing one revokes them.

## 6. Load tests

```bash
sbatch slurm/loadtest.sh steady                       # one workload, headless
sbatch slurm/loadtest.sh access --duration-scale 0.3  # a shorter rehearsal
```

Results go to `/data4/muthangiv/askiisc/results/locust/<job>-<workload>/`
and to the admin **Requirements** tab. The 16
workloads and the requirements each tests are in
`eval/locust/workloads.py` and in [DEMO.md](DEMO.md) section 3. Every
answer's citations are checked against the independent oracle; a forbidden
citation is a LEAK.

## 7. Quality gate

```bash
sbatch --wait slurm/gate.sh all test-int test-e2e test-sec test-regress
```

| Target | Checks |
|---|---|
| `fmt`, `lint`, `type` | ruff format, ruff (Google docstrings, security rules), mypy --strict |
| `test` | unit and conformance tests; 90% coverage overall, 100% on `access/`, `gateway/limits.py`, `gateway/drr.py`, `autoscaler/policy.py` |
| `test-int` | real Redis, HTTP access service killed mid-run, 10,000 FR-2 probes on Milvus Lite |
| `test-e2e` | the whole stack with the fake model |
| `test-sec` | bandit, the security tests (no query text in logs, loopback-only sockets, no real user ids, injection stays data), gitleaks |
| `test-regress` | golden prompts, audit record, cost model and grader report (`ASKIISC_UPDATE_GOLDENS=1` regenerates them after a deliberate change; check the diff first) |
| `test-perf` | performance thresholds (`--time=02:00:00`) |

Tests are hermetic: fakes for the model, clock and index, no network, no
GPU. Every bug fix adds a test that failed before it.

## 8. Monitoring

**Grafana "Ask IISc"** (`deploy/grafana/askiisc.json`, 5 s refresh):

| Row | Panels | Healthy |
|---|---|---|
| Overview | requests/s real and Locust, p95 latency, p95 first token, workers ready, sessions | p95 ≤ 3.5 s, first token ≤ 0.8 s, workers ≤ 2 |
| Traffic and latency | rate by origin, outcomes, 429 by department, latency p50/p95/p99, first token | errors near 0 |
| Pipeline stages | access / embed / search p95, access service p95, decisions, cache hit rate, fair queue and cold holds, tokens by department | search ≤ 150 ms, filter ≤ 20 ms |
| GPU workers | workers by state, GPU use and memory, vLLM tokens/s, requests, KV and prefix cache | 0 to 2 workers, scale to zero when idle |
| Accounts | logins by result, accounts, chat activity | |
| Locust | users, requests, response times, forbidden citations | forbidden citations must stay 0 |

**Grafana "Ask IISc requests"** (Loki): requests per minute by outcome,
the request log, every service's lines for one request id, errors in
service logs. The admin Request log's **logs** link opens it for one
request. No question or answer text is in any log.

**Alerts** (`deploy/grafana/alerts.yaml`, also on the admin Overview):
p95 above 3.5 s for 5 min, error rate above 1%, a Locust leak, a GPU
worker crash, access p95 above 20 ms, a monitored service down.

Dashboards come from the repo and reload with the serving job; edit the
JSON files rather than the UI.

## 9. When something goes wrong

| Symptom | Cause and fix |
|---|---|
| Serving log: "holds 768-dimension vectors but BAAI/bge-m3 gives 1024" | The job points at an index built with another embedder. Use the default (`state-v4`) or set `ASKIISC_EMBED_MODEL` to that index's model |
| Serve job pending, reason `InvalidAccount` | One of your `dev` jobs is running; wait or cancel it |
| `ports in use` | Another instance runs: `cat /data4/muthangiv/askiisc/run/current.json` |
| First answer takes about 30 s | Cold start after 5 minutes idle (the target is 150 s for 8B models) |
| `The service is busy (429)` | Admission limits: 24 in flight, shared by the departments asking; 20/s per department |
| A burst of 403s at the start of a load test | Access checks over 0.25 s are refused (fail closed); they settle once the filter cache is warm |
| Requests dashboard empty | Alloy ships every few seconds; check `loki.log`, `alloy.log` in the run folder |
| Slow everything | SQLite must be on local disk (`databases in /data2/scratch/...` in the log); `OMP_NUM_THREADS` set by `serve.sh` |

Run logs: `/data4/muthangiv/askiisc/results/serve/<job>/` (`access.log`,
`embed.log`, `vllm-N.log`, `requests.log`, `audit.jsonl`, `workers.json`
(each worker's lifecycle and its card's free memory and sharing at start),
`responses.jsonl` (each question request's status and milliseconds at the
serving node, FR-6), `loop_lag.jsonl` (event-loop lag per second),
Prometheus, Grafana, Loki, Locust logs). `python -m eval.fr6_report` turns
a W2 or `overload` run into FR-6 numbers. Check a load run's audit store
with `python -m eval.audit_check` (FR-12; see
[results/v3/FR-12](results/v3/FR-12/README.md)).
