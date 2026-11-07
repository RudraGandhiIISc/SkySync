# Deployment

How to put Ask IISc into service on the CSA `automation` server, keep it
running, update it and roll it back. Each step is a command to run from the
repository root (`/data4/muthangiv/AskIISc/ask-iisc`). The details
behind each step are in [OPERATIONS.md](OPERATIONS.md); the design is in
[ARCHITECTURE.md](ARCHITECTURE.md).

Nothing here runs on the login node except editing, `git`, `sbatch`,
`squeue` and `scancel`. Every service binds to `127.0.0.1` and dies with
its Slurm job; people reach it through an SSH tunnel.

## 1. What a deployment is

| Part | Where | In git |
|---|---|---|
| Code, job scripts, dashboards | `/data4/muthangiv/AskIISc/ask-iisc` | yes |
| Virtual environments | `/data2/scratch/muthangiv/askiisc/` | no |
| Model weights | `/data4/muthangiv/huggingface/` | no |
| Corpus and index (`corpus-v3`, `state-v4`) | `/data4/muthangiv/askiisc/` | no |
| People, allowlist, accounts, chats, secrets | `/data4/muthangiv/askiisc/private/` (mode 700) | **never** |
| Run logs and results | `/data4/muthangiv/askiisc/results/` | no |

One Slurm job (`slurm/serve.sh`) runs everything: the chat app and admin
dashboard, gateway, access service, embedding server, up to 2 vLLM workers
(scale to zero after 5 idle minutes), Redis, Prometheus, Grafana, Loki,
Alloy and the Locust UI.

## 2. Before the first deployment

Run once, in order; each is a short job.

```bash
sbatch slurm/setup_env.sh          # project venv (Python 3.11) and serve venv (vLLM 0.11)
sbatch slurm/fetch_services.sh     # Redis, gitleaks, Prometheus, Grafana, Loki, Alloy
sbatch slurm/fetch_models.sh Qwen/Qwen3-8B-AWQ Qwen/Qwen3-8B \
  ibm-granite/granite-3.3-8b-instruct BAAI/bge-m3 BAAI/bge-reranker-v2-m3
```

Granite-3.3-8B is needed even when another model serves: it is the judge
of the quality runs. Other models of `src/askiisc/llm/models.yaml` are
fetched the same way before they are served (runs are offline).

If `dev` jobs stay pending (they wait while any GPU-partition job runs),
add `--partition=low_2h_1gpu` to the short CPU jobs here and below.

### Secrets and e-mail

Created on the first start, mode 600, in `private/app/`, never printed:
`audit_salt`, `chat.key` (encrypts chat history), `grafana_admin_password`,
`prometheus_password`, `loki_password`, `locust_ui_password`.

Written by hand, mode 600:

- `private/app/smtp.json`, for registration and password-reset e-mails
  (fields in `src/askiisc/web/mailer.py`; for Gmail, an app password).
  Set `public_url` to the address people open, normally
  `http://localhost:18000`. Without this file, Create account and Forgot
  password answer "not set up".
- `private/app/alerting.json` (optional), where Grafana sends alerts.

### People

```bash
srun --partition=low_2h_1gpu --time=00:10:00 --pty \
  uv run python -m askiisc.web.cli add-institute-admin <name>@iisc.ac.in
sbatch --partition=low_2h_1gpu slurm/run.sh uv run python -m askiisc.web.cli seed-locust
```

The institute admin then opens the chat app, chooses **Create account**
and confirms the e-mailed link (10 minutes). Everyone else is added on the
admin **People** tab (only `@iisc.ac.in` addresses) and registers the same
way. `seed-locust` creates the load-test accounts (ids without "@") and
must be run again whenever departments change.

## 3. Release checklist

1. Work on a branch (`phase-N/topic`), never on `main` directly.
2. Run the whole gate and read its log:
   ```bash
   sbatch --wait --partition=low_2h_1gpu slurm/gate.sh all test-int test-e2e test-sec test-regress
   tail -3 logs/askiisc_gate_<job>.out     # ends "... passed"
   ```
   It covers format, lint, `mypy --strict`, unit tests (coverage at least
   90%, 100% on access and the limiters), integration, end-to-end,
   security (bandit, pip-audit, gitleaks, access probes, socket audit) and
   regression.
3. Check the pages render without script errors: `sbatch slurm/ui_preview.sh`.
4. Commit (Conventional Commits, naming the FR or NFR), merge to `main`,
   and tag the release: `git tag -a vYYYY.MM.DD -m "..."`.
5. Note in [STATUS.md](STATUS.md) what changed and what was measured.

## 4. Start the serving job

```bash
sbatch --export=ALL,ASKIISC_MODEL=qwen3-8b-awq,ASKIISC_RENEW=1 slurm/serve.sh
```

- `ASKIISC_MODEL` picks the model by its key in `models.yaml` (default
  `qwen3-8b`). Qwen3-8B-AWQ has the best measured accuracy (0.82) and needs
  about half the weight memory of Qwen3-8B
  ([results/fr8](results/fr8/README.md)).
- `ASKIISC_RENEW=1` resubmits the job 5 minutes before its time limit
  (default 4 h on `low_4h_2gpu`), so service continues with a gap of about
  a minute. For fewer restarts: `--partition=med_24h_2gpu --time=24:00:00`.
- One GPU only: `--partition=low_unl_1gpu --export=ALL,...,ASKIISC_MAX_WORKERS=1`.
- Only one serving job can run (it holds the ports). Stop it before
  starting another, and before FR-8 or load-test chains that start their
  own stack.

## 5. Check it

```bash
squeue -u $USER                                  # the job is R
grep -m1 "serving on" logs/askiisc_serve_<job>.out
cat /data4/muthangiv/askiisc/run/current.json    # job id and ports
```

From a laptop (one tunnel per person):

```bash
ssh -N -L 18000:localhost:18000 -L 13000:localhost:13000 <you>@automation.csa.iisc.ac.in
```

Then:

- http://localhost:18000 : sign in, ask a question; the first answer after
  idle takes about 30 s while a GPU worker starts.
- http://localhost:18000/admin : Overview shows the serving job, the model
  and the GPU; Requirements shows the scorecard.
- http://localhost:13000 : Grafana (`admin`, `private/app/grafana_admin_password`).

If start-up fails, see [OPERATIONS.md section 9](OPERATIONS.md#9-when-something-goes-wrong)
and the run folder `/data4/muthangiv/askiisc/results/serve/<job>/`.

## 6. Update

1. Pass the release checklist (section 3) on the new commit.
2. Stop the running job and start the new one:
   ```bash
   scancel <job>        # data is saved at shutdown; renewals stop with it
   sbatch --export=ALL,ASKIISC_MODEL=qwen3-8b-awq,ASKIISC_RENEW=1 slurm/serve.sh
   ```
3. Check it (section 5).

Changing only the model needs no code change: restart with another
`ASKIISC_MODEL`. Refreshing the corpus and index is a separate pipeline
([OPERATIONS.md section 4](OPERATIONS.md#4-building-the-corpus-and-the-index));
the serving job picks up the new index at its next start.

## 7. Roll back

```bash
scancel <job>
git checkout <previous tag>
sbatch --export=ALL,ASKIISC_MODEL=<model>,ASKIISC_RENEW=1 slurm/serve.sh
```

Data on `/data4` is not changed by code rollbacks. Database upgrades only
ever add tables and columns, and an older release leaves a newer
database's schema version as it is, so it runs on the newer file and
ignores what it does not know. To undo an upgrade's data too, restore the
backup taken before the update (section 8).

## 8. Backups

The serving job copies the accounts, chats and request log into
`private/app/` every 10 minutes and at shutdown. Before an update, copy
the whole private tree:

```bash
mkdir -p -m 700 /data4/muthangiv/askiisc/backups
cp -a /data4/muthangiv/askiisc/private /data4/muthangiv/askiisc/backups/private-$(date +%F)
```

`chat.key` must be kept with `app.db`: chats cannot be read without it.
Keep backups on `/data4` (mode 700); never copy them into the repository
or `/home`.

## 9. Stop

```bash
scancel <job>
```

All services stop with the job; nothing keeps running. Accounts, chats
and the request log are saved at shutdown.

## 10. Security checklist

- Services listen on `127.0.0.1` only (checked by the socket audit in the
  gate); there is no public address.
- Only allowlisted `@iisc.ac.in` people can register; passwords are
  Argon2id; sign-in is throttled per address and overall; sessions are
  HttpOnly, SameSite=Strict cookies; state-changing requests need the
  `X-AskIISc` header and a same-site origin.
- Access checks fail closed; filters run inside the search.
- No query or answer text in logs; users are salted hashes in the request
  log and audit; chats are encrypted with `chat.key`.
- Secrets live only in `private/app/` (mode 600); gitleaks runs in the
  gate. Never commit anything from `/data4`.
