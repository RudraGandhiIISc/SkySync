#!/bin/bash
# Serve Ask IISc for people: chat app, admin dashboard, Prometheus, Grafana
# and the Locust web UI, until the job ends. From the repo root:
#   sbatch slurm/serve.sh                   # 4 h, up to 2 GPU workers
# Longer session:
#   sbatch --partition=med_24h_2gpu --time=24:00:00 slurm/serve.sh
# One GPU only (a 1-GPU partition allows one card):
#   sbatch --partition=low_unl_1gpu --time=08:00:00 \
#     --export=ALL,ASKIISC_MAX_WORKERS=1 slurm/serve.sh
# Note: a pending reason "InvalidAccount" only means one of your dev jobs
# is running; GPU-partition and dev jobs never run at the same time.
# Keep it running (resubmits itself 5 min before the time limit):
#   sbatch --export=ALL,ASKIISC_RENEW=1 slurm/serve.sh
# Fault injection for the soak and failover load tests:
#   sbatch --export=ALL,ASKIISC_CHAOS=1 slurm/serve.sh
# How to connect: the log (logs/askiisc_serve_<job>.out) prints the SSH
# tunnel command; ports are in $ASKIISC_DATA/run/current.json.
#SBATCH --job-name=askiisc_serve
#SBATCH --partition=low_4h_2gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1
# Slurm pins no CPUs here, so numpy (BLAS) and torch would each start one
# thread per core of the 256-core node; under load that oversubscription
# made a 20 ms search take over a second. Cap them for every child.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
# At most 2 GPUs (server rule); set 1 on a 1-GPU partition.
export ASKIISC_MAX_WORKERS=${ASKIISC_MAX_WORKERS:-2}
# Departments share the GPUs: per-department admission rate and burst
# (FR-6). Sized for classroom use; the overload test lowers nothing.
export ASKIISC_TENANT_RATE_PER_S=${ASKIISC_TENANT_RATE_PER_S:-20}
export ASKIISC_TENANT_BURST=${ASKIISC_TENANT_BURST:-40}
# The access service runs in its own process; allow for a busy gateway loop.
export ASKIISC_ACCESS_TIMEOUT_S=${ASKIISC_ACCESS_TIMEOUT_S:-0.25}
# Qwen3-8B meets NFR-7 on the edition-aware set (84.5%; Qwen3-4B 79.8%).
export ASKIISC_MODEL=${ASKIISC_MODEL:-qwen3-8b}
# The job's GPUs: the least busy cards, at most ASKIISC_MAX_WORKERS. vLLM
# workers use only these; the embedding and reranking server (bge-m3 and
# bge-reranker-v2-m3, ~3 GB) shares the first one.
if [ -z "${ASKIISC_GPUS:-}" ]; then
  FIRST=$(pick_gpu | cut -d' ' -f1)
  GPUS=$FIRST
  if [ "$ASKIISC_MAX_WORKERS" -gt 1 ]; then
    GPUS="$GPUS,$(pick_gpu "$FIRST" | cut -d' ' -f1)"
  fi
  export ASKIISC_GPUS=$GPUS
fi
export ASKIISC_EMBED_DEVICE=${ASKIISC_EMBED_DEVICE:-cuda}
echo "GPUs for this job: $ASKIISC_GPUS (embedding on $ASKIISC_EMBED_DEVICE)"
STATE=$ASKIISC_STATE_DIR
# A driver that starts several stacks in one job (slurm/fr6_stack.sh) gives
# each its own folder.
RUN_DIR=${ASKIISC_RUN_DIR:-"$ASKIISC_DATA/results/serve/$SLURM_JOB_ID"}
mkdir -p "$RUN_DIR" "$ASKIISC_DATA/run"

uv run python -m deploy.serve --run-dir "$RUN_DIR" \
  --corpus "$ASKIISC_CORPUS_DIR" --state-dir "$STATE" &
PID=$!

renew() {
  echo "time limit near: stopping"
  if [ "${ASKIISC_RENEW:-0}" = "1" ]; then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
      --partition="$SLURM_JOB_PARTITION" --time="${ASKIISC_RENEW_TIME:-04:00:00}" \
      --export=ALL slurm/serve.sh)
    echo "submitted the next serving job $NEXT"
  fi
  kill -TERM "$PID" 2>/dev/null
}
trap renew USR1
trap 'kill -TERM "$PID" 2>/dev/null' TERM INT
wait "$PID"
STATUS=$?
# wait returns early when a trap fires; wait for the real exit.
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; STATUS=$?; done
# Nothing outlives the job: every process group deploy.serve started.
if [ -f "$RUN_DIR/pids" ]; then
  while read -r P; do kill -KILL -- "-$P" 2>/dev/null; done < "$RUN_DIR/pids"
fi
echo "serve exit status $STATUS"
exit $STATUS
