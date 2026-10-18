#!/bin/bash
# M1 end-to-end run: the whole stack in one GPU job, the smoke questions
# through the gateway, then teardown. Results go to
# $ASKIISC_DATA/results/stack/<job id>/. From the repo root:
#   sbatch slurm/run_stack.sh
# Another model: sbatch --export=ALL,ASKIISC_MODEL=qwen3-8b slurm/run_stack.sh
#SBATCH --job-name=askiisc_stack
#SBATCH --partition=low_2h_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1
export ASKIISC_MAX_WORKERS=${ASKIISC_MAX_WORKERS:-1}
# Report the real cold start instead of timing out at the NFR-2 limit.
export ASKIISC_MAX_COLD_HOLD_S=${ASKIISC_MAX_COLD_HOLD_S:-600}
RUN_DIR="$ASKIISC_DATA/results/stack/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

# Every child dies with the job.
trap 'pkill -TERM -P $$ 2>/dev/null; sleep 5; pkill -KILL -s 0 -u "$USER" -f "vllm serve|embed_server" 2>/dev/null' EXIT

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv | tee "$RUN_DIR/gpus_before.csv"
uv run python -m deploy.stack --run-dir "$RUN_DIR" "$@"
STATUS=$?
echo "stack exit status $STATUS; results in $RUN_DIR"
exit $STATUS
