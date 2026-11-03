#!/bin/bash
# Start one vLLM server for MODEL (a models.yaml key) on the least busy H200,
# run a command against it with ASKIISC_LLM_URL and ASKIISC_LLM_MODEL set,
# then stop the server. Used for question generation and the LLM judge.
# From the repo root:
#   sbatch --export=ALL,MODEL=granite-3.3-8b-instruct slurm/llm_job.sh \
#     uv run python -m eval.grader.judge RESULTS
#SBATCH --job-name=askiisc_llm
#SBATCH --partition=low_2h_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-qwen3-4b-instruct-2507}
OUT="$ASKIISC_DATA/results/llm/$SLURM_JOB_ID"
mkdir -p "$OUT"

read -r GPU FREE TOTAL < <(pick_gpu "${GPU_EXCLUDE:-}")
FRAC=$(mem_fraction "$FREE" "$TOTAL")
export CUDA_VISIBLE_DEVICES=$GPU
mapfile -t ARGS < <(uv run python -c "
from askiisc.llm.models import select_model
print('\n'.join(select_model('$MODEL').vllm_args()))")
SERVED=${ARGS[2]}
PORT=$(free_port)
"$SERVE_VENV/bin/vllm" serve "${ARGS[@]}" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 --enable-prefix-caching \
  --gpu-memory-utilization "$FRAC" ${VLLM_EXTRA_ARGS:-} > "$OUT/vllm.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
START=$(date +%s)
until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "vLLM exited"; tail -30 "$OUT/vllm.log"; exit 1; }
  [ $(( $(date +%s) - START )) -lt 900 ] || { echo "vLLM not healthy"; exit 1; }
  sleep 5
done
echo "$MODEL ($SERVED) healthy on GPU $GPU after $(( $(date +%s) - START )) s"
export ASKIISC_LLM_URL="http://127.0.0.1:$PORT" ASKIISC_LLM_MODEL="$SERVED"
"$@"
STATUS=$?
echo "command exit status $STATUS"
exit $STATUS
