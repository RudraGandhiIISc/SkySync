#!/bin/bash
# Week-1 spike: start one vLLM worker on the least busy H200, ask one question,
# record the load time, shut down. Needs slurm/setup_env.sh and the weights
# (slurm/fetch_models.sh). From the repo root: sbatch slurm/vllm_smoke.sh
# Other model: sbatch --export=ALL,MODEL=Qwen/Qwen3-8B slurm/vllm_smoke.sh
#SBATCH --job-name=askiisc_vllm_smoke
#SBATCH --partition=low_2h_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1   # runs never reach the internet (NFR-9)

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
OUT="$ASKIISC_DATA/results/smoke/$SLURM_JOB_ID"
mkdir -p "$OUT"

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv | tee "$OUT/gpus_before.csv"
read -r GPU FREE TOTAL < <(pick_gpu)
FRAC=$(mem_fraction "$FREE" "$TOTAL")
echo "card $GPU: ${FREE} MiB free of ${TOTAL}; --gpu-memory-utilization $FRAC"
if awk -v x="$FRAC" 'BEGIN { exit !(x < 0.25) }'; then
  echo "not enough free GPU memory on any card; try later"; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU

PORT=$(free_port)
START=$(date +%s)
"$SERVE_VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 --enable-prefix-caching \
  --gpu-memory-utilization "$FRAC" > "$OUT/vllm.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

until curl -sf "127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $PID 2>/dev/null || { echo "vLLM exited during start-up"; tail -40 "$OUT/vllm.log"; exit 1; }
  [ $(( $(date +%s) - START )) -lt 900 ] || { echo "vLLM not healthy after 900 s"; exit 1; }
  sleep 5
done
LOAD=$(( $(date +%s) - START ))
echo "healthy after ${LOAD} s"

curl -s "127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one line.\"}],\"max_tokens\":30,\"temperature\":0}" \
  | tee "$OUT/reply.json"
echo
printf 'model=%s gpu=%s mem_fraction=%s load_s=%s\n' "$MODEL" "$GPU" "$FRAC" "$LOAD" \
  | tee "$OUT/summary.txt"
echo "smoke test done"
