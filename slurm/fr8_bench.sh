#!/bin/bash
# FR-8 part 1: each model's vLLM server alone (NFR-3 throughput sweep and
# its start-up time), for the models named on the command line, using at
# most 2 GPUs. Models run one after another on the free card; before each
# one, a second card with room (NEED_MIB free, not ours) starts a second
# model in parallel. No card is held idle while waiting.
# From the repo root:
#   sbatch slurm/fr8_bench.sh phi-4-mini-instruct qwen3-8b ...
# Results: $ASKIISC_DATA/results/models/<key>/bench-<job>/ (sweep.json,
# ready_s.txt, vllm.log). Recording them for the Models tab is a separate
# step (python -m askiisc.web.models record).
#SBATCH --job-name=askiisc_fr8
#SBATCH --partition=low_4h_2gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1
NEED_MIB=${NEED_MIB:-40000}  # an 8B model in bf16 with a useful KV cache
MAX_LANES=2
RESULTS=${RESULTS:-$ASKIISC_DATA/results/versioned/v4/qwen3-8b-p2/answers.json}
declare -A LANE_GPU=()  # pid -> GPU

# A card with NEED_MIB free that none of our lanes uses, least busy first.
free_card() {
  local ours=",${LANE_GPU[*]// /,},"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total \
      --format=csv,noheader,nounits | tr -d ' ' \
    | awk -F, -v ours="$ours" -v need="$NEED_MIB" \
        'index(ours, "," $1 ",") == 0 && $3 >= need' \
    | sort -t, -k2,2n -k3,3nr | head -1 | awk -F, '{print $1, $3, $4}'
}

# bench <model> <gpu> <free MiB> <total MiB>: one model, start to finish.
bench() {
  local model=$1 gpu=$2 free=$3 total=$4
  local out="$ASKIISC_DATA/results/models/$model/bench-$SLURM_JOB_ID"
  mkdir -p "$out"
  local args port pid start
  mapfile -t args < <(uv run python -c "
from askiisc.llm.models import select_model
print('\n'.join(select_model('$model').vllm_args()))")
  port=$(free_port)
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu "$SERVE_VENV/bin/vllm" serve "${args[@]}" \
    --host 127.0.0.1 --port "$port" --max-model-len 16384 \
    --enable-prefix-caching \
    --gpu-memory-utilization "$(mem_fraction "$free" "$total")" \
    > "$out/vllm.log" 2>&1 &
  pid=$!
  until curl -sf "127.0.0.1:$port/health" > /dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$model: vLLM exited"; tail -20 "$out/vllm.log"; return 1
    fi
    if [ $(( $(date +%s) - start )) -ge 900 ]; then
      echo "$model: not healthy in 900 s"; kill "$pid"; return 1
    fi
    sleep 5
  done
  echo $(( $(date +%s) - start )) > "$out/ready_s.txt"
  echo "$model ready on GPU $gpu after $(cat "$out/ready_s.txt") s"
  ASKIISC_LLM_URL="http://127.0.0.1:$port" ASKIISC_LLM_MODEL="${args[2]}" \
    uv run python -m eval.llm_bench sweep "$RESULTS" "$ASKIISC_STATE_DIR" \
    "$out" > "$out/sweep.log" 2>&1
  local status=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  echo "$model: sweep exit status $status ($out)"
  return $status
}

queue=("$@")
failed=0
# Every 2 minutes: start the next model on any card that has room (at most
# 2 lanes), and collect lanes that finished.
while [ ${#queue[@]} -gt 0 ] || [ ${#LANE_GPU[@]} -gt 0 ]; do
  for pid in "${!LANE_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || failed=$((failed + 1))
      unset "LANE_GPU[$pid]"
    fi
  done
  if [ ${#queue[@]} -gt 0 ] && [ ${#LANE_GPU[@]} -lt $MAX_LANES ]; then
    read -r gpu free total < <(free_card)
    if [ -n "${gpu:-}" ]; then
      bench "${queue[0]}" "$gpu" "$free" "$total" &
      LANE_GPU[$!]=$gpu
      echo "$(date +%T) started ${queue[0]} on GPU $gpu ($free MiB free)"
      queue=("${queue[@]:1}")
      sleep 60  # let vLLM claim its memory before judging the next card
      continue
    fi
  fi
  sleep 120
done
echo "all models done, $failed failed"
exit $(( failed > 0 ))
