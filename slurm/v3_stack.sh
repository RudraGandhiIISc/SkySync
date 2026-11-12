#!/bin/bash
# The v3 runs after FR-6, one part after another in one GPU job.
# Stack parts start a serving stack with their own settings, warm a worker,
# run their workloads (Locust inside this job) and stop:
#   fresh    FR-5 freshness replay, then FR-3 100 scripted revocations
#   revoke   the FR-3 revocations alone
#   fresh5   the FR-5 freshness replay and its check alone
#   fr7      W1 x3 with the answer cache off, prefix caching on, then off
#   nfr1     steady x3 with Qwen3-8B, up to 2 workers (NFR-1)
#   cold     20 scripted cold starts each for Qwen3-4B, Qwen3-8B, Qwen3-8B-AWQ
#   e2       W1 for every registered model, E2_REPS times (default 3)
#   idle     W3 for T_idle in 0, 60, 300, 900, 86400 s, best model per class
#   faults   keys refuse unkeyed calls; kill the embedding server, then the
#            access service, and time the watchdog's recovery (eval/fault_check)
#   fair2    FR-11 with both departments backlogged: 6 LLM slots, cache off
#   soak     NFR-5: 1 hour at half load, one worker killed at 30 minutes
# Direct vLLM parts:
#   smoke    Falcon3 3B, 7B, 7B-AWQ start and answer (FR-8 replacements)
#   kv24     each 4-bit model and its parent at one fixed 24 GB budget
#   e0       throughput sweep for every registered model (slurm/fr8_bench.sh)
#   e0sel    the same for the selected Qwen3-8B and its 4-bit build (NFR-3)
#   kvq      kv24's Qwen pair twice more, on the quietest card
# From the repo root, with no other serving stack up (queue the standing
# serving job behind it, as for slurm/fr6_stack.sh):
#   sbatch --export=ALL,V3_PARTS="fresh fr7 nfr1 cold smoke kv24 e0" slurm/v3_stack.sh
# Results: $ASKIISC_DATA/results/v3/<job>/<part>/...
#SBATCH --job-name=askiisc_v3
#SBATCH --partition=low_unl_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_MAX_WORKERS=1
PARTS=${V3_PARTS:-"fresh fr7 nfr1 cold smoke kv24 e0"}
BASE="$ASKIISC_DATA/results/v3/$SLURM_JOB_ID"
CURRENT="$ASKIISC_DATA/run/current.json"
mkdir -p "$BASE"
echo "== v3 runs: $PARTS; results in $BASE"
nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total \
  --format=csv | tee "$BASE/gpus_at_start.csv"

SERVE=""
stop() {
  [ -n "$SERVE" ] || return 0
  kill -USR1 "$SERVE" 2>/dev/null; kill -TERM "$SERVE" 2>/dev/null
  wait "$SERVE"; SERVE=""
}
trap 'stop; exit 1' TERM INT

port() { python3 -c "import json;print(json.load(open('$CURRENT'))['ports']['web'])"; }

# other_job_gpus: cards our other running jobs hold (the account may use at
# most 2 cards at once, whichever jobs hold them).
other_job_gpus() {
  local pids="" j
  for j in $(squeue -h -u "$USER" -t R -o %i); do
    [ "$j" = "$SLURM_JOB_ID" ] && continue
    pids="$pids $(scontrol listpids "$j" 2>/dev/null | awk 'NR > 1 {printf "%s ", $1}')"
  done
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ' ' \
    | awk -F, -v pids=" $pids " 'index(pids, " " $1 " ") {print $2}' | sort -u | wc -l
}

# empty_gpu: the card with no other job's process and the least memory in
# use (vLLM counts another process's growth during its profiling against
# the KV budget, so the FR-8 comparison needs a quiet card).
empty_gpu() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits \
    | tr -d ' ' \
    | awk -F, -v busy="$busy" '{ print (index(busy, $2) ? 1 : 0), $3, $1 }' \
    | sort -k1,1n -k2,2n | head -1 | awk '{print $3}'
}

# up <run dir> [warm]: start a stack with the current environment.
up() {
  export ASKIISC_RUN_DIR=$1
  bash slurm/serve.sh &
  SERVE=$!
  local start; start=$(date +%s)
  until [ -f "$CURRENT" ] && grep -q "\"run_dir\": \"$1\"" "$CURRENT" \
      && curl -sf "127.0.0.1:$(port)/healthz" > /dev/null 2>&1; do
    kill -0 "$SERVE" 2>/dev/null || { echo "stack exited"; SERVE=""; return 1; }
    [ $(( $(date +%s) - start )) -lt 1200 ] || { echo "stack not up in 20 min"; stop; return 1; }
    sleep 10
  done
  echo "stack up after $(( $(date +%s) - start )) s ($1)"
  if [ "${2:-warm}" = "warm" ]; then
    uv run python -m eval.locust.warmup "http://127.0.0.1:$(port)" --wait-s 5400 \
      || { echo "no GPU worker"; stop; return 1; }
    echo "worker ready after $(( $(date +%s) - start )) s"
  fi
}

# stack <name> <warm|cold> <runs> "<VAR=value ...>" <workload> [locust args]
stack() {
  local name=$1 warm=$2 runs=$3 envs=$4; shift 4
  local dir="$BASE/$name"
  echo "== $name: $envs, $runs x $* ($(date +%T))"
  (
    for kv in $envs; do export "${kv?}"; done
    # Other users' jobs can fill every card for minutes (the embedding
    # server then fails to load): try the stack up to 3 times.
    local try=1
    until up "$dir/serve.$try" "$warm"; do
      [ "$try" -lt 3 ] || exit 1
      echo "$name: stack try $try failed; again in 5 min"
      sleep 300; try=$((try + 1))
    done
    ln -sfn "serve.$try" "$dir/serve"
    for k in $(seq 1 "$runs"); do
      mkdir -p "$dir/rep$k"
      ASKIISC_LOCUST_RESULTS="$dir/rep$k" bash slurm/loadtest.sh "$@" \
        || echo "$name rep $k: locust exited non-zero"
    done
    stop
  )
}

failed=0
for p in $PARTS; do
  case $p in
    fresh)
      stack fresh warm 1 "ASKIISC_MODEL=qwen3-8b-awq" freshness || failed=$((failed + 1))
      run=$(ls -d "$BASE"/fresh/rep1/*-freshness 2>/dev/null | head -1)
      [ -n "$run" ] && uv run python -m eval.freshness_check "$run" "$BASE/fresh/serve" \
        "$ASKIISC_STATE_DIR/versions.db" "$BASE/fresh/report"
      stack revocations warm 1 "ASKIISC_MODEL=qwen3-8b-awq" revocations || failed=$((failed + 1)) ;;
    revoke)
      stack revocations warm 1 "ASKIISC_MODEL=qwen3-8b-awq" revocations || failed=$((failed + 1)) ;;
    fresh5)
      stack fresh5 warm 1 "ASKIISC_MODEL=qwen3-8b-awq" freshness || failed=$((failed + 1))
      run=$(ls -d "$BASE"/fresh5/rep1/*-freshness 2>/dev/null | head -1)
      [ -n "$run" ] && uv run python -m eval.freshness_check "$run" "$BASE/fresh5/serve" \
        "$ASKIISC_STATE_DIR/versions.db" "$BASE/fresh5/report" ;;
    fr7)
      stack fr7-prefix-on warm 3 "ASKIISC_MODEL=qwen3-8b-awq ASKIISC_ANSWER_CACHE=off ASKIISC_PREFIX_CACHE=on" w1 || failed=$((failed + 1))
      stack fr7-prefix-off warm 3 "ASKIISC_MODEL=qwen3-8b-awq ASKIISC_ANSWER_CACHE=off ASKIISC_PREFIX_CACHE=off" w1 || failed=$((failed + 1)) ;;
    nfr1)
      # Up to 2 workers, as the serving job runs: at W1's 4.4 questions/s
      # the autoscaler wants a second one (4 per worker). Only 1 while
      # another of our jobs holds a card (2 cards per account).
      n=2; [ "$(other_job_gpus)" -gt 0 ] && n=1
      echo "nfr1: up to $n worker(s)"
      stack nfr1 warm 3 "ASKIISC_MODEL=qwen3-8b ASKIISC_MAX_WORKERS=$n" steady || failed=$((failed + 1)) ;;
    cold)
      for m in qwen3-4b-instruct-2507 qwen3-8b qwen3-8b-awq; do
        stack "cold-$m" cold 1 "ASKIISC_MODEL=$m ASKIISC_IDLE_TIMEOUT_S=20" cold20 || failed=$((failed + 1))
      done ;;
    e2)
      for m in $(uv run python -c "from askiisc.llm.models import load_models; print(' '.join(load_models()))"); do
        stack "e2-$m" warm "${E2_REPS:-3}" "ASKIISC_MODEL=$m" w1 || failed=$((failed + 1))
      done ;;
    idle)
      for m in qwen3-4b-instruct-2507 qwen3-8b qwen3-8b-awq; do
        for t in 0 60 300 900 86400; do
          stack "idle-$m-$t" cold 1 "ASKIISC_MODEL=$m ASKIISC_IDLE_TIMEOUT_S=$t" w3 || failed=$((failed + 1))
        done
      done ;;
    faults)
      (
        export ASKIISC_MODEL=qwen3-8b
        echo "== faults ($(date +%T))"
        up "$BASE/faults/serve" warm || exit 1
        uv run python -m eval.fault_check "$BASE/faults/serve" \
          "http://127.0.0.1:$(port)" "$BASE/faults/report"
        rc=$?
        stop
        exit $rc
      ) || failed=$((failed + 1)) ;;
    fair2)
      stack fairness-backlogged warm 1 \
        "ASKIISC_MODEL=qwen3-8b ASKIISC_ANSWER_CACHE=off ASKIISC_LLM_CONCURRENCY=6" \
        fairness || failed=$((failed + 1)) ;;
    soak)
      stack soak warm 1 "ASKIISC_MODEL=qwen3-8b ASKIISC_CHAOS=1" \
        soak --duration-scale 0.25 || failed=$((failed + 1)) ;;
    kvq)
      # The Qwen pair of kv24 again, twice each, on a quiet card.
      for rep in 1 2; do
        for pair in "Qwen/Qwen3-8B bfloat16" "Qwen/Qwen3-8B-AWQ float16"; do
          set -- $pair
          SMOKE_GPU=$(empty_gpu) SMOKE_OUT="$BASE/kvq/${1#*/}-$rep" MODEL=$1 DTYPE=$2 \
            MEM_FRACTION=0.17 bash slurm/vllm_smoke.sh \
            > "$BASE/kvq-${1#*/}-$rep.log" 2>&1 || echo "kvq ${1}: did not start"
        done
      done ;;
    e0sel)
      MAX_LANES=1 bash slurm/fr8_bench.sh qwen3-8b qwen3-8b-awq \
        > "$BASE/e0sel.log" 2>&1 || failed=$((failed + 1)) ;;
    smoke)
      for r in tiiuae/Falcon3-3B-Instruct tiiuae/Falcon3-7B-Instruct; do
        SMOKE_OUT="$BASE/smoke/${r#*/}" MODEL=$r bash slurm/vllm_smoke.sh \
          > "$BASE/smoke-${r#*/}.log" 2>&1 || failed=$((failed + 1))
      done
      SMOKE_OUT="$BASE/smoke/Falcon3-7B-Instruct-AWQ" MODEL=tiiuae/Falcon3-7B-Instruct-AWQ \
        DTYPE=float16 bash slurm/vllm_smoke.sh \
        > "$BASE/smoke-Falcon3-7B-Instruct-AWQ.log" 2>&1 || failed=$((failed + 1)) ;;
    kv24)
      # 24 GiB of the 140.4 GiB card: --gpu-memory-utilization 0.17.
      for pair in "Qwen/Qwen3-8B bfloat16" "Qwen/Qwen3-8B-AWQ float16" \
                  "ibm-granite/granite-3.3-8b-instruct bfloat16" \
                  "TevunahAi/Granite-3.3-8B-Instruct-GPTQ float16" \
                  "tiiuae/Falcon3-7B-Instruct bfloat16" "tiiuae/Falcon3-7B-Instruct-AWQ float16"; do
        set -- $pair
        SMOKE_OUT="$BASE/kv24/${1#*/}" MODEL=$1 DTYPE=$2 MEM_FRACTION=0.17 bash slurm/vllm_smoke.sh \
          > "$BASE/kv24-${1#*/}.log" 2>&1 || echo "kv24 ${1}: did not start at 24 GB"
      done ;;
    e0)
      bash slurm/fr8_bench.sh $(uv run python -c "from askiisc.llm.models import load_models; print(' '.join(load_models()))") \
        > "$BASE/e0.log" 2>&1 || failed=$((failed + 1)) ;;
  esac
done
echo "== v3 runs done, $failed part(s) failed ($(date +%T))"
exit $(( failed > 0 ))
