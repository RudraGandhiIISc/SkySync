#!/bin/bash
# FR-6 / E4: profile, then W2 with admission on and off, in one GPU job.
# Parts, each on its own serving stack and run folder:
#   profile      overload at 1x/2x/3x (2 min each), ASKIISC_PROFILE=1
#   on-rep1..3   W2 (W1 at 1x, 2x, 3x), ASKIISC_ADMISSION=on
#   off-rep1..3  W2, ASKIISC_ADMISSION=off (requests queue without bound)
# Each Locust run is turned into FR-6 numbers by eval/fr6_report.py.
# At the end it submits the standing serving job again (STANDING_* below).
# From the repo root, with no other serving stack up:
#   sbatch slurm/fr6_stack.sh
# Results: $ASKIISC_DATA/results/fr6/<job>/<part>/{serve,locust,report}
#SBATCH --job-name=askiisc_fr6
#SBATCH --partition=low_unl_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:30:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_MODEL=${ASKIISC_MODEL:-qwen3-8b-awq} ASKIISC_MAX_WORKERS=1
REPS=${FR6_REPS:-3}
PARTS=${FR6_PARTS:-"profile on off"}
BASE="$ASKIISC_DATA/results/fr6/$SLURM_JOB_ID"
CURRENT="$ASKIISC_DATA/run/current.json"
mkdir -p "$BASE"
echo "== FR-6 runs, model $ASKIISC_MODEL, results in $BASE"
nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total \
  --format=csv | tee "$BASE/gpus_at_start.csv"

SERVE=""
stop() {
  [ -n "$SERVE" ] || return 0
  kill -USR1 "$SERVE" 2>/dev/null; kill -TERM "$SERVE" 2>/dev/null
  wait "$SERVE"; SERVE=""
}
restore() {
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --partition=low_unl_1gpu --time=24:00:00 \
    --export=ALL,ASKIISC_MODEL=qwen3-8b-awq,ASKIISC_MAX_WORKERS=1,ASKIISC_RENEW=1,ASKIISC_RENEW_TIME=24:00:00,ASKIISC_ADMISSION=on,ASKIISC_PROFILE=0,ASKIISC_RUN_DIR= \
    slurm/serve.sh)
  echo "standing serving job submitted again: $NEXT"
}
trap 'stop; restore; exit 1' TERM INT

# up <run dir>: start a stack with the current environment and wait for it.
up() {
  export ASKIISC_RUN_DIR=$1
  bash slurm/serve.sh &
  SERVE=$!
  local start; start=$(date +%s)
  until [ -f "$CURRENT" ] && grep -q "\"run_dir\": \"$1\"" "$CURRENT" \
      && curl -sf "127.0.0.1:$(python3 -c "import json;print(json.load(open('$CURRENT'))['ports']['web'])")/healthz" > /dev/null 2>&1; do
    kill -0 "$SERVE" 2>/dev/null || { echo "stack exited"; SERVE=""; return 1; }
    [ $(( $(date +%s) - start )) -lt 1200 ] || { echo "stack not up in 20 min"; stop; return 1; }
    sleep 10
  done
  echo "stack up after $(( $(date +%s) - start )) s ($1)"
}

# part <name> <admission> <profile> <workload> [locust args]
part() {
  local name=$1 admission=$2 profile=$3; shift 3
  local dir="$BASE/$name"
  mkdir -p "$dir/locust"
  echo "== $name: admission $admission, profile $profile, $* ($(date +%T))"
  export ASKIISC_ADMISSION=$admission ASKIISC_PROFILE=$profile
  up "$dir/serve" || return 1
  ASKIISC_LOCUST_RESULTS="$dir/locust" bash slurm/loadtest.sh "$@" \
    || echo "$name: locust exited non-zero"
  stop
  local run; run=$(ls -d "$dir"/locust/*/ 2>/dev/null | head -1)
  [ -n "$run" ] && uv run python -m eval.fr6_report "$dir/serve" "$run" "$dir/report"
}

failed=0
for p in $PARTS; do
  case $p in
    profile) part profile on 1 overload --duration-scale 0.4 || failed=$((failed + 1)) ;;
    on|off)
      for k in $(seq 1 "$REPS"); do
        part "$p-rep$k" "$p" 0 w2 || failed=$((failed + 1))
      done ;;
  esac
done
restore
echo "== FR-6 runs done, $failed part(s) failed ($(date +%T))"
exit $(( failed > 0 ))
