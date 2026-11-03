#!/bin/bash
# FR-8 part 2: the whole system with one model, then the next model.
# For the first model named: start the serving stack with it (one GPU
# worker), run the steady, cold-start, diurnal and quality workloads
# against it, stop it, grade the answers (rubric, then the Granite judge),
# then submit this script again for the remaining models.
# From the repo root:
#   sbatch slurm/fr8_stack.sh phi-4-mini-instruct qwen3-8b ...
# While it runs, the chat app serves that model (only one serving stack
# can hold the ports). Results stay out of the scorecard:
#   $ASKIISC_DATA/results/models/<key>/locust/<job>-<workload>/
#   $ASKIISC_DATA/results/models/<key>/measurements/
#SBATCH --job-name=askiisc_fr8_stack
#SBATCH --partition=low_unl_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
[ $# -gt 0 ] || { echo "give at least one model key"; exit 2; }
MODEL=$1
shift
REST=("$@")
BASE="$ASKIISC_DATA/results/models/$MODEL"
export ASKIISC_MODEL=$MODEL ASKIISC_MAX_WORKERS=1
export ASKIISC_LOCUST_RESULTS="$BASE/locust" ASKIISC_MEASUREMENTS="$BASE/measurements"
mkdir -p "$ASKIISC_LOCUST_RESULTS" "$ASKIISC_MEASUREMENTS"
CURRENT="$ASKIISC_DATA/run/current.json"
echo "== $MODEL: serving stack"

bash slurm/serve.sh &
SERVE=$!
stop() { kill -USR1 "$SERVE" 2>/dev/null; kill -TERM "$SERVE" 2>/dev/null; wait "$SERVE"; }
trap 'stop; exit 1' TERM INT

# Wait until this job's stack is the one in current.json and answers.
start=$(date +%s)
until [ -f "$CURRENT" ] && grep -q "\"job\": \"$SLURM_JOB_ID\"" "$CURRENT" \
    && curl -sf "127.0.0.1:$(python3 -c "import json;print(json.load(open('$CURRENT'))['ports']['web'])")/healthz" > /dev/null 2>&1; do
  kill -0 "$SERVE" 2>/dev/null || { echo "serving stack exited"; exit 1; }
  [ $(( $(date +%s) - start )) -lt 1200 ] || { echo "stack not up in 20 min"; stop; exit 1; }
  sleep 10
done
echo "stack up after $(( $(date +%s) - start )) s"

failed=0
for workload in steady cold-start diurnal quality; do
  echo "== $MODEL: $workload ($(date +%T))"
  bash slurm/loadtest.sh "$workload" || { echo "$workload exited non-zero"; failed=$((failed + 1)); }
done
stop
echo "== $MODEL: stack stopped ($(date +%T))"

R="$ASKIISC_LOCUST_RESULTS/$SLURM_JOB_ID-quality"
if [ -f "$R/answers.json" ]; then
  uv run python -m eval.grader.rubric "$R/answers.json" "$ASKIISC_STATE_DIR" "$R/grade" \
    || failed=$((failed + 1))
  MODEL=granite-3.3-8b-instruct bash slurm/llm_job.sh \
    uv run python -m eval.grader.judge "$R/answers.json" "$ASKIISC_STATE_DIR" "$R/grade" \
    || failed=$((failed + 1))
else
  echo "no answers.json from the quality run"; failed=$((failed + 1))
fi
echo "== $MODEL done, $failed step(s) failed"

if [ ${#REST[@]} -gt 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" slurm/fr8_stack.sh "${REST[@]}")
  echo "next model ${REST[0]}: job $NEXT"
fi
exit $(( failed > 0 ))
