#!/bin/bash
# Headless Locust run against the serving job (slurm/serve.sh must be up).
# From the repo root:
#   sbatch slurm/loadtest.sh steady
#   sbatch slurm/loadtest.sh overload --peak-users 48
#   sbatch slurm/loadtest.sh steady --duration-scale 0.1      # rehearsal
#   sbatch --time=05:00:00 slurm/loadtest.sh soak
# Workloads and the requirements they test: eval/locust/workloads.py.
# Results: $ASKIISC_DATA/results/locust/<job>-<workload>/ (summary.json,
# report.html, CSVs); also on the admin dashboard's "Load tests" tab.
#SBATCH --job-name=askiisc_load
# dev waits while any GPU job (the serving job) runs, so this CPU job uses
# a free 1-GPU partition (approved for short CPU jobs) and no GPU.
#SBATCH --partition=low_2h_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
WORKLOAD=${1:-steady}
shift || true
CURRENT="$ASKIISC_DATA/run/current.json"
if [ ! -s "$CURRENT" ]; then
  echo "no serving job is running (start: sbatch slurm/serve.sh)"; exit 1
fi
PORT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ports']['web'])" "$CURRENT")
export ASKIISC_LOCUST_ACCOUNTS="$ASKIISC_DATA/private/app/locust_accounts.json"
# FR-8 runs of other models go elsewhere so the scorecard keeps the
# serving model's results (slurm/fr8_stack.sh sets this).
export ASKIISC_LOCUST_RESULTS=${ASKIISC_LOCUST_RESULTS:-$ASKIISC_DATA/results/locust}
export ASKIISC_PROM_TARGETS="$ASKIISC_DATA/run/prom_targets"
export ASKIISC_LOCUST_RUN="$SLURM_JOB_ID"
OUT="$ASKIISC_LOCUST_RESULTS/$SLURM_JOB_ID-$WORKLOAD"
mkdir -p "$OUT"
echo "== locust $WORKLOAD against 127.0.0.1:$PORT, results in $OUT"
STATUS=0
uv run locust -f eval/locust/locustfile.py --headless --only-summary \
  --host "http://127.0.0.1:$PORT" --workload "$WORKLOAD" \
  --csv "$OUT/locust" --html "$OUT/report.html" "$@" || STATUS=$?
# Scaling and cost from Prometheus for the runs that measure them.
case "$WORKLOAD" in
  steady|spike|cold-start|diurnal|fairness)
    uv run python -m eval.scaling "$OUT/summary.json" \
      ${ASKIISC_MEASUREMENTS:+--results "$ASKIISC_MEASUREMENTS"} \
      || echo "scaling query failed" ;;
esac
# Locust exits 1 when any request failed; report that after the query.
exit $STATUS
