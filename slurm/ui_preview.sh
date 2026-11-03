#!/bin/bash
# Screenshots of the web app on fakes (no GPU, no real data): starts
# tools/ui_preview.py, runs tools/ui_shots.js in headless Chromium, stops.
# From the repo root: sbatch --wait slurm/ui_preview.sh
# While dev is blocked: sbatch --wait --partition=low_2h_1gpu slurm/ui_preview.sh
# Needs Node and Playwright in $ASKIISC_TOOLS (default /data4/$USER/tools).
#SBATCH --job-name=askiisc_ui
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh

TOOLS="${ASKIISC_TOOLS:-/data4/$USER/tools}"
OUT="$ASKIISC_DATA/results/ui/$SLURM_JOB_ID"
mkdir -p "$OUT"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
PASSWORD=$(uv run python -c 'from tests.web_fixtures import PASSWORD; print(PASSWORD)')

uv run python -m tools.ui_preview "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 60); do
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break
  sleep 1
done

export PATH="$TOOLS/node-v20.18.0-linux-x64/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH="$TOOLS/ms-playwright"
export PLAYWRIGHT_MODULE="$TOOLS/pw/node_modules/playwright-core"
node tools/ui_shots.js "http://127.0.0.1:$PORT" "$OUT" "$PASSWORD"
echo "screenshots in $OUT"
