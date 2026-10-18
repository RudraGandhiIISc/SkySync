#!/bin/bash
# One-time environment setup (rerun after changing uv.lock or
# serve-requirements.txt). From the repo root: sbatch slurm/setup_env.sh
#   project venv: Python 3.11, from pyproject.toml / uv.lock (no torch, no vLLM)
#   serve venv:   Python 3.12, vLLM and the embedding server (GPU side)
#SBATCH --job-name=askiisc_setup
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh

mkdir -p "$ASKIISC_SCRATCH/bin" "$HF_HOME" \
  "$ASKIISC_DATA"/{data,state,results,models}

if [ -f pyproject.toml ]; then
  uv python install 3.11
  if [ -f uv.lock ]; then uv sync --locked; else uv sync; fi
  "$UV_PROJECT_ENVIRONMENT/bin/python" --version
else
  echo "no pyproject.toml yet (Phase 0): skipping the project venv"
fi

[ -d "$SERVE_VENV" ] || uv venv --python 3.12 "$SERVE_VENV"
uv pip install --python "$SERVE_VENV/bin/python" -r slurm/serve-requirements.txt
"$SERVE_VENV/bin/python" -c \
  'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'

quota -v -s | tail -n +2
echo "setup done"
