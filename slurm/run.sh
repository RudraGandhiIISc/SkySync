#!/bin/bash
# Run one command inside Slurm with the project environment.
# From the repo root:
#   sbatch slurm/run.sh uv run python -m <module> [args]
#   sbatch --partition=low_2h_1gpu --time=00:40:00 --mem=8G slurm/run.sh ...
#SBATCH --job-name=askiisc_run
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "== $* on $(hostname) at $(date -Is) =="
"$@"
