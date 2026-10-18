#!/bin/bash
# Run Makefile targets (default: all) inside Slurm, never on the login node.
# From the repo root: sbatch --wait slurm/gate.sh [target...]
# Keep going past failures: sbatch --export=ALL,MAKEFLAGS=-k slurm/gate.sh ...
# While dev is blocked by another user's GPU job, and a GPU is free:
#   sbatch --partition=low_2h_1gpu slurm/gate.sh ...
# Longer suites: sbatch --wait --time=02:00:00 slurm/gate.sh test-perf
#SBATCH --job-name=askiisc_gate
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh

TARGETS=("${@:-all}")
echo "== make ${TARGETS[*]} on $(hostname) at $(date -Is) =="
make "${TARGETS[@]}"
echo "gate ${TARGETS[*]} passed"
