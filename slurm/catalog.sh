#!/bin/bash
# Build the catalogue of a crawled corpus: remove duplicates, find editions,
# assign validity windows (catalog.jsonl), then rewrite policies.json and
# titles.json from it. CPU only (text extraction in parallel). From the
# repo root:
#   sbatch --export=ALL,CORPUS_DIR=/data4/$USER/askiisc/corpus-v3 slurm/catalog.sh
#SBATCH --job-name=askiisc_catalog
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_CORPUS_DIR=${CORPUS_DIR:-$ASKIISC_CORPUS_DIR}
# One thread per extraction process (Slurm pins no CPUs here).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv run python -m askiisc.ingest.catalog
uv run python -m askiisc.ingest.corpus
echo "catalogue done"
