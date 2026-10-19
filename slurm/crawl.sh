#!/bin/bash
# Crawl public IISc admissions, academic-policy, course and handbook pages
# plus the CDS and CSA sites into $ASKIISC_DATA/corpus/public, then rebuild
# policies.json (public + restricted). From the repo root:
#   sbatch slurm/crawl.sh
#SBATCH --job-name=askiisc_crawl
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_CORPUS_DIR="$ASKIISC_DATA/corpus"
mkdir -p "$ASKIISC_CORPUS_DIR"
uv run python -m askiisc.ingest.crawl
uv run python -m askiisc.ingest.corpus
du -sh "$ASKIISC_CORPUS_DIR"/public
echo "crawl done"
