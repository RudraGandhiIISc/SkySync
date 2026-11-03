#!/bin/bash
# Crawl every public IISc site into $ASKIISC_CORPUS_DIR (resumable).
# Each job crawls for ASKIISC_CRAWL_SECONDS, checkpoints, and resubmits
# itself until the crawl is complete (at most CRAWL_CHAIN_MAX jobs), then
# submits the catalogue rebuild and, after it, the GPU ingest (INGEST_AFTER=0
# to skip both). From the repo root:
#   sbatch slurm/crawl.sh
# While dev is blocked by another user's GPU job and a GPU is free:
#   sbatch --partition=low_2h_1gpu slurm/crawl.sh
#SBATCH --job-name=askiisc_crawl
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_CORPUS_DIR=${CORPUS_DIR:-$ASKIISC_CORPUS_DIR}
export ASKIISC_CRAWL_SECONDS=${ASKIISC_CRAWL_SECONDS:-6600}
CHAIN=${CRAWL_CHAIN:-1}
CHAIN_MAX=${CRAWL_CHAIN_MAX:-12}
mkdir -p "$ASKIISC_CORPUS_DIR"

uv run python -m askiisc.ingest.crawl
STOPPED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stopped_by"])' \
  "$ASKIISC_CORPUS_DIR/crawl_report.json")
echo "chain $CHAIN/$CHAIN_MAX stopped by: $STOPPED"
if [ "$STOPPED" = "deadline" ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  sbatch --partition="$SLURM_JOB_PARTITION" \
    --export=ALL,CRAWL_CHAIN=$((CHAIN + 1)) slurm/crawl.sh
  echo "resubmitted the next crawl job"
  exit 0
fi
du -sh "$ASKIISC_CORPUS_DIR"/public
echo "crawl done"
# Some sites were down (robots.txt failed): try them again in 30 minutes.
RETRIES=${CRAWL_RETRIES:-6}
if [ "$STOPPED" = "hosts unavailable" ] && [ "$RETRIES" -gt 0 ]; then
  sbatch --begin=now+30minutes --partition="$SLURM_JOB_PARTITION" \
    --export=ALL,CRAWL_CHAIN=1,CRAWL_RETRIES=$((RETRIES - 1)),INGEST_AFTER=0 \
    slurm/crawl.sh
  echo "hosts down: retry submitted ($RETRIES left)"
  exit 0
fi
if [ "${INGEST_AFTER:-1}" = "1" ]; then
  # Ingest reads catalog.jsonl, so the catalogue is rebuilt first; then
  # embed on a GPU (the ingest chain resubmits itself until done).
  CAT=$(sbatch --parsable --partition="$SLURM_JOB_PARTITION" \
    --export=ALL,CORPUS_DIR="$ASKIISC_CORPUS_DIR" slurm/catalog.sh)
  sbatch --dependency=afterok:"$CAT" --partition=low_2h_1gpu \
    --export=ALL,EMBED_DEVICE=cuda slurm/ingest.sh
  echo "submitted the catalogue ($CAT) and the GPU ingest after it"
fi
