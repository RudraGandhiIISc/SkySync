#!/bin/bash
# Fetch past editions of academic pages from the Internet Archive into
# $CORPUS_DIR/archive (resumable). Each job runs ASKIISC_ARCHIVE_SECONDS,
# checkpoints, and resubmits itself until every address is done (at most
# ARCHIVE_CHAIN_MAX jobs). Arguments: manifests whose academic addresses to
# look up (default: the corpus's own manifest.jsonl). From the repo root:
#   sbatch --export=ALL,CORPUS_DIR=/data4/$USER/askiisc/corpus-v3 \
#     slurm/archive.sh /data4/$USER/askiisc/corpus/manifest.jsonl
#SBATCH --job-name=askiisc_archive
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export ASKIISC_CORPUS_DIR=${CORPUS_DIR:-$ASKIISC_CORPUS_DIR}
export ASKIISC_ARCHIVE_SECONDS=${ASKIISC_ARCHIVE_SECONDS:-6600}
CHAIN=${ARCHIVE_CHAIN:-1}
CHAIN_MAX=${ARCHIVE_CHAIN_MAX:-12}

uv run python -m askiisc.ingest.archive "$@"
STOPPED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stopped_by"])' \
  "$ASKIISC_CORPUS_DIR/archive_report.json")
echo "chain $CHAIN/$CHAIN_MAX stopped by: $STOPPED"
if [ "$STOPPED" = "deadline" ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  sbatch --partition="$SLURM_JOB_PARTITION" \
    --export=ALL,ARCHIVE_CHAIN=$((CHAIN + 1)) slurm/archive.sh "$@"
  echo "resubmitted the next archive job"
fi
