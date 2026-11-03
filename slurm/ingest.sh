#!/bin/bash
# Chunk, embed (bge-m3) and store the corpus in the persistent index
# $ASKIISC_STATE_DIR (slurm/lib.sh). Rerunning ingests only new or changed documents.
# Each job stops at INGEST_SECONDS and resubmits itself until done.
# From the repo root:
#   sbatch slurm/ingest.sh                                  # CPU, real corpus
#   sbatch --partition=low_2h_1gpu --export=ALL,EMBED_DEVICE=cuda slurm/ingest.sh
#   sbatch --export=ALL,CORPUS=eval/datasets/sample/corpus,STATE=... slurm/ingest.sh
#SBATCH --job-name=askiisc_ingest
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # torch would see all 256 cores
CORPUS=${CORPUS:-$ASKIISC_CORPUS_DIR}
STATE=${STATE:-$ASKIISC_STATE_DIR}
EMBED_DEVICE=${EMBED_DEVICE:-cpu}
INGEST_SECONDS=${INGEST_SECONDS:-6300}
CHAIN=${INGEST_CHAIN:-1}
CHAIN_MAX=${INGEST_CHAIN_MAX:-12}
RUN_DIR="$ASKIISC_DATA/results/ingest/$SLURM_JOB_ID"
trap 'pkill -TERM -P $$ 2>/dev/null || true' EXIT

if [ "$EMBED_DEVICE" = "cuda" ]; then
  read -r GPU FREE TOTAL < <(pick_gpu)
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "embedding on GPU $GPU (${FREE} MiB free of ${TOTAL})"
fi
export ASKIISC_EMBED_DEVICE=$EMBED_DEVICE
export ASKIISC_RERANK_MODEL=""   # ingestion only embeds

# DB_DIR=<folder on /data2/scratch>: keep versions.db and milvus.db on
# local disk while ingesting (NFS commits are ~30x slower); they are copied
# into $STATE when the chain finishes.
DB_ARGS=()
if [ -n "${DB_DIR:-}" ]; then
  mkdir -p "$DB_DIR"
  DB_ARGS=(--db-dir "$DB_DIR")
fi
uv run python -m deploy.ingest --corpus "$CORPUS" --state-dir "$STATE" \
  --run-dir "$RUN_DIR" --deadline-s "$INGEST_SECONDS" "${DB_ARGS[@]}"
STATUS=$?
if [ -n "${DB_DIR:-}" ] && [ "$STATUS" -ne 3 ]; then
  for f in versions.db milvus.db; do   # milvus.db is a folder
    rm -rf "$STATE/$f.tmp"
    cp -a "$DB_DIR/$f" "$STATE/$f.tmp" || exit 1
    if [ -e "$STATE/$f" ]; then mv "$STATE/$f" "$STATE/$f.old"; fi
    mv "$STATE/$f.tmp" "$STATE/$f" && rm -rf "$STATE/$f.old"
  done
  echo "databases copied from $DB_DIR to $STATE"
fi
du -sh "$STATE"
if [ "$STATUS" -eq 3 ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  sbatch --partition="$SLURM_JOB_PARTITION" \
    --export=ALL,INGEST_CHAIN=$((CHAIN + 1)) slurm/ingest.sh
  echo "stopped at the deadline: resubmitted ($CHAIN/$CHAIN_MAX)"
  exit 0
fi
echo "ingest finished with status $STATUS"
exit $STATUS
