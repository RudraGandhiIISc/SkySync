#!/bin/bash
# Download model weights into $HF_HOME (/data4, no quota) so runs stay offline.
# From the repo root: sbatch slurm/fetch_models.sh Qwen/Qwen3-4B-Instruct-2507 [more repo ids]
# Gated models (Gemma-3, Llama-3.1) need an accepted licence and a token in
# $HF_HOME/token or ~/.cache/huggingface/token (mode 600).
#SBATCH --job-name=askiisc_fetch
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh

[ "$#" -gt 0 ] || { echo "usage: sbatch slurm/fetch_models.sh <hf repo id>..."; exit 2; }

"$SERVE_VENV/bin/python" - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download

for repo in sys.argv[1:]:
    path = snapshot_download(
        repo, ignore_patterns=["original/*", "*.pth", "*.gguf"])
    print(repo, "->", path)
EOF
du -sh "$HF_HOME/hub" 2>/dev/null || true
echo "fetch done"
