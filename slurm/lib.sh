# Shared paths and helpers for Ask IISc Slurm jobs. Source it; do not run it.
# Jobs are submitted from the repository root, so $SLURM_SUBMIT_DIR is the repo.

export ASKIISC_REPO=${ASKIISC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
export ASKIISC_SCRATCH=${ASKIISC_SCRATCH:-/data2/scratch/$USER/askiisc}
export ASKIISC_DATA=${ASKIISC_DATA:-/data4/$USER/askiisc}
export HF_HOME=${HF_HOME:-/data4/$USER/huggingface}
export UV_PROJECT_ENVIRONMENT="$ASKIISC_SCRATCH/.venv"
export UV_CACHE_DIR="$ASKIISC_SCRATCH/uv-cache"
export SERVE_VENV="$ASKIISC_SCRATCH/serve-venv"
export PATH="$HOME/.local/bin:$ASKIISC_SCRATCH/bin:$PATH"

# Everything a job writes is readable only by you (shared server).
umask 077
# Real users live only in this private file, never in git.
if [ -z "${ASKIISC_DIRECTORY_PATH:-}" ] && \
   [ -d "$ASKIISC_DATA/private/users" ]; then
  export ASKIISC_DIRECTORY_PATH="$ASKIISC_DATA/private/users"
fi

# pick_gpu [exclude_csv]
# Prints "<index> <free_MiB> <total_MiB>" for the least busy H200, ordered by
# utilisation and then free memory. Slurm reserves no GPU here (Gres=(null)),
# so the caller must export CUDA_VISIBLE_DEVICES=<index> itself.
pick_gpu() {
  local exclude=",${1:-},"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total \
      --format=csv,noheader,nounits \
    | tr -d ' ' \
    | awk -F, -v ex="$exclude" 'index(ex, "," $1 ",") == 0' \
    | sort -t, -k2,2n -k3,3nr \
    | head -1 \
    | awk -F, '{print $1, $3, $4}'
}

# mem_fraction <free_MiB> <total_MiB>
# Prints a vLLM --gpu-memory-utilization value that uses the card's free
# memory minus 4 GiB of headroom, capped at 0.90.
mem_fraction() {
  awk -v f="$1" -v t="$2" \
    'BEGIN { x = (f - 4096) / t; if (x > 0.90) x = 0.90; printf "%.2f\n", x }'
}

# free_port: prints a free TCP port on 127.0.0.1.
free_port() {
  python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}
