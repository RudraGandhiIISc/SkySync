#!/bin/bash
# Fetch or build the service binaries into $ASKIISC_SCRATCH/bin (no root):
# Redis (built from source) and gitleaks (secrets scan). MinIO no longer
# publishes community binaries (HTTP 410), so originals are stored as files.
# From the repo root: sbatch slurm/fetch_services.sh
#SBATCH --job-name=askiisc_services
#SBATCH --partition=dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/lib.sh
REDIS_VERSION=7.2.5      # last BSD-licensed line
GITLEAKS_VERSION=8.18.4
BIN="$ASKIISC_SCRATCH/bin"
BUILD=$(mktemp -d "$ASKIISC_SCRATCH/build.XXXX")
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BIN"

curl -fsSL "https://download.redis.io/releases/redis-$REDIS_VERSION.tar.gz" \
  | tar -xz -C "$BUILD"
make -s -C "$BUILD/redis-$REDIS_VERSION" -j8 BUILD_TLS=no MALLOC=libc >/dev/null
cp "$BUILD/redis-$REDIS_VERSION/src/redis-server" "$BIN/"
"$BIN/redis-server" --version

curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v$GITLEAKS_VERSION/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
  | tar -xz -C "$BIN" gitleaks
"$BIN/gitleaks" version
echo "services fetched"
