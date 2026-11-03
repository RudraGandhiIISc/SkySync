#!/bin/bash
# Fetch or build the service binaries (no root): Redis (built from source)
# and gitleaks into $ASKIISC_SCRATCH/bin; Prometheus and Grafana (for
# slurm/serve.sh) into $ASKIISC_DATA/services. Present ones are skipped.
# MinIO no longer publishes community binaries (HTTP 410), so originals are
# stored as files.
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
PROMETHEUS_VERSION=2.53.2   # long-term-support line
GRAFANA_VERSION=11.2.0
LOKI_VERSION=3.1.1          # log store (released 2024-08)
ALLOY_VERSION=1.3.1         # log shipper, successor of Promtail (2024-08)
BIN="$ASKIISC_SCRATCH/bin"
SERVICES="$ASKIISC_DATA/services"
BUILD=$(mktemp -d "$ASKIISC_SCRATCH/build.XXXX")
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BIN"

if [ ! -x "$BIN/redis-server" ]; then
  curl -fsSL "https://download.redis.io/releases/redis-$REDIS_VERSION.tar.gz" \
    | tar -xz -C "$BUILD"
  make -s -C "$BUILD/redis-$REDIS_VERSION" -j8 BUILD_TLS=no MALLOC=libc >/dev/null
  cp "$BUILD/redis-$REDIS_VERSION/src/redis-server" "$BIN/"
fi
"$BIN/redis-server" --version

if [ ! -x "$BIN/gitleaks" ]; then
  curl -fsSL "https://github.com/gitleaks/gitleaks/releases/download/v$GITLEAKS_VERSION/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
    | tar -xz -C "$BIN" gitleaks
fi
"$BIN/gitleaks" version

mkdir -p "$SERVICES/prometheus" "$SERVICES/grafana"
if [ ! -x "$SERVICES/prometheus/prometheus" ]; then
  curl -fsSL "https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz" \
    | tar -xz -C "$SERVICES/prometheus" --strip-components=1
fi
"$SERVICES/prometheus/prometheus" --version | head -1

if [ ! -x "$SERVICES/grafana/bin/grafana" ]; then
  curl -fsSL "https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.linux-amd64.tar.gz" \
    | tar -xz -C "$SERVICES/grafana" --strip-components=1
fi
"$SERVICES/grafana/bin/grafana" --version

mkdir -p "$SERVICES/loki" "$SERVICES/alloy"
if [ ! -x "$SERVICES/loki/loki" ]; then
  curl -fsSL -o "$BUILD/loki.zip" \
    "https://github.com/grafana/loki/releases/download/v$LOKI_VERSION/loki-linux-amd64.zip"
  unzip -q -o "$BUILD/loki.zip" -d "$BUILD"
  install -m 755 "$BUILD/loki-linux-amd64" "$SERVICES/loki/loki"
fi
"$SERVICES/loki/loki" --version | head -1

if [ ! -x "$SERVICES/alloy/alloy" ]; then
  curl -fsSL -o "$BUILD/alloy.zip" \
    "https://github.com/grafana/alloy/releases/download/v$ALLOY_VERSION/alloy-linux-amd64.zip"
  unzip -q -o "$BUILD/alloy.zip" -d "$BUILD"
  install -m 755 "$BUILD/alloy-linux-amd64" "$SERVICES/alloy/alloy"
fi
"$SERVICES/alloy/alloy" --version | head -1
echo "services fetched"
