#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-rfdiffusion}"
TAG="${TAG:-latest}"

SRC_DIR="$PROJECT_ROOT/third_parties/RFdiffusion"
DST_DIR="$SCRIPT_DIR/RFdiffusion"
MODELWORKER_SRC="$PROJECT_ROOT/modelworker"
MODELWORKER_DST="$SCRIPT_DIR/modelworker"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Expected source directory not found: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$MODELWORKER_SRC" ]]; then
  echo "[ERROR] Expected modelworker directory not found: $MODELWORKER_SRC" >&2
  exit 1
fi

cleanup() {
  rm -rf "$DST_DIR" "$MODELWORKER_DST"
}
trap cleanup EXIT

rm -rf "$DST_DIR" "$MODELWORKER_DST"
cp -R "$SRC_DIR" "$DST_DIR"
cp -R "$MODELWORKER_SRC" "$MODELWORKER_DST"

docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
