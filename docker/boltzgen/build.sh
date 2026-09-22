#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-boltzgen}"
TAG="${TAG:-latest}"

BOLTZGEN_SRC="$PROJECT_ROOT/third_parties/boltzgen"

if [[ ! -d "$BOLTZGEN_SRC" ]]; then
  echo "[ERROR] boltzgen directory not found: $BOLTZGEN_SRC" >&2
  exit 1
fi

cleanup() {
  echo "[INFO] Cleaning up temporary boltzgen files"
  rm -rf \
    "$SCRIPT_DIR/pyproject.toml" \
    "$SCRIPT_DIR/src" \
    "$SCRIPT_DIR/PYPI_DESCRIPTION.md" \
    "$SCRIPT_DIR/README.md"
}
trap cleanup EXIT

echo "[INFO] Staging boltzgen package into docker build context"

cp "$BOLTZGEN_SRC/pyproject.toml" "$SCRIPT_DIR/"
cp -r "$BOLTZGEN_SRC/src" "$SCRIPT_DIR/"

[[ -f "$BOLTZGEN_SRC/PYPI_DESCRIPTION.md" ]] && cp "$BOLTZGEN_SRC/PYPI_DESCRIPTION.md" "$SCRIPT_DIR/"
[[ -f "$BOLTZGEN_SRC/README.md" ]] && cp "$BOLTZGEN_SRC/README.md" "$SCRIPT_DIR/"

echo "[INFO] Building docker image ${IMAGE_NAME}:${TAG}"
docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"

echo "[INFO] Build finished successfully 🎉"
