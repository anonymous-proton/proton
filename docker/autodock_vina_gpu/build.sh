#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-autodock_vina_gpu}"
TAG="${TAG:-latest}"
SOURCE_IMAGE="${SOURCE_IMAGE:-fovus/vina-gpu-2.1:autodock-vina-gpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker pull "${SOURCE_IMAGE}"

MODE="${VINA_BUILD_MODE:-loadonly}"
if [[ "$MODE" == "source" ]]; then
	docker build -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
	exit 0
fi

ARCH="${ARCH:-sm_86}"
BIN_DIR="$SCRIPT_DIR/bins-${ARCH}"
STAGE="$SCRIPT_DIR/.kernels-staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"

if [[ -d "$BIN_DIR" ]]; then
	cp "$BIN_DIR"/Kernel1_Opt.bin "$BIN_DIR"/Kernel2_Opt.bin "$STAGE"/
else
	CID="$(docker create "$SOURCE_IMAGE" true)"
	docker cp "$CID":/vina/AutoDock-Vina-GPU-2.1/Kernel1_Opt.bin "$STAGE"/ >/dev/null
	docker cp "$CID":/vina/AutoDock-Vina-GPU-2.1/Kernel2_Opt.bin "$STAGE"/ >/dev/null
	docker rm "$CID" >/dev/null
fi

docker build -f "$SCRIPT_DIR/Dockerfile.loadonly" -t "${IMAGE_NAME}:${TAG}" "$SCRIPT_DIR"
