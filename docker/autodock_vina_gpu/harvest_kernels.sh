#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD=1
if [[ "${1:-}" == "--no-build" ]]; then
	BUILD=0
fi

ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '. ')"
HARVEST_IMAGE="${HARVEST_IMAGE:-autodock_vina_gpu:harvest}"
OUT="$SCRIPT_DIR/bins-${ARCH}"

RECEPTOR="$SCRIPT_DIR/harvest_input/receptor.pdbqt"
LIGAND="$SCRIPT_DIR/harvest_input/ligand.pdbqt"
[[ -f "$RECEPTOR" && -f "$LIGAND" ]] || {
	echo "missing $RECEPTOR / $LIGAND" >&2
	exit 2
}

if ! docker image inspect "$HARVEST_IMAGE" >/dev/null 2>&1; then
	echo "[harvest] building source-compile image $HARVEST_IMAGE ..."
	VINA_BUILD_MODE=source TAG="${HARVEST_IMAGE##*:}" bash "$SCRIPT_DIR/build.sh"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/ligand_input" "$WORK/output_ligands"
cp "$RECEPTOR" "$WORK/rec.pdbqt"
cp "$LIGAND" "$WORK/ligand_input/ligand.pdbqt"

docker run --rm --gpus all \
	-v "$WORK":/harvest -w /harvest \
	"$HARVEST_IMAGE" bash -c '
        ln -s /vina/AutoDock-Vina-GPU-2.1/OpenCL /harvest/OpenCL
        /vina/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1 \
            --opencl_binary_path /harvest \
            --receptor rec.pdbqt \
            --ligand_directory ligand_input/ \
            --output_directory output_ligands/ \
            --center_x -1.494 --center_y 0.135 --center_z -0.880 \
            --size_x 25 --size_y 25 --size_z 25 --thread 2048
    ' >/dev/null

mkdir -p "$OUT"
cp "$WORK"/Kernel1_Opt.bin "$WORK"/Kernel2_Opt.bin "$OUT"/
echo "[harvest] bins: $OUT/Kernel{1,2}_Opt.bin (arch sm_${ARCH})"

if [[ "$BUILD" == 1 ]]; then
	(cd "$REPO_ROOT" && ARCH="$ARCH" python3 docker/build.py autodock_vina_gpu)
	docker tag autodock_vina_gpu:worker "autodock_vina_gpu:sm_${ARCH}"
	echo "[harvest] built: autodock_vina_gpu:sm_${ARCH} (load-only binary + worker layer)"
fi
