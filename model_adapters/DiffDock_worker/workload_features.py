"""Workload feature extraction and retry hooks for DiffDock.

Stdlib only -- no torch / rdkit imports on the gateway host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "diffdock"
INPUT_SIZE_KEY = "protein_residues"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1]
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1])
    return None


def _resolve_path(raw: str, workdir: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if workdir:
        return Path(workdir) / p
    return p


def _count_protein_residues(path: Path) -> int:
    try:
        seen: set[tuple[str, str]] = set()
        for line in path.read_text().splitlines():
            if not line.startswith(("ATOM  ", "ATOM ")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain_id = line[21].strip()
            resseq = line[22:26].strip()
            if resseq:
                seen.add((chain_id, resseq))
        return len(seen)
    except Exception:
        return 0


def _count_ligand_atoms(path: Path) -> int:
    try:
        suffix = path.suffix.lower()
        lines = path.read_text().splitlines()
        if suffix == ".mol2":
            in_atom_block = False
            count = 0
            for line in lines:
                if line.startswith("@<TRIPOS>ATOM"):
                    in_atom_block = True
                    continue
                if line.startswith("@<TRIPOS>") and in_atom_block:
                    break
                if in_atom_block and line.strip():
                    count += 1
            return count
        if suffix == ".sdf":
            if len(lines) >= 4:
                counts = lines[3]
                try:
                    return int(counts[:3].strip() or "0")
                except Exception:
                    return 0
    except Exception:
        return 0
    return 0


def _positive_int(raw: Optional[str]) -> int:
    try:
        val = int(str(raw).strip())
    except Exception:
        return 0
    return val if val > 0 else 0


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    workdir = env.get("_GW_WORKDIR", "")

    protein_raw = _get(argv, ["protein_path"])
    if protein_raw:
        protein_path = _resolve_path(protein_raw, workdir)
        residues = _count_protein_residues(protein_path)
        if residues > 0:
            result["protein_residues"] = residues

    ligand_raw = _get(argv, ["ligand_description"])
    if ligand_raw:
        ligand_path = _resolve_path(ligand_raw, workdir)
        atoms = _count_ligand_atoms(ligand_path)
        if atoms > 0:
            result["ligand_atoms"] = atoms

    for key in ("samples_per_complex", "inference_steps", "actual_steps", "batch_size"):
        value = _positive_int(_get(argv, [key]))
        if value > 0:
            result[key] = value

    return result


UNLIMITED_RETRIES = True


def is_retryable_error(error_message: str) -> bool:
    """Retry all non-input/config DiffDock failures indefinitely.

    DiffDock has historically mixed genuine data / graph-construction
    / runtime errors with true input/config failures. For the gateway we
    want DiffDock process-level failures to stay inside the scheduler's
    retry loop instead of being elevated to Nextflow. Therefore only
    clearly permanent file/argv/config errors fail fast; graph build,
    model runtime, CUDA allocator, and similar process failures are all
    treated as retryable.
    """

    err = str(error_message or "").lower()

    permanent_markers = (
        "no such file or directory",
        "filenotfounderror",
        "does not exist",
        "permission denied",
    )
    for marker in permanent_markers:
        if marker in err:
            return False

    return True


def adjust_argv_for_retry(argv: List[str], attempt: int) -> List[str]:
    """Keep DiffDock argv unchanged across retry attempts.

    The retry is intended to re-enter placement under a different
    co-location state, not to alter DiffDock sampling semantics.
    """

    return argv
