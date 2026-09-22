"""Workload feature extraction for Vina GPU. Stdlib only — no heavy deps."""
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "vina_gpu"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1]
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1])
    return None


def _resolve_path(raw: str, workdir: str) -> Path:
    """Resolve a path that may be relative to workdir."""
    p = Path(raw)
    if p.is_absolute():
        return p
    if workdir:
        return Path(workdir) / p
    return p


def _count_pdbqt_atoms(path: Path) -> int:
    """Count ATOM/HETATM lines in a PDBQT file."""
    try:
        count = 0
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM  ", "ATOM ", "HETATM")):
                count += 1
        return count
    except Exception:
        return 0


def _count_pdbqt_torsions(path: Path) -> int:
    """Read TORSDOF from a PDBQT ligand file."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("TORSDOF"):
                return int(line.split()[1])
        return 0
    except Exception:
        return 0


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    workdir = env.get("_GW_WORKDIR", "")

    try:
        sx = float(_get(argv, ["size_x", "--size_x"]) or 0)
        sy = float(_get(argv, ["size_y", "--size_y"]) or 0)
        sz = float(_get(argv, ["size_z", "--size_z"]) or 0)
        vol = int(sx * sy * sz)
        if vol > 0:
            result["search_volume"] = vol
    except (ValueError, TypeError):
        pass

    ligand_raw = _get(argv, ["ligand", "--ligand"])
    if ligand_raw:
        ligand_path = _resolve_path(ligand_raw, workdir)
        n = _count_pdbqt_atoms(ligand_path)
        if n > 0:
            result["ligand_atoms"] = n
        t = _count_pdbqt_torsions(ligand_path)
        if t > 0:
            result["ligand_torsions"] = t

    receptor_raw = _get(argv, ["receptor", "--receptor"])
    if receptor_raw:
        receptor_path = _resolve_path(receptor_raw, workdir)
        n = _count_pdbqt_atoms(receptor_path)
        if n > 0:
            result["receptor_atoms"] = n

    return result


UNLIMITED_RETRIES = True


def is_retryable_error(error_message: str) -> bool:
    """Only retry on exit 255 (OpenCL OOM). Exit 1 = docking failure, not transient."""
    return "255" in error_message


def adjust_argv_for_retry(argv: List[str], attempt: int) -> List[str]:
    """Keep --thread unchanged on retry.

    Exit 255 (OpenCL OOM) is typically caused by transient GPU memory
    pressure from co-located tasks, not by the thread count itself.
    Reducing threads degrades docking parallelism and can cause exit 1
    (docking failure).  Retrying with the same argv after the gateway's
    re-scheduling cycle usually succeeds once co-located tasks finish.
    """
    return argv
