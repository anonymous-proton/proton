"""Workload feature extraction for ProteinMPNN. Stdlib only — no torch imports."""
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "proteinmpnn"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1].strip("'\"")
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1]).strip("'\"")
    return None


def _count_residues_from_pdb(pdb_path: str) -> int:
    """Count unique residues (CA atoms) in a PDB file."""
    try:
        seen: set = set()
        for line in Path(pdb_path).read_text().splitlines():
            if line.startswith(("ATOM  ", "ATOM ")) and line[12:16].strip() == "CA":
                seen.add((line[21], line[22:26].strip()))
        return len(seen)
    except Exception:
        return 0


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    pdb_path = _get(argv, ["pdb_path", "--pdb_path"])
    if pdb_path:
        n = _count_residues_from_pdb(pdb_path)
        if n > 0:
            result["num_residues"] = n
            return result

        import re
        m = re.search(r"(\d+)res", Path(pdb_path).name)
        if m:
            result["num_residues"] = int(m.group(1))

    return result
