"""Workload feature extraction for RFDiffusion. Stdlib only — no torch/rfdiffusion imports."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "rfdiffusion"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                val = str(a).split("=", 1)[1]
                val = val.strip("'\"")
                return val
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1]).strip("'\"")
    return None


def _contig_length(contigs: Any) -> int:
    if not contigs:
        return 0
    if isinstance(contigs, list):
        parts = [str(c) for c in contigs]
    else:
        parts = [str(contigs).strip().strip("[]").replace("'", "").replace('"', "")]
    total = 0
    for cstr in parts:
        for token in re.split(r"[,/\s]+", cstr):
            token = token.strip()
            if not token:
                continue
            m = re.fullmatch(r"[A-Za-z](\d+)-(\d+)", token)
            if m:
                total += int(m.group(2)) - int(m.group(1)) + 1
                continue
            m = re.fullmatch(r"(\d+)-(\d+)", token)
            if m:
                total += int(m.group(2))
                continue
            m = re.fullmatch(r"(\d+)", token)
            if m:
                total += int(m.group(1))
    return total


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

    raw = _get(argv, ["contigmap.contigs", "contigs"])
    total = _contig_length(raw)
    if total > 0:
        result["scaffold_length"] = total
        return result

    input_pdb = _get(argv, ["inference.input_pdb"])
    if input_pdb:
        n = _count_residues_from_pdb(input_pdb)
        if n > 0:
            result["num_residues"] = n

    return result
