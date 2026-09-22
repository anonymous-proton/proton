"""Workload feature extraction for ESM. Stdlib only — no torch imports."""
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "esm"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1].strip("'\"")
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1]).strip("'\"")
    return None


def _read_fasta_total_length(path: str) -> int:
    """Read a FASTA file and return total sequence length (all chains)."""
    try:
        total = 0
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            total += len(line)
        return total
    except Exception:
        return 0


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    seq = _get(argv, ["ref_sequence"]) or env.get("GW_SEQUENCE") or ""
    if seq:
        n = len(str(seq).strip())
        if n > 0:
            result["sequence_length"] = n
            return result

    fasta_path = _get(argv, ["-i", "--input", "--ref_fasta_file", "ref_fasta_file"])
    if fasta_path:
        n = _read_fasta_total_length(fasta_path)
        if n > 0:
            result["sequence_length"] = n

    return result
