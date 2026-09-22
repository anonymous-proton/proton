"""Workload feature extraction for Protenix. Stdlib only — no torch/protenix imports."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "protenix"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        for f in flags:
            if str(a).startswith(f + "="):
                return str(a).split("=", 1)[1]
            if str(a) in (f"--{f}", f"-{f}", f) and i + 1 < len(argv):
                return str(argv[i + 1])
    return None


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    seq = _get(argv, ["sequence"])
    if seq:
        result["num_residues"] = len(str(seq).strip())
        return result

    input_json_path = _get(argv, ["input_json_path"])
    if input_json_path:
        try:
            data = json.loads(Path(input_json_path).read_text())
            total = 0
            for item in (data if isinstance(data, list) else [data]):
                for chain in (item.get("sequences") or []):
                    for kind in ("protein", "proteinChain",
                                 "rna", "rnaChain",
                                 "dna", "dnaChain"):
                        seq_val = (chain.get(kind) or {}).get("sequence") or ""
                        total += len(str(seq_val))
            if total > 0:
                result["num_residues"] = total
        except Exception:
            pass
    return result
