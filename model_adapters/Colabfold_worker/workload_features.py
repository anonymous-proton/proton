"""Workload feature extraction for MMSeqs2/Colabfold. Stdlib only — no colabfold imports."""
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "mmseqs2"


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

    seq = _get(argv, ["sequence"]) or env.get("GW_SEQUENCE") or ""
    if seq:
        result["sequence_length"] = len(str(seq).strip())
        return result

    fasta_path = _get(argv, ["fasta_path", "-i"])
    if not fasta_path:
        for a in argv[1:]:
            if not str(a).startswith("-") and str(a).rstrip("/").endswith((".fasta", ".fa")):
                fasta_path = str(a)
                break
    if fasta_path:
        try:
            total = sum(
                len(line.strip())
                for line in Path(fasta_path).read_text().splitlines()
                if line.strip() and not line.startswith(">")
            )
            if total > 0:
                result["sequence_length"] = total
        except Exception:
            pass
    return result


UNLIMITED_RETRIES = True


def is_retryable_error(error_message: str) -> bool:
    """Retry transient GPU/contention failures indefinitely.

    Same precedent as vinaGPU (exit 255 OpenCL OOM → unlimited retry).
    For mmseqs2, the binary runs with ``--gpu 1`` and exits with status 1
    under heavy CUDA co-location pressure — observed in exp2c proton-naive
    smoke (): 3 mmseqs2 tasks in c2 hit the 5-attempt cap then
    surfaced as terminal FAILED.  Solo runs never reproduce this; under
    contention it self-resolves once co-located tasks finish.

    Permanent input-side errors (missing FASTA, missing DB, malformed
    sequence) surface as different strings and are NOT retried — they
    fall through to the gateway's terminal-failure path.
    """
    err = str(error_message or "").lower()

    permanent_markers = (
        "no such file or directory",
        "filenotfounderror",
        "database not found",
        "invalid sequence",
        "permission denied",
        "is a directory",
        "does not exist",
    )
    for m in permanent_markers:
        if m in err:
            return False

    transient_markers = (
        "mmseqs2 execution failed",
        "cuda out of memory",
        "execute_batch failed",
        "non-zero exit status",
    )
    for m in transient_markers:
        if m in err:
            return True

    return False
