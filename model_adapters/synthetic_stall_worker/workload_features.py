"""Workload feature extraction for synthetic_stall."""
from typing import Any, Dict, List, Optional

COMPONENT = "synthetic_stall"
INPUT_SIZE_KEY = "input_size"
UNLIMITED_RETRIES = True


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, arg in enumerate(argv):
        token = str(arg)
        for flag in flags:
            if token.startswith(flag + "="):
                return token.split("=", 1)[1]
            bare = flag.lstrip("-")
            if token in {flag, bare, f"-{bare}", f"--{bare}"} and i + 1 < len(argv):
                return str(argv[i + 1])
    return None


def _int(raw: Optional[str]) -> int:
    try:
        return int(str(raw).strip())
    except Exception:
        return 0


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    del env
    out: Dict[str, Any] = {}
    input_size = _int(_get(argv, ["--input-size"]))
    if input_size > 0:
        out["input_size"] = input_size
    for key, flag in {
        "profile": "--profile",
        "fault_profile": "--fault-profile",
        "scenario": "--scenario",
        "stage": "--stage",
        "hang_mode": "--hang-mode",
    }.items():
        value = _get(argv, [flag])
        if value:
            out[key] = str(value)
    return out


def is_retryable_error(error_message: str) -> bool:
    return "SYNTH_RETRYABLE" in str(error_message or "")


def adjust_argv_for_retry(argv: List[str], attempt: int) -> List[str]:
    del attempt
    return list(argv)
