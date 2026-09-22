"""Gateway-local profiling path/config helpers.

This module is intentionally self-contained so gateway runtime code does not
import `profile.*` modules from profiling/src.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_CONFIG_DIR = REPO_ROOT / "profiling" / "configs"

DEFAULT_STORAGE_ROOT = Path("/mnt/nfs/new/proton/profile")
DEFAULT_DATASET_ROOT = DEFAULT_STORAGE_ROOT.parent
DEFAULT_INPUT_ROOT = DEFAULT_STORAGE_ROOT / "inputs"
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "scratch"

PATHS_FILE = PROFILE_CONFIG_DIR / "paths.yaml"


class _Namespace(dict):
    def __getattr__(self, item: str):
        if item not in self:
            raise AttributeError(item)
        value = self[item]
        if isinstance(value, dict):
            value = _Namespace(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _resolve_context() -> Dict[str, str]:
    return {
        "storage_root": str(DEFAULT_STORAGE_ROOT),
        "data_root": str(DEFAULT_DATASET_ROOT),
        "input_root": str(DEFAULT_INPUT_ROOT),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
    }


def _resolve_value(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return Path(value.format(**context))
    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, context) for v in value]
    return value


def _load_paths() -> _Namespace:
    if not PATHS_FILE.exists():
        raise FileNotFoundError(f"Missing profiling paths config: {PATHS_FILE}")
    payload = yaml.safe_load(PATHS_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid profiling paths config: {PATHS_FILE}")
    resolved = _resolve_value(payload, _resolve_context())
    return _Namespace(resolved)


PATHS = _load_paths()

