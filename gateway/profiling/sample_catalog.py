"""Sample metadata lookup for gateway profiling runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import yaml

from .runtime_paths import PROFILE_CONFIG_DIR

SAMPLES_FILE = PROFILE_CONFIG_DIR / "samples.yaml"


@lru_cache(maxsize=1)
def _load_samples() -> Dict[str, Dict[str, str]]:
    if not SAMPLES_FILE.exists():
        return {}
    payload = yaml.safe_load(SAMPLES_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    groups = payload.get("samples")
    if not isinstance(groups, dict):
        return out
    for entries in groups.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sample_id = str(entry.get("sample_id") or "").strip()
            if not sample_id:
                continue
            out[sample_id] = {str(k): str(v) for k, v in entry.items() if v is not None}
    return out


def lookup_sample(sample_id: str) -> Dict[str, str]:
    return dict(_load_samples().get(str(sample_id), {}))

