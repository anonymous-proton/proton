"""Workload feature extraction for AlphaFold3. Stdlib only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

COMPONENT = "alphafold3"
INPUT_SIZE_KEY = "num_residues"


def _get(argv: List[str], flags: List[str]) -> Optional[str]:
    for i, arg in enumerate(argv):
        value = str(arg)
        for flag in flags:
            if value.startswith(flag + "="):
                return value.split("=", 1)[1].strip("'\"")
            bare = flag.lstrip("-")
            if value in (flag, f"--{bare}", f"-{bare}", bare) and i + 1 < len(argv):
                return str(argv[i + 1]).strip("'\"")
    return None


def _resolve(path: str, env: Dict[str, str]) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    workdir = env.get("_GW_WORKDIR") or env.get("PWD") or ""
    return Path(workdir) / resolved if workdir else resolved


def _chain_multiplier(entity: Dict[str, Any]) -> int:
    chain_id = entity.get("id")
    if isinstance(chain_id, list):
        return max(1, len(chain_id))
    return 1


def _sequence_length(entity: Dict[str, Any]) -> int:
    sequence = entity.get("sequence")
    if sequence:
        return len(str(sequence))
    return 0


def _json_num_residues(path: Path) -> int:
    data = json.loads(path.read_text())
    total = 0
    jobs = data if isinstance(data, list) else [data]
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for entry in job.get("sequences") or []:
            if not isinstance(entry, dict):
                continue
            for key in ("protein", "rna", "dna", "proteinChain", "rnaChain", "dnaChain"):
                entity = entry.get(key)
                if isinstance(entity, dict):
                    total += _sequence_length(entity) * _chain_multiplier(entity)
    return total


def extract_workload_features(argv: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    json_path = _get(argv, ["--json_path", "json_path"])
    if not json_path:
        return {}

    try:
        num_residues = _json_num_residues(_resolve(json_path, env))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}

    result: Dict[str, Any] = {}
    if num_residues > 0:
        result["num_residues"] = num_residues

    model_dir = _get(argv, ["--model_dir", "model_dir"]) or env.get("ALPHAFOLD3_MODEL_DIR")
    if model_dir:
        result["model_dir"] = Path(str(model_dir)).name or str(model_dir)

    config_sources = {
        "num_recycles": (
            ["--num_recycles", "num_recycles"],
            "ALPHAFOLD3_NUM_RECYCLES",
        ),
        "num_diffusion_steps": (
            ["--num_diffusion_steps", "num_diffusion_steps"],
            "ALPHAFOLD3_NUM_DIFFUSION_STEPS",
        ),
        "flash_attention": (
            ["--flash_attention_implementation", "flash_attention_implementation"],
            "ALPHAFOLD3_FLASH_ATTENTION_IMPLEMENTATION",
        ),
    }
    for key, (flags, env_key) in config_sources.items():
        value = _get(argv, flags) or env.get(env_key)
        if value:
            result[key] = str(value)

    return result
