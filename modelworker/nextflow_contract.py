"""Validation helpers for the Nextflow gateway contract.

This module validates task payloads sent from the Gateway to Workers.
Output file collection is handled by Nextflow using wildcard patterns,
so no manifest validation is needed on the worker side.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping


class ContractError(ValueError):
    pass


def _is_abs(p: str) -> bool:
    return os.path.isabs(p)


def _normalize_env(env: Dict[str, str], workdir: str) -> Dict[str, str]:
    out = dict(env)
    out.setdefault("PWD", workdir)
    out.setdefault("CURRENT_DIR", workdir)
    return out


def _normalize_profiling(profiling: Any) -> Dict[str, Any]:
    if profiling is None:
        return {}
    if not isinstance(profiling, Mapping):
        raise ContractError("payload.profiling must be an object when provided")
    if profiling.get("level") is not None:
        raise ContractError("payload.profiling.level is not supported in canonical runtime path")

    run_id = profiling.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ContractError("payload.profiling.run_id must be a string")

    output_dir = profiling.get("output_dir")
    if output_dir is not None and (not isinstance(output_dir, str) or not output_dir.strip()):
        raise ContractError("payload.profiling.output_dir must be a non-empty string")

    trace_input_path = profiling.get("trace_input_path")
    if trace_input_path is not None and (
        not isinstance(trace_input_path, str) or not trace_input_path.strip()
    ):
        raise ContractError("payload.profiling.trace_input_path must be a non-empty string")

    include_preprocess = profiling.get("include_preprocess", False)
    if not isinstance(include_preprocess, bool):
        raise ContractError("payload.profiling.include_preprocess must be a boolean")
    if include_preprocess:
        raise ContractError("payload.profiling.include_preprocess=true is not supported")

    if trace_input_path is not None:
        raise ContractError("payload.profiling.trace_input_path is not supported in canonical runtime path")

    normalized: Dict[str, Any] = {
        "include_preprocess": include_preprocess,
    }
    if run_id is not None:
        normalized["run_id"] = run_id
    if output_dir is not None:
        normalized["output_dir"] = output_dir
    return normalized


def validate_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a task payload.

    Expected payload:
        {
            "mode": "nextflow_task",
            "workdir": "/absolute/path/to/workdir",
            "argv": ["script.py", "--arg1", "value1", ...],
            "env": {"KEY": "value", ...},  # optional
        }

    The adapter receives this payload and is responsible for:
    - Parsing argv using its own argparse logic
    - Executing the task
    - Writing output files to workdir

    Nextflow collects output files using wildcard patterns defined in gw_signature.
    """
    if not isinstance(payload, dict):
        raise ContractError("payload must be a JSON object")

    mode = payload.get("mode")
    if mode != "nextflow_task":
        raise ContractError("payload.mode must be 'nextflow_task'")

    if "output_dir" in payload:
        raise ContractError("output_dir is not supported. Use workdir only.")

    workdir = payload.get("workdir")
    if not isinstance(workdir, str) or not workdir:
        raise ContractError("payload.workdir must be a non-empty string")
    if not _is_abs(workdir):
        raise ContractError("payload.workdir must be an absolute path")


    argv = payload.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(x, str) and x for x in argv)
    ):
        raise ContractError("payload.argv must be a non-empty list of strings")

    env = payload.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ContractError("payload.env must be a string->string map")

    out = dict(payload)
    out["mode"] = mode
    out["workdir"] = workdir
    out["argv"] = argv
    out["env"] = _normalize_env(env, workdir)
    out["profiling"] = _normalize_profiling(payload.get("profiling"))
    if "container_image" not in out:
        out["container_image"] = payload.get("container_image", "")
    return out
