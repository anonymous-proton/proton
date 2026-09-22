"""Deterministic run identity helpers for profiling runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Union


_NON_TOKEN = re.compile(r"[^a-zA-Z0-9._-]+")


def _normalize_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if float(value).is_integer():
            return int(value)
        return float(value)
    return value


def normalize_axis_value(value: Any) -> Any:
    """Convert a runtime value into a deterministic JSON-serializable value."""
    value = _normalize_number(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_axis_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_axis_value(item) for item in value]
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonicalize_axes(axes: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a key-sorted, normalized axis mapping."""
    out: Dict[str, Any] = {}
    for key_obj in sorted(axes.keys(), key=lambda item: str(item)):
        key = str(key_obj).strip()
        if not key:
            continue
        out[key] = normalize_axis_value(axes[key_obj])
    return out


def _identity_payload(component: str, level: str, identity: Mapping[str, Any]) -> str:
    payload = {
        "component": str(component).strip().lower(),
        "level": str(level).strip().lower(),
        "identity": canonicalize_axes(identity),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def run_identity_digest(component: str, level: str, identity: Mapping[str, Any]) -> str:
    raw = _identity_payload(component, level, identity)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_run_key(component: str, level: str, identity: Mapping[str, Any]) -> str:
    digest = run_identity_digest(component, level, identity)
    return f"{str(component).strip().lower()}:{str(level).strip().lower()}:{digest}"


def build_cell_schema_id(
    *,
    component: str,
    level: str,
    cell_axis_subset: Sequence[str],
    cell_schema_version: int = 1,
    algo_version: str = "v1",
) -> str:
    subset = [str(item).strip() for item in list(cell_axis_subset or []) if str(item).strip()]
    payload = {
        "component": str(component).strip().lower(),
        "level": str(level).strip().lower(),
        "cell_schema_version": int(cell_schema_version),
        "algo_version": str(algo_version).strip().lower() or "v1",
        "subset": sorted(dict.fromkeys(subset)),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"cs{int(cell_schema_version)}_{digest}"


def build_cell_key(
    *,
    component: str,
    level: str,
    axes: Mapping[str, Any],
    subset: Optional[Union[Mapping[str, Any], Sequence[str]]] = None,
    schema_id: Optional[str] = None,
) -> str:
    """Build a deterministic key for decision-time cell aggregation.

    `subset` accepts either:
    - iterable[str]: axis keys to include from `axes`
    - mapping[str, Any]: explicit axis payload to hash
    """
    if subset is None:
        selected = canonicalize_axes(dict(axes or {}))
    elif isinstance(subset, Mapping):
        selected = canonicalize_axes(dict(subset))
    else:
        selected = {}
        source = canonicalize_axes(dict(axes or {}))
        for key_obj in subset:
            key = str(key_obj).strip()
            if not key:
                continue
            if key in source:
                selected[key] = source[key]
    effective_schema_id = str(schema_id or "").strip()
    if not effective_schema_id:
        if isinstance(subset, Mapping):
            subset_keys = list(subset.keys())
        else:
            subset_keys = list(subset or [])
        effective_schema_id = build_cell_schema_id(
            component=component,
            level=level,
            cell_axis_subset=[str(item).strip() for item in subset_keys if str(item).strip()],
        )
    digest = run_identity_digest(component, level, selected)
    return f"{str(component).strip().lower()}@{effective_schema_id}:{str(level).strip().lower()}:cell:{digest}"


def _tokenize(value: str) -> str:
    token = _NON_TOKEN.sub("_", str(value).strip())
    token = token.strip("_")
    return token or "unknown"


def build_run_id(
    *,
    component: str,
    sample_id: str,
    input_batch_size: int,
    output_sample_count: int,
    repeat_idx: int,
    level: str,
    identity: Mapping[str, Any],
) -> str:
    digest = run_identity_digest(component, level, identity)[:8]
    return (
        f"{str(component).strip().lower()}-{_tokenize(sample_id)}-"
        f"inp{int(input_batch_size)}-out{int(output_sample_count)}-r{int(repeat_idx)}-{digest}"
    )
