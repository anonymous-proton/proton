"""Canonical runtime regime and query-context helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_MEMORY_BASIS = "total_execution_peak"
DEFAULT_PEAK_FIDELITY = "full_peak"
DEFAULT_ACTIVE_MEMORY_SCOPE = DEFAULT_MEMORY_BASIS

_ACTIVE_BUCKET_ORDER = {"a1": 0, "a2": 1, "a3plus": 2}
_QUEUE_BUCKET_ORDER = {"q0": 0, "q1": 1, "q2_3": 2, "q4plus": 3}
_DISPATCH_GROUP_BUCKET_ORDER = {"g1": 0, "g2": 1, "g3plus": 2}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_positive_int(value: Any) -> int | None:
    parsed = _to_int(value)
    if parsed is None or parsed < 1:
        return None
    return parsed


def _to_non_negative_int(value: Any) -> int | None:
    parsed = _to_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def bucket_batch_size(value: Any) -> str:
    batch_size = _to_positive_int(value) or 1
    if batch_size <= 1:
        return "b1"
    if batch_size == 2:
        return "b2"
    if batch_size <= 4:
        return "b3_4"
    if batch_size <= 8:
        return "b5_8"
    return "b9plus"


def bucket_active_requests(value: Any) -> str:
    active = max(1, _to_positive_int(value) or 1)
    if active <= 1:
        return "a1"
    if active == 2:
        return "a2"
    return "a3plus"


def bucket_queue_depth(value: Any) -> str:
    depth = max(0, _to_non_negative_int(value) or 0)
    if depth <= 0:
        return "q0"
    if depth == 1:
        return "q1"
    if depth <= 3:
        return "q2_3"
    return "q4plus"


def bucket_dispatch_group_size(value: Any) -> str:
    size = _to_positive_int(value) or 1
    if size <= 1:
        return "g1"
    if size == 2:
        return "g2"
    return "g3plus"


def build_local_regime_from_buckets(
    *,
    active_request_count_bucket: Any,
    effective_batch_bucket: Any,
    queue_depth_band: Any,
    dispatch_group_bucket: Any,
) -> str:
    active_bucket = _text(active_request_count_bucket) or "a1"
    batch_bucket = _text(effective_batch_bucket) or "b1"
    depth_bucket = _text(queue_depth_band) or "q0"
    group_bucket = _text(dispatch_group_bucket) or "g1"
    return f"{active_bucket}|{batch_bucket}|{depth_bucket}|{group_bucket}"


def build_local_regime_bucket(
    *,
    active_request_count: Any,
    batch_size: Any,
    queue_depth: Any,
    dispatch_group_size: Any,
) -> str:
    return build_local_regime_from_buckets(
        active_request_count_bucket=bucket_active_requests(active_request_count),
        effective_batch_bucket=bucket_batch_size(batch_size),
        queue_depth_band=bucket_queue_depth(queue_depth),
        dispatch_group_bucket=bucket_dispatch_group_size(dispatch_group_size),
    )


def co_location_signature_from_active_bucket(active_bucket: str) -> str:
    normalized = _text(active_bucket) or "a1"
    if normalized == "a1":
        return "solo_execute"
    return f"shared_execute:{normalized}"


def hardware_software_signature(worker_context: Mapping[str, Any]) -> str:
    explicit = _text(worker_context.get("hardware_software"))
    if explicit:
        return explicit
    parts = []
    worker_name = _text(worker_context.get("worker_name"))
    adapter = _text(worker_context.get("adapter"))
    image = _text(worker_context.get("image"))
    model_version = _text(worker_context.get("model_version"))
    gpu_ids = [
        _text(item) for item in list(worker_context.get("gpu_ids") or []) if _text(item)
    ]
    if worker_name:
        parts.append(f"worker:{worker_name}")
    if adapter:
        parts.append(f"adapter:{adapter}")
    if image:
        parts.append(f"image:{image}")
    if model_version:
        parts.append(f"model:{model_version}")
    if gpu_ids:
        parts.append(f"gpus:{len(gpu_ids)}")
    return "|".join(parts) or "default"


def normalize_worker_context(
    worker_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(worker_context or {}) if isinstance(worker_context, Mapping) else {}
    worker_name = _text(raw.get("worker_name"))
    worker_addr = _text(raw.get("worker_addr"))
    gpu_ids = [_text(item) for item in list(raw.get("gpu_ids") or []) if _text(item)]
    active_request_count = max(
        1, _to_positive_int(raw.get("active_request_count")) or 1
    )
    queue_depth = max(0, _to_non_negative_int(raw.get("queue_depth")) or 0)
    dispatch_group_size = _to_positive_int(raw.get("dispatch_group_size")) or 1
    active_request_count_bucket = _text(
        raw.get("active_request_count_bucket")
    ) or bucket_active_requests(active_request_count)
    queue_depth_band = _text(raw.get("queue_depth_band")) or bucket_queue_depth(
        queue_depth
    )
    dispatch_group_bucket = _text(
        raw.get("dispatch_group_bucket")
    ) or bucket_dispatch_group_size(dispatch_group_size)
    effective_batch_bucket = _text(raw.get("effective_batch_bucket"))
    co_location_signature = _text(
        raw.get("co_location_signature")
    ) or co_location_signature_from_active_bucket(active_request_count_bucket)
    local_regime_bucket = _text(
        raw.get("local_regime_bucket")
    ) or build_local_regime_from_buckets(
        active_request_count_bucket=active_request_count_bucket,
        effective_batch_bucket=effective_batch_bucket,
        queue_depth_band=queue_depth_band,
        dispatch_group_bucket=dispatch_group_bucket,
    )
    return {
        "worker_name": worker_name,
        "worker_addr": worker_addr,
        "gpu_ids": gpu_ids,
        "adapter": _text(raw.get("adapter")),
        "image": _text(raw.get("image")),
        "model_version": _text(raw.get("model_version")),
        "active_request_count": active_request_count,
        "active_request_count_bucket": active_request_count_bucket,
        "queue_depth": queue_depth,
        "queue_depth_band": queue_depth_band,
        "dispatch_group_size": dispatch_group_size,
        "dispatch_group_bucket": dispatch_group_bucket,
        "effective_batch_bucket": effective_batch_bucket,
        "local_regime_bucket": local_regime_bucket,
        "co_location_signature": co_location_signature,
        "hardware_software": hardware_software_signature(raw),
        "residency_state": _text(raw.get("residency_state")),
    }


def canonical_tool_mode(workload_features: Mapping[str, Any]) -> str:
    return _text(workload_features.get("tool_mode")).lower()


def canonical_stage_class(
    *,
    component: str,
    level: str | None,
    workload_features: Mapping[str, Any],
) -> str:
    base = (_text(level) or _text(component)).lower()
    tool_mode = canonical_tool_mode(workload_features)
    if tool_mode:
        return f"{base}:{tool_mode}"
    return base


def descriptor_bucket(
    *,
    config_fingerprint: str,
    input_fingerprint: str,
    workload_features: Mapping[str, Any],
) -> str:
    parts = []
    normalized_cfg = _text(config_fingerprint)
    normalized_inp = _text(input_fingerprint)
    sample_id = _text(workload_features.get("sample_id"))
    if normalized_cfg:
        parts.append(f"cfg:{normalized_cfg}")
    if normalized_inp:
        parts.append(f"inp:{normalized_inp}")
    if sample_id:
        parts.append(f"sample:{sample_id}")
    extras = []
    for key in sorted(workload_features):
        if key in {
            "sample_id",
            "input_batch_size",
            "output_sample_count",
            "backfill_gap_sec",
            "mem_safe_limit_mib",
            "tool_mode",
        }:
            continue
        value = workload_features.get(key)
        if value in (None, "", [], {}, ()):
            continue
        extras.append(f"{key}={value}")
    if extras:
        parts.append("axes:" + "|".join(extras))
    return "|".join(parts) or "generic"


def normalize_execution_envelope(
    *,
    execution_envelope: Mapping[str, Any] | None = None,
    requested_batch_size: Any = None,
    worker_context: Mapping[str, Any] | None = None,
    memory_basis: str = DEFAULT_MEMORY_BASIS,
    peak_fidelity: str = DEFAULT_PEAK_FIDELITY,
    active_memory_scope: str = DEFAULT_ACTIVE_MEMORY_SCOPE,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = (
        dict(execution_envelope or {})
        if isinstance(execution_envelope, Mapping)
        else {}
    )
    extra = dict(defaults or {}) if isinstance(defaults, Mapping) else {}
    normalized_worker = normalize_worker_context(worker_context)
    resolved_batch_size = (
        _to_positive_int(raw.get("requested_batch_size"))
        or _to_positive_int(requested_batch_size)
        or 1
    )
    effective_batch_bucket = _text(
        raw.get("effective_batch_bucket")
    ) or bucket_batch_size(resolved_batch_size)
    normalized_worker["effective_batch_bucket"] = effective_batch_bucket
    local_regime_bucket = _text(
        raw.get("local_regime_bucket")
    ) or build_local_regime_from_buckets(
        active_request_count_bucket=normalized_worker["active_request_count_bucket"],
        effective_batch_bucket=effective_batch_bucket,
        queue_depth_band=normalized_worker["queue_depth_band"],
        dispatch_group_bucket=normalized_worker["dispatch_group_bucket"],
    )
    normalized_memory_basis = (
        _text(raw.get("memory_basis"))
        or _text(memory_basis)
        or _text(raw.get("active_memory_scope"))
        or _text(active_memory_scope)
        or DEFAULT_MEMORY_BASIS
    )
    normalized_peak_fidelity = (
        _text(raw.get("peak_fidelity")) or _text(peak_fidelity) or DEFAULT_PEAK_FIDELITY
    )
    return {
        "model_id": _text(raw.get("model_id")) or _text(extra.get("model_id")).lower(),
        "stage_class": (
            _text(raw.get("stage_class")) or _text(extra.get("stage_class"))
        ).lower(),
        "tool_mode": (
            _text(raw.get("tool_mode")) or _text(extra.get("tool_mode"))
        ).lower(),
        "descriptor_bucket": _text(raw.get("descriptor_bucket"))
        or _text(extra.get("descriptor_bucket")),
        "requested_batch_size": resolved_batch_size,
        "effective_batch_bucket": effective_batch_bucket,
        "active_request_count": normalized_worker["active_request_count"],
        "active_request_count_bucket": normalized_worker["active_request_count_bucket"],
        "queue_depth": normalized_worker["queue_depth"],
        "queue_depth_band": normalized_worker["queue_depth_band"],
        "dispatch_group_size": normalized_worker["dispatch_group_size"],
        "dispatch_group_bucket": normalized_worker["dispatch_group_bucket"],
        "local_regime_bucket": local_regime_bucket,
        "co_location_signature": _text(raw.get("co_location_signature"))
        or normalized_worker["co_location_signature"],
        "hardware_software": _text(raw.get("hardware_software"))
        or normalized_worker["hardware_software"],
        "residency_state": _text(raw.get("residency_state"))
        or normalized_worker["residency_state"],
        "memory_basis": normalized_memory_basis,
        "peak_fidelity": normalized_peak_fidelity,
        "active_memory_scope": normalized_memory_basis,
    }


def build_query_context(
    *,
    component: str,
    level: str | None,
    config_fingerprint: str,
    input_fingerprint: str,
    workload_features: Mapping[str, Any],
    requested_batch_size: Any,
    worker_context: Mapping[str, Any] | None,
    campaign_id: str,
    memory_basis: str = DEFAULT_MEMORY_BASIS,
    peak_fidelity: str = DEFAULT_PEAK_FIDELITY,
    active_memory_scope: str = DEFAULT_ACTIVE_MEMORY_SCOPE,
) -> dict[str, Any]:
    normalized_component = _text(component).lower()
    normalized_workload = (
        dict(workload_features or {}) if isinstance(workload_features, Mapping) else {}
    )
    stage_class = canonical_stage_class(
        component=normalized_component,
        level=level,
        workload_features=normalized_workload,
    )
    tool_mode = canonical_tool_mode(normalized_workload)
    execution_envelope = normalize_execution_envelope(
        requested_batch_size=requested_batch_size,
        worker_context=worker_context,
        memory_basis=memory_basis,
        peak_fidelity=peak_fidelity,
        active_memory_scope=active_memory_scope,
        defaults={
            "model_id": normalized_component,
            "stage_class": stage_class,
            "tool_mode": tool_mode,
            "descriptor_bucket": descriptor_bucket(
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                workload_features=normalized_workload,
            ),
        },
    )
    baseline_context = {
        "model_id": normalized_component,
        "stage_class": stage_class,
        "tool_mode": tool_mode,
        "hardware_software": execution_envelope["hardware_software"],
    }
    return {
        "version": 2,
        "campaign_id": _text(campaign_id),
        "model_id": normalized_component,
        "stage_class": stage_class,
        "tool_mode": tool_mode,
        "execution_envelope": execution_envelope,
        "replica_baseline": baseline_context,
    }


def normalize_query_context(
    query_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(query_context or {}) if isinstance(query_context, Mapping) else {}
    execution_envelope = raw.get("execution_envelope")
    baseline_context = raw.get("replica_baseline")
    if not isinstance(execution_envelope, Mapping) or not isinstance(
        baseline_context, Mapping
    ):
        return {}
    normalized_execution_envelope = normalize_execution_envelope(
        execution_envelope=execution_envelope,
        worker_context=execution_envelope,
        memory_basis=_text(execution_envelope.get("memory_basis"))
        or DEFAULT_MEMORY_BASIS,
        peak_fidelity=_text(execution_envelope.get("peak_fidelity"))
        or DEFAULT_PEAK_FIDELITY,
        active_memory_scope=_text(execution_envelope.get("active_memory_scope"))
        or DEFAULT_ACTIVE_MEMORY_SCOPE,
        defaults={
            "model_id": raw.get("model_id"),
            "stage_class": raw.get("stage_class"),
            "tool_mode": raw.get("tool_mode"),
        },
    )
    return {
        "version": _to_int(raw.get("version")) or 2,
        "campaign_id": _text(raw.get("campaign_id")),
        "model_id": _text(
            raw.get("model_id") or normalized_execution_envelope.get("model_id")
        ).lower(),
        "stage_class": (
            _text(raw.get("stage_class"))
            or _text(normalized_execution_envelope.get("stage_class"))
        ).lower(),
        "tool_mode": (
            _text(raw.get("tool_mode"))
            or _text(normalized_execution_envelope.get("tool_mode"))
        ).lower(),
        "memory_basis": _text(
            raw.get("memory_basis") or normalized_execution_envelope.get("memory_basis")
        )
        or DEFAULT_MEMORY_BASIS,
        "peak_fidelity": _text(
            raw.get("peak_fidelity")
            or normalized_execution_envelope.get("peak_fidelity")
        )
        or DEFAULT_PEAK_FIDELITY,
        "execution_envelope": normalized_execution_envelope,
        "replica_baseline": {
            "model_id": _text(
                baseline_context.get("model_id")
                or raw.get("model_id")
                or normalized_execution_envelope.get("model_id")
            ).lower(),
            "stage_class": (
                _text(baseline_context.get("stage_class"))
                or _text(raw.get("stage_class"))
                or _text(normalized_execution_envelope.get("stage_class"))
            ).lower(),
            "tool_mode": (
                _text(baseline_context.get("tool_mode"))
                or _text(raw.get("tool_mode"))
                or _text(normalized_execution_envelope.get("tool_mode"))
            ).lower(),
            "hardware_software": _text(baseline_context.get("hardware_software"))
            or _text(normalized_execution_envelope.get("hardware_software"))
            or "default",
        },
    }


def regime_is_nearby(row_ctx: Mapping[str, Any], query_ctx: Mapping[str, Any]) -> bool:
    if _text(row_ctx.get("effective_batch_bucket")) != _text(
        query_ctx.get("effective_batch_bucket")
    ):
        return False
    if _text(row_ctx.get("dispatch_group_bucket")) != _text(
        query_ctx.get("dispatch_group_bucket")
    ):
        return False
    row_active_rank = _ACTIVE_BUCKET_ORDER.get(
        _text(row_ctx.get("active_request_count_bucket"))
    )
    query_active_rank = _ACTIVE_BUCKET_ORDER.get(
        _text(query_ctx.get("active_request_count_bucket"))
    )
    row_queue_rank = _QUEUE_BUCKET_ORDER.get(_text(row_ctx.get("queue_depth_band")))
    query_queue_rank = _QUEUE_BUCKET_ORDER.get(_text(query_ctx.get("queue_depth_band")))
    if (
        row_active_rank is None
        or query_active_rank is None
        or row_queue_rank is None
        or query_queue_rank is None
    ):
        return False
    return (
        abs(row_active_rank - query_active_rank) <= 1
        and abs(row_queue_rank - query_queue_rank) <= 1
    )


__all__ = [
    "DEFAULT_ACTIVE_MEMORY_SCOPE",
    "bucket_active_requests",
    "bucket_batch_size",
    "bucket_dispatch_group_size",
    "bucket_queue_depth",
    "build_local_regime_bucket",
    "build_local_regime_from_buckets",
    "build_query_context",
    "canonical_stage_class",
    "canonical_tool_mode",
    "co_location_signature_from_active_bucket",
    "descriptor_bucket",
    "hardware_software_signature",
    "normalize_execution_envelope",
    "normalize_query_context",
    "normalize_worker_context",
    "regime_is_nearby",
]
