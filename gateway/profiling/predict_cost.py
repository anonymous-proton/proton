"""Canonical Section-6 signals estimator entrypoint for scheduler-facing predictions."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import MODEL_INPUT_RUN_SOURCES, MODEL_INPUT_STATE_SUCCEEDED
from .index_models import RunQuery, include_run_for_modeling


_COLLECT_CACHE_TTL_SEC = float(os.environ.get("PREDICT_COST_COLLECT_CACHE_TTL_SEC", "1.0"))
_collect_cache: Dict[Tuple[Any, ...], Tuple[float, List[Any]]] = {}
_collect_cache_lock = threading.Lock()


def clear_collect_cache() -> None:
    """Manual invalidation hook for tests / explicit callers."""
    with _collect_cache_lock:
        _collect_cache.clear()
from .runtime_regime import build_query_context, normalize_query_context, regime_is_nearby
from .safety_inflation import (
    _actual_active_memory_mib as _safety_actual_active_memory_mib,
    _actual_hot_execution_sec as _safety_actual_hot_execution_sec,
    _actual_resident_memory_mib as _safety_actual_resident_memory_mib,
    _actual_transition_penalty_sec as _safety_actual_transition_penalty_sec,
    active_memory_upper as _safety_active_memory_upper,
    estimate_safety_inflation,
    resident_memory_upper as _safety_resident_memory_upper,
    runtime_hot_upper as _safety_runtime_hot_upper,
    runtime_transition_upper as _safety_runtime_transition_upper,
)

_RUNTIME_GUARD_Q = 0.90
_MEMORY_GUARD_Q = 0.95
_RECENCY_DECAY = 0.85
_SELECTION_SUPPORT = {"exact": 2, "nearby": 2, "coarse": 1}
_SUPPORT_TIER_THRESHOLDS = {"exact": 6, "nearby": 3, "coarse": 1}
_SUPPORT_GUARD_MULTIPLIER = {
    "exact": 1.0,
    "nearby": 1.08,
    "coarse": 1.18,
    "none": 1.0,
}
_FALLBACK_GUARD_MULTIPLIER = {
    "exact": 1.0,
    "nearby": 1.05,
    "coarse": 1.12,
    "none": 1.0,
}
_CAMPAIGN_CORRECTION_ALPHA = 0.55
_CAMPAIGN_CORRECTION_MAX_LOG = math.log(1.75)
_SUPPORT_ORDER = {"none": 0, "coarse": 1, "nearby": 2, "exact": 3}
_FALLBACK_ORDER = {"exact": 0, "nearby": 1, "coarse": 2, "none": 3}


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or math.isinf(parsed):
        return None
    return float(parsed)


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_positive_int(value: Any) -> Optional[int]:
    parsed = _to_int(value)
    if parsed is None or parsed < 1:
        return None
    return int(parsed)


def _row_str(row: Any, field: str) -> str:
    return str(getattr(row, field, "") or "").strip()


def _row_timestamp(row: Any) -> float:
    for field in ("finished_at", "updated_at", "started_at", "created_at"):
        parsed = _to_float(getattr(row, field, None))
        if parsed is not None:
            return float(parsed)
    return 0.0


def _sorted_rows_by_recency(rows: Sequence[Any]) -> List[Any]:
    return sorted(list(rows), key=_row_timestamp, reverse=True)


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = sorted(
        (
            (float(value), max(0.0, float(weight)))
            for value, weight in zip(values, weights)
            if value is not None and weight is not None
        ),
        key=lambda item: item[0],
    )
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return float(pairs[-1][0])
    threshold = max(0.0, min(1.0, float(q))) * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _recency_weights(n_items: int) -> List[float]:
    return [float(_RECENCY_DECAY**idx) for idx in range(max(0, int(n_items)))]


def _build_query_context(
    *,
    component: str,
    level: Optional[str],
    config_fingerprint: str,
    input_fingerprint: str,
    workload_features: Mapping[str, Any],
    execution_overrides: Mapping[str, Any],
    worker_context: Mapping[str, Any],
    campaign_id: str,
) -> Dict[str, Any]:
    requested_batch_size = (
        _to_positive_int(execution_overrides.get("batch_size"))
        or _to_positive_int(workload_features.get("input_batch_size"))
        or 1
    )
    return build_query_context(
        component=component,
        level=level,
        config_fingerprint=config_fingerprint,
        input_fingerprint=input_fingerprint,
        workload_features=workload_features,
        requested_batch_size=requested_batch_size,
        worker_context=worker_context,
        campaign_id=campaign_id,
    )


def _row_query_context(row: Any) -> Dict[str, Any]:
    raw_decision = getattr(row, "scheduler_decision", None)
    decision = dict(raw_decision or {}) if isinstance(raw_decision, Mapping) else {}
    query_context = normalize_query_context(decision.get("query_context"))
    return dict(query_context)


def _support_level_for_count(n_rows: int) -> str:
    if n_rows >= _SUPPORT_TIER_THRESHOLDS["exact"]:
        return "exact"
    if n_rows >= _SUPPORT_TIER_THRESHOLDS["nearby"]:
        return "nearby"
    if n_rows >= _SUPPORT_TIER_THRESHOLDS["coarse"]:
        return "coarse"
    return "none"


def _positive_target_value(value: Any) -> Optional[float]:
    parsed = _to_float(value)
    if parsed is None or parsed <= 0:
        return None
    return float(parsed)


def _transition_penalty_sec(row: Any) -> Optional[float]:
    timing = getattr(row, "worker_timing_us", None)
    if not isinstance(timing, Mapping):
        return None
    prepare_us = _to_float(timing.get("prepare_us"))
    finalize_us = _to_float(timing.get("finalize_us"))
    total_us = float((prepare_us or 0.0) + (finalize_us or 0.0))
    if total_us <= 0:
        return None
    return total_us / 1_000_000.0


_interference_correction_fn: Optional[Any] = None


def set_interference_correction(fn: Any) -> None:
    """Wire in the SignalService interference correction callback."""
    global _interference_correction_fn
    _interference_correction_fn = fn


def _hot_execution_time_sec(row: Any) -> Optional[float]:
    raw = _positive_target_value(getattr(row, "runtime_sec", None))
    if raw is None:
        return None
    if not bool(getattr(row, "concurrent_execute_overlap", False)):
        return raw
    if _interference_correction_fn is not None:
        component = str(getattr(row, "component", "") or "").strip().lower()
        if component:
            corrected = _interference_correction_fn(component, raw)
            if corrected is not None and corrected > 0:
                return float(corrected)
    return raw


def _active_memory_mib(row: Any) -> Optional[float]:
    active_vram = getattr(row, "active_vram_mib", None)
    if active_vram is not None:
        if not bool(getattr(row, "vram_memory_qc_keep", False)):
            return None
        return _positive_target_value(active_vram)
    if not bool(getattr(row, "memory_qc_keep", False)):
        return None
    return _positive_target_value(getattr(row, "active_memory_mib", None))


def _resident_memory_mib(row: Any) -> Optional[float]:
    return _positive_target_value(getattr(row, "resident_memory_mib", None))


def _collect_rows(
    *,
    run_index: Any,
    component: str,
    level: Optional[str],
    history_limit: int,
) -> List[Any]:
    """TTL-cached.  Hits sqlite at most once per
    ``(component, level, history_limit)`` tuple per
    ``_COLLECT_CACHE_TTL_SEC``."""
    normalized_component = str(component or "").strip().lower()
    normalized_level = str(level or "").strip().lower() or None
    history = int(max(1, history_limit))
    key = ("rows", id(run_index), normalized_component, normalized_level, history)
    now = time.monotonic()
    with _collect_cache_lock:
        cached = _collect_cache.get(key)
        if cached is not None:
            ts, cached_rows = cached
            if now - ts < _COLLECT_CACHE_TTL_SEC:
                return cached_rows

    import threading as _threading
    import os as _os
    cur_name = _threading.current_thread().name
    if not cur_name.startswith("sqlite-") and not _os.environ.get(
        "GATEWAY_SQLITE_THREAD_BYPASS"
    ):
        if cached is not None:
            return cached[1]
        return []

    rows: List[Any] = []
    for run_source in MODEL_INPUT_RUN_SOURCES:
        source_rows, _ = run_index.query_runs(
            RunQuery(
                run_source=run_source,
                component=normalized_component,
                level=normalized_level,
                state=MODEL_INPUT_STATE_SUCCEEDED,
                sort="finished_at",
                descending=True,
                limit=history,
                offset=0,
            )
        )
        rows.extend(source_rows)
    filtered = [row for row in rows if include_run_for_modeling(row)]
    with _collect_cache_lock:
        _collect_cache[key] = (now, filtered)
    return filtered


def _collect_recent_task_rows(
    *,
    run_index: Any,
    component: str,
    level: Optional[str],
    history_limit: int,
) -> List[Any]:
    """TTL-cached.  See ``_collect_rows`` for rationale.

     fix Option D — graceful fallback when called from
    non-sqlite thread (e.g. ``Validator.validate_and_dispatch`` path
    sync invoking via ``SignalService.query`` → ``predict_cost``).
    The ``RunIndexStore._connect`` guard would raise RuntimeError on
    such calls; here we return cached-or-empty before reaching that
    guard so the caller (Validator) gets a graceful default cost
    estimate instead of a hard exception that would break dispatch.
    Background refreshers (running on the sqlite-* thread) still
    populate the cache for subsequent reads.
    """
    normalized_component = str(component or "").strip().lower()
    normalized_level = str(level or "").strip().lower() or None
    history = int(max(1, history_limit))
    key = ("recent", id(run_index), normalized_component, normalized_level, history)
    now = time.monotonic()
    with _collect_cache_lock:
        cached = _collect_cache.get(key)
        if cached is not None:
            ts, cached_rows = cached
            if now - ts < _COLLECT_CACHE_TTL_SEC:
                return cached_rows

    import threading as _threading
    import os as _os
    cur_name = _threading.current_thread().name
    if not cur_name.startswith("sqlite-") and not _os.environ.get(
        "GATEWAY_SQLITE_THREAD_BYPASS"
    ):
        if cached is not None:
            return cached[1]
        return []

    rows, _ = run_index.query_runs(
        RunQuery(
            run_source="task",
            component=normalized_component,
            level=normalized_level,
            sort="finished_at",
            descending=True,
            limit=history,
            offset=0,
        )
    )
    result = list(rows)
    with _collect_cache_lock:
        _collect_cache[key] = (now, result)
    return result


def _match_execution_context(row_ctx: Mapping[str, Any], query_ctx: Mapping[str, Any], level: str) -> bool:
    if str(row_ctx.get("model_id") or "") != str(query_ctx.get("model_id") or ""):
        return False
    if str(row_ctx.get("stage_class") or "") != str(query_ctx.get("stage_class") or ""):
        return False
    if level == "exact":
        for key in (
            "descriptor_bucket",
            "local_regime_bucket",
            "queue_depth_band",
            "dispatch_group_bucket",
            "co_location_signature",
            "hardware_software",
        ):
            if str(row_ctx.get(key) or "") != str(query_ctx.get(key) or ""):
                return False
        return True
    if level == "nearby":
        for key in ("descriptor_bucket", "hardware_software"):
            if str(row_ctx.get(key) or "") != str(query_ctx.get(key) or ""):
                return False
        return regime_is_nearby(row_ctx, query_ctx)
    return True


def _match_baseline_context(row_ctx: Mapping[str, Any], query_ctx: Mapping[str, Any], level: str) -> bool:
    if str(row_ctx.get("model_id") or "") != str(query_ctx.get("model_id") or ""):
        return False
    if level == "coarse":
        return True
    if str(row_ctx.get("stage_class") or "") != str(query_ctx.get("stage_class") or ""):
        return False
    if level == "nearby":
        return True
    return str(row_ctx.get("hardware_software") or "") == str(query_ctx.get("hardware_software") or "")


def _select_target_rows(
    *,
    rows: Sequence[Any],
    query_context: Mapping[str, Any],
    context_getter: Callable[[Any], Mapping[str, Any]],
    matcher: Callable[[Mapping[str, Any], Mapping[str, Any], str], bool],
    extract_value: Callable[[Any], Optional[float]],
) -> Dict[str, Any]:
    buckets: Dict[str, List[Any]] = {"exact": [], "nearby": [], "coarse": []}
    for row in rows:
        observed = extract_value(row)
        if observed is None or observed <= 0:
            continue
        row_context = context_getter(row)
        if matcher(row_context, query_context, "coarse"):
            buckets["coarse"].append(row)
        if matcher(row_context, query_context, "nearby"):
            buckets["nearby"].append(row)
        if matcher(row_context, query_context, "exact"):
            buckets["exact"].append(row)
    if len(buckets["exact"]) >= _SELECTION_SUPPORT["exact"]:
        fallback_level = "exact"
    elif len(buckets["nearby"]) >= _SELECTION_SUPPORT["nearby"]:
        fallback_level = "nearby"
    elif len(buckets["coarse"]) >= _SELECTION_SUPPORT["coarse"]:
        fallback_level = "coarse"
    else:
        fallback_level = "none"
    selected_rows = list(buckets.get(fallback_level) or []) if fallback_level != "none" else []
    return {
        "fallback_level": fallback_level,
        "support_level": _support_level_for_count(len(selected_rows)),
        "selected_rows": selected_rows,
        "counts": {
            "n_exact_rows": int(len(buckets["exact"])),
            "n_nearby_rows": int(len(buckets["nearby"])),
            "n_coarse_rows": int(len(buckets["coarse"])),
        },
    }


def _select_compatible_rows(
    *,
    rows: Sequence[Any],
    query_context: Mapping[str, Any],
    context_getter: Callable[[Any], Mapping[str, Any]],
    matcher: Callable[[Mapping[str, Any], Mapping[str, Any], str], bool],
    level: str,
) -> List[Any]:
    if level == "none":
        return []
    out: List[Any] = []
    for row in rows:
        row_context = context_getter(row)
        if matcher(row_context, query_context, level):
            out.append(row)
    return out


def _campaign_correction_log(
    *,
    rows: Sequence[Any],
    selected_rows: Sequence[Any],
    campaign_id: str,
    extract_value: Callable[[Any], Optional[float]],
    shared_center_log: Optional[float],
) -> Tuple[float, int]:
    if not campaign_id or shared_center_log is None:
        return 0.0, 0
    residual_rows = [
        row
        for row in _sorted_rows_by_recency(rows)
        if str(getattr(row, "campaign_id", "") or "").strip() == campaign_id
    ]
    if not residual_rows:
        residual_rows = [
            row
            for row in _sorted_rows_by_recency(selected_rows)
            if str(getattr(row, "campaign_id", "") or "").strip() == campaign_id
        ]
    delta = 0.0
    used = 0
    for row in residual_rows:
        observed = extract_value(row)
        if observed is None or observed <= 0:
            continue
        residual = math.log(max(observed, 1e-12)) - float(shared_center_log)
        clipped = max(-_CAMPAIGN_CORRECTION_MAX_LOG, min(_CAMPAIGN_CORRECTION_MAX_LOG, residual))
        if used == 0:
            delta = clipped
        else:
            delta = (_CAMPAIGN_CORRECTION_ALPHA * clipped) + ((1.0 - _CAMPAIGN_CORRECTION_ALPHA) * delta)
        used += 1
        if used >= 8:
            break
    return float(delta), int(used)


def _empty_target_block(*, unit: str, scope: str = "", n_history_rows: int = 0, counts: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    evidence_counts = dict(counts or {})
    return {
        "estimate": {
            "center": None,
            "upper": None,
            "unit": unit,
            **({"scope": scope} if scope else {}),
        },
        "support": {
            "level": "none",
            "fallback_level": "none",
            "effective_support": 0,
            "status": "abstain",
        },
        "evidence": {
            "n_history_rows": int(n_history_rows),
            "n_exact_rows": int(_to_int(evidence_counts.get("n_exact_rows")) or 0),
            "n_nearby_rows": int(_to_int(evidence_counts.get("n_nearby_rows")) or 0),
            "n_coarse_rows": int(_to_int(evidence_counts.get("n_coarse_rows")) or 0),
            "n_selected_rows": 0,
            "n_campaign_correction_rows": 0,
        },
    }


def _empty_correction_block() -> Dict[str, Any]:
    return {
        "campaign_local_residual": None,
        "total": None,
    }


def _empty_guard_block() -> Dict[str, Any]:
    return {
        "base": None,
        "support": None,
        "fallback": None,
        "safety": None,
        "total": None,
    }


def _estimate_target(
    *,
    rows: Sequence[Any],
    safety_rows: Sequence[Any],
    query_context: Mapping[str, Any],
    context_getter: Callable[[Any], Mapping[str, Any]],
    matcher: Callable[[Mapping[str, Any], Mapping[str, Any], str], bool],
    extract_value: Callable[[Any], Optional[float]],
    safety_target_kind: str,
    safety_actual_getter: Callable[[Any], Optional[float]],
    safety_upper_getter: Callable[[Any], Optional[float]],
    campaign_id: str,
    guard_q: float,
    unit: str,
    scope: str = "",
    apply_campaign_correction: bool = True,
) -> Dict[str, Any]:
    selection = _select_target_rows(
        rows=rows,
        query_context=query_context,
        context_getter=context_getter,
        matcher=matcher,
        extract_value=extract_value,
    )
    if str(selection.get("fallback_level") or "none") == "none":
        return {
            "target": _empty_target_block(
                unit=unit,
                scope=scope,
                n_history_rows=len(rows),
                counts=selection.get("counts"),
            ),
            "corrections": _empty_correction_block(),
            "guards": _empty_guard_block(),
        }

    selected_rows = list(selection.get("selected_rows") or [])
    ordered_rows = _sorted_rows_by_recency(selected_rows)
    observations: List[float] = []
    value_rows: List[Any] = []
    for row in ordered_rows:
        observed = extract_value(row)
        if observed is None or observed <= 0:
            continue
        observations.append(float(observed))
        value_rows.append(row)
    if not observations:
        return {
            "target": _empty_target_block(
                unit=unit,
                scope=scope,
                n_history_rows=len(rows),
                counts=selection.get("counts"),
            ),
            "corrections": _empty_correction_block(),
            "guards": _empty_guard_block(),
        }

    weights = _recency_weights(len(observations))
    log_values = [math.log(max(value, 1e-12)) for value in observations]
    shared_center_log = _weighted_quantile(log_values, weights, 0.50)
    if shared_center_log is None:
        return {
            "target": _empty_target_block(
                unit=unit,
                scope=scope,
                n_history_rows=len(rows),
                counts=selection.get("counts"),
            ),
            "corrections": _empty_correction_block(),
            "guards": _empty_guard_block(),
        }

    positive_residuals = [max(0.0, value - shared_center_log) for value in log_values]
    fallback_level = str(selection.get("fallback_level") or "none")
    support_level = str(selection.get("support_level") or "none")
    base_guard_log = _weighted_quantile(positive_residuals, weights, guard_q) or 0.0
    support_guard_log = math.log(_SUPPORT_GUARD_MULTIPLIER.get(support_level, 1.0))
    fallback_guard_log = math.log(_FALLBACK_GUARD_MULTIPLIER.get(fallback_level, 1.0))
    safety_selected_rows = _select_compatible_rows(
        rows=safety_rows,
        query_context=query_context,
        context_getter=context_getter,
        matcher=matcher,
        level=fallback_level,
    )
    safety_guard_log = float(
        estimate_safety_inflation(
            rows=safety_selected_rows,
            target_kind=safety_target_kind,
            actual_getter=safety_actual_getter,
            upper_getter=safety_upper_getter,
        ).get("log_increment")
        or 0.0
    )
    campaign_delta_log, campaign_rows = (0.0, 0)
    if apply_campaign_correction:
        campaign_delta_log, campaign_rows = _campaign_correction_log(
            rows=value_rows,
            selected_rows=selected_rows,
            campaign_id=campaign_id,
            extract_value=extract_value,
            shared_center_log=shared_center_log,
        )

    center_log = shared_center_log + campaign_delta_log
    upper_log = center_log + base_guard_log + support_guard_log + fallback_guard_log + safety_guard_log

    target_block = {
        "estimate": {
            "center": math.exp(center_log),
            "upper": math.exp(upper_log),
            "unit": unit,
            **({"scope": scope} if scope else {}),
        },
        "support": {
            "level": support_level,
            "fallback_level": fallback_level,
            "effective_support": int(len(observations)),
            "status": "available",
        },
        "evidence": {
            "n_history_rows": int(len(rows)),
            "n_exact_rows": int(_to_int((selection.get("counts") or {}).get("n_exact_rows")) or 0),
            "n_nearby_rows": int(_to_int((selection.get("counts") or {}).get("n_nearby_rows")) or 0),
            "n_coarse_rows": int(_to_int((selection.get("counts") or {}).get("n_coarse_rows")) or 0),
            "n_selected_rows": int(len(observations)),
            "n_campaign_correction_rows": int(campaign_rows),
        },
    }
    return {
        "target": target_block,
        "corrections": {
            "campaign_local_residual": float(campaign_delta_log) if apply_campaign_correction else 0.0,
            "total": float(campaign_delta_log) if apply_campaign_correction else 0.0,
        },
        "guards": {
            "base": float(base_guard_log),
            "support": float(support_guard_log),
            "fallback": float(fallback_guard_log),
            "safety": float(safety_guard_log),
            "total": float(base_guard_log + support_guard_log + fallback_guard_log + safety_guard_log),
        },
    }


def _support_rank(level: str) -> int:
    return int(_SUPPORT_ORDER.get(str(level or "none"), 0))


def _fallback_rank(level: str) -> int:
    return int(_FALLBACK_ORDER.get(str(level or "none"), _FALLBACK_ORDER["none"]))


def _target_support_level(target: Mapping[str, Any]) -> str:
    support = dict(target.get("support") or {}) if isinstance(target.get("support"), Mapping) else {}
    return str(support.get("level") or "none").strip().lower() or "none"


def _target_fallback_level(target: Mapping[str, Any]) -> str:
    support = dict(target.get("support") or {}) if isinstance(target.get("support"), Mapping) else {}
    return str(support.get("fallback_level") or "none").strip().lower() or "none"


def _target_effective_support(target: Mapping[str, Any]) -> int:
    support = dict(target.get("support") or {}) if isinstance(target.get("support"), Mapping) else {}
    return int(_to_int(support.get("effective_support")) or 0)


def _target_center(target: Mapping[str, Any]) -> Optional[float]:
    estimate = dict(target.get("estimate") or {}) if isinstance(target.get("estimate"), Mapping) else {}
    return _to_float(estimate.get("center"))


def _target_upper(target: Mapping[str, Any]) -> Optional[float]:
    estimate = dict(target.get("estimate") or {}) if isinstance(target.get("estimate"), Mapping) else {}
    return _to_float(estimate.get("upper"))


def _aggregate_runtime_compatibility(hot: Mapping[str, Any], transition: Mapping[str, Any]) -> Dict[str, Any]:
    hot_center = _target_center(hot)
    hot_upper = _target_upper(hot)
    transition_center = _target_center(transition)
    transition_upper = _target_upper(transition)
    if hot_center is None or hot_upper is None or transition_center is None or transition_upper is None:
        return {
            "predicted_runtime_sec": None,
            "predicted_p90_sec": None,
            "runtime_guard_margin_sec": None,
            "support_level": "none",
            "fallback_level": "none",
        }
    support_level = min(
        (_target_support_level(hot), _target_support_level(transition)),
        key=_support_rank,
    )
    fallback_level = max(
        (_target_fallback_level(hot), _target_fallback_level(transition)),
        key=_fallback_rank,
    )
    runtime_center = float(hot_center + transition_center)
    runtime_upper = float(hot_upper + transition_upper)
    return {
        "predicted_runtime_sec": runtime_center,
        "predicted_p90_sec": runtime_upper,
        "runtime_guard_margin_sec": max(0.0, runtime_upper - runtime_center),
        "support_level": support_level,
        "fallback_level": fallback_level,
    }


def _compatibility_reasons(*, runtime: Mapping[str, Any], memory_target: Mapping[str, Any], baseline_target: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    runtime_support = str(runtime.get("support_level") or "none")
    runtime_fallback = str(runtime.get("fallback_level") or "none")
    reasons.append(f"runtime_support_{runtime_support}")
    if runtime_fallback not in {"", "none"}:
        reasons.append(f"runtime_fallback_{runtime_fallback}")
    if runtime_support == "none":
        reasons.append("runtime_abstained")

    memory_support = _target_support_level(memory_target)
    memory_fallback = _target_fallback_level(memory_target)
    reasons.append(f"active_memory_support_{memory_support}")
    if memory_fallback not in {"", "none"}:
        reasons.append(f"active_memory_fallback_{memory_fallback}")
    if memory_support == "none":
        reasons.append("active_memory_abstained")

    baseline_support = _target_support_level(baseline_target)
    if baseline_support == "none":
        reasons.append("resident_baseline_abstained")
    return list(dict.fromkeys(reasons))


def _estimate_execution_envelope(
    *,
    rows: Sequence[Any],
    safety_rows: Sequence[Any],
    query_context: Mapping[str, Any],
    campaign_id: str,
) -> Dict[str, Any]:
    hot = _estimate_target(
        rows=rows,
        safety_rows=safety_rows,
        query_context=query_context,
        context_getter=lambda row: _row_query_context(row).get("execution_envelope", {}),
        matcher=_match_execution_context,
        extract_value=_hot_execution_time_sec,
        safety_target_kind="runtime",
        safety_actual_getter=_safety_actual_hot_execution_sec,
        safety_upper_getter=_safety_runtime_hot_upper,
        campaign_id=campaign_id,
        guard_q=_RUNTIME_GUARD_Q,
        unit="sec",
        apply_campaign_correction=True,
    )
    transition = _estimate_target(
        rows=rows,
        safety_rows=safety_rows,
        query_context=query_context,
        context_getter=lambda row: _row_query_context(row).get("execution_envelope", {}),
        matcher=_match_execution_context,
        extract_value=_transition_penalty_sec,
        safety_target_kind="runtime",
        safety_actual_getter=_safety_actual_transition_penalty_sec,
        safety_upper_getter=_safety_runtime_transition_upper,
        campaign_id=campaign_id,
        guard_q=_RUNTIME_GUARD_Q,
        unit="sec",
        apply_campaign_correction=True,
    )
    active_memory = _estimate_target(
        rows=rows,
        safety_rows=safety_rows,
        query_context=query_context,
        context_getter=lambda row: _row_query_context(row).get("execution_envelope", {}),
        matcher=_match_execution_context,
        extract_value=_active_memory_mib,
        safety_target_kind="memory",
        safety_actual_getter=_safety_actual_active_memory_mib,
        safety_upper_getter=_safety_active_memory_upper,
        campaign_id=campaign_id,
        guard_q=_MEMORY_GUARD_Q,
        unit="mib",
        scope=str(query_context.get("memory_basis") or query_context.get("active_memory_scope") or "").strip(),
        apply_campaign_correction=True,
    )
    return {
        "query_context": dict(query_context),
        "hot_execution_time_sec": dict(hot.get("target") or {}),
        "transition_penalty_sec": dict(transition.get("target") or {}),
        "active_memory_mib": dict(active_memory.get("target") or {}),
        "corrections": {
            "hot_execution_time_sec": dict(hot.get("corrections") or {}),
            "transition_penalty_sec": dict(transition.get("corrections") or {}),
            "active_memory_mib": dict(active_memory.get("corrections") or {}),
        },
        "guards": {
            "hot_execution_time_sec": dict(hot.get("guards") or {}),
            "transition_penalty_sec": dict(transition.get("guards") or {}),
            "active_memory_mib": dict(active_memory.get("guards") or {}),
        },
    }


def _estimate_replica_baseline(
    *,
    rows: Sequence[Any],
    safety_rows: Sequence[Any],
    query_context: Mapping[str, Any],
) -> Dict[str, Any]:
    resident = _estimate_target(
        rows=rows,
        safety_rows=safety_rows,
        query_context=query_context,
        context_getter=lambda row: _row_query_context(row).get("replica_baseline", {}),
        matcher=_match_baseline_context,
        extract_value=_resident_memory_mib,
        safety_target_kind="memory",
        safety_actual_getter=_safety_actual_resident_memory_mib,
        safety_upper_getter=_safety_resident_memory_upper,
        campaign_id="",
        guard_q=_MEMORY_GUARD_Q,
        unit="mib",
        apply_campaign_correction=False,
    )
    return {
        "query_context": dict(query_context),
        "resident_memory_mib": dict(resident.get("target") or {}),
        "corrections": {
            "resident_memory_mib": dict(resident.get("corrections") or {}),
        },
        "guards": {
            "resident_memory_mib": dict(resident.get("guards") or {}),
        },
    }


def predict_cost(
    *,
    run_index: Any,
    component: str,
    level: Optional[str],
    config_fingerprint: str,
    input_fingerprint: str,
    workload_features: Optional[Mapping[str, Any]] = None,
    execution_overrides: Optional[Mapping[str, Any]] = None,
    worker_context: Optional[Mapping[str, Any]] = None,
    campaign_id: str = "",
    history_limit: int = 400,
) -> Dict[str, Any]:
    normalized_workload = dict(workload_features or {}) if isinstance(workload_features, Mapping) else {}
    normalized_overrides = dict(execution_overrides or {}) if isinstance(execution_overrides, Mapping) else {}
    normalized_worker_context = dict(worker_context or {}) if isinstance(worker_context, Mapping) else {}
    rows = _collect_rows(
        run_index=run_index,
        component=component,
        level=level,
        history_limit=history_limit,
    )
    safety_rows = _collect_recent_task_rows(
        run_index=run_index,
        component=component,
        level=level,
        history_limit=history_limit,
    )
    query_context = _build_query_context(
        component=component,
        level=level,
        config_fingerprint=str(config_fingerprint or "").strip(),
        input_fingerprint=str(input_fingerprint or "").strip(),
        workload_features=normalized_workload,
        execution_overrides=normalized_overrides,
        worker_context=normalized_worker_context,
        campaign_id=str(campaign_id or "").strip(),
    )
    execution_envelope = _estimate_execution_envelope(
        rows=rows,
        safety_rows=safety_rows,
        query_context=dict(query_context.get("execution_envelope") or {}),
        campaign_id=str(query_context.get("campaign_id") or "").strip(),
    )
    replica_baseline = _estimate_replica_baseline(
        rows=rows,
        safety_rows=safety_rows,
        query_context=dict(query_context.get("replica_baseline") or {}),
    )

    hot_target = dict(execution_envelope.get("hot_execution_time_sec") or {})
    transition_target = dict(execution_envelope.get("transition_penalty_sec") or {})
    active_memory_target = dict(execution_envelope.get("active_memory_mib") or {})
    resident_target = dict(replica_baseline.get("resident_memory_mib") or {})
    runtime_summary = _aggregate_runtime_compatibility(hot_target, transition_target)
    active_center = _target_center(active_memory_target)
    active_upper = _target_upper(active_memory_target)
    memory_basis = str(
        (
            dict(active_memory_target.get("estimate") or {})
            if isinstance(active_memory_target.get("estimate"), Mapping)
            else {}
        ).get("scope")
        or ""
    ).strip()
    if not memory_basis:
        memory_basis = str(
            dict(query_context.get("execution_envelope") or {}).get("memory_basis")
            or dict(query_context.get("execution_envelope") or {}).get("active_memory_scope")
            or ""
        ).strip()
    peak_fidelity = "incremental_peak" if memory_basis == "increment_over_resident" else "full_peak"
    resident_upper = _to_float(_target_upper(resident_target))
    total_upper_bound_mib = active_upper
    if memory_basis == "increment_over_resident" and active_upper is not None and resident_upper is not None:
        total_upper_bound_mib = float(active_upper + resident_upper)

    compatibility = {
        "predicted_runtime_sec": _to_float(runtime_summary.get("predicted_runtime_sec")),
        "predicted_p90_sec": _to_float(runtime_summary.get("predicted_p90_sec")),
        "predicted_peak_mem_p90_mib": _to_float(active_upper),
        "active_memory_scope": memory_basis,
        "memory_basis": memory_basis,
        "peak_fidelity": peak_fidelity,
        "total_upper_bound_mib": _to_float(total_upper_bound_mib),
        "runtime_guard_margin_sec": _to_float(runtime_summary.get("runtime_guard_margin_sec")),
        "memory_guard_margin_mib": (
            max(0.0, float(active_upper) - float(active_center))
            if active_upper is not None and active_center is not None
            else None
        ),
        "support_level": str(runtime_summary.get("support_level") or "none"),
        "fallback_level": str(runtime_summary.get("fallback_level") or "none"),
        "memory_support_level": _target_support_level(active_memory_target),
        "memory_fallback_level": _target_fallback_level(active_memory_target),
        "resident_support_level": _target_support_level(resident_target),
        "resident_fallback_level": _target_fallback_level(resident_target),
        "predicted_resident_memory_mib": _to_float(_target_center(resident_target)),
        "predicted_resident_memory_p95_mib": resident_upper,
    }
    compatibility["reasons"] = _compatibility_reasons(
        runtime=runtime_summary,
        memory_target=active_memory_target,
        baseline_target=resident_target,
    )

    return {
        "version": 2,
        "query_context": query_context,
        "execution_envelope": {
            "query_context": dict(execution_envelope.get("query_context") or {}),
            "hot_execution_time_sec": hot_target,
            "transition_penalty_sec": transition_target,
            "active_memory_mib": active_memory_target,
        },
        "replica_baseline": {
            "query_context": dict(replica_baseline.get("query_context") or {}),
            "resident_memory_mib": resident_target,
        },
        "corrections": {
            "space": "log",
            "execution_envelope": dict(execution_envelope.get("corrections") or {}),
            "replica_baseline": dict(replica_baseline.get("corrections") or {}),
        },
        "guards": {
            "space": "log",
            "execution_envelope": dict(execution_envelope.get("guards") or {}),
            "replica_baseline": dict(replica_baseline.get("guards") or {}),
        },
        "compatibility": compatibility,
    }


__all__ = ["predict_cost"]
