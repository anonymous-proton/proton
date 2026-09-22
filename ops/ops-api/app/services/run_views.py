from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from app.services.run_explanations import (
    build_recent_run_summary,
    build_run_takeaway,
    build_target_explanation,
)
from app.services.payload_utils import (
    float_payload as _float,
    mapping_payload as _mapping,
    normalize_gateway_payload,
    text_payload as _text,
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_ts(value: Any) -> str | None:
    parsed = _float(value)
    if parsed is None or parsed <= 0:
        return None
    return datetime.fromtimestamp(parsed, tz=timezone.utc).isoformat(timespec="seconds")


def _timestamp_for_row(row: Mapping[str, Any]) -> float:
    for key in ("updated_at", "finished_at", "started_at", "created_at"):
        parsed = _float(row.get(key))
        if parsed is not None and parsed > 0:
            return float(parsed)
    return 0.0


def _selected_worker_context_summary(row: Mapping[str, Any]) -> str:
    selected_worker_context = _mapping(row.get("selected_worker_context"))
    execution_context = _mapping(row.get("execution_context"))
    parts = [
        _text(selected_worker_context.get("worker_name")) or _text(_mapping(row.get("chosen_worker")).get("name")),
        _text(execution_context.get("local_regime_bucket")),
        _text(execution_context.get("co_location_signature")),
        _text(execution_context.get("hardware_software")),
    ]
    return " / ".join([part for part in parts if part]) or "-"
def _support_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    distribution = {"exact": 0, "nearby": 0, "coarse": 0, "none": 0}
    for row in rows:
        block = _mapping(row.get(key))
        level = _text(block.get("support_level")).lower() or "none"
        if level not in distribution:
            level = "none"
        distribution[level] += 1
    return distribution


def _fallback_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    distribution = {"exact": 0, "nearby": 0, "coarse": 0, "none": 0}
    for row in rows:
        block = _mapping(row.get(key))
        level = _text(block.get("fallback_level")).lower() or "none"
        if level not in distribution:
            level = "none"
        distribution[level] += 1
    return distribution


def _quantile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(q))) * float(len(ordered) - 1)
    lower = int(pos)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = pos - float(lower)
    return float(ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction))


def _effective_support_summary(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float | None]:
    values = []
    for row in rows:
        block = _mapping(row.get(key))
        support = _float(block.get("effective_support"))
        if support is not None:
            values.append(float(support))
    return {
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
    }


def _sum_distribution_values(distribution: Mapping[str, int]) -> int:
    return sum(int(value or 0) for value in distribution.values())


def _oom_like_failure(row: Mapping[str, Any]) -> bool:
    message = " ".join(
        value
        for value in (
            _text(row.get("error")),
            _text(_mapping(row.get("actual")).get("failure_outcome")),
        )
        if value
    ).lower()
    return "oom" in message or "out of memory" in message or "cuda error out of memory" in message


def _campaign_correction_magnitude(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float | None]:
    magnitudes: list[float] = []
    for row in rows:
        for key in ("runtime_correction_total_log", "active_memory_correction_total_log"):
            value = _float(row.get(key))
            if value is None or abs(value) <= 1e-12:
                continue
            magnitudes.append(abs(value))
    return {
        "p50_log": _quantile(magnitudes, 0.50),
        "p90_log": _quantile(magnitudes, 0.90),
    }


def _safety_transition_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    ordered_rows = sorted(rows, key=_timestamp_for_row)
    previous = None
    expansion_count = 0
    contraction_count = 0
    for row in ordered_rows:
        current = max(
            abs(_float(row.get("runtime_safety_guard_log")) or 0.0),
            abs(_float(row.get("active_memory_safety_guard_log")) or 0.0),
            abs(_float(row.get("resident_baseline_safety_guard_log")) or 0.0),
        )
        if previous is not None:
            if current > previous + 1e-12:
                expansion_count += 1
            elif current + 1e-12 < previous:
                contraction_count += 1
        previous = current
    return expansion_count, contraction_count


def _prediction_error_by_local_regime_bucket(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        regime_bucket = _text(_mapping(row.get("execution_context")).get("local_regime_bucket")) or "-"
        bucket = buckets.setdefault(
            regime_bucket,
            {
                "count": 0,
                "runtime_error_sum": 0.0,
                "runtime_error_count": 0,
                "memory_error_sum": 0.0,
                "memory_error_count": 0,
            },
        )
        bucket["count"] += 1
        runtime_pred = _float(_mapping(row.get("compatibility")).get("runtime_p50"))
        runtime_actual = _float(row.get("actual_total_runtime_sec"))
        if runtime_pred is not None and runtime_actual is not None:
            bucket["runtime_error_sum"] += abs(runtime_actual - runtime_pred)
            bucket["runtime_error_count"] += 1
        memory_pred = _float(_mapping(_mapping(row.get("active_target")).get("estimate")).get("center"))
        memory_actual = _float(row.get("actual_active_memory_mib"))
        if memory_pred is not None and memory_actual is not None:
            bucket["memory_error_sum"] += abs(memory_actual - memory_pred)
            bucket["memory_error_count"] += 1
    payload: Dict[str, Dict[str, Any]] = {}
    for regime_bucket, bucket in buckets.items():
        payload[regime_bucket] = {
            "count": int(bucket["count"]),
            "runtime_mae_sec": (
                float(bucket["runtime_error_sum"]) / float(bucket["runtime_error_count"])
                if int(bucket["runtime_error_count"]) > 0
                else None
            ),
            "memory_mae_mib": (
                float(bucket["memory_error_sum"]) / float(bucket["memory_error_count"])
                if int(bucket["memory_error_count"]) > 0
                else None
            ),
        }
    return payload


def build_recent_run_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    execution_context = _mapping(row.get("execution_context"))
    chosen_worker = _mapping(row.get("chosen_worker"))
    gateway = normalize_gateway_payload(_mapping(row.get("gateway")))
    worker_id = _text(row.get("dispatch_worker_name")) or _text(chosen_worker.get("name")) or "-"
    summary = build_recent_run_summary(row)
    return {
        "time": _iso_ts(_timestamp_for_row(row)),
        "run_key": _text(row.get("run_key")),
        "campaign_id": _text(row.get("campaign_id")) or "-",
        "stage_id": _text(_mapping(row.get("axes")).get("stage_id")) or "-",
        "model_id": _text(row.get("component")) or "-",
        "worker_id": worker_id,
        "status": _text(row.get("state")) or "UNKNOWN",
        "gateway": gateway,
        "gateway_label": _text(gateway.get("label")) or "unknown",
        "selected_context_applied": _text(row.get("selected_context_applied")) or "missing_worker_context",
        "selected_worker_context_summary": _selected_worker_context_summary(row),
        "local_regime_bucket": _text(execution_context.get("local_regime_bucket")) or "-",
        "co_location_signature": _text(execution_context.get("co_location_signature")) or "-",
        "takeaway": _text(summary.get("takeaway")),
        "runtime": _mapping(summary.get("runtime")),
        "active_memory": _mapping(summary.get("active_memory")),
        "resident_baseline": _mapping(summary.get("resident_baseline")),
        "telemetry_missing": bool(
            _text(_mapping(summary.get("runtime")).get("status")) == "telemetry_missing"
            or _text(_mapping(summary.get("active_memory")).get("status")) == "telemetry_missing"
        ),
    }


def summarize_estimator_health(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    recent_rows = [build_recent_run_row(row) for row in rows]
    runtime_fallback_distribution = _fallback_distribution(recent_rows, "runtime")
    active_memory_fallback_distribution = _fallback_distribution(recent_rows, "active_memory")
    resident_fallback_distribution = _fallback_distribution(recent_rows, "resident_baseline")
    safety_guard_expansion_count, safety_guard_contraction_count = _safety_transition_counts(rows)
    summary = {
        "last_1h_runs": len(recent_rows),
        "runtime_support_distribution": _support_distribution(recent_rows, "runtime"),
        "runtime_fallback_distribution": runtime_fallback_distribution,
        "runtime_fallback_ratio": (
            float(_sum_distribution_values(runtime_fallback_distribution) - int(runtime_fallback_distribution.get("exact") or 0))
            / float(_sum_distribution_values(runtime_fallback_distribution))
            if _sum_distribution_values(runtime_fallback_distribution) > 0
            else 0.0
        ),
        "runtime_effective_support_summary": _effective_support_summary(recent_rows, "runtime"),
        "active_memory_support_distribution": _support_distribution(recent_rows, "active_memory"),
        "active_memory_fallback_distribution": active_memory_fallback_distribution,
        "active_memory_fallback_ratio": (
            float(_sum_distribution_values(active_memory_fallback_distribution) - int(active_memory_fallback_distribution.get("exact") or 0))
            / float(_sum_distribution_values(active_memory_fallback_distribution))
            if _sum_distribution_values(active_memory_fallback_distribution) > 0
            else 0.0
        ),
        "active_memory_effective_support_summary": _effective_support_summary(recent_rows, "active_memory"),
        "resident_baseline_support_distribution": _support_distribution(recent_rows, "resident_baseline"),
        "resident_baseline_fallback_distribution": resident_fallback_distribution,
        "resident_baseline_effective_support_summary": _effective_support_summary(recent_rows, "resident_baseline"),
        "runtime_violation_count": 0,
        "active_memory_violation_count": 0,
        "runtime_abstention_count": 0,
        "active_memory_abstention_count": 0,
        "resident_baseline_abstention_count": 0,
        "campaign_correction_applied_count": 0,
        "campaign_correction_magnitude": _campaign_correction_magnitude(rows),
        "telemetry_missing_count": 0,
        "post_selection_retry_count": 0,
        "oom_incidence_count": 0,
        "safety_guard_expansion_count": safety_guard_expansion_count,
        "safety_guard_contraction_count": safety_guard_contraction_count,
        "prediction_error_by_local_regime_bucket": _prediction_error_by_local_regime_bucket(rows),
    }
    for row in rows:
        runtime = build_target_explanation(row=row, target_id="runtime")
        active_memory = build_target_explanation(row=row, target_id="active_memory")
        resident_baseline = build_target_explanation(row=row, target_id="resident_baseline")
        if _text(runtime.get("status")) == "predicted_violated":
            summary["runtime_violation_count"] += 1
        if _text(active_memory.get("status")) == "predicted_violated":
            summary["active_memory_violation_count"] += 1
        if _text(runtime.get("status")) == "abstained":
            summary["runtime_abstention_count"] += 1
        if _text(active_memory.get("status")) == "abstained":
            summary["active_memory_abstention_count"] += 1
        if _text(resident_baseline.get("status")) == "abstained":
            summary["resident_baseline_abstention_count"] += 1
        if _float(runtime.get("campaign_correction_total")) is not None or _float(active_memory.get("campaign_correction_total")) is not None:
            if abs(_float(runtime.get("campaign_correction_total")) or 0.0) > 1e-12 or abs(_float(active_memory.get("campaign_correction_total")) or 0.0) > 1e-12:
                summary["campaign_correction_applied_count"] += 1
        if _text(runtime.get("status")) == "telemetry_missing" or _text(active_memory.get("status")) == "telemetry_missing":
            summary["telemetry_missing_count"] += 1
        if _text(row.get("selected_context_applied")) == "post_selection_retry":
            summary["post_selection_retry_count"] += 1
        if _oom_like_failure(row):
            summary["oom_incidence_count"] += 1
    return summary


def build_recent_runs_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
    total: int,
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    return {
        "generated_at": _iso_now(),
        "filters": {
            "campaign_id": _text(filters.get("campaign_id")) or None,
            "component": _text(filters.get("component")) or None,
            "worker_name": _text(filters.get("worker_name")) or None,
            "include_gateway_instance_id": list(filters.get("include_gateway_instance_id") or []),
            "exclude_gateway_instance_id": list(filters.get("exclude_gateway_instance_id") or []),
            "include_gateway_git_commit": list(filters.get("include_gateway_git_commit") or []),
            "exclude_gateway_git_commit": list(filters.get("exclude_gateway_git_commit") or []),
        },
        "summary": summarize_estimator_health(summary_rows),
        "total": int(total),
        "limit": int(limit),
        "offset": int(offset),
        "runs": [build_recent_run_row(row) for row in rows],
    }


def build_run_detail_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    execution_context = _mapping(row.get("execution_context"))
    selected_worker_context = _mapping(row.get("selected_worker_context"))
    chosen_worker = _mapping(row.get("chosen_worker"))
    worker_id = _text(row.get("dispatch_worker_name")) or _text(chosen_worker.get("name")) or "-"
    stage_id = _text(_mapping(row.get("axes")).get("stage_id")) or "-"
    runtime = build_target_explanation(row=row, target_id="runtime")
    active_memory = build_target_explanation(row=row, target_id="active_memory")
    resident_baseline = build_target_explanation(row=row, target_id="resident_baseline")
    hot_execution = build_target_explanation(row=row, target_id="hot_execution_time")
    transition_penalty = build_target_explanation(row=row, target_id="transition_penalty")
    return {
        "generated_at": _iso_now(),
        "run_key": _text(row.get("run_key")),
        "overview": build_recent_run_row(row),
        "takeaway": build_run_takeaway(row),
        "request_trace_context": {
            "campaign_id": _text(row.get("campaign_id")) or "-",
            "stage_id": stage_id,
            "model_id": _text(row.get("component")) or "-",
            "stage_class": _text(_mapping(row.get("query_context")).get("stage_class"))
            or _text(execution_context.get("stage_class"))
            or "-",
            "descriptor_bucket": _text(execution_context.get("descriptor_bucket")) or "-",
            "requested_batch_size": execution_context.get("requested_batch_size"),
            "effective_batch_bucket": _text(execution_context.get("effective_batch_bucket")) or "-",
        },
        "selected_worker_context": {
            "worker_id": worker_id,
            "hardware_software": _text(execution_context.get("hardware_software"))
            or _text(selected_worker_context.get("hardware_software"))
            or "-",
            "active_request_count_bucket": _text(execution_context.get("active_request_count_bucket")) or "-",
            "co_location_signature": _text(execution_context.get("co_location_signature")) or "-",
            "residency_state": _text(selected_worker_context.get("residency_state")) or "-",
        },
        "estimator_explanations": {
            "runtime": runtime,
            "active_memory": active_memory,
            "resident_baseline": resident_baseline,
            "hot_execution_time": hot_execution,
            "transition_penalty": transition_penalty,
        },
        "actual_telemetry": {
            "actual_hot_execution_sec": _float(row.get("actual_hot_execution_sec")),
            "actual_transition_penalty_sec": _float(row.get("actual_transition_penalty_sec")),
            "actual_total_runtime_sec": _float(row.get("actual_total_runtime_sec")),
            "actual_active_memory_mib": _float(row.get("actual_active_memory_mib")),
            "dispatch_time_resident_baseline_mib": _float(row.get("actual_resident_memory_mib")),
            "dispatch_time_resident_baseline_state": _text(row.get("resident_baseline_state")) or "-",
            "failure_outcome": _text(row.get("error")) or (_text(row.get("state")) if _text(row.get("state")) in {"FAILED", "CANCELLED"} else ""),
            "runtime_violation": _text(_mapping(runtime).get("status")) == "predicted_violated",
            "memory_violation": _text(_mapping(active_memory).get("status")) == "predicted_violated",
            "telemetry_missing": bool(
                _text(_mapping(runtime).get("status")) == "telemetry_missing"
                or _text(_mapping(active_memory).get("status")) == "telemetry_missing"
            ),
        },
        "raw": {
            "run_record": dict(_mapping(row.get("raw"))),
            "scheduler_decision": dict(_mapping(row.get("scheduler_decision"))),
            "axes": dict(_mapping(row.get("axes"))),
        },
    }
