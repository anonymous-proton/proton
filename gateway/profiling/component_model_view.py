"""Canonical component-model payload serializer."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _extract_target_data_points(metrics: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    raw = metrics.get("target_data_points")
    if isinstance(raw, Mapping):
        for target, payload in raw.items():
            if not isinstance(payload, Mapping):
                continue
            target_key = str(target or "").strip()
            if not target_key:
                continue
            out[target_key] = {
                "validation_datapoints_total": _to_int(payload.get("validation_datapoints_total")),
                "training_datapoints_per_fold_avg": _to_float(payload.get("training_datapoints_per_fold_avg")),
                "k_folds_min": _to_int(payload.get("k_folds_min")),
                "k_folds_max": _to_int(payload.get("k_folds_max")),
                "k_folds_mode": _to_int(payload.get("k_folds_mode")),
                "segments": _to_int(payload.get("segments")),
            }
    return out


def choose_effective_model_view(
    *,
    latest_model: Optional[Mapping[str, Any]],
    active_model: Optional[Mapping[str, Any]],
    promotion_gate: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gate_passed = bool((promotion_gate or {}).get("passed", False))
    if gate_passed:
        chosen = latest_model or active_model or {}
    else:
        chosen = active_model or latest_model or {}
    return dict(chosen) if isinstance(chosen, Mapping) else {}


def build_component_model_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    readiness = dict(row.get("readiness") or {})
    priority = dict(row.get("priority") or {})
    delta = dict(row.get("delta") or {})
    decision_gate = dict(row.get("decision_gate") or {})
    history_metrics = (
        dict(metrics.get("history_metrics") or {})
        if isinstance(metrics.get("history_metrics"), Mapping)
        else {}
    )
    decision_quality = (
        dict(decision_gate.get("decision_quality") or {})
        if isinstance(decision_gate.get("decision_quality"), Mapping)
        else {}
    )
    run_meta = dict(metrics.get("run_meta") or {}) if isinstance(metrics.get("run_meta"), Mapping) else {}
    data_points_by_target = _extract_target_data_points(metrics)
    artifacts = dict(row.get("artifacts") or {})
    return {
        "component": str(row.get("component") or "").strip().lower(),
        "component_model_version_id": str(row.get("component_model_version_id") or ""),
        "parent_loop_model_version": str(row.get("parent_loop_model_version_id") or ""),
        "job_id": str(row.get("job_id") or ""),
        "loop_idx": int(row.get("loop_idx") or 0),
        "trained_at": row.get("trained_at"),
        "status": str(readiness.get("status") or row.get("status") or "CAUTION").upper(),
        "ready_ratio": readiness.get("ready_ratio"),
        "blocked_ratio": readiness.get("blocked_ratio"),
        "contam_cell_ratio": readiness.get("contam_cell_ratio"),
        "estimated_remaining_runs": readiness.get("estimated_remaining_runs"),
        "eta_gpu_sec_to_ready": readiness.get("eta_gpu_sec_to_ready"),
        "n_obs_recent": _to_int(history_metrics.get("n_obs_recent")),
        "unique_inputs": _to_int(history_metrics.get("unique_inputs")),
        "unique_configs": _to_int(history_metrics.get("unique_configs")),
        "median_runtime_sec": _to_float(history_metrics.get("median_runtime_sec")),
        "pi_width_rel": _to_float(history_metrics.get("pi_width_rel")),
        "runtime_ape_p50": _to_float(decision_quality.get("runtime_ape_p50")),
        "runtime_ape_p90": _to_float(decision_quality.get("runtime_ape_p90")),
        "runtime_bias_sec_median": _to_float(decision_quality.get("runtime_bias_sec_median")),
        "runtime_log_mae": _to_float(decision_quality.get("runtime_log_mae")),
        "mem_fn_rate": _to_float(decision_quality.get("mem_fn_rate")),
        "gap_fit_fp_rate": _to_float(decision_quality.get("gap_fit_fp_rate")),
        "residual_p90": _to_float(decision_quality.get("residual_p90")),
        "residual_p95": _to_float(decision_quality.get("residual_p95")),
        "calibration_error": _to_float(decision_quality.get("calibration_error")),
        "n_runtime_checks": _to_int(decision_quality.get("n_runtime_checks")),
        "n_mem_checks": _to_int(decision_quality.get("n_mem_checks")),
        "n_gap_fit_checks": _to_int(decision_quality.get("n_gap_fit_checks")),
        "n_residual_checks": _to_int(decision_quality.get("n_residual_checks")),
        "n_calibration_checks": _to_int(decision_quality.get("n_calibration_checks")),
        "quality_score": _to_float(decision_quality.get("quality_score")),
        "confidence": _to_float(decision_quality.get("confidence")),
        "priority_rank": priority.get("rank"),
        "priority_need": priority.get("need"),
        "priority_value": priority.get("value"),
        "priority_cost": priority.get("cost"),
        "priority_risk": priority.get("risk"),
        "priority_reasons": list(priority.get("reasons") or []),
        "delta_ready_ratio": delta.get("ready_ratio"),
        "delta_eta_gpu_sec_to_ready": delta.get("eta_gpu_sec_to_ready"),
        "delta_runtime_ape_p90": delta.get("runtime_ape_p90"),
        "delta_blocked_ratio": delta.get("blocked_ratio"),
        "delta_contam_cell_ratio": delta.get("contam_cell_ratio"),
        "delta_state": str(delta.get("state") or ""),
        "delta_reason": str(delta.get("reason") or ""),
        "holdout_group": str(run_meta.get("holdout_group") or ""),
        "data_points_by_target": data_points_by_target,
        "cell_schema_ids": list(row.get("cell_schema_ids") or []),
        "is_active": bool(row.get("is_active", False)),
        "error": str(row.get("error") or ""),
        "artifacts": artifacts,
    }
