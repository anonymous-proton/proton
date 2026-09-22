"""Canonical promotion-gate evaluation for component models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "status_not_blocked": True,
    "ready_ratio_min": 0.7,
    "runtime_ape_p90_max": 0.50,
    "mem_fn_rate_max": 0.02,
    "gap_fit_fp_rate_max": 0.05,
    "residual_p95_max": 0.70,
}


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def evaluate_promotion_gate(
    model_view: Optional[Mapping[str, Any]],
    *,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gate_thresholds = dict(_DEFAULT_THRESHOLDS)
    if isinstance(thresholds, Mapping):
        gate_thresholds.update(dict(thresholds))

    if not isinstance(model_view, Mapping):
        return {
            "passed": False,
            "reasons": ["latest_model_missing"],
            "measured": {},
            "thresholds": gate_thresholds,
        }

    status = str(model_view.get("status") or "").strip().upper()
    ready_ratio = _to_float(model_view.get("ready_ratio"))
    runtime_ape_p90 = _to_float(model_view.get("runtime_ape_p90"))
    mem_fn_rate = _to_float(model_view.get("mem_fn_rate"))
    gap_fit_fp_rate = _to_float(model_view.get("gap_fit_fp_rate"))
    residual_p95 = _to_float(model_view.get("residual_p95"))
    comp_ok = str(model_view.get("error") or "").strip() == ""

    reasons = []
    if not comp_ok:
        reasons.append("model_eval_error")
    if status == "BLOCKED":
        reasons.append("blocked_status")
    if ready_ratio is None:
        reasons.append("ready_ratio_missing")
    elif ready_ratio < float(gate_thresholds["ready_ratio_min"]):
        reasons.append("ready_ratio_below_threshold")
    if runtime_ape_p90 is None:
        reasons.append("runtime_ape_p90_missing")
    elif runtime_ape_p90 > float(gate_thresholds["runtime_ape_p90_max"]):
        reasons.append("runtime_ape_p90_above_threshold")
    if mem_fn_rate is None:
        reasons.append("mem_fn_rate_missing")
    elif mem_fn_rate > float(gate_thresholds["mem_fn_rate_max"]):
        reasons.append("mem_fn_rate_above_threshold")
    if gap_fit_fp_rate is not None and gap_fit_fp_rate > float(gate_thresholds["gap_fit_fp_rate_max"]):
        reasons.append("gap_fit_fp_rate_above_threshold")
    if residual_p95 is None:
        reasons.append("residual_p95_missing")
    elif residual_p95 > float(gate_thresholds["residual_p95_max"]):
        reasons.append("residual_p95_above_threshold")

    passed = (
        comp_ok
        and status != "BLOCKED"
        and ready_ratio is not None
        and ready_ratio >= float(gate_thresholds["ready_ratio_min"])
        and runtime_ape_p90 is not None
        and runtime_ape_p90 <= float(gate_thresholds["runtime_ape_p90_max"])
        and mem_fn_rate is not None
        and mem_fn_rate <= float(gate_thresholds["mem_fn_rate_max"])
        and (gap_fit_fp_rate is None or gap_fit_fp_rate <= float(gate_thresholds["gap_fit_fp_rate_max"]))
        and residual_p95 is not None
        and residual_p95 <= float(gate_thresholds["residual_p95_max"])
    )

    return {
        "passed": bool(passed),
        "reasons": reasons,
        "measured": {
            "comp_ok": comp_ok,
            "status": status,
            "ready_ratio": ready_ratio,
            "runtime_ape_p90": runtime_ape_p90,
            "mem_fn_rate": mem_fn_rate,
            "gap_fit_fp_rate": gap_fit_fp_rate,
            "residual_p95": residual_p95,
        },
        "thresholds": gate_thresholds,
    }
