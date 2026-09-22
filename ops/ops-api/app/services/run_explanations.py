from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from app.services.explanation_utils import (
    display_unit as _display_unit,
    effective_support as _effective_support,
    estimate_center as _estimate_center,
    estimate_scope as _estimate_scope,
    estimate_upper as _estimate_upper,
    fallback_level as _fallback_level,
    fallback_rank as _fallback_rank,
    normalized_evidence as _normalized_evidence,
    support_level as _support_level,
    support_rank as _support_rank,
)
from app.services.payload_utils import (
    float_payload as _float,
    int_payload as _int,
    mapping_payload as _mapping,
    text_payload as _text,
)

_TARGET_TITLES = {
    "runtime": "Runtime",
    "active_memory": "Active Memory",
    "resident_baseline": "Resident Baseline",
    "hot_execution_time": "Hot Execution Time",
    "transition_penalty": "Transition Penalty",
}
_RUNTIME_UNIT = "sec"
_MEMORY_UNIT = "mib"


def _min_or_none(*values: int) -> int:
    existing = [int(value) for value in values if value is not None]
    if not existing:
        return 0
    return int(min(existing))


def _sum_or_none(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values if value is not None))


def _reason_codes(row: Mapping[str, Any]) -> list[str]:
    scheduler_decision = _mapping(row.get("scheduler_decision"))
    placement = _mapping(scheduler_decision.get("placement"))
    return [
        _text(item)
        for item in list(scheduler_decision.get("reasons") or placement.get("reasons") or [])
        if _text(item)
    ]


def _target_blocks(row: Mapping[str, Any], target_id: str) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float | None, str]:
    corrections = _mapping(row.get("corrections"))
    guards = _mapping(row.get("guards"))
    execution_corrections = _mapping(corrections.get("execution_envelope"))
    baseline_corrections = _mapping(corrections.get("replica_baseline"))
    execution_guards = _mapping(guards.get("execution_envelope"))
    baseline_guards = _mapping(guards.get("replica_baseline"))
    if target_id == "hot_execution_time":
        return (
            _mapping(row.get("hot_target")),
            _mapping(execution_corrections.get("hot_execution_time_sec")),
            _mapping(execution_guards.get("hot_execution_time_sec")),
            _float(row.get("actual_hot_execution_sec")),
            _RUNTIME_UNIT,
        )
    if target_id == "transition_penalty":
        return (
            _mapping(row.get("transition_target")),
            _mapping(execution_corrections.get("transition_penalty_sec")),
            _mapping(execution_guards.get("transition_penalty_sec")),
            _float(row.get("actual_transition_penalty_sec")),
            _RUNTIME_UNIT,
        )
    if target_id == "active_memory":
        return (
            _mapping(row.get("active_target")),
            _mapping(execution_corrections.get("active_memory_mib")),
            _mapping(execution_guards.get("active_memory_mib")),
            _float(row.get("actual_active_memory_mib")),
            _MEMORY_UNIT,
        )
    if target_id == "resident_baseline":
        return (
            _mapping(row.get("resident_target")),
            _mapping(baseline_corrections.get("resident_memory_mib")),
            _mapping(baseline_guards.get("resident_memory_mib")),
            _float(row.get("actual_resident_memory_mib")),
            _MEMORY_UNIT,
        )
    raise ValueError(f"unsupported target_id: {target_id}")


def _guard_contributions(center: float | None, upper: float | None, guard_block: Mapping[str, Any]) -> Dict[str, float | None]:
    if center is None:
        return {
            "base": None,
            "support": None,
            "fallback": None,
            "safety": None,
            "total": None,
        }
    base_log = _float(guard_block.get("base"))
    support_log = _float(guard_block.get("support"))
    fallback_log = _float(guard_block.get("fallback"))
    safety_log = _float(guard_block.get("safety"))
    if all(value is None for value in (base_log, support_log, fallback_log, safety_log)):
        return {
            "base": None,
            "support": None,
            "fallback": None,
            "safety": None,
            "total": float(upper - center) if upper is not None else None,
        }

    running = float(center)
    running_log = 0.0
    pieces: Dict[str, float | None] = {}
    for key, raw_log in (
        ("base", base_log),
        ("support", support_log),
        ("fallback", fallback_log),
        ("safety", safety_log),
    ):
        if raw_log is None:
            pieces[key] = 0.0
            continue
        previous = float(center) * math.exp(running_log)
        running_log += float(raw_log)
        current = float(center) * math.exp(running_log)
        pieces[key] = max(0.0, current - previous)
        running = current
    pieces["total"] = max(0.0, running - float(center))
    return pieces


def _correction_delta_units(center: float | None, correction_total: float | None) -> float | None:
    if center is None or correction_total is None:
        return None
    shared_center = float(center) / math.exp(float(correction_total))
    return float(center - shared_center)


def _classify_status(*, center: float | None, upper: float | None, actual: float | None) -> str:
    if center is None or upper is None:
        return "abstained"
    if actual is None:
        return "telemetry_missing"
    if actual > upper:
        return "predicted_violated"
    return "predicted_ok"


def _generic_abstain_reason(*, evidence: Mapping[str, int], actual: float | None) -> tuple[str | None, str | None]:
    if int(evidence.get("n_history_rows") or 0) <= 0:
        return "no_history_rows", "no usable modeling history exists for this target"
    if (
        int(evidence.get("n_exact_rows") or 0) <= 0
        and int(evidence.get("n_nearby_rows") or 0) <= 0
        and int(evidence.get("n_coarse_rows") or 0) <= 0
    ):
        return "no_compatible_rows", "no compatible rows matched the requested target context"
    if actual is None:
        return "telemetry_missing", "actual telemetry is missing for this target"
    return "no_compatible_rows", "no compatible rows matched the requested target context"


def _active_memory_abstain_reason(row: Mapping[str, Any], evidence: Mapping[str, int], actual: float | None) -> tuple[str | None, str | None]:
    if int(evidence.get("n_history_rows") or 0) <= 0:
        return "no_history_rows", "no modeling history exists for active memory yet"
    if (
        actual is not None
        and int(evidence.get("n_selected_rows") or 0) <= 0
        and ("active_memory_abstained" in _reason_codes(row) or not bool(row.get("actual_memory_qc_keep")))
    ):
        return "no_qc_kept_rows", "no QC-kept compatible active-memory rows were available"
    if (
        int(evidence.get("n_exact_rows") or 0) <= 0
        and int(evidence.get("n_nearby_rows") or 0) <= 0
        and int(evidence.get("n_coarse_rows") or 0) <= 0
    ):
        return "no_compatible_rows", "no compatible active-memory rows matched the requested context"
    if actual is None:
        return "telemetry_missing", "active-memory telemetry is missing for this run"
    return "no_compatible_rows", "no compatible active-memory rows matched the requested context"


def _summary_sentence(
    *,
    title: str,
    status: str,
    support_level: str,
    effective_support: int,
    upper: float | None,
    actual: float | None,
    delta_vs_upper: float | None,
    abstain_reason: str | None,
    unit: str,
) -> str:
    if status == "predicted_violated":
        return (
            f"{title} used {support_level} support with {effective_support} usable rows, "
            f"predicted an upper bound of {upper:.1f} {_display_unit(unit)}, and actual was {actual:.1f} {_display_unit(unit)}, "
            f"which is {delta_vs_upper:.1f} {_display_unit(unit)} above the guard."
        )
    if status == "predicted_ok":
        margin = abs(delta_vs_upper or 0.0)
        return (
            f"{title} used {support_level} support with {effective_support} usable rows, "
            f"predicted an upper bound of {upper:.1f} {_display_unit(unit)}, and actual was {actual:.1f} {_display_unit(unit)}, "
            f"staying within the guard by {margin:.1f} {_display_unit(unit)}."
        )
    if status == "telemetry_missing":
        return (
            f"{title} produced a prediction but actual telemetry is missing, so the guard outcome cannot be checked."
        )
    return f"{title} abstained because {abstain_reason or 'no explanation is available'}."


def _build_single_target_explanation(
    *,
    row: Mapping[str, Any],
    target_id: str,
) -> Dict[str, Any]:
    target, correction_block, guard_block, actual, unit = _target_blocks(row, target_id)
    title = _TARGET_TITLES[target_id]
    center = _estimate_center(target)
    upper = _estimate_upper(target)
    support_level = _support_level(target)
    fallback_level = _fallback_level(target)
    effective_support = _effective_support(target)
    evidence = _normalized_evidence(target)
    guard_components = _guard_contributions(center, upper, guard_block)
    correction_total = _float(correction_block.get("total"))
    correction_delta_units = _correction_delta_units(center, correction_total)
    delta_vs_upper = float(actual - upper) if actual is not None and upper is not None else None
    status = _classify_status(center=center, upper=upper, actual=actual)
    if target_id == "active_memory":
        abstain_reason_code, abstain_reason = _active_memory_abstain_reason(row, evidence, actual)
    else:
        abstain_reason_code, abstain_reason = _generic_abstain_reason(evidence=evidence, actual=actual)
    if status != "abstained":
        abstain_reason_code, abstain_reason = None, None
    return {
        "target_id": target_id,
        "title": title,
        "status": status,
        "summary": _summary_sentence(
            title=title,
            status=status,
            support_level=support_level,
            effective_support=effective_support,
            upper=upper,
            actual=actual,
            delta_vs_upper=delta_vs_upper,
            abstain_reason=abstain_reason,
            unit=unit,
        ),
        "unit": unit,
        "scope": _estimate_scope(target),
        "center": center,
        "upper": upper,
        "actual": actual,
        "guard_margin": float(upper - center) if upper is not None and center is not None else None,
        "delta_vs_upper": delta_vs_upper,
        "support_level": support_level,
        "fallback_level": fallback_level,
        "effective_support": effective_support,
        "evidence": evidence,
        "guard_components": guard_components,
        "campaign_correction_total": correction_delta_units,
        "abstain_reason_code": abstain_reason_code,
        "abstain_reason": abstain_reason,
        "reason_codes": _reason_codes(row),
    }


def _runtime_guard_components(hot: Mapping[str, Any], transition: Mapping[str, Any]) -> Dict[str, float | None]:
    return {
        "base": _sum_or_none(_float(_mapping(hot.get("guard_components")).get("base")), _float(_mapping(transition.get("guard_components")).get("base"))),
        "support": _sum_or_none(_float(_mapping(hot.get("guard_components")).get("support")), _float(_mapping(transition.get("guard_components")).get("support"))),
        "fallback": _sum_or_none(_float(_mapping(hot.get("guard_components")).get("fallback")), _float(_mapping(transition.get("guard_components")).get("fallback"))),
        "safety": _sum_or_none(_float(_mapping(hot.get("guard_components")).get("safety")), _float(_mapping(transition.get("guard_components")).get("safety"))),
        "total": _sum_or_none(_float(_mapping(hot.get("guard_components")).get("total")), _float(_mapping(transition.get("guard_components")).get("total"))),
    }


def _runtime_evidence(hot: Mapping[str, Any], transition: Mapping[str, Any]) -> Dict[str, int]:
    hot_evidence = _mapping(hot.get("evidence"))
    transition_evidence = _mapping(transition.get("evidence"))
    if not hot_evidence:
        return {key: _int(transition_evidence.get(key)) for key in _normalized_evidence({}).keys()}
    if not transition_evidence:
        return {key: _int(hot_evidence.get(key)) for key in _normalized_evidence({}).keys()}
    return {
        "n_history_rows": _min_or_none(_int(hot_evidence.get("n_history_rows")), _int(transition_evidence.get("n_history_rows"))),
        "n_exact_rows": _min_or_none(_int(hot_evidence.get("n_exact_rows")), _int(transition_evidence.get("n_exact_rows"))),
        "n_nearby_rows": _min_or_none(_int(hot_evidence.get("n_nearby_rows")), _int(transition_evidence.get("n_nearby_rows"))),
        "n_coarse_rows": _min_or_none(_int(hot_evidence.get("n_coarse_rows")), _int(transition_evidence.get("n_coarse_rows"))),
        "n_selected_rows": _min_or_none(_int(hot_evidence.get("n_selected_rows")), _int(transition_evidence.get("n_selected_rows"))),
        "n_campaign_correction_rows": _min_or_none(
            _int(hot_evidence.get("n_campaign_correction_rows")),
            _int(transition_evidence.get("n_campaign_correction_rows")),
        ),
    }


def build_target_explanation(*, row: Mapping[str, Any], target_id: str) -> Dict[str, Any]:
    if target_id == "runtime":
        hot = _build_single_target_explanation(row=row, target_id="hot_execution_time")
        transition = _build_single_target_explanation(row=row, target_id="transition_penalty")
        compatibility = _mapping(row.get("compatibility"))
        center = _sum_or_none(_float(hot.get("center")), _float(transition.get("center")))
        upper = _sum_or_none(_float(hot.get("upper")), _float(transition.get("upper")))
        if center is None:
            center = _float(compatibility.get("runtime_p50"))
        if upper is None:
            upper = _float(compatibility.get("runtime_p90"))
        actual = _float(row.get("actual_total_runtime_sec"))
        support_level = min(
            (str(hot.get("support_level") or "none"), str(transition.get("support_level") or "none")),
            key=_support_rank,
        )
        fallback_level = max(
            (str(hot.get("fallback_level") or "none"), str(transition.get("fallback_level") or "none")),
            key=_fallback_rank,
        )
        effective_support = _min_or_none(_int(hot.get("effective_support")), _int(transition.get("effective_support")))
        evidence = _runtime_evidence(hot, transition)
        correction_delta_units = _sum_or_none(
            _float(hot.get("campaign_correction_total")),
            _float(transition.get("campaign_correction_total")),
        )
        guard_components = _runtime_guard_components(hot, transition)
        delta_vs_upper = float(actual - upper) if actual is not None and upper is not None else None
        status = _classify_status(center=center, upper=upper, actual=actual)
        abstain_reason_code = None
        abstain_reason = None
        if status == "abstained":
            abstain_reason_code, abstain_reason = _generic_abstain_reason(evidence=evidence, actual=actual)
        return {
            "target_id": "runtime",
            "title": _TARGET_TITLES["runtime"],
            "status": status,
            "summary": _summary_sentence(
                title=_TARGET_TITLES["runtime"],
                status=status,
                support_level=support_level,
                effective_support=effective_support,
                upper=upper,
                actual=actual,
                delta_vs_upper=delta_vs_upper,
                abstain_reason=abstain_reason,
                unit=_RUNTIME_UNIT,
            ),
            "unit": _RUNTIME_UNIT,
            "scope": None,
            "center": center,
            "upper": upper,
            "actual": actual,
            "guard_margin": float(upper - center) if upper is not None and center is not None else _float(compatibility.get("runtime_guard_margin_sec")),
            "delta_vs_upper": delta_vs_upper,
            "support_level": support_level,
            "fallback_level": fallback_level,
            "effective_support": effective_support,
            "evidence": evidence,
            "guard_components": guard_components,
            "campaign_correction_total": correction_delta_units,
            "abstain_reason_code": abstain_reason_code,
            "abstain_reason": abstain_reason,
            "reason_codes": _reason_codes(row),
        }
    if target_id in {"active_memory", "resident_baseline", "hot_execution_time", "transition_penalty"}:
        return _build_single_target_explanation(row=row, target_id=target_id)
    raise ValueError(f"unsupported target_id: {target_id}")


def _target_summary(explanation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "title": _text(explanation.get("title")),
        "status": _text(explanation.get("status")),
        "summary": _text(explanation.get("summary")),
        "unit": _text(explanation.get("unit")),
        "scope": _text(explanation.get("scope")) or None,
        "center": _float(explanation.get("center")),
        "upper": _float(explanation.get("upper")),
        "actual": _float(explanation.get("actual")),
        "guard_margin": _float(explanation.get("guard_margin")),
        "delta_vs_upper": _float(explanation.get("delta_vs_upper")),
        "support_level": _text(explanation.get("support_level")) or "none",
        "fallback_level": _text(explanation.get("fallback_level")) or "none",
        "effective_support": _int(explanation.get("effective_support")),
        "abstain_reason_code": _text(explanation.get("abstain_reason_code")) or None,
        "abstain_reason": _text(explanation.get("abstain_reason")) or None,
    }


def build_run_takeaway(row: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = build_target_explanation(row=row, target_id="runtime")
    active_memory = build_target_explanation(row=row, target_id="active_memory")
    notes: list[str] = []
    selected_context_applied = _text(row.get("selected_context_applied"))
    if selected_context_applied == "post_selection_retry":
        notes.append("worker was reselected before execution and the estimator was recomputed on the replacement worker")
    elif selected_context_applied == "missing_worker_context":
        notes.append("selected worker context was missing when the prediction was recorded")

    runtime_status = _text(runtime.get("status"))
    runtime_delta = _float(runtime.get("delta_vs_upper"))
    if runtime_status == "predicted_violated" and runtime_delta is not None:
        first = f"Runtime guard missed by {runtime_delta:.1f} sec."
        status = "attention"
    elif runtime_status == "predicted_ok":
        margin = abs(_float(runtime.get("delta_vs_upper")) or 0.0)
        first = f"Runtime stayed within the guard by {margin:.1f} sec."
        status = "ok"
    elif runtime_status == "telemetry_missing":
        first = "Runtime was predicted but actual telemetry is missing."
        status = "attention"
    else:
        reason = _text(runtime.get("abstain_reason")) or "the runtime estimator had no explanation"
        first = f"Runtime abstained because {reason}."
        status = "attention"

    active_status = _text(active_memory.get("status"))
    if active_status == "predicted_violated":
        delta = _float(active_memory.get("delta_vs_upper")) or 0.0
        second = f"Active memory guard missed by {delta:.1f} MiB."
        status = "attention"
    elif active_status == "predicted_ok":
        margin = abs(_float(active_memory.get("delta_vs_upper")) or 0.0)
        second = f"Active memory stayed within the guard by {margin:.1f} MiB."
    elif active_status == "telemetry_missing":
        second = "Active-memory telemetry is missing."
        status = "attention"
    else:
        reason = _text(active_memory.get("abstain_reason")) or "no explanation is available"
        second = f"Active memory abstained because {reason}."
        status = "attention"

    return {
        "headline": f"{first} {second}",
        "status": status,
        "notes": notes,
    }


def build_recent_run_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = build_target_explanation(row=row, target_id="runtime")
    active_memory = build_target_explanation(row=row, target_id="active_memory")
    resident_baseline = build_target_explanation(row=row, target_id="resident_baseline")
    takeaway = build_run_takeaway(row)
    return {
        "takeaway": _text(takeaway.get("headline")),
        "runtime": _target_summary(runtime),
        "active_memory": _target_summary(active_memory),
        "resident_baseline": _target_summary(resident_baseline),
    }


__all__ = [
    "build_recent_run_summary",
    "build_run_takeaway",
    "build_target_explanation",
]
