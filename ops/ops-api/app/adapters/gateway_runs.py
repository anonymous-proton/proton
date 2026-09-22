from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from app.services.payload_utils import (
    float_payload as _float,
    list_of_dicts as _list_of_dicts,
    mapping_payload as _mapping,
    normalize_gateway_payload,
    sum_payload as _sum,
    text_payload as _text,
)


def _worker_timings(row: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    worker_timing = _mapping(row.get("worker_timing_us"))
    execute_us = _float(worker_timing.get("execute_us"))
    prepare_us = _float(worker_timing.get("prepare_us"))
    finalize_us = _float(worker_timing.get("finalize_us"))

    hot = execute_us / 1_000_000.0 if execute_us is not None and execute_us > 0 else None
    transition_total = 0.0
    if prepare_us is not None and prepare_us > 0:
        transition_total += prepare_us
    if finalize_us is not None and finalize_us > 0:
        transition_total += finalize_us
    transition = transition_total / 1_000_000.0 if transition_total > 0 else None
    total = None
    if hot is not None and transition is not None:
        total = hot + transition
    elif hot is not None:
        total = hot
    return hot, transition, total


def normalize_run(row: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(row or {})
    dispatch = _mapping(raw.get("dispatch"))
    axes = _mapping(raw.get("axes"))
    scheduler_decision = _mapping(raw.get("scheduler_decision"))
    placement = _mapping(scheduler_decision.get("placement"))
    signal = _mapping(scheduler_decision.get("signal"))
    admission = _mapping(scheduler_decision.get("admission"))
    predicted = _mapping(scheduler_decision.get("predicted"))
    corrections = _mapping(scheduler_decision.get("corrections"))
    guards = _mapping(scheduler_decision.get("guards"))
    constraints = _mapping(scheduler_decision.get("constraints"))
    actual = _mapping(scheduler_decision.get("actual"))
    query_context = _mapping(scheduler_decision.get("query_context"))
    signal_query_context = _mapping(signal.get("query_context"))
    signal_artifacts = _mapping(signal.get("artifacts"))
    if not query_context:
        query_context = signal_query_context
    execution_context = _mapping(query_context.get("execution_envelope"))
    baseline_context = _mapping(query_context.get("replica_baseline"))
    selected_worker_context = _mapping(scheduler_decision.get("selected_worker_context"))
    if not selected_worker_context:
        selected_worker_context = _mapping(placement.get("selected_worker_context"))
    chosen_worker = _mapping(scheduler_decision.get("chosen_worker"))
    if not chosen_worker:
        chosen_worker = _mapping(placement.get("chosen_worker"))
    predicted_execution = _mapping(predicted.get("execution_envelope"))
    predicted_baseline = _mapping(predicted.get("replica_baseline"))
    if not predicted_execution:
        predicted_execution = _mapping(signal_artifacts.get("execution_envelope"))
    if not predicted_baseline:
        predicted_baseline = _mapping(signal_artifacts.get("replica_baseline"))
    compatibility = _mapping(predicted.get("compatibility"))
    if not corrections:
        corrections = _mapping(signal_artifacts.get("corrections"))
    if not guards:
        guards = _mapping(signal_artifacts.get("guards"))
    if not compatibility and signal:
        signal_runtime = _mapping(signal.get("runtime"))
        signal_memory = _mapping(signal.get("memory"))
        provenance = _mapping(signal.get("provenance"))
        active_scope = _text(_mapping(_mapping(predicted_execution.get("active_memory_mib")).get("estimate")).get("scope"))
        if not active_scope:
            active_scope = _text(signal_memory.get("memory_basis"))
        compatibility = {
            "runtime_p50": _float(signal_runtime.get("estimate_sec")),
            "runtime_p90": _float(signal_runtime.get("upper_sec")),
            "peak_mem_p90": _float(signal_memory.get("active_upper_mib")),
            "runtime_guard_margin_sec": _float(signal_runtime.get("guard_margin_sec")),
            "memory_guard_margin_mib": _float(signal_memory.get("guard_margin_mib")),
            "active_memory_scope": active_scope,
            "support_level": _text(provenance.get("runtime_support_level")),
            "fallback_level": _text(provenance.get("runtime_fallback_level")),
            "memory_support_level": _text(provenance.get("memory_support_level")),
            "memory_fallback_level": _text(provenance.get("memory_fallback_level")),
            "resident_support_level": _text(provenance.get("resident_support_level")),
            "resident_fallback_level": _text(provenance.get("resident_fallback_level")),
            "predicted_resident_memory_mib": _float(signal_memory.get("resident_estimate_mib")),
            "predicted_resident_memory_p95_mib": _float(signal_memory.get("resident_upper_mib")),
        }
    if not predicted and (compatibility or predicted_execution or predicted_baseline):
        predicted = {
            "compatibility": compatibility,
            "execution_envelope": predicted_execution,
            "replica_baseline": predicted_baseline,
        }
    if not constraints and admission:
        constraints = {
            "mem_safe_limit_mib": _float(admission.get("mem_safe_limit_mib")),
            "backfill_gap_sec": _float(admission.get("backfill_gap_sec")),
            "backfill_fit_pred": admission.get("backfill_fits"),
            "memory_composition_rule": _text(admission.get("memory_basis")),
            "memory_upper_bound_mib": _float(admission.get("memory_upper_bound_mib")),
            "memory_fit_pred": admission.get("memory_fits"),
            "batch_size_safe_cap": admission.get("batch_size_safe_cap"),
            "batch_size_applied": admission.get("batch_size_applied"),
        }
    hot_target = _mapping(predicted_execution.get("hot_execution_time_sec"))
    transition_target = _mapping(predicted_execution.get("transition_penalty_sec"))
    active_target = _mapping(predicted_execution.get("active_memory_mib"))
    resident_target = _mapping(predicted_baseline.get("resident_memory_mib"))
    runtime_support_level = _text(compatibility.get("support_level")) or _text(_mapping(hot_target.get("support")).get("level"))
    runtime_fallback_level = _text(compatibility.get("fallback_level")) or _text(_mapping(hot_target.get("support")).get("fallback_level"))
    active_support_level = _text(compatibility.get("memory_support_level")) or _text(_mapping(active_target.get("support")).get("level"))
    active_fallback_level = _text(compatibility.get("memory_fallback_level")) or _text(_mapping(active_target.get("support")).get("fallback_level"))
    resident_support_level = _text(compatibility.get("resident_support_level")) or _text(_mapping(resident_target.get("support")).get("level"))
    resident_fallback_level = _text(compatibility.get("resident_fallback_level")) or _text(_mapping(resident_target.get("support")).get("fallback_level"))
    execution_corrections = _mapping(corrections.get("execution_envelope"))
    baseline_corrections = _mapping(corrections.get("replica_baseline"))
    execution_guards = _mapping(guards.get("execution_envelope"))
    baseline_guards = _mapping(guards.get("replica_baseline"))

    actual_hot_execution_sec, actual_transition_penalty_sec, actual_total_runtime_sec = _worker_timings(raw)
    if actual_hot_execution_sec is None:
        actual_hot_execution_sec = _float(actual.get("runtime_sec"))
    if actual_total_runtime_sec is None:
        actual_total_runtime_sec = actual_hot_execution_sec

    return {
        "raw": raw,
        "run_key": _text(raw.get("run_key")),
        "campaign_id": _text(raw.get("campaign_id")),
        "component": _text(raw.get("component")).lower(),
        "state": _text(raw.get("state")).upper(),
        "gateway": normalize_gateway_payload(_mapping(raw.get("gateway"))),
        "dispatch": dispatch,
        "axes": axes,
        "error": _text(raw.get("error")),
        "created_at": _float(raw.get("created_at")),
        "started_at": _float(raw.get("started_at")),
        "finished_at": _float(raw.get("finished_at")),
        "updated_at": _float(raw.get("updated_at")),
        "worker_timing_us": _mapping(raw.get("worker_timing_us")),
        "scheduler_decision": scheduler_decision,
        "selected_context_applied": (
            _text(scheduler_decision.get("selected_context_applied"))
            or _text(placement.get("selected_context_applied"))
            or "missing_worker_context"
        ),
        "selected_worker_context": selected_worker_context,
        "chosen_worker": chosen_worker,
        "query_context": query_context,
        "execution_context": execution_context,
        "baseline_context": baseline_context,
        "predicted": predicted,
        "compatibility": compatibility,
        "constraints": constraints,
        "predicted_execution": predicted_execution,
        "predicted_baseline": predicted_baseline,
        "hot_target": hot_target,
        "transition_target": transition_target,
        "active_target": active_target,
        "resident_target": resident_target,
        "runtime_support_level": runtime_support_level or "none",
        "runtime_fallback_level": runtime_fallback_level or "none",
        "active_memory_support_level": active_support_level or "none",
        "active_memory_fallback_level": active_fallback_level or "none",
        "resident_baseline_support_level": resident_support_level or "none",
        "resident_baseline_fallback_level": resident_fallback_level or "none",
        "runtime_correction_total_log": _sum([
            _float(_mapping(execution_corrections.get("hot_execution_time_sec")).get("total")),
            _float(_mapping(execution_corrections.get("transition_penalty_sec")).get("total")),
        ]),
        "active_memory_correction_total_log": _float(_mapping(execution_corrections.get("active_memory_mib")).get("total")),
        "resident_baseline_correction_total_log": _float(_mapping(baseline_corrections.get("resident_memory_mib")).get("total")),
        "runtime_safety_guard_log": _sum([
            _float(_mapping(execution_guards.get("hot_execution_time_sec")).get("safety")),
            _float(_mapping(execution_guards.get("transition_penalty_sec")).get("safety")),
        ]),
        "active_memory_safety_guard_log": _float(_mapping(execution_guards.get("active_memory_mib")).get("safety")),
        "resident_baseline_safety_guard_log": _float(_mapping(baseline_guards.get("resident_memory_mib")).get("safety")),
        "corrections": corrections,
        "guards": guards,
        "actual": actual,
        "actual_hot_execution_sec": actual_hot_execution_sec,
        "actual_transition_penalty_sec": actual_transition_penalty_sec,
        "actual_total_runtime_sec": actual_total_runtime_sec,
        "actual_active_memory_mib": _float(actual.get("active_memory_mib")),
        "actual_total_memory_upper_bound_mib": _float(actual.get("total_upper_bound_mib")),
        "actual_resident_memory_mib": _float(actual.get("resident_memory_mib")),
        "bootstrap_memory_summary": _mapping(actual.get("bootstrap_memory_summary"))
        or _mapping(raw.get("bootstrap_memory_summary")),
        "dispatch_memory_window": _mapping(actual.get("dispatch_memory_window"))
        or _mapping(raw.get("dispatch_memory_window")),
        "resident_baseline_state": _text(actual.get("resident_baseline_state")) or _text(raw.get("resident_baseline_state")),
        "resident_baseline_source": _text(actual.get("resident_memory_source")) or _text(raw.get("resident_memory_source")),
        "resident_baseline_collected_at": _float(actual.get("resident_baseline_collected_at"))
        if actual.get("resident_baseline_collected_at") is not None
        else _float(raw.get("resident_baseline_collected_at")),
        "actual_memory_qc_keep": bool(actual.get("memory_qc_keep")) if actual.get("memory_qc_keep") is not None else bool(raw.get("memory_qc_keep")),
        "dispatch_worker_name": _text(dispatch.get("worker_name")),
        "dispatch_gpu_ids": [str(item).strip() for item in list(dispatch.get("gpu_ids") or []) if str(item).strip()],
        "job_links": _list_of_dicts(raw.get("job_links")),
    }


def normalize_runs_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    runs = [normalize_run(item) for item in list(payload.get("runs") or []) if isinstance(item, Mapping)]
    return {
        "runs": runs,
        "total": int(payload.get("total") or len(runs)),
        "limit": int(payload.get("limit") or len(runs)),
        "offset": int(payload.get("offset") or 0),
    }


def normalize_run_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return normalize_run(payload)
