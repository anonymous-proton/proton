"""Event-reactive guard inflation helpers."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

_RECENCY_DECAY = 0.82
_RUNTIME_EVENT_WEIGHTS = {
    "overrun": math.log(1.08),
    "failure": math.log(1.04),
    "oom": math.log(1.03),
}
_MEMORY_EVENT_WEIGHTS = {
    "overrun": math.log(1.10),
    "failure": math.log(1.05),
    "oom": math.log(1.14),
}
_CLEAN_DECAY_CREDIT = 0.18
_MAX_SAFETY_LOG = math.log(1.45)


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or math.isinf(parsed):
        return None
    return float(parsed)


def _row_text(row: Any, field: str) -> str:
    return str(getattr(row, field, "") or "").strip()


def _row_timestamp(row: Any) -> float:
    for field in ("finished_at", "updated_at", "started_at", "created_at"):
        parsed = _to_float(getattr(row, field, None))
        if parsed is not None:
            return float(parsed)
    return 0.0


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _predicted_target_upper(row: Any, *, family_key: str, target_key: str, compatibility_key: str) -> Optional[float]:
    scheduler_decision = _mapping(getattr(row, "scheduler_decision", None))
    signal_payload = _mapping(scheduler_decision.get("signal"))
    artifacts = _mapping(signal_payload.get("artifacts"))
    family = _mapping(artifacts.get(family_key))
    if not family:
        predicted = _mapping(scheduler_decision.get("predicted"))
        family = _mapping(predicted.get(family_key))
    target = _mapping(family.get(target_key))
    estimate = _mapping(target.get("estimate"))
    upper = _to_float(estimate.get("upper"))
    if upper is not None:
        return upper
    if not compatibility_key:
        return None
    runtime = _mapping(signal_payload.get("runtime"))
    memory = _mapping(signal_payload.get("memory"))
    if compatibility_key == "runtime_p90":
        upper = _to_float(runtime.get("upper_sec"))
    elif compatibility_key == "peak_mem_p90":
        upper = _to_float(memory.get("active_upper_mib"))
    elif compatibility_key == "resident_memory_p95_mib":
        upper = _to_float(memory.get("resident_upper_mib"))
    else:
        upper = None
    if upper is not None:
        return upper
    predicted = _mapping(scheduler_decision.get("predicted"))
    compatibility = _mapping(predicted.get("compatibility"))
    return _to_float(compatibility.get(compatibility_key))


def _actual_hot_execution_sec(row: Any) -> Optional[float]:
    return _to_float(getattr(row, "runtime_sec", None))


def _actual_transition_penalty_sec(row: Any) -> Optional[float]:
    worker_timing = _mapping(getattr(row, "worker_timing_us", None))
    prepare_us = _to_float(worker_timing.get("prepare_us"))
    finalize_us = _to_float(worker_timing.get("finalize_us"))
    total_us = float((prepare_us or 0.0) + (finalize_us or 0.0))
    if total_us <= 0:
        return None
    return total_us / 1_000_000.0


def _actual_active_memory_mib(row: Any) -> Optional[float]:
    value = getattr(row, "active_vram_mib", None)
    if value is not None:
        if not bool(getattr(row, "vram_memory_qc_keep", False)):
            return None
        return _to_float(value)
    if not bool(getattr(row, "memory_qc_keep", False)):
        return None
    return _to_float(getattr(row, "active_memory_mib", None))


def _actual_resident_memory_mib(row: Any) -> Optional[float]:
    return _to_float(getattr(row, "resident_memory_mib", None))


def _is_failure(row: Any) -> bool:
    return _row_text(row, "state").upper() not in {"", "SUCCEEDED"}


def _is_oom_like(row: Any) -> bool:
    scheduler_decision = _mapping(getattr(row, "scheduler_decision", None))
    actual = _mapping(scheduler_decision.get("actual"))
    message = " ".join(
        value
        for value in (
            _row_text(row, "error"),
            str(actual.get("failure_outcome") or "").strip(),
        )
        if value
    ).lower()
    return "oom" in message or "out of memory" in message or "cuda error out of memory" in message


def is_oom_like_failure(row: Any) -> bool:
    return _is_oom_like(row)


def estimate_safety_inflation(
    *,
    rows: Sequence[Any],
    target_kind: str,
    actual_getter: Callable[[Any], Optional[float]],
    upper_getter: Callable[[Any], Optional[float]],
) -> Dict[str, Any]:
    ordered_rows = sorted(list(rows), key=_row_timestamp, reverse=True)
    if not ordered_rows:
        return {
            "log_increment": 0.0,
            "overrun_count": 0,
            "failure_count": 0,
            "oom_count": 0,
            "clean_count": 0,
        }
    weights = _MEMORY_EVENT_WEIGHTS if str(target_kind or "").strip().lower() == "memory" else _RUNTIME_EVENT_WEIGHTS
    score = 0.0
    clean_credit = 0.0
    overrun_count = 0
    failure_count = 0
    oom_count = 0
    clean_count = 0
    for idx, row in enumerate(ordered_rows):
        weight = _RECENCY_DECAY**idx
        actual = actual_getter(row)
        upper = upper_getter(row)
        row_score = 0.0
        if actual is not None and upper is not None and actual > upper:
            row_score += weight * weights["overrun"]
            overrun_count += 1
        failure = _is_failure(row)
        oom_like = _is_oom_like(row)
        if failure:
            row_score += weight * weights["failure"]
            failure_count += 1
        if oom_like:
            row_score += weight * weights["oom"]
            oom_count += 1
        if row_score > 0:
            score += row_score
            continue
        if actual is not None and upper is not None:
            clean_credit += weight
            clean_count += 1
    score = max(0.0, score - (_CLEAN_DECAY_CREDIT * clean_credit))
    return {
        "log_increment": float(min(_MAX_SAFETY_LOG, score)),
        "overrun_count": int(overrun_count),
        "failure_count": int(failure_count),
        "oom_count": int(oom_count),
        "clean_count": int(clean_count),
    }


def runtime_hot_upper(row: Any) -> Optional[float]:
    return _predicted_target_upper(
        row,
        family_key="execution_envelope",
        target_key="hot_execution_time_sec",
        compatibility_key="runtime_p90",
    )


def runtime_transition_upper(row: Any) -> Optional[float]:
    return _predicted_target_upper(
        row,
        family_key="execution_envelope",
        target_key="transition_penalty_sec",
        compatibility_key="",
    )


def active_memory_upper(row: Any) -> Optional[float]:
    return _predicted_target_upper(
        row,
        family_key="execution_envelope",
        target_key="active_memory_mib",
        compatibility_key="peak_mem_p90",
    )


def resident_memory_upper(row: Any) -> Optional[float]:
    return _predicted_target_upper(
        row,
        family_key="replica_baseline",
        target_key="resident_memory_mib",
        compatibility_key="resident_memory_p95_mib",
    )


__all__ = [
    "active_memory_upper",
    "estimate_safety_inflation",
    "is_oom_like_failure",
    "resident_memory_upper",
    "runtime_hot_upper",
    "runtime_transition_upper",
    "_actual_active_memory_mib",
    "_actual_hot_execution_sec",
    "_actual_resident_memory_mib",
    "_actual_transition_penalty_sec",
]
