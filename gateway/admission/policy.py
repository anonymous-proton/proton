from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..signals.contracts import PlannerIntent, SignalBundle

_DEFAULT_MEM_SAFE_LIMIT_MIB = 0.9 * 24576.0

from ..gpu_capacity import default_mem_safe_limit_mib as _dmsl

_DEFAULT_MEM_SAFE_LIMIT_MIB = _dmsl()


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return float(parsed)


@dataclass(frozen=True)
class AdmissionDecision:
    requested_batch_size: int
    mem_safe_limit_mib: Optional[float]
    backfill_gap_sec: Optional[float]
    backfill_fits: Optional[bool]
    resident_memory_mib: Optional[float]
    active_memory_upper_mib: Optional[float]
    memory_basis: str
    peak_fidelity: str
    memory_upper_bound_mib: Optional[float]
    memory_fits: Optional[bool]
    batch_size_safe_cap: Optional[int]
    batch_size_applied: Optional[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_batch_size": int(self.requested_batch_size),
            "mem_safe_limit_mib": self.mem_safe_limit_mib,
            "backfill_gap_sec": self.backfill_gap_sec,
            "backfill_fits": self.backfill_fits,
            "resident_memory_mib": self.resident_memory_mib,
            "active_memory_upper_mib": self.active_memory_upper_mib,
            "memory_basis": str(self.memory_basis or ""),
            "peak_fidelity": str(self.peak_fidelity or ""),
            "memory_upper_bound_mib": self.memory_upper_bound_mib,
            "memory_fits": self.memory_fits,
            "batch_size_safe_cap": self.batch_size_safe_cap,
            "batch_size_applied": self.batch_size_applied,
        }


def evaluate_admission(
    *,
    signal: SignalBundle,
    workload_features: Optional[Mapping[str, Any]],
    execution_overrides: Optional[Mapping[str, Any]],
    resident_baseline_mib: Optional[float] = None,
    default_mem_safe_limit_mib: float = _DEFAULT_MEM_SAFE_LIMIT_MIB,
    planner_intent: Optional[PlannerIntent] = None,
    current_activation_mib: float = 0.0,
) -> AdmissionDecision:
    _ = planner_intent or PlannerIntent()
    workload = (
        dict(workload_features or {}) if isinstance(workload_features, Mapping) else {}
    )
    overrides = (
        dict(execution_overrides or {})
        if isinstance(execution_overrides, Mapping)
        else {}
    )
    mem_safe_limit_mib = _to_float(workload.get("mem_safe_limit_mib"))
    if mem_safe_limit_mib is None or mem_safe_limit_mib <= 0:
        mem_safe_limit_mib = float(default_mem_safe_limit_mib)
    backfill_gap_sec = _to_float(workload.get("backfill_gap_sec"))
    if backfill_gap_sec is not None and backfill_gap_sec <= 0:
        backfill_gap_sec = None
    runtime_upper_sec = signal.runtime.upper_sec
    backfill_fits = None
    if backfill_gap_sec is not None and runtime_upper_sec is not None:
        backfill_fits = bool(runtime_upper_sec <= backfill_gap_sec)
    requested_batch_size = _to_int(overrides.get("batch_size"))
    if requested_batch_size is None or requested_batch_size <= 0:
        requested_batch_size = _to_int(workload.get("input_batch_size"))
    requested_batch_size = max(1, requested_batch_size or 1)
    resident_memory = resident_baseline_mib
    if resident_memory is None:
        resident_memory = signal.memory.resident_upper_mib
    if resident_memory is None:
        resident_memory = signal.memory.resident_estimate_mib
    active_memory_upper = signal.memory.active_upper_mib
    memory_upper_bound_mib = signal.memory.total_upper_bound_mib
    if memory_upper_bound_mib is None:
        if signal.memory.memory_basis == "increment_over_resident":
            if resident_memory is not None and active_memory_upper is not None:
                memory_upper_bound_mib = float(resident_memory + active_memory_upper)
        elif active_memory_upper is not None:
            memory_upper_bound_mib = float(active_memory_upper)
    memory_fits = None
    if (
        memory_upper_bound_mib is not None
        and mem_safe_limit_mib is not None
        and mem_safe_limit_mib > 0
    ):
        memory_fits = bool(memory_upper_bound_mib <= mem_safe_limit_mib)
    batch_size_safe_cap = None
    if (
        active_memory_upper is not None
        and active_memory_upper > 0
        and mem_safe_limit_mib is not None
        and mem_safe_limit_mib > 0
    ):
        available_for_active = float(mem_safe_limit_mib)
        if signal.memory.memory_basis == "increment_over_resident":
            available_for_active = max(
                0.0, float(mem_safe_limit_mib) - float(resident_memory or 0.0)
            )
        available_for_active = max(
            0.0, available_for_active - float(current_activation_mib)
        )
        batch_size_safe_cap = int(
            max(
                1,
                int(
                    (float(requested_batch_size) * available_for_active)
                    / float(active_memory_upper)
                ),
            )
        )
    batch_size_applied = None
    override_batch_size = _to_int(overrides.get("batch_size"))
    if override_batch_size is not None and override_batch_size > 0:
        if batch_size_safe_cap is not None:
            batch_size_applied = int(
                max(1, min(int(override_batch_size), int(batch_size_safe_cap)))
            )
        else:
            batch_size_applied = int(override_batch_size)
    return AdmissionDecision(
        requested_batch_size=int(requested_batch_size),
        mem_safe_limit_mib=mem_safe_limit_mib,
        backfill_gap_sec=backfill_gap_sec,
        backfill_fits=backfill_fits,
        resident_memory_mib=_to_float(resident_memory),
        active_memory_upper_mib=_to_float(active_memory_upper),
        memory_basis=str(signal.memory.memory_basis or ""),
        peak_fidelity=str(signal.memory.peak_fidelity or ""),
        memory_upper_bound_mib=_to_float(memory_upper_bound_mib),
        memory_fits=memory_fits,
        batch_size_safe_cap=batch_size_safe_cap,
        batch_size_applied=batch_size_applied,
    )


__all__ = ["AdmissionDecision", "evaluate_admission"]
