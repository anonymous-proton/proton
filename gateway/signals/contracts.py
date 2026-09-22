from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class PlannerIntent:
    mode: str = "default"
    target_worker_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode or "default"),
            "target_worker_names": [
                str(item) for item in self.target_worker_names if str(item)
            ],
            "tags": [str(item) for item in self.tags if str(item)],
        }


@dataclass(frozen=True)
class SignalProvenance:
    runtime_support_level: str = "none"
    runtime_fallback_level: str = "none"
    memory_support_level: str = "none"
    memory_fallback_level: str = "none"
    resident_support_level: str = "none"
    resident_fallback_level: str = "none"
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runtime_support_level": str(self.runtime_support_level or "none"),
            "runtime_fallback_level": str(self.runtime_fallback_level or "none"),
            "memory_support_level": str(self.memory_support_level or "none"),
            "memory_fallback_level": str(self.memory_fallback_level or "none"),
            "resident_support_level": str(self.resident_support_level or "none"),
            "resident_fallback_level": str(self.resident_fallback_level or "none"),
            "reasons": [str(item) for item in self.reasons if str(item)],
        }


@dataclass(frozen=True)
class RuntimeSignal:
    estimate_sec: Optional[float]
    upper_sec: Optional[float]
    guard_margin_sec: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "estimate_sec": self.estimate_sec,
            "upper_sec": self.upper_sec,
            "guard_margin_sec": self.guard_margin_sec,
        }


@dataclass(frozen=True)
class MemorySignal:
    active_estimate_mib: Optional[float]
    active_upper_mib: Optional[float]
    resident_estimate_mib: Optional[float]
    resident_upper_mib: Optional[float]
    total_upper_bound_mib: Optional[float]
    memory_basis: str
    peak_fidelity: str
    guard_margin_mib: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active_estimate_mib": self.active_estimate_mib,
            "active_upper_mib": self.active_upper_mib,
            "resident_estimate_mib": self.resident_estimate_mib,
            "resident_upper_mib": self.resident_upper_mib,
            "total_upper_bound_mib": self.total_upper_bound_mib,
            "memory_basis": str(self.memory_basis or ""),
            "peak_fidelity": str(self.peak_fidelity or ""),
            "guard_margin_mib": self.guard_margin_mib,
        }


@dataclass(frozen=True)
class SignalArtifacts:
    query_context: Dict[str, Any] = field(default_factory=dict)
    execution_envelope: Dict[str, Any] = field(default_factory=dict)
    replica_baseline: Dict[str, Any] = field(default_factory=dict)
    corrections: Dict[str, Any] = field(default_factory=dict)
    guards: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_context": dict(self.query_context),
            "execution_envelope": dict(self.execution_envelope),
            "replica_baseline": dict(self.replica_baseline),
            "corrections": dict(self.corrections),
            "guards": dict(self.guards),
        }


@dataclass(frozen=True)
class IntrinsicSignalSummary:
    runtime_estimate_sec: Optional[float]
    runtime_upper_sec: Optional[float]
    active_memory_upper_mib: Optional[float]
    memory_basis: str
    runtime_support_level: str
    runtime_fallback_level: str
    planner_intent: PlannerIntent = field(default_factory=PlannerIntent)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runtime_estimate_sec": self.runtime_estimate_sec,
            "runtime_upper_sec": self.runtime_upper_sec,
            "active_memory_upper_mib": self.active_memory_upper_mib,
            "memory_basis": str(self.memory_basis or ""),
            "runtime_support_level": str(self.runtime_support_level or "none"),
            "runtime_fallback_level": str(self.runtime_fallback_level or "none"),
            "planner_intent": self.planner_intent.as_dict(),
        }


@dataclass(frozen=True)
class SignalBundle:
    runtime: RuntimeSignal
    memory: MemorySignal
    provenance: SignalProvenance
    artifacts: SignalArtifacts
    planner_intent: PlannerIntent = field(default_factory=PlannerIntent)

    def intrinsic_summary(self) -> IntrinsicSignalSummary:
        return IntrinsicSignalSummary(
            runtime_estimate_sec=self.runtime.estimate_sec,
            runtime_upper_sec=self.runtime.upper_sec,
            active_memory_upper_mib=self.memory.active_upper_mib,
            memory_basis=self.memory.memory_basis,
            runtime_support_level=self.provenance.runtime_support_level,
            runtime_fallback_level=self.provenance.runtime_fallback_level,
            planner_intent=self.planner_intent,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime.as_dict(),
            "memory": self.memory.as_dict(),
            "provenance": self.provenance.as_dict(),
            "query_context": dict(self.artifacts.query_context),
            "artifacts": self.artifacts.as_dict(),
            "planner_intent": self.planner_intent.as_dict(),
        }


@dataclass(frozen=True)
class SignalResult:
    bundle: SignalBundle
    worker_context_applied: bool
    worker_context: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = self.bundle.as_dict()
        payload["worker_context_applied"] = bool(self.worker_context_applied)
        payload["worker_context"] = dict(self.worker_context)
        return payload


InterferenceProvenance = Literal[
    "exact",
    "nearby",
    "cross-config",
    "class-prior",
    "fixed-prior",
    "unknown",
]


def slowdown_factor_to_delta(factor: float) -> float:
    """Canonical conversion from a slowdown factor to additive delta."""
    if factor < 1.0 or factor in (math.inf, -math.inf) or factor != factor:
        raise ValueError("slowdown_factor must be finite and >= 1")
    return factor - 1.0


@dataclass(frozen=True)
class ReciprocalInterferenceQuery:
    """Fully keyed directed self/pair interference lookup."""

    victim_component: str
    victim_config_fingerprint: str = ""
    victim_input_fingerprint: str = ""
    interferer_component: str = ""
    interferer_config_fingerprint: str = ""
    interferer_input_fingerprint: str = ""
    gpu_id: str = ""
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""
    self_n: int = 1
    pair_multiplicity: int = 1

    def __post_init__(self) -> None:
        if not self.victim_component:
            raise ValueError("victim_component is required")
        if self.self_n < 1 or self.pair_multiplicity < 1:
            raise ValueError("interference multiplicity must be >= 1")

    @property
    def is_self(self) -> bool:
        return not self.interferer_component or (
            self.interferer_component == self.victim_component
        )

    @property
    def has_exact_identity(self) -> bool:
        fields = (
            self.victim_config_fingerprint,
            self.victim_input_fingerprint,
            self.gpu_id,
            self.gpu_model,
            self.mps_mode,
            self.worker_backend,
            self.actor_model,
            self.adapter_version,
        )
        if not self.is_self:
            fields += (
                self.interferer_config_fingerprint,
                self.interferer_input_fingerprint,
            )
        return all(fields)

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ReciprocalInterferenceResult:
    """One factor-valued estimate with explicit support and provenance."""

    slowdown_factor: float
    uncertainty: float
    support: int
    provenance: InterferenceProvenance
    age_sec: float
    evidence_epoch: int

    def __post_init__(self) -> None:
        slowdown_factor_to_delta(self.slowdown_factor)
        if self.uncertainty < 0.0 or self.support < 0 or self.age_sec < 0.0:
            raise ValueError("uncertainty, support, and age must be non-negative")

    @property
    def delta(self) -> float:
        return slowdown_factor_to_delta(self.slowdown_factor)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "slowdown_factor": self.slowdown_factor,
            "delta": self.delta,
            "uncertainty": self.uncertainty,
            "support": self.support,
            "provenance": self.provenance,
            "age_sec": self.age_sec,
            "evidence_epoch": self.evidence_epoch,
        }


@dataclass(frozen=True)
class ProfileDriftEvent:
    """Emitted when a signal measurement deviates significantly from its established baseline.

    Covers three signal dimensions, each with its own detection mechanism:

    ``metric`` values and their sources:

    - ``"vram"`` — ``ResourceProfileRegistry`` cross-GPU pool baseline deviates ≥ 30%
      (requires ≥ 5 baseline samples).  ``config_fingerprint`` is populated.
    - ``"latency"`` — ``ResourceProfileRegistry`` cross-GPU pool baseline deviates ≥ 30%
      (requires ≥ 5 baseline samples).  ``config_fingerprint`` is populated.
    - ``"interference_latency"`` — ``InterferenceRegistry`` solo latency baseline regime
      change detected (> 50% deviation → flush).  ``config_fingerprint`` is ``""``.
    - ``"interference_vram"`` — ``InterferenceRegistry`` solo VRAM baseline regime
      change detected (> 50% deviation → flush).  ``config_fingerprint`` is ``""``.

    For ``"interference_*"`` metrics, ``n_baseline`` is the number of observations
    flushed from the solo baseline (not a Welford sample count).

    The Planner registers a callback via :py:meth:`SignalService.register_drift_callback`
    to receive these events and can use them to invalidate cached scheduling
    scenarios that relied on the stale baseline.
    """

    component: str
    config_fingerprint: str
    gpu_id: Optional[
        str
    ]
    metric: str
    observed: float
    predicted: float
    drift_ratio: float
    n_baseline: int
    input_size: float
    campaign_id: str
    timestamp: float
    primary_tail_reference_qualified: bool = False

    @classmethod
    def make(
        cls,
        *,
        component: str,
        config_fingerprint: str,
        gpu_id: Optional[str],
        metric: str,
        observed: float,
        predicted: float,
        drift_ratio: float,
        n_baseline: int,
        input_size: float = 0.0,
        campaign_id: str = "",
        primary_tail_reference_qualified: bool = False,
    ) -> "ProfileDriftEvent":
        return cls(
            component=component,
            config_fingerprint=config_fingerprint,
            gpu_id=gpu_id,
            metric=metric,
            observed=observed,
            predicted=predicted,
            drift_ratio=drift_ratio,
            n_baseline=n_baseline,
            input_size=input_size,
            campaign_id=campaign_id,
            timestamp=_time.time(),
            primary_tail_reference_qualified=primary_tail_reference_qualified,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "config_fingerprint": self.config_fingerprint,
            "gpu_id": self.gpu_id,
            "metric": self.metric,
            "observed": round(self.observed, 3),
            "predicted": round(self.predicted, 3),
            "drift_ratio": round(self.drift_ratio, 4),
            "n_baseline": self.n_baseline,
            "input_size": round(self.input_size, 1),
            "campaign_id": self.campaign_id,
            "timestamp": self.timestamp,
        }




@dataclass
class WorkerMemoryProfile:
    """Memory footprint projections for a candidate request.

    Populated from SignalService's historical profiling data.
    """

    required_vram_mb: float = 0.0
    expected_peak_vram_mb: float = 0.0
    safe_limit_vram_mb: float = 0.0
    batch_size_safe_cap: Optional[int] = None


@dataclass
class WorkerRuntimeProfile:
    """Execution time projections for a candidate request.

    Populated from SignalService's historical profiling data.
    """

    expected_duration_sec: float = 0.0
    upper_bound_duration_sec: float = 0.0


@dataclass
class WorkerConcurrencyProfile:
    """GPU sharing / interference profile for a request type.

    When multiple models share one GPU, memory-bound and compute-bound workloads
    can interfere differently (DRAM bandwidth contention vs. SM occupancy
    conflicts). This struct records observed interference behavior so the Planner
    can make bin-packing decisions that minimize throughput degradation.

    Fields are populated by the InterferenceRegistry in the Signal module from
    observed co-location latency data and workload classification.
    """

    can_overlap_execute: bool = False

    interference_vram_overhead_ratio: float = 0.0

    interference_compute_bound_slowdown: float = 0.0

    interference_memory_bound_slowdown: float = 0.0

    workload_class: str = "unknown"


    predicted_slowdown: float = 0.0

    prediction_basis: str = "none"

    prediction_confidence: str = "low"

    solo_baseline_sec: Optional[float] = None


@dataclass
class WorkerComputeSignal:
    """Extensible struct passed from the Signal module to the Pluggable Planner.

    Wraps per-request resource projections. New fields should be added to the
    appropriate sub-profile rather than directly here so that the Signal module
    owns the boundary cleanly and the Planner interface stays stable.
    """

    memory_profile: WorkerMemoryProfile = field(default_factory=WorkerMemoryProfile)
    runtime_profile: WorkerRuntimeProfile = field(default_factory=WorkerRuntimeProfile)
    concurrency_profile: WorkerConcurrencyProfile = field(
        default_factory=WorkerConcurrencyProfile
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "memory_profile": {
                "required_vram_mb": self.memory_profile.required_vram_mb,
                "expected_peak_vram_mb": self.memory_profile.expected_peak_vram_mb,
                "safe_limit_vram_mb": self.memory_profile.safe_limit_vram_mb,
                "batch_size_safe_cap": self.memory_profile.batch_size_safe_cap,
            },
            "runtime_profile": {
                "expected_duration_sec": self.runtime_profile.expected_duration_sec,
                "upper_bound_duration_sec": self.runtime_profile.upper_bound_duration_sec,
            },
            "concurrency_profile": {
                "can_overlap_execute": self.concurrency_profile.can_overlap_execute,
                "interference_vram_overhead_ratio": self.concurrency_profile.interference_vram_overhead_ratio,
                "interference_compute_bound_slowdown": self.concurrency_profile.interference_compute_bound_slowdown,
                "interference_memory_bound_slowdown": self.concurrency_profile.interference_memory_bound_slowdown,
                "workload_class": self.concurrency_profile.workload_class,
                "predicted_slowdown": self.concurrency_profile.predicted_slowdown,
                "prediction_basis": self.concurrency_profile.prediction_basis,
                "prediction_confidence": self.concurrency_profile.prediction_confidence,
                "solo_baseline_sec": self.concurrency_profile.solo_baseline_sec,
            },
        }


__all__ = [
    "IntrinsicSignalSummary",
    "MemorySignal",
    "PlannerIntent",
    "ProfileDriftEvent",
    "ReciprocalInterferenceQuery",
    "ReciprocalInterferenceResult",
    "RuntimeSignal",
    "SignalArtifacts",
    "SignalBundle",
    "SignalProvenance",
    "SignalResult",
    "WorkerComputeSignal",
    "WorkerConcurrencyProfile",
    "WorkerMemoryProfile",
    "WorkerRuntimeProfile",
    "slowdown_factor_to_delta",
]
