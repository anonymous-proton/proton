"""Interference modeling for co-located GPU workloads.

When multiple models share a GPU, they contend for shared resources (SM cores,
memory bandwidth, L2 cache).  The degree of contention depends on each model's
**workload class**:

- **compute_bound**: Dominated by SM utilization (e.g. many FLOPS, low memory
  traffic).  Two compute-bound models compete for SM warps → high slowdown.
- **memory_bound**: Dominated by DRAM bandwidth (e.g. large embedding lookups,
  attention with long sequences).  Two memory-bound models saturate HBM → high
  slowdown.
- **balanced**: Significant use of both compute and memory bandwidth.

Interference between orthogonal classes (compute + memory) is lower because
they stress different hardware units.

This module provides:

1. ``WorkloadClassifier`` — infers workload class from profiling history
   (GPU utilization, arithmetic intensity proxy).
2. ``InterferenceRegistry`` — tracks pairwise correction factors (latency,
   VRAM, GPU util) from observed data with temporal decay and regime change
   detection.
"""

from __future__ import annotations

import contextlib
import logging
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from .contracts import (
    InterferenceProvenance,
    ReciprocalInterferenceQuery,
    ReciprocalInterferenceResult,
    slowdown_factor_to_delta,
)
from .latency_tracker import LatencyObservation

_LOG = logging.getLogger(__name__)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _query_payload(query: ReciprocalInterferenceQuery | None) -> dict[str, Any]:
    return query.as_dict() if query is not None else {}


def _as_provenance(value: Any) -> InterferenceProvenance:
    text = str(value or "unknown")
    if text == "exact":
        return "exact"
    if text == "nearby":
        return "nearby"
    if text == "cross-config":
        return "cross-config"
    if text == "class-prior":
        return "class-prior"
    if text == "fixed-prior":
        return "fixed-prior"
    return "unknown"


def _query_from_payload(payload: Any) -> ReciprocalInterferenceQuery | None:
    if not isinstance(payload, Mapping) or not payload:
        return None
    allowed = ReciprocalInterferenceQuery.__dataclass_fields__
    try:
        return ReciprocalInterferenceQuery(
            **{key: value for key, value in payload.items() if key in allowed}
        )
    except (TypeError, ValueError):
        return None


@contextlib.contextmanager
def _writable_text(path: str):
    import os

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yield handle
    except OSError as exc:
        raise RuntimeError(f"cannot write interference history: {path}") from exc


class _ObservedAt(Protocol):
    observed_at: float


_ObservedT = TypeVar("_ObservedT", bound=_ObservedAt)


COMPUTE_BOUND = "compute_bound"
MEMORY_BOUND = "memory_bound"
BALANCED = "balanced"
UNKNOWN = "unknown"


@dataclass
class _SelfSlowdownObs:
    """Plan — N-stratified self-slowdown
    observation indexed by ``N_concurrency`` only.

    Phase A reference normalization (input-aware GP latency at the
    observation's input_size) factors out the input-length dimension
    at *recording time*, so the stored ``delta`` is approximately
    input-length-independent.  Storage therefore drops the
    ``input_size`` axis entirely; the estimate reads the arithmetic
    mean of deltas at the exact queried N (naive mean-only policy,
    2026-08 — no cross-N kernel smoothing).

    GP input: ``x = N_concurrency``; output: slowdown delta
    (``actual / reference - 1``).
    """

    n_concurrency: int
    delta: float
    observed_at: float
    gpu_id: str = ""
    query: ReciprocalInterferenceQuery | None = None
    uncertainty: float = 0.0
    provenance: InterferenceProvenance = "unknown"


@dataclass
class WorkloadProfile:
    """Aggregated profiling metrics for one component used for classification."""

    component: str
    mean_gpu_util_percent: float | None = None
    mean_execute_us: float | None = None
    mean_active_memory_mib: float | None = None
    sample_count: int = 0


_DEFAULT_COMPUTE_BOUND_GPU_UTIL = 65.0
_DEFAULT_MEMORY_BOUND_GPU_UTIL = 35.0
_DEFAULT_COMPUTE_BOUND_INTENSITY = 500.0
_DEFAULT_MEMORY_BOUND_INTENSITY = 100.0


class WorkloadClassifier:
    """Infers workload class (compute/memory/balanced) from profiling data.

    Two signals are used, in priority order:

    1. ``mean_gpu_util_percent`` — directly indicates SM saturation.
    2. **Arithmetic intensity proxy** — ``execute_us / active_memory_mib``.

    Uses EMA (α=0.3) to adapt to changing workload patterns.
    Only classifies after ≥2 samples to avoid single-observation noise.
    """

    def __init__(
        self,
        *,
        compute_bound_gpu_util: float = _DEFAULT_COMPUTE_BOUND_GPU_UTIL,
        memory_bound_gpu_util: float = _DEFAULT_MEMORY_BOUND_GPU_UTIL,
        compute_bound_intensity: float = _DEFAULT_COMPUTE_BOUND_INTENSITY,
        memory_bound_intensity: float = _DEFAULT_MEMORY_BOUND_INTENSITY,
    ) -> None:
        self._compute_gpu = _as_float(compute_bound_gpu_util)
        self._memory_gpu = _as_float(memory_bound_gpu_util)
        self._compute_intensity = _as_float(compute_bound_intensity)
        self._memory_intensity = _as_float(memory_bound_intensity)
        self._cache: dict[str, str] = {}
        self._profiles: dict[str, WorkloadProfile] = {}

    def update_profile(
        self,
        component: str,
        *,
        gpu_util_percent: float | None = None,
        execute_us: float | None = None,
        active_memory_mib: float | None = None,
        is_solo: bool = True,
        concurrent_task_count: int = 1,
    ) -> str:
        """Update profiling data for a component and re-classify.

        Both solo and concurrent observations are accepted.  Concurrent GPU
        utilization is normalized by task count (aggregate / N) since
        nvidia-smi reports device-wide utilization.  ``execute_us`` and
        ``active_memory_mib`` come from the per-task worker engine, so they
        are already per-task and need no normalization.  Concurrent observations
        are weighted lower (α=0.15 vs α=0.30) to reduce noise from the
        normalization approximation.
        """
        if not is_solo and concurrent_task_count > 1 and gpu_util_percent is not None:
            gpu_util_percent = gpu_util_percent / concurrent_task_count

        alpha = 0.3 if is_solo else 0.15
        profile = self._profiles.get(component)
        if profile is None:
            profile = WorkloadProfile(component=component)
            self._profiles[component] = profile

        if gpu_util_percent is not None:
            if profile.mean_gpu_util_percent is None:
                profile.mean_gpu_util_percent = gpu_util_percent
            else:
                profile.mean_gpu_util_percent = (
                    alpha * gpu_util_percent
                    + (1 - alpha) * profile.mean_gpu_util_percent
                )
        if execute_us is not None:
            if profile.mean_execute_us is None:
                profile.mean_execute_us = execute_us
            else:
                profile.mean_execute_us = (
                    alpha * execute_us + (1 - alpha) * profile.mean_execute_us
                )
        if active_memory_mib is not None:
            if profile.mean_active_memory_mib is None:
                profile.mean_active_memory_mib = active_memory_mib
            else:
                profile.mean_active_memory_mib = (
                    alpha * active_memory_mib
                    + (1 - alpha) * profile.mean_active_memory_mib
                )
        profile.sample_count += 1

        wclass = self._classify(profile)
        if wclass != self._cache.get(component):
            _LOG.info(
                "[workload-classifier] %s classified as %s (gpu_util=%.1f%%, intensity=%.1f us/MiB, n=%d)",
                component,
                wclass,
                profile.mean_gpu_util_percent or 0.0,
                (
                    (profile.mean_execute_us / profile.mean_active_memory_mib)
                    if profile.mean_execute_us and profile.mean_active_memory_mib
                    else 0.0
                ),
                profile.sample_count,
            )
        self._cache[component] = wclass
        return wclass

    def classify(self, component: str) -> str:
        return self._cache.get(component, UNKNOWN)

    def _classify(self, profile: WorkloadProfile) -> str:
        if profile.mean_gpu_util_percent is not None and profile.sample_count >= 2:
            util = profile.mean_gpu_util_percent
            if util >= self._compute_gpu:
                return COMPUTE_BOUND
            if util <= self._memory_gpu:
                return MEMORY_BOUND
            return BALANCED
        if (
            profile.mean_execute_us is not None
            and profile.mean_active_memory_mib is not None
            and profile.mean_active_memory_mib > 0
            and profile.sample_count >= 2
        ):
            intensity = profile.mean_execute_us / profile.mean_active_memory_mib
            if intensity >= self._compute_intensity:
                return COMPUTE_BOUND
            if intensity <= self._memory_intensity:
                return MEMORY_BOUND
            return BALANCED
        return UNKNOWN

    def get_profile(self, component: str) -> WorkloadProfile | None:
        return self._profiles.get(component)

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "workload_classifier_v1",
            "thresholds": {
                "compute_bound_gpu_util": self._compute_gpu,
                "memory_bound_gpu_util": self._memory_gpu,
                "compute_bound_intensity": self._compute_intensity,
                "memory_bound_intensity": self._memory_intensity,
            },
            "cache": dict(self._cache),
            "profiles": {
                comp: {
                    "component": profile.component,
                    "mean_gpu_util_percent": profile.mean_gpu_util_percent,
                    "mean_execute_us": profile.mean_execute_us,
                    "mean_active_memory_mib": profile.mean_active_memory_mib,
                    "sample_count": _as_int(profile.sample_count),
                }
                for comp, profile in sorted(self._profiles.items())
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        thresholds = data.get("thresholds") or {}
        if isinstance(thresholds, Mapping):
            self._compute_gpu = _as_float(
                thresholds.get("compute_bound_gpu_util", self._compute_gpu)
                or self._compute_gpu
            )
            self._memory_gpu = _as_float(
                thresholds.get("memory_bound_gpu_util", self._memory_gpu)
                or self._memory_gpu
            )
            self._compute_intensity = _as_float(
                thresholds.get("compute_bound_intensity", self._compute_intensity)
                or self._compute_intensity
            )
            self._memory_intensity = _as_float(
                thresholds.get("memory_bound_intensity", self._memory_intensity)
                or self._memory_intensity
            )
        self._cache = (
            {str(k): str(v) for k, v in (data.get("cache") or {}).items() if str(k)}
            if isinstance(data.get("cache"), Mapping)
            else {}
        )
        self._profiles = {}
        profiles = data.get("profiles") or {}
        for comp, payload in profiles.items() if isinstance(profiles, Mapping) else []:
            if not isinstance(payload, Mapping):
                continue
            component = str(payload.get("component", comp) or comp)
            profile = WorkloadProfile(component=component)
            for field_name in (
                "mean_gpu_util_percent",
                "mean_execute_us",
                "mean_active_memory_mib",
            ):
                value = payload.get(field_name)
                if value is not None:
                    with contextlib.suppress(Exception):
                        setattr(profile, field_name, _as_float(value))
            profile.sample_count = _as_int(payload.get("sample_count", 0) or 0)
            self._profiles[component] = profile
            self._cache.setdefault(component, self._classify(profile))




@dataclass
class _Timestamped:
    """A scalar observation with a monotonic timestamp for decay.

    Plan fix — ``gpu_id`` annotation enables per-GPU
    maturity fallback on top of component-global maturity.  Primary
    gating signal is component-global (plan  EFT "Self vs Pairwise
    Interference Separation"), but when a specific GPU accumulates
    ``>= observations_per_dim × input_dim`` observations on its own,
    its per-GPU posterior is also considered mature so co-location
    gating on that GPU skips the primary-protect branch — avoiding
    the self-reinforcing imbalance where one GPU hoards tasks while
    others remain blocked forever.
    """

    value: float
    observed_at: float
    gpu_id: str = ""
    query: ReciprocalInterferenceQuery | None = None
    uncertainty: float = 0.0
    provenance: InterferenceProvenance = "unknown"




@dataclass
class _CorrectionRecord:
    """Tracks correction factors for a specific component pair.

    Stores timestamped observations so that stale data can be discarded
    and recent data can be weighted higher via exponential decay.
    """

    latency_slowdowns: list[_Timestamped] = field(default_factory=list)
    vram_overheads: list[_Timestamped] = field(default_factory=list)
    gpu_util_shares: list[_Timestamped] = field(default_factory=list)



_DEFAULT_CLASS_INTERFERENCE: dict[tuple[str, str], float] = {
    (COMPUTE_BOUND, COMPUTE_BOUND): 0.30,
    (COMPUTE_BOUND, MEMORY_BOUND): 0.10,
    (COMPUTE_BOUND, BALANCED): 0.20,
    (MEMORY_BOUND, COMPUTE_BOUND): 0.10,
    (MEMORY_BOUND, MEMORY_BOUND): 0.25,
    (MEMORY_BOUND, BALANCED): 0.18,
    (BALANCED, COMPUTE_BOUND): 0.20,
    (BALANCED, MEMORY_BOUND): 0.18,
    (BALANCED, BALANCED): 0.15,
}

_UNKNOWN_INTERFERENCE = 0.20


@dataclass(frozen=True)
class InterferencePrediction:
    """Predicted interference for a component given its co-located neighbors."""

    component: str
    predicted_slowdown: float
    predicted_vram_overhead: float
    workload_class: str
    neighbor_components: frozenset[str]
    basis: str
    confidence: float
    sample_count: int
    estimate_source: str = ""

    def adjusted_duration(self, solo_duration_sec: float) -> float:
        return solo_duration_sec * (1.0 + self.predicted_slowdown)

    def adjusted_vram(self, solo_vram_mib: float) -> float:
        return solo_vram_mib * (1.0 + self.predicted_vram_overhead)


class InterferenceRegistry:
    """Tracks and predicts interference between co-located GPU workloads.

    Features:
    - **Sliding window** — observations evicted by max_age (time-based) and
      max_history (count-based), aligned with GPEstimate's sliding window.
    - **Exponential confidence decay** — older observations contribute less.
    - **Correction matrix** — tracks latency slowdown, VRAM overhead, and
      GPU utilization share per component pair.

    Regime change flush has been removed — the GP's probabilistic model
    and sliding window naturally handle baseline shifts via posterior
    variance widening.
    """

    def __init__(
        self,
        classifier: WorkloadClassifier,
        *,
        class_interference: dict[tuple[str, str], float] | None = None,
        min_empirical_samples: int = 3,
        max_history: int = 200,
        max_age_sec: float = 7200.0,
        decay_half_life_sec: float = 900.0,
        maturity_observations_per_dim: int = 10,
    ) -> None:
        self._classifier = classifier
        self._class_interference = dict(
            class_interference or _DEFAULT_CLASS_INTERFERENCE
        )
        self._min_empirical = max(1, _as_int(min_empirical_samples))
        self._max_history = max(10, _as_int(max_history))
        self._max_age_sec = _as_float(max_age_sec)
        self._decay_half_life = _as_float(decay_half_life_sec)
        self._maturity_observations_per_dim = max(
            1,
            _as_int(maturity_observations_per_dim),
        )
        self._self_slowdown_prior: float = 2.0
        self._pairwise_slowdown_prior: float = 1.3
        self._self_interference_prior_coeff: float = 1.0
        self._slowdown_prior_variance: float = 0.25
        self._slowdown_prior_strength: float = 1.0

        self._self_slowdown_obs: dict[
            tuple[str, str],
            list[_SelfSlowdownObs],
        ] = defaultdict(list)
        self._pbbc_solo_evidence: set[tuple[str, str]] = set()
        self._pbbc_self_evidence_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._pbbc_self_evidence_gpu_counts: dict[tuple[str, str, str], int] = (
            defaultdict(int)
        )
        self._self_slowdown_ls_N: float = 2.0

        self._solo_baselines: dict[
            tuple[str, str],
            list[_Timestamped],
        ] = defaultdict(list)

        self._solo_vram: dict[
            tuple[str, str],
            list[_Timestamped],
        ] = defaultdict(list)

        self._pairwise: dict[
            tuple[str, str, str],
            _CorrectionRecord,
        ] = defaultdict(_CorrectionRecord)

        self._self_slowdown_cache: dict[
            tuple[str, str, int],
            tuple[float, float, int, str],
        ] = {}
        self._pairwise_slowdown_cache: dict[
            tuple[str, str, str], tuple[float, float] | None
        ] = {}
        self._solo_vram_value_cache: dict[tuple[str, str], float | None] = {}
        self._pairwise_slowdown_value_cache: dict[
            tuple[str, str, str], float | None
        ] = {}

        self._signal_service_ref: Any | None = None
        self._evidence_epoch = 0
        self._evidence_callback: Callable[[int, tuple[str, ...]], None] | None = None
        self._observation_counters: dict[str, int] = defaultdict(int)

    @staticmethod
    def _split_component_fp_key(raw_key: Any) -> tuple[str, str]:
        """Normalize legacy component-only keys and current ``(component, fp)`` keys."""
        if isinstance(raw_key, tuple):
            if len(raw_key) >= 2:
                return str(raw_key[0] or ""), str(raw_key[1] or "")
            if len(raw_key) == 1:
                return str(raw_key[0] or ""), ""
            return "", ""
        return str(raw_key or ""), ""

    def _mark_pbbc_solo_evidence(self, component: str, fp: str = "") -> None:
        component = str(component or "")
        if not component:
            return
        self._pbbc_solo_evidence.add((component, str(fp or "")))

    def _mark_pbbc_self_evidence(
        self,
        component: str,
        fp: str = "",
        *,
        gpu_id: str = "",
    ) -> None:
        component = str(component or "")
        if not component:
            return
        fp = str(fp or "")
        self._pbbc_self_evidence_counts[(component, fp)] += 1
        gid = str(gpu_id or "")
        if gid:
            self._pbbc_self_evidence_gpu_counts[(component, fp, gid)] += 1

    def _has_pbbc_solo_evidence(
        self,
        component: str,
        *,
        fp: str = "",
        exact_fp: bool = False,
    ) -> bool:
        component = str(component or "")
        fp = str(fp or "")
        if not component:
            return False
        if exact_fp and fp:
            return (component, fp) in self._pbbc_solo_evidence
        if fp and (component, fp) in self._pbbc_solo_evidence:
            return True
        if (component, "") in self._pbbc_solo_evidence:
            return True
        return any(comp == component for comp, _other_fp in self._pbbc_solo_evidence)

    def _pbbc_self_evidence_count(
        self,
        component: str,
        *,
        fp: str = "",
        exact_fp: bool = False,
        gpu_id: str | None = None,
    ) -> int:
        component = str(component or "")
        fp = str(fp or "")
        if not component:
            return 0

        def _count_for(key_fp: str) -> int:
            if gpu_id is None:
                return _as_int(
                    self._pbbc_self_evidence_counts.get((component, key_fp), 0)
                )
            gid = str(gpu_id)
            return _as_int(
                self._pbbc_self_evidence_gpu_counts.get((component, key_fp, gid), 0)
            )

        if exact_fp and fp:
            return _count_for(fp)
        if fp:
            exact_count = _count_for(fp)
            if exact_count > 0:
                return exact_count
        cross_fp_count = _count_for("")
        if cross_fp_count > 0:
            return cross_fp_count

        if gpu_id is None:
            return sum(
                _as_int(count)
                for (comp, _other_fp), count in self._pbbc_self_evidence_counts.items()
                if comp == component
            )
        gid = str(gpu_id)
        return sum(
            _as_int(count)
            for (comp, _other_fp, obs_gid), count in (
                self._pbbc_self_evidence_gpu_counts.items()
            )
            if comp == component and obs_gid == gid
        )

    def set_signal_service_ref(self, signal_service: Any) -> None:
        """Inject SignalService back-reference for latency-GP-based
        solo reference computation.  Idempotent; pass ``None`` to
        revert to legacy ``_solo_baselines`` median path."""
        self._signal_service_ref = signal_service

    @property
    def evidence_epoch(self) -> int:
        return self._evidence_epoch

    @property
    def observation_counters(self) -> dict[str, int]:
        return dict(self._observation_counters)

    def register_evidence_callback(
        self,
        callback: Callable[[int, tuple[str, ...]], None] | None,
    ) -> None:
        self._evidence_callback = callback

    def _advance_evidence_epoch(self, *components: str) -> None:
        self._evidence_epoch += 1
        if self._evidence_callback is not None:
            self._evidence_callback(
                self._evidence_epoch,
                tuple(sorted({component for component in components if component})),
            )

    def record_reciprocal_observation(
        self,
        query: ReciprocalInterferenceQuery,
        slowdown_factor: float,
        *,
        uncertainty: float = 0.0,
        provenance: InterferenceProvenance = "exact",
    ) -> bool:
        """Record one identifiable direct self/pair observation."""
        if provenance in {"class-prior", "fixed-prior", "unknown"}:
            self._observation_counters["prior_training_skips"] += 1
            return False
        delta = slowdown_factor_to_delta(slowdown_factor)
        now = time.monotonic()
        if query.is_self:
            observations = self._self_slowdown_obs[
                (query.victim_component, query.victim_config_fingerprint)
            ]
            observations.append(
                _SelfSlowdownObs(
                    n_concurrency=query.self_n,
                    delta=delta,
                    observed_at=now,
                    gpu_id=query.gpu_id,
                    query=query,
                    uncertainty=max(0.0, uncertainty),
                    provenance=provenance,
                )
            )
            if len(observations) > self._max_history:
                del observations[: len(observations) - self._max_history]
            self._mark_pbbc_self_evidence(
                query.victim_component,
                query.victim_config_fingerprint,
                gpu_id=query.gpu_id,
            )
            self._invalidate_self_slowdown_cache(query.victim_component)
            affected = (query.victim_component,)
        else:
            per_instance_delta = delta / query.pair_multiplicity
            record = self._pairwise[
                self._pair_key(
                    query.victim_component,
                    query.interferer_component,
                    query.victim_config_fingerprint,
                )
            ]
            record.latency_slowdowns.append(
                _Timestamped(
                    value=per_instance_delta,
                    observed_at=now,
                    gpu_id=query.gpu_id,
                    query=query,
                    uncertainty=max(0.0, uncertainty) / query.pair_multiplicity,
                    provenance=provenance,
                )
            )
            if len(record.latency_slowdowns) > self._max_history:
                del record.latency_slowdowns[
                    : len(record.latency_slowdowns) - self._max_history
                ]
            self._invalidate_pairwise_slowdown_cache(
                query.victim_component, query.interferer_component
            )
            self._invalidate_pairwise_slowdown_value_cache(
                query.victim_component, query.interferer_component
            )
            affected = (query.victim_component, query.interferer_component)
        self._observation_counters["direct_observations"] += 1
        self._advance_evidence_epoch(*affected)
        return True

    @staticmethod
    def _identity_provenance(
        requested: ReciprocalInterferenceQuery,
        observed: ReciprocalInterferenceQuery | None,
    ) -> InterferenceProvenance:
        if observed is None:
            return "cross-config"
        if requested == observed and requested.has_exact_identity:
            return "exact"
        same_environment = (
            requested.victim_component == observed.victim_component
            and requested.interferer_component == observed.interferer_component
            and requested.victim_config_fingerprint
            == observed.victim_config_fingerprint
            and requested.interferer_config_fingerprint
            == observed.interferer_config_fingerprint
            and requested.gpu_id == observed.gpu_id
            and requested.gpu_model == observed.gpu_model
            and requested.mps_mode == observed.mps_mode
            and requested.worker_backend == observed.worker_backend
            and requested.actor_model == observed.actor_model
            and requested.adapter_version == observed.adapter_version
            and requested.self_n == observed.self_n
            and requested.pair_multiplicity == observed.pair_multiplicity
        )
        return "nearby" if same_environment else "cross-config"

    def query_reciprocal(
        self, query: ReciprocalInterferenceQuery
    ) -> ReciprocalInterferenceResult:
        """Return the best directed keyed evidence, otherwise an explicit prior.

        Self queries are N-stratified (2026-08 naive mean-only policy):
        only observations recorded at the queried ``self_n`` concurrency
        degree are eligible candidates; different-N observations never
        merge into the mean.  When no same-N observation exists, the
        explicit N-linear fixed prior applies.  Pairwise (cross-component)
        queries keep the existing provenance-ranked pooling.
        """
        now = time.monotonic()
        ranked = {"cross-config": 1, "nearby": 2, "exact": 3}
        candidates: list[tuple[InterferenceProvenance, float, float, float]] = []
        if query.is_self:
            pools = [
                observations
                for (component, _fp), observations in self._self_slowdown_obs.items()
                if component == query.victim_component
            ]
            for item in (item for pool in pools for item in self._get_valid(pool)):
                if _as_int(item.n_concurrency) != _as_int(query.self_n):
                    continue
                identity = self._identity_provenance(query, item.query)
                source_rank = ranked.get(item.provenance, 0)
                if source_rank:
                    identity = min(
                        (identity, item.provenance), key=lambda value: ranked[value]
                    )
                candidates.append(
                    (identity, item.delta, item.uncertainty, item.observed_at)
                )
        else:
            for (victim, interferer, _fp), record in self._pairwise.items():
                if (victim, interferer) != (
                    query.victim_component,
                    query.interferer_component,
                ):
                    continue
                for item in self._get_valid(record.latency_slowdowns):
                    identity = self._identity_provenance(query, item.query)
                    source_rank = ranked.get(item.provenance, 0)
                    if source_rank:
                        identity = min(
                            (identity, item.provenance), key=lambda value: ranked[value]
                        )
                    candidates.append(
                        (identity, item.value, item.uncertainty, item.observed_at)
                    )

        if candidates:
            best_rank = max(ranked[provenance] for provenance, *_rest in candidates)
            selected = [item for item in candidates if ranked[item[0]] == best_rank]
            provenance = selected[0][0]
            deltas = [item[1] for item in selected]
            mean_delta = statistics.fmean(deltas)
            if not query.is_self:
                mean_delta *= query.pair_multiplicity
            spread = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            uncertainty = max(spread, *(item[2] for item in selected))
            if not query.is_self:
                uncertainty *= query.pair_multiplicity
            return ReciprocalInterferenceResult(
                slowdown_factor=1.0 + max(0.0, mean_delta),
                uncertainty=max(0.0, uncertainty),
                support=len(selected),
                provenance=provenance,
                age_sec=max(0.0, now - max(item[3] for item in selected)),
                evidence_epoch=self._evidence_epoch,
            )

        if query.is_self:
            delta = max(
                0.0,
                max(1.0, query.self_n * self._self_interference_prior_coeff) - 1.0,
            )
            provenance: InterferenceProvenance = "fixed-prior"
        else:
            victim_class = self._classifier.classify(query.victim_component)
            interferer_class = self._classifier.classify(query.interferer_component)
            class_delta = self._class_interference.get((victim_class, interferer_class))
            if class_delta is None:
                delta = self._pairwise_slowdown_prior - 1.0
                provenance = "fixed-prior"
            else:
                delta = class_delta
                provenance = "class-prior"
            delta *= query.pair_multiplicity
        return ReciprocalInterferenceResult(
            slowdown_factor=1.0 + max(0.0, delta),
            uncertainty=math.sqrt(self._slowdown_prior_variance),
            support=0,
            provenance=provenance,
            age_sec=0.0,
            evidence_epoch=self._evidence_epoch,
        )


    def _compute_reference_solo_latency(
        self,
        component: str,
        config_fingerprint: str,
        input_size: float,
        gpu_id: str | None = None,
    ) -> float | None:
        """Plan — latency-GP-based input-aware solo reference.

        Returns the solo-equivalent expected latency for *component* at
        the given workload point ``(config_fingerprint, input_size, gpu_id)``,
        used as the reference denominator in slowdown computation:

            slowdown = (duration / reference) - 1.0

        Why GP-based instead of ``_get_solo_baseline_value`` median:
        the median pools observations across all input_sizes, so when the
        recording obs has a different input_size from the median-source
        tasks, the resulting ratio bakes input-scaling drift into the
        slowdown.  The latency GP is parameterised on input_size and
        already applies an interference-aware Kalman R_t correction
        (see ``signals/service.py`` corrected_latency path) — its
        prediction approximates the solo-equivalent latency at the
        query point, eliminating the input-scaling pollution.

        Fallback chain (mirrors WSJF tiered fallback):
          (1) per-(component, fp, gpu_id) GP — most specific
          (2) per-(component, fp, *) GP — cross-GPU pool for same fp
          (3) per-(component, *, gpu_id) GP — cross-fp on same GPU
          (4) per-(component, *, *) GP — cross-fp cross-GPU pool
          (5) ``_get_solo_baseline_value(component)`` — legacy median
              (cold-start safety net before GP matures)

        Returns ``None`` only when every tier fails (true cold-start
        with no solo baseline at all); caller should skip recording in
        that window so noisy first observations don't poison the
        slowdown distribution.

        ``signal_service_ref`` is set via ``set_signal_service_ref``;
        when it's ``None`` (e.g. unit tests with bare InterferenceRegistry),
        the function skips tiers (1-4) and goes straight to the legacy
        path.
        """
        ss = self._signal_service_ref
        if ss is not None:
            rp = getattr(ss, "resource_profiles", None)
            predict_fn = getattr(rp, "predict_latency", None)
            if predict_fn is not None and rp is not None:
                fp = config_fingerprint or ""
                gid = str(gpu_id or "") or None
                try:
                    if fp and gid:
                        v = predict_fn(component, fp, _as_float(input_size), gpu_id=gid)
                        if v is not None and v > 0:
                            return _as_float(v)
                    if fp:
                        v = predict_fn(
                            component, fp, _as_float(input_size), gpu_id=None
                        )
                        if v is not None and v > 0:
                            return _as_float(v)
                    profile = getattr(rp, "_profiles", {}).get(component)
                    if profile is not None:
                        cfg_baselines = getattr(profile, "_config_baselines", {})
                        if gid:
                            for cfg in cfg_baselines.values():
                                pred = getattr(cfg, "predict_latency", None)
                                if pred is None:
                                    continue
                                v = pred(_as_float(input_size), gpu_id=gid)
                                if v is not None and v > 0:
                                    return _as_float(v)
                        for cfg in cfg_baselines.values():
                            pred = getattr(cfg, "predict_latency", None)
                            if pred is None:
                                continue
                            v = pred(_as_float(input_size), gpu_id=None)
                            if v is not None and v > 0:
                                return _as_float(v)
                except Exception:
                    _LOG.warning(
                        "[interference-ref] predict_latency raised for "
                        "component=%s fp=%s input=%.1f gpu=%s — falling back "
                        "to legacy solo baseline median.",
                        component,
                        fp,
                        _as_float(input_size),
                        gid,
                        exc_info=True,
                    )
        return self._get_solo_baseline_value(component)


    def record_observation(
        self,
        obs: LatencyObservation,
    ) -> tuple[float, float, float, int] | None:
        """Record a completed latency observation.

        Solo observations update baselines.
        Concurrent observations compute correction factors — even without a
        solo baseline, concurrent-only observations are stored to enable
        relative comparison between pairs once enough data accumulates.

        Returns None (regime change flush removed; sliding window handles eviction).
        """
        component = obs.component
        duration = obs.duration_sec
        now = time.monotonic()

        obs_fp = str(getattr(obs, "config_fingerprint", "") or "")

        if obs.was_solo_throughout:
            return self._record_solo_baseline(component, duration, now, fp=obs_fp)

        solo_fraction = getattr(obs, "solo_fraction", 0.0)
        if solo_fraction > 0.8:
            return self._record_solo_baseline(component, duration, now, fp=obs_fp)

        if not obs.co_located_components:
            return
        correction_provenance = set(getattr(obs, "correction_provenance", ()) or ())
        if getattr(obs, "correction_applied", False) and correction_provenance & {
            "class-prior",
            "fixed-prior",
            "unknown",
        }:
            self._observation_counters["prior_training_skips"] += 1
            return

        segments = obs.composition_segments
        peers: tuple[Any, ...]
        if segments:
            compositions = {
                tuple(
                    sorted(
                        segment.neighbors,
                        key=lambda item: (item.task_id, item.component),
                    )
                )
                for segment in segments
                if segment.end_mono > segment.start_mono
            }
            if len(compositions) != 1:
                self._observation_counters["variable_composition_skips"] += 1
                return
            peers = next(iter(compositions))
            neighbor_components = {peer.component for peer in peers}
        else:
            neighbor_components = set(obs.co_located_components)
            if len(neighbor_components) != 1 or obs.max_concurrent != 2:
                self._observation_counters["ambiguous_composition_skips"] += 1
                return
            peers = ()
        if len(neighbor_components) != 1:
            self._observation_counters["ambiguous_composition_skips"] += 1
            return

        neighbor_component = next(iter(neighbor_components))
        representative = peers[0] if peers else None
        pair_multiplicity = (
            sum(peer.component == neighbor_component for peer in peers) or 1
        )
        obs_gpu_id = obs.gpu_ids[0] if len(obs.gpu_ids) == 1 else ""
        query = ReciprocalInterferenceQuery(
            victim_component=component,
            victim_config_fingerprint=obs_fp,
            victim_input_fingerprint=str(
                getattr(obs, "input_fingerprint", "") or obs.registry_key[2] or ""
            ),
            interferer_component=neighbor_component,
            interferer_config_fingerprint=str(
                getattr(representative, "config_fingerprint", "") or ""
            ),
            interferer_input_fingerprint=str(
                getattr(representative, "input_fingerprint", "") or ""
            ),
            gpu_id=obs_gpu_id,
            gpu_model=str(getattr(obs, "gpu_model", "") or ""),
            mps_mode=str(getattr(obs, "mps_mode", "") or ""),
            worker_backend=str(getattr(obs, "worker_backend", "") or ""),
            actor_model=str(getattr(obs, "actor_model", "") or ""),
            adapter_version=str(getattr(obs, "adapter_version", "") or ""),
            self_n=max(1, obs.max_concurrent),
            pair_multiplicity=pair_multiplicity,
        )
        evidence_provenance: InterferenceProvenance = (
            "exact" if query.has_exact_identity else "cross-config"
        )

        obs_gpu_id_for_ref = obs.gpu_ids[0] if len(obs.gpu_ids) == 1 else None
        input_size_for_ref = self._extract_input_size_from_obs(obs)
        solo_baseline = self._compute_reference_solo_latency(
            component,
            obs_fp,
            input_size_for_ref,
            gpu_id=obs_gpu_id_for_ref,
        )

        if solo_baseline is not None and solo_baseline > 0:
            slowdown = max(0.0, (duration / solo_baseline) - 1.0)
            obs_gpu_id = str(obs_gpu_id)
            N_concurrency = _as_int(getattr(obs, "max_concurrent", 2) or 2)
            for neighbor in obs.co_located_components:
                if neighbor == component:
                    self._self_slowdown_obs[(component, obs_fp)].append(
                        _SelfSlowdownObs(
                            n_concurrency=N_concurrency,
                            delta=slowdown,
                            observed_at=now,
                            gpu_id=obs_gpu_id,
                            query=query,
                            provenance=evidence_provenance,
                        )
                    )
                    self._mark_pbbc_self_evidence(
                        component,
                        obs_fp,
                        gpu_id=obs_gpu_id,
                    )
                    self._invalidate_self_slowdown_cache(component)
                    continue
                pair_key = self._pair_key(component, neighbor, obs_fp)
                record = self._pairwise[pair_key]
                self._append_timestamped(
                    record.latency_slowdowns,
                    slowdown / query.pair_multiplicity,
                    now,
                    obs_gpu_id,
                    query=query,
                    provenance=evidence_provenance,
                )
                self._invalidate_pairwise_slowdown_cache(
                    component,
                    neighbor,
                )
                self._invalidate_pairwise_slowdown_value_cache(
                    component,
                    neighbor,
                )
            self._observation_counters["direct_observations"] += 1
            self._advance_evidence_epoch(component, neighbor_component)
            _LOG.debug(
                "[interference] %s duration=%.2fs baseline=%.2fs slowdown=%.1f%% neighbors=%s",
                component,
                duration,
                solo_baseline,
                slowdown * 100,
                sorted(obs.co_located_components),
            )
        else:
            self._observation_counters["prior_training_skips"] += 1
        return None

    def record_solo_vram(
        self,
        component: str,
        vram_mib: float,
        *,
        fp: str = "",
    ) -> tuple[float, float, float, int] | None:
        """Record solo VRAM observation for baseline.

        Plan Phase B — keyed by ``(component, fp)``.  Default
        ``fp = ""`` keeps backward compat (cross-fp pool).  Regime
        change flush removed — sliding window handles eviction.
        Returns None (no flush events).
        """
        if vram_mib <= 0:
            return None
        now = time.monotonic()
        self._append_timestamped(self._solo_vram[(component, fp)], vram_mib, now)
        self._invalidate_solo_vram_value_cache(component, fp)
        self._advance_evidence_epoch(component)
        return None

    def record_concurrent_vram(
        self,
        component: str,
        neighbor: str,
        vram_mib: float,
        *,
        fp: str = "",
    ) -> None:
        """Record VRAM observation during concurrent execution with neighbor.

        Plan Phase B — keyed by ``(component, neighbor, fp)`` so
        adapter-config variants stay isolated.  Default ``fp = ""``
        keeps backward compat.  Computes overhead ratio against the
        solo VRAM baseline (per-fp lookup with cross-fp fallback) and
        stores it.
        """
        solo_vram = self._get_solo_vram_value(component, fp=fp)
        if solo_vram is None or solo_vram <= 0 or vram_mib <= 0:
            return
        overhead = max(0.0, (vram_mib / solo_vram) - 1.0)
        now = time.monotonic()
        pair_key = self._pair_key(component, neighbor, fp)
        record = self._pairwise[pair_key]
        self._append_timestamped(record.vram_overheads, overhead, now)
        self._advance_evidence_epoch(component, neighbor)

    def record_gpu_util_share(
        self,
        component: str,
        neighbor: str,
        util_share: float,
        *,
        fp: str = "",
    ) -> None:
        """Record estimated GPU util share for component when co-located.

        ``util_share``: fraction of total GPU util attributable to component
        (e.g. 0.45 = 45% of total GPU util).  Plan Phase B — keyed
        by ``(component, neighbor, fp)``; default ``fp = ""``.
        """
        if not (0.0 <= util_share <= 1.0):
            return
        now = time.monotonic()
        pair_key = self._pair_key(component, neighbor, fp)
        record = self._pairwise[pair_key]
        self._append_timestamped(record.gpu_util_shares, util_share, now)
        self._advance_evidence_epoch(component, neighbor)


    def predict(
        self,
        component: str,
        co_located_components: frozenset[str],
        *,
        self_n: int | None = None,
    ) -> InterferencePrediction:
        """Predict interference for *component* given its neighbors.

        Returns slowdown and VRAM overhead as multiplicative factors.
        Confidence is continuous (0.0–1.0) based on sample count and recency.

        ``self_n`` — projected self-interference concurrency degree for the
        candidate (live same-component slots + 1, from the worker latency
        tracker).  When the candidate's component appears in
        ``co_located_components`` (self co-location), the N-stratified
        naive mean-only self estimate is consumed via
        ``get_self_slowdown_estimate_detail``: ``observed_mean`` from the
        first valid same-N observation, otherwise the existing N-linear
        zero-evidence fallback.  Cross-component (pairwise) neighbors keep
        the empirical/class-prior path unchanged.
        """
        if not co_located_components:
            return InterferencePrediction(
                component=component,
                predicted_slowdown=0.0,
                predicted_vram_overhead=0.0,
                workload_class=self._classifier.classify(component),
                neighbor_components=frozenset(),
                basis="none",
                confidence=1.0,
                sample_count=0,
            )

        comp_class = self._classifier.classify(component)
        total_slowdown = 0.0
        total_vram_overhead = 0.0
        total_samples = 0
        all_empirical = True
        confidence_sum = 0.0
        estimate_source = ""

        for neighbor in co_located_components:
            if neighbor == component:
                detail = self.get_self_slowdown_estimate_detail(
                    component,
                    n_concurrent=max(2, _as_int(self_n or 2)),
                )
                if detail is not None and detail["source"] == "observed_mean":
                    total_slowdown += max(0.0, _as_float(detail["mean_delta"]))
                    total_samples += _as_int(detail["sample_count"])
                    n_obs = _as_int(detail["sample_count"])
                    confidence_sum += n_obs / (n_obs + 5.0)
                    estimate_source = "observed_mean"
                elif detail is not None:
                    total_slowdown += max(0.0, _as_float(detail["mean_delta"]))
                    estimate_source = "fallback_prior"
                    all_empirical = False
                    conf = 0.3 if comp_class != UNKNOWN else 0.1
                    confidence_sum += conf
                else:
                    prior = self._class_interference.get(
                        (comp_class, comp_class), _UNKNOWN_INTERFERENCE
                    )
                    total_slowdown += prior
                    all_empirical = False
                    conf = 0.3 if comp_class != UNKNOWN else 0.1
                    confidence_sum += conf
                total_vram_overhead += 0.05
                continue
            pair_key = self._pair_key(component, neighbor)
            record = self._pairwise.get(pair_key)
            valid_slowdowns = (
                self._get_valid(record.latency_slowdowns) if record else []
            )

            if len(valid_slowdowns) >= self._min_empirical:
                slowdown = self._weighted_median(valid_slowdowns)
                total_slowdown += slowdown
                total_samples += len(valid_slowdowns)
                conf = self._observation_confidence(valid_slowdowns)
                confidence_sum += conf

                valid_vram = self._get_valid(record.vram_overheads) if record else []
                if valid_vram:
                    total_vram_overhead += self._weighted_median(valid_vram)
            else:
                all_empirical = False
                neigh_class = self._classifier.classify(neighbor)
                pair_classes = (comp_class, neigh_class)
                prior = self._class_interference.get(
                    pair_classes, _UNKNOWN_INTERFERENCE
                )
                total_slowdown += prior
                total_vram_overhead += 0.05
                conf = 0.3 if comp_class != UNKNOWN else 0.1
                confidence_sum += conf

        total_slowdown = min(total_slowdown, 2.0)
        total_vram_overhead = min(total_vram_overhead, 1.0)

        n_neighbors = len(co_located_components)
        avg_confidence = confidence_sum / n_neighbors if n_neighbors > 0 else 0.0

        if all_empirical or total_samples > 0:
            basis = "empirical"
        else:
            basis = "class_prior"

        return InterferencePrediction(
            component=component,
            predicted_slowdown=total_slowdown,
            predicted_vram_overhead=total_vram_overhead,
            workload_class=comp_class,
            neighbor_components=frozenset(co_located_components),
            basis=basis,
            confidence=round(min(1.0, avg_confidence), 3),
            sample_count=total_samples,
            estimate_source=estimate_source,
        )


    def get_solo_baseline(
        self,
        component: str,
        *,
        fp: str = "",
    ) -> float | None:
        return self._get_solo_baseline_value(component, fp=fp)

    def has_solo_baseline(
        self,
        component: str,
        *,
        fp: str = "",
        exact_fp: bool = False,
        include_expired: bool = False,
    ) -> bool:
        """Return whether a valid solo latency baseline exists.

        This is the existence-only counterpart to ``get_solo_baseline`` for
        hot planner gates that need the PBBC phase, not the baseline value.
        It preserves the same fp fallback chain as the value accessor while
        avoiding weighted-median work.  When ``exact_fp`` is true and ``fp`` is
        non-empty, only the exact ``(component, fp)`` bucket is considered; this
        is used by PBBC so one config's solo anchor does not release another.

        ``include_expired`` is intentionally separate from value prediction:
        PBBC needs "has this workload point ever completed a solo run in this
        process?" evidence and must not regress to cold mode merely because the
        latency GP's sliding window expired during a long benchmark.  Callers
        that need a fresh baseline value keep the default ``False``.
        """
        if exact_fp and fp:
            values = self._solo_baselines.get((component, fp), [])
            if include_expired:
                return self._has_pbbc_solo_evidence(
                    component,
                    fp=fp,
                    exact_fp=True,
                )
            cutoff = time.monotonic() - self._max_age_sec
            return any(o.observed_at >= cutoff for o in values)
        if include_expired:
            return self._has_pbbc_solo_evidence(
                component,
                fp=fp,
                exact_fp=exact_fp,
            )
        return self._has_solo_baseline(component, fp=fp)

    def get_solo_vram_baseline(
        self,
        component: str,
        *,
        fp: str = "",
    ) -> float | None:
        return self._get_solo_vram_value(component, fp=fp)

    def get_pairwise_slowdown(
        self,
        component_a: str,
        component_b: str,
        *,
        fp: str = "",
    ) -> float | None:
        cache_key = self._pair_key(component_a, component_b, fp)
        if cache_key in self._pairwise_slowdown_value_cache:
            return self._pairwise_slowdown_value_cache[cache_key]
        record = self._get_pairwise_record(component_a, component_b, fp=fp)
        if record is None:
            self._pairwise_slowdown_value_cache[cache_key] = None
            return None
        valid = self._get_valid(record.latency_slowdowns)
        result = self._weighted_median(valid) if valid else None
        self._pairwise_slowdown_value_cache[cache_key] = result
        return result

    def _invalidate_pairwise_slowdown_value_cache(
        self,
        component_a: str,
        component_b: str,
    ) -> None:
        """Drop cached victim→interferer slowdown values across all fp."""
        if not self._pairwise_slowdown_value_cache:
            return
        stale = [
            k
            for k in self._pairwise_slowdown_value_cache
            if (k[0], k[1]) == (component_a, component_b)
        ]
        for k in stale:
            del self._pairwise_slowdown_value_cache[k]

    def get_pairwise_vram_overhead(
        self,
        component_a: str,
        component_b: str,
        *,
        fp: str = "",
    ) -> float | None:
        record = self._get_pairwise_record(component_a, component_b, fp=fp)
        if record is None:
            return None
        valid = self._get_valid(record.vram_overheads)
        return self._weighted_median(valid) if valid else None


    def get_self_observation_count(
        self,
        component: str,
        gpu_id: str | None = None,
        *,
        fp: str = "",
        exact_fp: bool = False,
        include_expired: bool = False,
    ) -> int:
        """Return the count of valid (non-expired) self-interference
        observations for *component*.

        Plan fix Stage 3 — storage moved from
        ``_pairwise[(c, __self__:c)]`` to the 2-D-feature
        ``_self_slowdown_obs[(component, fp)]`` list.  Recency filter
        applied via ``_max_age_sec``.  ``gpu_id`` filter preserved
        for API compat (per-GPU count still supported though
        fix fallback retired in Stage 1 fix).

        Plan Phase B — ``fp`` keyword filters by config_fingerprint
        (default ``""`` walks the cross-fp fallback chain in
        ``_iter_self_slowdown_obs``).

        Returns 0 if no observations or all expired.  ``include_expired`` is
        for PBBC evidence only: the gate needs to know whether a 2-way sample
        has ever completed in this process, while the learned GP still uses
        the default fresh-window count for maturity and posterior confidence.
        """
        if include_expired:
            evidence_count = self._pbbc_self_evidence_count(
                component,
                fp=fp,
                exact_fp=exact_fp,
                gpu_id=gpu_id,
            )
            if evidence_count > 0:
                return evidence_count
        if exact_fp and fp:
            obs_list = list(self._self_slowdown_obs.get((component, fp), []))
        else:
            obs_list = self._iter_self_slowdown_obs(component, fp=fp)
        if not obs_list:
            return 0
        if include_expired:
            if gpu_id is None:
                return len(obs_list)
            target = str(gpu_id)
            return sum(1 for o in obs_list if o.gpu_id == target)
        now = time.monotonic()
        valid = [o for o in obs_list if (now - o.observed_at) <= self._max_age_sec]
        if gpu_id is None:
            return len(valid)
        target = str(gpu_id)
        return sum(1 for o in valid if o.gpu_id == target)

    def get_pair_observation_count(
        self,
        component_a: str,
        component_b: str,
        *,
        fp: str = "",
    ) -> int:
        """Return the count of valid pairwise (inter-component)
        observations for *(component_a, component_b)*.  Reserved
        self-pair key ``__self__:*`` is routed via
        ``get_self_observation_count`` instead.

        Plan Phase B — ``fp`` keyword filters by config_fingerprint
        (default ``""`` walks the cross-fp fallback chain).
        """
        if component_a == component_b or component_b == f"__self__:{component_a}":
            return self.get_self_observation_count(component_a, fp=fp)
        record = self._get_pairwise_record(component_a, component_b, fp=fp)
        if record is None:
            return 0
        return len(self._get_valid(record.latency_slowdowns))

    def get_self_input_dimension(self, component: str) -> int:
        """Self-interference GP input feature dimension.  Plan 
         C — "Default 10 = 1-D GP (component).   
         ."  Currently 1-D (component identifier only); future
        extensions (input_size / batch_size stratification) will return
        higher values.  Conservative default 1 when unknown.
        """
        return 1

    def get_pair_input_dimension(
        self,
        component_a: str,
        component_b: str,
    ) -> int:
        """Pairwise-interference GP input feature dimension.  Plan 
         C — 1-D (pair identifier only) by default; extensions
        may stratify by overlap duration or workload features.
        """
        return 1

    def is_self_mature(
        self,
        component: str,
        *,
        observations_per_dim: int | None = None,
        gpu_id: str | None = None,
        fp: str = "",
    ) -> bool:
        """Plan fix — self-interference GP maturity gate.

        Hybrid maturity (Plan fix "Per-GPU mature fallback"):
          - When ``gpu_id`` is None → component-global maturity
            (legacy / default).  True when
            ``get_self_observation_count(component) >= threshold``.
          - When ``gpu_id`` is given → **OR of two conditions**:
              (a) per-GPU local maturity — observations specifically
                  recorded on ``gpu_id`` already exceed threshold, OR
              (b) component-global maturity still holds.
            Either one is sufficient.  The per-GPU branch lets a
            less-loaded GPU that genuinely accumulated enough
            observations on its own skip the cold-start primary-
            protect gate independently, preventing self-reinforcing
            imbalance where one GPU hoards tasks while others
            remain blocked.

        *observations_per_dim* default: ``self._maturity_observations_per_dim``
        (10 per Plan Category C, injected at construction via
        SupervisorConfig.gp_maturity_observations_per_dim).

        Plan fix — **early-warm (solo + pair = mature)**:
        Bench <stamp> (9/9 success, 1969 s)   
        `self_obs`   component  10 threshold   
        (protenix 9  ) — cold-phase gate  co-location 
         self-pair observation   bench  
         feedback loop .  **user intent** ("solo + 2-way
           ")    early-warm  :
        solo baseline    pair obs ≥ 1  maturity .
        PBBC (fix)  "solo-baseline anchor   pairwise
        "      — PBBC flag  solo ,
         self-pair observation  "cold    co-location
        " .  Legacy threshold (10×dim)   OR  
        (solo baseline   pair obs   edge case).

        Plan fix — **early-warm    **:
         fix code  early-warm   `get_self_
        observation_count(component, gpu_id=gpu_id)`  per-GPU 
         "solo  pair   GPU   "   
        .   fix  `gpu_id`   ** signal
         component-level  **   — plan  entry
           .

        Plan fix — **10×d legacy OR fallback + fix
        per-GPU fallback  **:
        DACE folklore `n = 10·d` rule-of-thumb  hyperparameter MLE
           (Loeppky-Sacks-Welch 2009)  GP posterior
        ** **  .  GP-UCB (Srinivas et al. 2010)
        regret bound  **N≥1 observation  valid** — posterior_std
          N  conservative   (μ + β·σ).   10×d
        gate  .  PBBC (fix) + bootstrap   first
        dispatch  `solo_exists=True`     → legacy
        `global_count >= 10×dim`   PBBC    **
        ** (dead code).  fix per-GPU fallback 
        `obs.gpu_id` vs `gpu_ids` annotation bug  per-GPU count 
         0 → dead code.      , `is_self_mature`
         early-warm    .  `observations_per_dim` /
        `gpu_id`   API     .
        """
        del observations_per_dim, gpu_id
        if not self._has_solo_baseline(component, fp=fp):
            return False
        return self.get_self_observation_count(component, fp=fp) >= 1

    def is_pair_mature(
        self,
        component_a: str,
        component_b: str,
        *,
        observations_per_dim: int | None = None,
    ) -> bool:
        """Return empirical maturity only when directed observations exist."""
        required = max(1, observations_per_dim or 1)
        return self.get_pair_observation_count(component_a, component_b) >= required

    def set_maturity_observations_per_dim(self, n: int) -> None:
        """Plan Fix-3+4 — inject Category C maturity threshold
        from SupervisorConfig after construction.  Called by the gateway
        wiring layer (``http_server._ensure_core_loop_components``) once
        both SignalService and WorkerSupervisor are initialized.

         fix note: legacy 10×d gate removed, so this value
        no longer affects ``is_self_mature`` / ``is_pair_mature``.
        Preserved for API compatibility + potential future use as a
        hyperparameter-tuning trigger (MLE stability heuristic per
        Loeppky-Sacks-Welch 2009, not a gate).
        """
        self._maturity_observations_per_dim = max(1, _as_int(n))

    def set_slowdown_prior_means(
        self,
        *,
        self_slowdown: float | None = None,
        pairwise_slowdown: float | None = None,
    ) -> None:
        """Plan fix — inject GP prior mean function
        baselines from SupervisorConfig after construction.  Called
        by the gateway wiring layer alongside
        ``set_maturity_observations_per_dim``.

        Semantics: when ``get_{self,pairwise}_slowdown_if_mature`` is
        called and the underlying observation record is empty (N=0),
        the registry returns the prior-based delta (value - 1.0).
        Values >= 1.0 required (slowdown factor, 1.0 = no slowdown).
        """
        if self_slowdown is not None:
            v = _as_float(self_slowdown)
            if v >= 1.0:
                self._self_slowdown_prior = v
        if pairwise_slowdown is not None:
            v = _as_float(pairwise_slowdown)
            if v >= 1.0:
                self._pairwise_slowdown_prior = v

    def set_self_interference_prior_coeff(self, coeff: float) -> None:
        """ v5.0 fix — inject single global N-linear self-interference
        coefficient from ``workers.yaml shared.self_interference_prior``
        (or ``SupervisorConfig.self_interference_prior`` default 1.0).

        ``coeff`` is non-negative; 0 = no slowdown prior (per-process
        throughput unaffected by N), 1.0 = full Roofline-N upper bound,
        >1 = super-Roofline (memory-bound contention worst case).

        Wired by ``http_server._ensure_core_loop_components`` after
        InterferenceRegistry construction.  ``get_self_slowdown_if_mature``
        N=0 prior  ``sd = max(1.0, N * coeff)``  .
        """
        if coeff < 0.0:
            return
        self._self_interference_prior_coeff = _as_float(coeff)


    def _invalidate_self_slowdown_cache(self, component: str) -> None:
        """fix — drop every cached self-posterior tuple whose key
        starts with ``component``.  Plan Phase B: invalidates across
        all fp variants for the component (key tuple is now ``(component,
        fp, input_size, n_concurrency)`` so we still match on index 0)."""
        if not self._self_slowdown_cache:
            return
        stale = [k for k in self._self_slowdown_cache if k[0] == component]
        for k in stale:
            del self._self_slowdown_cache[k]

    def _invalidate_pairwise_slowdown_cache(
        self,
        component_a: str,
        component_b: str,
    ) -> None:
        """Drop cached victim→interferer posteriors across all fp."""
        if not self._pairwise_slowdown_cache:
            return
        if component_a == component_b:
            self._invalidate_self_slowdown_cache(component_a)
            return
        stale = [
            k
            for k in self._pairwise_slowdown_cache
            if (k[0], k[1]) == (component_a, component_b)
        ]
        for k in stale:
            del self._pairwise_slowdown_cache[k]

    def get_self_slowdown_if_mature(
        self,
        component: str,
        *,
        n_concurrent: int = 2,
        input_size: float = 0.0,
        observations_per_dim: int | None = None,
        fp: str = "",
    ) -> tuple[float, float] | None:
        """Self-slowdown (delta form) — N-stratified naive mean-only
        estimate, returned as ``(mean, std)``.

        2026-08 policy change — maturity gating removed from estimate
        consumption.  Storage indexes only by ``N_concurrency`` (input_size
        dimension dropped — Phase A reference normalization factored it out
        at recording time) and the estimate is the **arithmetic mean of the
        valid deltas recorded at the exact queried N**:

          - sample_count(N) >= 1 → ``(mean(deltas at N), pstdev(deltas at
            N))`` — used immediately from the first valid observation; no
            prior blending, no kernel smoothing across N, no minimum sample
            count.  ``std`` is the descriptive spread of the same-N
            observations (0.0 for a single sample) so callers that apply a
            conservative ``z·std`` term keep their existing semantics.
          - sample_count(N) == 0 → existing zero-evidence fallback:
            ``(max(1, N·coeff) - 1, sqrt(prior_variance))`` — the
            intentional fresh-run cold-start prior is unchanged.
          - None only when there is no observation at the queried N AND no
            solo baseline (component never dispatched — PBBC guarantees
            non-None after first dispatch).

        Scope: interference-only.  Does not affect latency (SignalService)
        or VRAM (ResourceProfileRegistry) predictions.

        Args:
          input_size: legacy parameter retained for API compat; ignored
            (storage axis removed in  redesign).
        """
        del observations_per_dim
        del input_size
        estimate = self._cached_self_slowdown_estimate(
            component,
            fp or "",
            _as_int(n_concurrent),
        )
        if estimate is None:
            return None
        mean_delta, std_delta, _sample_count, _source = estimate
        return (mean_delta, std_delta)

    def get_self_slowdown_estimate_detail(
        self,
        component: str,
        *,
        n_concurrent: int = 2,
        fp: str = "",
    ) -> dict[str, Any] | None:
        """Observability companion to ``get_self_slowdown_if_mature``.

        Returns the same N-stratified mean-only estimate with explicit
        metadata, or ``None`` under the identical zero-evidence + no-solo
        condition:

          - ``mean_delta`` / ``std_delta``: the estimate tuple values.
          - ``sample_count``: valid observations at the queried N
            (0 on the fallback path).
          - ``queried_n``: the concurrency degree this estimate answers.
          - ``source``: ``"observed_mean"`` when at least one valid
            same-N observation exists, ``"fallback_prior"`` for the
            zero-evidence cold-start prior.
        """
        estimate = self._cached_self_slowdown_estimate(
            component,
            fp or "",
            _as_int(n_concurrent),
        )
        if estimate is None:
            return None
        mean_delta, std_delta, sample_count, source = estimate
        return {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "sample_count": sample_count,
            "queried_n": _as_int(n_concurrent),
            "source": source,
        }

    def _cached_self_slowdown_estimate(
        self,
        component: str,
        fp: str,
        n: int,
    ) -> tuple[float, float, int, str] | None:
        """fix cache wrapper around ``_self_slowdown_estimate``.

        Cache key ``(component, fp, n)`` (input_size axis removed in the
         redesign).  The ``None`` (no observation at N and no solo
        baseline) result is intentionally not cached — it re-evaluates
        the solo gate per call, matching pre-change behavior.
        """
        cache_key = (component, fp, n)
        cached = self._self_slowdown_cache.get(cache_key)
        if cached is not None:
            return cached
        estimate = self._self_slowdown_estimate(component, fp=fp, n_concurrent=n)
        if estimate is not None:
            self._self_slowdown_cache[cache_key] = estimate
        return estimate

    def _self_slowdown_estimate(
        self,
        component: str,
        *,
        fp: str,
        n_concurrent: int,
    ) -> tuple[float, float, int, str] | None:
        """Compute the N-stratified naive mean-only self-slowdown estimate.

        Returns ``(mean_delta, std_delta, sample_count, source)`` where
        ``source`` is ``"observed_mean"`` or ``"fallback_prior"``; None
        only when zero observations exist at the queried N and the solo
        baseline is absent (pre-first-dispatch defensive gate, preserved
        from the pre-mean-only behavior).
        """
        n = _as_int(n_concurrent)
        at_n = [
            o
            for o in self._valid_self_obs(component, fp=fp)
            if _as_int(o.n_concurrency) == n
        ]
        if at_n:
            deltas = [_as_float(o.delta) for o in at_n]
            mean_delta = statistics.fmean(deltas)
            std_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            return (mean_delta, std_delta, len(at_n), "observed_mean")
        if self._get_solo_baseline_value(component, fp=fp) is None:
            return None
        coeff = self._self_interference_prior_coeff
        prior_sd = max(1.0, _as_float(n) * coeff)
        return (
            max(0.0, prior_sd - 1.0),
            math.sqrt(max(0.0, self._slowdown_prior_variance)),
            0,
            "fallback_prior",
        )

    def get_pairwise_slowdown_if_mature(
        self,
        component_a: str,
        component_b: str,
        *,
        observations_per_dim: int | None = None,
        fp: str = "",
    ) -> tuple[float, float] | None:
        """Plan fix/10/11 — pairwise slowdown (delta form)
        returned as ``(mean, std)``.

        Returns ``(mean_delta, std_delta)``:
          - N>=1 observations: recency-weighted mean + std of recorded
            pair slowdowns (delta form).
          - N=0: ``(pairwise_prior - 1.0, prior_std)`` — default
            ``(0.3, 0.5)`` before the scheduler applies its selected z.

        Self-pair lookup (``component_a == component_b``) redirects to
        ``get_self_slowdown_if_mature`` for consistent semantic.

        Unlike pre- behavior, this API never returns None under
        normal operation — None path retained only for self-lookup
        redirect when solo baseline is absent.

        Scope: interference-only (see class-level note).
        """
        del observations_per_dim
        if component_a == component_b or component_b == f"__self__:{component_a}":
            return self.get_self_slowdown_if_mature(component_a, fp=fp)
        cache_key = self._pair_key(component_a, component_b, fp)
        if cache_key in self._pairwise_slowdown_cache:
            return self._pairwise_slowdown_cache[cache_key]
        prior_mean_delta = _as_float(self._pairwise_slowdown_prior) - 1.0
        record = self._get_pairwise_record(component_a, component_b, fp=fp)
        obs_list: list[_Timestamped] = []
        if record is not None:
            obs_list = self._get_valid(record.latency_slowdowns)
        result = self._bayesian_slowdown_posterior(prior_mean_delta, obs_list)
        self._pairwise_slowdown_cache[cache_key] = result
        return result

    def get_interference_matrix(self) -> dict[str, Any]:
        """Plan Phase B — cross-fp aggregated view.  Internal
        storage is keyed by ``(comp_a, comp_b, fp)`` but the matrix
        outer key remains the legacy ``"comp_a:comp_b"`` format so
        ops dashboards / API readers see the union across fp variants.
        Per-fp introspection is available via ``as_dict_per_fp``
        (future extension) when needed."""
        matrix: dict[str, Any] = {}
        aggregated: dict[
            tuple[str, str],
            tuple[
                list[_Timestamped],
                list[_Timestamped],
                list[_Timestamped],
            ],
        ] = defaultdict(lambda: ([], [], []))
        for (comp_a, comp_b, _fp), record in self._pairwise.items():
            slot = aggregated[(comp_a, comp_b)]
            slot[0].extend(record.latency_slowdowns)
            slot[1].extend(record.vram_overheads)
            slot[2].extend(record.gpu_util_shares)
        for (comp_a, comp_b), (sd_list, vr_list, gu_list) in sorted(aggregated.items()):
            valid_s = self._get_valid(sd_list)
            valid_v = self._get_valid(vr_list)
            valid_g = self._get_valid(gu_list)
            key = f"{comp_a}:{comp_b}"
            matrix[key] = {
                "slowdown_median": self._weighted_median(valid_s) if valid_s else None,
                "vram_overhead_median": self._weighted_median(valid_v)
                if valid_v
                else None,
                "gpu_util_share_median": self._weighted_median(valid_g)
                if valid_g
                else None,
                "n_slowdown": len(valid_s),
                "n_vram": len(valid_v),
                "n_gpu_util": len(valid_g),
                "confidence": self._observation_confidence(valid_s) if valid_s else 0.0,
            }
        return matrix

    def max_observed_slowdown(self) -> float:
        """Return the maximum pairwise slowdown observed across all pairs.

        Used as σ²_bias fallback when interference data is unavailable for a
        specific pair (Case 3 in Kalman correction pipeline).
        """
        max_sd = 0.0
        for record in self._pairwise.values():
            valid = self._get_valid(record.latency_slowdowns)
            if valid:
                median = self._weighted_median(valid)
                if median is not None and median > max_sd:
                    max_sd = median
        return max_sd

    def evict_stale(self) -> int:
        """Remove expired observations from all records. Returns count removed."""
        removed = 0
        for baselines in self._solo_baselines.values():
            before = len(baselines)
            self._evict_expired(baselines)
            removed += before - len(baselines)
        for baselines in self._solo_vram.values():
            before = len(baselines)
            self._evict_expired(baselines)
            removed += before - len(baselines)
        for record in self._pairwise.values():
            for obs_list in (
                record.latency_slowdowns,
                record.vram_overheads,
                record.gpu_util_shares,
            ):
                before = len(obs_list)
                self._evict_expired(obs_list)
                removed += before - len(obs_list)
        for observations in self._self_slowdown_obs.values():
            before = len(observations)
            self._evict_expired(observations)
            removed += before - len(observations)
        if removed:
            self._self_slowdown_cache.clear()
            self._pairwise_slowdown_cache.clear()
            self._solo_vram_value_cache.clear()
            self._pairwise_slowdown_value_cache.clear()
            self._advance_evidence_epoch()
        return removed

    def as_dict(self) -> dict[str, Any]:
        """Plan Phase B — cross-fp aggregated view.  Internal
        storage is keyed by ``(component, fp)`` for ``solo_baselines`` /
        ``solo_vram`` / ``_self_slowdown_obs`` and ``(comp_a, comp_b,
        fp)`` for ``_pairwise``.  This export collapses observations
        across fp variants under the legacy component-only key so ops
        dashboards / API readers see the union; per-fp introspection is
        a future extension when needed."""
        solo_agg: dict[str, list[_Timestamped]] = defaultdict(list)
        for raw_key, vals in self._solo_baselines.items():
            comp, _fp = self._split_component_fp_key(raw_key)
            solo_agg[comp].extend(vals)
        vram_agg: dict[str, list[_Timestamped]] = defaultdict(list)
        for raw_key, vals in self._solo_vram.items():
            comp, _fp = self._split_component_fp_key(raw_key)
            vram_agg[comp].extend(vals)
        components_seen = set(solo_agg.keys()) | {
            c for (a, b, _fp) in self._pairwise for c in (a, b)
        }
        return {
            "solo_baselines": {
                comp: {
                    "median_sec": self._weighted_median(self._get_valid(vals)),
                    "n": len(self._get_valid(vals)),
                    "n_total": len(vals),
                }
                for comp, vals in sorted(solo_agg.items())
            },
            "solo_vram": {
                comp: {
                    "median_mib": self._weighted_median(self._get_valid(vals)),
                    "n": len(self._get_valid(vals)),
                }
                for comp, vals in sorted(vram_agg.items())
            },
            "pairwise": self.get_interference_matrix(),
            "self_slowdown": self._self_slowdown_snapshot(),
            "workload_classes": {
                comp: self._classifier.classify(comp)
                for comp in sorted(components_seen)
            },
            "evidence_epoch": self._evidence_epoch,
            "observation_counters": dict(self._observation_counters),
            "config": {
                "max_age_sec": self._max_age_sec,
                "max_history": self._max_history,
                "decay_half_life_sec": self._decay_half_life,
                "min_empirical_samples": self._min_empirical,
            },
        }


    def dump_to_jsonl(self, path: str) -> int:
        """Dump observation history to JSONL for post-hoc convergence trace.

        Writes one JSON record per line with fields:
          - ``kind``: "solo_baseline" | "solo_vram" | "pairwise" |
                       "self_slowdown" | "pair_count_global"
          - ``key``: kind-specific tuple (component / pair / etc.)
          - ``observations``: list of `{value, observed_at, gpu_id, ...}`

        Each observation preserves its monotonic timestamp + value +
        any structural metadata (gpu_id, n_concurrency for self-slowdown,
        latency/vram subkind for pairwise) so the consumer can reproduce
        the time-ordered ingest order and correlate with Planner
        prediction trajectories from `profile_runs`.

        Returns the number of records written.
        """
        import json

        records = 0
        with _writable_text(path) as fh:
            for raw_key, vals in self._solo_baselines.items():
                comp, fp = self._split_component_fp_key(raw_key)
                fh.write(
                    json.dumps(
                        {
                            "kind": "solo_baseline",
                            "component": comp,
                            "config_fingerprint": fp,
                            "observations": [
                                {
                                    "value": o.value,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in vals
                            ],
                        }
                    )
                    + "\n"
                )
                records += 1
            for raw_key, vals in self._solo_vram.items():
                comp, fp = self._split_component_fp_key(raw_key)
                fh.write(
                    json.dumps(
                        {
                            "kind": "solo_vram",
                            "component": comp,
                            "config_fingerprint": fp,
                            "observations": [
                                {
                                    "value": o.value,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in vals
                            ],
                        }
                    )
                    + "\n"
                )
                records += 1
            for (comp_a, comp_b, fp), record in self._pairwise.items():
                fh.write(
                    json.dumps(
                        {
                            "kind": "pairwise",
                            "comp_a": comp_a,
                            "comp_b": comp_b,
                            "config_fingerprint": fp,
                            "latency_slowdowns": [
                                {
                                    "value": o.value,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in record.latency_slowdowns
                            ],
                            "vram_overheads": [
                                {
                                    "value": o.value,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in record.vram_overheads
                            ],
                            "gpu_util_shares": [
                                {
                                    "value": o.value,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in record.gpu_util_shares
                            ],
                        }
                    )
                    + "\n"
                )
                records += 1
            for (comp, fp), obs_list in self._self_slowdown_obs.items():
                fh.write(
                    json.dumps(
                        {
                            "kind": "self_slowdown",
                            "component": comp,
                            "config_fingerprint": fp,
                            "observations": [
                                {
                                    "n_concurrency": o.n_concurrency,
                                    "delta": o.delta,
                                    "observed_at": o.observed_at,
                                    "gpu_id": o.gpu_id,
                                }
                                for o in obs_list
                            ],
                        }
                    )
                    + "\n"
                )
                records += 1
            fh.write(
                json.dumps(
                    {
                        "kind": "config",
                        "max_age_sec": self._max_age_sec,
                        "max_history": self._max_history,
                        "decay_half_life_sec": self._decay_half_life,
                        "min_empirical_samples": self._min_empirical,
                        "pairwise_slowdown_prior": self._pairwise_slowdown_prior,
                        "self_slowdown_prior": getattr(
                            self, "_self_slowdown_prior", None
                        ),
                        "slowdown_prior_variance": self._slowdown_prior_variance,
                        "slowdown_prior_strength": self._slowdown_prior_strength,
                        "self_slowdown_ls_N": self._self_slowdown_ls_N,
                    }
                )
                + "\n"
            )
            records += 1
        return records


    def _valid_self_obs(
        self,
        component: str,
        *,
        gpu_id: str | None = None,
        fp: str = "",
    ) -> list[_SelfSlowdownObs]:
        """Return non-expired ``_self_slowdown_obs`` entries for *component*.

        Mirrors ``_get_valid`` semantics for ``_Timestamped`` but applies
        to the 2-D feature storage.  ``gpu_id`` filter is optional — when
        provided, only observations with matching ``gpu_id`` pass.
        Plan Phase B — ``fp`` keyword routes through
        ``_iter_self_slowdown_obs`` (per-fp → cross-fp → any-fp fallback).
        """
        obs_list = self._iter_self_slowdown_obs(component, fp=fp)
        if not obs_list:
            return []
        now = time.monotonic()
        cutoff = now - self._max_age_sec
        valid = [o for o in obs_list if o.observed_at >= cutoff]
        if gpu_id is not None:
            target = str(gpu_id)
            valid = [o for o in valid if o.gpu_id == target]
        return valid

    def _self_slowdown_snapshot(self) -> dict[str, Any]:
        """Plan fix — export 2-D self-slowdown storage to
        the ``interference.self_slowdown`` API section.  Plan 
        Phase B — cross-fp aggregated view (component-level union of
        all fp variants).

        Per-component payload:
          - ``n_obs``     : raw observation count (includes expired).
          - ``n_valid``   : count passing ``_max_age_sec`` recency filter.
          - ``median_delta`` : simple median over valid ``delta`` values
            (returns 0.0 when no valid obs).
          - ``mean_delta``   : arithmetic mean over valid ``delta`` values
            (0.0 fallback).
          - ``recent_samples`` : last 5 observations (raw list order) —
            each entry is a flat dict suitable for JSON.
        """
        out: dict[str, Any] = {}
        components_seen: set = set()
        for comp, _fp in self._self_slowdown_obs:
            components_seen.add(comp)
        for component in sorted(components_seen):
            raw: list[_SelfSlowdownObs] = []
            for (comp, _fp), entries in self._self_slowdown_obs.items():
                if comp == component:
                    raw.extend(entries)
            valid = self._valid_self_obs(component)
            deltas = [_as_float(o.delta) for o in valid]
            if deltas:
                median_delta = _as_float(statistics.median(deltas))
                mean_delta = sum(deltas) / len(deltas)
            else:
                median_delta = 0.0
                mean_delta = 0.0
            out[component] = {
                "n_obs": len(raw),
                "n_valid": len(valid),
                "median_delta": median_delta,
                "mean_delta": mean_delta,
                "recent_samples": [
                    {
                        "n_concurrency": _as_int(o.n_concurrency),
                        "delta": _as_float(o.delta),
                        "gpu_id": o.gpu_id,
                        "observed_at": _as_float(o.observed_at),
                    }
                    for o in list(raw)[-5:]
                ],
            }
        return out


    def _record_solo_baseline(
        self,
        component: str,
        duration: float,
        now: float,
        fp: str = "",
    ) -> tuple[float, float, float, int] | None:
        """Record a solo baseline observation.

        Regime change detection and flush have been removed — the GP's
        probabilistic model and sliding window handle regime shifts
        naturally via posterior variance widening.  Old observations
        are evicted by the time-based sliding window (max_age_sec).

        Plan Phase B — keyed by ``(component, fp)``.  Default
        ``fp = ""`` keeps backward compat for the cross-fp pool used
        by legacy callers and by the ``was_solo_throughout`` /
        ``solo_fraction > 0.8`` paths in ``record_observation`` that
        already extract ``obs.config_fingerprint``.

        Returns None (no flush events).
        """
        self._append_timestamped(self._solo_baselines[(component, fp)], duration, now)
        self._mark_pbbc_solo_evidence(component, fp)
        self._advance_evidence_epoch(component)
        return None

    def _append_timestamped(
        self,
        obs_list: list[_Timestamped],
        value: float,
        now: float,
        gpu_id: str = "",
        *,
        query: ReciprocalInterferenceQuery | None = None,
        provenance: InterferenceProvenance = "unknown",
    ) -> None:
        obs_list.append(
            _Timestamped(
                value=value,
                observed_at=now,
                gpu_id=str(gpu_id or ""),
                query=query,
                provenance=provenance,
            )
        )
        if len(obs_list) > self._max_history:
            del obs_list[: len(obs_list) - self._max_history]

    def _get_valid(self, obs_list: Sequence[_ObservedT]) -> list[_ObservedT]:
        """Return non-expired observations."""
        if not obs_list:
            return []
        now = time.monotonic()
        cutoff = now - self._max_age_sec
        return [o for o in obs_list if o.observed_at >= cutoff]

    def _evict_expired(self, obs_list: list[_ObservedT]) -> None:
        now = time.monotonic()
        cutoff = now - self._max_age_sec
        obs_list[:] = [o for o in obs_list if o.observed_at >= cutoff]

    def _decay_weight(self, observed_at: float) -> float:
        """Exponential decay weight: recent observations weigh more."""
        age = time.monotonic() - observed_at
        if age <= 0:
            return 1.0
        return math.exp(-age * math.log(2) / self._decay_half_life)

    def _weighted_mean(self, obs_list: list[_Timestamped]) -> float:
        """Compute a recency-weighted arithmetic mean.

        Plan — replaces the legacy ``_weighted_median`` summary used
        by every empirical-aggregation site (``get_pairwise_slowdown``,
        ``_get_solo_baseline_value``, ``predict``'s neighbor loop,
        ``as_dict`` / ``get_interference_matrix`` exports, etc.).
        Switching to mean unifies the registry's empirical statistic
        with the GP-UCB ``μ + z·σ`` Bayesian posterior already used by
        the scheduling-critical ``get_*_if_mature`` API
        (fix Stage 3) — no more split-statistic semantics where
        scheduling decisions used mean while neighbouring code paths
        used median for the same observation list.

        Empty list → 0.0; single obs → its value (degenerate weighted
        mean).  ``_decay_weight`` total ≤ 0 falls back to plain
        arithmetic mean (matches legacy median fallback behaviour).
        """
        if not obs_list:
            return 0.0
        if len(obs_list) == 1:
            return obs_list[0].value
        weighted = [(o.value, self._decay_weight(o.observed_at)) for o in obs_list]
        total_w = sum(w for _, w in weighted)
        if total_w <= 0:
            return statistics.fmean([o.value for o in obs_list])
        return sum(v * w for v, w in weighted) / total_w

    _weighted_median = _weighted_mean

    @staticmethod
    def _extract_input_size_from_obs(obs: LatencyObservation) -> float:
        """Plan fix Stage 3 — single-source input_size
        extraction.  Uses ``ResourceProfileRegistry.extract_input_size``
        (same method already called in ``SignalService.record_observation``
        for consistency with per-component feature parsing — e.g.,
        residue count for protein sequences, etc.).

        Lazy import to avoid circular dependency (resource_profile
        imports nothing from interference but the canonical rule is
        single-direction).
        """
        try:
            from .resource_profile import ResourceProfileRegistry

            return _as_float(
                ResourceProfileRegistry.extract_input_size(
                    getattr(obs, "workload_features", {}) or {},
                    component=getattr(obs, "component", ""),
                )
            )
        except Exception:
            return 0.0

    def _bayesian_slowdown_posterior(
        self,
        prior_mean: float,
        obs_list: list[_Timestamped],
    ) -> tuple[float, float]:
        """Plan fix/11 (deviation fix) — Bayesian Normal-
        Normal conjugate posterior for slowdown GP.

        Prior: ``N(prior_mean, σ₀² = _slowdown_prior_variance)``
        Data: *obs_list* treated as samples with variance equal to
        the weighted sample variance (or 0 for N≤1).

        Posterior (pseudo-count form, α = _slowdown_prior_strength):
          - posterior_mean = (α·μ₀ + N·x̄) / (α + N)
          - posterior_var  = (α·σ₀² + N·s²) / (α + N)

        Behavior at key N values (defaults α=1, σ₀²=0.25 → σ₀=0.5):
          - N=0: (μ₀, σ₀)                 — pure prior, large σ
          - N=1: ((μ₀+x)/2, σ₀/√2)        — weighted avg with prior
          - N=10, varied data: close to (x̄, √s²·rough)  — empirical
          - N=10, identical data (s²=0): (shrunk mean, σ₀/√11)

        Matches Plan Fix 2 "N=0 posterior_std = sqrt(prior_variance)
         conservative" + Fix 3 "N=1 Bayesian weighted average"
        semantic.  Stage 1/2 initial landing returned (prior, 0.0) and
        (observation, 0.0) respectively — deviation from plan.
        """
        alpha = self._slowdown_prior_strength
        prior_var = self._slowdown_prior_variance
        N = len(obs_list)
        if N == 0:
            return (prior_mean, math.sqrt(max(0.0, prior_var)))
        sample_mean, sample_std = self._weighted_mean_std(obs_list)
        sample_var = sample_std**2
        denom = alpha + N
        posterior_mean = (alpha * prior_mean + N * sample_mean) / denom
        posterior_var = (alpha * prior_var + N * sample_var) / denom
        return (posterior_mean, math.sqrt(max(0.0, posterior_var)))

    def _weighted_mean_std(
        self,
        obs_list: list[_Timestamped],
    ) -> tuple[float, float]:
        """Plan fix — recency-weighted mean + std for
        slowdown posterior (μ + z·σ structure in D2 EFT).

        Plan Stage 2 (Fix 3) adopts mean/std in place of median/IQR
        because Plan L3946 pseudocode references
        ``posterior_mean(N=n_self+1)`` — an arithmetic-mean posterior
        summary, not median.  The accompanying std gives caller the
        GP-UCB-style ``μ + z·σ`` conservative quantile for low-N
        uncertainty handling (Srinivas et al. 2010).

        Weighting uses the same recency decay as ``_weighted_median``
        so both APIs share a consistent temporal model.  Returns
        ``(mean, std)`` in the same units as the observations (delta
        form for slowdowns).  For N=1, std=0.0 (degenerate variance).
        """
        if not obs_list:
            return 0.0, 0.0
        if len(obs_list) == 1:
            return obs_list[0].value, 0.0
        weighted_values = [
            (o.value, self._decay_weight(o.observed_at)) for o in obs_list
        ]
        total_w = sum(w for _, w in weighted_values)
        if total_w <= 0:
            vals = [o.value for o in obs_list]
            m = statistics.fmean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return m, s
        mean = sum(v * w for v, w in weighted_values) / total_w
        var = sum(w * (v - mean) ** 2 for v, w in weighted_values) / total_w
        std = math.sqrt(max(0.0, var))
        return mean, std

    def _observation_confidence(self, obs_list: list[_Timestamped]) -> float:
        """Compute confidence score (0.0–1.0) based on count and recency.

        - More observations → higher confidence (saturates at ~10 samples).
        - Recent observations → higher confidence.
        """
        if not obs_list:
            return 0.0
        n = len(obs_list)
        count_conf = n / (n + 5.0)
        avg_decay = sum(self._decay_weight(o.observed_at) for o in obs_list) / n
        return round(count_conf * avg_decay, 3)

    def _get_pairwise_record(
        self,
        comp_a: str,
        comp_b: str,
        fp: str = "",
    ) -> _CorrectionRecord | None:
        """Plan Phase B — fp-keyed pairwise record lookup with
        cross-fp fallback.  Tier 1: ``(comp_a, comp_b, fp)``; Tier 2:
        ``(comp_a, comp_b, "")``; Tier 3: any fp registered for this
        pair.  Returns ``None`` when no record exists at any tier.

        Note: ``defaultdict.get`` does NOT trigger the factory, so this
        helper is read-only — pairwise records are only created by the
        record-side ``self._pairwise[key]`` access.
        """
        if fp:
            rec = self._pairwise.get(self._pair_key(comp_a, comp_b, fp))
            if rec is not None:
                return rec
        rec = self._pairwise.get(self._pair_key(comp_a, comp_b, ""))
        if rec is not None:
            return rec
        for (a, b, _other_fp), record in self._pairwise.items():
            if a == comp_a and b == comp_b:
                return record
        return None

    def _iter_self_slowdown_obs(
        self,
        component: str,
        fp: str = "",
    ) -> list[_SelfSlowdownObs]:
        """Plan Phase B — fp-keyed self-slowdown observation
        retrieval with cross-fp fallback.  Tier 1: ``(component, fp)``;
        Tier 2: ``(component, "")``; Tier 3: aggregate from any fp for
        the component.  Returns the merged list (caller is responsible
        for any further validity / kernel weighting).
        """
        if fp:
            obs_list = self._self_slowdown_obs.get((component, fp))
            if obs_list:
                return list(obs_list)
        obs_list = self._self_slowdown_obs.get((component, ""))
        if obs_list:
            return list(obs_list)
        merged: list[_SelfSlowdownObs] = []
        for (comp, _other_fp), entries in self._self_slowdown_obs.items():
            if comp == component:
                merged.extend(entries)
        return merged

    def _iter_self_slowdown_obs_count(
        self,
        component: str,
        fp: str = "",
    ) -> int:
        """Plan Phase B — count helper for self-slowdown
        observations across the same fp-fallback chain as
        ``_iter_self_slowdown_obs``.  Returns the **first** non-zero
        tier rather than summing across tiers (matches WSJF
        first-match semantics)."""
        if fp:
            obs_list = self._self_slowdown_obs.get((component, fp))
            if obs_list:
                return len(obs_list)
        obs_list = self._self_slowdown_obs.get((component, ""))
        if obs_list:
            return len(obs_list)
        total = 0
        for (comp, _other_fp), entries in self._self_slowdown_obs.items():
            if comp == component:
                total += len(entries)
        return total

    def _has_solo_baseline(self, component: str, fp: str = "") -> bool:
        """ fix — existence-only equivalent of
        ``_get_solo_baseline_value`` for callers that only need a boolean
        (: ``is_self_mature``).  Bypasses ``_get_valid`` listcomp +
        ``_weighted_mean`` listcomp + decay_weight calculation entirely
        — first cutoff-valid observation triggers early-exit return True.

        3-tier fallback mirrors ``_get_solo_baseline_value``:
          Tier 1: ``(component, fp)`` exact.
          Tier 2: ``(component, "")`` cross-fp pool.
          Tier 3: any fp registered for the component.

        - py-spy hot path: try_evict path  evaluate_gpu ×
        evict_subset × GPU candidates  ``_get_solo_baseline_value``
          listcomp + decay_weight   → main loop CPU .
        Existence-only check  listcomp 0 + early-exit .
        """
        cutoff = time.monotonic() - self._max_age_sec
        if fp:
            for o in self._solo_baselines.get((component, fp), []):
                if o.observed_at >= cutoff:
                    return True
        for o in self._solo_baselines.get((component, ""), []):
            if o.observed_at >= cutoff:
                return True
        for raw_key, baselines in self._solo_baselines.items():
            comp, _other_fp = self._split_component_fp_key(raw_key)
            if comp != component:
                continue
            for o in baselines:
                if o.observed_at >= cutoff:
                    return True
        return False

    def _get_solo_baseline_value(
        self,
        component: str,
        fp: str = "",
    ) -> float | None:
        """Plan Phase B — fp-keyed solo baseline lookup with
        cross-fp fallback.  Tries ``(component, fp)`` first; if absent
        / empty / all-stale, falls back to ``(component, "")`` (legacy
        / cross-fp pool).  When neither has a valid observation,
        iterates every fp registered for the component and returns the
        first non-empty weighted median (mirrors WSJF Tier B fallback).
        Default ``fp = ""`` keeps backward compat for legacy callers
        — they automatically get the cross-fp pool.
        """
        if fp:
            valid = self._get_valid(self._solo_baselines.get((component, fp), []))
            if valid:
                return self._weighted_median(valid)
        valid = self._get_valid(self._solo_baselines.get((component, ""), []))
        if valid:
            return self._weighted_median(valid)
        for raw_key, baselines in self._solo_baselines.items():
            comp, _other_fp = self._split_component_fp_key(raw_key)
            if comp != component:
                continue
            valid = self._get_valid(baselines)
            if valid:
                return self._weighted_median(valid)
        return None

    def _get_solo_vram_value(
        self,
        component: str,
        fp: str = "",
    ) -> float | None:
        """Plan Phase B — fp-keyed solo VRAM baseline lookup.
        Same fallback chain as ``_get_solo_baseline_value``.

         fix — value cache (fix).  ``record_solo_vram``
          (component, fp) key invalidate.
        """
        cache_key = (component, fp)
        if cache_key in self._solo_vram_value_cache:
            return self._solo_vram_value_cache[cache_key]
        if fp:
            valid = self._get_valid(self._solo_vram.get((component, fp), []))
            if valid:
                result = self._weighted_median(valid)
                self._solo_vram_value_cache[cache_key] = result
                return result
        valid = self._get_valid(self._solo_vram.get((component, ""), []))
        if valid:
            result = self._weighted_median(valid)
            self._solo_vram_value_cache[cache_key] = result
            return result
        for raw_key, baselines in self._solo_vram.items():
            comp, _other_fp = self._split_component_fp_key(raw_key)
            if comp != component:
                continue
            valid = self._get_valid(baselines)
            if valid:
                result = self._weighted_median(valid)
                self._solo_vram_value_cache[cache_key] = result
                return result
        self._solo_vram_value_cache[cache_key] = None
        return None

    def _invalidate_solo_vram_value_cache(
        self,
        component: str,
        fp: str,
    ) -> None:
        """ fix — drop cached solo_vram value(s) on observation
        update.  Cross-fp pool key (``(component, "")``)   drop —
        Tier 2 fallback     .  ``component``   fp
        variant  drop (Tier 3 fallback  affected)."""
        if not self._solo_vram_value_cache:
            return
        stale = [k for k in self._solo_vram_value_cache if k[0] == component]
        for k in stale:
            del self._solo_vram_value_cache[k]


    def export_snapshot(self) -> dict[str, Any]:
        """Export current state as a JSON-serializable snapshot.

        Stores raw observation values (not monotonic timestamps) so the
        snapshot can be imported across process restarts.  On import,
        values are assigned fresh timestamps at the current monotonic
        time.  Plan Phase B — fp dimension preserved via the
        ``"comp|fp"`` key encoding for ``solo_baselines`` /
        ``solo_vram`` and ``"comp_a|comp_b|fp"`` for ``pairwise``.
        Pre-Phase-B snapshots (without ``|fp``) import cleanly into
        the cross-fp pool ``fp = ""`` (see ``import_snapshot``).
        """
        solo_bl: dict[str, list[float]] = {}
        for raw_key, vals in self._solo_baselines.items():
            comp, fp = self._split_component_fp_key(raw_key)
            valid = self._get_valid(vals)
            if valid:
                key = f"{comp}|{fp}" if fp else comp
                solo_bl[key] = [o.value for o in valid]

        solo_vr: dict[str, list[float]] = {}
        for raw_key, vals in self._solo_vram.items():
            comp, fp = self._split_component_fp_key(raw_key)
            valid = self._get_valid(vals)
            if valid:
                key = f"{comp}|{fp}" if fp else comp
                solo_vr[key] = [o.value for o in valid]

        pairwise: dict[str, dict[str, list[Any]]] = {}
        for (comp_a, comp_b, fp), record in self._pairwise.items():
            key = f"{comp_a}:{comp_b}|{fp}" if fp else f"{comp_a}:{comp_b}"
            entry: dict[str, list[Any]] = {}
            valid_s = self._get_valid(record.latency_slowdowns)
            if valid_s:
                entry["slowdowns"] = [
                    {
                        "value": item.value,
                        "identity": _query_payload(item.query),
                        "uncertainty": item.uncertainty,
                        "provenance": item.provenance,
                    }
                    for item in valid_s
                ]
            valid_v = self._get_valid(record.vram_overheads)
            if valid_v:
                entry["vram_overheads"] = [o.value for o in valid_v]
            valid_g = self._get_valid(record.gpu_util_shares)
            if valid_g:
                entry["gpu_util_shares"] = [o.value for o in valid_g]
            if entry:
                pairwise[key] = entry

        self_slowdown: dict[str, list[dict[str, Any]]] = {}
        for (comp, fp), obs_list in self._self_slowdown_obs.items():
            valid = self._get_valid(obs_list)
            if not valid:
                continue
            key = f"{comp}|{fp}" if fp else comp
            self_slowdown[key] = [
                {
                    "n_concurrency": _as_int(o.n_concurrency),
                    "delta": _as_float(o.delta),
                    "gpu_id": str(o.gpu_id or ""),
                    "identity": _query_payload(o.query),
                    "uncertainty": o.uncertainty,
                    "provenance": o.provenance,
                }
                for o in valid
            ]

        return {
            "schema": "interference_snapshot_v3_directed_identity",
            "evidence_epoch": self._evidence_epoch,
            "solo_baselines": solo_bl,
            "solo_vram": solo_vr,
            "pairwise": pairwise,
            "self_slowdown": self_slowdown,
        }

    def export_state(self) -> dict[str, Any]:
        """Export exact runtime state for completed-run bootstrap."""
        now = time.monotonic()

        def _ts_list(values: list[_Timestamped]) -> list[dict[str, Any]]:
            return [
                {
                    "value": _as_float(item.value),
                    "age_sec": max(0.0, now - _as_float(item.observed_at)),
                    "gpu_id": str(item.gpu_id or ""),
                    "identity": _query_payload(item.query),
                    "uncertainty": item.uncertainty,
                    "provenance": item.provenance,
                }
                for item in values
            ]

        return {
            "schema": "interference_registry_v3_directed_identity",
            "evidence_epoch": self._evidence_epoch,
            "config": {
                "min_empirical_samples": _as_int(self._min_empirical),
                "max_history": _as_int(self._max_history),
                "max_age_sec": _as_float(self._max_age_sec),
                "decay_half_life_sec": _as_float(self._decay_half_life),
                "maturity_observations_per_dim": _as_int(
                    self._maturity_observations_per_dim
                ),
                "pairwise_slowdown_prior": _as_float(self._pairwise_slowdown_prior),
                "self_slowdown_prior": _as_float(self._self_slowdown_prior),
                "self_interference_prior_coeff": _as_float(
                    self._self_interference_prior_coeff
                ),
                "slowdown_prior_variance": _as_float(self._slowdown_prior_variance),
                "slowdown_prior_strength": _as_float(self._slowdown_prior_strength),
                "self_slowdown_ls_N": _as_float(self._self_slowdown_ls_N),
                "class_interference": {
                    f"{a}|{b}": _as_float(v)
                    for (a, b), v in sorted(self._class_interference.items())
                },
            },
            "solo_baselines": {
                f"{comp}|{fp}": _ts_list(values)
                for (comp, fp), values in sorted(self._solo_baselines.items())
            },
            "solo_vram": {
                f"{comp}|{fp}": _ts_list(values)
                for (comp, fp), values in sorted(self._solo_vram.items())
            },
            "pairwise": {
                f"{a}|{b}|{fp}": {
                    "latency_slowdowns": _ts_list(record.latency_slowdowns),
                    "vram_overheads": _ts_list(record.vram_overheads),
                    "gpu_util_shares": _ts_list(record.gpu_util_shares),
                }
                for (a, b, fp), record in sorted(self._pairwise.items())
            },
            "self_slowdown": {
                f"{comp}|{fp}": [
                    {
                        "n_concurrency": _as_int(item.n_concurrency),
                        "delta": _as_float(item.delta),
                        "age_sec": max(0.0, now - _as_float(item.observed_at)),
                        "gpu_id": str(item.gpu_id or ""),
                        "identity": _query_payload(item.query),
                        "uncertainty": item.uncertainty,
                        "provenance": item.provenance,
                    }
                    for item in values
                ]
                for (comp, fp), values in sorted(self._self_slowdown_obs.items())
            },
            "pbbc": {
                "solo_evidence": [
                    {"component": comp, "config_fingerprint": fp}
                    for comp, fp in sorted(self._pbbc_solo_evidence)
                ],
                "self_evidence_counts": [
                    {
                        "component": comp,
                        "config_fingerprint": fp,
                        "count": _as_int(count),
                    }
                    for (comp, fp), count in sorted(
                        self._pbbc_self_evidence_counts.items()
                    )
                ],
                "self_evidence_gpu_counts": [
                    {
                        "component": comp,
                        "config_fingerprint": fp,
                        "gpu_id": gid,
                        "count": _as_int(count),
                    }
                    for (comp, fp, gid), count in sorted(
                        self._pbbc_self_evidence_gpu_counts.items()
                    )
                ],
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> dict[str, int]:
        """Replace exact runtime state from ``export_state``."""
        now = time.monotonic()
        directed_pairwise = str(data.get("schema") or "").startswith(
            ("interference_registry_v2_directed", "interference_registry_v3_directed")
        )

        def _split_key(raw: Any, expected: int) -> tuple[str, ...]:
            parts = str(raw or "").split("|")
            if len(parts) < expected:
                parts.extend([""] * (expected - len(parts)))
            return tuple(parts[:expected])

        def _load_ts_list(raw_items: Any) -> list[_Timestamped]:
            loaded: list[_Timestamped] = []
            for item in raw_items if isinstance(raw_items, list) else []:
                if not isinstance(item, Mapping):
                    continue
                try:
                    raw_value = item.get("value")
                    if raw_value is None:
                        continue
                    value = _as_float(raw_value)
                    age = max(0.0, _as_float(item.get("age_sec", 0.0) or 0.0))
                except Exception:
                    continue
                if age > self._max_age_sec:
                    continue
                loaded.append(
                    _Timestamped(
                        value=value,
                        observed_at=now - age,
                        gpu_id=str(item.get("gpu_id") or ""),
                        query=_query_from_payload(item.get("identity")),
                        uncertainty=max(
                            0.0, _as_float(item.get("uncertainty", 0.0) or 0.0)
                        ),
                        provenance=_as_provenance(item.get("provenance")),
                    )
                )
            return loaded[-self._max_history :]

        config = data.get("config") or {}
        if isinstance(config, Mapping):
            self._min_empirical = _as_int(
                config.get("min_empirical_samples", self._min_empirical)
                or self._min_empirical
            )
            self._max_history = _as_int(
                config.get("max_history", self._max_history) or self._max_history
            )
            self._max_age_sec = _as_float(
                config.get("max_age_sec", self._max_age_sec) or self._max_age_sec
            )
            self._decay_half_life = _as_float(
                config.get("decay_half_life_sec", self._decay_half_life)
                or self._decay_half_life
            )
            self._maturity_observations_per_dim = _as_int(
                config.get(
                    "maturity_observations_per_dim",
                    self._maturity_observations_per_dim,
                )
                or self._maturity_observations_per_dim
            )
            self._pairwise_slowdown_prior = _as_float(
                config.get("pairwise_slowdown_prior", self._pairwise_slowdown_prior)
                or self._pairwise_slowdown_prior
            )
            self._self_slowdown_prior = _as_float(
                config.get("self_slowdown_prior", self._self_slowdown_prior)
                or self._self_slowdown_prior
            )
            self._self_interference_prior_coeff = _as_float(
                config.get(
                    "self_interference_prior_coeff",
                    self._self_interference_prior_coeff,
                )
                or self._self_interference_prior_coeff
            )
            self._slowdown_prior_variance = _as_float(
                config.get("slowdown_prior_variance", self._slowdown_prior_variance)
                or self._slowdown_prior_variance
            )
            self._slowdown_prior_strength = _as_float(
                config.get("slowdown_prior_strength", self._slowdown_prior_strength)
                or self._slowdown_prior_strength
            )
            self._self_slowdown_ls_N = _as_float(
                config.get("self_slowdown_ls_N", self._self_slowdown_ls_N)
                or self._self_slowdown_ls_N
            )
            class_payload = config.get("class_interference") or {}
            if isinstance(class_payload, Mapping):
                self._class_interference = {}
                for raw_key, value in class_payload.items():
                    a, b = _split_key(raw_key, 2)
                    try:
                        self._class_interference[(a, b)] = _as_float(value)
                    except Exception:
                        continue

        self._solo_baselines = defaultdict(list)
        self._solo_vram = defaultdict(list)
        self._pairwise = defaultdict(_CorrectionRecord)
        self._self_slowdown_obs = defaultdict(list)
        self._pbbc_solo_evidence = set()
        self._pbbc_self_evidence_counts = defaultdict(int)
        self._pbbc_self_evidence_gpu_counts = defaultdict(int)
        self._self_slowdown_cache.clear()
        self._pairwise_slowdown_cache.clear()
        self._solo_vram_value_cache.clear()
        self._pairwise_slowdown_value_cache.clear()

        counts = {
            "solo_baselines": 0,
            "solo_vram": 0,
            "pairwise": 0,
            "self_slowdown": 0,
            "pbbc_solo": 0,
            "pbbc_self": 0,
        }

        for raw_key, values in (data.get("solo_baselines") or {}).items():
            comp, fp = _split_key(raw_key, 2)
            loaded = _load_ts_list(values)
            self._solo_baselines[(comp, fp)].extend(loaded)
            counts["solo_baselines"] += len(loaded)

        for raw_key, values in (data.get("solo_vram") or {}).items():
            comp, fp = _split_key(raw_key, 2)
            loaded = _load_ts_list(values)
            self._solo_vram[(comp, fp)].extend(loaded)
            counts["solo_vram"] += len(loaded)

        pairwise_payload = data.get("pairwise") if directed_pairwise else {}
        for raw_key, entry in (pairwise_payload or {}).items():
            if not isinstance(entry, Mapping):
                continue
            comp_a, comp_b, fp = _split_key(raw_key, 3)
            record = self._pairwise[(comp_a, comp_b, fp)]
            record.latency_slowdowns.extend(
                _load_ts_list(entry.get("latency_slowdowns"))
            )
            record.vram_overheads.extend(_load_ts_list(entry.get("vram_overheads")))
            record.gpu_util_shares.extend(_load_ts_list(entry.get("gpu_util_shares")))
            counts["pairwise"] += (
                len(record.latency_slowdowns)
                + len(record.vram_overheads)
                + len(record.gpu_util_shares)
            )

        for raw_key, values in (data.get("self_slowdown") or {}).items():
            comp, fp = _split_key(raw_key, 2)
            for item in values if isinstance(values, list) else []:
                if not isinstance(item, Mapping):
                    continue
                try:
                    n_concurrency = _as_int(item.get("n_concurrency") or 0)
                    raw_delta = item.get("delta")
                    if raw_delta is None:
                        continue
                    delta = _as_float(raw_delta)
                    age = max(0.0, _as_float(item.get("age_sec", 0.0) or 0.0))
                except Exception:
                    continue
                if n_concurrency < 2 or age > self._max_age_sec:
                    continue
                self._self_slowdown_obs[(comp, fp)].append(
                    _SelfSlowdownObs(
                        n_concurrency=n_concurrency,
                        delta=max(0.0, delta),
                        observed_at=now - age,
                        gpu_id=str(item.get("gpu_id") or ""),
                        query=_query_from_payload(item.get("identity")),
                        uncertainty=max(
                            0.0, _as_float(item.get("uncertainty", 0.0) or 0.0)
                        ),
                        provenance=_as_provenance(item.get("provenance")),
                    )
                )
                counts["self_slowdown"] += 1

        pbbc = data.get("pbbc") or {}
        if isinstance(pbbc, Mapping):
            for item in pbbc.get("solo_evidence") or []:
                if not isinstance(item, Mapping):
                    continue
                comp = str(item.get("component") or "")
                if not comp:
                    continue
                fp = str(item.get("config_fingerprint") or "")
                self._pbbc_solo_evidence.add((comp, fp))
                counts["pbbc_solo"] += 1
            for item in pbbc.get("self_evidence_counts") or []:
                if not isinstance(item, Mapping):
                    continue
                comp = str(item.get("component") or "")
                if not comp:
                    continue
                fp = str(item.get("config_fingerprint") or "")
                count = _as_int(item.get("count", 0) or 0)
                if count > 0:
                    self._pbbc_self_evidence_counts[(comp, fp)] = count
                    counts["pbbc_self"] += count
            for item in pbbc.get("self_evidence_gpu_counts") or []:
                if not isinstance(item, Mapping):
                    continue
                comp = str(item.get("component") or "")
                gid = str(item.get("gpu_id") or "")
                if not comp or not gid:
                    continue
                fp = str(item.get("config_fingerprint") or "")
                count = _as_int(item.get("count", 0) or 0)
                if count > 0:
                    self._pbbc_self_evidence_gpu_counts[(comp, fp, gid)] = count
        imported_epoch = _as_int(data.get("evidence_epoch", 0) or 0)
        self._evidence_epoch = max(self._evidence_epoch, imported_epoch)
        affected = (
            {component for component, _fp in self._solo_baselines}
            | {component for component, _fp in self._self_slowdown_obs}
            | {
                component
                for victim, interferer, _fp in self._pairwise
                for component in (victim, interferer)
            }
        )
        self._advance_evidence_epoch(*affected)
        return counts

    def import_snapshot(self, data: dict[str, Any]) -> dict[str, int]:
        """Import a previously exported snapshot into the registry.

        Values are assigned fresh monotonic timestamps so decay and
        eviction work normally from this point forward.  Existing
        observations (if any) are preserved — snapshot values are
        appended.  Plan Phase B — handles both legacy
        component-only keys (imported as cross-fp pool ``fp = ""``)
        and  ``"comp|fp"`` / ``"comp_a:comp_b|fp"`` encoding.

        Returns counts of imported items per category.
        """
        now = time.monotonic()
        directed_pairwise = str(data.get("schema") or "").startswith(
            ("interference_snapshot_v2_directed", "interference_snapshot_v3_directed")
        )
        counts = {
            "solo_baselines": 0,
            "solo_vram": 0,
            "pairwise": 0,
            "self_slowdown": 0,
        }

        for raw_key, values in (data.get("solo_baselines") or {}).items():
            comp, _, fp = str(raw_key).partition("|")
            for v in values if isinstance(values, list) else []:
                if isinstance(v, (int, float)) and v > 0:
                    self._append_timestamped(
                        self._solo_baselines[(comp, fp)],
                        _as_float(v),
                        now,
                    )
                    self._mark_pbbc_solo_evidence(comp, fp)
                    counts["solo_baselines"] += 1

        for raw_key, values in (data.get("solo_vram") or {}).items():
            comp, _, fp = str(raw_key).partition("|")
            for v in values if isinstance(values, list) else []:
                if isinstance(v, (int, float)) and v > 0:
                    self._append_timestamped(
                        self._solo_vram[(comp, fp)],
                        _as_float(v),
                        now,
                    )
                    counts["solo_vram"] += 1

        pairwise_payload = data.get("pairwise") if directed_pairwise else {}
        for raw_pair_key, entry in (pairwise_payload or {}).items():
            head, _, fp = str(raw_pair_key).partition("|")
            parts = head.split(":", 1)
            if len(parts) != 2:
                continue
            comp_a, comp_b = parts
            key = self._pair_key(comp_a, comp_b, fp)
            record = self._pairwise[key]
            for item in entry.get("slowdowns") or []:
                if isinstance(item, Mapping):
                    raw_value = item.get("value")
                    if not isinstance(raw_value, (int, float)):
                        continue
                    self._append_timestamped(
                        record.latency_slowdowns,
                        _as_float(raw_value),
                        now,
                        query=_query_from_payload(item.get("identity")),
                        provenance=_as_provenance(item.get("provenance")),
                    )
                    record.latency_slowdowns[-1].uncertainty = max(
                        0.0, _as_float(item.get("uncertainty", 0.0) or 0.0)
                    )
                    counts["pairwise"] += 1
                elif isinstance(item, (int, float)):
                    self._append_timestamped(
                        record.latency_slowdowns, _as_float(item), now
                    )
                    counts["pairwise"] += 1
            for v in entry.get("vram_overheads") or []:
                if isinstance(v, (int, float)):
                    self._append_timestamped(record.vram_overheads, _as_float(v), now)
                    counts["pairwise"] += 1
            for v in entry.get("gpu_util_shares") or []:
                if isinstance(v, (int, float)):
                    self._append_timestamped(record.gpu_util_shares, _as_float(v), now)
                    counts["pairwise"] += 1

        for raw_key, observations in (data.get("self_slowdown") or {}).items():
            comp, _, fp = str(raw_key).partition("|")
            if not comp:
                continue
            for item in observations if isinstance(observations, list) else []:
                if isinstance(item, Mapping):
                    try:
                        n_concurrency = _as_int(item.get("n_concurrency") or 0)
                        raw_delta = item.get("delta")
                        if raw_delta is None:
                            continue
                        delta = _as_float(raw_delta)
                    except Exception:
                        continue
                    gpu_id = str(item.get("gpu_id") or "")
                    query = _query_from_payload(item.get("identity"))
                    uncertainty = max(
                        0.0, _as_float(item.get("uncertainty", 0.0) or 0.0)
                    )
                    provenance = _as_provenance(item.get("provenance"))
                elif isinstance(item, (int, float)):
                    n_concurrency = 2
                    delta = _as_float(item)
                    gpu_id = ""
                    query = None
                    uncertainty = 0.0
                    provenance = "unknown"
                else:
                    continue
                if n_concurrency < 2:
                    continue
                self._self_slowdown_obs[(comp, fp)].append(
                    _SelfSlowdownObs(
                        n_concurrency=n_concurrency,
                        delta=max(0.0, delta),
                        observed_at=now,
                        gpu_id=gpu_id,
                        query=query,
                        uncertainty=uncertainty,
                        provenance=provenance,
                    )
                )
                self._mark_pbbc_self_evidence(comp, fp, gpu_id=gpu_id)
                counts["self_slowdown"] += 1
            observations_for_key = self._self_slowdown_obs[(comp, fp)]
            if len(observations_for_key) > self._max_history:
                del observations_for_key[
                    : len(observations_for_key) - self._max_history
                ]
            if observations_for_key:
                self._invalidate_self_slowdown_cache(comp)

        if any(counts.values()):
            self._pairwise_slowdown_cache.clear()
            self._pairwise_slowdown_value_cache.clear()
            imported_epoch = _as_int(data.get("evidence_epoch", 0) or 0)
            self._evidence_epoch = max(self._evidence_epoch, imported_epoch)
            affected = (
                {component for component, _fp in self._solo_baselines}
                | {component for component, _fp in self._self_slowdown_obs}
                | {
                    component
                    for victim, interferer, _fp in self._pairwise
                    for component in (victim, interferer)
                }
            )
            self._advance_evidence_epoch(*affected)
        return counts

    @staticmethod
    def _pair_key(
        victim: str,
        interferer: str,
        fp: str = "",
    ) -> tuple[str, str, str]:
        """Directed victim→interferer pairwise storage key."""
        return (str(victim), str(interferer), str(fp or ""))


__all__ = [
    "BALANCED",
    "COMPUTE_BOUND",
    "InterferencePrediction",
    "InterferenceRegistry",
    "MEMORY_BOUND",
    "UNKNOWN",
    "WorkloadClassifier",
    "WorkloadProfile",
]
