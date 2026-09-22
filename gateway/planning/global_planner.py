"""Global Planner — HEFT-based unified scheduling solver.

Produces fully-specified ``DispatchPlan`` for each task by combining:
- DAG topology + fan-out from ``PipelineDAG``
- GP predictions (latency, VRAM, interference) from ``SignalService``
- Campaign priority (FIFO or WSJF)
- Stochastic b-rank ordering (D1)
- Stochastic EFT placement (D2)
- Integrated pre-init scheduling (D3)
- GP-enhanced backfill admission (D4)

All parameters are data-driven (GP observations) — zero arbitrary constants.

Architecture:
  Global Planner  →  Reality Validator  →  Event Handler
  (this file)        (http_server.py)      (campaign_scheduler.py events)
"""

from __future__ import annotations

import logging
import math
import random
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Protocol

from .constraint_tracker import ConstraintTracker, PlanningExhausted
from .contracts import (
    ConstraintAssumptions,
    ConstraintViolation,
    DispatchPlan,
    PreInitAction,
)
from .scenario import SchedulingScenario

_LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _discover_gpu_models() -> dict[str, str]:
    """Discover GPU names once; failure keeps the existing coarse fallback."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return {
        index.strip(): model.strip()
        for line in result.stdout.splitlines()
        if "," in line
        for index, model in [line.split(",", 1)]
        if index.strip() and model.strip()
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default




def _active_entries_for_component(timeline: Any, component: str) -> list[Any]:
    getter = getattr(timeline, "active_entries_for_component", None)
    if callable(getter):
        entries = getter(component)
        return list(entries) if isinstance(entries, (list, tuple, set)) else []
    return [
        e
        for e in getattr(timeline, "active_entries", [])
        if getattr(e, "component", None) == component
    ]


class PriorityStrategy(Protocol):
    """Task ordering strategy — determines solve() loop iteration order."""

    def compute_rank(
        self,
        component: str,
        campaign_id: str,
        planner: GlobalPlanner,
    ) -> float: ...


class PlacementStrategy(Protocol):
    """GPU selection — evaluates each GPU for a given task."""

    def evaluate_gpu(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        extra_same_component_active: int,
        planner: GlobalPlanner,
        *,
        task_id: str = "",
    ) -> float | None: ...



class BackfillStrategy(Protocol):
    """Backfill admission — decides if backfill task should be admitted."""

    def should_admit(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        primary_deadline: float,
        planner: GlobalPlanner,
    ) -> bool: ...


class PreInitStrategy(Protocol):
    """Pre-init decision — determines when/where to pre-init downstream."""

    def plan_preinit(
        self,
        task_id: str,
        component: str,
        assigned_gpu: str,
        campaign_id: str,
        eft: float,
        planner: GlobalPlanner,
    ) -> list[PreInitAction]: ...


@dataclass
class CurrentSolvePlacements:
    """Same-solve placements not yet represented in the timeline.

    Planned occupancy counts predicted entries directly.  This shadow counter
    is still needed by PBBC: stale/lookahead predicted entries are ignored by
    the cold gate, but siblings already selected in this solve should count as
    imminent same-component occupancy.
    """

    counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
    expected_task_ids: frozenset[str] = field(default_factory=frozenset)
    remaining_task_ids: set[str] = field(default_factory=set)
    attempted_task_ids: set[str] = field(default_factory=set)
    completed_task_ids: set[str] = field(default_factory=set)

    @classmethod
    def for_task_ids(cls, task_ids: list[str]) -> CurrentSolvePlacements:
        normalized = [str(task_id) for task_id in task_ids if str(task_id)]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for task_id in normalized:
            if task_id in seen:
                duplicates.add(task_id)
            seen.add(task_id)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise PlanningExhausted(f"duplicate task_id in planning batch: {names}")
        expected = frozenset(normalized)
        return cls(expected_task_ids=expected, remaining_task_ids=set(expected))

    @property
    def remaining_task_count(self) -> int:
        return len(self.remaining_task_ids)

    def begin_task(self, task_id: str) -> None:
        if not self.expected_task_ids:
            return
        normalized = str(task_id)
        if normalized not in self.expected_task_ids:
            raise PlanningExhausted(
                f"task_id is not part of the planning batch: {normalized}"
            )
        if normalized not in self.remaining_task_ids:
            raise PlanningExhausted(
                f"task_id already attempted in planning batch: {normalized}"
            )
        self.remaining_task_ids.remove(normalized)
        self.attempted_task_ids.add(normalized)

    @staticmethod
    def _key(gpu_id: str, component: str, campaign_id: str) -> tuple[str, str, str]:
        return (str(gpu_id), str(component), str(campaign_id))

    def count_same(self, gpu_id: str, component: str, campaign_id: str) -> int:
        return int(self.counts.get(self._key(gpu_id, component, campaign_id), 0))

    def count_component(self, gpu_id: str, component: str) -> int:
        target_gpu = str(gpu_id)
        target_component = str(component)
        return sum(
            int(count)
            for (gid, comp, _cid), count in self.counts.items()
            if gid == target_gpu and comp == target_component
        )

    def record(
        self,
        gpu_id: str,
        component: str,
        campaign_id: str,
        *,
        task_id: str = "",
    ) -> None:
        normalized = str(task_id)
        if self.expected_task_ids:
            if normalized not in self.attempted_task_ids:
                raise PlanningExhausted(
                    f"task_id was not claimed from the planning batch: {normalized}"
                )
            if normalized in self.completed_task_ids:
                raise PlanningExhausted(
                    f"task_id already placed in planning batch: {normalized}"
                )
            self.completed_task_ids.add(normalized)
        key = self._key(gpu_id, component, campaign_id)
        self.counts[key] = int(self.counts.get(key, 0)) + 1


@dataclass(frozen=True)
class ReciprocalProjection:
    """Read-only incumbent-delay projection for one candidate GPU."""

    gpu_id: str
    timeline_state_version: int
    candidate_eft: float
    candidate_duration_sec: float
    candidate_base_duration_sec: float
    candidate_slowdown_sec: float
    uncertainty_z: float
    interference_drain_at: float
    primary_tail_before: float
    primary_tail_after: float
    global_makespan_before: float = 0.0
    global_makespan_after: float = 0.0
    incumbent_delays: tuple[tuple[str, float], ...] = ()
    incumbent_projected_latencies: tuple[tuple[str, float], ...] = ()
    self_delay_total_sec: float = 0.0
    pair_delay_total_sec: float = 0.0
    scoring_elapsed_ns: int = 0
    scanned_entries: int = 0
    temporal_segments: int = 0
    interference_lookups: int = 0


@dataclass(frozen=True)
class ReciprocalTaskOverlay:
    """Immutable task-owned state consumed by the canonical event sweep."""

    task_id: str
    execution_attempt_id: str
    component: str
    campaign_id: str
    gpu_id: str
    start_wall: float
    total_base_work_sec: float
    remaining_base_work_sec: float
    current_multiplier: float = 1.0
    last_accounted_mono: float | None = None
    evidence_epoch: int = 0
    is_init: bool = False
    is_predicted: bool = False
    config_fingerprint: str = ""
    input_fingerprint: str = ""
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""
    cancel_wall: float | None = None


@dataclass(frozen=True)
class ReciprocalMultiplierSegment:
    task_id: str
    start_mono: float
    end_mono: float
    multiplier: float


@dataclass(frozen=True)
class ReciprocalEvaluation:
    """Versioned, read-only result from one GPU-local reciprocal sweep."""

    gpu_id: str
    timeline_state_version: int
    evidence_epoch: int
    completion_wall_by_task: tuple[tuple[str, float], ...]
    multiplier_by_task: tuple[tuple[str, float], ...]
    segments: tuple[ReciprocalMultiplierSegment, ...]
    cancelled_task_ids: tuple[str, ...] = ()
    scanned_entries: int = 0
    interference_lookups: int = 0

    def completion_wall(self, task_id: str) -> float | None:
        return next(
            (value for key, value in self.completion_wall_by_task if key == task_id),
            None,
        )

    def multiplier(self, task_id: str) -> float:
        return next(
            (value for key, value in self.multiplier_by_task if key == task_id),
            1.0,
        )


@dataclass(frozen=True)
class ReciprocalDispatchPreparation:
    event_id: str
    task_id: str
    component: str
    gpu_id: str
    candidate_base_work_sec: float
    uncertainty_z: float
    candidate_cancel_at: float | None
    expected_timeline_version: int
    prepared_timeline_version: int
    version_retried: bool
    rollback_entries: tuple[Any, ...]
    preexisting_entry_ids: frozenset[int]
    duplicate: bool = False


@dataclass(frozen=True)
class _ReciprocalEntryRollback:
    entry: Any
    predicted_end_time: float
    total_base_work_sec: float | None
    remaining_base_work_sec: float | None
    last_accounted_mono: float | None
    current_multiplier: float
    evidence_epoch: int
    execution_attempt_id: str
    cancel_at: float | None


@dataclass(frozen=True)
class _AppliedReciprocalAttempt:
    preparation: ReciprocalDispatchPreparation
    entry_rollbacks: tuple[_ReciprocalEntryRollback, ...]


@dataclass
class PrimaryDeadlineTracker:
    """Solve-scoped projected primary budget.

    Tracks the earliest blocking completion for the current primary
    campaign across both already-running timeline entries and primary
    plans committed earlier in the same solve() pass.
    """

    primary_campaign_id: str = ""
    deadline_ts: float = float("inf")

    def current(self) -> float:
        return float(self.deadline_ts)

    def observe(self, campaign_id: str, predicted_end_time: float) -> None:
        if not self.primary_campaign_id:
            return
        if str(campaign_id) != str(self.primary_campaign_id):
            return
        try:
            predicted_end_time = float(predicted_end_time)
        except Exception:
            return
        if predicted_end_time > 0.0 and predicted_end_time < self.deadline_ts:
            self.deadline_ts = predicted_end_time


class CampaignStrategy(Protocol):
    """Campaign priority — determines which campaign is primary."""

    def select_primary(
        self,
        campaign_queues: dict[str, Any],
    ) -> Any | None: ...


class EvictionStrategy(Protocol):
    """Select a subset of candidate backfills to evict for a primary task.

    Responsibility:
      - Decide **which** backfills to evict when the primary cannot fit
        on any of its preferred GPUs due to capacity constraints.
      - Return (selected_subset, net_benefit) or (None, 0.0) if no
        eviction subset is viable.
      - Planner-internal only — Validator / Core Loop MUST NOT invoke.
    """

    def select_backfills_to_evict(
        self,
        task_info: Any,
        candidates: list[Any],
        cost: dict[str, float],
        urgency: float,
        eft_no_evict: float,
        planner: GlobalPlanner,
    ) -> tuple[list[Any] | None, float]: ...





class HEFTPriority:
    """HEFT b-rank ordering (D1 — Topcuoglu, Hariri & Wu, IEEE TPDS 13(3):260-274, 2002).

    Plan Integration JT row (fix) — the
    ``fanout_barrier_force`` ablation knob.  When True, every fan-out
    is scored with ``Φ⁻¹(α^{1/N})`` regardless of ``join_type``
    (pre- legacy behavior).  Default False matches plan +
    independent fan-out.

    Empirically strong on heterogeneous platforms (Topcuoglu 2002
    Table 3: 7% over CPOP, 14% over MH, 7% over DLS, 34% over LMT at
    β=1.0 computation-heterogeneity range).  No constant-factor
    approximation bound exists on heterogeneous platforms (Bleuse et
    al., IEEE TPDS 28(9):2689-2702, 2017 — modified HEFT empirically
    under-performs moldable 3/2+ε on benchmark instances; Canon et al.,
    IEEE TPDS 31(3):721-732, 2020 — online heterogeneous platform
    Ω(√(m/k)) lower bound as indirect evidence).  Graham (1966)'s
    (2-1/m) bound applies only to identical-machines list scheduling.

    Stochastic b-rank (Plan):
      independent fan-out (default) ⇒ μ + σ·Φ⁻¹(α)
      barrier collector (join_type=barrier) ⇒ μ + σ·Φ⁻¹(α^{1/N})
    """

    def __init__(self, *, fanout_barrier_force: bool = False) -> None:
        self._cache: dict[str, dict[str, float]] = {}
        self._cache_lock = threading.RLock()
        self._fanout_barrier_force = bool(fanout_barrier_force)

    def compute_rank(
        self,
        component: str,
        campaign_id: str,
        planner: GlobalPlanner,
    ) -> float:
        with self._cache_lock:
            cache = self._cache.get(campaign_id, {})
            if component in cache:
                return cache[component]

        cs = planner.campaign_scheduler
        cq = cs._campaign_queues.get(campaign_id)
        dag = cq.dag_context if cq else None
        if not dag or not dag.component_order:
            return 0.0

        rank = self._compute(component, campaign_id, planner, set())
        with self._cache_lock:
            self._cache.setdefault(campaign_id, {})[component] = rank
        return rank

    def _compute(
        self,
        component: str,
        campaign_id: str,
        planner: GlobalPlanner,
        visiting: set[str],
    ) -> float:
        if component in visiting:
            return 0.0
        visiting = visiting | {component}

        with self._cache_lock:
            cache = self._cache.get(campaign_id, {})
            if component in cache:
                return cache[component]

        cs = planner.campaign_scheduler
        cq = cs._campaign_queues.get(campaign_id)
        dag = cq.dag_context if cq else None
        if not dag:
            return 0.0

        avg_input = cq.avg_input_size(component) if cq else 0.0
        fp = planner._campaign_component_fingerprint(campaign_id, component) or ""
        cold_infer = (
            cs._cold_inference_latency_sec()
            if hasattr(cs, "_cold_inference_latency_sec")
            else 30.0
        )
        mu = cs._predict_latency(
            component,
            avg_input,
            "",
            config_fingerprint=fp,
        )
        if mu is None or mu <= 0.0:
            mu = cold_infer
        sigma = (
            cs._predict_latency_sigma(
                component,
                avg_input,
                "",
                config_fingerprint=fp,
            )
            or 0.0
        )

        alpha = self._get_alpha(cs, component, avg_input, fp)

        fan_out = (cq.effective_fan_out(component) if cq else None) or 1
        N = max(1, int(fan_out))
        join_type = (
            dag.join_type(component) if hasattr(dag, "join_type") else "independent"
        )
        force_barrier = getattr(self, "_fanout_barrier_force", False)
        effective_barrier = (join_type == "barrier") or force_barrier

        stage_dur = self._compute_stage_duration(
            mu,
            sigma,
            alpha,
            effective_barrier,
            N,
        )

        downstream = dag.downstream_map.get(component, [])
        if not downstream:
            rank = stage_dur
        else:
            rank = stage_dur + max(
                self._compute(d, campaign_id, planner, visiting) for d in downstream
            )

        with self._cache_lock:
            self._cache.setdefault(campaign_id, {})[component] = rank
        return rank

    def _compute_stage_duration(
        self,
        mu: float,
        sigma: float,
        alpha: float,
        effective_barrier: bool,
        N: int,
    ) -> float:
        """Stage duration α-quantile — Plan stochastic b-rank.

        Subclasses override this to swap stochastic ↔ deterministic
        formulations without duplicating the rest of ``_compute``.
        Default (HEFTPriority): adds σ-quantile padding per fan-out
        semantics (independent vs explicit barrier/collector).
        """
        if effective_barrier and N > 1 and sigma > 0 and alpha > 0:
            alpha_adj = alpha ** (1.0 / N)
            return mu + sigma * _norm_ppf(alpha_adj)
        if sigma > 0 and alpha > 0:
            return mu + sigma * _norm_ppf(alpha)
        return mu

    @staticmethod
    def _get_alpha(
        cs: Any,
        component: str,
        input_size: float,
        config_fingerprint: str = "",
    ) -> float:
        """Plan A — continuous α derived from GP posterior.

        Uses GP coefficient-of-variation (σ/μ) when available:
        - CV ≤ 0.1  (very confident)  → α = 0.95 (Φ⁻¹≈1.645)
        - CV ≤ 0.3  (confident)       → α = 0.90
        - CV ≤ 0.5  (moderate)        → α = 0.80
        - CV  > 0.5 (low confidence)  → α = 0.50 (μ-only, wide margin)

        Formula: α = clamp(1 − CV, 0.5, 0.95).  The categorical
        ``_get_confidence`` bucket remains as a secondary signal when
        the signal service cannot provide numeric posterior stats
        (e.g., unit-test stubs).
        """
        try:
            mu = (
                cs._predict_latency(
                    component,
                    input_size,
                    "",
                    config_fingerprint=config_fingerprint,
                )
                or 0.0
            )
            sigma = (
                cs._predict_latency_sigma(
                    component,
                    input_size,
                    "",
                    config_fingerprint=config_fingerprint,
                )
                or 0.0
            )
            if mu > 0 and sigma > 0:
                cv = sigma / mu
                alpha = 1.0 - cv
                return max(0.5, min(0.95, alpha))
        except (AttributeError, TypeError, ZeroDivisionError):
            pass
        try:
            confidence_str = cs._get_confidence(
                component,
                input_size,
                "",
                config_fingerprint=config_fingerprint,
            )
        except TypeError:
            confidence_str = cs._get_confidence(component, input_size, "")
        if confidence_str == "high":
            return 0.95
        if confidence_str == "medium":
            return 0.80
        return 0.50

    def invalidate(self, campaign_id: str | None = None) -> None:
        with self._cache_lock:
            if campaign_id:
                self._cache.pop(campaign_id, None)
            else:
                self._cache.clear()


class DeterministicHEFTPriority(HEFTPriority):
    """μ-only HEFT b-rank — proton-naive baseline.

    Drops the σ-quantile padding from the stage-duration formula,
    leaving ``b_rank = μ_lat + max(b_rank(downstream))`` (no Φ⁻¹ term,
    no barrier order-statistic correction).  Used as the "naive HEFT"
    ablation against the stochastic ``HEFTPriority`` default so the
    paper submission can isolate the stochastic enhancement
    contribution.

    Composition with fan-out semantics (independent / barrier) is
    inherited from ``HEFTPriority``: ``join_type=barrier`` still
    drives downstream traversal exactly as in the stochastic variant
    — only the σ-quantile padding is removed at the stage-duration
    helper.  Downstream traversal, fan-out detection, cycle guard,
    cache, and α resolution all stay unchanged.
    """

    def _compute_stage_duration(
        self,
        mu: float,
        sigma: float,
        alpha: float,
        effective_barrier: bool,
        N: int,
    ) -> float:
        return mu


class FIFOPriority:
    """Simple FIFO ordering — arrival time as priority."""

    def compute_rank(
        self,
        component: str,
        campaign_id: str,
        planner: GlobalPlanner,
    ) -> float:
        cq = planner.campaign_scheduler._campaign_queues.get(campaign_id)
        return -cq.arrival_time if cq else 0.0

    def invalidate(self, campaign_id: str | None = None) -> None:
        pass


class FIFOCampaign:
    """Strict campaign-FIFO — earliest arrival is primary.

     fix (): the only remaining ``CampaignStrategy``.
    The WSPT-inspired ``SmithRuleCampaign`` / ``WSJFCampaign`` were retired
    in  because (a) Plan fix already labelled them an
    "aging heuristic" with no formal bound on the makespan objective, and
    (b) they consumed ~12-15% of main-thread CPU through ``refresh_
    remaining_est`` on every drift cascade ( py-spy 50-sample
    profile).  See ``<docs>``  entry.
    """

    def select_primary(self, campaign_queues: dict[str, Any]) -> Any | None:
        active = [cq for cq in campaign_queues.values() if not cq.is_empty]
        if not active:
            return None
        return min(active, key=lambda cq: cq.arrival_time)


class GreedyEvictionStrategy:
    """Default eviction strategy — gate-aware greedy (fix).

    The primary cannot fit on any preferred GPU.  Strategy decides which
    subset of backfills to evict using **cheap gate-aware checks**, NOT
    full EFT recompute ( py-spy: full-EFT recompute per candidate
    held the event loop for minutes — fix tracker for ).

    Algorithm (matching user spec, ):

      1. Cost ascending sort.

      2. **VRAM-gated path** (per-GPU): for each preferred GPU where the
         primary's ``vram_budget_mb`` exceeds the GPU's
         ``available_vram_mb``, walk same-GPU backfills cheapest-first
         and accumulate ``predicted_vram_mb`` until the cumulative
         freed VRAM covers the shortfall.  The first GPU that achieves
         feasibility wins (cheapest cost).  No EFT recompute — just
         summing predicted VRAM.

      3. **Concurrency-gated path** (per-worker): if no GPU is VRAM-
         gated (or all VRAM gates fail), the block must be on
         worker concurrency (max_concurrency saturated).  Pick the
         cheapest backfill on the *same worker* (worker_name match,
         GPU-agnostic — workers may span multiple GPUs) and evict
         enough to free the slot the primary needs.  No feasibility
         recompute — pure counting.  If fewer same-worker backfills
         exist than slots needed, return None → PlanningExhausted.

    Returns:
      - (selected, 0.0) on success — net_benefit not computed in +
        (Greedy is feasibility-driven, not benefit-driven; net=0.0
        signals "feasibility achieved").
      - (None, 0.0) when neither gate path admits a viable subset.
    """

    requires_eft_no_evict = False

    def select_backfills_to_evict(
        self,
        task_info: Any,
        candidates: list[Any],
        cost: dict[str, float],
        urgency: float,
        eft_no_evict: float,
        planner: GlobalPlanner,
    ) -> tuple[list[Any] | None, float]:
        if not candidates:
            return None, 0.0
        ordered = sorted(
            candidates,
            key=lambda bf: cost.get(getattr(bf, "task_id", ""), 0.0),
        )

        preferred_gpus: list[str] = list(
            getattr(task_info, "preferred_gpu_ids", None) or []
        )
        vram_needed = float(getattr(task_info, "vram_budget_mb", 0) or 0)
        ram_needed = float(getattr(task_info, "ram_budget_mb", 0) or 0)
        scenario = planner.timelines

        if not preferred_gpus or (vram_needed <= 0 and ram_needed <= 0):
            return None, 0.0

        best_selected: list[Any] | None = None
        best_cost: float = math.inf
        duration = max(
            0.0,
            _safe_float(getattr(task_info, "required_duration_sec", 0.0)),
        )
        candidate_task_id = str(getattr(task_info, "task_id", "") or "")
        candidate_component = str(getattr(task_info, "component", "") or "")
        candidate_config = str(getattr(task_info, "config_fingerprint", "") or "")
        candidate_input_size = _safe_float(getattr(task_info, "input_size", 0.0))

        def _fits_after_masking(gpu_id: str, selected: list[Any]) -> bool:
            restored: list[tuple[Any, bool]] = []
            masking_supported = callable(getattr(scenario, "get", None))
            at_time = time.time()
            try:
                affected_timelines: set[Any] = set()
                for victim in selected:
                    previous = bool(getattr(victim, "is_evict_masked", False))
                    restored.append((victim, previous))
                    victim.is_evict_masked = True
                    timeline = (
                        scenario.get(str(getattr(victim, "gpu_id", "")))
                        if masking_supported
                        else None
                    )
                    if timeline is not None:
                        affected_timelines.add(timeline)
                for timeline in affected_timelines:
                    timeline._bump_state_version()
                earliest_fit = getattr(scenario, "earliest_dual_fit_time", None)
                if duration > 0 and callable(earliest_fit):
                    return (
                        earliest_fit(
                            gpu_id,
                            vram_needed,
                            ram_needed,
                            horizon_sec=120.0,
                            exclude_task_id=candidate_task_id,
                            required_duration_sec=duration,
                            candidate_component=candidate_component,
                            candidate_config_fingerprint=candidate_config,
                            candidate_input_size=candidate_input_size,
                        )
                        is not None
                    )
                try:
                    available_vram = scenario.available_vram_at(
                        gpu_id,
                        exclude_task_id=candidate_task_id,
                    )
                except TypeError:
                    available_vram = scenario.available_vram_at(gpu_id)
                try:
                    available_ram = scenario.available_host_ram_at(
                        exclude_task_id=candidate_task_id,
                    )
                except TypeError:
                    available_ram = scenario.available_host_ram_at()
                if not masking_supported:
                    for victim in selected:
                        reserved_at = getattr(victim, "reserved_vram_at", None)
                        available_vram += _safe_float(
                            reserved_at(at_time)
                            if callable(reserved_at)
                            else getattr(victim, "predicted_vram_mb", 0.0)
                        )
                        available_ram += _safe_float(
                            getattr(victim, "predicted_ram_mb", 0.0)
                        )
                return bool(
                    available_vram >= vram_needed and available_ram >= ram_needed
                )
            finally:
                restored_timelines: set[Any] = set()
                for victim, previous in restored:
                    victim.is_evict_masked = previous
                    timeline = (
                        scenario.get(str(getattr(victim, "gpu_id", "")))
                        if masking_supported
                        else None
                    )
                    if timeline is not None:
                        restored_timelines.add(timeline)
                for timeline in restored_timelines:
                    timeline._bump_state_version()

        for gpu_id in preferred_gpus:
            try:
                if _fits_after_masking(gpu_id, []):
                    continue
            except Exception:
                continue
            gpu_bfs = [
                bf for bf in ordered if str(getattr(bf, "gpu_id", "")) == str(gpu_id)
            ]
            selected: list[Any] = []
            for bf in gpu_bfs:
                selected.append(bf)
                if _fits_after_masking(gpu_id, selected):
                    subset_cost = sum(
                        cost.get(getattr(b, "task_id", ""), 0.0) for b in selected
                    )
                    if subset_cost < best_cost:
                        best_cost = subset_cost
                        best_selected = list(selected)
                    break
        if best_selected is not None:
            return best_selected, 0.0

        target_workers: set = set()
        for gpu_id in preferred_gpus:
            try:
                wname, _addr, _cold = planner._resolve_worker(
                    str(getattr(task_info, "component", "") or ""),
                    str(gpu_id),
                )
                if wname:
                    target_workers.add(wname)
            except Exception:
                continue
        needed_slots = 1
        for wname in target_workers:
            same_worker_bfs = [
                bf
                for bf in ordered
                if str(getattr(bf, "worker_name", "")) == str(wname)
            ]
            if len(same_worker_bfs) >= needed_slots:
                return same_worker_bfs[:needed_slots], 0.0

        return None, 0.0


class GreedyRamEvictionStrategy:
    """Host-RAM eviction strategy — cost-greedy, GPU-agnostic.

    Mirrors the VRAM greedy policy surface (cost-ascending cumulative
    relief) but on the correct host-global locality for RAM:

      1. Compute the primary's host RAM shortfall against the current
         projected host RAM availability.
      2. Sort eligible backfills by cost ascending.
      3. Evict the minimum-cost prefix whose cumulative
         ``predicted_ram_mb`` covers the shortfall.

    This strategy is intentionally planner-only.  Runtime idle host-RAM
    worker eviction remains owned by ``ResourceAdmissionTracker`` /
    ``WorkerSupervisor``.
    """

    requires_eft_no_evict = False

    def select_backfills_to_evict(
        self,
        task_info: Any,
        candidates: list[Any],
        cost: dict[str, float],
        urgency: float,
        eft_no_evict: float,
        planner: GlobalPlanner,
    ) -> tuple[list[Any] | None, float]:
        del urgency, eft_no_evict
        if not candidates:
            return None, 0.0
        ram_needed = float(getattr(task_info, "ram_budget_mb", 0.0) or 0.0)
        if ram_needed <= 0.0:
            return None, 0.0
        try:
            ram_avail = float(planner.timelines.available_host_ram_at() or 0.0)
        except Exception:
            ram_avail = 0.0
        ram_shortfall = max(0.0, ram_needed - ram_avail)
        if ram_shortfall <= 0.0:
            return None, 0.0
        ordered = sorted(
            candidates,
            key=lambda bf: cost.get(getattr(bf, "task_id", ""), 0.0),
        )
        freed_ram = 0.0
        selected: list[Any] = []
        for bf in ordered:
            predicted_ram = float(getattr(bf, "predicted_ram_mb", 0.0) or 0.0)
            if predicted_ram <= 0.0:
                continue
            selected.append(bf)
            freed_ram += predicted_ram
            if freed_ram >= ram_shortfall:
                return selected, 0.0
        return None, 0.0


class NoEviction:
    """No-op eviction — standalone no-preemption ablation.

    Returns ``(None, 0.0)`` unconditionally so the planner falls
    through to ``PlanningExhausted`` → Core Loop ``SKIP_THIS_CYCLE``
    → SchedulingSupervisor wake/retry.  Matches Slurm's default
    behaviour (running backfill jobs are not preempted by higher-
    priority arrivals; the new arrival waits its turn) and K8s's
    default scheduling (no preemption unless a PriorityClass with
    ``preemptionPolicy: PreemptLowerPriority`` is configured AND the
    arriving pod has higher priority).

    Not used by the clean proton-naive runtime config; proton-naive now
    shares PROTON's greedy eviction path and only changes mean-only /
    no-safeguard scheduling knobs.
    """

    requires_eft_no_evict = False

    def select_backfills_to_evict(
        self,
        task_info: Any,
        candidates: list[Any],
        cost: dict[str, float],
        urgency: float,
        eft_no_evict: float,
        planner: GlobalPlanner,
    ) -> tuple[list[Any] | None, float]:
        return None, 0.0


class NoRamEviction:
    """No-op host RAM eviction.

    Preserves the current naive behavior where host RAM pressure is
    handled only by runtime admission / idle worker relief, not by
    planner-owned running-backfill eviction.
    """

    requires_eft_no_evict = False

    def select_backfills_to_evict(
        self,
        task_info: Any,
        candidates: list[Any],
        cost: dict[str, float],
        urgency: float,
        eft_no_evict: float,
        planner: GlobalPlanner,
    ) -> tuple[list[Any] | None, float]:
        return None, 0.0


class StochasticEFTPlacement:
    """Stochastic EFT placement (D2)."""

    estimator_basis = "ucb"

    def evaluate_gpu(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        extra_same_component_active: int,
        planner: GlobalPlanner,
        *,
        task_id: str = "",
        timelines: Any = None,
        fixed_now: float | None = None,
    ) -> float | None:
        """Stochastic EFT per D2 — interference enters via ``_interference_cost``
        (`Σ (sd - 1) × overlap`) below.  No ``_would_slow_primary`` gate here
        (fix — HEFT min-EFT rule naturally avoids high-interference
        GPUs, a separate hard admission threshold would double-optimize and
        conflict with HEFT).
        """
        cs = planner.campaign_scheduler
        timelines = cs._timelines if timelines is None else timelines

        tl = timelines.get(gpu_id)
        if not tl:
            return None

        fp = (
            planner._campaign_component_fingerprint(
                campaign_id,
                component,
            )
            or ""
        )

        if task_id and planner.dynamic_batch_blocked(task_id, gpu_id):
            return None
        batch_profile = (
            planner.dynamic_batch_profile(task_id, gpu_id) if task_id else None
        )
        use_upper_resources = self._use_resource_upper_bounds()
        if batch_profile is not None:
            mu = float(batch_profile.latency_mean)
            sigma = float(batch_profile.latency_sigma)
            vram = float(
                batch_profile.vram_upper
                if use_upper_resources
                else batch_profile.vram_mean
            )
            ram = float(
                batch_profile.ram_upper
                if use_upper_resources
                else batch_profile.ram_mean
            )
        else:
            cold_infer = (
                cs._cold_inference_latency_sec()
                if hasattr(cs, "_cold_inference_latency_sec")
                else 30.0
            )
            mu = cs._predict_latency(
                component,
                input_size,
                gpu_id,
                config_fingerprint=fp,
            )
            if mu is None or mu <= 0.0:
                mu = cold_infer
            sigma = (
                cs._predict_latency_sigma(
                    component,
                    input_size,
                    gpu_id,
                    config_fingerprint=fp,
                )
                or 0.0
            )
            vram = cs._predict_vram(
                component,
                input_size,
                gpu_id,
                use_upper=use_upper_resources,
                config_fingerprint=fp,
            )
            ram = cs._predict_ram(
                component,
                input_size,
                gpu_id,
                use_upper=use_upper_resources,
                config_fingerprint=fp,
            )
            vram = float(vram) if isinstance(vram, (int, float)) else 0.0
            ram = float(ram) if isinstance(ram, (int, float)) else 0.0

        try:
            feasible = planner.is_gpu_feasible_for_task(
                gpu_id=gpu_id,
                component=component,
                worker_name="",
                vram_budget_mb=int(max(0.0, vram)),
            )
        except (AttributeError, KeyError) as exc:
            _LOG.warning(
                "[placement] is_gpu_feasible_for_task raised %s — proceeding",
                type(exc).__name__,
            )
            feasible = True
        if not feasible:
            return None

        alpha = HEFTPriority._get_alpha(
            cs,
            component,
            input_size,
            fp,
        )
        uncertainty_z = self._uncertainty_z(alpha)
        exec_time = self._compute_exec_time(mu, sigma, alpha)

        sup = getattr(cs, "_supervisor", None)
        warm = True
        if sup:
            resolved = planner._resolve_worker(component, gpu_id)
            warm = bool(
                isinstance(resolved, tuple)
                and len(resolved) == 3
                and resolved[0]
                and resolved[1]
                and not resolved[2]
            )
        init_cost = 0.0
        if sup and not warm:
            init_cost = cs._get_init_latency(component, gpu_id)
        now = (
            fixed_now
            if fixed_now is not None
            else getattr(planner, "_solve_now_wall", None)
        )
        if not isinstance(now, (int, float)) or not math.isfinite(float(now)):
            now = time.time()
        else:
            now = float(now)
        fixed_start = fixed_now is not None or bool(
            getattr(self, "requires_fixed_now_start", False)
        )
        fit_time = self._fit_start(
            timelines,
            gpu_id,
            vram,
            ram,
            exec_time,
            task_id=task_id,
            component=component,
            fp=fp,
            input_size=input_size,
            start=now + init_cost,
            now=now,
            fixed=fixed_start,
        )
        if fit_time is None:
            return None
        est = max(fit_time - init_cost, now)

        intf_reg = (
            cs._signal_service.interference_registry if cs._signal_service else None
        )

        sup = getattr(cs, "_supervisor", None)
        gp_maturity_threshold = int(
            getattr(sup, "gp_maturity_min_observations", 10),
        )

        _now_ts = time.time()

        def _is_current_task_predicted(e: Any) -> bool:
            return (
                bool(task_id)
                and str(getattr(e, "task_id", "") or "") == str(task_id)
                and bool(getattr(e, "is_predicted", False))
            )

        def _is_stale_pure_prediction(e: Any) -> bool:
            return (
                bool(getattr(e, "is_predicted", False))
                and not bool(getattr(e, "is_dispatching", False))
                and float(
                    getattr(e, "predicted_end_time", float("inf")) or float("inf")
                )
                < _now_ts
            )

        def _same_component_gate_entry(e: Any) -> bool:
            return (
                e.component == component
                and not getattr(e, "is_init", False)
                and not getattr(e, "is_evict_masked", False)
                and not _is_current_task_predicted(e)
                and not _is_stale_pure_prediction(e)
            )

        def _actual_same_component(e: Any) -> bool:
            return _same_component_gate_entry(e) and (
                not getattr(e, "is_predicted", False)
                or getattr(e, "is_dispatching", False)
            )

        def _projected_same_component(e: Any) -> bool:
            return (
                _same_component_gate_entry(e)
                and getattr(e, "is_predicted", False)
                and not getattr(e, "is_dispatching", False)
            )

        def _real_pending_dispatch_prediction(e: Any) -> bool:
            if not _projected_same_component(e):
                return False
            if getattr(e, "is_init", False):
                return False
            predicted_task_id = str(getattr(e, "task_id", "") or "")
            if not predicted_task_id:
                return False
            return not predicted_task_id.startswith(
                ("__lookahead_", "__reproject_", "__preinit_")
            )

        component_active_entries = _active_entries_for_component(tl, component)
        actual_self_count = sum(
            1 for e in component_active_entries if _actual_same_component(e)
        )
        timeline_projected_self_count = sum(
            1 for e in component_active_entries if _projected_same_component(e)
        )
        pending_dispatch_self_count = sum(
            1 for e in component_active_entries if _real_pending_dispatch_prediction(e)
        )
        try:
            same_solve_self_count = max(
                0,
                int(extra_same_component_active or 0),
            )
        except Exception:
            same_solve_self_count = 0
        projected_self_count = int(timeline_projected_self_count)
        shadow_projected_self_count = max(
            0,
            int(same_solve_self_count) - int(timeline_projected_self_count),
        )
        pbbc_pending_self_count = max(
            int(same_solve_self_count),
            int(pending_dispatch_self_count),
        )
        actual_post_count = actual_self_count + pbbc_pending_self_count + 1
        effective_post_count = actual_self_count + projected_self_count + 1

        self_obs_count = 0
        self_mature = False
        self_solo_seen = False
        self_solo_evidence_known = False
        pbbc_self_obs_count = 0
        pbbc_solo_seen = False
        if intf_reg is not None:
            try:
                exact_pbbc_fp = bool(str(fp or ""))
                try:
                    self_obs_count = intf_reg.get_self_observation_count(
                        component,
                        fp=fp,
                        exact_fp=exact_pbbc_fp,
                    )
                except TypeError:
                    self_obs_count = intf_reg.get_self_observation_count(
                        component,
                        fp=fp,
                    )
                has_solo_baseline = getattr(
                    intf_reg,
                    "has_solo_baseline",
                    None,
                )
                if callable(has_solo_baseline):
                    self_solo_evidence_known = True
                    try:
                        self_solo_seen = bool(
                            has_solo_baseline(
                                component,
                                fp=fp,
                                exact_fp=exact_pbbc_fp,
                            ),
                        )
                    except TypeError:
                        self_solo_seen = bool(
                            has_solo_baseline(component, fp=fp),
                        )
                else:
                    get_solo_baseline = getattr(
                        intf_reg,
                        "get_solo_baseline",
                        None,
                    )
                    if callable(get_solo_baseline):
                        self_solo_evidence_known = True
                        self_solo_seen = get_solo_baseline(component, fp=fp) is not None
                self_mature = intf_reg.is_self_mature(
                    component,
                    observations_per_dim=gp_maturity_threshold,
                    gpu_id=gpu_id,
                    fp=fp,
                )
                if self_solo_evidence_known:
                    self_mature = bool(
                        self_mature and self_solo_seen and self_obs_count >= 1
                    )
                pbbc_self_obs_count = self_obs_count
                pbbc_solo_seen = self_solo_seen
                try:
                    pbbc_self_obs_count = intf_reg.get_self_observation_count(
                        component,
                        fp=fp,
                        exact_fp=exact_pbbc_fp,
                        include_expired=True,
                    )
                except TypeError:
                    pbbc_self_obs_count = self_obs_count
                if callable(has_solo_baseline):
                    try:
                        pbbc_solo_seen = bool(
                            has_solo_baseline(
                                component,
                                fp=fp,
                                exact_fp=exact_pbbc_fp,
                                include_expired=True,
                            ),
                        )
                    except TypeError:
                        pbbc_solo_seen = self_solo_seen
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "self_maturity = False",
                    exc_info=True,
                )
                self_obs_count = 0
                self_mature = False
                self_solo_seen = False
                self_solo_evidence_known = False
                pbbc_self_obs_count = 0
                pbbc_solo_seen = False

        if _LOG.isEnabledFor(logging.DEBUG) and (
            effective_post_count > 1 or (tl and tl.active_entries)
        ):
            _entry_summary = [
                f"{e.task_id[:8]}({e.component},is_pred={e.is_predicted},"
                f"is_disp={e.is_dispatching},was_pri={getattr(e, 'was_primary_at_dispatch', False)})"
                for e in (tl.active_entries if tl else [])
            ]
            _LOG.debug(
                "[-gate-entry] component=%s gpu=%s actual_post_count=%d "
                "projected_n=%d effective_n=%d self_mature=%s entries=%s",
                component,
                gpu_id,
                actual_post_count,
                projected_self_count,
                effective_post_count,
                self_mature,
                _entry_summary,
            )

        _naive_disable_self_int = bool(
            getattr(sup, "disable_self_interference_gate", False)
        )
        _naive_disable_pbbc = bool(getattr(sup, "disable_pbbc_gate", False))
        _naive_disable_slo = bool(getattr(sup, "disable_primary_slo_protection", False))
        profiling_runtime = getattr(sup, "profiling_runtime", {}) or {}
        _force_single_self_concurrency = getattr(
            sup, "force_single_self_concurrency_gate", False
        ) is True or (
            isinstance(profiling_runtime, dict)
            and profiling_runtime.get("force_single_self_concurrency_gate") is True
        )
        if _force_single_self_concurrency and actual_post_count > 1:
            planner._log_info_throttled(
                ("force-single-self-concurrency", component, gpu_id, actual_post_count),
                "[force-single-self-concurrency] INFEASIBLE: component=%s "
                "gpu=%s actual_n=%d projected_n=%d effective_n=%d max=1",
                component,
                gpu_id,
                actual_post_count,
                projected_self_count,
                effective_post_count,
            )
            return None

        if not self_mature and actual_post_count > 1:
            if getattr(planner, "_cached_primary_id_set", False):
                current_primary_id = getattr(planner, "_cached_primary_id", None)
            else:
                current_primary_cq = cs._primary_campaign()
                current_primary_id = (
                    current_primary_cq.campaign_id if current_primary_cq else None
                )
            has_active_primary_on_worker = False
            try:
                for e in component_active_entries:
                    if not _actual_same_component(e):
                        continue
                    if (
                        current_primary_id is not None
                        and e.campaign_id == current_primary_id
                    ):
                        has_active_primary_on_worker = True
                        break
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "has_active_primary_on_worker = False",
                    exc_info=True,
                )
                has_active_primary_on_worker = False

            if has_active_primary_on_worker and not _naive_disable_slo:
                planner._log_info_throttled(
                    ("primary-protect", component, gpu_id, actual_post_count),
                    "[-primary-protect] INFEASIBLE: component=%s "
                    "gpu=%s actual_n=%d projected_n=%d effective_n=%d "
                    "(active primary on worker, self_obs=%d)",
                    component,
                    gpu_id,
                    actual_post_count,
                    projected_self_count,
                    effective_post_count,
                    self_obs_count,
                )
                return None

            pbbc_phase = "disabled"
            pbbc_max_n: int | None = None
            if _naive_disable_self_int:
                pass
            else:
                if not _naive_disable_pbbc and not pbbc_solo_seen:
                    pbbc_max_n = int(
                        getattr(
                            sup,
                            "self_concurrency_max_primary_cold_start",
                            1,
                        ),
                    )
                    pbbc_phase = "await_solo"
                elif pbbc_self_obs_count < 1:
                    pbbc_max_n = int(
                        getattr(
                            sup,
                            "self_concurrency_max_backfill_cold_start",
                            2,
                        ),
                    )
                    pbbc_phase = "await_2way"
                else:
                    pbbc_phase = "observed_2way"
            if (
                not _naive_disable_self_int
                and pbbc_max_n is not None
                and actual_post_count > pbbc_max_n
            ):
                planner._log_info_throttled(
                    (
                        "self-gate",
                        component,
                        gpu_id,
                        actual_post_count,
                        pbbc_max_n,
                        pbbc_phase,
                    ),
                    "[-self-gate] INFEASIBLE: component=%s gpu=%s "
                    "actual_n=%d projected_n=%d effective_n=%d max=%d "
                    "(pbbc_phase=%s, solo_seen=%s, self_obs=%d, "
                    "fresh_solo=%s, fresh_self_obs=%d)",
                    component,
                    gpu_id,
                    actual_post_count,
                    projected_self_count,
                    effective_post_count,
                    pbbc_max_n,
                    pbbc_phase,
                    pbbc_solo_seen,
                    pbbc_self_obs_count,
                    self_solo_seen,
                    self_obs_count,
                )
                return None

        if (
            not _naive_disable_self_int
            and self_mature
            and effective_post_count >= 2
            and intf_reg is not None
        ):
            safety_factor = float(
                getattr(
                    sup,
                    "concurrency_gate_safety_factor",
                    1.0,
                )
            )
            existing_planned_n = int(actual_self_count + projected_self_count)
            candidate_n = int(effective_post_count)

            max_cycle_candidate_n = candidate_n
            try:
                for other_gpu_id in getattr(timelines, "gpu_ids", []):
                    other_tl = timelines.get(str(other_gpu_id))
                    other_component_entries = _active_entries_for_component(
                        other_tl,
                        component,
                    )
                    other_actual = sum(
                        1 for e in other_component_entries if _actual_same_component(e)
                    )
                    other_projected = sum(
                        1
                        for e in other_component_entries
                        if _projected_same_component(e)
                    )
                    max_cycle_candidate_n = max(
                        max_cycle_candidate_n,
                        int(other_actual + other_projected + 1),
                    )
            except Exception:
                _LOG.warning(
                    "[-self-concurrency-gate] failed to compute "
                    "cycle max-N component=%s fp=%s",
                    component,
                    fp,
                    exc_info=True,
                )

            score_cache: dict[
                int, tuple[bool, float, float, float, float, float] | None
            ] = {}

            def _score_self_n(
                n_concurrent: int,
            ) -> tuple[bool, float, float, float, float, float] | None:
                if int(n_concurrent) in score_cache:
                    return score_cache[int(n_concurrent)]
                if n_concurrent <= 1:
                    score_cache[int(n_concurrent)] = (
                        True,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    return score_cache[int(n_concurrent)]
                sd_tuple_gate: tuple[float, float] | None = None
                try:
                    sd_tuple_gate = intf_reg.get_self_slowdown_if_mature(
                        component,
                        n_concurrent=n_concurrent,
                        fp=fp,
                    )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed an exception; body=%s",
                        __name__,
                        "sd_tuple_gate = None",
                        exc_info=True,
                    )
                    sd_tuple_gate = None
                if sd_tuple_gate is None:
                    score_cache[int(n_concurrent)] = None
                    return None
                sd_mean_d, sd_std_d = sd_tuple_gate
                crossover = float(n_concurrent - 1) * safety_factor
                gate_value = sd_mean_d
                efficiency = (1.0 + max(0.0, sd_mean_d)) / float(n_concurrent)
                score_cache[int(n_concurrent)] = (
                    gate_value < crossover,
                    efficiency,
                    sd_mean_d,
                    sd_std_d,
                    gate_value,
                    crossover,
                )
                return score_cache[int(n_concurrent)]

            cache_key = (
                component,
                str(fp or ""),
                int(max_cycle_candidate_n),
                round(float(safety_factor), 6),
            )
            cached_best = getattr(
                planner,
                "_self_concurrency_best_n_cache",
                {},
            ).get(cache_key)
            if cached_best is not None:
                best_n, best_efficiency = cached_best
            else:
                best_n = 1
                best_efficiency = 1.0
                current_search_n = 1
                current_search_efficiency = 1.0
                while current_search_n < max_cycle_candidate_n:
                    next_n = current_search_n + 1
                    next_score = _score_self_n(next_n)
                    if next_score is None:
                        break
                    (
                        next_is_viable,
                        next_efficiency,
                        _next_mean,
                        _next_std,
                        _next_lower,
                        _next_crossover,
                    ) = next_score
                    if not next_is_viable:
                        break
                    if next_efficiency < current_search_efficiency - 1e-9:
                        best_n = next_n
                        best_efficiency = next_efficiency
                        current_search_n = next_n
                        current_search_efficiency = next_efficiency
                        continue
                    break
                planner._self_concurrency_best_n_cache[cache_key] = (
                    best_n,
                    best_efficiency,
                )

            current_score = _score_self_n(candidate_n)
            if current_score is None:
                current_is_viable = False
                current_efficiency = float("inf")
                current_mean = 0.0
                current_std = 0.0
                current_gate_value = float("inf")
                current_crossover = float(candidate_n - 1) * safety_factor
            else:
                (
                    current_is_viable,
                    current_efficiency,
                    current_mean,
                    current_std,
                    current_gate_value,
                    current_crossover,
                ) = current_score

            if (not current_is_viable) or best_n is None or candidate_n > best_n:
                pruned_predicted = 0
                if best_n is not None and best_n < existing_planned_n:
                    timeline_keep_count = max(
                        0,
                        int(best_n)
                        - int(actual_self_count)
                        - int(shadow_projected_self_count),
                    )
                    drop_order_task_ids: list[str] = []
                    try:
                        alpha = planner._get_confidence_alpha()
                        predicted_candidates = [
                            e
                            for e in component_active_entries
                            if _projected_same_component(e)
                        ]
                        cost_by_task = {
                            str(
                                getattr(e, "task_id", "") or ""
                            ): planner._compute_eviction_cost(e, alpha)
                            for e in predicted_candidates
                        }
                        drop_order_task_ids = [
                            str(getattr(e, "task_id", "") or "")
                            for e in sorted(
                                predicted_candidates,
                                key=lambda entry: (
                                    -float(getattr(entry, "start_time", 0.0) or 0.0),
                                    cost_by_task.get(
                                        str(getattr(entry, "task_id", "") or ""),
                                        0.0,
                                    ),
                                ),
                            )
                        ]
                    except Exception:
                        _LOG.warning(
                            "[-self-concurrency-gate] greedy prune order "
                            "computation failed component=%s gpu=%s",
                            component,
                            gpu_id,
                            exc_info=True,
                        )
                    try:
                        prune_backlog = getattr(
                            cs._timelines,
                            "prune_predicted_component_backlog",
                            None,
                        )
                        if callable(prune_backlog):
                            pruned_predicted = _safe_int(
                                prune_backlog(
                                    component,
                                    gpu_id,
                                    keep_count=timeline_keep_count,
                                    exclude_task_id=task_id,
                                    drop_order_task_ids=drop_order_task_ids,
                                )
                            )
                    except Exception:
                        _LOG.warning(
                            "[-self-concurrency-gate] predicted backlog "
                            "prune failed component=%s gpu=%s best_n=%s "
                            "existing_n=%s",
                            component,
                            gpu_id,
                            best_n,
                            existing_planned_n,
                            exc_info=True,
                        )
                planner._log_info_throttled(
                    ("self-concurrency-gate", component, gpu_id, candidate_n),
                    "[-self-concurrency-gate] INFEASIBLE: component=%s "
                    "gpu=%s actual_n=%d projected_n=%d effective_n=%d "
                    "existing_n=%d best_n=%d cycle_max_n=%d "
                    "pruned_predicted=%d "
                    "current_eff=%.3f best_eff=%.3f "
                    "current_gate_mu=%.3f (μ=%.3f, σ=%.3f) "
                    "crossover=%.3f self_obs=%d",
                    component,
                    gpu_id,
                    actual_post_count,
                    projected_self_count,
                    candidate_n,
                    existing_planned_n,
                    best_n or 0,
                    max_cycle_candidate_n,
                    pruned_predicted,
                    current_efficiency,
                    best_efficiency,
                    current_gate_value,
                    current_mean,
                    current_std,
                    current_crossover,
                    self_obs_count,
                )
                return None

        slowdown = 0.0
        _naive_disable_slowdown_cost = bool(
            getattr(sup, "disable_slowdown_cost", False)
        ) or bool(planner.reciprocal_interference_correction)
        active_entries = tl.active_entries if not _naive_disable_slowdown_cost else []
        same_component_cost_entries = (
            component_active_entries if not _naive_disable_slowdown_cost else []
        )
        candidate_base_start = est + init_cost
        same_overlap_intervals = [
            interval
            for e in same_component_cost_entries
            if not getattr(e, "is_init", False)
            and not getattr(e, "is_evict_masked", False)
            and not _is_current_task_predicted(e)
            and not _is_stale_predicted_entry(e, now)
            if (
                interval := _base_work_overlap_interval(
                    e,
                    candidate_base_start,
                    exec_time,
                )
            )
            is not None
        ]
        slowdown += _candidate_self_slowdown(
            intf_reg,
            component,
            same_overlap_intervals,
            fp=fp,
            uncertainty_z=uncertainty_z,
        )

        for e in active_entries:
            if e.component == component:
                continue
            if getattr(e, "is_init", False):
                continue
            if getattr(e, "is_evict_masked", False):
                continue
            if _is_stale_predicted_entry(e, now):
                continue
            pair_delta = _effective_pair_delta(
                intf_reg,
                component,
                e.component,
                fp=fp,
                uncertainty_z=uncertainty_z,
            )
            overlap = _base_work_overlap(e, candidate_base_start, exec_time)
            slowdown += pair_delta * overlap

        final_duration = max(0.0, exec_time + slowdown)
        if final_duration > exec_time:
            final_fit_time = self._fit_start(
                timelines,
                gpu_id,
                vram,
                ram,
                final_duration,
                task_id=task_id,
                component=component,
                fp=fp,
                input_size=input_size,
                start=now + init_cost,
                now=now,
                fixed=fixed_start,
            )
            if final_fit_time is None:
                return None
            est = max(final_fit_time - init_cost, now)
        eft = est + final_duration + init_cost
        if task_id:
            planner._record_candidate_eft_detail(
                str(task_id),
                str(gpu_id),
                {
                    "duration_sec": float(final_duration),
                    "slowdown_sec": max(0.0, float(slowdown)),
                    "uncertainty_z": float(uncertainty_z),
                    "vram_mb": max(0.0, float(vram)),
                    "ram_mb": max(0.0, float(ram)),
                },
            )
        return eft

    @staticmethod
    def _fit_start(
        timelines: Any,
        gpu_id: str,
        vram: float,
        ram: float,
        duration: float,
        *,
        task_id: str,
        component: str,
        fp: str,
        input_size: float,
        start: float,
        now: float,
        fixed: bool,
    ) -> float | None:
        kwargs = {
            "exclude_task_id": task_id,
            "candidate_component": component,
            "candidate_config_fingerprint": fp,
            "candidate_input_size": input_size,
            "_captured_now": now,
        }
        if fixed:
            return (
                start
                if timelines.candidate_interval_fits(
                    gpu_id, start, start + duration, vram, ram, **kwargs
                )
                else None
            )
        return timelines.earliest_dual_fit_time(
            gpu_id,
            vram,
            ram,
            horizon_sec=120.0,
            required_duration_sec=duration,
            earliest_start_time=start,
            **kwargs,
        )

    def _uncertainty_z(self, alpha: float) -> float:
        if self.estimator_basis == "mean" or alpha <= 0.5:
            return 0.0
        return max(0.0, _norm_ppf(alpha))

    def _compute_exec_time(
        self,
        mu: float,
        sigma: float,
        alpha: float,
    ) -> float:
        """α-quantile execution time — Plan EFT.

        Subclasses override to swap stochastic ↔ deterministic
        formulations.  Default (StochasticEFTPlacement): adds σ-quantile
        padding ``μ + σ·Φ⁻¹(α)`` when σ and α are positive; otherwise
        falls back to ``μ`` alone.
        """
        if sigma > 0 and alpha > 0:
            return mu + sigma * self._uncertainty_z(alpha)
        return mu

    def _use_resource_upper_bounds(self) -> bool:
        return True


class GreedyVRAMPlacement:
    """Pluggable alternative to ``StochasticEFTPlacement`` (plan Built-in).

    Ignores GP uncertainty entirely and picks the GPU with the largest
    currently-available VRAM.  Useful as a baseline for ablation.
    """

    estimator_basis = "mean"

    def evaluate_gpu(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        extra_same_component_active: int,
        planner: GlobalPlanner,
        *,
        task_id: str = "",
    ) -> float | None:
        cs = planner.campaign_scheduler
        tl = cs._timelines.get(gpu_id)
        if not tl:
            return None
        fp = (
            planner._campaign_component_fingerprint(
                campaign_id,
                component,
            )
            or ""
        )
        batch_profile = (
            planner.dynamic_batch_profile(task_id, gpu_id) if task_id else None
        )
        if batch_profile is not None:
            vram = float(batch_profile.vram_mean)
            ram = float(batch_profile.ram_mean)
        else:
            vram = cs._predict_vram(
                component,
                input_size,
                gpu_id,
                use_upper=False,
                config_fingerprint=fp,
            )
            ram = cs._predict_ram(
                component,
                input_size,
                gpu_id,
                use_upper=False,
                config_fingerprint=fp,
            )
            vram = float(vram) if isinstance(vram, (int, float)) else 0.0
            ram = float(ram) if isinstance(ram, (int, float)) else 0.0
        avail = tl.available_vram_at(exclude_task_id=task_id)
        ram_avail = cs._timelines.available_host_ram_at(
            exclude_task_id=task_id,
        )
        duration = _safe_float(
            batch_profile.latency_mean
            if batch_profile is not None
            else cs._predict_latency(
                component,
                input_size,
                gpu_id,
                config_fingerprint=fp,
            ),
            0.0,
        )
        if duration <= 0.0:
            duration = _safe_float(
                cs._cold_inference_latency_sec()
                if hasattr(cs, "_cold_inference_latency_sec")
                else 30.0,
                30.0,
            )
        now = time.time()
        if not cs._timelines.candidate_interval_fits(
            gpu_id,
            now,
            now + duration,
            vram,
            ram,
            exclude_task_id=task_id,
            candidate_component=component,
            candidate_config_fingerprint=fp,
            candidate_input_size=input_size,
            _captured_now=now,
        ):
            return None
        return -(avail + ram_avail)


class DeterministicEFTPlacement(StochasticEFTPlacement):
    """μ-only EFT placement — proton-naive baseline.

    Drops the σ-quantile padding from execution time, leaving
    ``EFT = EST + μ_lat + slowdown + init_if_cold``.  All other
    components — feasibility gate, VRAM check, cold-start detection,
    self/pairwise interference cost (μ-only forms already), primary
    SLO protection — are inherited unchanged from
    ``StochasticEFTPlacement``.

    Intended use: proton-naive baseline against the stochastic default
    so the paper submission can isolate the σ-quantile contribution.
    """

    estimator_basis = "mean"

    def _compute_exec_time(
        self,
        mu: float,
        sigma: float,
        alpha: float,
    ) -> float:
        return mu

    def _use_resource_upper_bounds(self) -> bool:
        return False


class ReactivePreInit:
    """Pluggable alternative to ``DAGLookaheadPreInit`` (plan Built-in).

    Does NOT pre-init — relies on the Validator's cold-start path
    (``WorkerActivator``) to create workers reactively when a plan
    targets a cold GPU.  Equivalent to ``pre_init_depth=0`` in the
    plan's Policy Integration table.
    """

    def plan_preinit(
        self,
        task_id: str,
        component: str,
        assigned_gpu: str,
        campaign_id: str,
        eft: float,
        planner: GlobalPlanner,
    ) -> list[PreInitAction]:
        return []


class NoPreInit(ReactivePreInit):
    """Explicit "pre-init disabled" strategy (plan Built-in).

    Identical to ``ReactivePreInit`` semantically — kept as a distinct
    class so config files can set ``preinit_strategy: no_preinit``
    unambiguously.
    """


class DAGLookaheadPreInit:
    """DAG-aware pre-init scheduling (D3).

    ORION (OSDI 2022): DAG-based proactive container warm-up.
    trigger_at = EFT - (μ_init + σ_init × Φ⁻¹(α_init)).
    P(ready) = α_init = confidence (one-sided, same as α).
    Depth-1 only to avoid over-reservation.
    """

    def plan_preinit(
        self,
        task_id: str,
        component: str,
        assigned_gpu: str,
        campaign_id: str,
        eft: float,
        planner: GlobalPlanner,
    ) -> list[PreInitAction]:
        cs = planner.campaign_scheduler
        cq = cs._campaign_queues.get(campaign_id)
        if not cq or not cq.dag_context:
            return []

        downstream = cq.dag_context.downstream_map.get(component, [])
        if not downstream:
            return []

        actions: list[PreInitAction] = []
        for next_comp in downstream[:1]:
            init_mu = cs._get_init_latency(next_comp, assigned_gpu)
            init_sigma = 0.0
            if cs._init_tracker:
                _, init_var = cs._init_tracker.predict(next_comp, assigned_gpu)
                if init_var and init_var > 0:
                    init_sigma = init_var**0.5

            alpha = HEFTPriority._get_alpha(cs, next_comp, 0.0)
            if init_sigma > 0 and alpha > 0:
                init_duration = init_mu + init_sigma * _norm_ppf(alpha)
            else:
                init_duration = init_mu

            trigger_at = eft - init_duration
            weight_mb = cs._predict_vram(next_comp, 0.0, assigned_gpu)

            actions.append(
                PreInitAction(
                    component=next_comp,
                    target_gpu_id=assigned_gpu,
                    trigger_at=trigger_at,
                    init_duration_sec=init_duration,
                    weight_vram_mb=weight_mb,
                )
            )

        return actions


class PointEstimateBackfill:
    """Simple point-estimate backfill admission (alternative to BayesianBackfill).

    Admits if μ ≤ budget. No GP CDF, no interference cost.
    """

    def should_admit(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        primary_deadline: float,
        planner: GlobalPlanner,
    ) -> bool:
        cs = planner.campaign_scheduler
        budget = primary_deadline - time.time()
        if budget <= 0:
            return False
        fp = (
            planner._campaign_component_fingerprint(campaign_id, component)
            if campaign_id
            else ""
        ) or ""
        task_id = str(getattr(planner, "_dynamic_batch_admission_task_id", "") or "")
        batch_profile = planner.dynamic_batch_profile(task_id, gpu_id)
        mu = (
            float(batch_profile.latency_mean)
            if batch_profile is not None
            else cs._predict_latency(
                component,
                input_size,
                gpu_id,
                config_fingerprint=fp,
            )
        )
        if mu is None:
            return False
        return mu <= budget


class NoBackfill:
    """Disable backfill admission entirely (the paper ablation).

    Returns ``False`` for every candidate, so backfill tasks are never
    admitted to a co-locate slot.  Primary tasks still flow through
    the planner normally — only the backfill admission path is gated
    off.  Selected via runtime YAML ``global_planner.backfill_admission:
    none`` so paper experiment #3 can compare ``cdf`` (default
    Bayesian) vs ``point`` vs ``none`` cells without code changes.

    Plan Integration SB row — "OFF" toggle materialised as a
    distinct strategy class rather than a boolean flag inside
    BayesianBackfill, so the Protocol surface stays uniform and unit
    tests for the disabled path can be written against this class
    directly.
    """

    def should_admit(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        primary_deadline: float,
        planner: GlobalPlanner,
    ) -> bool:
        return False


class BayesianBackfill:
    """GP-enhanced binary admission (D4).

    Plan fixa — threshold formula corrected.
    Classical Bayesian 2-action rule (Berger 1985 ): minimum-expected-cost
    decision under asymmetric loss (Cu = reject opportunity cost, Co = admit
    failure cost).

    Decision derivation (success-amortization model, plan ):
        E[cost | reject]  = Cu = budget                           (opportunity)
        E[cost | admit]   = (1 − p) · Co + p · 0                  (success amortizes)
        Admit iff E[admit] < E[reject]
            iff  (1 − p) · Co < Cu
            iff  p > 1 − Cu/Co
        threshold = 1 − Cu/Co  =  (Co − Cu) / Co

    Pre- used `Cu/(Cu+Co)` (continuous critical-fractile form);
    that formula is for continuous order-quantity decisions, not binary
    admit/reject — applying it produced the *opposite* direction (admit
    became more strict when interference shrank).

    Primary protection mechanism is in Co's *weighted* summation
    (`weight_fn` parameter, default w≡1 → Phase A behavior).  Phase B
    primary-weighted variant retired (Phase D ablation archive — see
    changelog  ablation block).
    """

    _EPSILON_SIGMA = 1e-6

    def should_admit(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        primary_deadline: float,
        planner: GlobalPlanner,
        *,
        weight_fn: Any | None = None,
    ) -> bool:
        cs = planner.campaign_scheduler
        now = time.time()
        budget = primary_deadline - now

        if budget <= 0:
            return False

        fp = (
            planner._campaign_component_fingerprint(campaign_id, component)
            if campaign_id
            else ""
        ) or ""
        task_id = str(getattr(planner, "_dynamic_batch_admission_task_id", "") or "")
        batch_profile = planner.dynamic_batch_profile(task_id, gpu_id)
        if batch_profile is not None:
            mu_inf = float(batch_profile.latency_mean)
            sigma = float(batch_profile.latency_sigma)
        else:
            mu_inf = cs._predict_latency(
                component,
                input_size,
                gpu_id,
                config_fingerprint=fp,
            )
            if mu_inf is None:
                return False
            sigma = (
                cs._predict_latency_sigma(
                    component,
                    input_size,
                    gpu_id,
                    config_fingerprint=fp,
                )
                or 0.0
            )

        if not (math.isfinite(mu_inf) and math.isfinite(sigma)):
            return False

        if planner._has_warm_on_gpu(component, gpu_id):
            mu_total = mu_inf
        else:
            mu_init = cs._get_init_latency(component, gpu_id) or 0.0
            mu_total = mu_inf + mu_init

        if sigma < self._EPSILON_SIGMA:
            p_success = 1.0 if mu_total <= budget else 0.0
        else:
            z = (budget - mu_total) / sigma
            p_success = _norm_cdf(z)

        intf = self._interference_cost(
            component,
            gpu_id,
            mu_total,
            planner,
            input_size=input_size,
            campaign_id=campaign_id,
            weight_fn=weight_fn,
        )
        co = intf

        if co <= 0:
            return True

        threshold = 1.0 - budget / co

        if threshold > 1.0:
            return False

        return p_success >= threshold

    @staticmethod
    def _interference_cost(
        component: str,
        gpu_id: str,
        mu_bf: float,
        planner: GlobalPlanner,
        *,
        input_size: float = 0.0,
        campaign_id: str = "",
        weight_fn: Any | None = None,
    ) -> float:
        """Canonical interference cost per D2/D4 ( P0 formula unification).

        interference_cost = Σ_{k co-located} w(k) × (sd(bf, k) - 1) × overlap_k    [seconds]

        `sd - 1` captures the *additional* delay caused by co-location, not the
        total co-execution time.  Using `sd × overlap` would double-count the
        baseline execution (already in μ_lat), inflating the admission threshold
        and unnecessarily rejecting feasible backfills.  D2 EFT penalty and D4
        admission Co share this exact formula (fix unification).

        Plan fixa — `weight_fn(neighbor_entry) -> float` parameter
        added.  Default `weight_fn ≡ 1` (Phase A behavior).  Phase B primary-
        weighted variant retired (Phase D ablation archive — see changelog).
        """
        if weight_fn is None:
            weight_fn = lambda _e: 1.0
        cs = planner.campaign_scheduler
        tl = cs._timelines.get(gpu_id)
        if not tl:
            return 0.0
        intf_reg = (
            cs._signal_service.interference_registry if cs._signal_service else None
        )
        fp = (
            planner._campaign_component_fingerprint(campaign_id, component)
            if campaign_id
            else ""
        ) or ""
        alpha = HEFTPriority._get_alpha(cs, component, input_size, fp)
        uncertainty_z = (
            0.0
            if planner.interference_estimator_basis == "mean"
            else max(0.0, _norm_ppf(alpha))
        )
        now = time.time()

        same_comp_overlaps = [
            (e, overlap)
            for e in _active_entries_for_component(tl, component)
            if not getattr(e, "is_init", False)
            and not getattr(e, "is_evict_masked", False)
            and not _is_stale_predicted_entry(e, now)
            if (overlap := _base_work_overlap(e, now, mu_bf)) > 0.0
        ]
        post_count = len(same_comp_overlaps) + 1
        self_cost = 0.0
        if same_comp_overlaps and intf_reg is not None:
            self_weight = sum(weight_fn(e) for e, _ in same_comp_overlaps) / len(
                same_comp_overlaps
            )
            self_delta = _effective_self_delta(
                intf_reg,
                component,
                n_concurrent=post_count,
                fp=fp,
                uncertainty_z=uncertainty_z,
            )
            self_overlap = max(overlap for _e, overlap in same_comp_overlaps)
            self_cost = self_weight * self_delta * self_overlap

        pair_cost = 0.0
        for e in tl.active_entries:
            if e.component == component:
                continue
            if getattr(e, "is_init", False):
                continue
            if getattr(e, "is_evict_masked", False):
                continue
            if _is_stale_predicted_entry(e, now):
                continue
            if intf_reg is None:
                continue
            pair_delta = _effective_pair_delta(
                intf_reg,
                component,
                e.component,
                fp=fp,
                uncertainty_z=uncertainty_z,
            )
            overlap = _base_work_overlap(e, now, mu_bf)
            pair_cost += weight_fn(e) * pair_delta * overlap

        return self_cost + pair_cost





class GlobalPlanner:
    """HEFT-based unified scheduling solver with Validator feedback.

    Produces ``DispatchPlan`` for each task by running a single-pass
    solve loop that integrates ordering, placement, pre-init, backfill
    admission, and campaign priority.

    All strategies are pluggable via Protocol interfaces.
    """

    def __init__(
        self,
        campaign_scheduler: Any,
        priority: Any = None,
        placement: Any = None,
        preinit: Any = None,
        backfill: Any = None,
        campaign: Any = None,
        eviction_strategy: Any = None,
        ram_eviction_strategy: Any = None,
        *,
        fanout_barrier_force: bool = False,
    ) -> None:
        self.campaign_scheduler = campaign_scheduler
        self.eviction_strategy = eviction_strategy or GreedyEvictionStrategy()
        self.ram_eviction_strategy = ram_eviction_strategy or NoRamEviction()
        self.priority = priority or HEFTPriority(
            fanout_barrier_force=fanout_barrier_force,
        )
        self.placement = placement or StochasticEFTPlacement()
        self.interference_estimator_basis = str(
            getattr(self.placement, "estimator_basis", "ucb")
        )
        self.preinit = preinit or DAGLookaheadPreInit()
        self.backfill = backfill or BayesianBackfill()
        self.campaign = campaign or FIFOCampaign()
        try:
            if hasattr(campaign_scheduler, "set_campaign_strategy"):
                campaign_scheduler.set_campaign_strategy(self.campaign)
        except Exception:
            _LOG.warning(
                "[planner] failed to bind CampaignStrategy onto CampaignScheduler",
                exc_info=True,
            )

        self._cached_primary_id: str | None = None
        self._cached_primary_id_set: bool = False
        self._self_concurrency_best_n_cache: dict[
            tuple[str, str, int, float], tuple[int, float]
        ] = {}
        self.reciprocal_interference_correction = False
        self._candidate_eft_details: dict[tuple[str, str], dict[str, float]] = {}
        self._reciprocal_query_results: dict[tuple[int, Any], Any] = {}
        self._dynamic_batch_context: Any = None
        self._owns_dynamic_batch_context = False
        self._dynamic_batch_profiles: dict[tuple[str, str], Any] = {}
        self._dynamic_batch_blocked: set[tuple[str, str]] = set()
        self._dynamic_batch_admission_task_id = ""
        self._gpu_models = _discover_gpu_models()
        self._applied_reciprocal_dispatch_events: dict[
            str, _AppliedReciprocalAttempt
        ] = {}

        self._constraint_tracker = ConstraintTracker()
        try:
            self._constraint_tracker.attach_scenario(self.timelines)
        except Exception:
            _LOG.warning(
                "[silent-except] %s:%d (%s)",
                __name__,
                0,
                "swallowed_pass",
                exc_info=True,
            )

        def _init_latency_accessor(component: str, gpu_id: str):
            cs = self.campaign_scheduler
            mu = 0.0
            sigma = 0.0
            if hasattr(cs, "_get_init_latency"):
                try:
                    mu = float(cs._get_init_latency(component, gpu_id) or 0.0)
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed an exception; body=%s",
                        __name__,
                        "mu = 0.0",
                        exc_info=True,
                    )
                    mu = 0.0
            tracker = getattr(cs, "_init_tracker", None)
            if tracker is not None:
                try:
                    _, var = tracker.predict(component, gpu_id)
                    if var and var > 0:
                        sigma = var**0.5
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed an exception; body=%s",
                        __name__,
                        "sigma = 0.0",
                        exc_info=True,
                    )
                    sigma = 0.0
            return mu, sigma

        try:
            self._constraint_tracker.attach_init_latency_accessor(
                _init_latency_accessor,
            )
        except Exception:
            _LOG.warning(
                "[silent-except] %s:%d (%s)",
                __name__,
                0,
                "swallowed_pass",
                exc_info=True,
            )

        self._recent_eviction_at: dict[tuple[str, str], float] = {}

        self._eft_median_window: int = 100
        self._recent_eft_samples: deque[float] = deque(maxlen=self._eft_median_window)
        self._info_log_state: dict[tuple[Any, ...], tuple[float, int]] = {}

    def _log_info_throttled(
        self,
        key: tuple[Any, ...],
        message: str,
        *args: Any,
        interval_sec: float = 10.0,
    ) -> None:
        """Emit INFO at most once per key/window and summarize suppressions."""
        now = time.time()
        last, suppressed = self._info_log_state.get(key, (0.0, 0))
        if last > 0.0 and now - last < interval_sec:
            self._info_log_state[key] = (last, suppressed + 1)
            return
        if suppressed:
            message = f"{message} (suppressed=%d in %.0fs)"
            args = (*args, suppressed, interval_sec)
        _LOG.info(message, *args)
        self._info_log_state[key] = (now, 0)

    def _record_eft_sample(self, eft_wall_time: float) -> None:
        """Record a positive finite remaining completion horizon."""
        if eft_wall_time is None:
            return
        remaining_sec = float(eft_wall_time) - time.time()
        if not math.isfinite(remaining_sec) or remaining_sec <= 0:
            return
        self._recent_eft_samples.append(remaining_sec)

    def _recent_eft_median(self) -> float:
        """Return the median recent completion horizon, or zero if empty."""
        if not self._recent_eft_samples:
            return 0.0
        xs = sorted(self._recent_eft_samples)
        n = len(xs)
        mid = n // 2
        return xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])

    def _is_gpu_warm_for(self, component: str, gpu_id: str) -> bool:
        """Plan fix — helper for tie-break edge case
        (cold+warm within bucket).  Returns True iff the supervisor
        reports a ready worker for ``component`` on ``gpu_id``.  Missing
        supervisor ref defaults to False (caller treats tie as
        undifferentiated — no warm preference)."""
        if getattr(self.campaign_scheduler, "_supervisor", None) is None:
            return False
        worker_name, worker_addr, needs_cold = self._resolve_worker(
            component,
            gpu_id,
        )
        return bool(worker_name and worker_addr and not needs_cold)

    def _tie_break_select(
        self,
        component: str,
        candidates: list[tuple[str, float]],
    ) -> str:
        """Plan fix — adaptive-bucket + uniform-random
        tie-break over ``(gpu_id, eft)`` candidates (EFT not None).

        - Bucket size = ``max(1ms, 0.1% × recent_eft_median)`` → task-
          duration scale-invariant.
        - Cold+warm edge: when both cold and warm GPUs truly tie within
          the duration-scaled bucket, cold wins to expand residency.
        - Uniform random over remaining tied candidates.

        Caller is responsible for having filtered infeasible candidates
        (eft is None).  ``candidates`` must be non-empty.
        """
        assert candidates, "_tie_break_select: empty candidate list"
        min_eft = min(eft for _, eft in candidates)
        median_eft = self._recent_eft_median()
        bucket_sec = max(0.001, 0.001 * median_eft)
        tied = [gpu for gpu, eft in candidates if (eft - min_eft) <= bucket_sec]
        if len(tied) == 1:
            return tied[0]
        tied_cold = [g for g in tied if not self._is_gpu_warm_for(component, g)]
        if tied_cold and len(tied_cold) < len(tied):
            tied = tied_cold
        if len(tied) == 1:
            return tied[0]
        return random.choice(tied)


    @property
    def _saturated_gpus(self) -> dict[str, float]:
        return self._constraint_tracker._saturated_gpus

    @property
    def _dead_workers(self) -> set[str]:
        return self._constraint_tracker._dead_workers

    @property
    def _excluded_workers(self) -> set[str]:
        return self._constraint_tracker._excluded_workers

    @property
    def _vram_corrections(self) -> dict[str, float]:
        return self._constraint_tracker._vram_corrections

    @property
    def _unhealthy_gpus(self) -> dict[str, float]:
        return self._constraint_tracker._unhealthy_gpus

    @property
    def _gpu_reset_required(self) -> set[str]:
        return self._constraint_tracker._gpu_reset_required

    @property
    def _rma_qualifying_gpus(self) -> set[str]:
        return self._constraint_tracker._rma_qualifying_gpus

    @property
    def _unreachable_workers(self) -> dict[str, float]:
        return self._constraint_tracker._unreachable_workers

    @property
    def _activation_excluded(self) -> dict[tuple[str, str], float]:
        return self._constraint_tracker._activation_excluded

    @property
    def timelines(self) -> SchedulingScenario:
        return self.campaign_scheduler._timelines


    def solve(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> list[DispatchPlan]:
        _solve_start = time.time()
        try:
            return self._solve_impl(pending_tasks)
        finally:
            self._cached_primary_id = None
            self._cached_primary_id_set = False
            self._self_concurrency_best_n_cache.clear()
            self._candidate_eft_details.clear()
            if getattr(self, "_owns_dynamic_batch_context", False):
                self.set_dynamic_batch_context()
            _solve_elapsed_ms = (time.time() - _solve_start) * 1000.0
            if _solve_elapsed_ms > 50.0:
                _LOG.warning(
                    "[hot-path] GlobalPlanner.solve n_pending=%d elapsed=%.1fms",
                    len(pending_tasks or []),
                    _solve_elapsed_ms,
                )

    async def solve_admissible(
        self,
        pending_tasks: list[dict[str, Any]],
        *,
        commit_predictions: bool = True,
    ) -> tuple[list[DispatchPlan], dict[str, PlanningExhausted]]:
        """Batch solve that keeps feasible plans and parks infeasible tasks.

        ``solve()`` remains all-or-nothing for legacy callers.  The
        SchedulingSupervisor uses this admission-oriented variant so one
        currently-infeasible handle does not force every other ready handle
        back through a single-task ``plan()`` storm.

        When ``commit_predictions`` is false, the solve may still use
        predicted entries internally to model the sequential HEFT placement
        frontier, but those speculative entries are detached before returning.
        The caller must explicitly commit only the plans it is about to
        dispatch.  This keeps batch lookahead from leaking unlaunched future
        work into the live timeline.
        """
        _solve_start = time.time()
        try:
            return self._solve_admissible_impl(
                pending_tasks,
                commit_predictions=commit_predictions,
            )
        finally:
            self._cached_primary_id = None
            self._cached_primary_id_set = False
            self._self_concurrency_best_n_cache.clear()
            self._candidate_eft_details.clear()
            if getattr(self, "_owns_dynamic_batch_context", False):
                self.set_dynamic_batch_context()
            _solve_elapsed_ms = (time.time() - _solve_start) * 1000.0
            if _solve_elapsed_ms > 50.0:
                _LOG.warning(
                    "[hot-path] GlobalPlanner.solve_admissible "
                    "n_pending=%d elapsed=%.1fms",
                    len(pending_tasks or []),
                    _solve_elapsed_ms,
                )

    def _solve_impl(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> list[DispatchPlan]:
        """Batch solve: plan ALL pending tasks in a single pass.

        Implements the plan's 3-phase solve loop:
          Phase 1: Campaign priority → primary/backfill classification
          Phase 2: b-rank ordering (all tasks sorted)
          Phase 3: Sequential placement with backfill/pre-init

        Args:
            pending_tasks: List of dicts with keys:
                task_id, campaign_id, component, input_size, is_backfill
        """
        cs = self.campaign_scheduler
        self._solve_now_wall = time.time()
        self._self_concurrency_best_n_cache.clear()
        self._candidate_eft_details.clear()
        self._reciprocal_query_results.clear()

        try:
            self.timelines.gc_orphaned_predicted_entries()
        except Exception:
            _LOG.warning(
                "[solve] gc_orphaned_predicted_entries skipped",
                exc_info=True,
            )


        gpu_ids = list(cs._timelines.gpu_ids)

        primary_cq = self.campaign.select_primary(cs._campaign_queues)
        primary_budget = self._build_primary_deadline_tracker(primary_cq, gpu_ids)

        self._cached_primary_id = primary_cq.campaign_id if primary_cq else None
        self._cached_primary_id_set = True

        for t in pending_tasks:
            t["b_rank"] = self.priority.compute_rank(
                t["component"],
                t["campaign_id"],
                self,
            )

        def _campaign_arrival(t: dict[str, Any]) -> float:
            cq = cs._campaign_queues.get(t["campaign_id"])
            return _safe_float(getattr(cq, "arrival_time", None), math.inf)

        tasks_sorted = sorted(
            pending_tasks,
            key=lambda t: (
                bool(t.get("is_backfill", False)),
                _campaign_arrival(t),
                -t["b_rank"],
                str(t.get("task_id", "")),
            ),
        )

        plans: list[DispatchPlan] = []
        current_solve = CurrentSolvePlacements.for_task_ids(
            [str(task.get("task_id", "") or "") for task in tasks_sorted]
        )

        for t in tasks_sorted:
            tid = t["task_id"]
            current_solve.begin_task(tid)

            cid = t["campaign_id"]
            comp = t["component"]
            inp = t.get("input_size", 0.0)
            is_bf = t.get("is_backfill", False)

            plan = self._place_single(
                tid,
                cid,
                comp,
                inp,
                is_bf,
                primary_budget,
                gpu_ids,
                current_solve,
                config_fingerprint=str(t.get("config_fingerprint", "") or ""),
                input_fingerprint=str(t.get("input_fingerprint", "") or ""),
            )
            plans.append(plan)

        return plans

    def _solve_admissible_impl(
        self,
        pending_tasks: list[dict[str, Any]],
        *,
        commit_predictions: bool = True,
    ) -> tuple[list[DispatchPlan], dict[str, PlanningExhausted]]:
        cs = self.campaign_scheduler
        self._solve_now_wall = time.time()
        self._self_concurrency_best_n_cache.clear()
        self._candidate_eft_details.clear()
        self._reciprocal_query_results.clear()

        try:
            self.timelines.gc_orphaned_predicted_entries()
        except Exception:
            _LOG.warning(
                "[solve_admissible] gc_orphaned_predicted_entries skipped",
                exc_info=True,
            )

        gpu_ids = list(cs._timelines.gpu_ids)
        primary_cq = self.campaign.select_primary(cs._campaign_queues)
        primary_budget = self._build_primary_deadline_tracker(primary_cq, gpu_ids)
        self._cached_primary_id = (
            str(getattr(primary_cq, "campaign_id", "") or "") if primary_cq else None
        )
        self._cached_primary_id_set = True

        for t in pending_tasks:
            t["b_rank"] = self.priority.compute_rank(
                t["component"],
                t["campaign_id"],
                self,
            )

        def _campaign_arrival(t: dict[str, Any]) -> float:
            cq = cs._campaign_queues.get(t["campaign_id"])
            return _safe_float(getattr(cq, "arrival_time", None), math.inf)

        tasks_sorted = sorted(
            pending_tasks,
            key=lambda t: (
                bool(t.get("is_backfill", False)),
                _campaign_arrival(t),
                -t["b_rank"],
                str(t.get("task_id", "")),
            ),
        )

        plans: list[DispatchPlan] = []
        skipped: dict[str, PlanningExhausted] = {}
        current_solve = CurrentSolvePlacements.for_task_ids(
            [str(task.get("task_id", "") or "") for task in tasks_sorted]
        )
        pending_task_ids = [
            str(t.get("task_id", "") or "")
            for t in pending_tasks
            if str(t.get("task_id", "") or "")
        ]

        try:
            for t in tasks_sorted:
                tid = str(t.get("task_id", ""))
                current_solve.begin_task(tid)
                try:
                    plan = self._place_single(
                        tid,
                        t["campaign_id"],
                        t["component"],
                        _safe_float(t.get("input_size")),
                        bool(t.get("is_backfill", False)),
                        primary_budget,
                        gpu_ids,
                        current_solve,
                        config_fingerprint=str(t.get("config_fingerprint", "") or ""),
                        input_fingerprint=str(t.get("input_fingerprint", "") or ""),
                    )
                except PlanningExhausted as exc:
                    skipped[tid] = exc
                    continue
                plans.append(plan)

            return plans, skipped
        finally:
            if not commit_predictions:
                self._detach_batch_solve_predictions(pending_task_ids)

    def _detach_batch_solve_predictions(self, task_ids: list[str]) -> int:
        """Remove speculative predictions created during a detached batch solve."""
        removed_total = 0
        remove_many = getattr(
            self.timelines,
            "remove_predicted_entries_for_tasks",
            None,
        )
        if callable(remove_many):
            try:
                removed_total = _safe_int(
                    remove_many(task_ids, include_dispatching=False)
                )
            except TypeError:
                removed_total = _safe_int(remove_many(task_ids))
            if removed_total:
                _LOG.debug(
                    "[batch-plan] detached %d speculative predicted entries",
                    removed_total,
                )
            return removed_total
        remove_fn = getattr(self.timelines, "remove_predicted_entries_for_task", None)
        if not callable(remove_fn):
            return 0
        for task_id in task_ids:
            try:
                removed_total += _safe_int(
                    remove_fn(task_id, include_dispatching=False)
                )
            except TypeError:
                removed_total += _safe_int(remove_fn(task_id))
        if removed_total:
            _LOG.debug(
                "[batch-plan] detached %d speculative predicted entries",
                removed_total,
            )
        return removed_total

    def commit_dispatch_plan_prediction(
        self,
        plan: DispatchPlan,
        *,
        input_size: float = 0.0,
    ) -> None:
        """Commit one launch-ready batch plan to the live timeline.

        Detached batch solving computes a speculative schedule in the live
        data structure and then removes it before returning.  Only plans that
        the HTTP handoff is about to drive should become live predicted
        entries; future lookahead plans stay out of the live timeline.
        """
        meta = dict(getattr(plan, "worker_metadata", {}) or {})
        projection = dict(meta.get("planning_projection") or {})
        reciprocal = dict(meta.get("reciprocal_interference") or {})
        if self.reciprocal_interference_correction:
            required = {
                "candidate_base_duration_sec",
                "estimator_basis",
                "uncertainty_z",
            }
            missing = sorted(required.difference(reciprocal))
            if missing:
                raise ValueError(
                    f"reciprocal handoff metadata missing: {', '.join(missing)}"
                )
            if (
                str(reciprocal["estimator_basis"]).strip().lower()
                != self.interference_estimator_basis
            ):
                raise ValueError("reciprocal handoff estimator basis mismatch")
        start_time = _safe_float(
            projection.get("planned_start_time")
            or getattr(plan, "planned_start_time", 0.0)
            or time.time()
        )
        predicted_end_time = _safe_float(
            projection.get("predicted_end_time")
            or (
                start_time
                + max(
                    0.0,
                    _safe_float(getattr(plan, "predicted_latency_sec", 0.0)),
                )
            )
        )
        vram = _safe_float(
            projection.get("predicted_vram_mb")
            if "predicted_vram_mb" in projection
            else getattr(plan, "vram_budget_mb", 0)
        )
        ram = _safe_float(
            projection.get("predicted_ram_mb")
            if "predicted_ram_mb" in projection
            else getattr(plan, "ram_budget_mb", 0)
        )
        config_fp = str(projection.get("config_fingerprint") or "")
        input_fp = str(projection.get("input_fingerprint") or "")
        was_primary = bool(projection.get("was_primary_at_dispatch", False))
        batch_profile = dict(meta.get("dynamic_batch_profile") or {})
        entry = self.timelines.add_predicted_entry(
            task_id=str(plan.task_id),
            component=str(plan.component),
            gpu_id=str(plan.target_gpu_id),
            worker_name=str(plan.target_worker_name or ""),
            start_time=start_time,
            predicted_end_time=predicted_end_time,
            predicted_vram_mb=vram,
            predicted_ram_mb=ram,
            campaign_id=str(plan.campaign_id or ""),
            is_backfill=bool(plan.is_backfill),
            is_dispatching=True,
            input_size=float(input_size or projection.get("input_size") or 0.0),
            config_fingerprint=config_fp,
            input_fingerprint=input_fp,
            gpu_model=str(projection.get("gpu_model") or ""),
            mps_mode=str(projection.get("mps_mode") or ""),
            worker_backend=str(projection.get("worker_backend") or ""),
            actor_model=str(projection.get("actor_model") or ""),
            adapter_version=str(projection.get("adapter_version") or ""),
            was_primary_at_dispatch=was_primary,
            logical_batch_size=_safe_int(
                batch_profile.get("logical_batch_size"),
            ),
            execution_batch_size=_safe_int(
                batch_profile.get("execution_batch_size"),
            ),
        )
        if reciprocal:
            entry.reciprocal_base_duration_sec = _safe_float(
                reciprocal["candidate_base_duration_sec"]
            )

    def evaluate_reciprocal_gpu(
        self,
        gpu_id: str,
        *,
        overlays: tuple[ReciprocalTaskOverlay, ...] = (),
        exclude_task_ids: frozenset[str] = frozenset(),
        actual_incumbents_only: bool = False,
        uncertainty_z: float = 0.0,
        now_wall: float | None = None,
        now_mono: float | None = None,
    ) -> ReciprocalEvaluation:
        """Evaluate one immutable, versioned GPU-local reciprocal overlay."""
        timeline = self.timelines.get(str(gpu_id))
        return _evaluate_reciprocal_entries(
            self,
            str(gpu_id),
            tuple(getattr(timeline, "active_entries", ()) if timeline else ()),
            overlays=overlays,
            exclude_task_ids=exclude_task_ids,
            actual_incumbents_only=actual_incumbents_only,
            uncertainty_z=uncertainty_z,
            now_wall=now_wall,
            now_mono=now_mono,
            timeline_state_version=int(getattr(timeline, "_state_version", 0) or 0),
        )

    def reconcile_reciprocal_gpu(
        self,
        gpu_id: str,
        *,
        event_attempt_id: str = "",
        expected_timeline_version: int | None = None,
        uncertainty_z: float = 0.0,
        now_wall: float | None = None,
        now_mono: float | None = None,
        metrics: dict[str, int] | None = None,
    ) -> bool:
        """Synchronously account base work and refresh affected completions."""
        if not self.reciprocal_interference_correction:
            return False
        timeline = self.timelines.get(str(gpu_id))
        if timeline is None:
            return False
        version = int(getattr(timeline, "_state_version", 0) or 0)
        if expected_timeline_version is not None and version != int(
            expected_timeline_version
        ):
            return False
        entries = [
            entry
            for entry in getattr(timeline, "active_entries", ())
            if not entry.is_completed and not entry.is_predicted and not entry.is_init
        ]
        if event_attempt_id and not any(
            str(getattr(entry, "execution_attempt_id", "") or "")
            == str(event_attempt_id)
            for entry in entries
        ):
            return False

        wall = float(time.time() if now_wall is None else now_wall)
        mono = float(time.monotonic() if now_mono is None else now_mono)
        for entry in entries:
            _account_reciprocal_entry(entry, now_wall=wall, now_mono=mono)
        evaluation = self.evaluate_reciprocal_gpu(
            str(gpu_id),
            actual_incumbents_only=True,
            uncertainty_z=uncertainty_z,
            now_wall=wall,
            now_mono=mono,
        )
        if metrics is not None:
            metrics["scanned_entries"] = evaluation.scanned_entries
            metrics["temporal_segments"] = len(evaluation.segments)
            metrics["interference_lookups"] = evaluation.interference_lookups
        changed = False
        changed_entries: list[Any] = []
        for entry in entries:
            completion = evaluation.completion_wall(str(entry.task_id))
            if completion is None:
                continue
            multiplier = evaluation.multiplier(str(entry.task_id))
            entry_changed = (
                not math.isclose(
                    float(entry.predicted_end_time),
                    completion,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(entry.current_reciprocal_multiplier),
                    multiplier,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or int(entry.reciprocal_evidence_epoch)
                != int(evaluation.evidence_epoch)
            )
            if entry_changed:
                changed = True
                changed_entries.append(entry)
            entry.predicted_end_time = completion
            entry.current_reciprocal_multiplier = multiplier
            entry.reciprocal_evidence_epoch = evaluation.evidence_epoch
        if changed:
            timeline._bump_state_version()
            pending_pre_inits = getattr(
                self.campaign_scheduler,
                "_pending_pre_inits",
                None,
            )
            schedule_pre_init = getattr(
                self.campaign_scheduler,
                "_schedule_downstream_pre_init",
                None,
            )
            if isinstance(pending_pre_inits, dict) and callable(schedule_pre_init):
                for entry in changed_entries:
                    source_task_id = str(entry.task_id)
                    for key in list(pending_pre_inits):
                        scheduled_task_id = str(key[2]).rsplit("_", 1)[0]
                        if scheduled_task_id == source_task_id:
                            pending_pre_inits.pop(key, None)
                    schedule_pre_init(
                        str(entry.task_id),
                        str(entry.campaign_id),
                        str(entry.component),
                        str(entry.gpu_id),
                        float(entry.predicted_end_time),
                    )
        return True

    def is_reciprocal_attempt_current(
        self,
        task_id: str,
        event_id: str,
        gpu_id: str,
    ) -> bool:
        """Return false only when an event targets a superseded live attempt."""
        if not event_id or not self.reciprocal_interference_correction:
            return True
        entry = self.timelines.find_entry(str(task_id))
        if entry is None or entry.is_completed:
            return str(event_id) in self._applied_reciprocal_dispatch_events
        active_event = str(getattr(entry, "execution_attempt_id", "") or "")
        return str(entry.gpu_id) == str(gpu_id) and (
            not active_event or active_event == str(event_id)
        )

    def reconcile_reciprocal_event(
        self,
        gpu_ids: tuple[str, ...] | list[str],
        event_kind: str,
        *,
        component: str = "",
        metric: str = "",
        uncertainty_z: float | None = None,
        now_wall: float | None = None,
        now_mono: float | None = None,
    ) -> dict[str, int]:
        """Serialize one runtime boundary through the canonical evaluator."""
        stats = {
            "gpus": 0,
            "entries": 0,
            "revision_conflicts": 0,
            "scanned_entries": 0,
            "temporal_segments": 0,
            "interference_lookups": 0,
            "elapsed_ns": 0,
        }
        if not self.reciprocal_interference_correction:
            return stats
        started_ns = time.perf_counter_ns()
        wall = float(time.time() if now_wall is None else now_wall)
        mono = float(time.monotonic() if now_mono is None else now_mono)
        uncertainty_fn = getattr(self.placement, "_uncertainty_z", None)
        resolved_uncertainty_z = (
            _safe_float(uncertainty_z)
            if uncertainty_z is not None
            else (
                _safe_float(uncertainty_fn(self._get_confidence_alpha()))
                if callable(uncertainty_fn)
                else 0.0
            )
        )
        for gpu_id in dict.fromkeys(str(item) for item in gpu_ids if str(item)):
            timeline = self.timelines.get(gpu_id)
            if timeline is None:
                continue
            active_entries = [
                entry
                for entry in getattr(timeline, "active_entries", ())
                if not entry.is_completed
                and not entry.is_predicted
                and not entry.is_init
            ]
            if event_kind == "profile_drift" and metric == "latency":
                for entry in active_entries:
                    if component and str(entry.component) != str(component):
                        continue
                    _account_reciprocal_entry(
                        entry,
                        now_wall=wall,
                        now_mono=mono,
                    )
                    completed_base = max(
                        0.0,
                        _reciprocal_total_base(entry)
                        - float(entry.remaining_base_work_sec or 0.0),
                    )
                    if _safe_int(getattr(entry, "execution_batch_size", 0)) > 0:
                        refreshed_total = _reciprocal_total_base(entry)
                    else:
                        refreshed_total = self.campaign_scheduler._predict_latency(
                            entry.component,
                            entry.input_size,
                            gpu_id,
                            config_fingerprint=str(entry.config_fingerprint or ""),
                        )
                    refreshed_total = _safe_float(refreshed_total, -1.0)
                    if (
                        not math.isfinite(refreshed_total)
                        or refreshed_total <= completed_base
                    ):
                        stats["revision_conflicts"] += 1
                        continue
                    entry.total_base_work_sec = refreshed_total
                    entry.remaining_base_work_sec = refreshed_total - completed_base
                    entry.prediction_stale = False
            gpu_metrics: dict[str, int] = {}
            if self.reconcile_reciprocal_gpu(
                gpu_id,
                uncertainty_z=max(0.0, resolved_uncertainty_z),
                now_wall=wall,
                now_mono=mono,
                metrics=gpu_metrics,
            ):
                stats["gpus"] += 1
                stats["entries"] += len(active_entries)
                for key in (
                    "scanned_entries",
                    "temporal_segments",
                    "interference_lookups",
                ):
                    stats[key] += gpu_metrics.get(key, 0)
        stats["elapsed_ns"] = max(0, time.perf_counter_ns() - started_ns)
        if stats["gpus"]:
            _LOG.debug(
                "[reciprocal-reconcile] event=%s component=%s metric=%s stats=%s",
                event_kind,
                component,
                metric,
                stats,
            )
        return stats

    def validate_reciprocal_dispatch_reservation(
        self,
        task_id: str,
        component: str,
        gpu_id: str,
        reciprocal: dict[str, Any],
    ) -> Any:
        """Validate reciprocal metadata and the exact launch-owned prediction."""
        if not self.reciprocal_interference_correction:
            raise ValueError(
                "reciprocal dispatch validated while correction is disabled"
            )
        required = {
            "candidate_base_duration_sec",
            "estimator_basis",
            "uncertainty_z",
            "timeline_state_version",
        }
        missing = sorted(required.difference(reciprocal))
        if missing:
            raise ValueError(f"reciprocal metadata missing: {', '.join(missing)}")
        estimator_basis = str(reciprocal["estimator_basis"]).strip().lower()
        if estimator_basis != self.interference_estimator_basis:
            raise ValueError(
                f"reciprocal estimator basis mismatch: "
                f"{estimator_basis} != {self.interference_estimator_basis}"
            )
        candidate_base = _safe_float(reciprocal["candidate_base_duration_sec"], -1.0)
        uncertainty_z = _safe_float(reciprocal["uncertainty_z"], -1.0)
        expected_version = _safe_int(reciprocal["timeline_state_version"], -1)
        candidate_cancel_at = (
            _safe_float(reciprocal.get("candidate_cancel_at"), -1.0)
            if reciprocal.get("candidate_cancel_at") is not None
            else None
        )
        if (
            not math.isfinite(candidate_base)
            or candidate_base < 0.0
            or not math.isfinite(uncertainty_z)
            or uncertainty_z < 0.0
            or expected_version < 0
            or (
                candidate_cancel_at is not None
                and (
                    not math.isfinite(candidate_cancel_at) or candidate_cancel_at < 0.0
                )
            )
        ):
            raise ValueError("invalid reciprocal dispatch metadata")

        matches = [
            entry
            for entry in self.timelines.get_predicted_entry_refs()
            if str(entry.task_id) == str(task_id)
        ]
        if (
            len(matches) != 1
            or str(matches[0].component) != str(component)
            or str(matches[0].gpu_id) != str(gpu_id)
            or not matches[0].is_dispatching
        ):
            raise ValueError("reciprocal prediction reservation missing or stale")
        return matches[0]

    def prepare_reciprocal_dispatch_start(
        self,
        task_id: str,
        event_id: str,
        component: str,
        gpu_id: str,
        reciprocal: dict[str, Any],
    ) -> ReciprocalDispatchPreparation:
        """Validate one attempt and capture its removable prediction state."""
        event_key = str(event_id or "").strip()
        if not event_key or ":" not in event_key:
            raise ValueError("reciprocal dispatch requires task_id:batch_id")
        applied = self._applied_reciprocal_dispatch_events.get(event_key)
        if applied is not None:
            return replace(applied.preparation, duplicate=True)

        candidate = self.validate_reciprocal_dispatch_reservation(
            task_id,
            component,
            gpu_id,
            reciprocal,
        )
        candidate_base = _safe_float(reciprocal["candidate_base_duration_sec"], -1.0)
        uncertainty_z = _safe_float(reciprocal["uncertainty_z"], -1.0)
        expected_version = _safe_int(reciprocal["timeline_state_version"], -1)
        candidate_cancel_at = (
            _safe_float(reciprocal.get("candidate_cancel_at"), -1.0)
            if reciprocal.get("candidate_cancel_at") is not None
            else None
        )
        timeline = self.timelines.get(str(gpu_id))
        prepared_version = int(getattr(timeline, "_state_version", 0) or 0)
        cid_prefix = str(getattr(candidate, "campaign_id", "") or "")[:8]
        all_entries = tuple(
            entry
            for current_gpu in getattr(self.timelines, "gpu_ids", ())
            for entry in getattr(self.timelines.get(str(current_gpu)), "_entries", ())
        )
        rollback_entries = tuple(
            entry
            for entry in all_entries
            if not entry.is_completed
            and (
                str(entry.task_id) == str(task_id)
                or str(entry.task_id).startswith(
                    (f"__lookahead_{cid_prefix}_", f"__reproject_{cid_prefix}_")
                )
            )
        )
        return ReciprocalDispatchPreparation(
            event_id=event_key,
            task_id=str(task_id),
            component=str(component),
            gpu_id=str(gpu_id),
            candidate_base_work_sec=candidate_base,
            uncertainty_z=uncertainty_z,
            candidate_cancel_at=candidate_cancel_at,
            expected_timeline_version=expected_version,
            prepared_timeline_version=prepared_version,
            version_retried=prepared_version
            not in {expected_version, expected_version + 1},
            rollback_entries=rollback_entries,
            preexisting_entry_ids=frozenset(id(entry) for entry in all_entries),
        )

    def commit_reciprocal_dispatch_start(
        self,
        preparation: ReciprocalDispatchPreparation,
        reciprocal: dict[str, Any],
        *,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        gpu_model: str = "",
        mps_mode: str = "",
        worker_backend: str = "",
        actor_model: str = "",
        adapter_version: str = "",
    ) -> tuple[int, float]:
        """Atomically recompute and commit one actual exec-in attempt."""
        if preparation.duplicate:
            return 0, 0.0
        if preparation.event_id in self._applied_reciprocal_dispatch_events:
            return 0, 0.0
        candidate = self.timelines.find_entry(preparation.task_id)
        if (
            candidate is None
            or candidate.is_predicted
            or candidate.is_completed
            or str(candidate.component) != preparation.component
            or str(candidate.gpu_id) != preparation.gpu_id
        ):
            raise ValueError("actual reciprocal candidate missing or topology changed")
        timeline = self.timelines.get(preparation.gpu_id)
        if timeline is None:
            raise ValueError("actual reciprocal GPU timeline missing")
        entries = tuple(
            entry
            for entry in getattr(timeline, "active_entries", ())
            if not entry.is_completed and not entry.is_predicted and not entry.is_init
        )
        rollbacks = tuple(
            _ReciprocalEntryRollback(
                entry=entry,
                predicted_end_time=float(entry.predicted_end_time),
                total_base_work_sec=entry.total_base_work_sec,
                remaining_base_work_sec=entry.remaining_base_work_sec,
                last_accounted_mono=entry.last_accounted_mono,
                current_multiplier=float(entry.current_reciprocal_multiplier),
                evidence_epoch=int(entry.reciprocal_evidence_epoch),
                execution_attempt_id=str(entry.execution_attempt_id or ""),
                cancel_at=getattr(entry, "reciprocal_cancel_at", None),
            )
            for entry in entries
        )
        now_wall = time.time()
        now_mono = time.monotonic()
        try:
            candidate.execution_attempt_id = preparation.event_id
            candidate.total_base_work_sec = preparation.candidate_base_work_sec
            candidate.remaining_base_work_sec = preparation.candidate_base_work_sec
            candidate.last_accounted_mono = now_mono
            candidate.current_reciprocal_multiplier = 1.0
            candidate.config_fingerprint = str(config_fingerprint or "")
            candidate.input_fingerprint = str(input_fingerprint or "")
            candidate.gpu_model = str(gpu_model or "")
            candidate.mps_mode = str(mps_mode or "")
            candidate.worker_backend = str(worker_backend or "")
            candidate.actor_model = str(actor_model or "")
            candidate.adapter_version = str(adapter_version or "")
            candidate.reciprocal_cancel_at = preparation.candidate_cancel_at
            stats = self.reconcile_reciprocal_event(
                [preparation.gpu_id],
                "dispatch",
                uncertainty_z=preparation.uncertainty_z,
                now_wall=now_wall,
                now_mono=now_mono,
            )
            if stats["gpus"] != 1 or candidate.predicted_end_time <= now_wall:
                raise ValueError("actual reciprocal candidate missing from evaluation")
        except Exception:
            for rollback in rollbacks:
                rollback.entry.predicted_end_time = rollback.predicted_end_time
                rollback.entry.total_base_work_sec = rollback.total_base_work_sec
                rollback.entry.remaining_base_work_sec = (
                    rollback.remaining_base_work_sec
                )
                rollback.entry.last_accounted_mono = rollback.last_accounted_mono
                rollback.entry.current_reciprocal_multiplier = (
                    rollback.current_multiplier
                )
                rollback.entry.reciprocal_evidence_epoch = rollback.evidence_epoch
                rollback.entry.execution_attempt_id = rollback.execution_attempt_id
                rollback.entry.reciprocal_cancel_at = rollback.cancel_at
            timeline._bump_state_version()
            raise

        self._applied_reciprocal_dispatch_events[preparation.event_id] = (
            _AppliedReciprocalAttempt(preparation, rollbacks)
        )
        before_by_task = {
            str(rollback.entry.task_id): rollback.predicted_end_time
            for rollback in rollbacks
        }
        incumbent_delays = [
            max(
                0.0,
                float(entry.predicted_end_time) - before_by_task[str(entry.task_id)],
            )
            for entry in entries
            if str(entry.task_id) != preparation.task_id
        ]
        total_delay = sum(incumbent_delays)
        reciprocal["applied_incumbents"] = sum(
            delay > 0.0 for delay in incumbent_delays
        )
        reciprocal["applied_total_sec"] = float(total_delay)
        reciprocal["applied_at"] = now_wall
        reciprocal["committed_timeline_version"] = int(timeline._state_version)
        reciprocal["version_retried"] = bool(preparation.version_retried)
        for key in (
            "scanned_entries",
            "temporal_segments",
            "interference_lookups",
            "elapsed_ns",
        ):
            reciprocal[f"commit_{key}"] = int(stats.get(key, 0))
        return int(reciprocal["applied_incumbents"]), float(total_delay)

    def rollback_reciprocal_dispatch_start(
        self,
        preparation: ReciprocalDispatchPreparation,
    ) -> None:
        """Compensate a committed attempt before worker execution begins."""
        applied = self._applied_reciprocal_dispatch_events.pop(
            preparation.event_id, None
        )
        restored_timelines: set[Any] = set()
        if applied is not None:
            for rollback in applied.entry_rollbacks:
                rollback.entry.predicted_end_time = rollback.predicted_end_time
                rollback.entry.total_base_work_sec = rollback.total_base_work_sec
                rollback.entry.remaining_base_work_sec = (
                    rollback.remaining_base_work_sec
                )
                rollback.entry.last_accounted_mono = rollback.last_accounted_mono
                rollback.entry.current_reciprocal_multiplier = (
                    rollback.current_multiplier
                )
                rollback.entry.reciprocal_evidence_epoch = rollback.evidence_epoch
                rollback.entry.execution_attempt_id = rollback.execution_attempt_id
                rollback.entry.reciprocal_cancel_at = rollback.cancel_at
                timeline = self.timelines.get(str(rollback.entry.gpu_id))
                if timeline is not None:
                    restored_timelines.add(timeline)
        for timeline in restored_timelines:
            timeline._bump_state_version()
        init_task_id = f"{preparation.task_id}__init_phase"
        rollback_task_ids = {preparation.task_id, init_task_id}
        for current_gpu in getattr(self.timelines, "gpu_ids", ()):
            timeline = self.timelines.get(str(current_gpu))
            if timeline is None:
                continue
            before = len(timeline._entries)
            timeline._entries = [
                entry
                for entry in timeline._entries
                if not (
                    id(entry) not in preparation.preexisting_entry_ids
                    and str(entry.task_id) in rollback_task_ids
                )
            ]
            if len(timeline._entries) != before:
                timeline._rebuild_component_index()
                timeline._bump_state_version()
        existing_ids = {
            id(entry)
            for current_gpu in getattr(self.timelines, "gpu_ids", ())
            for entry in getattr(self.timelines.get(str(current_gpu)), "_entries", ())
        }
        for entry in preparation.rollback_entries:
            if id(entry) not in existing_ids:
                self.timelines.add_entry(entry)
                existing_ids.add(id(entry))

    def finish_reciprocal_attempt(
        self,
        task_id: str,
        event_id: str,
        gpu_id: str,
    ) -> bool:
        """Remove exact-once state only for the matching execution attempt."""
        applied = self._applied_reciprocal_dispatch_events.get(str(event_id))
        if applied is None:
            return False
        preparation = applied.preparation
        if preparation.task_id != str(task_id) or preparation.gpu_id != str(gpu_id):
            return False
        self._applied_reciprocal_dispatch_events.pop(str(event_id), None)
        return True

    def set_dynamic_batch_context(
        self,
        context: Any = None,
        *,
        borrowed: bool = False,
    ) -> None:
        """Install one solve-local context, owned here unless Phase lends it."""
        self._dynamic_batch_context = context
        self._owns_dynamic_batch_context = context is not None and not borrowed
        if context is not None:
            self._dynamic_batch_profiles.clear()
            self._dynamic_batch_blocked.clear()

    def set_dynamic_batch_profiles(
        self,
        profiles: dict[tuple[str, str], Any] | None = None,
        blocked: set[tuple[str, str]] | None = None,
    ) -> None:
        """Install one preselected solve-local profile per task/GPU."""
        self.set_dynamic_batch_context()
        self._dynamic_batch_profiles = dict(profiles or {})
        self._dynamic_batch_blocked = set(blocked or ())

    def dynamic_batch_profile(self, task_id: str, gpu_id: str) -> Any | None:
        key = (str(task_id), str(gpu_id))
        profile = getattr(self, "_dynamic_batch_profiles", {}).get(key)
        if profile is not None:
            return profile
        context = getattr(self, "_dynamic_batch_context", None)
        return context.active_profile(*key) if context is not None else None

    def dynamic_batch_blocked(self, task_id: str, gpu_id: str) -> bool:
        key = (str(task_id), str(gpu_id))
        if key in getattr(self, "_dynamic_batch_blocked", set()):
            return True
        context = getattr(self, "_dynamic_batch_context", None)
        return bool(
            context is not None
            and context.has_joint_candidates(key[0])
            and context.active_profile(*key) is None
        )

    def _record_candidate_eft_detail(
        self,
        task_id: str,
        gpu_id: str,
        detail: dict[str, float],
    ) -> None:
        context = getattr(self, "_dynamic_batch_context", None)
        if context is not None and context.active_profile(task_id, gpu_id) is not None:
            context.record_evaluation(task_id, gpu_id, detail)
            return
        self._candidate_eft_details[(str(task_id), str(gpu_id))] = dict(detail)

    def _candidate_eft_detail(
        self,
        task_id: str,
        gpu_id: str,
        option: Any = None,
    ) -> dict[str, float]:
        context = getattr(self, "_dynamic_batch_context", None)
        if context is not None and option is not None:
            return context.evaluation(task_id, gpu_id, option)
        return dict(self._candidate_eft_details.get((str(task_id), str(gpu_id)), {}))

    def plan(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float = 0.0,
        is_backfill: bool = False,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        logical_batch_size: int = 0,
        execution_overrides: dict[str, Any] | None = None,
    ) -> DispatchPlan:
        """Single-task convenience wrapper around solve().

        Used by the HTTP handler for individual task submission.
        Internally calls solve() with one task for consistency.
        """
        results = self.solve(
            [
                {
                    "task_id": task_id,
                    "campaign_id": campaign_id,
                    "component": component,
                    "input_size": input_size,
                    "logical_batch_size": int(logical_batch_size or 0),
                    "execution_overrides": dict(execution_overrides or {}),
                    "config_fingerprint": config_fingerprint,
                    "input_fingerprint": input_fingerprint,
                    "is_backfill": is_backfill,
                }
            ]
        )
        if not results:
            raise PlanningExhausted(
                f"Planner.solve returned empty for task_id={task_id}",
            )
        return results[0]


    def _evaluate_gpu(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        extra_same_component_active: int,
        *,
        task_id: str = "",
    ) -> float | None:
        """Call the placement strategy with current-task context when supported."""
        try:
            return self.placement.evaluate_gpu(
                component,
                input_size,
                gpu_id,
                campaign_id,
                extra_same_component_active,
                self,
                task_id=task_id,
            )
        except TypeError as exc:
            if "task_id" not in str(exc):
                raise
            return self.placement.evaluate_gpu(
                component,
                input_size,
                gpu_id,
                campaign_id,
                extra_same_component_active,
                self,
            )

    def _place_single(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float,
        is_backfill: bool,
        primary_budget: PrimaryDeadlineTracker,
        gpu_ids: list[str],
        current_solve: CurrentSolvePlacements,
        *,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        reciprocal_candidate_filter: Callable[[ReciprocalProjection], bool]
        | None = None,
        candidate_filter: Callable[[str, float, ReciprocalProjection | None, Any], bool]
        | None = None,
    ) -> DispatchPlan:
        """Place a single task on the best GPU (Phase 3b)."""
        cs = self.campaign_scheduler
        reciprocal_enabled = bool(self.reciprocal_interference_correction)
        candidate_config_fingerprint = str(config_fingerprint or "")
        if reciprocal_enabled and not candidate_config_fingerprint:
            candidate_config_fingerprint = str(
                self._campaign_component_fingerprint(campaign_id, component) or ""
            )
        candidate_input_fingerprint = str(input_fingerprint or "")
        candidates: list[
            tuple[str, float, float, ReciprocalProjection | None, Any]
        ] = []
        reciprocal_scoring_elapsed_ns = 0
        reciprocal_scoring_candidate_count = 0
        reciprocal_scanned_entries = 0
        reciprocal_temporal_segments = 0
        reciprocal_interference_lookups = 0
        context = self._dynamic_batch_context
        if context is not None and context.has_joint_candidates(task_id):
            joint_dynamic = True
            candidate_pairs = context.candidate_pairs(task_id, gpu_ids)
        else:
            joint_dynamic = False
            candidate_pairs = [(str(gpu_id), None) for gpu_id in gpu_ids]
        joint_context = context
        reciprocal_baseline_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}

        def _candidate_for_gpu(
            gpu_id: str,
            option: Any,
        ) -> tuple[str, float, float, ReciprocalProjection | None, Any] | None:
            nonlocal reciprocal_interference_lookups
            nonlocal reciprocal_scanned_entries
            nonlocal reciprocal_scoring_candidate_count
            nonlocal reciprocal_scoring_elapsed_ns
            nonlocal reciprocal_temporal_segments
            if option is not None:
                if joint_context is None:
                    raise RuntimeError("dynamic batch candidate context is missing")
                if candidate_filter is None and not joint_context.candidate_allowed(
                    task_id,
                    gpu_id,
                    option,
                    is_backfill=is_backfill,
                ):
                    return None
                joint_context.begin_evaluation(task_id, gpu_id, option)
            try:
                if is_backfill and not self._call_should_admit(
                    component,
                    input_size,
                    gpu_id,
                    campaign_id,
                    primary_budget.current(),
                    task_id,
                ):
                    return None
                eft = self._evaluate_gpu(
                    component,
                    input_size,
                    gpu_id,
                    campaign_id,
                    current_solve.count_component(gpu_id, component),
                    task_id=task_id,
                )
            finally:
                if option is not None and joint_context is not None:
                    joint_context.end_evaluation()
            if eft is None:
                return None
            if not reciprocal_enabled:
                if candidate_filter is not None and not candidate_filter(
                    gpu_id,
                    eft,
                    None,
                    option,
                ):
                    return None
                return (gpu_id, eft, eft, None, option)
            detail = self._candidate_eft_detail(task_id, gpu_id, option)
            required_detail = {
                "duration_sec",
                "slowdown_sec",
                "uncertainty_z",
                "vram_mb",
                "ram_mb",
            }
            missing_detail = sorted(required_detail.difference(detail))
            if missing_detail:
                raise ValueError(
                    f"candidate EFT metadata missing: {', '.join(missing_detail)}"
                )
            duration = float(detail["duration_sec"])
            candidate_environment = self._worker_environment_identity(
                component,
                gpu_id,
            )
            projection = _project_reciprocal_interference(
                self,
                current_solve,
                task_id=task_id,
                campaign_id=campaign_id,
                component=component,
                gpu_id=gpu_id,
                candidate_eft=eft,
                candidate_duration_sec=duration,
                uncertainty_z=float(detail["uncertainty_z"]),
                candidate_slowdown_sec=float(detail["slowdown_sec"]),
                candidate_config_fingerprint=candidate_config_fingerprint,
                candidate_input_fingerprint=candidate_input_fingerprint,
                candidate_gpu_model=candidate_environment["gpu_model"],
                candidate_mps_mode=candidate_environment["mps_mode"],
                candidate_worker_backend=candidate_environment["worker_backend"],
                candidate_actor_model=candidate_environment["actor_model"],
                candidate_adapter_version=candidate_environment["adapter_version"],
                baseline_cache=reciprocal_baseline_cache,
                query_results=self._reciprocal_query_results,
            )
            reciprocal_scoring_candidate_count += 1
            reciprocal_scoring_elapsed_ns += projection.scoring_elapsed_ns
            reciprocal_scanned_entries += projection.scanned_entries
            reciprocal_temporal_segments += projection.temporal_segments
            reciprocal_interference_lookups += projection.interference_lookups
            canonical_eft = projection.candidate_eft
            canonical_start = canonical_eft - projection.candidate_duration_sec
            if not self.campaign_scheduler._timelines.candidate_interval_fits(
                gpu_id,
                canonical_start,
                canonical_eft,
                float(detail["vram_mb"]),
                float(detail["ram_mb"]),
                exclude_task_id=task_id,
                candidate_component=component,
                candidate_config_fingerprint=candidate_config_fingerprint,
                candidate_input_size=input_size,
                _captured_now=(
                    float(self._solve_now_wall)
                    if isinstance(
                        getattr(self, "_solve_now_wall", None),
                        (int, float),
                    )
                    and math.isfinite(float(self._solve_now_wall))
                    else time.time()
                ),
            ):
                return None
            if (
                reciprocal_candidate_filter is not None
                and not reciprocal_candidate_filter(projection)
            ):
                return None
            if candidate_filter is not None and not candidate_filter(
                gpu_id,
                canonical_eft,
                projection,
                option,
            ):
                return None
            return (gpu_id, canonical_eft, canonical_eft, projection, option)

        def _select_candidate(
            choices: list[tuple[str, float, float, ReciprocalProjection | None, Any]],
        ) -> tuple[str, float, float, ReciprocalProjection | None, Any]:
            per_gpu: dict[
                str, tuple[str, float, float, ReciprocalProjection | None, Any]
            ] = {}
            for choice in choices:
                previous = per_gpu.get(choice[0])
                if previous is None or choice[2] < previous[2] - 1e-12:
                    per_gpu[choice[0]] = choice
            representatives = [
                per_gpu[str(gpu_id)] for gpu_id in gpu_ids if str(gpu_id) in per_gpu
            ]
            gpu = self._tie_break_select(
                component,
                [(choice[0], choice[1]) for choice in representatives],
            )
            return per_gpu[gpu]

        for gpu_id, option in candidate_pairs:
            if gpu_id in self._saturated_gpus:
                now = time.time()
                expiry = self._saturated_gpus[gpu_id]
                if expiry > now:
                    continue
                del self._saturated_gpus[gpu_id]
            candidate = _candidate_for_gpu(gpu_id, option)
            if candidate is not None:
                candidates.append(candidate)

        best_gpu: str | None = None
        best_eft = float("inf")
        _best_score = float("inf")
        best_projection: ReciprocalProjection | None = None
        best_option: Any = None
        if candidates:
            (
                best_gpu,
                best_eft,
                _best_score,
                best_projection,
                best_option,
            ) = _select_candidate(candidates)
            self._record_eft_sample(
                min(eft for _, eft, _score, _projection, _option in candidates)
            )

        if best_gpu is None:
            if is_backfill:
                raise PlanningExhausted(
                    f"Backfill {task_id}: no GPU admits it within primary deadline",
                )

            class _EvictionTaskShim:
                def __init__(
                    self,
                    tid,
                    cid,
                    comp,
                    isize,
                    gids,
                    vram,
                    ram,
                    duration,
                    config,
                ):
                    self.task_id = tid
                    self.campaign_id = cid
                    self.component = comp
                    self.input_size = isize
                    self.preferred_gpu_ids = list(gids)
                    self.arrival_time = time.time()
                    self.last_planning_attempt_at = time.time()
                    self.vram_budget_mb = vram
                    self.ram_budget_mb = ram
                    self.required_duration_sec = duration
                    self.config_fingerprint = config

            use_upper_resources = not isinstance(
                self.placement, DeterministicEFTPlacement
            )

            def _eviction_duration(
                latency_mean: Any,
                latency_sigma: Any,
                config_fp: str,
            ) -> float:
                mean = _safe_float(latency_mean, 0.0)
                if mean <= 0.0:
                    mean = _safe_float(
                        cs._cold_inference_latency_sec()
                        if hasattr(cs, "_cold_inference_latency_sec")
                        else 30.0,
                        30.0,
                    )
                sigma = max(0.0, _safe_float(latency_sigma, 0.0))
                compute = getattr(self.placement, "_compute_exec_time", None)
                if callable(compute):
                    alpha = HEFTPriority._get_alpha(
                        cs,
                        component,
                        input_size,
                        config_fp,
                    )
                    duration = _safe_float(compute(mean, sigma, alpha), mean)
                    if duration > 0.0:
                        return duration
                return max(0.0, mean)

            eviction_tasks: list[_EvictionTaskShim] = []
            if joint_dynamic:
                seen_requirements: set[tuple[str, float, float]] = set()
                for gpu_id, option in candidate_pairs:
                    if option is None:
                        continue
                    worker_name, _worker_addr, _needs_cold = self._resolve_worker(
                        component,
                        gpu_id,
                    )
                    if worker_name is None or not self.is_gpu_feasible_for_task(
                        gpu_id,
                        component,
                        worker_name,
                        0,
                    ):
                        continue
                    profile = option.profile
                    vram_est = float(
                        profile.vram_upper if use_upper_resources else profile.vram_mean
                    )
                    ram_est = float(
                        profile.ram_upper if use_upper_resources else profile.ram_mean
                    )
                    requirement = (str(gpu_id), vram_est, ram_est)
                    if requirement in seen_requirements:
                        continue
                    seen_requirements.add(requirement)
                    eviction_tasks.append(
                        _EvictionTaskShim(
                            task_id,
                            campaign_id,
                            component,
                            input_size,
                            [gpu_id],
                            vram_est,
                            ram_est,
                            _eviction_duration(
                                profile.latency_mean,
                                profile.latency_sigma,
                                candidate_config_fingerprint,
                            ),
                            candidate_config_fingerprint,
                        )
                    )
            else:
                ref_gpu = gpu_ids[0] if gpu_ids else ""
                try:
                    component_fp = self._campaign_component_fingerprint(
                        campaign_id,
                        component,
                    )
                except RuntimeError:
                    component_fp = ""
                vram_est = cs._predict_vram(
                    component,
                    input_size,
                    ref_gpu,
                    use_upper=use_upper_resources,
                    config_fingerprint=component_fp,
                )
                ram_est = cs._predict_ram(
                    component,
                    input_size,
                    ref_gpu,
                    use_upper=use_upper_resources,
                    config_fingerprint=component_fp,
                )
                latency_est = cs._predict_latency(
                    component,
                    input_size,
                    ref_gpu,
                    config_fingerprint=component_fp,
                )
                latency_sigma = cs._predict_latency_sigma(
                    component,
                    input_size,
                    ref_gpu,
                    config_fingerprint=component_fp,
                )
                eviction_tasks.append(
                    _EvictionTaskShim(
                        task_id,
                        campaign_id,
                        component,
                        input_size,
                        gpu_ids,
                        (
                            float(vram_est)
                            if isinstance(vram_est, (int, float))
                            else 0.0
                        ),
                        (float(ram_est) if isinstance(ram_est, (int, float)) else 0.0),
                        _eviction_duration(
                            latency_est,
                            latency_sigma,
                            component_fp,
                        ),
                        component_fp,
                    )
                )

            evict_success = any(
                self.try_evict_backfills_for(task, scope="cross_cycle")
                for task in eviction_tasks
            )
            if evict_success:
                now_ts = time.time()
                retry_candidates: list[
                    tuple[str, float, float, ReciprocalProjection | None, Any]
                ] = []
                for gpu_id, option in candidate_pairs:
                    if gpu_id in self._saturated_gpus:
                        if self._saturated_gpus[gpu_id] > now_ts:
                            continue
                        del self._saturated_gpus[gpu_id]
                    candidate = _candidate_for_gpu(gpu_id, option)
                    if candidate is not None:
                        retry_candidates.append(candidate)
                if retry_candidates:
                    (
                        best_gpu,
                        best_eft,
                        _best_score,
                        best_projection,
                        best_option,
                    ) = _select_candidate(retry_candidates)
                    self._record_eft_sample(
                        min(
                            eft
                            for _, eft, _score, _projection, _option in retry_candidates
                        ),
                    )
            if best_gpu is None:
                if self._constraint_tracker.is_component_exhausted(
                    component,
                    list(gpu_ids),
                ):
                    from .contracts import (
                        ConstraintViolation as _CV,
                    )
                    from .contracts import (
                        ViolationType as _VT,
                    )

                    raise PlanningExhausted(
                        _CV(
                            violation_type=_VT.COMPONENT_EXHAUSTED,
                            gpu_id="",
                            worker_name="",
                        ),
                    )
                raise PlanningExhausted(
                    f"Primary {task_id}: no feasible GPU after MCPSE eviction",
                )

        selected_context = None
        if best_option is not None:
            if context is None:
                raise RuntimeError("selected dynamic batch context is missing")
            selected_context = context
            selected_context.select(
                task_id,
                best_gpu,
                best_option,
                is_backfill=is_backfill,
            )
        candidate_duration = (
            best_projection.candidate_duration_sec
            if best_projection is not None
            else self._candidate_eft_detail(
                task_id,
                best_gpu,
                best_option,
            ).get("duration_sec")
        )
        try:
            plan = self._build_plan(
                task_id,
                campaign_id,
                component,
                input_size,
                is_backfill,
                best_gpu,
                best_eft,
                candidate_duration_sec=(
                    float(candidate_duration)
                    if candidate_duration is not None
                    else None
                ),
            )
        except Exception:
            if selected_context is not None:
                selected_context.release_selected(task_id)
            raise
        if not is_backfill:
            primary_budget.observe(campaign_id, best_eft)
        current_solve.record(
            plan.target_gpu_id,
            component,
            campaign_id,
            task_id=task_id,
        )
        if best_projection is not None:
            selected_environment = self._worker_environment_identity(
                component,
                best_gpu,
                str(plan.target_worker_name or ""),
            )
            predicted_entry = self.timelines.find_entry(str(task_id))
            if predicted_entry is not None:
                predicted_entry.reciprocal_base_duration_sec = float(
                    best_projection.candidate_base_duration_sec
                )
                predicted_entry.config_fingerprint = candidate_config_fingerprint
                predicted_entry.input_fingerprint = candidate_input_fingerprint
                predicted_entry.gpu_model = selected_environment["gpu_model"]
                predicted_entry.mps_mode = selected_environment["mps_mode"]
                predicted_entry.worker_backend = selected_environment["worker_backend"]
                predicted_entry.actor_model = selected_environment["actor_model"]
                predicted_entry.adapter_version = selected_environment[
                    "adapter_version"
                ]
            plan.worker_metadata.setdefault("planning_projection", {}).update(
                {
                    "input_fingerprint": candidate_input_fingerprint,
                    **selected_environment,
                }
            )
            plan.worker_metadata["reciprocal_interference"] = {
                "candidate_eft": float(best_projection.candidate_eft),
                "candidate_duration_sec": float(best_projection.candidate_duration_sec),
                "candidate_base_duration_sec": float(
                    best_projection.candidate_base_duration_sec
                ),
                "candidate_slowdown_sec": float(best_projection.candidate_slowdown_sec),
                "estimator_basis": self.interference_estimator_basis,
                "uncertainty_z": float(best_projection.uncertainty_z),
                "interference_drain_at": float(best_projection.interference_drain_at),
                "primary_tail_before": float(best_projection.primary_tail_before),
                "primary_tail_after": float(best_projection.primary_tail_after),
                "global_makespan_before": float(best_projection.global_makespan_before),
                "global_makespan_after": float(best_projection.global_makespan_after),
                "affected_incumbents": len(best_projection.incumbent_delays),
                "self_delay_total_sec": float(best_projection.self_delay_total_sec),
                "pair_delay_total_sec": float(best_projection.pair_delay_total_sec),
                "timeline_state_version": int(best_projection.timeline_state_version),
                "scoring_candidate_count": reciprocal_scoring_candidate_count,
                "scoring_elapsed_ns": reciprocal_scoring_elapsed_ns,
                "scanned_entries": reciprocal_scanned_entries,
                "temporal_segments": reciprocal_temporal_segments,
                "interference_lookups": reciprocal_interference_lookups,
            }
        return plan

    def _build_plan(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float,
        is_backfill: bool,
        best_gpu: str,
        best_eft: float,
        candidate_duration_sec: float | None = None,
    ) -> DispatchPlan:
        """Build a DispatchPlan + update scenario with projected entry."""
        cs = self.campaign_scheduler

        worker_name, worker_addr, needs_cold = self._resolve_worker(
            component,
            best_gpu,
        )
        if worker_name is None:
            raise PlanningExhausted(
                f"No viable worker spec for ({component}, {best_gpu}) — "
                f"all matching supervisor specs dead/excluded.  "
                f"fix recovery sweeper will reopen when "
                f"supervisor reports ready=True."
            )
        use_upper_resources = not isinstance(self.placement, DeterministicEFTPlacement)
        try:
            component_fp = self._campaign_component_fingerprint(
                campaign_id,
                component,
            )
        except RuntimeError:
            component_fp = ""
        batch_profile = self.dynamic_batch_profile(task_id, best_gpu)
        if batch_profile is not None:
            vram = float(
                batch_profile.vram_upper
                if use_upper_resources
                else batch_profile.vram_mean
            )
            ram = float(
                batch_profile.ram_upper
                if use_upper_resources
                else batch_profile.ram_mean
            )
        else:
            vram = cs._predict_vram(
                component,
                input_size,
                best_gpu,
                use_upper=use_upper_resources,
                config_fingerprint=component_fp,
            )
            ram = cs._predict_ram(
                component,
                input_size,
                best_gpu,
                use_upper=use_upper_resources,
                config_fingerprint=component_fp,
            )
            vram = float(vram) if isinstance(vram, (int, float)) else 0.0
            ram = float(ram) if isinstance(ram, (int, float)) else 0.0

        pre_inits = self.preinit.plan_preinit(
            task_id,
            component,
            best_gpu,
            campaign_id,
            best_eft,
            self,
        )

        now = time.time()
        latency = (
            float(batch_profile.latency_mean)
            if batch_profile is not None
            else (
                cs._predict_latency(
                    component,
                    input_size,
                    best_gpu,
                    config_fingerprint=component_fp,
                )
                or 0.0
            )
        )
        predicted_duration = max(
            0.0,
            float(candidate_duration_sec)
            if candidate_duration_sec is not None
            else float(latency),
        )
        start = best_eft - predicted_duration if best_eft < float("inf") else now
        end_time = best_eft if best_eft < float("inf") else now + latency
        try:
            _was_primary = bool(cs._is_primary(campaign_id))
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "_was_primary = False",
                exc_info=True,
            )
            _was_primary = False
        entry = cs._timelines.add_predicted_entry(
            task_id=task_id,
            component=component,
            gpu_id=best_gpu,
            worker_name=worker_name,
            start_time=max(now, start),
            predicted_end_time=end_time,
            predicted_vram_mb=vram,
            predicted_ram_mb=ram,
            campaign_id=campaign_id,
            is_backfill=is_backfill,
            is_dispatching=False,
            input_size=input_size,
            config_fingerprint=component_fp,
            was_primary_at_dispatch=_was_primary,
            logical_batch_size=(batch_profile.logical_n if batch_profile else 0),
            execution_batch_size=(batch_profile.k if batch_profile else 0),
        )

        tl = cs._timelines.get(best_gpu)
        assumptions = ConstraintAssumptions(
            assumed_available_vram_mb=(
                tl.available_vram_at(exclude_task_id=task_id) if tl else 0.0
            ),
            assumed_gpu_active_count=(tl.planned_occupancy_count() if tl else 0),
            assumed_worker_ready=not needs_cold,
            assumed_interference_slowdown=0.0,
            assumed_available_host_ram_mb=cs._timelines.available_host_ram_at(
                exclude_task_id=task_id,
            ),
        )

        worker_meta: dict[str, Any] = {}
        if batch_profile is not None:
            worker_meta["dynamic_batch_profile"] = batch_profile.as_dict()
        worker_meta["planning_projection"] = {
            "planned_start_time": float(entry.start_time),
            "predicted_end_time": float(end_time),
            "candidate_duration_sec": float(predicted_duration),
            "predicted_vram_mb": float(vram),
            "predicted_ram_mb": float(ram),
            "input_size": float(input_size or 0.0),
            "config_fingerprint": str(component_fp or ""),
            "was_primary_at_dispatch": bool(_was_primary),
        }
        sup = getattr(cs, "_supervisor", None)
        if sup and worker_name:
            st = sup.states.get(worker_name)
            if st:
                gpu_ids = [str(g) for g in (st.assigned_gpus or [])]
                worker_inflight = (
                    int(getattr(st, "queue_prepare_inflight", 0) or 0)
                    + int(getattr(st, "queue_execute_inflight", 0) or 0)
                    + int(getattr(st, "queue_finalize_inflight", 0) or 0)
                )
                hidden_backlog = (
                    int(getattr(st, "dispatch_pending", 0) or 0)
                    + int(getattr(st, "queue_in_queue", 0) or 0)
                    + int(getattr(st, "queue_prepare_inflight", 0) or 0)
                    + int(getattr(st, "queue_prepared_queue", 0) or 0)
                )
                worker_meta["estimator_worker_context"] = {
                    "worker_name": worker_name,
                    "gpu_ids": gpu_ids,
                    "active_request_count": max(1, worker_inflight + 1),
                    "queue_depth": max(0, hidden_backlog),
                    "dispatch_group_size": 1,
                    "co_location_signature": "solo_execute",
                    "hardware_software": f"worker:{worker_name}|gpus:{len(gpu_ids)}",
                    "residency_state": "hot" if (st.ready and st.addr) else "cold",
                }
                rbs = getattr(st, "resident_baseline_snapshot", None)
                if rbs:
                    worker_meta["resident_baseline_snapshot"] = dict(rbs)

        return DispatchPlan(
            task_id=task_id,
            campaign_id=campaign_id,
            component=component,
            target_gpu_id=best_gpu,
            target_worker_name=worker_name,
            target_worker_addr=worker_addr,
            vram_budget_mb=int(vram) if vram > 0 else 0,
            predicted_latency_sec=best_eft - now
            if best_eft < float("inf")
            else latency,
            is_backfill=is_backfill,
            needs_cold_start=needs_cold,
            ram_budget_mb=int(ram) if ram > 0 else 0,
            planned_start_time=float(entry.start_time),
            pre_init_schedule=pre_inits,
            constraint_assumptions=assumptions,
            worker_metadata=worker_meta,
        )


    def incorporate_constraint(self, violation: ConstraintViolation) -> bool:
        """Update world model based on Validator's gap detection.

        Returns:
            True  = world-model lattice progress made (constraint accumulated)
            False = no-op (already saturated or fallout drop)

        Raises:
            PlanningExhausted: on ``component_exhausted`` (re_plan infeasible).

        NOTE ( +  + ): ``gpu_redirect`` and
        ``backfill_admission_stale`` violation types are REMOVED —
        acquire_compute_slot is planned-GPU-only with no Validator
        re-routing, and Dispatcher cannot make admission decisions.
        The full 15-type taxonomy is handled by ``ConstraintTracker``
        (Plan).  This method delegates to the
        tracker and returns its monotone-lattice progress signal.
        """
        return self._constraint_tracker.incorporate(violation)


    def is_gpu_feasible_for_task(
        self,
        gpu_id: str,
        component: str,
        worker_name: str = "",
        vram_budget_mb: int = 0,
    ) -> bool:
        """5-step feasibility gate ( Issue 4/5,  RMA 2-signal).

        Planner's ``solve()`` / ``evaluate_gpu`` MUST call this before
        computing EFT — otherwise infeasible GPUs enter the optimization
        and the resulting plan is rejected at dispatch time, causing a
        re-plan loop.

        The 5 gates (in order):
          1. GPU saturation TTL
          2. GPU unhealthy TTL + mandatory_reset + RMA 2-signal
          3. (component, gpu) activation_excluded TTL
          4. target_worker dead/excluded/unreachable
          5. VRAM feasibility against projected scenario
        """

        class _TaskInfoShim:
            def __init__(self, comp: str, worker: str, vram: int) -> None:
                self.component = comp
                self.target_worker_name = worker
                self.vram_budget_mb = vram

        return self._constraint_tracker.is_gpu_feasible_for_task(
            gpu_id,
            _TaskInfoShim(component, worker_name, vram_budget_mb),
        )

    def on_reset_completed(self, gpu_id: str) -> None:
        """GPU reset completion signal (from GpuHealthObserver / admin).

        Clears ``_gpu_reset_required`` but NOT ``_rma_qualifying_gpus`` —
        those require ``admin_confirm_rma_complete`` ( P0-SAFETY).
        Also fires the ``gpu_health_change`` wake trigger so the
        SchedulingSupervisor re-evaluates previously-blocked tasks.
        """
        self._constraint_tracker.on_reset_completed(gpu_id)
        self._fire_health_wake()

    def admin_confirm_rma_complete(self, gpu_id: str) -> None:
        """Hardware RMA / diagnostic completion signal ( P0-SAFETY).

        Re-admission of Xid 64 GPUs requires BOTH ``on_reset_completed``
        and this explicit operator confirmation (2-signal invariant).
        """
        self._constraint_tracker.admin_confirm_rma_complete(gpu_id)
        self._fire_health_wake()

    def attach_supervisor_wake(self, supervisor: Any) -> None:
        """Register a ``SchedulingSupervisor`` for wake triggers.

        After this call, ``on_reset_completed`` /
        ``admin_confirm_rma_complete`` fire ``gpu_health_change`` wakes
        on the supervisor.
        """
        self._supervisor_wake_hook = supervisor

    def _fire_health_wake(self) -> None:
        sup = getattr(self, "_supervisor_wake_hook", None)
        if sup is None:
            return
        notify = getattr(sup, "notify_wake", None)
        if callable(notify):
            try:
                notify("gpu_health_change")
            except Exception:
                _LOG.warning(
                    "[global-planner] notify_wake failed",
                    exc_info=True,
                )


    EXACT_MCPSE_THRESHOLD: int = 8

    def try_evict_backfills_for(
        self,
        task_info: Any,
        scope: str = "cross_cycle",
    ) -> bool:
        _mcpse_start = time.time()
        try:
            return self._try_evict_backfills_for_impl(task_info, scope)
        finally:
            _mcpse_elapsed_ms = (time.time() - _mcpse_start) * 1000.0
            if _mcpse_elapsed_ms > 20.0:
                _LOG.warning(
                    "[hot-path] try_evict_backfills_for task=%s scope=%s elapsed=%.1fms",
                    str(getattr(task_info, "task_id", "?"))[:8],
                    scope,
                    _mcpse_elapsed_ms,
                )

    def _try_evict_backfills_for_impl(
        self,
        task_info: Any,
        scope: str = "cross_cycle",
    ) -> bool:
        """Select a backfill subset to evict for *task_info*.

        Returns ``True`` if the injected ``EvictionStrategy`` selected a
        non-empty subset and the eviction markers were applied; ``False``
        otherwise (caller raises ``PlanningExhausted``).

        This is a **Planner-internal** method — Validator / Core Loop MUST
        NOT invoke it directly.  ``scope="cross_cycle"`` filters by the
        task's ``last_planning_attempt_at`` so only backfills started after
        the primary's first planning attempt are considered.

        Cost / benefit model (Plan):
          cost_α = C_sunk + C_reinit + C_rerun
          benefit = urgency × ΔEST
          net = benefit − Σ cost_α

        + refactor: eviction decisions are delegated to
        ``self.eviction_strategy`` (``GreedyEvictionStrategy`` by default,
        ``MCPSEEvictionStrategy`` as ablation).  Shared helpers still compute
        costs, urgency, no-eviction EFT, and timeline/cancel side effects.
        """
        task_id_short = str(getattr(task_info, "task_id", "?"))[:8]
        component = str(getattr(task_info, "component", "") or "")
        strategy_name = type(self.eviction_strategy).__name__
        ram_strategy_name = type(self.ram_eviction_strategy).__name__
        if isinstance(self.eviction_strategy, NoEviction) and isinstance(
            self.ram_eviction_strategy,
            NoRamEviction,
        ):
            self._log_info_throttled(
                (
                    "eviction-disabled",
                    component,
                    scope,
                    strategy_name,
                    ram_strategy_name,
                ),
                "[eviction] DISABLED primary=%s scope=%s strategy=%s "
                "ram_strategy=%s — skipping candidate scan",
                task_id_short,
                scope,
                strategy_name,
                ram_strategy_name,
            )
            return False

        no_candidate_cache_key = self._eviction_negative_cache_key(
            task_info,
            scope,
            [],
        )
        if self._eviction_negative_cache_hit(no_candidate_cache_key):
            self._log_info_throttled(
                (
                    "eviction-negative-cache-hit",
                    component,
                    scope,
                    strategy_name,
                    ram_strategy_name,
                    0,
                ),
                "[eviction] NEGATIVE_CACHE_HIT primary=%s scope=%s strategy=%s "
                "ram_strategy=%s candidates=0 — unchanged no-candidate "
                "eviction trial, returning SKIP_THIS_CYCLE",
                task_id_short,
                scope,
                strategy_name,
                ram_strategy_name,
            )
            return False

        candidates = self._collect_eviction_candidates(task_info, scope)
        negative_cache_key = self._eviction_negative_cache_key(
            task_info,
            scope,
            candidates,
        )

        if self._eviction_negative_cache_hit(negative_cache_key):
            self._log_info_throttled(
                (
                    "eviction-negative-cache-hit",
                    component,
                    scope,
                    strategy_name,
                    ram_strategy_name,
                    len(candidates),
                ),
                "[eviction] NEGATIVE_CACHE_HIT primary=%s scope=%s strategy=%s "
                "ram_strategy=%s candidates=%d — unchanged failed eviction "
                "trial, returning SKIP_THIS_CYCLE",
                task_id_short,
                scope,
                strategy_name,
                ram_strategy_name,
                len(candidates),
            )
            return False

        if not candidates:
            self._log_info_throttled(
                (
                    "eviction-no-candidates",
                    component,
                    scope,
                    strategy_name,
                    ram_strategy_name,
                ),
                "[eviction] NO_CANDIDATES primary=%s scope=%s strategy=%s "
                "ram_strategy=%s "
                "— no eligible backfills on preferred GPUs (Planner returns "
                "False → SKIP_THIS_CYCLE)",
                task_id_short,
                scope,
                strategy_name,
                ram_strategy_name,
            )
            self._record_eviction_negative(no_candidate_cache_key, "no-candidates")
            return False

        alpha = self._get_confidence_alpha()
        cost = {bf.task_id: self._compute_eviction_cost(bf, alpha) for bf in candidates}
        urgency = self._compute_urgency(task_info)
        eft_no_evict = float("inf")

        best_S, best_net = self.eviction_strategy.select_backfills_to_evict(
            task_info=task_info,
            candidates=candidates,
            cost=cost,
            urgency=urgency,
            eft_no_evict=eft_no_evict,
            planner=self,
        )
        if not best_S:
            try:
                ram_avail = float(self.timelines.available_host_ram_at() or 0.0)
            except Exception:
                ram_avail = 0.0
            ram_needed = float(getattr(task_info, "ram_budget_mb", 0.0) or 0.0)
            if max(0.0, ram_needed - ram_avail) > 0.0:
                best_S, best_net = self.ram_eviction_strategy.select_backfills_to_evict(
                    task_info=task_info,
                    candidates=candidates,
                    cost=cost,
                    urgency=urgency,
                    eft_no_evict=eft_no_evict,
                    planner=self,
                )

        if not best_S:
            total_cost = sum(cost.values())
            self._log_info_throttled(
                (
                    "eviction-reject",
                    component,
                    strategy_name,
                    ram_strategy_name,
                    len(candidates),
                ),
                "[eviction] REJECT primary=%s strategy=%s ram_strategy=%s urgency=%.3f "
                "eft_no_evict=%.1fs candidates=%d total_cost=%.1fs "
                "— no feasible eviction subset",
                task_id_short,
                strategy_name,
                ram_strategy_name,
                urgency,
                eft_no_evict,
                len(candidates),
                total_cost,
            )
            self._record_eviction_negative(negative_cache_key, "reject")
            return False

        applied = 0
        for victim in best_S:
            if self._apply_eviction_mark(victim):
                applied += 1
        if applied != len(best_S):
            _LOG.warning(
                "[eviction] only %d/%d selected victims accepted cancel scheduling "
                "for primary=%s",
                applied,
                len(best_S),
                task_id_short,
            )
            if applied == 0:
                self._record_eviction_negative(
                    negative_cache_key,
                    "cancel-not-applied",
                )
            return False

        cache = getattr(self, "_eviction_negative_cache", None)
        if cache is not None:
            cache.clear()

        _LOG.info(
            "[eviction] evicted %d backfills for primary=%s "
            "(strategy=%s, ram_strategy=%s, net_benefit=%.2fs, urgency=%.3f, candidates=%d)",
            applied,
            task_id_short,
            strategy_name,
            ram_strategy_name,
            best_net,
            urgency,
            len(candidates),
        )
        return True

    def _eviction_negative_cache_key(
        self,
        task_info: Any,
        scope: str,
        candidates: list[Any],
    ) -> tuple[Any, ...]:
        """Fingerprint an eviction trial whose negative result is reusable.

        The key intentionally excludes the primary task id.  Fan-out-heavy
        phases often present many same-component tasks with the same resource
        need against the same active candidate set; a failed eviction trial for
        one such task is reusable for its siblings.  Active-candidate
        identities and per-GPU timeline versions still fence correctness:
        dispatch/complete/cancel/predicted-prune mutations bump the timeline
        version, and candidate add/remove or resource prediction changes alter
        the candidate tuple.
        """
        preferred = tuple(
            sorted(
                str(g)
                for g in (
                    getattr(task_info, "preferred_gpu_ids", None)
                    or list(self.timelines.gpu_ids)
                )
            )
        )
        versions: list[tuple[str, int, int]] = []
        for gpu_id in preferred:
            tl = self.timelines.get(str(gpu_id))
            if tl is None:
                versions.append((str(gpu_id), -1, 0))
                continue
            versions.append(
                (
                    str(gpu_id),
                    int(getattr(tl, "_state_version", 0) or 0),
                    int(getattr(tl, "active_count", 0) or 0),
                )
            )
        cand_sig = tuple(
            sorted(
                (
                    str(getattr(e, "task_id", "") or ""),
                    str(getattr(e, "gpu_id", "") or ""),
                    str(getattr(e, "component", "") or ""),
                    bool(getattr(e, "is_predicted", False)),
                    bool(getattr(e, "is_dispatching", False)),
                    round(float(getattr(e, "predicted_vram_mb", 0.0) or 0.0), 1),
                    round(float(getattr(e, "predicted_ram_mb", 0.0) or 0.0), 1),
                    round(float(getattr(e, "start_time", 0.0) or 0.0), 3),
                )
                for e in candidates
            )
        )
        return (
            str(scope or ""),
            str(getattr(task_info, "component", "") or ""),
            str(getattr(task_info, "campaign_id", "") or ""),
            round(float(getattr(task_info, "vram_budget_mb", 0.0) or 0.0), 1),
            round(float(getattr(task_info, "ram_budget_mb", 0.0) or 0.0), 1),
            round(float(getattr(task_info, "required_duration_sec", 0.0) or 0.0), 3),
            str(getattr(task_info, "config_fingerprint", "") or ""),
            preferred,
            tuple(versions),
            cand_sig,
        )

    def _eviction_negative_cache_hit(self, key: tuple[Any, ...]) -> bool:
        cache = getattr(self, "_eviction_negative_cache", None)
        if cache is None:
            self._eviction_negative_cache = {}
            return False
        entry = cache.get(key)
        if entry is None:
            return False
        ts, _reason = entry
        ttl = float(
            getattr(
                self,
                "_eviction_negative_cache_ttl_sec",
                2.0,
            )
            or 2.0
        )
        if (time.time() - ts) <= ttl:
            return True
        cache.pop(key, None)
        return False

    def _record_eviction_negative(
        self,
        key: tuple[Any, ...],
        reason: str,
    ) -> None:
        cache = getattr(self, "_eviction_negative_cache", None)
        if cache is None:
            cache = {}
            self._eviction_negative_cache = cache
        cache[key] = (time.time(), str(reason or "reject"))
        max_entries = int(
            getattr(
                self,
                "_eviction_negative_cache_max",
                4096,
            )
            or 4096
        )
        if len(cache) <= max_entries:
            return
        excess = len(cache) - max_entries
        for old_key, _ in sorted(cache.items(), key=lambda kv: kv[1][0])[:excess]:
            cache.pop(old_key, None)


    def _collect_eviction_candidates(
        self,
        task_info: Any,
        scope: str,
    ) -> list[Any]:
        """Gather backfill entries eligible for eviction (Plan).

        Scans active entries on the task's preferred GPUs and filters by
        scope. ``cross_cycle`` cheaply admits eligible new and pre-existing
        backfills; the eviction strategy decides whether removal helps without
        repeating a full EFT solve per candidate. ``current_cycle`` keeps only
        entries marked ``admitted_this_cycle``.
        """
        cs = self.campaign_scheduler
        preferred = getattr(task_info, "preferred_gpu_ids", None) or list(
            self.timelines.gpu_ids
        )
        eg_policy = getattr(cs, "_eviction_grace_policy", None)
        recent = self._recent_eviction_at
        now = time.time()

        def _within_grace(comp: str, gpu_id: str) -> bool:
            if eg_policy is None or not getattr(eg_policy, "enabled", False):
                return False
            last = recent.get((comp, gpu_id))
            if last is None:
                return False
            try:
                grace = float(eg_policy.grace_sec(comp, gpu_id, self))
            except Exception:
                _LOG.warning(
                    "[mcpse] EvictionGrace.grace_sec raised for (%s, %s)",
                    comp,
                    gpu_id,
                    exc_info=True,
                )
                return False
            if grace <= 0.0:
                return False
            return (now - last) < grace

        candidates: list[Any] = []
        for gpu_id in preferred:
            tl = cs._timelines.get(gpu_id)
            if tl is None:
                continue
            for e in getattr(tl, "active_entries", []):
                if not getattr(e, "is_backfill", False):
                    continue
                if getattr(e, "is_init", False):
                    continue
                if getattr(e, "is_predicted", False) and getattr(
                    e, "is_dispatching", False
                ):
                    continue
                if _within_grace(
                    getattr(e, "component", "") or "",
                    getattr(e, "gpu_id", "") or gpu_id,
                ):
                    continue
                if not getattr(e, "is_predicted", False) and not getattr(
                    e, "active_cancel_safe", True
                ):
                    continue
                if scope == "cross_cycle":
                    candidates.append(e)
                elif scope == "current_cycle":
                    if not getattr(e, "admitted_this_cycle", False):
                        continue
                    candidates.append(e)
        return candidates

    def _compute_eviction_cost(self, bf: Any, alpha: float) -> float:
        """``cost_α(b) = C_sunk + C_reinit + C_rerun`` (Plan D6).

        Falls back to conservative zeros when GP posterior fields are
        missing so the sum never becomes negative.
        """
        elapsed = max(0.0, float(getattr(bf, "elapsed", 0.0) or 0.0))
        c_sunk = elapsed

        if self._has_warm_alternative(
            getattr(bf, "component", ""),
            exclude_gpu=getattr(bf, "gpu_id", ""),
        ):
            c_reinit = 0.0
        else:
            mu_init, sigma_init = self._init_posterior(
                getattr(bf, "component", ""),
                getattr(bf, "gpu_id", ""),
            )
            c_reinit = mu_init + _norm_ppf(alpha) * sigma_init

        mu_rem, sigma_rem = self._remaining_posterior(bf)
        remaining_alpha = mu_rem + _norm_ppf(alpha) * sigma_rem
        p_abandon = self._estimate_abandon_probability(bf)
        c_rerun = 1.0 * remaining_alpha * p_abandon

        return max(0.0, c_sunk + c_reinit + c_rerun)

    def _compute_urgency(self, task_info: Any) -> float:
        """Plan ``urgency(P) = b_rank(P) / max_b_rank``.

        Normalizes b_rank across all currently-pending primaries.  Plan
         B2 acknowledges single-pending-primary collapses to 1.0;
        there is no absolute anchor available in the current scheduler
        (no ``T_max_wait`` configuration), so this implementation matches
        the plan text directly: ``b_rank / max(b_rank across pending)``.
        """
        try:
            rank = float(
                self.priority.compute_rank(
                    getattr(task_info, "component", ""),
                    getattr(task_info, "campaign_id", ""),
                    self,
                )
            )
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "return 1.0",
                exc_info=True,
            )
            return 1.0
        if rank <= 0.0:
            return 0.0

        cs = self.campaign_scheduler
        queues = getattr(cs, "_campaign_queues", {}) or {}
        pending_ranks: list[float] = []
        for cq in queues.values():
            try:
                dag = getattr(cq, "dag_context", None)
                components = list(
                    getattr(dag, "component_order", None)
                    or getattr(dag, "components", [])
                    or []
                )
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "components = []",
                    exc_info=True,
                )
                components = []
            for comp in components:
                try:
                    r = float(
                        self.priority.compute_rank(
                            comp,
                            cq.campaign_id,
                            self,
                        )
                    )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed an exception; body=%s",
                        __name__,
                        "continue",
                        exc_info=True,
                    )
                    continue
                if r > 0.0:
                    pending_ranks.append(r)
        max_rank = max(pending_ranks) if pending_ranks else rank
        if max_rank <= 0.0:
            return 1.0
        return max(0.0, min(1.0, rank / max_rank))

    _DEFAULT_REENQUEUE_WAIT_SEC: float = 30.0

    def _estimate_abandon_probability(self, bf: Any) -> float:
        """Exponential survival model (  /  P12-6).

        ``P_abandon = 1 − exp(−expected_wait / deadline_margin)``.  Returns
        ``0.0`` when the campaign deadline is disabled (infinite margin).

        Plan naming-mismatch fix: the prior
        ``getattr(bf, "expected_reenqueue_wait", 30.0)`` masked the
        fact that ``TimelineEntry`` has no such attribute — the
        getattr default was the real value.  Until
        ``SchedulingSupervisor.mean_backfill_wait_time`` is implemented
        per plan, use the named constant
        ``_DEFAULT_REENQUEUE_WAIT_SEC`` so the fallback is auditable.
        """
        deadline = getattr(bf, "campaign_deadline", None)
        if deadline is None or deadline == float("inf"):
            return 0.0
        now = time.time()
        margin = max(1.0, float(deadline) - now)
        expected_wait = self._DEFAULT_REENQUEUE_WAIT_SEC
        return 1.0 - math.exp(-expected_wait / margin)

    def _call_should_admit(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
        primary_deadline: float,
        task_id: str = "",
    ) -> bool:
        """Backfill admission delegate.  Phase D ablation archive: 
        ``PrimaryWeightedCancellation`` weight_fn injection retired.
        """
        self._dynamic_batch_admission_task_id = str(task_id or "")
        try:
            return self.backfill.should_admit(
                component,
                input_size,
                gpu_id,
                campaign_id,
                primary_deadline,
                self,
            )
        finally:
            self._dynamic_batch_admission_task_id = ""

    def _campaign_component_fingerprint(
        self,
        campaign_id: str,
        component: str,
    ) -> str:
        cs = self.campaign_scheduler
        queues = getattr(cs, "_campaign_queues", None)
        if queues is None:
            return ""
        cq = queues.get(campaign_id)
        if cq is None:
            return ""
        helper = getattr(cq, "_campaign_component_fingerprint", None)
        if helper is None:
            return ""
        return str(helper(component) or "").strip()

    def _has_warm_alternative(self, component: str, exclude_gpu: str) -> bool:
        return any(
            str(gpu_id) != str(exclude_gpu)
            and self._has_warm_on_gpu(component, str(gpu_id))
            for gpu_id in self.timelines.gpu_ids
        )

    def _has_warm_on_gpu(self, component: str, gpu_id: str) -> bool:
        """True iff a warm worker for ``component`` is assigned to ``gpu_id``.

        Plan (post): determines whether a backfill admission
        on ``gpu_id`` will incur cold-start latency.  Distinct from
        ``_has_warm_alternative`` (which probes warm workers on *other*
        GPUs for MCPSE re-execution cost — those probe sets are deliberately
        complementary, never reused interchangeably).
        """
        if getattr(self.campaign_scheduler, "_supervisor", None) is None:
            return False
        worker_name, worker_addr, needs_cold = self._resolve_worker(
            component,
            gpu_id,
        )
        return bool(worker_name and worker_addr and not needs_cold)

    def _init_posterior(self, component: str, gpu_id: str) -> tuple[float, float]:
        cs = self.campaign_scheduler
        mu = 0.0
        sigma = 0.0
        if hasattr(cs, "_get_init_latency"):
            try:
                mu = float(cs._get_init_latency(component, gpu_id) or 0.0)
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "mu = 0.0",
                    exc_info=True,
                )
                mu = 0.0
        if hasattr(cs, "_get_init_sigma"):
            try:
                sigma = float(cs._get_init_sigma(component, gpu_id) or 0.0)
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "sigma = 0.0",
                    exc_info=True,
                )
                sigma = 0.0
        return mu, sigma

    def _remaining_posterior(self, bf: Any) -> tuple[float, float]:
        """Plan ``_compute_eviction_cost`` — GP posterior for a
        backfill's remaining duration.

        Plan naming-mismatch fix (analogous to fix):
        ``TimelineEntry`` exposes ``remaining`` (wall-clock seconds
        left until ``predicted_end_time``) and ``predicted_duration``;
        there are no ``remaining_mu`` / ``remaining_sigma`` fields.
        The prior ``getattr(bf, "remaining_mu", 0.0)`` silently
        returned 0.0, collapsing C_rerun to 0 for every MCPSE
        candidate and under-estimating eviction cost.

        Resolution: derive μ from the scenario-maintained
        ``bf.remaining`` (clock-based, monotonic) and query GP σ from
        ``campaign_scheduler._predict_latency_sigma`` at the entry's
        (component, input_size, gpu_id).  Fall back to 0.0 only when
        the GP has no posterior for that key (explicit, logged).
        """
        cs = self.campaign_scheduler
        component = str(getattr(bf, "component", "") or "").strip().lower()
        gpu_id = str(getattr(bf, "gpu_id", "") or "")
        input_size = _safe_float(getattr(bf, "input_size", 0.0))
        try:
            mu = _safe_float(getattr(bf, "remaining", 0.0))
        except Exception:
            _LOG.warning(
                "[mcpse] failed to read bf.remaining for cost computation",
                exc_info=True,
            )
            mu = 0.0
        sigma = 0.0
        predict_sigma = getattr(cs, "_predict_latency_sigma", None)
        if callable(predict_sigma) and component and input_size > 0:
            try:
                sigma = _safe_float(predict_sigma(component, input_size, gpu_id))
            except Exception:
                _LOG.warning(
                    "[mcpse] _predict_latency_sigma raised for (%s, %s, %s) "
                    "— falling back to sigma=0",
                    component,
                    input_size,
                    gpu_id,
                    exc_info=True,
                )
                sigma = 0.0
        return mu, sigma

    def _get_confidence_alpha(self) -> float:
        cs = self.campaign_scheduler
        for attr in ("_confidence_alpha", "confidence_alpha"):
            val = getattr(cs, attr, None)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s:%d (%s)",
                        __name__,
                        0,
                        "swallowed_pass",
                        exc_info=True,
                    )
        return 0.95

    @staticmethod
    def _cancel_task_failed(task: Any) -> bool:
        try:
            if task.cancelled():
                return True
            return task.result() is False
        except Exception:
            return True

    def _rollback_eviction_mark_if_cancel_failed(
        self,
        task: Any,
        task_id: str,
        killed_at_token: float,
        predicted_end_time: float,
    ) -> None:
        if not self._cancel_task_failed(task):
            return
        restore_fn = getattr(self.timelines, "restore_killed_entry", None)
        restored = False
        if callable(restore_fn):
            try:
                restored = bool(
                    restore_fn(
                        task_id,
                        killed_at_token=killed_at_token,
                        predicted_end_time=predicted_end_time,
                    ),
                )
            except Exception:
                _LOG.warning(
                    "[MCPSE] failed to rollback eviction marker task=%s",
                    task_id,
                    exc_info=True,
                )
                return
        _LOG.warning(
            "[MCPSE] cancel task failed; rollback eviction marker task=%s restored=%s",
            task_id,
            restored,
        )

    def _remove_predicted_eviction_projection(self, task_id: str) -> int:
        """Remove plan-only predicted entries without touching workers.

        Predicted lookahead entries reserve planner capacity but have not
        reached the worker.  Evicting them should only discard the projection;
        sending a worker cancel creates noisy ``cancel returned False`` events
        for synthetic ids such as ``__lookahead_*``.
        """
        removed = 0
        timelines = getattr(self.timelines, "_timelines", {}) or {}
        for tl in timelines.values():
            entries = getattr(tl, "_entries", None)
            if entries is None:
                continue
            before = len(entries)
            tl._entries = [
                e
                for e in entries
                if not (
                    getattr(e, "task_id", "") == task_id
                    and getattr(e, "is_predicted", False)
                    and not getattr(e, "is_dispatching", False)
                    and not getattr(e, "is_completed", False)
                )
            ]
            delta = before - len(tl._entries)
            if delta:
                removed += delta
                rebuild = getattr(tl, "_rebuild_component_index", None)
                bump = getattr(tl, "_bump_state_version", None)
                if callable(rebuild):
                    rebuild()
                if callable(bump):
                    bump()
        return removed

    def _apply_eviction_mark(self, victim: Any) -> bool:
        """Mark an eviction victim: record the eviction on the timeline
        and signal the campaign scheduler's cancel path.

        Two steps:
          1. ``campaign_scheduler.mark_for_eviction`` / the supervisor's
             ``CancelBatch`` path kills the in-flight task on the worker.
          2. ``scenario.mark_entry_killed(task_id)`` flags the **active**
             timeline entry as killed (sets ``killed_at``, freezes
             ``predicted_end_time``) and drops pure predictions for the
             same task_id.  The killed active entry **stays on the
             timeline** so the ops dashboard renders the eviction as a
             red-X badge — operators can see MCPSE evict / cancel events
             actually landed.  Previously ``remove_entry`` stripped the
             entry silently, leaving eviction invisible on the Gantt.
             Idempotent (no-op on a second call once the entry is
             already killed).

        Side-effect (Plan Integration DG row consumer): records
        ``(component, gpu_id) → time.time()`` in
        ``self._recent_eviction_at`` so ``_collect_eviction_candidates``
        can honour ``EvictionGrace.grace_sec`` and skip repeat evictions
        within the grace window.  DG knob OFF → grace_sec == 0 →
        cooldown is a no-op.
        """
        task_id = getattr(victim, "task_id", None)
        if not task_id:
            return False
        try:
            existing = self.timelines.find_entry(task_id)
        except Exception:
            existing = None
        victim_is_plan_only = bool(getattr(victim, "is_predicted", False)) and not bool(
            getattr(victim, "is_dispatching", False)
        )
        existing_is_plan_only = (
            existing is not None
            and bool(getattr(existing, "is_predicted", False))
            and not bool(getattr(existing, "is_dispatching", False))
            and not bool(getattr(existing, "is_completed", False))
        )
        if victim_is_plan_only or existing_is_plan_only:
            removed = self._remove_predicted_eviction_projection(str(task_id))
            if removed:
                _LOG.debug(
                    "[MCPSE] removed %d predicted eviction projection(s) task=%s",
                    removed,
                    task_id,
                )
                return True
            _LOG.debug(
                "[MCPSE] predicted eviction projection missing task=%s",
                task_id,
            )
            return False
        if bool(getattr(victim, "is_init", False)) or (
            existing is not None and bool(getattr(existing, "is_init", False))
        ):
            _LOG.debug(
                "[MCPSE] skip eviction for init timeline marker task=%s",
                task_id,
            )
            return False
        mark_fn = getattr(self.campaign_scheduler, "mark_for_eviction", None)
        cancel_task = None
        if callable(mark_fn):
            try:
                cancel_task = mark_fn(task_id)
            except Exception as e:
                _LOG.warning("[MCPSE] mark_for_eviction failed: %s", e)
                return False
        if cancel_task is None:
            _LOG.warning(
                "[MCPSE] no cancel task scheduled for eviction task=%s",
                task_id,
            )
            return False
        killed = None
        predicted_end_before = 0.0
        try:
            predicted_end_before = float(
                getattr(existing, "predicted_end_time", 0.0) or 0.0,
            )
        except Exception:
            predicted_end_before = 0.0
        try:
            kill_fn = getattr(self.timelines, "mark_entry_killed", None)
            if callable(kill_fn):
                killed = kill_fn(task_id)
            else:
                remove_fn = getattr(self.timelines, "remove_entry", None)
                if callable(remove_fn):
                    remove_fn(task_id, missing_ok=True)
        except Exception as e:
            _LOG.warning("[MCPSE] scenario.mark_entry_killed failed: %s", e)
            return False
        if killed is not None:
            killed_at_token = float(getattr(killed, "killed_at", 0.0) or 0.0)
            try:

                def _rollback_if_failed(done: Any) -> None:
                    self._rollback_eviction_mark_if_cancel_failed(
                        done,
                        task_id,
                        killed_at_token,
                        predicted_end_before,
                    )

                add_done_callback = getattr(cancel_task, "add_done_callback", None)
                if not callable(add_done_callback):
                    raise TypeError("eviction cancel task has no done callback")
                add_done_callback(_rollback_if_failed)
            except Exception:
                _LOG.warning(
                    "[MCPSE] failed to attach eviction rollback callback task=%s",
                    task_id,
                    exc_info=True,
                )
        try:
            comp = getattr(victim, "component", "") or ""
            gpu_id = getattr(victim, "gpu_id", "") or ""
            if comp and gpu_id:
                self._recent_eviction_at[(comp, gpu_id)] = time.time()
        except Exception:
            _LOG.warning(
                "[MCPSE] recording recent_eviction_at failed",
                exc_info=True,
            )
        return True

    def re_plan(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float,
        is_backfill: bool,
        violation: ConstraintViolation,
    ) -> DispatchPlan:
        """Re-plan after constraint violation, with updated world model.

         architectural compliance: ``incorporate_constraint`` updates the
        world model (VRAM correction / saturation TTL / worker exclusion);
        the Planner then re-runs ``solve()`` on a fresh scenario snapshot.
        No ``gpu_redirect``-style override — the Planner re-decides placement
        holistically based on the corrected world model.
        """
        if violation.violation_type == "host_ram_saturated":
            try:
                sync_host_ram = getattr(
                    self.campaign_scheduler, "_sync_host_ram_capacity", None
                )
                if callable(sync_host_ram):
                    sync_host_ram()
            except Exception:
                _LOG.warning(
                    "[host-ram-replan] failed to sync host RAM capacity before replanning",
                    exc_info=True,
                )
        entry = self.timelines.find_entry(task_id)
        config_fingerprint = str(getattr(entry, "config_fingerprint", "") or "")
        input_fingerprint = str(getattr(entry, "input_fingerprint", "") or "")
        self.incorporate_constraint(violation)
        return self.plan(
            task_id,
            campaign_id,
            component,
            input_size,
            is_backfill,
            config_fingerprint,
            input_fingerprint,
        )



    def _worker_environment_identity(
        self,
        component: str,
        gpu_id: str,
        worker_name: str = "",
    ) -> dict[str, str]:
        """Return known execution identity without inventing missing evidence."""
        identity = {
            "gpu_model": str(self._gpu_models.get(str(gpu_id), "") or ""),
            "mps_mode": "",
            "worker_backend": "",
            "actor_model": "",
            "adapter_version": "",
        }
        sup = getattr(self.campaign_scheduler, "_supervisor", None)
        states = getattr(sup, "states", None)
        if not isinstance(states, dict):
            return identity
        state = states.get(worker_name) if worker_name else None
        if state is None:
            for candidate in states.values():
                spec = getattr(candidate, "spec", None)
                assigned = list(
                    getattr(candidate, "assigned_gpus", None)
                    or getattr(spec, "gpus", None)
                    or []
                )
                if getattr(spec, "component", None) == component and str(gpu_id) in {
                    str(item) for item in assigned
                }:
                    state = candidate
                    break
        if state is None:
            return identity
        spec = getattr(state, "spec", None)
        worker_server = dict(getattr(spec, "worker_server", None) or {})
        configured_mps = worker_server.get("cuda_mps")
        mps_enabled = True if configured_mps is None else configured_mps is True
        identity.update(
            {
                "mps_mode": "enabled" if mps_enabled else "disabled",
                "worker_backend": "persistent_actor",
                "actor_model": "shared_cuda_ipc",
                "adapter_version": str(
                    getattr(getattr(state, "caps", None), "model_version", "") or ""
                ),
            }
        )
        return identity

    def _resolve_worker(
        self,
        component: str,
        gpu_id: str,
    ) -> tuple[str | None, str | None, bool]:
        """Find worker for (component, gpu_id). Returns (name, addr, needs_cold).

        Plan fix amendment — **no synthetic cold-name
        fallback**.  Previously this helper returned
        ``f"{component}_cold_{gpu_id}"`` as a placeholder when no real
        worker spec matched OR when all matching specs were in
        ``_dead_workers`` / ``_excluded_workers``.  The Validator then
        tried to activate that synthetic name, which had no
        registered spec → ``activation_failed`` → exclusion of
        ``(component, gpu_id)``.  Compounding over multiple GPUs this
        produced ``component_exhausted`` → TASK_FAILED cascade
        observed in the benchmark.

        Behaviour changes:
          - **Real spec exists + ready + addr** → return real
            ``(name, addr, needs_cold=False)``.
          - **Real spec exists but cold (not ready / no addr)** →
            return real ``(name, None, needs_cold=True)`` so Stage 4
            activates the existing spec (supervisor can cold-start).
          - **No matching real spec at all (empty component pool for
            GPU)** → return ``(name=None, addr=None, needs_cold=True)``.
            Caller treats ``name is None`` as infeasible for
            ``(component, gpu_id)``; Planner picks another GPU or
            raises ``PlanningExhausted (transient)`` → SKIP_THIS_CYCLE.
            No spurious ``activation_failed`` exclusion registers.
          - **All matching specs are in ``_dead_workers``/
            ``_excluded_workers``** → same as "no matching spec",
            return ``(None, None, True)``.  fix recovery
            sweeper will clear ``_dead_workers`` when supervisor
            restores ``ready=True``, reopening this pair naturally.
        """
        sup = getattr(self.campaign_scheduler, "_supervisor", None)
        if not sup:
            return f"{component}_0", None, True

        states_iter = getattr(sup, "states", {}) or {}
        has_any_spec = False
        tracker = getattr(self, "_constraint_tracker", None)
        for st in states_iter.values():
            spec = getattr(st, "spec", None)
            if spec is None or getattr(spec, "component", None) != component:
                continue
            assigned = [str(g) for g in (getattr(st, "assigned_gpus", None) or [])]
            if gpu_id not in assigned:
                continue
            has_any_spec = True
            spec_name = getattr(spec, "name", "")
            if spec_name in self._dead_workers or spec_name in self._excluded_workers:
                continue
            cold_marker = False
            if tracker is not None and hasattr(
                tracker, "worker_needs_cold_on_next_plan"
            ):
                try:
                    cold_marker = tracker.worker_needs_cold_on_next_plan(spec_name)
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed an exception; body=%s",
                        __name__,
                        "cold_marker = False",
                        exc_info=True,
                    )
                    cold_marker = False
            if cold_marker:
                return spec_name, None, True
            if getattr(st, "ready", False) and getattr(st, "addr", ""):
                return spec_name, st.addr, False
            else:
                return spec_name, None, True

        if has_any_spec:
            return None, None, True
        return f"{component}_cold_{gpu_id}", None, True

    def _build_primary_deadline_tracker(
        self,
        primary_cq: Any,
        gpu_ids: list[str],
    ) -> PrimaryDeadlineTracker:
        """Solve-scoped projected primary budget.

        Seed from current timeline state, then allow `_place_single()`
        to tighten it with primary plans committed earlier in the same
        solve() pass.
        """
        tracker = PrimaryDeadlineTracker(
            primary_campaign_id=(
                str(getattr(primary_cq, "campaign_id", "") or "")
                if primary_cq is not None
                else ""
            ),
            deadline_ts=float("inf"),
        )
        if not tracker.primary_campaign_id:
            return tracker

        now = time.time()
        for gpu_id in gpu_ids:
            tl = self.timelines.get(gpu_id)
            if not tl:
                continue
            for e in tl.active_entries:
                if (
                    str(getattr(e, "campaign_id", "") or "")
                    != tracker.primary_campaign_id
                ):
                    continue
                predicted_end = float(
                    getattr(e, "predicted_end_time", float("inf")) or float("inf")
                )
                if predicted_end > now:
                    tracker.observe(tracker.primary_campaign_id, predicted_end)
        return tracker

    def on_profile_drift(self, component: str, metric: str) -> None:
        """Invalidate caches and apply new evidence prospectively."""
        if hasattr(self.priority, "invalidate"):
            self.priority.invalidate()
        affected_gpu_ids = [
            str(gpu_id)
            for gpu_id in getattr(self.timelines, "gpu_ids", ())
            if any(
                str(getattr(entry, "component", "") or "") == str(component)
                and not entry.is_completed
                and not entry.is_predicted
                for entry in getattr(
                    self.timelines.get(str(gpu_id)), "active_entries", ()
                )
            )
        ]
        self.reconcile_reciprocal_event(
            affected_gpu_ids,
            "profile_drift",
            component=component,
            metric=metric,
        )




def _is_stale_predicted_entry(entry: Any, now: float) -> bool:
    return (
        bool(getattr(entry, "is_predicted", False))
        and not bool(getattr(entry, "is_dispatching", False))
        and float(getattr(entry, "predicted_end_time", float("inf")) or float("inf"))
        < float(now)
    )


@dataclass
class _ReciprocalSweepTask:
    spec: ReciprocalTaskOverlay
    start_mono: float
    remaining_base_work_sec: float
    delta_index: int
    completed_mono: float | None = None
    cancelled_mono: float | None = None


def _reciprocal_total_base(entry: Any) -> float:
    total = getattr(entry, "total_base_work_sec", None)
    if total is None:
        total = max(
            0.0,
            _safe_float(getattr(entry, "predicted_end_time", 0.0))
            - _safe_float(getattr(entry, "start_time", 0.0)),
        )
    total = _safe_float(total, -1.0)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError(f"invalid reciprocal base work: {total!r}")
    return total


def _account_reciprocal_entry(
    entry: Any,
    *,
    now_wall: float,
    now_mono: float,
) -> None:
    """Consume elapsed monotonic time exactly once from authoritative base work."""
    total = _reciprocal_total_base(entry)
    remaining_value = getattr(entry, "remaining_base_work_sec", None)
    remaining = total if remaining_value is None else _safe_float(remaining_value, -1.0)
    multiplier = _safe_float(
        getattr(entry, "current_reciprocal_multiplier", 1.0),
        -1.0,
    )
    if (
        not math.isfinite(remaining)
        or remaining < 0.0
        or remaining > total + 1e-6
        or not math.isfinite(multiplier)
        or multiplier < 1.0
    ):
        raise ValueError(f"invalid reciprocal execution state: {entry.task_id}")
    anchor = getattr(entry, "last_accounted_mono", None)
    if anchor is None:
        elapsed_wall = max(
            0.0,
            float(now_wall) - _safe_float(getattr(entry, "start_time", now_wall)),
        )
        anchor = float(now_mono) - elapsed_wall
    elapsed = max(0.0, float(now_mono) - _safe_float(anchor, now_mono))
    entry.total_base_work_sec = total
    entry.remaining_base_work_sec = max(0.0, remaining - elapsed / multiplier)
    entry.last_accounted_mono = float(now_mono)


def _entry_reciprocal_overlay(
    entry: Any,
    *,
    now_wall: float,
    now_mono: float,
) -> ReciprocalTaskOverlay:
    total = _reciprocal_total_base(entry)
    remaining_value = getattr(entry, "remaining_base_work_sec", None)
    remaining = total if remaining_value is None else _safe_float(remaining_value, -1.0)
    multiplier = _safe_float(
        getattr(entry, "current_reciprocal_multiplier", 1.0),
        -1.0,
    )
    if (
        not math.isfinite(remaining)
        or remaining < -1e-6
        or remaining > total + 1e-6
        or not math.isfinite(multiplier)
        or multiplier < 1.0
    ):
        raise ValueError(f"invalid reciprocal overlay state: {entry.task_id}")
    if not getattr(entry, "is_predicted", False):
        anchor = getattr(entry, "last_accounted_mono", None)
        if anchor is None:
            elapsed_wall = max(
                0.0,
                float(now_wall) - _safe_float(getattr(entry, "start_time", now_wall)),
            )
            anchor = float(now_mono) - elapsed_wall
        remaining -= (
            max(
                0.0,
                float(now_mono) - _safe_float(anchor, now_mono),
            )
            / multiplier
        )
        if not math.isfinite(remaining):
            raise ValueError(f"invalid reciprocal overlay state: {entry.task_id}")
        remaining = max(0.0, remaining)
    return ReciprocalTaskOverlay(
        task_id=str(getattr(entry, "task_id", "") or ""),
        execution_attempt_id=str(getattr(entry, "execution_attempt_id", "") or ""),
        component=str(getattr(entry, "component", "") or ""),
        campaign_id=str(getattr(entry, "campaign_id", "") or ""),
        gpu_id=str(getattr(entry, "gpu_id", "") or ""),
        start_wall=_safe_float(getattr(entry, "start_time", now_wall)),
        total_base_work_sec=total,
        remaining_base_work_sec=max(0.0, remaining),
        current_multiplier=multiplier,
        last_accounted_mono=float(now_mono),
        evidence_epoch=_safe_int(getattr(entry, "reciprocal_evidence_epoch", 0)),
        is_init=bool(getattr(entry, "is_init", False)),
        is_predicted=bool(getattr(entry, "is_predicted", False)),
        config_fingerprint=str(getattr(entry, "config_fingerprint", "") or ""),
        input_fingerprint=str(getattr(entry, "input_fingerprint", "") or ""),
        gpu_model=str(getattr(entry, "gpu_model", "") or ""),
        mps_mode=str(getattr(entry, "mps_mode", "") or ""),
        worker_backend=str(getattr(entry, "worker_backend", "") or ""),
        actor_model=str(getattr(entry, "actor_model", "") or ""),
        adapter_version=str(getattr(entry, "adapter_version", "") or ""),
        cancel_wall=getattr(entry, "reciprocal_cancel_at", None),
    )


def _reciprocal_delta(
    registry: Any,
    victim: ReciprocalTaskOverlay,
    interferer: ReciprocalTaskOverlay | None,
    *,
    self_n: int,
    uncertainty_z: float,
    query_results: dict[tuple[int, Any], Any] | None = None,
) -> tuple[float, int]:
    if registry is None:
        return 0.0, victim.evidence_epoch
    query_api = getattr(registry, "query_reciprocal", None)
    if callable(query_api):
        from gateway.signals.contracts import ReciprocalInterferenceQuery

        query = ReciprocalInterferenceQuery(
            victim_component=victim.component,
            victim_config_fingerprint=victim.config_fingerprint,
            victim_input_fingerprint=victim.input_fingerprint,
            interferer_component=(interferer.component if interferer else ""),
            interferer_config_fingerprint=(
                interferer.config_fingerprint if interferer else ""
            ),
            interferer_input_fingerprint=(
                interferer.input_fingerprint if interferer else ""
            ),
            gpu_id=victim.gpu_id,
            gpu_model=victim.gpu_model,
            mps_mode=victim.mps_mode,
            worker_backend=victim.worker_backend,
            actor_model=victim.actor_model,
            adapter_version=victim.adapter_version,
            self_n=max(1, int(self_n)),
            pair_multiplicity=1,
        )
        query_key = (_safe_int(getattr(registry, "evidence_epoch", 0)), query)
        if query_results is not None and query_key in query_results:
            result = query_results[query_key]
        else:
            result = query_api(query)
            if query_results is not None:
                query_results[query_key] = result
        delta = _safe_float(getattr(result, "delta", 0.0), -1.0)
        uncertainty = _safe_float(getattr(result, "uncertainty", 0.0), -1.0)
        epoch = _safe_int(getattr(result, "evidence_epoch", 0))
        if (
            not math.isfinite(delta)
            or delta < 0.0
            or not math.isfinite(uncertainty)
            or uncertainty < 0.0
        ):
            raise ValueError("invalid reciprocal interference result")
        return delta + max(0.0, float(uncertainty_z)) * uncertainty, epoch
    if interferer is None:
        return (
            _effective_self_delta(
                registry,
                victim.component,
                n_concurrent=max(1, int(self_n)),
                fp=victim.config_fingerprint,
                uncertainty_z=uncertainty_z,
            ),
            victim.evidence_epoch,
        )
    return (
        _effective_pair_delta(
            registry,
            victim.component,
            interferer.component,
            fp=victim.config_fingerprint,
            uncertainty_z=uncertainty_z,
        ),
        victim.evidence_epoch,
    )


def _evaluate_reciprocal_entries(
    planner: GlobalPlanner,
    gpu_id: str,
    entries: tuple[Any, ...],
    *,
    overlays: tuple[ReciprocalTaskOverlay, ...] = (),
    exclude_task_ids: frozenset[str] = frozenset(),
    actual_incumbents_only: bool = False,
    uncertainty_z: float = 0.0,
    now_wall: float | None = None,
    now_mono: float | None = None,
    timeline_state_version: int = 0,
    query_results: dict[tuple[int, Any], Any] | None = None,
) -> ReciprocalEvaluation:
    """Canonical additive directed event sweep over remaining base work."""
    wall = float(time.time() if now_wall is None else now_wall)
    mono = float(time.monotonic() if now_mono is None else now_mono)
    specs: dict[str, ReciprocalTaskOverlay] = {}
    for entry in entries:
        task_id = str(getattr(entry, "task_id", "") or "")
        if (
            not task_id
            or task_id in exclude_task_ids
            or getattr(entry, "is_completed", False)
            or getattr(entry, "is_invalidated", False)
            or getattr(entry, "is_evict_masked", False)
            or getattr(entry, "is_init", False)
            or _is_stale_predicted_entry(entry, wall)
            or (
                actual_incumbents_only
                and getattr(entry, "is_predicted", False)
                and not getattr(entry, "is_dispatching", False)
            )
        ):
            continue
        specs[task_id] = _entry_reciprocal_overlay(
            entry,
            now_wall=wall,
            now_mono=mono,
        )
    for overlay in overlays:
        if overlay.task_id:
            specs[overlay.task_id] = overlay

    states: dict[str, _ReciprocalSweepTask] = {}
    for task_id, spec in specs.items():
        if str(spec.gpu_id) != str(gpu_id):
            continue
        start_mono = mono + max(0.0, float(spec.start_wall) - wall)
        if not spec.is_predicted and spec.start_wall <= wall:
            start_mono = mono
        states[task_id] = _ReciprocalSweepTask(
            spec=spec,
            start_mono=start_mono,
            remaining_base_work_sec=max(0.0, spec.remaining_base_work_sec),
            delta_index=len(states),
        )

    registry = getattr(
        getattr(planner.campaign_scheduler, "_signal_service", None),
        "interference_registry",
        None,
    )
    if query_results is None:
        query_results = {}
    pair_deltas: list[list[tuple[int, float, int] | None]] = [
        [None] * len(states) for _ in states
    ]
    self_deltas: list[dict[tuple[int, int], tuple[float, int]]] = [{} for _ in states]

    def _delta_for(
        victim: _ReciprocalSweepTask,
        interferer: _ReciprocalSweepTask | None,
        self_n: int,
    ) -> tuple[float, int]:
        if registry is not None and not callable(
            getattr(registry, "query_reciprocal", None)
        ):
            return _reciprocal_delta(
                registry,
                victim.spec,
                interferer.spec if interferer is not None else None,
                self_n=self_n,
                uncertainty_z=uncertainty_z,
                query_results=query_results,
            )
        registry_epoch = _safe_int(getattr(registry, "evidence_epoch", 0))
        victim_index = victim.delta_index
        if interferer is None:
            key = (registry_epoch, max(1, int(self_n)))
            cached_self = self_deltas[victim_index].get(key)
            if cached_self is not None:
                return cached_self
            result = _reciprocal_delta(
                registry,
                victim.spec,
                None,
                self_n=self_n,
                uncertainty_z=uncertainty_z,
                query_results=query_results,
            )
            self_deltas[victim_index][key] = result
            return result
        interferer_index = interferer.delta_index
        cached_pair = pair_deltas[victim_index][interferer_index]
        if cached_pair is not None and cached_pair[0] == registry_epoch:
            return cached_pair[1], cached_pair[2]
        delta, result_epoch = _reciprocal_delta(
            registry,
            victim.spec,
            interferer.spec,
            self_n=1,
            uncertainty_z=uncertainty_z,
            query_results=query_results,
        )
        pair_deltas[victim_index][interferer_index] = (
            registry_epoch,
            delta,
            result_epoch,
        )
        return delta, result_epoch

    segments: list[ReciprocalMultiplierSegment] = []
    first_multipliers: dict[str, float] = {}
    interference_lookups = 0
    evidence_epoch = max(
        (spec.evidence_epoch for spec in specs.values()),
        default=_safe_int(getattr(registry, "evidence_epoch", 0)),
    )
    cursor = mono
    max_events = max(4, len(states) * 4 + 1)
    events = 0
    epsilon = 1e-9

    def _cancel_mono(state: _ReciprocalSweepTask) -> float | None:
        if state.spec.cancel_wall is None:
            return None
        return mono + max(0.0, float(state.spec.cancel_wall) - wall)

    while any(
        state.completed_mono is None and state.cancelled_mono is None
        for state in states.values()
    ):
        events += 1
        if events > max_events:
            raise RuntimeError("reciprocal event sweep failed to converge")
        for state in states.values():
            cancel_mono = _cancel_mono(state)
            if (
                state.completed_mono is None
                and state.cancelled_mono is None
                and cancel_mono is not None
                and cancel_mono <= cursor + epsilon
            ):
                state.cancelled_mono = cancel_mono
            if (
                state.completed_mono is None
                and state.cancelled_mono is None
                and state.start_mono <= cursor + epsilon
                and state.remaining_base_work_sec <= epsilon
            ):
                state.remaining_base_work_sec = 0.0
                state.completed_mono = cursor
        unfinished = [
            state
            for state in states.values()
            if state.completed_mono is None and state.cancelled_mono is None
        ]
        if not unfinished:
            break
        active = [state for state in unfinished if state.start_mono <= cursor + epsilon]
        if not active:
            cursor = min(state.start_mono for state in unfinished)
            continue

        exec_active = [state for state in active if not state.spec.is_init]
        multipliers: dict[str, float] = {}
        for state in active:
            if state.spec.is_init:
                multipliers[state.spec.task_id] = 1.0
                continue
            same_n = sum(
                other.spec.component == state.spec.component for other in exec_active
            )
            multiplier = 1.0
            if same_n > 1:
                interference_lookups += 1
                delta, epoch = _delta_for(state, None, same_n)
                multiplier += delta
                evidence_epoch = max(evidence_epoch, epoch)
            for other in exec_active:
                if (
                    other.spec.task_id == state.spec.task_id
                    or other.spec.component == state.spec.component
                ):
                    continue
                interference_lookups += 1
                delta, epoch = _delta_for(state, other, 1)
                multiplier += delta
                evidence_epoch = max(evidence_epoch, epoch)
            if not math.isfinite(multiplier) or multiplier < 1.0:
                raise ValueError("invalid composed reciprocal multiplier")
            multipliers[state.spec.task_id] = multiplier
        for task_id, multiplier in multipliers.items():
            first_multipliers.setdefault(task_id, multiplier)

        boundaries = [
            state.start_mono
            for state in unfinished
            if state.start_mono > cursor + epsilon
        ]
        boundaries.extend(
            cancel_mono
            for state in unfinished
            if (cancel_mono := _cancel_mono(state)) is not None
            and cancel_mono > cursor + epsilon
        )
        completion_boundaries = [
            (
                cursor
                + state.remaining_base_work_sec * multipliers[state.spec.task_id],
                state,
            )
            for state in active
        ]
        coalesced = [
            state
            for boundary, state in completion_boundaries
            if boundary <= cursor + epsilon
        ]
        if coalesced:
            for state in coalesced:
                state.remaining_base_work_sec = 0.0
                state.completed_mono = cursor
            continue
        boundaries.extend(boundary for boundary, _state in completion_boundaries)
        next_cursor = min(boundaries)
        if not math.isfinite(next_cursor) or next_cursor <= cursor + epsilon:
            raise RuntimeError("reciprocal event sweep made no progress")
        elapsed = next_cursor - cursor
        for state in active:
            multiplier = multipliers[state.spec.task_id]
            state.remaining_base_work_sec = max(
                0.0,
                state.remaining_base_work_sec - elapsed / multiplier,
            )
            segments.append(
                ReciprocalMultiplierSegment(
                    task_id=state.spec.task_id,
                    start_mono=cursor,
                    end_mono=next_cursor,
                    multiplier=multiplier,
                )
            )
        cursor = next_cursor

    completions = tuple(
        sorted(
            (
                task_id,
                wall + float(state.completed_mono or mono) - mono,
            )
            for task_id, state in states.items()
            if state.completed_mono is not None
        )
    )
    return ReciprocalEvaluation(
        gpu_id=str(gpu_id),
        timeline_state_version=int(timeline_state_version),
        evidence_epoch=int(evidence_epoch),
        completion_wall_by_task=completions,
        multiplier_by_task=tuple(sorted(first_multipliers.items())),
        segments=tuple(segments),
        cancelled_task_ids=tuple(
            sorted(
                task_id
                for task_id, state in states.items()
                if state.cancelled_mono is not None
            )
        ),
        scanned_entries=len(entries) + len(overlays),
        interference_lookups=interference_lookups,
    )


def _project_reciprocal_interference(
    planner: GlobalPlanner,
    current_solve: CurrentSolvePlacements,
    *,
    task_id: str,
    campaign_id: str,
    component: str,
    gpu_id: str,
    candidate_eft: float,
    candidate_duration_sec: float,
    uncertainty_z: float,
    candidate_slowdown_sec: float = 0.0,
    now: float | None = None,
    actual_incumbents_only: bool = False,
    candidate_config_fingerprint: str = "",
    candidate_input_fingerprint: str = "",
    candidate_gpu_model: str = "",
    candidate_mps_mode: str = "",
    candidate_worker_backend: str = "",
    candidate_actor_model: str = "",
    candidate_adapter_version: str = "",
    candidate_start_at: float | None = None,
    candidate_cancel_at: float | None = None,
    baseline_cache: dict[tuple[Any, ...], tuple[Any, ...]] | None = None,
    query_results: dict[tuple[int, Any], Any] | None = None,
) -> ReciprocalProjection:
    """Project one candidate through the canonical read-only event sweep."""
    del current_solve
    scoring_started_ns = time.perf_counter_ns()
    cache_key = (
        str(task_id),
        bool(actual_incumbents_only),
        float(uncertainty_z),
        float(now) if now is not None else None,
    )
    cached_baseline = (
        baseline_cache.get(cache_key) if baseline_cache is not None else None
    )
    baseline_was_computed = cached_baseline is None
    entries_by_gpu: dict[str, tuple[Any, ...]] = {}
    baseline_by_gpu: dict[str, ReciprocalEvaluation] = {}
    if cached_baseline is None:
        now_wall = float(time.time() if now is None else now)
        now_mono = time.monotonic()
    else:
        now_wall = float(cached_baseline[0])
        now_mono = float(cached_baseline[1])
        entries_by_gpu = cached_baseline[2]
        baseline_by_gpu = cached_baseline[3]
    legacy_duration = max(0.0, float(candidate_duration_sec))
    legacy_slowdown = min(
        legacy_duration,
        max(0.0, float(candidate_slowdown_sec)),
    )
    candidate_base = max(0.0, legacy_duration - legacy_slowdown)
    candidate_start = max(
        now_wall,
        (
            float(candidate_start_at)
            if candidate_start_at is not None
            else float(candidate_eft) - legacy_duration
        ),
    )
    uncertainty_z = max(0.0, float(uncertainty_z))
    cs = planner.campaign_scheduler
    timelines = cs._timelines
    timeline = timelines.get(str(gpu_id))
    timeline_version = int(getattr(timeline, "_state_version", 0) or 0)
    primary_id = planner._cached_primary_id if planner._cached_primary_id_set else None
    if not planner._cached_primary_id_set:
        primary = cs._primary_campaign()
        primary_id = str(primary.campaign_id) if primary is not None else None

    if cached_baseline is None:
        for current_gpu in getattr(timelines, "gpu_ids", []):
            current_timeline = timelines.get(str(current_gpu))
            entries = tuple(
                getattr(current_timeline, "active_entries", ())
                if current_timeline
                else ()
            )
            entries_by_gpu[str(current_gpu)] = entries
            baseline_by_gpu[str(current_gpu)] = _evaluate_reciprocal_entries(
                planner,
                str(current_gpu),
                entries,
                exclude_task_ids=frozenset({str(task_id)}),
                actual_incumbents_only=actual_incumbents_only,
                uncertainty_z=uncertainty_z,
                now_wall=now_wall,
                now_mono=now_mono,
                timeline_state_version=int(
                    getattr(current_timeline, "_state_version", 0) or 0
                ),
                query_results=query_results,
            )
        if baseline_cache is not None:
            baseline_cache[cache_key] = (
                now_wall,
                now_mono,
                entries_by_gpu,
                baseline_by_gpu,
            )
    baseline = baseline_by_gpu.get(str(gpu_id))
    if baseline is None:
        baseline = _evaluate_reciprocal_entries(
            planner,
            str(gpu_id),
            (),
            uncertainty_z=uncertainty_z,
            now_wall=now_wall,
            now_mono=now_mono,
            timeline_state_version=timeline_version,
            query_results=query_results,
        )
    candidate = ReciprocalTaskOverlay(
        task_id=str(task_id),
        execution_attempt_id=str(task_id),
        component=str(component),
        campaign_id=str(campaign_id),
        gpu_id=str(gpu_id),
        start_wall=candidate_start,
        total_base_work_sec=candidate_base,
        remaining_base_work_sec=candidate_base,
        is_predicted=True,
        config_fingerprint=str(candidate_config_fingerprint or ""),
        input_fingerprint=str(candidate_input_fingerprint or ""),
        gpu_model=str(candidate_gpu_model or ""),
        mps_mode=str(candidate_mps_mode or ""),
        worker_backend=str(candidate_worker_backend or ""),
        actor_model=str(candidate_actor_model or ""),
        adapter_version=str(candidate_adapter_version or ""),
        cancel_wall=candidate_cancel_at,
    )
    projected = _evaluate_reciprocal_entries(
        planner,
        str(gpu_id),
        entries_by_gpu.get(str(gpu_id), ()),
        overlays=(candidate,),
        exclude_task_ids=frozenset({str(task_id)}),
        actual_incumbents_only=actual_incumbents_only,
        uncertainty_z=uncertainty_z,
        now_wall=now_wall,
        now_mono=now_mono,
        timeline_state_version=timeline_version,
        query_results=query_results,
    )

    baseline_completion = dict(baseline.completion_wall_by_task)
    projected_completion = dict(projected.completion_wall_by_task)
    target_entries = {
        str(getattr(entry, "task_id", "") or ""): entry
        for entry in entries_by_gpu.get(str(gpu_id), ())
        if not getattr(entry, "is_init", False)
    }
    delays = {
        incumbent_id: max(0.0, projected_completion[incumbent_id] - before)
        for incumbent_id, before in baseline_completion.items()
        if incumbent_id in projected_completion
        and incumbent_id in target_entries
        and projected_completion[incumbent_id] > before + 1e-9
    }
    candidate_completion = projected.completion_wall(str(task_id))
    candidate_cancelled = str(task_id) in projected.cancelled_task_ids
    if candidate_completion is None and not candidate_cancelled:
        raise RuntimeError("canonical reciprocal projection lost candidate")
    candidate_action_end = (
        float(candidate_completion)
        if candidate_completion is not None
        else max(candidate_start, float(candidate_cancel_at or candidate_start))
    )
    candidate_duration = max(0.0, candidate_action_end - candidate_start)
    candidate_slowdown = (
        max(0.0, candidate_duration - candidate_base)
        if candidate_completion is not None
        else 0.0
    )

    def _primary_tail(
        evaluations: dict[str, ReciprocalEvaluation],
        *,
        target_override: ReciprocalEvaluation | None = None,
    ) -> float:
        tail = 0.0
        for current_gpu, evaluation in evaluations.items():
            if target_override is not None and current_gpu == str(gpu_id):
                evaluation = target_override
            completions = dict(evaluation.completion_wall_by_task)
            for entry in entries_by_gpu.get(current_gpu, ()):
                if (
                    primary_id is not None
                    and not getattr(entry, "is_init", False)
                    and str(getattr(entry, "campaign_id", "") or "") == str(primary_id)
                ):
                    tail = max(
                        tail,
                        completions.get(str(getattr(entry, "task_id", "")), 0.0),
                    )
        if (
            target_override is not None
            and primary_id is not None
            and str(campaign_id) == str(primary_id)
            and candidate_completion is not None
        ):
            tail = max(tail, candidate_completion)
        return tail

    def _global_tail(
        evaluations: dict[str, ReciprocalEvaluation],
        *,
        target_override: ReciprocalEvaluation | None = None,
    ) -> float:
        tails = []
        for current_gpu, evaluation in evaluations.items():
            if target_override is not None and current_gpu == str(gpu_id):
                evaluation = target_override
            tails.extend(
                value for _task_id, value in evaluation.completion_wall_by_task
            )
        if target_override is not None and str(gpu_id) not in evaluations:
            tails.extend(
                value for _task_id, value in target_override.completion_wall_by_task
            )
        return max(tails, default=now_wall)

    primary_tail_before = _primary_tail(baseline_by_gpu)
    primary_tail_after = _primary_tail(
        baseline_by_gpu,
        target_override=projected,
    )
    global_makespan_before = _global_tail(baseline_by_gpu)
    global_makespan_after = _global_tail(
        baseline_by_gpu,
        target_override=projected,
    )
    interference_drain_at = max(
        baseline_completion.values(),
        default=now_wall,
    )
    self_total = sum(
        delay
        for incumbent_id, delay in delays.items()
        if str(getattr(target_entries[incumbent_id], "component", "") or "")
        == str(component)
    )
    pair_total = sum(delays.values()) - self_total
    measured_evaluations = (
        tuple(baseline_by_gpu.values()) if baseline_was_computed else ()
    ) + (projected,)
    return ReciprocalProjection(
        gpu_id=str(gpu_id),
        timeline_state_version=timeline_version,
        candidate_eft=candidate_action_end,
        candidate_duration_sec=candidate_duration,
        candidate_base_duration_sec=candidate_base,
        candidate_slowdown_sec=candidate_slowdown,
        uncertainty_z=uncertainty_z,
        interference_drain_at=float(interference_drain_at),
        primary_tail_before=float(primary_tail_before),
        primary_tail_after=float(primary_tail_after),
        global_makespan_before=float(global_makespan_before),
        global_makespan_after=float(global_makespan_after),
        incumbent_delays=tuple(sorted(delays.items())),
        incumbent_projected_latencies=tuple(
            sorted(
                (
                    task_id,
                    max(
                        0.0,
                        projected_completion[task_id]
                        - _entry_reciprocal_overlay(
                            entry, now_wall=now_wall, now_mono=now_mono
                        ).start_wall,
                    ),
                )
                for task_id, entry in target_entries.items()
                if task_id in projected_completion
            )
        ),
        self_delay_total_sec=float(self_total),
        pair_delay_total_sec=float(pair_total),
        scoring_elapsed_ns=max(0, time.perf_counter_ns() - scoring_started_ns),
        scanned_entries=sum(
            evaluation.scanned_entries for evaluation in measured_evaluations
        ),
        temporal_segments=sum(
            len(evaluation.segments) for evaluation in measured_evaluations
        ),
        interference_lookups=sum(
            evaluation.interference_lookups for evaluation in measured_evaluations
        ),
    )


def _base_work_overlap_interval(
    entry: Any,
    candidate_start: float,
    candidate_base_duration: float,
    *,
    require_base: bool = False,
) -> tuple[float, float] | None:
    entry_start = _safe_float(getattr(entry, "start_time", candidate_start))
    canonical_duration = max(
        0.0,
        _safe_float(getattr(entry, "predicted_end_time", 0.0)) - entry_start,
    )
    base_duration = getattr(entry, "reciprocal_base_duration_sec", None)
    if base_duration is None:
        if require_base and not bool(getattr(entry, "is_predicted", False)):
            raise ValueError(
                f"active reciprocal entry missing base duration: "
                f"{getattr(entry, 'task_id', '')}"
            )
        base_duration = canonical_duration
    overlap_start = max(entry_start, float(candidate_start))
    overlap_end = min(
        entry_start + max(0.0, _safe_float(base_duration)),
        float(candidate_start) + max(0.0, float(candidate_base_duration)),
    )
    if overlap_end <= overlap_start:
        return None
    return overlap_start, overlap_end


def _base_work_overlap(
    entry: Any,
    candidate_start: float,
    candidate_base_duration: float,
    *,
    require_base: bool = False,
) -> float:
    interval = _base_work_overlap_interval(
        entry,
        candidate_start,
        candidate_base_duration,
        require_base=require_base,
    )
    return 0.0 if interval is None else interval[1] - interval[0]


def _candidate_self_slowdown(
    registry: Any,
    component: str,
    incumbent_intervals: list[tuple[float, float]],
    *,
    fp: str,
    uncertainty_z: float,
) -> float:
    """Integrate candidate self slowdown at the actual N of each segment."""
    boundaries = sorted(
        {point for interval in incumbent_intervals for point in interval}
    )
    total = 0.0
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active = sum(a <= midpoint < b for a, b in incumbent_intervals)
        if active <= 0:
            continue
        total += _effective_self_delta(
            registry,
            component,
            n_concurrent=active + 1,
            fp=fp,
            uncertainty_z=uncertainty_z,
        ) * (end - start)
    return total


def _incumbent_self_marginal_delay(
    registry: Any,
    component: str,
    victim_interval: tuple[float, float],
    incumbent_intervals: list[tuple[float, float]],
    *,
    fp: str,
    uncertainty_z: float,
) -> float:
    """Integrate the victim's N→N+1 marginal over candidate overlap."""
    victim_start, victim_end = victim_interval
    boundaries = {victim_start, victim_end}
    for start, end in incumbent_intervals:
        if end > victim_start and start < victim_end:
            boundaries.add(max(victim_start, start))
            boundaries.add(min(victim_end, end))
    total = 0.0
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:], strict=False):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        pre_n = sum(a <= midpoint < b for a, b in incumbent_intervals)
        before = _effective_self_delta(
            registry,
            component,
            n_concurrent=max(1, pre_n),
            fp=fp,
            uncertainty_z=uncertainty_z,
        )
        after = _effective_self_delta(
            registry,
            component,
            n_concurrent=max(2, pre_n + 1),
            fp=fp,
            uncertainty_z=uncertainty_z,
        )
        total += max(0.0, after - before) * (end - start)
    return total


def _effective_self_delta(
    intf_reg: Any,
    component: str,
    *,
    n_concurrent: int,
    fp: str = "",
    uncertainty_z: float,
) -> float:
    if intf_reg is None or int(n_concurrent) <= 1:
        return 0.0
    sd_tuple: tuple[float, float] | None
    try:
        sd_tuple = intf_reg.get_self_slowdown_if_mature(
            component,
            n_concurrent=int(n_concurrent),
            fp=fp,
        )
    except Exception:
        _LOG.warning(
            "[silent-except] %s swallowed an exception; body=%s",
            __name__,
            "sd_tuple = None",
            exc_info=True,
        )
        sd_tuple = None
    if sd_tuple is None:
        prior_mean = float(getattr(intf_reg, "_self_slowdown_prior", 2.0))
        prior_var = float(getattr(intf_reg, "_slowdown_prior_variance", 0.25))
        sd_mean_delta = max(0.0, prior_mean - 1.0)
        sd_std_delta = math.sqrt(max(0.0, prior_var))
    else:
        sd_mean_delta, sd_std_delta = sd_tuple
    return max(
        0.0,
        sd_mean_delta + max(0.0, float(uncertainty_z)) * sd_std_delta,
    )


def _effective_pair_delta(
    intf_reg: Any,
    component: str,
    neighbor_component: str,
    *,
    fp: str = "",
    uncertainty_z: float,
) -> float:
    if intf_reg is None:
        return 0.0
    sd_tuple: tuple[float, float] | None
    try:
        sd_tuple = intf_reg.get_pairwise_slowdown_if_mature(
            component,
            neighbor_component,
            fp=fp,
        )
    except Exception:
        _LOG.warning(
            "[silent-except] %s swallowed an exception; body=%s",
            __name__,
            "sd_tuple = None",
            exc_info=True,
        )
        sd_tuple = None
    if sd_tuple is None:
        prior_mean = float(getattr(intf_reg, "_pairwise_slowdown_prior", 1.3))
        prior_var = float(getattr(intf_reg, "_slowdown_prior_variance", 0.25))
        sd_mean_delta = max(0.0, prior_mean - 1.0)
        sd_std_delta = math.sqrt(max(0.0, prior_var))
    else:
        sd_mean_delta, sd_std_delta = sd_tuple
    return max(
        0.0,
        sd_mean_delta + max(0.0, float(uncertainty_z)) * sd_std_delta,
    )


def _norm_cdf(z: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (rational approximation).

    Abramowitz & Stegun 26.2.23, |error| < 4.5e-4.
    """
    if p <= 0.0:
        return -5.0
    if p >= 1.0:
        return 5.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_norm_ppf(1.0 - p)

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
