"""Small phase/campaign wrapper for ``scheduler_implementation=proton_phase``.

``GlobalPlanner`` still owns validator/timeline contracts.  This class owns the
phase policy: primary campaign first, round-robin in-phase placement, residency
reuse/pre-init hints, and conservative finish-before-sync backfill.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, cast

from ..extraction.component_features import get_dynamic_batching
from .constraint_tracker import PlanningExhausted
from .contracts import DispatchPlan, PreInitAction
from .global_planner import CurrentSolvePlacements, GlobalPlanner
from .predictive_support import (
    _DYNAMIC_BATCH_SELECTORS as _DYNAMIC_BATCH_SELECTORS,
    DynamicBatchCandidate as DynamicBatchCandidate,
    PredictiveTaskSupport,
    _fixed_n as _fixed_n,
    _largest_safe as _largest_safe,
    _safe_float as _safe_float,
    _safe_int as _safe_int,
    _throughput_optimal as _throughput_optimal,
    primary_incumbent_slowdown_ratio,
)

_LOG = logging.getLogger(__name__)

_FREE_WINDOW_HORIZON_SEC = 3600.0


@dataclass(frozen=True)
class DynamicBatchOption:
    profile: DynamicBatchCandidate
    phase: str
    policy: str
    batch_size_arg: str

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "batch_size_arg": self.batch_size_arg,
            "logical_batch_size": self.profile.logical_n,
            "selected_batch_size": self.profile.k,
            "phase": self.phase,
            "policy": self.policy,
            "fallback_reason": "",
            "profile_source": self.profile.source,
        }


@dataclass
class DynamicBatchTaskCandidates:
    task: dict[str, Any]
    options_by_gpu: dict[str, tuple[DynamicBatchOption, ...]] = field(
        default_factory=dict
    )
    fallback_metadata: dict[str, Any] = field(default_factory=dict)
    fallback_mode: bool = False
    selected_gpu: str = ""
    selected_option: DynamicBatchOption | None = None
    selected_admission: Any = None
    evaluations: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)


@dataclass
class DynamicBatchSolveContext:
    """One synchronous solve-local owner for candidates and selection state."""

    owner: Any
    tasks: dict[str, DynamicBatchTaskCandidates] = field(default_factory=dict)
    envelope: Any = None
    active_task_id: str = ""
    active_gpu_id: str = ""
    active_option: DynamicBatchOption | None = None

    def has_joint_candidates(self, task_id: str) -> bool:
        task = self.tasks.get(str(task_id))
        return bool(task is not None and not task.fallback_mode)

    def candidate_pairs(
        self,
        task_id: str,
        gpu_ids: list[str],
    ) -> list[tuple[str, DynamicBatchOption]]:
        task = self.tasks.get(str(task_id))
        if task is None or task.fallback_mode:
            return []
        gpu_rank = {str(gpu_id): rank for rank, gpu_id in enumerate(gpu_ids)}
        pairs = [
            (str(gpu_id), option)
            for gpu_id in gpu_ids
            for option in task.options_by_gpu.get(str(gpu_id), ())
        ]
        return sorted(
            pairs,
            key=lambda pair: (
                -(pair[1].profile.logical_n / max(pair[1].profile.latency_mean, 1e-9)),
                -pair[1].profile.k,
                gpu_rank[pair[0]],
            ),
        )

    def begin_evaluation(
        self,
        task_id: str,
        gpu_id: str,
        option: DynamicBatchOption,
    ) -> None:
        self.active_task_id = str(task_id)
        self.active_gpu_id = str(gpu_id)
        self.active_option = option

    def end_evaluation(self) -> None:
        self.active_task_id = ""
        self.active_gpu_id = ""
        self.active_option = None

    def active_profile(self, task_id: str, gpu_id: str) -> DynamicBatchCandidate | None:
        task = self.tasks.get(str(task_id))
        if (
            task is not None
            and task.selected_option is not None
            and task.selected_gpu == str(gpu_id)
        ):
            return task.selected_option.profile
        if (
            self.active_option is not None
            and self.active_task_id == str(task_id)
            and self.active_gpu_id == str(gpu_id)
        ):
            return self.active_option.profile
        return None

    def record_evaluation(
        self,
        task_id: str,
        gpu_id: str,
        detail: dict[str, float],
    ) -> None:
        task = self.tasks.get(str(task_id))
        if task is None or self.active_option is None:
            return
        task.evaluations[(str(gpu_id), self.active_option.profile.k)] = dict(detail)

    def evaluation(
        self,
        task_id: str,
        gpu_id: str,
        option: DynamicBatchOption,
    ) -> dict[str, float]:
        task = self.tasks.get(str(task_id))
        return (
            dict(task.evaluations.get((str(gpu_id), option.profile.k), {}))
            if task is not None
            else {}
        )

    def candidate_allowed(
        self,
        task_id: str,
        gpu_id: str,
        option: DynamicBatchOption,
        *,
        is_backfill: bool,
    ) -> bool:
        if not is_backfill:
            return True
        find_slack = getattr(self.owner, "_find_backfill_slack", None)
        if not callable(find_slack):
            return True
        task = self.tasks[str(task_id)].task
        return (
            find_slack(
                task,
                self.envelope,
                consume=False,
                gpu_id=str(gpu_id),
                profile=option.profile,
            )
            is not None
        )

    def select(
        self,
        task_id: str,
        gpu_id: str,
        option: DynamicBatchOption,
        *,
        is_backfill: bool,
    ) -> None:
        task = self.tasks[str(task_id)]
        admission = None
        find_slack = getattr(self.owner, "_find_backfill_slack", None)
        if is_backfill and callable(find_slack):
            admission = find_slack(
                task.task,
                self.envelope,
                consume=True,
                gpu_id=str(gpu_id),
                profile=option.profile,
            )
            if admission is None:
                raise PlanningExhausted("phase slack changed before pair commit")
        task.selected_gpu = str(gpu_id)
        task.selected_option = option
        task.selected_admission = admission

    def release_selected(self, task_id: str) -> None:
        task = self.tasks.get(str(task_id))
        if task is None:
            return
        release = getattr(self.owner, "_release_backfill_slack", None)
        if callable(release):
            release(task.selected_admission, self.envelope)
        task.selected_gpu = ""
        task.selected_option = None
        task.selected_admission = None

    def selected_task_ids(self) -> tuple[str, ...]:
        return tuple(
            task_id
            for task_id, task in self.tasks.items()
            if task.selected_option is not None
        )

    def release_all_selected(self) -> None:
        for task_id in tuple(self.tasks):
            self.release_selected(task_id)

    def selected_admission(self, task_id: str) -> Any:
        task = self.tasks.get(str(task_id))
        return task.selected_admission if task is not None else None

    def metadata(self, task_id: str) -> dict[str, Any]:
        task = self.tasks.get(str(task_id))
        if task is None:
            return {}
        if task.selected_option is not None:
            return task.selected_option.metadata()
        return dict(task.fallback_metadata)


@dataclass(frozen=True)
class PhaseKey:
    campaign_id: str
    phase_index: int
    phase_name: str
    component: str
    config_fingerprint: str = ""
    barrier_kind: str = "hard"


@dataclass
class PhaseEstimate:
    source: str = "fallback_upper"
    latency_mean_sec: float = 0.0
    latency_upper_sec: float = 0.0
    confidence: float = 0.0
    observation_count: int = 0
    observed_runtimes_sec: list[float] = field(default_factory=list)
    remaining_tail_sec: float = 0.0
    last_invalidation_reason: str = ""


@dataclass
class PhaseState:
    key: PhaseKey
    ready_task_ids: list[str] = field(default_factory=list)
    running_task_ids: list[str] = field(default_factory=list)
    completed_task_ids: list[str] = field(default_factory=list)
    estimate: PhaseEstimate = field(default_factory=PhaseEstimate)
    next_phase_keys: list[PhaseKey] = field(default_factory=list)
    predicted_completion_time: float = 0.0


@dataclass
class PhaseWorkingSet:
    key: PhaseKey
    desired_worker_count: int = 1
    weight_vram_mb: int = 0
    weight_ram_mb: int = 0
    activation_vram_mb: int = 0
    activation_ram_mb: int = 0
    init_mean_sec: float = 0.0
    init_upper_sec: float = 0.0
    expected_phase_runtime_sec: float = 0.0
    resident_workers: list[str] = field(default_factory=list)
    reusable_workers: list[str] = field(default_factory=list)

    @property
    def missing_worker_count(self) -> int:
        return max(0, self.desired_worker_count - len(self.resident_workers))

    @property
    def effective_init_upper_sec(self) -> float:
        return self.missing_worker_count * _safe_float(self.init_upper_sec)


@dataclass
class ResidencyLease:
    gpu_id: str
    component: str
    config_fingerprint: str = ""
    worker_name: str = ""
    worker_addr: str = ""
    owner_campaign_id: str = ""
    owner_phase_name: str = ""
    reuse_source_campaign_id: str = ""
    reuse_source_phase_name: str = ""
    lease_start_time: float = 0.0
    lease_end_time: float = 0.0
    protected_until: float = 0.0
    evictable_after: float = 0.0
    reason: str = "primary_current_phase"
    lease_id: str = ""

    def __post_init__(self) -> None:
        if self.lease_id:
            return
        worker = self.worker_name or self.worker_addr or "resident"
        self.lease_id = (
            f"{self.owner_campaign_id}:{self.owner_phase_name}:"
            f"{self.component}:{self.gpu_id}:{worker}:{self.reason}"
        )


@dataclass
class SlackWindow:
    gpu_id: str
    start_time: float
    end_time: float
    free_vram_mb: int = 0
    free_ram_mb: int = 0
    resident_components: list[str] = field(default_factory=list)
    latest_safe_finish_time: float = 0.0
    window_id: str = ""

    def __post_init__(self) -> None:
        if not self.latest_safe_finish_time:
            self.latest_safe_finish_time = self.end_time
        if not self.window_id:
            self.window_id = f"{self.gpu_id}:{self.start_time:.3f}:{self.end_time:.3f}"


@dataclass
class CampaignFrontier:
    campaign_id: str
    phase_keys: list[PhaseKey] = field(default_factory=list)
    ready_task_ids: list[str] = field(default_factory=list)
    running_task_ids: list[str] = field(default_factory=list)
    downstream_ready_task_ids: list[str] = field(default_factory=list)
    preinit_candidates: list[PhaseWorkingSet] = field(default_factory=list)
    reusable_workers: list[str] = field(default_factory=list)
    protected_sync_point: float = 0.0
    latest_safe_backfill_deadline: float = 0.0
    eta_reduction_sec: float = 0.0
    confidence: float = 0.0


@dataclass
class ResidencyAction:
    action_id: str
    working_set: PhaseWorkingSet
    target_gpu_id: str
    campaign_id: str
    reason: str
    latest_safe_finish_time: float
    source_component: str = ""
    slack_window_id: str = ""
    scheduled: bool = False


@dataclass
class SlackAdmission:
    window_id: str
    gpu_id: str
    start_time: float
    finish_time: float
    duration_sec: float
    cancel_tier: bool = False
    reason: str = "finish_before_sync"


@dataclass
class PrimaryTailSloEpoch:
    """Primary-only reference and cap for one non-ratcheting frontier epoch."""

    campaign_id: str
    phase_index: int
    anchor_wall: float
    baseline_tail_wall: float
    reference_tail_wall: float
    factor: float
    phase_reference_tails: dict[int, float] = field(default_factory=dict)
    qualified_revision: int = 0
    qualified_update_at: float = 0.0

    @property
    def cap_tail_wall(self) -> float:
        return self.anchor_wall + self.factor * max(
            0.0, self.reference_tail_wall - self.anchor_wall
        )


@dataclass
class PhaseEnvelope:
    frontier: CampaignFrontier
    protected_gpu_intervals: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict,
    )
    protected_model_leases: list[ResidencyLease] = field(default_factory=list)
    allowed_slack_windows: list[SlackWindow] = field(default_factory=list)
    residency_actions: list[ResidencyAction] = field(default_factory=list)
    next_sync_point: float = 0.0
    protected_idle_cap_sec: float = 0.0


class PhaseCampaignPlanner(PredictiveTaskSupport):
    """Drop-in phase-aware facade around ``GlobalPlanner``."""

    @staticmethod
    def _get_dynamic_batching(component: str) -> dict[str, Any]:
        return get_dynamic_batching(component)

    def __init__(
        self,
        *args: Any,
        delegate: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.phase_shadow_mode = bool(kwargs.pop("phase_shadow_mode", False))
        self.cleanup_margin_sec = _safe_float(kwargs.pop("cleanup_margin_sec", 0.0))
        self.reciprocal_interference_correction = bool(
            kwargs.pop("reciprocal_interference_correction", False)
        )
        configured_slo_factor = kwargs.pop("backfill_primary_tail_slo_factor", None)
        self.backfill_primary_tail_slo_factor = (
            None
            if configured_slo_factor is None
            else _safe_float(configured_slo_factor, math.nan)
        )
        if self.backfill_primary_tail_slo_factor is not None and (
            not math.isfinite(self.backfill_primary_tail_slo_factor)
            or self.backfill_primary_tail_slo_factor < 1.0
        ):
            raise ValueError("backfill_primary_tail_slo_factor must be >= 1.0")
        if (
            self.backfill_primary_tail_slo_factor is not None
            and not self.reciprocal_interference_correction
        ):
            raise ValueError(
                "backfill_primary_tail_slo_factor requires reciprocal_interference_correction"
            )
        configured_basis = kwargs.pop("backfill_latency_basis", None)
        self.dynamic_batch_cold_policy = str(
            kwargs.pop("dynamic_batch_cold_policy", "largest_safe")
        )
        self.dynamic_batch_warm_policy = str(
            kwargs.pop("dynamic_batch_warm_policy", "throughput_optimal")
        )
        observer = kwargs.pop("primary_tail_slo_observer", None)
        trace_path = str(kwargs.pop("primary_tail_slo_observability_path", "") or "")
        if self.dynamic_batch_cold_policy not in {
            "constant_memory_linear_latency",
            "fixed_n",
            "largest_safe",
        }:
            raise ValueError(
                f"unsupported dynamic batch cold policy: {self.dynamic_batch_cold_policy}"
            )
        if self.dynamic_batch_warm_policy not in {
            "fixed_n",
            "largest_safe",
            "throughput_optimal",
        }:
            raise ValueError(
                f"unsupported dynamic batch warm policy: {self.dynamic_batch_warm_policy}"
            )
        self._delegate = delegate or GlobalPlanner(*args, **kwargs)
        delegate_basis = getattr(
            self._delegate,
            "interference_estimator_basis",
            None,
        )
        basis = str(configured_basis or delegate_basis or "mean").strip().lower()
        if basis not in {"mean", "ucb"}:
            raise ValueError(f"unsupported backfill_latency_basis: {basis}")
        if (
            configured_basis is not None
            and delegate_basis in {"mean", "ucb"}
            and basis != delegate_basis
        ):
            raise ValueError(
                "backfill_latency_basis must match global_planner.placement_strategy"
            )
        self.backfill_latency_basis = basis
        self._delegate.interference_estimator_basis = basis
        self._delegate.reciprocal_interference_correction = bool(
            self.reciprocal_interference_correction and not self.phase_shadow_mode
        )
        self._phase_estimates: dict[PhaseKey, PhaseEstimate] = {}
        self._phase_states: dict[PhaseKey, PhaseState] = {}
        self._residency_leases: list[ResidencyLease] = []
        self._residency_actions: list[ResidencyAction] = []
        self._committed_action_ids: set[str] = set()
        self._rr_cursor = 0
        self._last_phase_envelope: PhaseEnvelope | None = None
        self._invalidated_components: dict[str, str] = {}
        self._dynamic_batch_context: DynamicBatchSolveContext | None = None
        self._dynamic_batch_profiles: dict[tuple[str, str], DynamicBatchCandidate] = {}
        self._dynamic_batch_blocked: set[tuple[str, str]] = set()
        self._dynamic_batch_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        self._dynamic_batch_fallbacks: dict[str, dict[str, Any]] = {}
        self._primary_tail_slo_epochs: dict[str, PrimaryTailSloEpoch] = {}
        self._qualified_latency_updates: set[tuple[str, str]] = set()
        self._primary_tail_slo_rejections: set[str] = set()
        self._alternate_gpu_retry_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
        }
        self._alternate_gpu_retry_rescued_task_ids: set[str] = set()
        self._primary_tail_slo_observer: Any = observer if callable(observer) else None
        if trace_path:

            def _write_slo_record(record: dict[str, Any]) -> None:
                try:
                    with open(trace_path, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                except OSError:
                    _LOG.warning(
                        "Unable to write primary-tail SLO trace", exc_info=True
                    )

            self._primary_tail_slo_observer = _write_slo_record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def on_profile_drift_event(self, event: Any) -> None:
        """Accept only solo-equivalent primary latency revisions for SLO rebasing."""
        if not bool(getattr(event, "primary_tail_reference_qualified", False)):
            return
        if str(getattr(event, "metric", "") or "") != "latency":
            return
        campaign_id = str(getattr(event, "campaign_id", "") or "")
        component = str(getattr(event, "component", "") or "").strip().lower()
        if campaign_id and component:
            self._qualified_latency_updates.add((campaign_id, component))

    def export_runtime_state(self) -> dict[str, Any]:
        """Persist immutable SLO anchors and qualified primary-only revisions."""
        return {
            "schema": "phase_campaign_planner_runtime_state_v1",
            "primary_tail_slo_epochs": {
                key: {
                    "campaign_id": epoch.campaign_id,
                    "phase_index": epoch.phase_index,
                    "anchor_wall": epoch.anchor_wall,
                    "baseline_tail_wall": epoch.baseline_tail_wall,
                    "reference_tail_wall": epoch.reference_tail_wall,
                    "factor": epoch.factor,
                    "phase_reference_tails": epoch.phase_reference_tails,
                    "qualified_revision": epoch.qualified_revision,
                    "qualified_update_at": epoch.qualified_update_at,
                }
                for key, epoch in self._primary_tail_slo_epochs.items()
            },
        }

    def import_runtime_state(self, data: Any) -> int:
        """Restore SLO anchors without allowing a bootstrap rebase."""
        if not isinstance(data, dict):
            return 0
        restored = 0
        for key, raw in dict(data.get("primary_tail_slo_epochs") or {}).items():
            if not isinstance(raw, dict):
                continue
            factor = _safe_float(raw.get("factor"), math.nan)
            anchor = _safe_float(raw.get("anchor_wall"), math.nan)
            baseline = _safe_float(raw.get("baseline_tail_wall"), math.nan)
            reference = _safe_float(raw.get("reference_tail_wall"), math.nan)
            if (
                not all(
                    math.isfinite(value)
                    for value in (factor, anchor, baseline, reference)
                )
                or factor < 1.0
                or reference < anchor
            ):
                continue
            self._primary_tail_slo_epochs[str(key)] = PrimaryTailSloEpoch(
                campaign_id=str(raw.get("campaign_id", "") or ""),
                phase_index=_safe_int(raw.get("phase_index")),
                anchor_wall=anchor,
                baseline_tail_wall=baseline,
                reference_tail_wall=reference,
                factor=factor,
                phase_reference_tails={
                    _safe_int(phase): _safe_float(tail, math.nan)
                    for phase, tail in dict(
                        raw.get("phase_reference_tails") or {}
                    ).items()
                    if math.isfinite(_safe_float(tail, math.nan))
                },
                qualified_revision=_safe_int(raw.get("qualified_revision")),
                qualified_update_at=_safe_float(raw.get("qualified_update_at")),
            )
            restored += 1
        return restored

    def _primary_tail_slo_phase_keys(self, envelope: PhaseEnvelope) -> list[PhaseKey]:
        active = [
            key
            for key, state in self._phase_states.items()
            if key.campaign_id == envelope.frontier.campaign_id
            and (state.ready_task_ids or state.running_task_ids)
        ]
        return active or list(envelope.frontier.phase_keys)

    def _primary_tail_slo_key(self, envelope: PhaseEnvelope) -> str:
        return envelope.frontier.campaign_id

    def _primary_campaign_tail_reference(
        self, envelope: PhaseEnvelope, now: float
    ) -> tuple[int, dict[int, float], float]:
        """Forecast the primary-only DAG tail from its active phase onward."""
        active_keys = self._primary_tail_slo_phase_keys(envelope)
        active_index = min((key.phase_index for key in active_keys), default=0)
        tails = {active_index: max(now, envelope.frontier.protected_sync_point)}
        campaign_id = envelope.frontier.campaign_id
        states_by_phase: dict[int, list[PhaseState]] = {}
        for key, state in self._phase_states.items():
            if key.campaign_id == campaign_id:
                states_by_phase.setdefault(key.phase_index, []).append(state)
        for index, component in enumerate(self._component_order(campaign_id, [])):
            if index <= active_index:
                continue
            states = states_by_phase.get(index, [])
            if states:
                duration = max(
                    (state.estimate.remaining_tail_sec for state in states), default=0.0
                )
            else:
                duration = self._estimate_for_key(
                    PhaseKey(
                        campaign_id=campaign_id,
                        phase_index=index,
                        phase_name=component,
                        component=component,
                        barrier_kind="soft",
                    ),
                    [],
                ).latency_upper_sec
            tails[index] = tails[max(tails)] + max(0.0, duration)
        return active_index, tails, max(tails.values())

    def _primary_tail_slo_epoch(
        self, envelope: PhaseEnvelope | None
    ) -> PrimaryTailSloEpoch | None:
        if envelope is None or self.backfill_primary_tail_slo_factor is None:
            return None
        frontier = envelope.frontier
        if not (frontier.ready_task_ids or frontier.running_task_ids):
            return None
        now = time.time()
        active_index, phase_tails, campaign_tail = (
            self._primary_campaign_tail_reference(envelope, now)
        )
        key = self._primary_tail_slo_key(envelope)
        epoch = self._primary_tail_slo_epochs.get(key)
        if epoch is None:
            epoch = PrimaryTailSloEpoch(
                campaign_id=frontier.campaign_id,
                phase_index=active_index,
                anchor_wall=now,
                baseline_tail_wall=phase_tails[active_index],
                reference_tail_wall=campaign_tail,
                factor=self.backfill_primary_tail_slo_factor,
                phase_reference_tails=phase_tails,
            )
            self._primary_tail_slo_epochs[key] = epoch
            return epoch

        active_components = {
            key.component
            for key in self._primary_tail_slo_phase_keys(envelope)
            if key.phase_index == active_index
        }
        if any(
            (frontier.campaign_id, component) in self._qualified_latency_updates
            for component in active_components
        ):
            if campaign_tail > epoch.reference_tail_wall:
                epoch.reference_tail_wall = campaign_tail
                epoch.phase_reference_tails = phase_tails
                epoch.qualified_revision += 1
                epoch.qualified_update_at = now
            self._qualified_latency_updates.difference_update(
                (frontier.campaign_id, component) for component in active_components
            )
        return epoch

    def _primary_tail_slo_filter(
        self,
        task_id: str,
        epoch: PrimaryTailSloEpoch | None,
        active_phase_index: int,
    ) -> Any:
        if epoch is None:
            return None

        def _allows(projection: Any) -> bool:
            projected_tail = primary_incumbent_slowdown_ratio(
                self.timelines, epoch.campaign_id, projection
            )
            headroom = epoch.factor - projected_tail
            allowed = headroom >= 0.0
            if not allowed:
                self._primary_tail_slo_rejections.add(task_id)
            if self._primary_tail_slo_observer is not None:
                self._primary_tail_slo_observer(
                    {
                        "task_id": task_id,
                        "candidate_gpu": str(getattr(projection, "gpu_id", "")),
                        "epoch": epoch.campaign_id,
                        "active_phase": active_phase_index,
                        "reference": epoch.reference_tail_wall,
                        "cap": epoch.cap_tail_wall,
                        "projected": projected_tail,
                        "headroom": headroom,
                        "allowed": allowed,
                    }
                )
            return allowed

        return _allows

    @property
    def campaign_scheduler(self) -> Any:
        return self._delegate.campaign_scheduler

    @property
    def timelines(self) -> Any:
        return self._delegate.timelines

    @property
    def phase_estimates(self) -> dict[PhaseKey, PhaseEstimate]:
        return dict(self._phase_estimates)

    @property
    def phase_states(self) -> dict[PhaseKey, PhaseState]:
        return dict(self._phase_states)

    @property
    def residency_leases(self) -> list[ResidencyLease]:
        return list(self._residency_leases)

    @property
    def last_phase_envelope(self) -> PhaseEnvelope | None:
        return self._last_phase_envelope

    @property
    def residency_actions(self) -> list[ResidencyAction]:
        return list(self._residency_actions)

    def has_active_residency_lease(
        self,
        component: str,
        gpu_id: str,
        worker_name: str = "",
        *,
        now: float | None = None,
    ) -> bool:
        """Return whether the current phase envelope protects this worker."""
        envelope = self._last_phase_envelope
        if envelope is None:
            return False
        current = time.time() if now is None else _safe_float(now)
        wanted_component = str(component or "").strip().lower()
        wanted_gpu = str(gpu_id or "").strip()
        wanted_worker = str(worker_name or "").strip()
        return any(
            _safe_float(lease.lease_start_time)
            <= current
            < _safe_float(lease.evictable_after)
            and str(lease.component or "").strip().lower() == wanted_component
            and str(lease.gpu_id or "").strip() == wanted_gpu
            and (
                not str(lease.worker_name or "").strip()
                or str(lease.worker_name).strip() == wanted_worker
            )
            for lease in envelope.protected_model_leases
        )

    @property
    def _constraint_tracker(self) -> Any:
        return self._delegate._constraint_tracker

    def _prepare_dynamic_batch_context(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> DynamicBatchSolveContext:
        context = DynamicBatchSolveContext(owner=self)
        now = time.time()
        for task in pending_tasks:
            task_id = str(task.get("task_id", "") or "")
            component = str(task.get("component", "") or "").strip().lower()
            config = get_dynamic_batching(component)
            if not task_id or not config or not config.get("enabled", False):
                continue
            logical_n = _safe_int(task.get("logical_batch_size"))
            input_size = _safe_float(task.get("input_size"))
            batch_size_arg = str(config["batch_size_arg"])
            task_context = DynamicBatchTaskCandidates(task=task)
            context.tasks[task_id] = task_context
            if logical_n <= 1 or input_size <= 0:
                task_context.fallback_mode = True
                task_context.fallback_metadata = {
                    "enabled": True,
                    "batch_size_arg": batch_size_arg,
                    "logical_batch_size": logical_n,
                    "selected_batch_size": 0,
                    "phase": "exact",
                    "policy": "fixed_n",
                    "fallback_reason": "exact_n_missing_dynamic_axes",
                }
                continue
            overrides = task.get("execution_overrides")
            pinned_k = _safe_int(
                overrides.get("batch_size") if isinstance(overrides, dict) else 0
            )
            if pinned_k and not 1 <= pinned_k <= logical_n:
                raise ValueError(
                    f"dynamic batch override must be in [1, {logical_n}], got {pinned_k}"
                )
            config_fingerprint = str(task.get("config_fingerprint", "") or "")
            scalar_evidence = False
            for gpu_id in self._gpu_ids():
                available_vram = self.timelines.available_vram_at(
                    gpu_id,
                    now,
                    exclude_task_id=task_id,
                )
                available_ram = self.timelines.available_host_ram_at(
                    now,
                    exclude_task_id=task_id,
                )
                cold = self._cold_batch_candidates(
                    component=component,
                    input_size=input_size,
                    config_fingerprint=config_fingerprint,
                    gpu_id=gpu_id,
                    logical_n=logical_n,
                    available_vram=available_vram,
                    available_ram=available_ram,
                )
                if not cold:
                    continue
                scalar_evidence = True
                warm = self._warm_batch_candidates(
                    component=component,
                    input_size=input_size,
                    config_fingerprint=config_fingerprint,
                    gpu_id=gpu_id,
                    logical_n=logical_n,
                    available_vram=available_vram,
                    available_ram=available_ram,
                )
                phase = "warm" if warm else "cold"
                candidates = warm or cold
                policy = (
                    "pinned"
                    if pinned_k
                    else (
                        self.dynamic_batch_warm_policy
                        if warm
                        else self.dynamic_batch_cold_policy
                    )
                )
                if pinned_k:
                    eligible = tuple(
                        item
                        for item in candidates
                        if item.k == pinned_k and item.supported
                    )
                    if not eligible and warm:
                        phase = "cold"
                        eligible = tuple(
                            item
                            for item in cold
                            if item.k == pinned_k and item.supported
                        )
                elif policy == "fixed_n":
                    eligible = tuple(
                        item
                        for item in candidates
                        if item.k == logical_n and item.supported
                    )
                elif policy == "throughput_optimal":
                    eligible = tuple(
                        item
                        for item in candidates
                        if item.supported and item.latency_mean > 0
                    )
                else:
                    eligible = tuple(item for item in candidates if item.supported)
                task_context.options_by_gpu[str(gpu_id)] = tuple(
                    DynamicBatchOption(
                        profile=item,
                        phase=phase,
                        policy=policy,
                        batch_size_arg=batch_size_arg,
                    )
                    for item in eligible
                )
            if not scalar_evidence:
                task_context.fallback_mode = True
                task_context.fallback_metadata = {
                    "enabled": True,
                    "batch_size_arg": batch_size_arg,
                    "logical_batch_size": logical_n,
                    "selected_batch_size": 0,
                    "phase": "exact",
                    "policy": "fixed_n",
                    "fallback_reason": "missing_scalar_evidence",
                }
        self._dynamic_batch_context = context
        setter = getattr(self._delegate, "set_dynamic_batch_context", None)
        if callable(setter):
            if isinstance(self._delegate, GlobalPlanner):
                setter(context, borrowed=True)
            else:
                setter(context)
        return context

    def _rollback_dynamic_batch_context(
        self,
        context: DynamicBatchSolveContext,
    ) -> None:
        for task_id in context.selected_task_ids():
            self.timelines.remove_predicted_entries_for_task(task_id)
        context.release_all_selected()

    def _clear_dynamic_batch_context(self) -> None:
        context = self._dynamic_batch_context
        if context is not None:
            context.end_evaluation()
        setter = getattr(self._delegate, "set_dynamic_batch_context", None)
        if callable(setter):
            setter()
        self._dynamic_batch_context = None

    async def solve_admissible(
        self,
        pending_tasks: list[dict[str, Any]],
        *,
        commit_predictions: bool = True,
    ) -> tuple[list[DispatchPlan], dict[str, PlanningExhausted]]:
        if self.phase_shadow_mode:
            self._refresh_phase_model(pending_tasks)
            delegate_result = self._delegate.solve_admissible(
                pending_tasks,
                commit_predictions=commit_predictions,
            )
            if isinstance(delegate_result, Awaitable):
                delegate_result = await cast(Awaitable, delegate_result)
            return delegate_result
        try:
            self._prepare_legacy_dynamic_batch_profiles(pending_tasks)
            envelope = self._refresh_phase_model(pending_tasks)
            return await self._solve_with_phase_order(
                pending_tasks,
                envelope=envelope,
                commit_predictions=commit_predictions,
            )
        finally:
            getattr(self._delegate, "_reciprocal_query_results", {}).clear()
            self._clear_legacy_dynamic_batch_profiles()

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        if self.phase_shadow_mode:
            plan = self._delegate.plan(*args, **kwargs)
            if isinstance(plan, DispatchPlan):
                self._annotate_plan(plan)
            return plan
        names = (
            "task_id",
            "campaign_id",
            "component",
            "input_size",
            "is_backfill",
            "config_fingerprint",
            "input_fingerprint",
            "logical_batch_size",
            "execution_overrides",
        )
        values: dict[str, Any] = dict(zip(names, args, strict=False))
        values.update(kwargs)
        task = {name: values.get(name) for name in names}
        try:
            self._prepare_legacy_dynamic_batch_profiles([task])
            self._refresh_phase_model([task])
            plan = self._delegate.plan(*args, **kwargs)
            if isinstance(plan, DispatchPlan):
                self._annotate_plan(plan, task=task)
            return plan
        finally:
            getattr(self._delegate, "_reciprocal_query_results", {}).clear()
            self._clear_legacy_dynamic_batch_profiles()

    def commit_dispatch_plan_prediction(self, *args: Any, **kwargs: Any) -> Any:
        result = self._delegate.commit_dispatch_plan_prediction(*args, **kwargs)
        self._schedule_residency_actions(self._last_phase_envelope)
        return result

    def incorporate_constraint(self, violation: Any) -> Any:
        failed = getattr(violation, "failed_plan", None)
        component = str(getattr(failed, "component", "") or "")
        if component:
            self._invalidate_component(
                component,
                str(getattr(violation, "violation_type", "constraint_violation")),
            )
        return self._delegate.incorporate_constraint(violation)

    def on_profile_drift(self, component: str, metric: str) -> Any:
        self._invalidate_component(component, f"profile_drift:{metric}")
        return self._delegate.on_profile_drift(component, metric)

    def attach_supervisor_wake(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.attach_supervisor_wake(*args, **kwargs)


    def _refresh_phase_model(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> PhaseEnvelope | None:
        self._residency_leases = []
        self._residency_actions = []
        self._last_phase_envelope = None
        states = self._build_phase_states(pending_tasks)
        self._phase_states = states
        envelope = self._build_primary_envelope(states, pending_tasks)
        self._last_phase_envelope = envelope
        return envelope

    def _build_phase_states(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> dict[PhaseKey, PhaseState]:
        states: dict[PhaseKey, PhaseState] = {}
        for task in pending_tasks:
            key = self._phase_key_for_task(task, pending_tasks)
            state = states.setdefault(key, PhaseState(key=key))
            task_id = str(task.get("task_id", "") or "")
            if task_id:
                state.ready_task_ids.append(task_id)

        for entry in self._timeline_entries():
            if bool(getattr(entry, "is_completed", False)):
                continue
            campaign_id = str(getattr(entry, "campaign_id", "") or "")
            component = str(getattr(entry, "component", "") or "").strip().lower()
            task_id = str(getattr(entry, "task_id", "") or "")
            if not campaign_id or not component or not task_id:
                continue
            key = self._phase_key_for_values(
                campaign_id=campaign_id,
                component=component,
                config_fingerprint=str(getattr(entry, "config_fingerprint", "") or ""),
                pending_tasks=pending_tasks,
            )
            state = states.setdefault(key, PhaseState(key=key))
            state.running_task_ids.append(task_id)
            state.predicted_completion_time = max(
                state.predicted_completion_time,
                _safe_float(getattr(entry, "predicted_end_time", 0.0)),
            )

        self._add_completed_records(states, pending_tasks)
        self._update_estimates(states, pending_tasks)
        self._link_next_phases(states, pending_tasks)
        return states

    def _add_completed_records(
        self,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
    ) -> None:
        tasks = self._gateway_tasks()
        for record in tasks.values():
            if not bool(getattr(record, "ok", False)):
                continue
            campaign_id = str(getattr(record, "campaign_id", "") or "")
            component = str(getattr(record, "component", "") or "").strip().lower()
            task_id = str(getattr(record, "task_id", "") or "")
            if not campaign_id or not component or not task_id:
                continue
            key = self._phase_key_for_values(
                campaign_id=campaign_id,
                component=component,
                config_fingerprint=str(getattr(record, "config_fingerprint", "") or ""),
                process_name=self._record_process_name(record),
                pending_tasks=pending_tasks,
            )
            states.setdefault(key, PhaseState(key=key)).completed_task_ids.append(
                task_id
            )

    def _update_estimates(
        self,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
    ) -> None:
        now = time.time()
        for key, state in states.items():
            estimate = self._estimate_for_key(key, pending_tasks)
            active_count = max(
                1, len(state.ready_task_ids) + len(state.running_task_ids)
            )
            estimate.remaining_tail_sec = estimate.latency_upper_sec * active_count
            state.estimate = estimate
            state.predicted_completion_time = max(
                state.predicted_completion_time,
                now + estimate.remaining_tail_sec,
            )
            self._phase_estimates[key] = estimate

    def _estimate_for_key(
        self,
        key: PhaseKey,
        pending_tasks: list[dict[str, Any]],
    ) -> PhaseEstimate:
        gpu_id = self._first_gpu_id()
        input_size = self._phase_input_size(key, pending_tasks)
        uncertainty_z = self._latency_uncertainty_z(
            key.component,
            input_size,
            key.config_fingerprint,
        )
        observed = self._observed_runtimes(key)
        if observed:
            mean = sum(observed) / len(observed)
            sigma = math.sqrt(
                sum((runtime - mean) ** 2 for runtime in observed) / len(observed)
            )
            return PhaseEstimate(
                source="wave_observed" if len(observed) > 1 else "first_task_observed",
                latency_mean_sec=mean,
                latency_upper_sec=self._latency_with_basis(mean, sigma, uncertainty_z),
                confidence=0.85 if len(observed) > 1 else 0.55,
                observation_count=len(observed),
                observed_runtimes_sec=observed,
            )

        mean = self._predict_latency(
            key.component, input_size, gpu_id, key.config_fingerprint
        )
        sigma = self._predict_latency_sigma(
            key.component,
            input_size,
            gpu_id,
            key.config_fingerprint,
        )
        confidence_name = self._confidence_name(
            key.component,
            input_size,
            gpu_id,
            key.config_fingerprint,
        )
        confidence = {"high": 0.7, "medium": 0.45, "low": 0.2}.get(
            confidence_name, 0.25
        )
        source = (
            "historical_gp"
            if confidence_name in {"high", "medium"}
            else "fallback_upper"
        )
        if (
            self._campaign_rank(key.campaign_id, pending_tasks) > 0
            and source != "fallback_upper"
        ):
            source = "cross_campaign_gp"
        upper = self._latency_with_basis(mean, sigma, uncertainty_z)
        reason = self._invalidated_components.get(key.component, "")
        if reason:
            confidence = min(confidence, 0.25)
            source = "fallback_upper"
        return PhaseEstimate(
            source=source,
            latency_mean_sec=mean,
            latency_upper_sec=upper,
            confidence=confidence,
            last_invalidation_reason=reason,
        )

    def _link_next_phases(
        self,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
    ) -> None:
        for key, state in states.items():
            order = self._component_order(key.campaign_id, pending_tasks)
            if key.component not in order:
                continue
            idx = order.index(key.component)
            if idx + 1 >= len(order):
                continue
            next_key = self._phase_key_for_values(
                campaign_id=key.campaign_id,
                component=order[idx + 1],
                pending_tasks=pending_tasks,
            )
            state.next_phase_keys = [next_key]

    def _build_primary_envelope(
        self,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
    ) -> PhaseEnvelope | None:
        primary_id = self._primary_campaign_id(pending_tasks)
        if not primary_id:
            return None

        primary_states = [s for s in states.values() if s.key.campaign_id == primary_id]
        if not primary_states:
            now = time.time()
            return PhaseEnvelope(
                frontier=CampaignFrontier(campaign_id=primary_id),
                next_sync_point=now,
            )

        active_states = [
            s for s in primary_states if s.ready_task_ids or s.running_task_ids
        ]
        if active_states:
            active_index = min(s.key.phase_index for s in active_states)
        else:
            completed_states = [s for s in primary_states if s.completed_task_ids]
            active_index = (
                max(s.key.phase_index for s in completed_states)
                if completed_states
                else min(s.key.phase_index for s in primary_states)
            )
        frontier_states = [
            s for s in primary_states if s.key.phase_index == active_index
        ]
        downstream_ready = [
            s
            for s in primary_states
            if s.key.phase_index > active_index and s.ready_task_ids
        ]
        if active_states:
            sync_point = (
                max(s.predicted_completion_time for s in frontier_states) or time.time()
            )
        else:
            sync_point = time.time() + max(
                1.0,
                *(s.estimate.latency_upper_sec for s in frontier_states),
            )
        frontier = CampaignFrontier(
            campaign_id=primary_id,
            phase_keys=[
                s.key for s in sorted(primary_states, key=lambda s: s.key.phase_index)
            ],
            ready_task_ids=[
                tid
                for s in frontier_states + downstream_ready
                for tid in s.ready_task_ids
            ],
            running_task_ids=[
                tid for s in frontier_states for tid in s.running_task_ids
            ],
            downstream_ready_task_ids=[
                tid for s in downstream_ready for tid in s.ready_task_ids
            ],
            protected_sync_point=sync_point,
            latest_safe_backfill_deadline=sync_point,
            confidence=min(
                (s.estimate.confidence for s in frontier_states), default=0.0
            ),
        )
        frontier.preinit_candidates = self._preinit_candidates(frontier_states, states)
        working_sets = [self._build_working_set(s) for s in frontier_states]
        frontier.reusable_workers = [
            w for ws in working_sets for w in ws.reusable_workers
        ]
        frontier.eta_reduction_sec = sum(
            ws.effective_init_upper_sec for ws in frontier.preinit_candidates
        )
        primary_gpus: set[str] = set()
        for tid in frontier.running_task_ids:
            entry = self.timelines.find_entry(tid)
            if entry is not None:
                gpu = getattr(entry, "gpu_id", None)
                if gpu:
                    primary_gpus.add(str(gpu))
        slack_windows = self._slack_windows(sync_point, primary_gpus=primary_gpus)
        actions = self._residency_actions_for(
            primary_id,
            frontier,
            states,
            pending_tasks,
            slack_windows,
        )
        leases = self._leases_for(
            primary_id, working_sets + frontier.preinit_candidates, sync_point
        ) + self._leases_for_actions(actions)
        self._residency_leases = leases
        self._residency_actions = actions
        return PhaseEnvelope(
            frontier=frontier,
            protected_gpu_intervals=self._protected_intervals(primary_id),
            protected_model_leases=leases,
            allowed_slack_windows=slack_windows,
            residency_actions=actions,
            next_sync_point=sync_point,
        )

    def _preinit_candidates(
        self,
        frontier_states: list[PhaseState],
        states: dict[PhaseKey, PhaseState],
    ) -> list[PhaseWorkingSet]:
        ready_keys = {s.key for s in states.values() if s.ready_task_ids}
        seen: set[PhaseKey] = set()
        candidates: list[PhaseWorkingSet] = []
        for state in frontier_states:
            for key in state.next_phase_keys:
                if key in ready_keys or key in seen:
                    continue
                seen.add(key)
                ws = self._build_working_set(
                    PhaseState(
                        key=key,
                        estimate=self._phase_estimates.get(key, PhaseEstimate()),
                    )
                )
                if ws.missing_worker_count:
                    candidates.append(ws)
        return candidates

    def _residency_actions_for(
        self,
        primary_id: str,
        frontier: CampaignFrontier,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
        slack_windows: list[SlackWindow],
    ) -> list[ResidencyAction]:
        actions: list[ResidencyAction] = []
        for ws in frontier.preinit_candidates:
            action = self._action_for_working_set(
                ws,
                campaign_id=primary_id,
                reason="primary_next_phase_preinit",
                source_component=frontier.phase_keys[0].component
                if frontier.phase_keys
                else "",
                windows=slack_windows,
            )
            if action:
                actions.append(action)

        return actions

    def _successor_campaign_id(
        self,
        primary_id: str,
        pending_tasks: list[dict[str, Any]],
    ) -> str:
        for campaign_id in self._ordered_campaign_ids(pending_tasks):
            if campaign_id != primary_id:
                return campaign_id
        return ""

    def _successor_residency_action(
        self,
        campaign_id: str,
        states: dict[PhaseKey, PhaseState],
        pending_tasks: list[dict[str, Any]],
        slack_windows: list[SlackWindow],
    ) -> ResidencyAction | None:
        campaign_states = [
            s for s in states.values() if s.key.campaign_id == campaign_id
        ]
        if campaign_states:
            phase_index = min(s.key.phase_index for s in campaign_states)
            state = next(s for s in campaign_states if s.key.phase_index == phase_index)
        else:
            order = self._component_order(campaign_id, pending_tasks)
            if not order:
                return None
            state = PhaseState(
                key=self._phase_key_for_values(
                    campaign_id=campaign_id,
                    component=order[0],
                    pending_tasks=pending_tasks,
                )
            )
        ws = self._build_working_set(state)
        if ws.missing_worker_count <= 0:
            return None
        return self._action_for_working_set(
            ws,
            campaign_id=campaign_id,
            reason="successor_frontier_preinit",
            source_component=state.key.component,
            windows=slack_windows,
        )

    def _action_for_working_set(
        self,
        ws: PhaseWorkingSet,
        *,
        campaign_id: str,
        reason: str,
        source_component: str,
        windows: list[SlackWindow],
    ) -> ResidencyAction | None:
        window = self._window_for_residency(ws, windows)
        if not window:
            return None
        return ResidencyAction(
            action_id=f"{reason}:{campaign_id}:{ws.key.component}:{window.gpu_id}",
            working_set=ws,
            target_gpu_id=window.gpu_id,
            campaign_id=campaign_id,
            reason=reason,
            latest_safe_finish_time=window.latest_safe_finish_time,
            source_component=source_component,
            slack_window_id=window.window_id,
        )

    def _window_for_residency(
        self,
        ws: PhaseWorkingSet,
        windows: list[SlackWindow],
    ) -> SlackWindow | None:
        duration = max(0.0, _safe_float(ws.effective_init_upper_sec))
        for window in windows:
            if ws.weight_vram_mb > window.free_vram_mb:
                continue
            if ws.weight_ram_mb > window.free_ram_mb:
                continue
            if duration <= window.latest_safe_finish_time - window.start_time:
                window.free_vram_mb = max(0, window.free_vram_mb - ws.weight_vram_mb)
                for candidate_window in windows:
                    candidate_window.free_ram_mb = max(
                        0, candidate_window.free_ram_mb - ws.weight_ram_mb
                    )
                _LOG.info(
                    "[phase-plan] residency action claimed slack window=%s "
                    "component=%s campaign=%s duration=%.3f remaining=%.3f",
                    window.window_id,
                    ws.key.component,
                    ws.key.campaign_id,
                    duration,
                    max(0.0, window.latest_safe_finish_time - window.start_time),
                )
                return window
        return None

    def _claim_successor_residency_actions(
        self,
        primary_id: str,
        pending_tasks: list[dict[str, Any]],
        envelope: PhaseEnvelope | None,
    ) -> None:
        if envelope is None:
            return
        existing = {action.action_id for action in envelope.residency_actions}
        new_actions: list[ResidencyAction] = []
        for campaign_id in self._ordered_campaign_ids(pending_tasks):
            if campaign_id == primary_id:
                continue
            action = self._successor_residency_action(
                campaign_id,
                self._phase_states,
                pending_tasks,
                envelope.allowed_slack_windows,
            )
            if action is None or action.action_id in existing:
                continue
            existing.add(action.action_id)
            new_actions.append(action)
        if not new_actions:
            return
        envelope.residency_actions.extend(new_actions)
        envelope.protected_model_leases.extend(self._leases_for_actions(new_actions))
        self._residency_actions = envelope.residency_actions
        self._residency_leases = envelope.protected_model_leases

    def _schedule_residency_actions(self, envelope: PhaseEnvelope | None) -> None:
        if envelope is None:
            return
        fire = getattr(self.campaign_scheduler, "_fire_pre_init", None)
        for action in envelope.residency_actions:
            ws = action.working_set
            if ws.missing_worker_count <= 0:
                continue
            if action.action_id in self._committed_action_ids:
                continue
            if not callable(fire):
                continue
            try:
                fire(
                    ws.key.component,
                    action.target_gpu_id,
                    action.campaign_id,
                    action.source_component or ws.key.component,
                    weight_vram_mb=ws.weight_vram_mb,
                    weight_ram_mb=ws.weight_ram_mb,
                )
                action.scheduled = True
                self._committed_action_ids.add(action.action_id)
            except Exception:
                _LOG.warning(
                    "[phase-plan] residency pre-init action failed action=%s",
                    action.action_id,
                    exc_info=True,
                )

    def _build_working_set(self, state: PhaseState) -> PhaseWorkingSet:
        key = state.key
        gpu_id = self._first_gpu_id()
        input_size = self._avg_input_size(key.campaign_id, key.component)
        resident = self._resident_workers(key.component, key.config_fingerprint)
        desired = max(1, len(state.ready_task_ids) + len(state.running_task_ids))
        return PhaseWorkingSet(
            key=key,
            desired_worker_count=desired,
            weight_vram_mb=_safe_int(
                self._predict_vram(
                    key.component, input_size, gpu_id, key.config_fingerprint
                )
            ),
            weight_ram_mb=_safe_int(
                self._predict_ram(
                    key.component, input_size, gpu_id, key.config_fingerprint
                )
            ),
            activation_vram_mb=_safe_int(
                self._predict_vram(
                    key.component, input_size, gpu_id, key.config_fingerprint
                )
            ),
            activation_ram_mb=_safe_int(
                self._predict_ram(
                    key.component, input_size, gpu_id, key.config_fingerprint
                )
            ),
            init_mean_sec=self._get_init_latency(key.component, gpu_id),
            init_upper_sec=self._get_init_latency(key.component, gpu_id) * 1.5,
            expected_phase_runtime_sec=state.estimate.latency_upper_sec,
            resident_workers=[r.worker_name for r in resident if r.worker_addr],
            reusable_workers=[r.worker_name for r in resident if r.reusable],
        )

    def _leases_for(
        self,
        primary_id: str,
        working_sets: list[PhaseWorkingSet],
        sync_point: float,
    ) -> list[ResidencyLease]:
        now = time.time()
        leases: list[ResidencyLease] = []
        for ws in working_sets:
            for worker in self._resident_workers(
                ws.key.component, ws.key.config_fingerprint
            ):
                if not worker.reusable:
                    continue
                cross_campaign = bool(
                    worker.campaign_id and worker.campaign_id != primary_id
                )
                leases.append(
                    ResidencyLease(
                        gpu_id=worker.gpu_id,
                        component=ws.key.component,
                        config_fingerprint=ws.key.config_fingerprint,
                        worker_name=worker.worker_name,
                        worker_addr=worker.worker_addr,
                        owner_campaign_id=primary_id,
                        owner_phase_name=ws.key.phase_name,
                        reuse_source_campaign_id=worker.campaign_id,
                        lease_start_time=now,
                        lease_end_time=sync_point,
                        protected_until=sync_point,
                        evictable_after=sync_point,
                        reason="cross_campaign_reuse"
                        if cross_campaign
                        else "primary_current_phase",
                    )
                )
        return leases

    def _leases_for_actions(
        self, actions: list[ResidencyAction]
    ) -> list[ResidencyLease]:
        now = time.time()
        leases: list[ResidencyLease] = []
        for action in actions:
            ws = action.working_set
            leases.append(
                ResidencyLease(
                    gpu_id=action.target_gpu_id,
                    component=ws.key.component,
                    config_fingerprint=ws.key.config_fingerprint,
                    owner_campaign_id=action.campaign_id,
                    owner_phase_name=ws.key.phase_name,
                    lease_start_time=now,
                    lease_end_time=action.latest_safe_finish_time,
                    protected_until=action.latest_safe_finish_time,
                    evictable_after=action.latest_safe_finish_time,
                    reason=action.reason,
                    lease_id=action.action_id,
                )
            )
        return leases

    def _protected_intervals(
        self, primary_id: str
    ) -> dict[str, list[tuple[float, float]]]:
        intervals: dict[str, list[tuple[float, float]]] = {}
        for entry in self._timeline_entries():
            if str(getattr(entry, "campaign_id", "") or "") != primary_id:
                continue
            gpu_id = str(getattr(entry, "gpu_id", "") or "")
            if not gpu_id:
                continue
            intervals.setdefault(gpu_id, []).append(
                (
                    _safe_float(getattr(entry, "start_time", 0.0)),
                    _safe_float(getattr(entry, "predicted_end_time", 0.0)),
                )
            )
        return intervals

    def _slack_windows(
        self,
        sync_point: float,
        *,
        primary_gpus: set[str] | None = None,
    ) -> list[SlackWindow]:
        now = time.time()
        if sync_point <= now:
            return []
        windows: list[SlackWindow] = []
        temporal_mode = (
            getattr(self.timelines, "_vram_reservation_model_name", "full_wall")
            == "temporal_peak_interval"
        )
        free_horizon = now + _FREE_WINDOW_HORIZON_SEC
        for gpu_id in self._gpu_ids():
            local_sync = (
                sync_point
                if (primary_gpus is None or gpu_id in primary_gpus)
                else free_horizon
            )
            start = now
            if start >= local_sync:
                continue
            tl = self.timelines.get(gpu_id)
            boundaries = {start, local_sync}
            entries = getattr(tl, "active_entries", None) if tl else None
            if entries is None:
                entries = getattr(tl, "entries", ()) if tl else ()
            for entry in entries or ():
                if getattr(entry, "is_completed", False):
                    continue
                if getattr(entry, "is_invalidated", False) or getattr(
                    entry, "is_evict_masked", False
                ):
                    continue
                if getattr(entry, "is_predicted", False):
                    vram = self._as_float(
                        getattr(entry, "predicted_vram_mb", None), default=0.0
                    )
                    ram = self._as_float(
                        getattr(entry, "predicted_ram_mb", None), default=0.0
                    )
                    if (vram or 0.0) <= 0.0 and (ram or 0.0) <= 0.0:
                        continue
                    for field in ("start_time", "predicted_end_time"):
                        boundary = self._as_float(
                            getattr(entry, field, None), default=math.inf
                        )
                        if start < boundary < local_sync:
                            boundaries.add(boundary)
                has_valid_temporal = getattr(
                    entry, "_has_valid_temporal_reservation", None
                )
                if not temporal_mode or not callable(has_valid_temporal):
                    continue
                if not has_valid_temporal():
                    continue
                peak_start = self._as_float(
                    getattr(entry, "peak_start_time", None),
                    default=math.inf,
                )
                peak_end_getter = getattr(
                    entry,
                    "temporal_peak_reservation_end_time",
                    None,
                )
                peak_end = self._as_float(
                    peak_end_getter()
                    if callable(peak_end_getter)
                    else getattr(entry, "peak_end_time", None),
                    default=math.inf,
                )
                reservation_end_getter = getattr(
                    entry,
                    "vram_reservation_end_time",
                    None,
                )
                reservation_end = self._as_float(
                    reservation_end_getter()
                    if callable(reservation_end_getter)
                    else getattr(entry, "predicted_end_time", None),
                    default=math.inf,
                )
                for boundary in (peak_start, peak_end, reservation_end):
                    if start < boundary < local_sync:
                        boundaries.add(boundary)
            ordered_boundaries = sorted(boundaries)
            for boundary_index in range(len(ordered_boundaries) - 1):
                window_start = ordered_boundaries[boundary_index]
                window_end = ordered_boundaries[boundary_index + 1]
                if window_end <= window_start:
                    continue
                free_vram = _safe_int(
                    tl.available_vram_at(
                        window_start,
                        exclude_task_id="__phase_slack_window__",
                    )
                    if tl
                    else 0
                )
                windows.append(
                    SlackWindow(
                        gpu_id=gpu_id,
                        start_time=window_start,
                        end_time=window_end,
                        free_vram_mb=max(0, free_vram),
                        free_ram_mb=_safe_int(
                            self._as_float(
                                self.timelines.available_host_ram_at(window_start)
                            )
                        ),
                        resident_components=self._resident_components(gpu_id),
                        latest_safe_finish_time=window_end,
                    )
                )
        return windows


    async def _solve_with_phase_order(
        self,
        pending_tasks: list[dict[str, Any]],
        *,
        envelope: PhaseEnvelope | None,
        commit_predictions: bool,
    ) -> tuple[list[DispatchPlan], dict[str, PlanningExhausted]]:
        import asyncio

        self._delegate._self_concurrency_best_n_cache.clear()
        getattr(self._delegate, "_reciprocal_query_results", {}).clear()
        self.timelines.gc_orphaned_predicted_entries()
        primary_id = (
            envelope.frontier.campaign_id
            if envelope
            else self._primary_campaign_id(pending_tasks)
        )
        self._delegate._cached_primary_id = primary_id
        self._delegate._cached_primary_id_set = True

        primary_cq = self._campaign_queues().get(primary_id or "")
        primary_budget = self._delegate._build_primary_deadline_tracker(
            primary_cq, self._gpu_ids()
        )
        primary_tail_slo_epoch = self._primary_tail_slo_epoch(envelope)
        primary_tail_slo_phase_index = min(
            (key.phase_index for key in self._primary_tail_slo_phase_keys(envelope))
            if envelope is not None
            else (),
            default=0,
        )
        self._primary_tail_slo_rejections.clear()
        current_solve = CurrentSolvePlacements.for_task_ids(
            [str(task.get("task_id", "") or "") for task in pending_tasks]
        )
        campaign_ids = {
            str(task.get("campaign_id", "") or "") for task in pending_tasks
        }
        component_orders = {
            campaign_id: self._component_order(campaign_id, pending_tasks)
            for campaign_id in campaign_ids
        }
        phase_keys: dict[int, PhaseKey] = {}
        phase_groups: dict[PhaseKey, list[dict[str, Any]]] = {}
        for task in pending_tasks:
            campaign_id = str(task.get("campaign_id", "") or "")
            key = self._phase_key_for_task(
                task,
                pending_tasks,
                component_order=component_orders[campaign_id],
            )
            phase_keys[id(task)] = key
            phase_groups.setdefault(key, []).append(task)

        rr_ranks: dict[int, int] = {}
        for phase_tasks in phase_groups.values():
            first_rank_by_task_id: dict[str, int] = {}
            for rank, task in enumerate(
                sorted(phase_tasks, key=self._task_sequence_key)
            ):
                task_id = str(task.get("task_id", "") or "")
                if task_id:
                    first_rank_by_task_id.setdefault(task_id, rank)
            for task in phase_tasks:
                task_id = str(task.get("task_id", "") or "")
                rr_ranks[id(task)] = first_rank_by_task_id.get(task_id, 0)

        ordered_campaign_ids = self._ordered_campaign_ids(pending_tasks)
        campaign_ranks = {
            campaign_id: rank for rank, campaign_id in enumerate(ordered_campaign_ids)
        }
        task_campaign_ranks = {
            id(task): campaign_ranks.get(
                str(task.get("campaign_id", "") or ""), len(ordered_campaign_ids)
            )
            for task in pending_tasks
        }
        plans: list[DispatchPlan] = []
        skipped: dict[str, PlanningExhausted] = {}
        task_ids = [str(t.get("task_id", "") or "") for t in pending_tasks]
        try:
            for task in sorted(
                pending_tasks,
                key=lambda t: self._sort_key(
                    t,
                    pending_tasks,
                    primary_id,
                    phase_key=phase_keys[id(t)],
                    campaign_rank=task_campaign_ranks[id(t)],
                    phase_task_rr_rank=rr_ranks[id(t)],
                ),
            ):
                await asyncio.sleep(0)
                task_id = str(task.get("task_id", "") or "")
                campaign_id = str(task.get("campaign_id", "") or "")
                component = str(task.get("component", "") or "").strip().lower()
                if task_id:
                    current_solve.begin_task(task_id)
                if not task_id or not campaign_id or not component:
                    skipped[task_id] = PlanningExhausted(
                        "phase planner task invariant missing"
                    )
                    continue
                is_primary = campaign_id == primary_id
                task["_phase_campaign_rank"] = task_campaign_ranks[id(task)]
                task["_phase_task_rr_rank"] = rr_ranks[id(task)]
                context = self._dynamic_batch_context
                joint_dynamic = bool(
                    context is not None and context.has_joint_candidates(task_id)
                )
                admission: SlackAdmission | None = None
                candidate_evaluations: list[dict[str, Any]] = []
                primary_tail_filter = (
                    self._primary_tail_slo_filter(
                        task_id,
                        primary_tail_slo_epoch,
                        primary_tail_slo_phase_index,
                    )
                    if not is_primary
                    else None
                )

                def _phase_candidate_filter(
                    gpu_id: str,
                    eft: float,
                    projection: Any,
                    option: Any,
                    *,
                    _primary_tail_filter: Any = primary_tail_filter,
                    _component: str = component,
                    _envelope: PhaseEnvelope | None = envelope,
                    _task: dict[str, Any] = task,
                    _evaluations: list[dict[str, Any]] = candidate_evaluations,
                ) -> bool:
                    reason = ""
                    if _primary_tail_filter is not None and (
                        projection is None or not _primary_tail_filter(projection)
                    ):
                        reason = "primary_tail_slo"
                    elif self._lease_blocks_gpu(
                        gpu_id,
                        _component,
                        _envelope,
                        _task,
                    ):
                        reason = "hard_lease"
                    elif not self._candidate_worker_front_has_capacity(
                        _component,
                        gpu_id,
                    ):
                        reason = "worker_front"
                    else:
                        profile = getattr(option, "profile", None)
                        if (
                            self._find_backfill_slack(
                                _task,
                                _envelope,
                                consume=False,
                                gpu_id=gpu_id,
                                profile=profile,
                            )
                            is None
                        ):
                            reason = "phase_slack"
                    _evaluations.append(
                        {
                            "gpu_id": str(gpu_id),
                            "eft": float(eft),
                            "allowed": not reason,
                            "reason": reason or "admitted",
                        }
                    )
                    return not reason

                def _release_admission(
                    is_joint: bool,
                    solve_context: DynamicBatchSolveContext | None,
                    current_task_id: str,
                    current_admission: SlackAdmission | None,
                    current_envelope: PhaseEnvelope | None,
                ) -> None:
                    if is_joint and solve_context is not None:
                        solve_context.release_selected(current_task_id)
                    else:
                        self._release_backfill_slack(
                            current_admission,
                            current_envelope,
                        )

                try:
                    plan = self._build_round_robin_plan(
                        task,
                        is_backfill=not is_primary,
                        admission=admission,
                        primary_budget=primary_budget,
                        current_solve=current_solve,
                        candidate_filter=(
                            _phase_candidate_filter if not is_primary else None
                        ),
                    )
                    if joint_dynamic and context is not None:
                        admission = context.selected_admission(task_id)
                except PlanningExhausted as failure:
                    _release_admission(
                        joint_dynamic,
                        context,
                        task_id,
                        admission,
                        envelope,
                    )
                    self._record_alternate_gpu_retry(
                        task,
                        candidate_evaluations,
                        selected_gpu=None,
                    )
                    skipped[task_id] = failure
                    if task_id in self._primary_tail_slo_rejections:
                        self._log_backfill_decision(task, "reject", "primary_tail_slo")
                    continue
                if not is_primary and self._violates_hard_lease(plan, envelope, task):
                    _release_admission(
                        joint_dynamic,
                        context,
                        task_id,
                        admission,
                        envelope,
                    )
                    self.timelines.remove_predicted_entries_for_task(task_id)
                    skipped[task_id] = PlanningExhausted(
                        "phase hard residency lease rejected placement"
                    )
                    self._log_backfill_decision(task, "reject", "hard_lease")
                    continue
                if not is_primary and not self._plan_worker_front_has_capacity(plan):
                    _release_admission(
                        joint_dynamic,
                        context,
                        task_id,
                        admission,
                        envelope,
                    )
                    self.timelines.remove_predicted_entries_for_task(task_id)
                    skipped[task_id] = PlanningExhausted("phase worker front saturated")
                    self._log_backfill_decision(task, "reject", "worker_front")
                    continue
                if not is_primary and not joint_dynamic:
                    admission = self._find_backfill_slack(
                        task,
                        envelope,
                        consume=True,
                        gpu_id=str(plan.target_gpu_id),
                    )
                    if admission is None:
                        self.timelines.remove_predicted_entries_for_task(task_id)
                        skipped[task_id] = PlanningExhausted(
                            "slack feasibility rejected"
                        )
                        reason = str(
                            task.pop("_phase_slack_rejection_reason", "no_slack")
                        )
                        self._log_backfill_decision(task, "reject", reason)
                        continue
                if not is_primary:
                    self._log_backfill_decision(
                        task,
                        "accept",
                        admission.reason if admission else "slack_admitted",
                        admission,
                    )
                self._annotate_plan(
                    plan,
                    envelope=envelope,
                    task=task,
                    slack_admission=admission,
                )
                retry_record = self._record_alternate_gpu_retry(
                    task,
                    candidate_evaluations,
                    selected_gpu=str(plan.target_gpu_id),
                )
                if retry_record is not None:
                    plan.worker_metadata["phase_scheduler"]["alternate_gpu_retry"] = (
                        retry_record
                    )
                if not is_primary and primary_tail_slo_epoch is not None:
                    reciprocal = dict(
                        (getattr(plan, "worker_metadata", {}) or {}).get(
                            "reciprocal_interference", {}
                        )
                        or {}
                    )
                    plan.worker_metadata["primary_tail_slo"] = {
                        "factor": primary_tail_slo_epoch.factor,
                        "baseline_tail_wall": primary_tail_slo_epoch.baseline_tail_wall,
                        "reference_tail_wall": primary_tail_slo_epoch.reference_tail_wall,
                        "cap_tail_wall": primary_tail_slo_epoch.cap_tail_wall,
                        "projected_tail_wall": _safe_float(
                            reciprocal.get("primary_tail_after"), math.inf
                        ),
                        "qualified_revision": primary_tail_slo_epoch.qualified_revision,
                    }
                plans.append(plan)
            self._claim_successor_residency_actions(
                primary_id or "", pending_tasks, envelope
            )
            return plans, skipped
        finally:
            if not commit_predictions:
                self._delegate._detach_batch_solve_predictions(task_ids)

    def _build_round_robin_plan(
        self,
        task: dict[str, Any],
        *,
        is_backfill: bool,
        admission: SlackAdmission | None = None,
        primary_budget: Any = None,
        current_solve: CurrentSolvePlacements | None = None,
        reciprocal_candidate_filter: Any = None,
        candidate_filter: Any = None,
    ) -> DispatchPlan:
        task_id = str(task.get("task_id", "") or "")
        campaign_id = str(task.get("campaign_id", "") or "")
        component = str(task.get("component", "") or "").strip().lower()
        input_size = _safe_float(task.get("input_size"))
        gpu_ids = [admission.gpu_id] if admission else self._gpu_ids()
        placement_args = (
            task_id,
            campaign_id,
            component,
            input_size,
            is_backfill,
            primary_budget,
            gpu_ids,
            current_solve or CurrentSolvePlacements(),
        )
        placement_kwargs: dict[str, Any] = {
            "config_fingerprint": str(task.get("config_fingerprint", "") or ""),
            "input_fingerprint": str(task.get("input_fingerprint", "") or ""),
        }
        if reciprocal_candidate_filter is not None:
            placement_kwargs["reciprocal_candidate_filter"] = (
                reciprocal_candidate_filter
            )
        if candidate_filter is not None:
            placement_kwargs["candidate_filter"] = candidate_filter
        plan = self._delegate._place_single(*placement_args, **placement_kwargs)
        if not isinstance(plan, DispatchPlan):
            raise PlanningExhausted("phase round-robin delegate returned bad plan")
        if not is_backfill:
            self._log_backfill_decision(
                task,
                "primary",
                admission.reason if admission else "round_robin_eft",
                admission,
            )
        return plan

    def _rotated_gpu_ids(self) -> list[str]:
        gpu_ids = self._gpu_ids()
        if not gpu_ids:
            raise PlanningExhausted("phase round-robin has no GPU ids")
        idx = self._rr_cursor % len(gpu_ids)
        self._rr_cursor = (idx + 1) % len(gpu_ids)
        return gpu_ids[idx:] + gpu_ids[:idx]

    def _next_round_robin_gpu(self, task: dict[str, Any]) -> str:
        gpu_ids = self._gpu_ids()
        if not gpu_ids:
            raise PlanningExhausted("phase round-robin has no GPU ids")
        count = len(gpu_ids)
        for offset in range(count):
            idx = (self._rr_cursor + offset) % count
            gpu_id = gpu_ids[idx]
            if self._planned_start_time(task, gpu_id) < math.inf:
                self._rr_cursor = (idx + 1) % count
                return gpu_id
        raise PlanningExhausted("phase round-robin found no resource-feasible GPU")

    def _planned_start_time(self, task: dict[str, Any], gpu_id: str) -> float:
        component = str(task.get("component", "") or "").strip().lower()
        config = str(task.get("config_fingerprint", "") or "")
        input_size = _safe_float(task.get("input_size"))
        if self._force_single_self_concurrency_blocks_gpu(
            component,
            gpu_id,
            str(task.get("task_id", "") or ""),
        ):
            return math.inf
        vram = self._predict_vram(component, input_size, gpu_id, config)
        ram = self._predict_ram(component, input_size, gpu_id, config)
        dual_fit = getattr(self.timelines, "earliest_dual_fit_time", None)
        if callable(dual_fit):
            duration = self._candidate_upper_duration(task, gpu_id)
            fit = dual_fit(
                gpu_id,
                vram,
                ram,
                horizon_sec=max(120.0, duration),
                exclude_task_id=str(task.get("task_id", "") or ""),
                required_duration_sec=duration,
                candidate_component=component,
                candidate_config_fingerprint=config,
            )
            return self._as_float(fit) if fit is not None else math.inf
        return time.time()

    def _violates_hard_lease(
        self,
        plan: DispatchPlan,
        envelope: PhaseEnvelope | None,
        task: dict[str, Any],
    ) -> bool:
        return self._lease_blocks_gpu(
            str(plan.target_gpu_id), plan.component, envelope, task
        )

    def _lease_blocks_gpu(
        self,
        gpu_id: str,
        component: str,
        envelope: PhaseEnvelope | None,
        task: dict[str, Any],
    ) -> bool:
        _ = (gpu_id, component, envelope, task)
        return False

    def _sort_key(
        self,
        task: dict[str, Any],
        pending_tasks: list[dict[str, Any]],
        primary_id: str | None,
        *,
        phase_key: PhaseKey | None = None,
        campaign_rank: int | None = None,
        phase_task_rr_rank: int | None = None,
    ) -> tuple[Any, ...]:
        campaign_id = str(task.get("campaign_id", "") or "")
        component = str(task.get("component", "") or "").strip().lower()
        config = str(task.get("config_fingerprint", "") or "")
        key = phase_key or self._phase_key_for_task(task, pending_tasks)
        campaign_rank = (
            self._campaign_rank(campaign_id, pending_tasks)
            if campaign_rank is None
            else campaign_rank
        )
        phase_task_rr_rank = (
            self._phase_task_rr_rank(task, pending_tasks)
            if phase_task_rr_rank is None
            else phase_task_rr_rank
        )
        is_primary = bool(primary_id and campaign_id == primary_id)
        return (
            0 if is_primary else 2,
            key.phase_index if is_primary else campaign_rank,
            phase_task_rr_rank if is_primary else key.phase_index,
            0 if is_primary else phase_task_rr_rank,
            0 if self._has_reusable_residency(component, config) else 1,
            0 if self._has_warm_component(component) else 1,
            0 if self._cancel_safe(task) else 1,
            str(task.get("task_id", "") or ""),
        )

    def _task_sequence_key(self, task: dict[str, Any]) -> tuple[Any, ...]:
        return (
            self._as_float(task.get("created_at"), default=0.0),
            str(task.get("nf_task_id", "") or ""),
            str(task.get("task_id", "") or ""),
        )

    def _phase_task_rr_rank(
        self, task: dict[str, Any], pending_tasks: list[dict[str, Any]]
    ) -> int:
        task_id = str(task.get("task_id", "") or "")
        if not task_id:
            return 0
        key = self._phase_key_for_task(task, pending_tasks)
        phase_tasks = [
            candidate
            for candidate in pending_tasks
            if self._phase_key_for_task(candidate, pending_tasks) == key
        ]
        ordered_ids = [
            str(candidate.get("task_id", "") or "")
            for candidate in sorted(phase_tasks, key=self._task_sequence_key)
        ]
        try:
            return ordered_ids.index(task_id)
        except ValueError:
            return 0

    def _hierarchy_tier_name(
        self,
        plan: DispatchPlan,
        envelope: PhaseEnvelope | None,
        task: dict[str, Any],
    ) -> str:
        if not plan.is_backfill:
            return "primary_executable_frontier"
        primary_id = envelope.frontier.campaign_id if envelope else ""
        campaign_id = str(task.get("campaign_id", "") or plan.campaign_id or "")
        rank = _safe_int(task.get("_phase_campaign_rank"))
        if primary_id and campaign_id != primary_id and rank <= 1:
            return "successor_ready_backfill"
        return "later_ready_backfill"

    def _fits_slack(self, task: dict[str, Any], envelope: PhaseEnvelope | None) -> bool:
        return self._find_backfill_slack(task, envelope, consume=False) is not None

    def _claim_backfill_slack(
        self,
        task: dict[str, Any],
        envelope: PhaseEnvelope | None,
    ) -> SlackAdmission | None:
        return self._find_backfill_slack(task, envelope, consume=True)

    def _find_backfill_slack(
        self,
        task: dict[str, Any],
        envelope: PhaseEnvelope | None,
        *,
        consume: bool,
        gpu_id: str | None = None,
        profile: DynamicBatchCandidate | None = None,
    ) -> SlackAdmission | None:
        if envelope is None:
            target_gpu = str(gpu_id) if gpu_id is not None else self._first_gpu_id()
            return SlackAdmission("", target_gpu, time.time(), time.time(), 0.0)
        component = str(task.get("component", "") or "").strip().lower()
        task_id = str(task.get("task_id", "") or "")
        task_uses_full_wall = getattr(
            self.timelines,
            "task_uses_full_wall_reservation",
            None,
        )
        temporal_windows = (
            getattr(
                self.timelines,
                "_vram_reservation_model_name",
                "full_wall",
            )
            == "temporal_peak_interval"
        )
        temporal_mode = temporal_windows and not (
            callable(task_uses_full_wall) and task_uses_full_wall(task_id)
        )
        now = time.time()
        gpu_latest_safe_finish: dict[str, float] = {}
        if temporal_windows:
            for candidate in envelope.allowed_slack_windows:
                gpu_latest_safe_finish[candidate.gpu_id] = max(
                    gpu_latest_safe_finish.get(candidate.gpu_id, -math.inf),
                    candidate.latest_safe_finish_time,
                )
        rejection_reasons: set[str] = set()
        windows = sorted(
            envelope.allowed_slack_windows,
            key=lambda w: (
                max(now, w.start_time)
                + self._candidate_upper_duration(task, w.gpu_id, profile=profile)
                + self.cleanup_margin_sec
            ),
        )
        for window in windows:
            if gpu_id is not None and window.gpu_id != str(gpu_id):
                continue
            if self._lease_blocks_gpu(window.gpu_id, component, envelope, task):
                rejection_reasons.add("lease")
                continue
            if self._force_single_self_concurrency_blocks_gpu(
                component,
                window.gpu_id,
                str(task.get("task_id", "") or ""),
            ):
                rejection_reasons.add("force_single")
                continue
            vram, ram = self._task_resource_need(
                task,
                window.gpu_id,
                profile=profile,
            )
            config = str(task.get("config_fingerprint", "") or "")
            if ram > window.free_ram_mb:
                rejection_reasons.add("resource")
                continue
            if temporal_mode:
                tl = self.timelines.get(window.gpu_id)
                total_vram = getattr(tl, "total_vram_mb", 0.0) or 0.0
                if total_vram > 0.0 and vram > total_vram:
                    rejection_reasons.add("resource")
                    continue
            elif vram > window.free_vram_mb:
                rejection_reasons.add("resource")
                continue
            duration = self._candidate_upper_duration(
                task,
                window.gpu_id,
                profile=profile,
            )
            start = max(now, window.start_time)
            interval_fit = getattr(self.timelines, "candidate_interval_fits", None)
            finish = start + duration + self.cleanup_margin_sec
            latest_safe_finish = gpu_latest_safe_finish.get(
                window.gpu_id, window.latest_safe_finish_time
            )
            if finish <= latest_safe_finish:
                complete_fit = not callable(interval_fit) or bool(
                    interval_fit(
                        window.gpu_id,
                        start,
                        start + duration,
                        vram,
                        ram,
                        exclude_task_id=task_id,
                        candidate_component=component,
                        candidate_config_fingerprint=config,
                        candidate_force_full_wall=not temporal_mode,
                        _captured_now=now,
                    )
                )
                if complete_fit:
                    if consume:
                        window.start_time = finish
                    return SlackAdmission(
                        window_id=window.window_id,
                        gpu_id=window.gpu_id,
                        start_time=start,
                        finish_time=finish,
                        duration_sec=duration,
                    )
                rejection_reasons.add("interval")
            else:
                rejection_reasons.add("duration")
            if self._cancel_safe(task):
                init_cost = 0.0
                if not self._has_reusable_residency_on_gpu(
                    component, window.gpu_id, config
                ):
                    init_cost = self._get_init_latency(component, window.gpu_id)
                cancel_cost = (
                    init_cost
                    + self._cancel_requeue_cost(task)
                    + self.cleanup_margin_sec
                )
                cancel_at = gpu_latest_safe_finish.get(
                    window.gpu_id, window.latest_safe_finish_time
                )
                if start + cancel_cost <= cancel_at:
                    cancel_fit = not callable(interval_fit) or bool(
                        interval_fit(
                            window.gpu_id,
                            start,
                            cancel_at,
                            vram,
                            ram,
                            exclude_task_id=task_id,
                            candidate_component=component,
                            candidate_config_fingerprint=config,
                            candidate_wall_end_time=start + duration,
                            candidate_force_full_wall=not temporal_mode,
                            _captured_now=now,
                        )
                    )
                    if cancel_fit:
                        if consume:
                            window.start_time = cancel_at
                        return SlackAdmission(
                            window_id=window.window_id,
                            gpu_id=window.gpu_id,
                            start_time=start,
                            finish_time=start + duration,
                            duration_sec=duration,
                            cancel_tier=True,
                            reason="cooperative_cancel_before_sync",
                        )
                    rejection_reasons.add("interval")
                else:
                    rejection_reasons.add("cancel_window")
        if consume:
            detail = "+".join(sorted(rejection_reasons)) or "no_window"
            task["_phase_slack_rejection_reason"] = f"no_slack:{detail}"
            v2, r2 = self._task_resource_need(task, "2", profile=profile)
            _LOG.warning(
                "[backfill-debug] task=%s comp=%s reject=%s task_vram=%.0f task_ram=%.0f "
                "windows=[%s]",
                task_id,
                component,
                detail,
                v2,
                r2,
                ", ".join(
                    f"{w.gpu_id}(vram_free={w.free_vram_mb},ram_free={w.free_ram_mb},end={w.latest_safe_finish_time - now:.0f}s)"
                    for w in (envelope.allowed_slack_windows if envelope else [])
                ),
            )
        return None

    def _release_backfill_slack(
        self,
        admission: SlackAdmission | None,
        envelope: PhaseEnvelope | None,
    ) -> None:
        if admission is None or envelope is None:
            return
        for window in envelope.allowed_slack_windows:
            if window.window_id == admission.window_id:
                window.start_time = min(window.start_time, admission.start_time)
                return

    def _task_resource_need(
        self,
        task: dict[str, Any],
        gpu_id: str,
        *,
        profile: DynamicBatchCandidate | None = None,
    ) -> tuple[float, float]:
        task_id = str(task.get("task_id", "") or "")
        profile = profile or self._dynamic_batch_profiles.get((task_id, str(gpu_id)))
        if profile is not None:
            return max(0.0, profile.vram_upper), max(0.0, profile.ram_upper)
        component = str(task.get("component", "") or "").strip().lower()
        config = str(task.get("config_fingerprint", "") or "")
        input_size = _safe_float(task.get("input_size"))
        return (
            self._predict_vram(component, input_size, gpu_id, config),
            self._predict_ram(component, input_size, gpu_id, config),
        )

    def _explicit_cancel_safe(self, task: dict[str, Any]) -> bool:
        return bool(task.get("active_cancel_safe", task.get("cancel_safe", False)))

    def _cancel_safe(self, task: dict[str, Any]) -> bool:
        if "active_cancel_safe" in task or "cancel_safe" in task:
            return self._explicit_cancel_safe(task)
        if "preemptible" in task:
            return bool(task.get("preemptible"))
        return False

    def _cancel_requeue_cost(self, task: dict[str, Any]) -> float:
        for key in ("cancel_requeue_cost_sec", "requeue_cost_sec", "cancel_cost_sec"):
            if key in task:
                return max(0.0, self._as_float(task.get(key), default=math.inf))
        return 1.0 if self._cancel_safe(task) else math.inf

    def _candidate_upper_duration(
        self,
        task: dict[str, Any],
        gpu_id: str | None = None,
        *,
        profile: DynamicBatchCandidate | None = None,
    ) -> float:
        component = str(task.get("component", "") or "").strip().lower()
        config = str(task.get("config_fingerprint", "") or "")
        input_size = _safe_float(task.get("input_size"))
        target_gpu = gpu_id or self._first_gpu_id()
        task_id = str(task.get("task_id", "") or "")
        profile = profile or self._dynamic_batch_profiles.get(
            (task_id, str(target_gpu))
        )
        if profile is not None:
            runtime = (
                profile.latency_mean
                if self.backfill_latency_basis == "mean"
                else profile.latency_upper
            )
        else:
            mean = self._predict_latency(component, input_size, target_gpu, config)
            sigma = self._predict_latency_sigma(
                component, input_size, target_gpu, config
            )
            uncertainty_z = self._latency_uncertainty_z(component, input_size, config)
            runtime = self._latency_with_basis(mean, sigma, uncertainty_z)
        if not self._has_reusable_residency_on_gpu(component, target_gpu, config):
            runtime += self._get_init_latency(component, target_gpu)
        return max(0.0, runtime)

    def _log_backfill_decision(
        self,
        task: dict[str, Any],
        decision: str,
        reason: str,
        admission: SlackAdmission | None = None,
    ) -> None:
        if str(task.get("campaign_id", "") or "") == self._primary_campaign_id([task]):
            return
        _LOG.info(
            "[phase-plan] backfill decision=%s reason=%s task=%s campaign=%s "
            "component=%s window=%s gpu=%s finish=%.3f cancel_tier=%s",
            decision,
            reason,
            str(task.get("task_id", "") or ""),
            str(task.get("campaign_id", "") or ""),
            str(task.get("component", "") or ""),
            admission.window_id if admission else "",
            admission.gpu_id if admission else "",
            admission.finish_time if admission else 0.0,
            admission.cancel_tier if admission else False,
        )

    def _annotate_plan(
        self,
        plan: DispatchPlan,
        *,
        envelope: PhaseEnvelope | None = None,
        task: dict[str, Any] | None = None,
        slack_admission: SlackAdmission | None = None,
    ) -> None:
        task = task or {
            "task_id": plan.task_id,
            "campaign_id": plan.campaign_id,
            "component": plan.component,
        }
        key = self._phase_key_for_task(task, [task])
        estimate = self._phase_estimates.get(key, PhaseEstimate())
        envelope = envelope or self._last_phase_envelope
        meta = plan.worker_metadata.setdefault("phase_scheduler", {})
        batch_meta = self._attach_dynamic_batch_metadata(plan)
        meta.update(
            {
                "phase_key": {
                    "campaign_id": key.campaign_id,
                    "phase_index": key.phase_index,
                    "phase_name": key.phase_name,
                    "component": key.component,
                    "config_fingerprint": key.config_fingerprint,
                    "barrier_kind": key.barrier_kind,
                },
                "estimate_source": estimate.source,
                "estimate_confidence": estimate.confidence,
                "residency_reused": self._has_reusable_residency(
                    plan.component, key.config_fingerprint
                ),
                "phase_tier": self._phase_tier(plan, envelope, task),
                "inner_phase_ranking": "phase_task_round_robin",
                "inner_phase_placement": "eft",
                "runtime_prediction_basis": self.backfill_latency_basis,
                "resource_prediction_basis": (
                    "dynamic_batch_profile" if batch_meta else "mean"
                ),
                "backfill_admission_basis": self.backfill_latency_basis,
                "hierarchy_tier": self._hierarchy_tier_name(plan, envelope, task),
                "phase_task_rr_rank": _safe_int(task.get("_phase_task_rr_rank")),
                "cooperative_cancel_backfill": bool(
                    slack_admission.cancel_tier
                    if slack_admission
                    else (
                        plan.is_backfill
                        and self._cancel_safe(task)
                        and self._cancel_requeue_cost(task) < math.inf
                    )
                ),
            }
        )
        if envelope is None:
            return
        meta["protected_sync_point"] = envelope.next_sync_point
        meta["residency_lease_ids"] = [
            lease.lease_id
            for lease in envelope.protected_model_leases
            if lease.component == plan.component
        ]
        meta["slack_window_ids"] = (
            [slack_admission.window_id]
            if slack_admission
            else [
                window.window_id
                for window in envelope.allowed_slack_windows
                if window.gpu_id == str(plan.target_gpu_id)
            ]
        )
        if slack_admission:
            meta["backfill_admission"] = {
                "window_id": slack_admission.window_id,
                "gpu_id": slack_admission.gpu_id,
                "start_time": slack_admission.start_time,
                "finish_time": slack_admission.finish_time,
                "duration_sec": slack_admission.duration_sec,
                "cancel_tier": slack_admission.cancel_tier,
                "reason": slack_admission.reason,
            }
        meta["residency_action_ids"] = [
            action.action_id
            for action in envelope.residency_actions
            if action.working_set.key.component == plan.component
        ]
        meta["hard_residency_lease"] = bool(meta["residency_lease_ids"])
        meta["standalone_residency_actions"] = [
            {
                "action_id": action.action_id,
                "component": action.working_set.key.component,
                "campaign_id": action.campaign_id,
                "target_gpu_id": action.target_gpu_id,
                "reason": action.reason,
                "scheduled": action.scheduled,
            }
            for action in envelope.residency_actions
        ]
        self._attach_preinit(plan, envelope)

    def _phase_tier(
        self,
        plan: DispatchPlan,
        envelope: PhaseEnvelope | None,
        task: dict[str, Any],
    ) -> str:
        if envelope is None:
            return (
                "primary_executable_frontier"
                if not plan.is_backfill
                else "inter_campaign_backfill"
            )
        if plan.campaign_id == envelope.frontier.campaign_id:
            key = self._phase_key_for_task(task, [task])
            if key.phase_index > min(
                (k.phase_index for k in envelope.frontier.phase_keys),
                default=key.phase_index,
            ):
                return "primary_downstream_ready_frontier"
            return "primary_executable_frontier"
        if _safe_int(task.get("_phase_campaign_rank"), 9999) <= 1:
            return "successor_campaign_backfill"
        return "later_campaign_backfill"

    def _attach_preinit(self, plan: DispatchPlan, envelope: PhaseEnvelope) -> None:
        if plan.campaign_id != envelope.frontier.campaign_id or plan.is_backfill:
            return
        existing = {
            (action.component, action.target_gpu_id)
            for action in plan.pre_init_schedule
        }
        for ws in envelope.frontier.preinit_candidates:
            key = (ws.key.component, str(plan.target_gpu_id))
            if key in existing:
                continue
            plan.pre_init_schedule.append(
                PreInitAction(
                    component=ws.key.component,
                    target_gpu_id=str(plan.target_gpu_id),
                    trigger_at=max(time.time(), _safe_float(plan.planned_start_time)),
                    init_duration_sec=ws.init_upper_sec,
                    weight_vram_mb=_safe_float(ws.weight_vram_mb),
                )
            )
            existing.add(key)


    def _phase_key_for_task(
        self,
        task: dict[str, Any],
        pending_tasks: list[dict[str, Any]],
        *,
        component_order: list[str] | None = None,
    ) -> PhaseKey:
        return self._phase_key_for_values(
            campaign_id=str(task.get("campaign_id", "") or ""),
            component=str(task.get("component", "") or "").strip().lower(),
            config_fingerprint=str(task.get("config_fingerprint", "") or ""),
            process_name=str(task.get("process_name", "") or ""),
            pending_tasks=pending_tasks,
            component_order=component_order,
        )

    def _phase_key_for_values(
        self,
        *,
        campaign_id: str,
        component: str,
        pending_tasks: list[dict[str, Any]],
        config_fingerprint: str = "",
        process_name: str = "",
        component_order: list[str] | None = None,
    ) -> PhaseKey:
        order = (
            component_order
            if component_order is not None
            else self._component_order(campaign_id, pending_tasks)
        )
        phase_index = order.index(component) if component in order else len(order)
        return PhaseKey(
            campaign_id=campaign_id,
            phase_index=phase_index,
            phase_name=process_name or component or f"phase_{phase_index}",
            component=component,
            config_fingerprint=config_fingerprint,
            barrier_kind=self._barrier_kind(campaign_id, component),
        )

    def _component_order(
        self, campaign_id: str, pending_tasks: list[dict[str, Any]]
    ) -> list[str]:
        cq = self._campaign_queues().get(campaign_id)
        dag_context = getattr(cq, "dag_context", None) if cq else None
        raw_order = getattr(dag_context, "component_order", [])
        order = (
            [str(c).strip().lower() for c in raw_order]
            if isinstance(raw_order, list)
            else []
        )
        if not order:
            pipeline = getattr(self.campaign_scheduler, "_pipeline_dag", None)
            topological = getattr(pipeline, "topological_order", None)
            if callable(topological):
                raw = topological()
                order = (
                    [str(c).strip().lower() for c in raw]
                    if isinstance(raw, list)
                    else []
                )
        for task in pending_tasks:
            if str(task.get("campaign_id", "") or "") != campaign_id:
                continue
            component = str(task.get("component", "") or "").strip().lower()
            if component and component not in order:
                order.append(component)
        return order

    def _barrier_kind(self, campaign_id: str, component: str) -> str:
        cq = self._campaign_queues().get(campaign_id)
        dag_context = getattr(cq, "dag_context", None) if cq else None
        join_type = ""
        join = getattr(dag_context, "join_type", None)
        if callable(join):
            join_type = str(join(component) or "")
        if not join_type:
            pipeline = getattr(self.campaign_scheduler, "_pipeline_dag", None)
            join = getattr(pipeline, "join_type", None)
            join_type = str(join(component) or "") if callable(join) else ""
        if join_type == "barrier":
            return "hard"
        if join_type == "streaming":
            return "streaming"
        return "soft"

    def _timeline_entries(self) -> list[Any]:
        entries: list[Any] = []
        for gpu_id in self._gpu_ids():
            tl = self.timelines.get(gpu_id)
            if tl:
                entries.extend(list(getattr(tl, "entries", []) or []))
        return entries

    def _observed_runtimes(self, key: PhaseKey) -> list[float]:
        runtimes: list[float] = []
        for record in self._gateway_tasks().values():
            if str(getattr(record, "campaign_id", "") or "") != key.campaign_id:
                continue
            if (
                str(getattr(record, "component", "") or "").strip().lower()
                != key.component
            ):
                continue
            runtime = getattr(record, "signal_runtime_sec", None)
            timing = getattr(record, "worker_timing_us", None) or {}
            if runtime is None and isinstance(timing, dict) and timing.get("total_us"):
                runtime = _safe_float(timing["total_us"]) / 1_000_000.0
            if runtime:
                runtimes.append(_safe_float(runtime))
        return runtimes

    def _record_process_name(self, record: Any) -> str:
        context = getattr(record, "_handle_context", None)
        return (
            str(context.get("process_name") or "") if isinstance(context, dict) else ""
        )

    def _invalidate_component(self, component: str, reason: str) -> None:
        component = str(component or "").strip().lower()
        if component:
            self._invalidated_components[component] = reason

    def _gateway_tasks(self) -> dict[str, Any]:
        raw = getattr(self.campaign_scheduler, "_gateway_tasks", {}) or {}
        return raw if isinstance(raw, dict) else {}

    def _first_gpu_id(self) -> str:
        gpu_ids = self._gpu_ids()
        return gpu_ids[0] if gpu_ids else "0"

    def _phase_input_size(
        self, key: PhaseKey, pending_tasks: list[dict[str, Any]]
    ) -> float:
        sizes: list[float] = []
        for task in pending_tasks:
            if str(task.get("campaign_id", "") or "") != key.campaign_id:
                continue
            if str(task.get("component", "") or "").strip().lower() != key.component:
                continue
            value = self._as_float(task.get("input_size"), default=0.0)
            if value > 0.0:
                sizes.append(value)
        for entry in self._timeline_entries():
            if str(getattr(entry, "campaign_id", "") or "") != key.campaign_id:
                continue
            if (
                str(getattr(entry, "component", "") or "").strip().lower()
                != key.component
            ):
                continue
            value = self._as_float(getattr(entry, "input_size", 0.0), default=0.0)
            if value > 0.0:
                sizes.append(value)
        return (
            max(sizes)
            if sizes
            else self._avg_input_size(key.campaign_id, key.component)
        )

    def _avg_input_size(self, campaign_id: str, component: str) -> float:
        cq = self._campaign_queues().get(campaign_id)
        avg = getattr(cq, "avg_input_size", None) if cq else None
        return self._as_float(avg(component)) if callable(avg) else 0.0

    def _latency_with_basis(
        self,
        mean: float,
        sigma: float,
        uncertainty_z: float,
    ) -> float:
        return max(0.0, mean + max(0.0, uncertainty_z) * max(0.0, sigma))

    def _predict_latency(
        self, component: str, input_size: float, gpu_id: str, config: str = ""
    ) -> float:
        return self._call_float(
            "_predict_latency", component, input_size, gpu_id, config_fingerprint=config
        )

    def _predict_latency_sigma(
        self, component: str, input_size: float, gpu_id: str, config: str = ""
    ) -> float:
        return self._call_float(
            "_predict_latency_sigma",
            component,
            input_size,
            gpu_id,
            config_fingerprint=config,
        )

    def _predict_vram(
        self, component: str, input_size: float, gpu_id: str, config: str = ""
    ) -> float:
        return self._call_float(
            "_predict_vram",
            component,
            input_size,
            gpu_id,
            use_upper=False,
            config_fingerprint=config,
        )

    def _predict_ram(
        self, component: str, input_size: float, gpu_id: str, config: str = ""
    ) -> float:
        return self._call_float(
            "_predict_ram",
            component,
            input_size,
            gpu_id,
            use_upper=False,
            config_fingerprint=config,
        )

    def _get_init_latency(self, component: str, gpu_id: str) -> float:
        fn = getattr(self.campaign_scheduler, "_get_init_latency", None)
        return self._as_float(fn(component, gpu_id)) if callable(fn) else 0.0

    def _confidence_name(
        self, component: str, input_size: float, gpu_id: str, config: str = ""
    ) -> str:
        fn = getattr(self.campaign_scheduler, "_get_confidence", None)
        if not callable(fn):
            return "low"
        try:
            return str(
                fn(component, input_size, gpu_id, config_fingerprint=config) or "low"
            )
        except TypeError:
            fallback = fn(component, input_size, gpu_id)
            return str(fallback) if fallback else "low"

    def _call_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        fn = getattr(self.campaign_scheduler, name, None)
        if not callable(fn):
            return 0.0
        try:
            return self._as_float(fn(*args, **kwargs))
        except TypeError:
            return self._as_float(fn(*args[:3]))

    def _resident_workers(
        self, component: str, config: str = ""
    ) -> list[_ResidentWorker]:
        supervisor = getattr(self.campaign_scheduler, "_supervisor", None)
        raw_states = getattr(supervisor, "states", {}) if supervisor else {}
        states = raw_states if isinstance(raw_states, dict) else {}
        workers: list[_ResidentWorker] = []
        for name, state in states.items():
            spec = getattr(state, "spec", None)
            state_component = str(getattr(spec, "component", "") or "").strip().lower()
            if state_component != component:
                continue
            state_config = str(
                getattr(state, "config_fingerprint", "")
                or getattr(spec, "config_fingerprint", "")
                or ""
            )
            if config and state_config and state_config != config:
                continue
            gpus = self._assigned_gpus(state)
            ready = bool(getattr(state, "ready", False) and getattr(state, "addr", ""))
            workers.append(
                _ResidentWorker(
                    worker_name=str(name),
                    worker_addr=str(getattr(state, "addr", "") or ""),
                    gpu_id=gpus[0] if gpus else "",
                    reusable=ready and not self._worker_busy(state),
                    campaign_id=str(getattr(state, "campaign_id", "") or ""),
                )
            )
        return workers

    def _has_reusable_residency(self, component: str, config: str = "") -> bool:
        return any(
            worker.reusable for worker in self._resident_workers(component, config)
        )

    def _has_reusable_residency_on_gpu(
        self, component: str, gpu_id: str, config: str = ""
    ) -> bool:
        target_gpu = str(gpu_id)
        return any(
            worker.reusable and worker.gpu_id == target_gpu
            for worker in self._resident_workers(component, config)
        )

    def _has_warm_component(self, component: str) -> bool:
        return bool(self._resident_workers(component))

    def _resident_components(self, gpu_id: str) -> list[str]:
        supervisor = getattr(self.campaign_scheduler, "_supervisor", None)
        raw_states = getattr(supervisor, "states", {}) if supervisor else {}
        states = raw_states if isinstance(raw_states, dict) else {}
        components: list[str] = []
        for state in states.values():
            if gpu_id not in self._assigned_gpus(state):
                continue
            spec = getattr(state, "spec", None)
            component = str(getattr(spec, "component", "") or "").strip().lower()
            if component and component not in components:
                components.append(component)
        return components

    def _plan_worker_front_has_capacity(self, plan: DispatchPlan) -> bool:
        worker_name = str(getattr(plan, "target_worker_name", "") or "").strip()
        return self._worker_front_has_capacity(worker_name)

    def _assigned_gpus(self, state: Any) -> list[str]:
        raw = getattr(state, "assigned_gpus", None) or []
        return [str(gpu) for gpu in raw] if isinstance(raw, (list, tuple, set)) else []


@dataclass(frozen=True)
class _ResidentWorker:
    worker_name: str
    worker_addr: str
    gpu_id: str
    reusable: bool
    campaign_id: str = ""
