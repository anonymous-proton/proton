"""Planner Service: wiring, pre-plan housekeeping, GlobalPlanner initialization.

3-layer architecture: PlannerService connects GlobalPlanner to the gateway
runtime.  ``prepare_for_planning()`` refreshes GPU state and prunes stale
entries before each ``GlobalPlanner.plan()`` call.  GlobalPlanner initialization
is lazy — created on first ``_connect_supervisor_refs()`` after supervisor is
available.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from ..signals.contracts import (
    ProfileDriftEvent,
    SignalResult,
    WorkerComputeSignal,
    WorkerConcurrencyProfile,
    WorkerMemoryProfile,
    WorkerRuntimeProfile,
)
from .campaign_scheduler import CampaignScheduler
from .pipeline_dag import PipelineDAG
from .placement_context import PlacementContextManager

_LOG = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _require_strategy_choice(
    field: str,
    value: Any,
    allowed: set[str],
    *,
    aliases: dict[str, str] | None = None,
    default: str,
) -> str:
    """Decode a planner strategy knob without silent unknown fallbacks."""
    raw = "" if value is None else str(value).strip().lower()
    if raw == "":
        return default
    aliases = aliases or {}
    raw = aliases.get(raw, raw)
    if raw not in allowed:
        allowed_msg = ", ".join(sorted(allowed | set(aliases.keys())))
        raise ValueError(
            f"Unknown global_planner.{field}={value!r}; expected one of: {allowed_msg}",
        )
    return raw


@dataclasses.dataclass
class PlannerTaskRequest:
    """Extensible Nextflow task payload mapped into the Planner.

    Everything Nextflow sends about a task is normalised here so that
    the Planner interface stays stable even as the HTTP contract evolves.
    """

    component: str
    config_fingerprint: str
    input_fingerprint: str
    workload_features: dict[str, Any]
    execution_overrides: dict[str, Any]
    campaign_id: str | None
    preferred_worker_addr: str | None = None
    preferred_gpu_ids: list[str] | None = None
    scheduling_hints: dict[str, Any] = dataclasses.field(default_factory=dict)
    timeout_s: int = 300
    output_sample_count: int | None = None
    process_name: str | None = None


class PlannerService:
    """Stage 1 Pluggable Planner.

    Responsibilities (Plan 3-layer architecture):
    1. Query the SignalService for cached/historical resource projections.
    2. Wrap the projections in the extensible WorkerComputeSignal struct.
    3. Delegate placement to ``GlobalPlanner.solve()`` — HEFT min-EFT +
       MCPSE (plan  removed the
       per-worker ``SchedulingPolicy`` picker).
    4. Return (worker, reservation, intrinsic_signal) for the Dispatcher.
    5. Expose DAG context (fan-out, topology, join_type) for stage-aware
       scheduling consumed by the GlobalPlanner / RealityValidator.

    The Signal module is the sole author of WorkerComputeSignal; the Planner
    is only a consumer. Profile *construction* happens asynchronously in Stage 3.
    """

    def __init__(
        self,
        gateway_service: Any,
        pipeline_dag: PipelineDAG | None = None,
        scheduler_policy: str = "campaign_fifo",
    ) -> None:
        self._gateway = gateway_service
        self._pipeline_dag: PipelineDAG = pipeline_dag or PipelineDAG()
        self._placement_ctx = PlacementContextManager()

        self._campaign_scheduler = CampaignScheduler.create(
            scheduler_policy,
            signal_service=self._gateway._signal_service,
            pipeline_dag=self._pipeline_dag,
            gateway_tasks=self._gateway._tasks,
        )
        self._campaign_scheduler._planner_ref = self

        self._global_planner: Any = None

    def _connect_supervisor_refs(self) -> None:
        """Lazily connect InitProfile and ResourceAdmissionTracker from WorkerSupervisor.

        ``gateway_service.supervisor`` is set by ``__main__.py`` after app creation.
        """
        supervisor = getattr(self._gateway, "supervisor", None)
        if not supervisor:
            return
        if self._campaign_scheduler._init_tracker is None:
            init = getattr(supervisor, "init_tracker", None)
            if init:
                self._campaign_scheduler._init_tracker = init
        if self._campaign_scheduler._resource_tracker is None:
            vt = getattr(supervisor, "resource_tracker", None)
            if vt:
                self._campaign_scheduler._resource_tracker = vt
        if getattr(self._campaign_scheduler, "_supervisor", None) is None:
            self._campaign_scheduler._supervisor = supervisor

        if self._global_planner is None:
            try:
                from .global_planner import (
                    BayesianBackfill,
                    DAGLookaheadPreInit,
                    DeterministicEFTPlacement,
                    DeterministicHEFTPriority,
                    FIFOCampaign,
                    FIFOPriority,
                    GlobalPlanner,
                    GreedyEvictionStrategy,
                    GreedyRamEvictionStrategy,
                    HEFTPriority,
                    NoBackfill,
                    NoEviction,
                    NoPreInit,
                    NoRamEviction,
                    PointEstimateBackfill,
                    ReactivePreInit,
                    StochasticEFTPlacement,
                )

                gw_cfg = getattr(self._gateway, "_runtime_config", None) or {}
                gp_cfg = (
                    gw_cfg.get("global_planner", {}) if isinstance(gw_cfg, dict) else {}
                )
                if not gp_cfg:
                    gp_cfg = (
                        getattr(self._gateway, "_profiling_runtime", {}) or {}
                    ).get("global_planner", {})
                if not gp_cfg:
                    sup_cfg = getattr(supervisor, "config", None)
                    pr = getattr(sup_cfg, "profiling_runtime", {}) if sup_cfg else {}
                    gp_cfg = (
                        pr.get("global_planner", {}) if isinstance(pr, dict) else {}
                    )

                resource_upper_percentile = (
                    gp_cfg.get(
                        "resource_upper_percentile",
                        gw_cfg.get("resource_upper_percentile", 0.95)
                        if isinstance(gw_cfg, dict)
                        else 0.95,
                    )
                    if isinstance(gp_cfg, dict)
                    else 0.95
                )
                self._campaign_scheduler.set_resource_upper_percentile(
                    resource_upper_percentile
                )
                vram_reservation_model = (
                    gw_cfg.get("vram_reservation_model", "full_wall")
                    if isinstance(gw_cfg, dict)
                    else "full_wall"
                )
                self._campaign_scheduler.set_vram_reservation_model(
                    str(vram_reservation_model)
                )

                fanout_barrier_force = bool(gp_cfg.get("fanout_barrier_force", False))
                if "wave_projection" in gp_cfg or "wave_strategy" in gp_cfg:
                    _LOG.warning(
                        "[planner] wave_projection/wave_strategy are retired "
                        "and ignored; fan-out siblings use individual "
                        "placement."
                    )
                preinit_strategy_name = str(gp_cfg.get("pre_init_depth", "dag")).lower()

                _priority_name = (
                    str(gp_cfg.get("priority_strategy", "")).strip().lower()
                )
                if _priority_name:
                    _priority_name = _require_strategy_choice(
                        "priority_strategy",
                        _priority_name,
                        {"deterministic_heft", "fifo", "heft"},
                        default="heft",
                    )
                if _priority_name == "deterministic_heft":
                    priority = DeterministicHEFTPriority(
                        fanout_barrier_force=fanout_barrier_force,
                    )
                elif _priority_name == "fifo":
                    priority = FIFOPriority()
                elif _priority_name == "heft":
                    priority = HEFTPriority(
                        fanout_barrier_force=fanout_barrier_force,
                    )
                elif gp_cfg.get("use_heft_ordering", True):
                    priority = HEFTPriority(
                        fanout_barrier_force=fanout_barrier_force,
                    )
                else:
                    priority = FIFOPriority()

                _placement_name = (
                    str(gp_cfg.get("placement_strategy", "")).strip().lower()
                )
                if _placement_name:
                    _placement_name = _require_strategy_choice(
                        "placement_strategy",
                        _placement_name,
                        {"deterministic_eft", "stochastic_eft", "reactive_current"},
                        default="stochastic_eft",
                    )
                if _placement_name == "reactive_current":
                    from .react_campaign_planner import ReactiveCurrentPlacement

                    placement = ReactiveCurrentPlacement()
                elif _placement_name == "deterministic_eft":
                    placement = DeterministicEFTPlacement()
                else:
                    placement = StochasticEFTPlacement()

                _backfill_alias = gp_cfg.get("backfill_strategy") or gp_cfg.get(
                    "backfill_admission", "cdf"
                )
                _backfill_mode = _require_strategy_choice(
                    "backfill_strategy/backfill_admission",
                    _backfill_alias,
                    {"cdf", "point", "none"},
                    aliases={"point_estimate": "point"},
                    default="cdf",
                )
                if _backfill_mode == "point":
                    backfill = PointEstimateBackfill()
                elif _backfill_mode == "none":
                    backfill = NoBackfill()
                else:
                    backfill = BayesianBackfill()
                _camp_pri = _require_strategy_choice(
                    "campaign_priority",
                    gp_cfg.get("campaign_priority", "fifo"),
                    {"fifo", "wsjf"},
                    default="fifo",
                )
                if _camp_pri == "wsjf":
                    _LOG.warning(
                        "[planner] campaign_priority=wsjf is deprecated () — "
                        "SmithRuleCampaign retired. Falling back to FIFOCampaign."
                    )
                campaign = FIFOCampaign()
                _eviction_name = (
                    str(gp_cfg.get("eviction_strategy", "greedy")).strip().lower()
                )
                _eviction_name = _require_strategy_choice(
                    "eviction_strategy",
                    _eviction_name,
                    {"greedy", "mcpse", "min_elapsed", "no_eviction"},
                    default="greedy",
                )
                if _eviction_name == "no_eviction":
                    eviction = NoEviction()
                elif _eviction_name == "mcpse":
                    _LOG.warning(
                        "[planner] eviction_strategy=mcpse is archived. "
                        "Falling back to GreedyEvictionStrategy."
                    )
                    eviction = GreedyEvictionStrategy()
                elif _eviction_name == "min_elapsed":
                    _LOG.warning(
                        "[planner] eviction_strategy=min_elapsed is archived. "
                        "Falling back to GreedyEvictionStrategy."
                    )
                    eviction = GreedyEvictionStrategy()
                else:
                    eviction = GreedyEvictionStrategy()
                _ram_eviction_name = (
                    str(gp_cfg.get("ram_eviction_strategy", "no_eviction"))
                    .strip()
                    .lower()
                )
                _ram_eviction_name = _require_strategy_choice(
                    "ram_eviction_strategy",
                    _ram_eviction_name,
                    {"greedy", "no_eviction"},
                    default="no_eviction",
                )
                if _ram_eviction_name == "greedy":
                    ram_eviction = GreedyRamEvictionStrategy()
                else:
                    ram_eviction = NoRamEviction()
                _preinit_alias = (
                    str(gp_cfg.get("pre_init_strategy", "")).strip().lower()
                )
                _effective_preinit = (
                    _preinit_alias if _preinit_alias else preinit_strategy_name
                )
                _effective_preinit = _require_strategy_choice(
                    "pre_init_strategy/pre_init_depth",
                    _effective_preinit,
                    {
                        "0",
                        "1",
                        "dag",
                        "dag_lookahead",
                        "lookahead",
                        "reactive",
                        "none",
                        "no_preinit",
                    },
                    default="dag",
                )
                if _effective_preinit in ("0", "none", "no_preinit"):
                    preinit = NoPreInit()
                elif _effective_preinit == "reactive":
                    preinit = ReactivePreInit()
                else:
                    preinit = DAGLookaheadPreInit()

                scheduler_impl = (
                    str(gw_cfg.get("scheduler_implementation", "proton"))
                    .strip()
                    .lower()
                    if isinstance(gw_cfg, dict)
                    else "proton"
                )
                if scheduler_impl != "proton_react" and _placement_name == (
                    "reactive_current"
                ):
                    raise ValueError("reactive_current requires proton_react")
                if scheduler_impl == "proton_react" and (
                    _placement_name != "reactive_current"
                    or _backfill_mode != "point"
                    or not isinstance(priority, DeterministicHEFTPriority)
                    or not isinstance(preinit, NoPreInit)
                    or gp_cfg.get("backfill_admission", "point") != "point"
                    or str(gp_cfg.get("pre_init_depth", 0)).lower()
                    not in {"0", "none", "no_preinit"}
                    or gp_cfg.get("dynamic_grace", False)
                    or gp_cfg.get("drift_eviction", False)
                ):
                    raise ValueError(
                        "proton_react requires reactive_current/point backfill, "
                        "deterministic_heft and no_preinit"
                    )
                planner_kwargs = {}
                planner_cls = GlobalPlanner
                if scheduler_impl in {"proton_phase", "proton_heft", "proton_react"}:
                    if scheduler_impl == "proton_react":
                        from .react_campaign_planner import ReactCampaignPlanner

                        planner_cls = ReactCampaignPlanner
                        config_key = "react_scheduler"
                    elif scheduler_impl == "proton_heft":
                        from .heft_campaign_planner import HeftCampaignPlanner

                        planner_cls = HeftCampaignPlanner
                        config_key = "heft_scheduler"
                    else:
                        from .phase_campaign_planner import PhaseCampaignPlanner

                        planner_cls = PhaseCampaignPlanner
                        config_key = "phase_scheduler"
                    scheduler_cfg = gw_cfg.get(config_key, {})
                    if scheduler_impl == "proton_react":
                        allowed = {
                            "enabled",
                            "reciprocal_interference_correction",
                            "backfill_latency_basis",
                            "backfill_primary_tail_slo_factor",
                            "dynamic_batching",
                        }
                        if not isinstance(scheduler_cfg, dict):
                            raise ValueError("react_scheduler must be an object")
                        if set(scheduler_cfg) - allowed:
                            raise ValueError("react_scheduler has unsupported settings")
                        if scheduler_cfg.get("enabled", True) is not True:
                            raise ValueError("proton_react cannot disable its facade")
                    if scheduler_impl == "proton_phase":
                        planner_kwargs["phase_shadow_mode"] = bool(
                            scheduler_cfg.get("phase_shadow_mode", False),
                        )
                    planner_kwargs["reciprocal_interference_correction"] = bool(
                        scheduler_cfg.get("reciprocal_interference_correction", False),
                    )
                    planner_kwargs["backfill_latency_basis"] = str(
                        scheduler_cfg.get("backfill_latency_basis", "mean")
                    )
                    planner_kwargs["backfill_primary_tail_slo_factor"] = (
                        scheduler_cfg.get("backfill_primary_tail_slo_factor")
                    )
                    planner_kwargs["primary_tail_slo_observability_path"] = os.getenv(
                        "PROTON_PRIMARY_TAIL_SLO_TRACE_PATH", ""
                    )
                    dynamic_cfg = scheduler_cfg.get("dynamic_batching", {})
                    if not isinstance(dynamic_cfg, dict):
                        raise ValueError(
                            f"{config_key}.dynamic_batching must be an object"
                        )
                    unknown_dynamic = set(dynamic_cfg) - {
                        "cold_policy",
                        "warm_policy",
                    }
                    if unknown_dynamic:
                        raise ValueError(
                            f"{config_key}.dynamic_batching has unsupported keys: "
                            + ", ".join(sorted(str(key) for key in unknown_dynamic))
                        )
                    planner_kwargs["dynamic_batch_cold_policy"] = (
                        _require_strategy_choice(
                            f"{config_key}.dynamic_batching.cold_policy",
                            dynamic_cfg.get("cold_policy", "largest_safe"),
                            {
                                "constant_memory_linear_latency",
                                "fixed_n",
                                "largest_safe",
                            },
                            default="largest_safe",
                        )
                    )
                    planner_kwargs["dynamic_batch_warm_policy"] = (
                        _require_strategy_choice(
                            f"{config_key}.dynamic_batching.warm_policy",
                            dynamic_cfg.get("warm_policy", "throughput_optimal"),
                            {"fixed_n", "largest_safe", "throughput_optimal"},
                            default="throughput_optimal",
                        )
                    )
                self._global_planner = planner_cls(
                    campaign_scheduler=self._campaign_scheduler,
                    priority=priority,
                    placement=placement,
                    preinit=preinit,
                    backfill=backfill,
                    campaign=campaign,
                    eviction_strategy=eviction,
                    ram_eviction_strategy=ram_eviction,
                    **planner_kwargs,
                )
                if isinstance(preinit, NoPreInit):
                    self._campaign_scheduler._pre_init_disabled = True
                if scheduler_impl in {"proton_heft", "proton_react"}:
                    self._campaign_scheduler._downstream_projection_disabled = True
                _LOG.info(
                    "[planner] %s initialized (priority=%s, placement=%s, "
                    "backfill=%s, preinit=%s, campaign=%s, eviction=%s, "
                    "ram_eviction=%s, resource_p=%.3f, resource_z=%.3f)",
                    type(self._global_planner).__name__,
                    type(priority).__name__,
                    type(placement).__name__,
                    type(backfill).__name__,
                    type(preinit).__name__,
                    type(campaign).__name__,
                    type(eviction).__name__,
                    type(ram_eviction).__name__,
                    self._campaign_scheduler.resource_upper_percentile,
                    self._campaign_scheduler.resource_upper_z,
                )
                try:
                    ct = getattr(self._global_planner, "_constraint_tracker", None)
                    if ct is not None and hasattr(ct, "attach_supervisor"):
                        ct.attach_supervisor(supervisor)
                except Exception:
                    _LOG.warning(
                        "[planner] attach_supervisor on ConstraintTracker failed",
                        exc_info=True,
                    )

                try:
                    ct = getattr(self._global_planner, "_constraint_tracker", None)
                    if ct is not None and isinstance(gw_cfg, dict):
                        _simple_overrides = {
                            "activation_ttl_sweep_interval_sec": "_activation_ttl_sweep_interval_sec",
                            "worker_unreachable_ttl_sec": "_worker_unreachable_ttl_sec",
                            "default_saturation_ttl_sec": "_default_saturation_ttl_sec",
                            "correlation_window_sec": "_correlation_window_sec",
                            "gpu_unhealthy_ttl_sec_recoverable_default": "_gpu_unhealthy_ttl_sec_recoverable_default",
                            "numerical_retry_max": "_numerical_retry_max",
                            "numerical_retry_ttl_sec": "_numerical_retry_ttl_sec",
                        }
                        for cfg_key, attr_name in _simple_overrides.items():
                            if cfg_key in gw_cfg and hasattr(ct, attr_name):
                                val = gw_cfg[cfg_key]
                                try:
                                    if cfg_key == "numerical_retry_max":
                                        setattr(ct, attr_name, int(val))
                                    else:
                                        setattr(ct, attr_name, float(val))
                                except Exception:
                                    _LOG.warning(
                                        "[planner] tracker override failed for %s=%r",
                                        cfg_key,
                                        val,
                                        exc_info=True,
                                    )
                except Exception:
                    _LOG.warning(
                        "[planner] tracker Category B/C overrides failed",
                        exc_info=True,
                    )
            except Exception as exc:
                raise RuntimeError("GlobalPlanner initialization failed") from exc

        self._gateway._signal_service.register_drift_callback(self.on_profile_drift)

    @property
    def pipeline_dag(self) -> PipelineDAG:
        return self._pipeline_dag

    @property
    def placement_ctx(self) -> PlacementContextManager:
        return self._placement_ctx

    @property
    def campaign_scheduler(self) -> CampaignScheduler:
        return self._campaign_scheduler

    def export_runtime_state(self) -> dict[str, Any]:
        export = getattr(self._global_planner, "export_runtime_state", None)
        if not callable(export):
            return {}
        try:
            state = export()
        except Exception:
            _LOG.warning("[runtime-state] Planner export failed", exc_info=True)
            return {}
        return dict(state) if isinstance(state, dict) else {}

    def import_runtime_state(self, data: Any) -> int:
        restore = getattr(self._global_planner, "import_runtime_state", None)
        if not callable(restore):
            return 0
        try:
            restored = restore(data)
        except Exception:
            _LOG.warning("[runtime-state] Planner import failed", exc_info=True)
            return 0
        return restored if isinstance(restored, int) else 0


    def on_profile_drift(self, event: ProfileDriftEvent) -> None:
        """Receive a baseline drift notification from SignalService.

        Called synchronously whenever a signal measurement deviates significantly
        from its established baseline (≥ 30% for resource profiles, ≥ 50% for
        interference regime changes).

        This callback fires for **every** drift detection (including burst
        duplicates from fast-completing tasks).  The actual actions
        (PlacementContextManager.on_drift, CampaignScheduler.on_profile_drift) are
        idempotent, so repeated calls are safe.  Logging is at DEBUG level
        to avoid flooding — the authoritative drift log is emitted once by
        SignalService._emit_drift_event with deduplication.
        """
        if event.metric == "interference_evidence" and not bool(
            getattr(self._global_planner, "reciprocal_interference_correction", False)
        ):
            return
        _LOG.debug(
            "[planner-drift-ack] component=%s metric=%s observed=%.3f predicted=%.3f "
            "drift=%.1f%% n=%d config=%s gpu_id=%s campaign=%s",
            event.component,
            event.metric,
            event.observed,
            event.predicted,
            event.drift_ratio * 100,
            event.n_baseline,
            event.config_fingerprint or "__default__",
            event.gpu_id,
            event.campaign_id,
        )
        self._placement_ctx.on_drift(event)
        self._campaign_scheduler.on_profile_drift(
            event.component,
            event.metric,
            event=event,
        )
        if self._global_planner:
            try:
                on_event = getattr(self._global_planner, "on_profile_drift_event", None)
                if callable(on_event):
                    on_event(event)
                self._global_planner.on_profile_drift(event.component, event.metric)
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )

    def on_profile_drift_batch(
        self,
        by_component: dict[str, dict[str, ProfileDriftEvent]],
    ) -> None:
        """Plan v4.x B1 — batched drift cascade entry-point.

        ``signals.SignalService._flush_pending_drifts`` groups buffered
        events by component (same component → unified cascade) and calls
        this once per flush instead of per-event.  Cascade-invariant
        work (primary lookup, list_worker_snapshots, refresh_gpu_views,
        prune_stale_predicted, compute-waiter notify) runs **once** per
        flush regardless of how many events were buffered; per-component
        work iterates only the unique component set.

        Backward-compatible: callers without a batched view still hit
        ``on_profile_drift`` per event, which collapses through the
        same code path via a single-component dict.
        """
        if not by_component:
            return
        if not bool(
            getattr(self._global_planner, "reciprocal_interference_correction", False)
        ):
            by_component = {
                component: {
                    metric: event
                    for metric, event in events_by_metric.items()
                    if event.metric != "interference_evidence"
                }
                for component, events_by_metric in by_component.items()
            }
            by_component = {
                component: events
                for component, events in by_component.items()
                if events
            }
            if not by_component:
                return
        for events_by_metric in by_component.values():
            for ev in events_by_metric.values():
                _LOG.debug(
                    "[planner-drift-ack-batch] component=%s metric=%s "
                    "observed=%.3f predicted=%.3f drift=%.1f%%",
                    ev.component,
                    ev.metric,
                    ev.observed,
                    ev.predicted,
                    ev.drift_ratio * 100,
                )
                self._placement_ctx.on_drift(ev)

        self._campaign_scheduler.on_profile_drift_batch(by_component)

        if self._global_planner:
            try:
                for component, events_by_metric in by_component.items():
                    for ev in events_by_metric.values():
                        self._global_planner.on_profile_drift(
                            component,
                            ev.metric,
                        )
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )


    def on_task_dispatched_event(
        self,
        gpu_id: str,
        component: str,
        predicted_vram: float,
    ) -> None:
        """Notify scenario that a task was dispatched (VRAM reserved)."""
        self._placement_ctx.notify_gpu_change(
            gpu_id,
            predicted_vram,
            component,
            is_release=False,
        )

    def on_task_complete_event(
        self,
        gpu_id: str,
        component: str,
        predicted_vram: float,
    ) -> None:
        """Notify scenario that a task completed (VRAM released)."""
        self._placement_ctx.notify_gpu_change(
            gpu_id,
            predicted_vram,
            component,
            is_release=True,
        )


    _HORIZON_SEC = 120.0

    def _has_predicted_work(
        self,
        component: str,
        gpu_id: str,
        *,
        include_actual: bool = False,
    ) -> bool:
        """Check for near-horizon predicted or optionally actual work.

        Also checks pending pre-inits not yet added to the timeline.
        """
        import time as _time

        scenario = self._campaign_scheduler._timelines
        now = _time.time()
        tl = scenario.get(gpu_id)
        if tl and any(
            (e.is_predicted or include_actual)
            and not e.is_completed
            and e.start_time < now + self._HORIZON_SEC
            for e in tl.entries_for_component(component)
        ):
            return True
        for (_, pc, _), (
            _,
            tgpu,
            _,
        ) in self._campaign_scheduler._pending_pre_inits.items():
            if pc == component and tgpu == gpu_id:
                return True
        return False

    def select_for_eviction(
        self,
        candidates: list,
        gpu_id: str,
        needed_mb: int,
        freeable_fn: Any = None,
    ) -> list:
        """Protect predictive work and any optional phase residency leases."""
        lease_query = getattr(
            self._global_planner,
            "has_active_residency_lease",
            None,
        )
        runtime_cfg = getattr(self._gateway, "_runtime_config", {}) or {}
        protect_actual_work = callable(lease_query) or (
            str(runtime_cfg.get("scheduler_implementation", "")).strip().lower()
            in {"proton_heft", "proton_react"}
        )

        def _protected(st: Any) -> bool:
            if self._has_predicted_work(
                st.spec.component,
                gpu_id,
                include_actual=protect_actual_work,
            ):
                return True
            if not callable(lease_query):
                return False
            return bool(
                lease_query(
                    st.spec.component,
                    gpu_id,
                    getattr(st.spec, "name", ""),
                )
            )

        return self._select_for_idle_eviction_with_predicted_protection(
            candidates,
            needed_mb,
            is_protected=_protected,
            freeable_fn=freeable_fn,
            notify=lambda selected, get_weight: self._notify_eviction(
                selected,
                gpu_id,
                get_weight,
            ),
            allow_protected_fallback=not protect_actual_work,
            require_positive_freeable=protect_actual_work,
        )

    def select_for_host_ram_eviction(
        self,
        candidates: list,
        needed_mb: int,
        freeable_fn: Any = None,
    ) -> list:
        """Scenario-aware host-RAM idle eviction.

        Host RAM is cluster-wide, but warm worker protection still follows
        VRAM idle eviction semantics: first avoid workers with near-future
        predicted work on any assigned GPU, then fall back to all candidates
        only when needed to make progress.
        """
        supervisor = getattr(self._gateway, "supervisor", None)
        get_weight = supervisor.get_component_weight if supervisor else lambda _: 0
        freeable = freeable_fn or (lambda st: get_weight(st.spec.component))

        def _protected(st: Any) -> bool:
            component = getattr(getattr(st, "spec", None), "component", "")
            for gpu_id in getattr(st, "assigned_gpus", []) or []:
                if self._has_predicted_work(component, str(gpu_id)):
                    return True
            return False

        return self._select_for_idle_eviction_with_predicted_protection(
            candidates,
            needed_mb,
            freeable_fn=freeable,
            is_protected=_protected,
            notify=lambda selected, get_weight: self._notify_host_eviction(
                selected,
                get_weight,
            ),
        )

    def _select_for_idle_eviction_with_predicted_protection(
        self,
        candidates: list,
        needed_mb: int,
        *,
        is_protected: Any,
        notify: Any,
        freeable_fn: Any = None,
        allow_protected_fallback: bool = True,
        require_positive_freeable: bool = False,
    ) -> list:
        """Prefer unprotected idle workers, with an optional legacy fallback."""
        supervisor = getattr(self._gateway, "supervisor", None)
        get_weight = supervisor.get_component_weight if supervisor else lambda _: 0

        def _freeable(st: Any) -> int:
            if freeable_fn:
                return _safe_int(freeable_fn(st))
            return _safe_int(get_weight(st.spec.component))

        if require_positive_freeable:
            candidates = [st for st in candidates if _freeable(st) > 0]

        eviction_policy = (
            getattr(supervisor, "eviction_policy", None) if supervisor else None
        )
        if not eviction_policy:
            return candidates

        protected = [st for st in candidates if is_protected(st)]
        protected_ids = {id(st) for st in protected}
        unprotected = [st for st in candidates if id(st) not in protected_ids]

        if unprotected:
            result = eviction_policy.select_for_eviction(
                unprotected,
                needed_mb,
                freeable_fn=freeable_fn,
            )
            freed = sum(_freeable(st) for st in result)
            if freed >= needed_mb or not allow_protected_fallback:
                notify(result, get_weight)
                return result

        if not allow_protected_fallback:
            return []

        result = eviction_policy.select_for_eviction(
            candidates,
            needed_mb,
            freeable_fn=freeable_fn,
        )
        notify(result, get_weight)
        return result

    def _notify_host_eviction(self, evicted: list, get_weight) -> None:
        for st in evicted:
            for gpu_id in getattr(st, "assigned_gpus", []) or []:
                self._notify_eviction([st], str(gpu_id), get_weight)

    def _notify_eviction(self, evicted: list, gpu_id: str, get_weight) -> None:
        """Update PlacementContext: evicted workers release their weight VRAM."""
        for st in evicted:
            weight = get_weight(st.spec.component)
            if weight > 0:
                self._placement_ctx.notify_gpu_change(
                    gpu_id,
                    _safe_float(weight),
                    st.spec.component,
                    is_release=True,
                )

    def select_for_recovery(
        self,
        candidates: list,
        resource_tracker: Any,
    ) -> list:
        """Scenario-based recovery with VRAM capacity check.

        Step 1: Only recover workers whose component has predicted future
                entries in SchedulingScenario (scheduler actually needs them).
        Step 2: Among those, filter by VRAM capacity — recovering should
                not starve pending VRAM waiters.
        Step 3: If no worker is both needed and safe, recover nothing.
                Dispatch/pre-init demand paths remain responsible for
                re-initializing intentionally evicted idle workers.
        """
        supervisor = getattr(self._gateway, "supervisor", None)
        if supervisor is None:
            return []
        recovery_policy = getattr(supervisor, "recovery_policy", None)
        if not recovery_policy:
            return []

        get_weight = supervisor.get_component_weight

        needed = [
            st
            for st in candidates
            if st.assigned_gpus
            and self._has_predicted_work(st.spec.component, str(st.assigned_gpus[0]))
        ]

        gpus_with_waiters = (
            resource_tracker.gpus_with_pending_requests() if resource_tracker else set()
        )
        safe = []
        for st in needed:
            gpu_id = str(st.assigned_gpus[0])
            if gpu_id not in gpus_with_waiters:
                safe.append(st)
            else:
                weight = get_weight(st.spec.component)
                avail = resource_tracker.get_available_memory(gpu_id)
                max_pending = resource_tracker.get_max_pending_mb(gpu_id)
                if avail - weight >= max_pending:
                    safe.append(st)

        if not safe:
            return []

        cs = self._campaign_scheduler
        safe.sort(
            key=lambda st: (
                0 if cs and cs.is_needed_by_primary(st.spec.component) else 1,
                -st.last_used_at,
            )
        )

        result = recovery_policy.select_for_recovery(
            safe,
            resource_tracker,
            all_states=list(supervisor.states.values()),
        )
        for st in result:
            weight = get_weight(st.spec.component)
            gpu_id = str(st.assigned_gpus[0]) if st.assigned_gpus else ""
            if weight > 0 and gpu_id:
                self._placement_ctx.notify_gpu_change(
                    gpu_id,
                    _safe_float(weight),
                    st.spec.component,
                    is_release=False,
                )
        return result


    def build_compute_signal(
        self,
        intrinsic_result: SignalResult,
        component: str = "",
    ) -> WorkerComputeSignal:
        """Map the incoming SignalBundle into the Planner-facing WorkerComputeSignal."""
        bundle = intrinsic_result.bundle

        mem_prof = WorkerMemoryProfile(
            required_vram_mb=_safe_float(bundle.memory.active_estimate_mib),
            expected_peak_vram_mb=_safe_float(bundle.memory.active_upper_mib),
            safe_limit_vram_mb=_safe_float(bundle.memory.total_upper_bound_mib),
        )
        run_prof = WorkerRuntimeProfile(
            expected_duration_sec=_safe_float(bundle.runtime.estimate_sec),
            upper_bound_duration_sec=_safe_float(bundle.runtime.upper_sec),
        )

        signal_svc = self._gateway._signal_service
        comp = str(component or "").strip().lower()
        wclass = signal_svc.workload_classifier.classify(comp) if comp else "unknown"

        interference_info = (bundle.artifacts.guards or {}).get("interference", {})
        predicted_slowdown = _safe_float(interference_info.get("predicted_slowdown"))
        predicted_vram_overhead = _safe_float(
            interference_info.get("predicted_vram_overhead")
        )
        prediction_basis = str(interference_info.get("basis", "none"))
        prediction_confidence = str(interference_info.get("confidence", "low"))

        from ..signals.interference import COMPUTE_BOUND, MEMORY_BOUND

        intf_reg = signal_svc.interference_registry
        compute_slowdown = (
            intf_reg.get_pairwise_slowdown(comp, COMPUTE_BOUND) if comp else None
        )
        memory_slowdown = (
            intf_reg.get_pairwise_slowdown(comp, MEMORY_BOUND) if comp else None
        )

        solo_baseline = intf_reg.get_solo_baseline(comp) if comp else None

        con_prof = WorkerConcurrencyProfile(
            workload_class=wclass,
            predicted_slowdown=predicted_slowdown,
            prediction_basis=prediction_basis,
            prediction_confidence=prediction_confidence,
            solo_baseline_sec=solo_baseline,
            interference_compute_bound_slowdown=_safe_float(compute_slowdown),
            interference_memory_bound_slowdown=_safe_float(memory_slowdown),
            interference_vram_overhead_ratio=predicted_vram_overhead,
        )

        return WorkerComputeSignal(
            memory_profile=mem_prof,
            runtime_profile=run_prof,
            concurrency_profile=con_prof,
        )

    def prepare_for_planning(self) -> None:
        """Pre-plan housekeeping: refresh GPU state, prune stale entries.

        Called by the Core Loop before GlobalPlanner.plan().  Extracted
        from the legacy generate_plan() path so the same housekeeping
        runs regardless of which planning path is active.
        """
        flush_expired = getattr(
            self._campaign_scheduler, "_flush_expired_batches", None
        )
        if flush_expired:
            flush_expired()

        self._campaign_scheduler.check_pending_pre_inits()

        self._campaign_scheduler._timelines.prune_stale_predicted()

        self._connect_supervisor_refs()

        supervisor = getattr(self._gateway, "supervisor", None)
        resource_tracker = (
            getattr(supervisor, "resource_tracker", None) if supervisor else None
        )
        worker_snapshots = (
            supervisor.list_worker_snapshots(include_live_rss=False)
            if supervisor
            else []
        )
        self._placement_ctx.refresh_gpu_views(
            worker_snapshots, resource_tracker=resource_tracker
        )

        if resource_tracker and supervisor:
            idle_weights = {}
            idle_ram = 0.0
            try:
                snapshot_fn = getattr(
                    supervisor, "snapshot_idle_evictable_resources", None
                )
                if callable(snapshot_fn):
                    idle_snapshot = snapshot_fn(ignore_grace=True)
                    idle_ram = float(getattr(idle_snapshot, "idle_ram_mib", 0.0) or 0.0)
                    idle_weights = dict(
                        getattr(idle_snapshot, "idle_weight_by_gpu", {}) or {}
                    )
                else:
                    global_idle = supervisor._idle_evictable_states(ignore_grace=True)
                    for st in global_idle:
                        idle_ram += float(
                            supervisor._estimated_idle_host_ram_mib(st) or 0.0
                        )
                    for gpu_id in resource_tracker.total_vram:
                        gpu_idle = supervisor._idle_evictable_states(
                            gpu_id=gpu_id,
                            ignore_grace=True,
                        )
                        gpu_idle_weight = sum(
                            int(
                                supervisor.get_component_weight(st.spec.component)
                                or st.memory_reserved_mb
                                or st.actual_vram_mb
                                or 0
                            )
                            for st in gpu_idle
                        )
                        idle_weights[str(gpu_id)] = gpu_idle_weight
                for gpu_id in resource_tracker.total_vram:
                    idle_weights.setdefault(str(gpu_id), 0)
            except Exception:
                pass
            self._campaign_scheduler.sync_gpu_pool(
                dict(resource_tracker.total_vram),
                reserved_weights=dict(resource_tracker.reserved_vram),
                idle_weights=idle_weights,
                idle_ram=idle_ram,
            )
        elif resource_tracker:
            self._campaign_scheduler.sync_gpu_pool(dict(resource_tracker.total_vram))

