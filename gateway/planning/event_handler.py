"""EventHandler — reactive event routing (Plan).

Plan third layer.  Bridges
supervisor / dispatcher / drift events into scenario / campaign
accounting, and owns two plan-mandated periodic tasks:

    1. Safety-net GC (``safety_net_gc_interval_sec``, default 60 s).
    2. GP snapshot refresh (``snapshot_max_age_sec``, default 10 s).

Submodules (plan-mandated):

    - ``ScenarioUpdater``  : ``on_task_complete`` / ``on_worker_killed``
                             timeline updates + predicted-entry Layer-1 GC.
    - ``DriftResponder``   : ``on_profile_drift`` timeline invalidation +
                             backfill reassessment.
    - ``EvictionGrace``    : dynamic grace period (DG — init-latency based).
    - ``DriftEviction``    : drift-reactive backfill eviction (DR).

``EvictionGrace`` / ``DriftEviction`` are pluggable on/off per the plan's
Policy Integration table (DG / DR knobs).  Their default is OFF.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from .scenario import _checked_float, _checked_int

_LOG = logging.getLogger(__name__)


def _log_background_task_failure(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        _LOG.exception("[event-handler] background task failed")


class ScenarioUpdater:
    """Core (non-pluggable) — timeline mutations on task/worker events.

    Wraps the CampaignScheduler's ``on_task_complete`` / ``on_worker_killed``
    call sites so the 3-layer EventHandler owns the dispatch, not the
    ad-hoc bulk-call pattern of pre- code.
    """

    def __init__(self, campaign_scheduler: Any) -> None:
        self._cs = campaign_scheduler

    def on_task_complete(
        self,
        *,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
        from_eviction: bool = False,
    ) -> None:
        self._cs.on_task_complete(
            task_id=task_id,
            campaign_id=campaign_id,
            component=component,
            gpu_id=gpu_id,
            from_eviction=from_eviction,
        )

    def on_worker_killed(self, *, component: str, gpu_ids: list[str]) -> int:
        return self._cs.on_worker_killed(component, gpu_ids)


class DriftResponder:
    """Core — GP drift invalidation + backfill reassessment.

    Plan — on profile drift,
    invalidate predictions for the affected component and request re-plan
    for any active tasks on the drifting (component, gpu) pair.

    Plan Integration DR row: when ``DriftEviction`` is ON,
    profile-drift events additionally trigger eviction of the affected
    component's live backfills (stale prediction → evict rather than
    re-plan on drift).  ``drift_eviction`` is an optional injection
    so the Core path (invalidate + re_plan) stays on when the knob is
    OFF per plan default.
    """

    def __init__(
        self,
        global_planner: Any,
        campaign_scheduler: Any,
        drift_eviction: Any = None,
        *,
        supervisor: Any = None,
    ) -> None:
        self._gp = global_planner
        self._cs = campaign_scheduler
        self._drift_eviction = drift_eviction
        self._supervisor = supervisor
        self._supervisor_missing_warned = False

    def on_profile_drift(self, *, component: str, metric: str) -> int:
        """Invalidate scenario predictions for *component*, return count.

        Plan — for ``metric`` in
        ``{"vram", "interference_vram"}``, additionally upcap the
        admission-time activation reservations of every active timeline
        entry of this component so subsequent ``acquire_activation``
        decisions evaluate against the fresh GP upper bound rather than
        the stale cold-start fallback.  Latency-only drifts skip the
        upcap path (no VRAM implication).
        """
        try:
            count = self._cs._timelines.invalidate_component(
                component,
                metrics={metric},
            )
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "count = 0",
                exc_info=True,
            )
            count = 0
        invalidate = getattr(self._gp.priority, "invalidate", None)
        if callable(invalidate):
            try:
                invalidate()
            except Exception:
                _LOG.warning(
                    "[drift-responder] priority.invalidate failed",
                    exc_info=True,
                )
        if metric in ("vram", "interference_vram"):
            self._upcap_active_reservations(component)
        de = self._drift_eviction
        if de is not None and getattr(de, "should_evict_on_drift", None):
            try:
                if de.should_evict_on_drift(component):
                    evicted = self._evict_component_backfills(component)
                    if evicted > 0:
                        _LOG.info(
                            "[drift-eviction] drift=%s metric=%s evicted=%d "
                            "backfill entries (DR knob ON)",
                            component,
                            metric,
                            evicted,
                        )
            except Exception:
                _LOG.warning(
                    "[drift-responder] drift_eviction hook failed",
                    exc_info=True,
                )
        return count


    def _upcap_active_reservations(self, component: str) -> int:
        """Walk active timeline entries of *component*, schedule upcap of
        each entry's reservation to the latest GP mean prediction.

        Synchronous walk + intent collection runs on the drift-callback
        thread (asyncio main loop).  Apply (which awaits supervisor's
        async ``adjust_current_activation`` + idle eviction) is dispatched
        via ``asyncio.create_task`` so the drift cascade itself is not
        blocked on per-worker VRAM tracker mutations.

        Monotone-up only — entries whose current ``predicted_vram_mb`` is
        already at-or-above the new GP mean prediction are skipped (downward
        cap during execution would yank reservation away from a peak that
        may still be in flight; release is owned by ``on_task_end``).

        Returns: number of entries scheduled for upcap.  Zero if no
        active entry of *component* found, no GP upper bound available,
        no supervisor wired, or all entries already at-or-above the new
        bound.
        """
        sup = self._supervisor
        if sup is None:
            if not self._supervisor_missing_warned:
                _LOG.warning(
                    "[drift-vram-upcap] supervisor not wired into "
                    "DriftResponder — vram drift upcap is a no-op for "
                    "component=%s.  Wire via EventHandler(gateway, ...).",
                    component,
                )
                self._supervisor_missing_warned = True
            return 0
        sig_svc = getattr(self._cs, "_signal_service", None)
        if sig_svc is None:
            _LOG.warning(
                "[drift-vram-upcap] CampaignScheduler._signal_service "
                "missing — cannot query GP for component=%s",
                component,
            )
            return 0
        rp = getattr(sig_svc, "resource_profiles", None)
        if rp is None:
            _LOG.warning(
                "[drift-vram-upcap] ResourceProfileRegistry missing — "
                "cannot query VRAM upper bound for component=%s",
                component,
            )
            return 0
        timelines = getattr(self._cs, "_timelines", None)
        if timelines is None:
            return 0
        gateway_tasks = getattr(self._cs, "_gateway_tasks", None)

        intents: list[dict[str, Any]] = []
        for gpu_id in list(getattr(timelines, "gpu_ids", [])):
            tl = timelines.get(gpu_id)
            if tl is None:
                continue
            for entry in list(getattr(tl, "active_entries", [])):
                if entry.component != component:
                    continue
                if getattr(entry, "is_predicted", False):
                    continue
                if getattr(entry, "is_completed", False):
                    continue
                record = gateway_tasks.get(entry.task_id) if gateway_tasks else None
                addr = (
                    str(getattr(record, "dispatch_worker_addr", "") or "")
                    if record is not None
                    else ""
                )
                if not addr:
                    _LOG.debug(
                        "[drift-vram-upcap] task=%s component=%s gpu=%s: "
                        "no dispatch_worker_addr in TaskRecord — skipping",
                        entry.task_id,
                        component,
                        gpu_id,
                    )
                    continue
                config_fp = (
                    str(getattr(record, "config_fingerprint", "") or "")
                    or "__default__"
                )
                input_size = _checked_float(getattr(entry, "input_size", 0.0) or 0.0)
                new_pred = rp.predict_vram(
                    component=component,
                    config_fingerprint=config_fp,
                    input_size=input_size,
                    gpu_id=entry.gpu_id,
                )
                if new_pred is None or new_pred <= 0:
                    continue
                current = _checked_float(
                    getattr(entry, "predicted_vram_mb", 0.0) or 0.0
                )
                new_mb = _checked_int(new_pred)
                if new_mb <= _checked_int(current):
                    continue
                delta = new_mb - _checked_int(current)
                intents.append(
                    {
                        "addr": addr,
                        "delta_mb": delta,
                        "entry": entry,
                        "timeline": tl,
                        "new_mb": new_mb,
                        "current_mb": _checked_int(current),
                        "task_id": entry.task_id,
                        "gpu_id": entry.gpu_id,
                    }
                )

        if not intents:
            return 0

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            _LOG.error(
                "[drift-vram-upcap] no running asyncio loop — cannot "
                "schedule upcap for component=%s (%d intents lost): %s",
                component,
                len(intents),
                exc,
            )
            return 0
        task = loop.create_task(self._apply_upcaps_async(intents, sup))
        task.add_done_callback(_log_background_task_failure)
        return len(intents)

    async def _apply_upcaps_async(
        self,
        intents: list[dict[str, Any]],
        sup: Any,
    ) -> None:
        """Apply each upcap intent via supervisor.adjust_current_activation,
        update timeline entry's ``predicted_vram_mb``, then trigger
        idle-first eviction on every GPU that became oversubscribed.

        Plan — only idle workers are evicted
        from this path (CampaignAwareEvictionPolicy via supervisor's
        ``evict_idle_workers``).  Backfill eviction stays Planner-owned
        (Greedy via ``try_evict_backfills_for`` invoked through the normal
        Core Loop on next admission).
        """
        affected_gpus: set[str] = set()
        for it in intents:
            try:
                await sup.adjust_current_activation(
                    it["addr"],
                    delta_mb=it["delta_mb"],
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                _LOG.warning(
                    "[drift-vram-upcap] failed task=%s addr=%s delta=%d: %s",
                    it["task_id"],
                    it["addr"],
                    it["delta_mb"],
                    exc,
                )
                continue
            it["entry"].predicted_vram_mb = _checked_float(it["new_mb"])
            it["timeline"]._bump_state_version()
            affected_gpus.add(it["gpu_id"])
            _LOG.info(
                "[drift-vram-upcap] %s/%s gpu=%s: %d → %d MB (+%d)",
                it["entry"].component,
                it["task_id"],
                it["gpu_id"],
                it["current_mb"],
                it["new_mb"],
                it["delta_mb"],
            )

        for gpu_id in affected_gpus:
            try:
                tracker = sup.resource_tracker
                total = _checked_int(tracker.total_vram.get(gpu_id, 0))
                reserved = _checked_int(tracker.reserved_vram.get(gpu_id, 0))
                if total <= 0:
                    continue
                saturation_limit = _checked_int(total * 0.95)
                shortfall = reserved - saturation_limit
                if shortfall <= 0:
                    continue
                idle_freeable = _checked_int(sup.estimate_idle_freeable(gpu_id))
                if idle_freeable <= 0:
                    _LOG.warning(
                        "[drift-vram-upcap] gpu=%s oversubscribed "
                        "(reserved=%d > sat_limit=%d, shortfall=%d) but "
                        "no idle workers — residual surfaces via next "
                        "admission's Core Loop",
                        gpu_id,
                        reserved,
                        saturation_limit,
                        shortfall,
                    )
                    continue
                target = min(shortfall, idle_freeable)
                evicted = await sup.evict_idle_workers(gpu_id, target)
                _LOG.info(
                    "[drift-vram-upcap] gpu=%s post-upcap idle eviction: "
                    "shortfall=%d target=%d evicted=%d MB",
                    gpu_id,
                    shortfall,
                    target,
                    evicted,
                )
            except Exception as exc:
                _LOG.warning(
                    "[drift-vram-upcap] eviction trigger raised on gpu=%s: %s",
                    gpu_id,
                    exc,
                    exc_info=True,
                )

    def _evict_component_backfills(self, component: str) -> int:
        """DR knob — mark every live backfill entry of *component* as
        killed (Plan via scenario.mark_entry_killed so the
        eviction is visible on the ops dashboard as a red-X badge)."""
        evicted = 0
        timelines = getattr(self._cs, "_timelines", None)
        if timelines is None:
            return 0
        for gpu_id in list(getattr(timelines, "gpu_ids", [])):
            tl = timelines.get(gpu_id)
            if tl is None:
                continue
            for e in list(getattr(tl, "active_entries", [])):
                if (
                    getattr(e, "component", "") == component
                    and getattr(e, "is_backfill", False)
                    and getattr(e, "killed_at", None) is None
                    and getattr(e, "completed_at", None) is None
                    and not getattr(e, "is_predicted", False)
                ):
                    mark_fn = getattr(
                        self._cs,
                        "mark_for_eviction",
                        None,
                    )
                    if callable(mark_fn):
                        try:
                            mark_fn(e.task_id, reason="drift_eviction")
                        except Exception:
                            _LOG.warning(
                                "[drift-eviction] mark_for_eviction failed for task=%s",
                                e.task_id,
                                exc_info=True,
                            )
                    if hasattr(timelines, "mark_entry_killed"):
                        timelines.mark_entry_killed(e.task_id)
                    evicted += 1
        return evicted


class EvictionGrace:
    """Pluggable — dynamic grace period before backfill eviction.

    Plan table: ``EvictionGrace`` on/off knob
    DG.  When enabled, extends eviction latency by the GP-predicted
    init_latency upper-bound so newly arriving primary work does not
    race past a cold-starting worker.  Default OFF.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def grace_sec(self, component: str, gpu_id: str, global_planner: Any) -> float:
        if not self.enabled:
            return 0.0
        gp = getattr(global_planner, "campaign_scheduler", None)
        if gp is None:
            return 0.0
        return _checked_float(gp._get_init_latency(component, gpu_id) or 0.0)


class DriftEviction:
    """Pluggable — drift-reactive backfill eviction.

    Plan : ``DriftEviction`` on/off knob DR.
    When enabled, profile-drift events trigger eviction of backfills
    whose prediction is now stale.  Default OFF — drift flows through
    the normal ConstraintTracker → re_plan path unless explicitly on.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def should_evict_on_drift(self, component: str) -> bool:
        return self.enabled


class EventHandler:
    """Plan — Event Handler layer.

    Composes ``ScenarioUpdater`` / ``DriftResponder`` / ``EvictionGrace`` /
    ``DriftEviction`` and owns the two periodic tasks.  Lifecycle:

        handler = EventHandler(gateway, global_planner, campaign_scheduler)
        await handler.start()
        ...
        await handler.stop()
    """

    def __init__(
        self,
        gateway: Any,
        global_planner: Any,
        campaign_scheduler: Any,
        *,
        safety_net_gc_interval_sec: float = 60.0,
        stale_event_grace_period_sec: float = 30.0,
        snapshot_max_age_sec: float = 10.0,
        pre_init_fire_interval_sec: float = 1.0,
        eviction_grace: bool = False,
        drift_eviction: bool = False,
    ) -> None:
        self._gateway = gateway
        self._gp = global_planner
        self._cs = campaign_scheduler
        self.scenario_updater = ScenarioUpdater(campaign_scheduler)
        self.eviction_grace = EvictionGrace(enabled=eviction_grace)
        self.drift_eviction = DriftEviction(enabled=drift_eviction)
        self.drift_responder = DriftResponder(
            global_planner,
            campaign_scheduler,
            drift_eviction=self.drift_eviction,
            supervisor=getattr(gateway, "supervisor", None),
        )
        try:
            campaign_scheduler._eviction_grace_policy = self.eviction_grace
        except Exception:
            _LOG.warning(
                "[event-handler] failed to attach eviction_grace to "
                "CampaignScheduler — DG knob will be a no-op",
                exc_info=True,
            )

        self._safety_net_gc_interval_sec = _checked_float(safety_net_gc_interval_sec)
        self._stale_event_grace_period_sec = _checked_float(
            stale_event_grace_period_sec
        )
        self._snapshot_max_age_sec = _checked_float(snapshot_max_age_sec)
        self._pre_init_fire_interval_sec = _checked_float(pre_init_fire_interval_sec)

        self._running = False
        self._gc_task: asyncio.Task | None = None
        self._snapshot_task: asyncio.Task | None = None
        self._pre_init_fire_task: asyncio.Task | None = None


    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._gc_task = asyncio.create_task(self._safety_net_gc_loop())
        self._snapshot_task = asyncio.create_task(self._snapshot_refresh_loop())
        self._pre_init_fire_task = asyncio.create_task(self._pre_init_fire_loop())

    async def stop(self) -> None:
        self._running = False
        for task in (self._gc_task, self._snapshot_task, self._pre_init_fire_task):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._gc_task = None
        self._snapshot_task = None
        self._pre_init_fire_task = None


    def on_task_complete(
        self,
        *,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
    ) -> None:
        self.scenario_updater.on_task_complete(
            task_id=task_id,
            campaign_id=campaign_id,
            component=component,
            gpu_id=gpu_id,
        )

    def on_worker_killed(self, *, component: str, gpu_ids: list[str]) -> int:
        return self.scenario_updater.on_worker_killed(
            component=component,
            gpu_ids=gpu_ids,
        )

    def on_profile_drift(self, *, component: str, metric: str = "") -> int:
        return self.drift_responder.on_profile_drift(
            component=component,
            metric=metric,
        )


    async def _safety_net_gc_loop(self) -> None:
        """Plan tasks item 1 — 60 s cadence.

        Calls ``SchedulingScenario.gc_stale_entries_safety_net`` which is
        itself a no-op under a healthy system (Layer-1/2 paths clean up
        first).  A non-zero return logs WARNING — investigate.
        """
        grace = self._stale_event_grace_period_sec
        while self._running:
            try:
                await asyncio.sleep(self._safety_net_gc_interval_sec)
                tl = getattr(self._cs, "_timelines", None)
                if tl is not None and hasattr(tl, "gc_stale_entries_safety_net"):
                    tl.gc_stale_entries_safety_net(grace_period_sec=grace)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOG.warning(
                    "[event-handler] safety-net GC raised: %s (continuing)",
                    exc,
                    exc_info=True,
                )

    async def _snapshot_refresh_loop(self) -> None:
        """Plan tasks item 2 — GP snapshot refresh cadence.

        Calls ``SignalService.refresh_if_stale`` when available.  Stable
        GP posteriors yield a no-op; drifted metrics invalidate stale
        snapshots so the next ``solve()`` queries fresh values.
        """
        while self._running:
            try:
                await asyncio.sleep(self._snapshot_max_age_sec)
                sig = getattr(self._gp.campaign_scheduler, "_signal_service", None)
                if sig is not None and hasattr(sig, "refresh_if_stale"):
                    try:
                        sig.refresh_if_stale(max_age_sec=self._snapshot_max_age_sec)
                    except Exception:
                        _LOG.warning(
                            "[event-handler] signal.refresh_if_stale failed",
                            exc_info=True,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOG.warning(
                    "[event-handler] snapshot refresh raised: %s (continuing)",
                    exc,
                    exc_info=True,
                )

    async def _pre_init_fire_loop(self) -> None:
        """Periodic pre-init firing tick ().

        Plan — third periodic loop owned by EventHandler.
        ``CampaignScheduler.check_pending_pre_inits`` previously fired only
        on event hooks (``on_task_dispatched``, ``_on_scenario_changed``,
        ``prepare_for_planning``).  When no such event coincides with a
        scheduled pre-init's ``trigger_at`` (typical ~20s lookahead while
        upstream task is still running and no other state change occurs),
        firing was delayed until the next external event — observed as
        rfdiffusion → proteinmpnn pre-init scheduled at T+20s but firing
        at T+30s when rfdiffusion completion event finally arrived,
        cancelling the "warm worker ready" benefit.

        ``check_pending_pre_inits`` short-circuits when:
          - ``_pre_init_disabled`` (NoPreInit / proton-naive baseline)
          - ``_pending_pre_inits`` empty (steady state with no pending)
        so the per-tick cost is essentially a single dict-empty check
        when nothing is due.  1s cadence bounds late-fire to ≤ 1s.
        """
        while self._running:
            try:
                await asyncio.sleep(self._pre_init_fire_interval_sec)
                if hasattr(self._cs, "check_pending_pre_inits"):
                    self._cs.check_pending_pre_inits()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOG.warning(
                    "[event-handler] pre_init fire loop raised: %s (continuing)",
                    exc,
                    exc_info=True,
                )
