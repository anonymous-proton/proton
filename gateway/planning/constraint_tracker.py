"""ConstraintTracker — Planner-internal world-model state for Validator feedback.

Plan (v3.x).  Owned by GlobalPlanner via composition
(``planner._constraint_tracker``).  Receives ConstraintViolations via
``incorporate()`` and updates a monotone lattice of short-TTL avoidance state.

The Planner's ``solve()`` reads this state (via ``is_gpu_feasible_for_task``)
to filter infeasible GPUs before EFT evaluation.

This module preserves all existing GlobalPlanner fields (``_saturated_gpus``,
``_dead_workers``, ``_excluded_workers``, ``_vram_corrections``) as a
non-breaking extension.  + additions:

- ``_gpu_reset_required`` / ``_rma_qualifying_gpus`` / ``_gpu_reset_requested_at``
  for mandatory_reset Xid ( P0 safety +  recovery channel).
- ``_unhealthy_gpus`` / ``_unreachable_workers`` / ``_activation_excluded``
  TTL-bearing dicts (v3.x violation types).
- ``_partial_progress`` for ``stage_failed`` progress_info.

 section refs preserved in comments (Bach /, Krause-Golovin 
for submodularity — relevant to D6.7 ΔEST monotonicity, not ConstraintTracker
itself).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, cast

from .contracts import ConstraintViolation, ViolationType

_LOG = logging.getLogger(__name__)




class ConstraintView:
    """Scoped handle wrapping a ``ConstraintTracker``.

    Per plan  ( rename): "not a snapshot" — a
    write-through proxy whose ``incorporate`` updates the underlying
    tracker and applies an O(1) diff to the view's local copy.  Inner
    loop fixed-point accounting uses the same True/False return semantics
    as ``ConstraintTracker.incorporate`` so callers can test progress
    without touching the tracker.

    Unknown violation types trigger a full refresh (O(|lattice|)) with
    WARNING log — a guardrail against silent drift when new enum members
    are added.
    """

    def __init__(self, tracker: "ConstraintTracker") -> None:
        self._tracker = tracker
        self.vram_corrections: dict[str, float] = dict(tracker._vram_corrections)
        self.saturated_gpus: dict[str, float] = dict(tracker._saturated_gpus)
        self.unhealthy_gpus: dict[str, float] = dict(tracker._unhealthy_gpus)
        self.gpu_reset_required: set[str] = set(tracker._gpu_reset_required)
        self.rma_qualifying_gpus: set[str] = set(tracker._rma_qualifying_gpus)
        self.dead_workers: set[str] = set(tracker._dead_workers)
        self.excluded_workers: set[str] = set(tracker._excluded_workers)
        self.unreachable_workers: dict[str, float] = dict(tracker._unreachable_workers)
        self.activation_excluded: dict[tuple[str, str], float] = dict(
            tracker._activation_excluded
        )
        self.partial_progress: dict[str, dict[str, Any]] = dict(
            tracker._partial_progress
        )
        self.taken_at: float = time.time()

    def incorporate(self, violation: ConstraintViolation) -> bool:
        """Apply *violation* to the underlying tracker and mirror the
        change into this view via O(1) diff.  Returns the tracker's
        monotone-lattice progress signal (True/False)."""
        result = self._tracker.incorporate(violation)
        if result:
            self._apply_diff(violation)
        return result

    def _apply_diff(self, v: ConstraintViolation) -> None:
        """O(1) per-field diff.  Must be updated when a new violation
        type is added; unknown types fall through to the full-refresh
        branch with a WARNING log (plan  P1)."""
        t = self._tracker
        vt = v.violation_type
        if vt == ViolationType.VRAM_INSUFFICIENT:
            self.vram_corrections[v.gpu_id] = t._vram_corrections.get(v.gpu_id, 0.0)
        elif vt == ViolationType.COMPUTE_SATURATED:
            self.saturated_gpus[v.gpu_id] = t._saturated_gpus.get(v.gpu_id, 0.0)
        elif vt in (ViolationType.WORKER_DEAD, ViolationType.HOST_RESOURCE_EXHAUSTED):
            if v.worker_name:
                self.dead_workers.add(v.worker_name)
        elif vt == ViolationType.MEMORY_ADMISSION_REJECTED:
            if v.worker_name:
                self.excluded_workers.add(v.worker_name)
        elif vt == ViolationType.ACTIVATION_FAILED:
            if v.failed_plan is not None:
                key = (v.failed_plan.component, v.gpu_id)
                self.activation_excluded[key] = t._activation_excluded.get(key, 0.0)
        elif vt == ViolationType.WORKER_UNREACHABLE:
            if v.worker_name:
                self.unreachable_workers[v.worker_name] = t._unreachable_workers.get(
                    v.worker_name,
                    0.0,
                )
        elif vt == ViolationType.GPU_UNHEALTHY:
            self.unhealthy_gpus[v.gpu_id] = t._unhealthy_gpus.get(v.gpu_id, 0.0)
            if v.gpu_id in t._gpu_reset_required:
                self.gpu_reset_required.add(v.gpu_id)
            if v.gpu_id in t._rma_qualifying_gpus:
                self.rma_qualifying_gpus.add(v.gpu_id)
        elif vt == ViolationType.PREEMPTED:
            self.saturated_gpus[v.gpu_id] = t._saturated_gpus.get(v.gpu_id, 0.0)
        elif vt == ViolationType.STAGE_FAILED:
            if v.failed_plan is not None:
                tid = v.failed_plan.task_id
                if tid in t._partial_progress:
                    self.partial_progress[tid] = t._partial_progress[tid]
        elif vt == ViolationType.WORKER_NOT_READY:
            pass
        elif vt == ViolationType.NUMERICAL_FAILURE:
            pass
        elif vt == ViolationType.HOST_RAM_SATURATED:
            pass
        elif vt == ViolationType.WORKER_QUEUE_SATURATED:
            pass
        elif vt == ViolationType.SCHEDULER_INTERNAL_ERROR:
            self._full_refresh_from_tracker()
        else:
            _LOG.warning(
                "[constraint-view] _apply_diff: unhandled violation_type=%s; "
                "falling back to full snapshot refresh.  Add explicit branch.",
                vt,
            )
            self._full_refresh_from_tracker()

    def _full_refresh_from_tracker(self) -> None:
        """Full O(|lattice|) re-copy.  Invoked on unknown violation types
        as a guardrail; the happy path never reaches here."""
        t = self._tracker
        self.vram_corrections = dict(t._vram_corrections)
        self.saturated_gpus = dict(t._saturated_gpus)
        self.unhealthy_gpus = dict(t._unhealthy_gpus)
        self.gpu_reset_required = set(t._gpu_reset_required)
        self.rma_qualifying_gpus = set(t._rma_qualifying_gpus)
        self.dead_workers = set(t._dead_workers)
        self.excluded_workers = set(t._excluded_workers)
        self.unreachable_workers = dict(t._unreachable_workers)
        self.activation_excluded = dict(t._activation_excluded)
        self.partial_progress = dict(t._partial_progress)



XID_TTL_MAP_PCIE: dict[int, tuple[float, bool]] = {
    13: (5.0, False),
    43: (5.0, False),
    45: (5.0, False),
    31: (5.0, False),
    63: (60.0, False),
    64: (300.0, True),
    79: (float("inf"), False),
    94: (60.0, False),
    95: (60.0, True),
    119: (30.0, True),
    120: (30.0, True),
}




class ConstraintTracker:
    """Planner's short-TTL avoidance state + Validator feedback absorber.

    Composed into ``GlobalPlanner`` (``planner._constraint_tracker``).  All
    state is owned by the Planner; Validator never writes directly.
    """

    def __init__(
        self,
        *,
        gpu_unhealthy_ttl_sec_recoverable_default: float = 300.0,
        worker_unreachable_ttl_sec: float = 30.0,
        default_saturation_ttl_sec: float = 10.0,
        correlation_window_sec: float = 5.0,
        activation_ttl_sweep_interval_sec: float = 2.0,
        topology: str = "pcie",
    ) -> None:
        self._saturated_gpus: dict[str, float] = {}
        self._dead_workers: set[str] = set()
        self._excluded_workers: set[str] = set()
        self._vram_corrections: dict[str, float] = {}

        self._unhealthy_gpus: dict[str, float] = {}
        self._unreachable_workers: dict[str, float] = {}
        self._activation_excluded: dict[tuple[str, str], float] = {}
        self._partial_progress: dict[str, dict[str, Any]] = {}

        self._gpu_reset_required: set[str] = set()
        self._gpu_reset_requested_at: dict[str, float] = {}
        self._rma_qualifying_gpus: set[str] = set()

        self._recent_violations_by_corr: dict[str, ConstraintViolation] = {}
        self._fallout_drop_count: dict[str, int] = defaultdict(int)

        self._worker_transient_failures: dict[str, int] = defaultdict(int)
        self._transient_failure_threshold: int = 3

        self._worker_cold_on_next_plan: set[str] = set()

        self._task_numerical_retry_budget: dict[str, tuple[int, float]] = {}
        self._numerical_retry_max: int = 3
        self._numerical_retry_ttl_sec: float = 120.0

        self._primary_dispatched_components: set[tuple[str, str]] = set()

        self._gpu_unhealthy_ttl_sec_recoverable_default = (
            gpu_unhealthy_ttl_sec_recoverable_default
        )
        self._worker_unreachable_ttl_sec = worker_unreachable_ttl_sec
        self._default_saturation_ttl_sec = default_saturation_ttl_sec
        self._correlation_window_sec = correlation_window_sec
        self._topology = topology

        self._activation_ttl_sweep_interval_sec: float = float(
            activation_ttl_sweep_interval_sec,
        )
        self._sweeper_task: Any | None = None
        self._running: bool = False
        self._wake_hook: Any | None = None
        self._sweeper_heartbeat_at: float = 0.0
        self._heartbeat_task: Any | None = None
        self._scenario_ref: Any | None = None
        self._supervisor_ref: Any | None = None

    def attach_scenario(self, scenario: Any) -> None:
        """Register the ``SchedulingScenario`` (or compatible) used for
        GP-derived TTL calculations on ``compute_saturated`` violations."""
        self._scenario_ref = scenario

    def attach_supervisor(self, supervisor: Any) -> None:
        """Plan fix — register the supervisor reference so
        the dead-worker recovery sweeper can scan ``supervisor.states``
        and auto-clear ``_dead_workers`` entries whose worker has been
        restarted (``ready=True``) by supervisor autoscale/crash-restart.

        Plan (fix): worker_dead
        is now a supervisor-health-backed transient state, not a
        permanent one — recovery symmetry is mandatory.
        """
        self._supervisor_ref = supervisor

    @staticmethod
    def _resolve_supervisor_worker_state(
        supervisor: Any,
        states: Any,
        worker_name: str,
    ) -> Any:
        resolver = getattr(supervisor, "_resolve_worker", None)
        if callable(resolver):
            return resolver(worker_name)
        return states.get(worker_name)


    def record_transient_grpc_failure(self, worker_name: str) -> int:
        """Bump consecutive-transient-failure counter for ``worker_name``.

        Returns the new count.  Validator calls this on UNAVAILABLE /
        DEADLINE_EXCEEDED when the supervisor's health-check reports
        ``ready=True`` (i.e. the supervisor still believes the worker
        is alive — classify as transient first).
        """
        if not worker_name:
            return 0
        self._worker_transient_failures[worker_name] += 1
        return self._worker_transient_failures[worker_name]

    def reset_transient_grpc_failures(self, worker_name: str) -> None:
        """Clear the counter on dispatch success or worker recovery."""
        if not worker_name:
            return
        self._worker_transient_failures.pop(worker_name, None)

    def get_transient_grpc_failures(self, worker_name: str) -> int:
        """Read current count (does not mutate)."""
        if not worker_name:
            return 0
        return self._worker_transient_failures.get(worker_name, 0)

    @property
    def transient_failure_threshold(self) -> int:
        """Plan escalation threshold (default 3)."""
        return self._transient_failure_threshold

    def worker_needs_cold_on_next_plan(self, worker_name: str) -> bool:
        """Plan fix (f) — caller (Planner `_resolve_worker`)
        queries whether the given worker should be treated as cold on
        the next plan attempt regardless of supervisor.states ready
        flag.  Set by ``incorporate(worker_not_ready)``, cleared on
        dispatch success via ``clear_worker_cold_marker``."""
        return bool(worker_name) and worker_name in self._worker_cold_on_next_plan

    def clear_worker_cold_marker(self, worker_name: str) -> None:
        """Clear the worker_not_ready cold marker — called on dispatch
        success (``promote_predicted_to_active`` path) + recovery sweep."""
        if worker_name:
            self._worker_cold_on_next_plan.discard(worker_name)

    def clear_worker_unreachable(self, worker_name: str) -> bool:
        """Clear a stale worker_unreachable TTL after readiness/dispatch success."""
        if not worker_name:
            return False
        return self._unreachable_workers.pop(worker_name, None) is not None

    def clear_activation_exclusion(self, component: str, gpu_id: str) -> bool:
        """Clear a stale activation_excluded TTL after the worker becomes ready."""
        if not component or gpu_id is None:
            return False
        key = (str(component), str(gpu_id))
        return self._activation_excluded.pop(key, None) is not None

    def record_numerical_failure(self, task_id: str) -> bool:
        """Plan fix (g) — bump numerical_failure retry
        budget for ``task_id``.  Returns True while the task is still
        retryable (attempts < max), False when budget exhausted."""
        if not task_id:
            return True
        now = time.time()
        attempts, expiry = self._task_numerical_retry_budget.get(task_id, (0, 0.0))
        if expiry < now:
            attempts = 0
        attempts += 1
        self._task_numerical_retry_budget[task_id] = (
            attempts,
            now + self._numerical_retry_ttl_sec,
        )
        return attempts < self._numerical_retry_max

    def mark_primary_dispatched(
        self,
        component: str,
        gpu_id: str | None = None,
        config_fingerprint: str = "",
    ) -> None:
        """Record that a primary task for ``(component, config_fingerprint)``
        has been observed.

        Called from dispatch-time only:

        - ``CampaignScheduler.on_task_dispatched`` when
          ``_actual_was_primary=True``.  Plan-time placement is deliberately
          insufficient: relaxing PBBC before the primary has become an actual
          timeline entry lets same-cycle backfills slip onto the primary's
          still-cold worker.

        ``gpu_id`` is accepted for backward-compat and ignored.
        ``config_fingerprint`` defaults to ``""`` for legacy/test paths,
        but the planner should pass it explicitly so PBBC anchoring stays
        aligned with the per-fingerprint self-interference posterior.
        """
        if not component:
            return
        _ = gpu_id
        fp = str(config_fingerprint or "").strip()
        self._primary_dispatched_components.add((str(component), fp))

    def is_primary_dispatched(
        self,
        component: str,
        gpu_id: str | None = None,
        config_fingerprint: str = "",
    ) -> bool:
        """True iff a primary task for ``(component, config_fingerprint)``
        has reached actual dispatch.

        This is a legacy/diagnostic marker.  Current PBBC release semantics
        use completed solo and completed 2-way self observations instead.

        ``gpu_id`` is accepted for backward-compat and ignored.
        """
        if not component:
            return False
        _ = gpu_id
        fp = str(config_fingerprint or "").strip()
        return (str(component), fp) in self._primary_dispatched_components

    def task_numerical_exhausted(self, task_id: str) -> bool:
        """Returns True when the task has exhausted its numerical
        retry budget (Planner should skip it in subsequent plans)."""
        if not task_id:
            return False
        now = time.time()
        attempts, expiry = self._task_numerical_retry_budget.get(task_id, (0, 0.0))
        if expiry < now:
            self._task_numerical_retry_budget.pop(task_id, None)
            return False
        return attempts >= self._numerical_retry_max

    def active_correlations_count(self) -> int:
        """Plan — number of live
        correlation-window entries.  Used by
        ``GatewayHTTPService._compute_max_refinements`` to size the
        inner-loop lattice bound per Plan spec:

            max = gpus×3 + workers×2 + components×gpus
                  + min(50, active_correlations_count() + 10)

        Returns the count of root-cause correlations whose timestamp
        falls within ``_correlation_window_sec`` of now.  Expired
        entries are excluded.
        """
        now = time.time()
        window = self._correlation_window_sec
        count = 0
        for _corr_id, viol in self._recent_violations_by_corr.items():
            ts = getattr(viol, "timestamp", None)
            if ts is None:
                count += 1
                continue
            if (now - float(ts)) <= window:
                count += 1
        return count

    def is_component_exhausted(
        self,
        component: str,
        candidate_gpu_ids: list[str],
    ) -> bool:
        """Return true only when every GPU needs operator intervention.

        TTL-backed activation, saturation, unhealthy, and unreachable states
        remain transient so their recovery wake can re-enqueue the task.
        """
        if not candidate_gpu_ids:
            return True
        for gpu_id in candidate_gpu_ids:
            reset_required = gpu_id in self._gpu_reset_required
            rma = gpu_id in self._rma_qualifying_gpus
            if not (reset_required or rma):
                return False
        return True

    def attach_init_latency_accessor(self, accessor: Any) -> None:
        """Register a ``fn(component, gpu_id) -> (mu_init, sigma_init)``
        so ``ACTIVATION_FAILED`` can compute ``μ_init + 2σ_init`` TTLs
        without importing SignalService directly.
        """
        self._init_latency_accessor = accessor

    def clear_saturation(self, gpu_id: str) -> bool:
        """Clear transient ``compute_saturated`` state for a GPU.

        ``compute_saturated`` TTLs are prediction-backed guards.  A real task
        completion on the GPU is stronger evidence than the predicted end
        time, so completion should immediately make the GPU eligible for the
        next planning pass instead of waiting for the TTL to expire.
        """
        gid = str(gpu_id or "").strip()
        if not gid:
            return False
        return self._saturated_gpus.pop(gid, None) is not None

    def _compute_activation_exclusion_ttl(
        self,
        component: str,
        gpu_id: str,
    ) -> float:
        """Plan — ``μ_init + 2σ_init``.
        Falls back to ``gpu_unhealthy_ttl_sec_recoverable_default``
        when the GP accessor is not wired or returns zero-variance.
        """
        accessor = getattr(self, "_init_latency_accessor", None)
        if callable(accessor):
            try:
                mu, sigma = cast(tuple[Any, Any], accessor(component, gpu_id))
                mu = float(mu) if mu is not None else 0.0
                sigma = float(sigma) if sigma is not None else 0.0
                if mu > 0:
                    return mu + 2.0 * max(0.0, sigma)
            except Exception:
                _LOG.warning(
                    "[constraint-tracker] init latency accessor failed",
                    exc_info=True,
                )
        return self._gpu_unhealthy_ttl_sec_recoverable_default

    def _compute_saturation_ttl(
        self,
        gpu_id: str,
        violation: ConstraintViolation | None = None,
    ) -> float:
        """Derive saturation TTL from the earliest predicted_end_time on
        the GPU's timeline (plan).
        Falls back to ``default_saturation_ttl_sec`` when no timeline or
        active entries are available."""
        scenario = self._scenario_ref
        now = time.time()
        if (
            scenario is None
            and violation is not None
            and violation.failed_plan is not None
        ):
            scenario = getattr(violation.failed_plan, "_scenario_ref", None)
        if scenario is None:
            return self._default_saturation_ttl_sec
        try:
            tl = scenario.get(gpu_id)
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "tl = None",
                exc_info=True,
            )
            tl = None
        if tl is None:
            return self._default_saturation_ttl_sec
        if hasattr(tl, "planned_occupancy_entries"):
            active = list(tl.planned_occupancy_entries(now))
        else:
            active = [
                e
                for e in getattr(tl, "active_entries", [])
                if getattr(e, "predicted_end_time", 0.0) > now
                and not getattr(e, "is_completed", False)
                and not getattr(e, "is_evict_masked", False)
            ]
        if not active:
            return self._default_saturation_ttl_sec
        earliest_end = min(e.predicted_end_time for e in active)
        return max(0.0, earliest_end - now)


    def snapshot(self) -> ConstraintView:
        """Return a ``ConstraintView`` — scoped write-through proxy.

        The Core Loop's inner fixed-point detection uses the returned
        view: call ``view.incorporate(violation)`` inside the loop; a
        ``False`` return means "no lattice progress", exit the inner
        loop and escape to the outer retry.
        """
        return ConstraintView(self)


    def set_wake_hook(self, hook: Any) -> None:
        """Register a ``fn(trigger: str) -> None`` called when TTL sweep
        reaps entries.  The SchedulingSupervisor (or the Core Loop) uses
        this to trigger an immediate wake instead of waiting for the 10 s
        periodic tick."""
        self._wake_hook = hook

    async def start_sweeper(self) -> None:
        """Launch the background TTL sweeper + heartbeat monitor.

        Plan PR5 — two tasks: (1) the supervised sweeper loop
        and (2) the heartbeat watchdog that escalates a CRITICAL alert
        if the sweeper heartbeat goes stale (>20 s).  Correctness is
        still guaranteed by lazy expiry in
        ``is_gpu_feasible_for_task`` when the sweeper stalls.
        """
        if self._running:
            return
        self._running = True
        self._sweeper_heartbeat_at = time.time()
        try:
            import asyncio

            self._sweeper_task = asyncio.create_task(self._ttl_sweeper_wrapped())
            self._heartbeat_task = asyncio.create_task(self._sweeper_heartbeat())
        except RuntimeError:
            self._running = False

    async def stop_sweeper(self) -> None:
        import asyncio

        self._running = False
        for task in (self._sweeper_task, self._heartbeat_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._sweeper_task = None
        self._heartbeat_task = None

    async def _sweeper_heartbeat(self) -> None:
        """Monitor sweeper liveness (plan  PR5).

        Every 10 s, compare ``_sweeper_heartbeat_at`` to wall-clock.
        >20 s stale ⇒ CRITICAL log.  Correctness is still guaranteed
        by lazy expiry inside ``is_gpu_feasible_for_task`` — only
        wake-trigger latency degrades when the sweeper hangs.
        """
        import asyncio

        while self._running:
            await asyncio.sleep(10.0)
            stale_sec = time.time() - self._sweeper_heartbeat_at
            if stale_sec > 20.0:
                _LOG.critical(
                    "[constraint-tracker] TTL sweeper heartbeat stale for %.0fs — "
                    "lazy expiry still provides correctness, but activation TTL "
                    "wake latency degraded. Investigate sweeper liveness.",
                    stale_sec,
                )

    async def _ttl_sweeper_wrapped(self) -> None:
        """Supervised wrapper — crash-restart with 1 s backoff."""
        import asyncio

        while self._running:
            try:
                await self._ttl_sweeper()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _LOG.error(
                    "[constraint-tracker] TTL sweeper crashed: %s. Restarting in 1 s.",
                    e,
                )
                await asyncio.sleep(1.0)

    async def _ttl_sweeper(self) -> None:
        """Single sweep pass.  ``_ttl_sweeper_wrapped`` drives the loop +
        crash-restart; this method performs exactly one scan and returns.
        Running the loop here too (as earlier drafts did) would bury any
        unhandled exception inside an inner retry and bypass the outer
        supervisor's restart behavior."""
        import asyncio

        await asyncio.sleep(self._activation_ttl_sweep_interval_sec)
        self._sweeper_heartbeat_at = time.time()
        now = self._sweeper_heartbeat_at
        expired_keys = [
            key for key, expiry in self._activation_excluded.items() if expiry <= now
        ]
        if expired_keys:
            for key in expired_keys:
                self._activation_excluded.pop(key, None)
            _LOG.debug(
                "[constraint-tracker] TTL sweeper reaped %d activation exclusions",
                len(expired_keys),
            )
            hook = self._wake_hook
            if hook is not None:
                try:
                    hook("activation_ttl_expired")
                except Exception:
                    _LOG.warning(
                        "[constraint-tracker] wake_hook raised — ignoring",
                        exc_info=True,
                    )

        self._sweep_dead_worker_recovery()

        self._sweep_worker_cold_marker_recovery()

    def _sweep_worker_cold_marker_recovery(self) -> None:
        """Plan fix (F-xx) — scan ``supervisor.states`` for
        workers still carrying the fix (f) ``worker_not_ready``
        cold marker but actually ready in supervisor state.  Symmetric
        extension of ``_sweep_dead_worker_recovery`` — covers the case
        where a worker entered ``_worker_cold_on_next_plan`` via a
        transient ``worker_not_ready`` violation but never entered
        ``_dead_workers`` (so the dead-worker sweep could not clear the
        marker).

        No-op when no supervisor ref is attached (unit-test stubs) or
        when the marker set is empty.  Fires no wake trigger — marker
        staleness only affects the *next* plan's ``_resolve_worker``
        decision and does not unblock any pending task.
        """
        sup = self._supervisor_ref
        if sup is None or not self._worker_cold_on_next_plan:
            return
        states = getattr(sup, "states", None)
        if not states:
            return
        cleared: list = []
        for worker_name in list(self._worker_cold_on_next_plan):
            st = self._resolve_supervisor_worker_state(sup, states, worker_name)
            if st is None:
                continue
            if bool(getattr(st, "ready", False)) and getattr(st, "addr", ""):
                self._worker_cold_on_next_plan.discard(worker_name)
                cleared.append(worker_name)
        if cleared:
            _LOG.info(
                "[constraint-tracker] fix cleared cold marker for %d "
                "worker(s) now ready: %s",
                len(cleared),
                cleared,
            )

    def _sweep_dead_worker_recovery(self) -> None:
        """Plan fix — scan ``supervisor.states`` for
        workers in ``_dead_workers`` that have been restarted
        (``ready=True`` with an ``addr``) and clear them.  This provides
        recovery symmetry: Planner state tracks supervisor lifecycle
        instead of holding a stale " dead" classification after
        supervisor autoscale/crash-restart has revived the worker.

        No-op when no supervisor ref is attached (unit-test stubs) or
        when ``_dead_workers`` is empty.  Fires wake trigger
        ``gpu_health_change`` when at least one worker recovers so the
        Scheduling Supervisor re-evaluates pending tasks immediately.
        """
        sup = self._supervisor_ref
        if sup is None or not self._dead_workers:
            return
        states = getattr(sup, "states", None)
        if not states:
            return
        recovered: list = []
        for worker_name in list(self._dead_workers):
            st = self._resolve_supervisor_worker_state(sup, states, worker_name)
            if st is None:
                continue
            if bool(getattr(st, "ready", False)) and getattr(st, "addr", ""):
                self._dead_workers.discard(worker_name)
                self._worker_transient_failures.pop(worker_name, None)
                self._worker_cold_on_next_plan.discard(worker_name)
                recovered.append(worker_name)
        if recovered:
            _LOG.info(
                "[constraint-tracker] fix recovered %d worker(s) "
                "after supervisor restart: %s",
                len(recovered),
                recovered,
            )
            hook = self._wake_hook
            if hook is not None:
                try:
                    hook("gpu_health_change")
                except Exception:
                    _LOG.warning(
                        "[constraint-tracker] wake_hook raised — ignoring",
                        exc_info=True,
                    )


    def _topology_for_gpu(self, gpu_id: str) -> str:
        """Plan TTL topology-aware resolution — per-GPU topology
        ("pcie" | "nvswitch" | "unknown").  Operators may override via
        ``attach_topology_map({gpu_id: "nvswitch", ...})``; falls back
        to the tracker-wide ``_topology`` default.
        """
        topo_map = getattr(self, "_topology_map", None) or {}
        override = topo_map.get(str(gpu_id))
        if override:
            return str(override)
        return self._topology

    def _add_vram_correction(self, gpu_id: str, correction: float) -> bool:
        """Plan _constraint — accumulate a VRAM correction.

        Implements the plan's `_add_vram_correction(gpu_id, value)` API.
        Monotone accumulation is intentional (plan 
        — "lattice strictly monotone "): repeated ``vram_insufficient``
        violations for the same GPU grow the phantom correction so future
        plans account for the cumulative discrepancy until the
        scenario's scheduled entries clear.  Returns True when the
        correction changes the stored value, False otherwise
        (plan table: " correction True,    False").
        """
        if correction is None or correction <= 0:
            return False
        prev = self._vram_corrections.get(gpu_id, 0.0)
        new_val = prev + float(correction)
        if new_val != prev:
            self._vram_corrections[gpu_id] = new_val
            return True
        return False

    def attach_topology_map(self, mapping: dict[str, str]) -> None:
        """Register a per-GPU topology override map."""
        self._topology_map = dict(mapping or {})

    def _resolve_xid_ttl(
        self,
        xid: int,
        gpu_id: str = "",
    ) -> tuple[float, bool]:
        """Resolve (ttl_sec, mandatory_reset) for a given Xid code and GPU.

         topology multiplier: NVSwitch/DGX ×3 vs PCIe baseline.  Based
        on empirical Xid propagation measurement (not DGX H100 doc citation).
        Heterogeneous clusters (mixed PCIe + NVSwitch) use
        ``_topology_for_gpu(gpu_id)`` so each GPU gets its correct
        multiplier.
        """
        base_ttl, mandatory = XID_TTL_MAP_PCIE.get(
            xid, (self._gpu_unhealthy_ttl_sec_recoverable_default, False)
        )
        if base_ttl == float("inf"):
            return base_ttl, mandatory
        topology = self._topology_for_gpu(gpu_id) if gpu_id else self._topology
        mult = 3.0 if topology == "nvswitch" else 1.0
        return base_ttl * mult, mandatory


    def is_gpu_feasible_for_task(
        self,
        gpu_id: str,
        task_info: Any,
    ) -> bool:
        """5-step feasibility check ( Issue 4/5 +  mandatory_reset).

        Called by ``PlacementStrategy.evaluate_gpu`` *before* EFT calculation.
        Returns False if the GPU is infeasible for this task.  TTL-expired
        state is auto-cleaned here.

        Steps:
          1. GPU saturation (compute_saturated TTL)
          2. GPU unhealthy (Xid quarantine TTL) + mandatory_reset gate ()
          3. (component, gpu) activation excluded
          4. Target worker dead / unreachable / excluded
          5. VRAM budget via Planner's scenario (delegated back to caller —
             the VRAM projection requires scenario timeline access)

        NOTE on step 5: To keep the ConstraintTracker free of scenario
        dependencies, VRAM feasibility is checked by the caller
        (StochasticEFTPlacement.evaluate_gpu) using the per-GPU timeline.
        """
        now = time.time()

        expiry = self._saturated_gpus.get(gpu_id, 0.0)
        if expiry > now:
            return False
        if gpu_id in self._saturated_gpus:
            del self._saturated_gpus[gpu_id]

        unhealthy_expiry = self._unhealthy_gpus.get(gpu_id, 0.0)
        if unhealthy_expiry > now:
            return False
        if gpu_id in self._unhealthy_gpus:
            del self._unhealthy_gpus[gpu_id]
        if gpu_id in self._gpu_reset_required:
            return False
        if gpu_id in self._rma_qualifying_gpus:
            return False

        component = getattr(task_info, "component", None)
        if component is not None:
            key = (component, gpu_id)
            act_expiry = self._activation_excluded.get(key, 0.0)
            if act_expiry > now:
                return False
            if key in self._activation_excluded:
                del self._activation_excluded[key]

        target_worker = getattr(task_info, "target_worker_name", None)
        if target_worker:
            if target_worker in self._dead_workers:
                return False
            if target_worker in self._excluded_workers:
                return False
            unreach_expiry = self._unreachable_workers.get(target_worker, 0.0)
            if unreach_expiry > now:
                return False

        vram_budget = int(getattr(task_info, "vram_budget_mb", 0) or 0)
        if vram_budget > 0 and self._scenario_ref is not None:
            try:
                tl = self._scenario_ref.get(gpu_id)
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "tl = None",
                    exc_info=True,
                )
                tl = None
            if tl is not None:
                total_mb = getattr(tl, "total_vram_mb", 0.0)
                if total_mb and total_mb > 0 and total_mb < vram_budget:
                    return False

        return True


    def incorporate(self, violation: ConstraintViolation) -> bool:
        """Absorb a ConstraintViolation into the world-model lattice.

        Returns:
            True  = lattice progress (constraint accumulated OR correlation updated)
            False = no-op (already saturated state OR fallout drop)

        Raises:
            PlanningExhausted: on ``component_exhausted``.

        Inner loop fixed-point detection: when this returns False, the Core
        Loop's inner loop breaks and escalates to the Supervisor's outer
        loop ( architectural principle).
        """
        vt = violation.violation_type
        now = time.time()

        CORRELATION_WINDOW_SEC = self._correlation_window_sec
        if violation.correlation_id:
            recent = self._recent_violations_by_corr.get(violation.correlation_id)
            if recent is not None:
                age = now - recent.timestamp
                if age < CORRELATION_WINDOW_SEC and self._is_fallout(recent, violation):
                    self._fallout_drop_count[violation.correlation_id] += 1
                    return False
            self._recent_violations_by_corr[violation.correlation_id] = violation


        if vt == ViolationType.VRAM_INSUFFICIENT:
            if violation.requested_vram_mb and violation.available_vram_mb is not None:
                correction = violation.requested_vram_mb - violation.available_vram_mb
                return self._add_vram_correction(violation.gpu_id, correction)
            return False

        if vt == ViolationType.COMPUTE_SATURATED:
            ttl = self._compute_saturation_ttl(
                violation.gpu_id,
                violation,
            )
            new_expiry = now + ttl
            old_expiry = self._saturated_gpus.get(violation.gpu_id, 0.0)
            if new_expiry > old_expiry:
                self._saturated_gpus[violation.gpu_id] = new_expiry
                return True
            return False

        if vt == ViolationType.WORKER_QUEUE_SATURATED:
            _LOG.debug(
                "worker_queue_saturated: gpu=%s worker=%s backlog=%s — retry on next wake",
                violation.gpu_id,
                violation.worker_name,
                violation.active_count,
            )
            return False

        if vt == ViolationType.HOST_RAM_SATURATED:
            _LOG.warning(
                "host_ram_saturated: gpu=%s worker=%s "
                "available=%s MiB threshold=%s MiB — retry on next wake",
                violation.gpu_id,
                violation.worker_name,
                violation.host_mem_available_mib,
                violation.host_mem_threshold_mib,
            )
            return False

        if vt == ViolationType.MEMORY_ADMISSION_REJECTED:
            if (
                violation.worker_name
                and violation.worker_name not in self._excluded_workers
            ):
                self._excluded_workers.add(violation.worker_name)
                return True
            return False

        if vt in (ViolationType.WORKER_DEAD, ViolationType.HOST_RESOURCE_EXHAUSTED):
            if violation.worker_name:
                changed = False
                if violation.worker_name not in self._dead_workers:
                    self._dead_workers.add(violation.worker_name)
                    changed = True
                if violation.worker_name not in self._worker_cold_on_next_plan:
                    self._worker_cold_on_next_plan.add(violation.worker_name)
                    changed = True
                self._worker_transient_failures.pop(violation.worker_name, None)
                return changed
            return False

        if vt == ViolationType.GRPC_ERROR:
            return False

        if vt == ViolationType.WORKER_UNREACHABLE:
            new_expiry = now + self._worker_unreachable_ttl_sec
            old_expiry = self._unreachable_workers.get(violation.worker_name, 0.0)
            if violation.worker_name and new_expiry > old_expiry:
                self._unreachable_workers[violation.worker_name] = new_expiry
                return True
            return False

        if vt == ViolationType.WORKER_NOT_READY:
            if violation.worker_name:
                if violation.worker_name not in self._worker_cold_on_next_plan:
                    self._worker_cold_on_next_plan.add(violation.worker_name)
                    return True
            return False

        if vt == ViolationType.ACTIVATION_FAILED:
            comp = violation.failed_plan.component if violation.failed_plan else ""
            if not comp:
                return False
            key = (comp, violation.gpu_id)
            ttl = self._compute_activation_exclusion_ttl(comp, violation.gpu_id)
            new_expiry = now + ttl
            old_expiry = self._activation_excluded.get(key, 0.0)
            if new_expiry > old_expiry:
                self._activation_excluded[key] = new_expiry
                return True
            return False

        if vt == ViolationType.GPU_UNHEALTHY:
            xid = violation.cuda_xid or 0
            ttl, mandatory = self._resolve_xid_ttl(xid, violation.gpu_id)
            new_expiry = now + ttl
            old_expiry = self._unhealthy_gpus.get(violation.gpu_id, 0.0)
            if mandatory:
                self._gpu_reset_required.add(violation.gpu_id)
                self._gpu_reset_requested_at[violation.gpu_id] = now
                if xid == 64:
                    self._rma_qualifying_gpus.add(violation.gpu_id)
            if new_expiry > old_expiry:
                self._unhealthy_gpus[violation.gpu_id] = new_expiry
                return True
            return False

        if vt == ViolationType.PREEMPTED:
            preempt_until = now + 60.0
            if violation.memory_guard_details:
                preempt_until = violation.memory_guard_details.get(
                    "preempt_until", preempt_until
                )
            old_expiry = self._saturated_gpus.get(violation.gpu_id, 0.0)
            if preempt_until > old_expiry:
                self._saturated_gpus[violation.gpu_id] = preempt_until
                return True
            return False

        if vt == ViolationType.NUMERICAL_FAILURE:
            tid = violation.failed_plan.task_id if violation.failed_plan else ""
            if tid:
                self.record_numerical_failure(tid)
                return True
            return False

        if vt == ViolationType.SCHEDULER_INTERNAL_ERROR:
            _LOG.critical(
                "scheduler_internal_error: gpu=%s worker=%s details=%s "
                "— clearing short-TTL state (fix (h))",
                violation.gpu_id,
                violation.worker_name,
                violation.memory_guard_details,
            )
            had_state = bool(
                self._saturated_gpus
                or self._unhealthy_gpus
                or self._activation_excluded
                or self._worker_cold_on_next_plan
            )
            self._saturated_gpus.clear()
            self._unhealthy_gpus.clear()
            self._activation_excluded.clear()
            self._worker_cold_on_next_plan.clear()
            self._unreachable_workers.clear()
            return had_state

        if vt == ViolationType.STAGE_FAILED:
            if violation.failed_plan and violation.progress_info:
                tid = violation.failed_plan.task_id
                if tid and tid not in self._partial_progress:
                    self._partial_progress[tid] = violation.progress_info
                    return True
            return False

        if vt == ViolationType.COMPONENT_EXHAUSTED:
            raise PlanningExhausted(violation)

        return False


    def on_reset_completed(self, gpu_id: str) -> None:
        """GPU hardware reset completed — clear reset_required state.

         P0-SAFETY separation: this only clears ``_gpu_reset_required``
        and ``_gpu_reset_requested_at``.  ``_rma_qualifying_gpus`` is NOT
        touched here — a successful reset does not confirm RMA completion.
        Operator must call ``admin_confirm_rma_complete`` separately.
        """
        self._gpu_reset_required.discard(gpu_id)
        self._gpu_reset_requested_at.pop(gpu_id, None)

    def admin_confirm_rma_complete(self, gpu_id: str) -> None:
        """RMA / hardware replacement completed signal ( P0-SAFETY).

        Clears only ``_rma_qualifying_gpus``.  Xid 64 requires BOTH
        ``on_reset_completed`` AND this method to re-admit the GPU.
        """
        self._rma_qualifying_gpus.discard(gpu_id)


    def _is_fallout(
        self,
        earlier: ConstraintViolation,
        later: ConstraintViolation,
    ) -> bool:
        """Detect if ``later`` is a fallout of ``earlier`` (same root cause).

        Classic pattern (Blocker 2): worker guard-kill → gRPC in-flight break
        (grpc_error) → health-check fail (worker_dead) → admission reject
        (memory_admission_rejected).  First root cause is ``worker_dead``;
        the others are fallouts that should not generate separate lattice
        progress.
        """
        if earlier.violation_type == later.violation_type:
            return True
        root_cause_types = (
            ViolationType.WORKER_DEAD,
            ViolationType.HOST_RESOURCE_EXHAUSTED,
        )
        fallout_types = (
            ViolationType.GRPC_ERROR,
            ViolationType.MEMORY_ADMISSION_REJECTED,
            ViolationType.WORKER_UNREACHABLE,
        )
        return (
            earlier.violation_type in root_cause_types
            and later.violation_type in fallout_types
        )




class PlanningExhausted(Exception):
    """Raised when the Planner's feasible set for a task is empty.

    The Core Loop catches this and returns SKIP_THIS_CYCLE — the
    SchedulingSupervisor re-enqueues the task and retries later.

    Accepts either:
      - A ``ConstraintViolation`` (e.g., ``component_exhausted`` branch
        in ``ConstraintTracker.incorporate``).
      - A free-form string (used by ``GlobalPlanner.solve`` /
        ``_place_single`` when the placement loop cannot unblock a
        primary even after MCPSE eviction).

    The string form lets the Planner emit informative messages without
    synthesising a full ``ConstraintViolation`` for every dead-end.
    """

    def __init__(self, reason: Any) -> None:
        if isinstance(reason, ConstraintViolation):
            super().__init__(
                f"PlanningExhausted: component="
                f"{getattr(reason.failed_plan, 'component', '?')}"
                f" gpu={reason.gpu_id}"
            )
            self.violation = reason
        else:
            super().__init__(str(reason))
            self.violation = None


__all__ = [
    "ConstraintTracker",
    "PlanningExhausted",
    "XID_TTL_MAP_PCIE",
]
