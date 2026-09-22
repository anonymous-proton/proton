"""SchedulingSupervisor — outer-loop scheduler (Plan).

Plan B2 /  invariant.  Per-task Core Loop (``_run_task``) returns
``SKIP_THIS_CYCLE`` when the Planner cannot produce a feasible plan
(``PlanningExhausted``).  This module owns the retry-after-SKIP layer:

  - PriorityQueue of ``(priority_class, arrival_time, task_id, TaskHandle)``
  - Event-driven wake (``notify_wake(trigger)``) + jittered periodic tick
  - Atomic ``submit(task, reason)`` = put + notify (single enqueue path)
  - ``_drain_pending`` pops all pending handles per wake cycle
    (Plan : "pop all pending handles" —
    no per-cycle batch cap).  Outer bound is ``campaign_timeout_sec``
    (Plan Hybrid Bounded-Inner / Time-Bounded-Outer).

The Supervisor **does not** run the per-task Core Loop itself — it is the
layer that schedules per-task invocations.  ``TaskHandle.run`` is a
callable the caller supplies (typically a bound method returning either
``SKIP_THIS_CYCLE`` or a terminal result).

Wake triggers ( invariant — all must be wired):
  - ``task_completion``       — Event Handler ``on_task_complete``
  - ``eviction_complete``     — EvictionCoordinator ``on_eviction_complete``
  - ``gpu_health_change``     — GlobalPlanner ``on_reset_completed`` / admin_confirm_rma_complete
  - ``activation_ttl_expired`` — ConstraintTracker TTL sweeper ``_wake_hook``
  - ``periodic_tick``         — internal timer safety net

Plan invariants (tests I6, I4):
  - I6 enqueue atomicity: every put must pair with ``_wake_event.set()``.
    The only public API is ``submit`` — direct ``PriorityQueue.put_nowait``
    from the outside is forbidden.
  - I4 ``activation_ttl_expired`` dispatch: sweeper's ``_wake_hook`` ends
    up here and sets ``_wake_event``.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)


SKIP_THIS_CYCLE = object()


_SKIP_BACKOFF_SEC: float = 1.0


_REPLACEMENT_REENQUEUE_REASONS = {
    "eviction_reenqueue",
    "retry_after_adapter_failure",
}


_ACTIVE_LAUNCH_GRACE_SEC: float = 5.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class RetryDeferral:
    """Typed non-terminal retry deferral returned by a task driver.

    ``SKIP_THIS_CYCLE`` remains the generic retry-on-next-wake sentinel.
    ``RetryDeferral(reason="worker_front", component=...,
    config_fingerprint=...)`` is more specific: the task is waiting for a
    dispatch-front backlog in a component/config cohort to drain, so
    unrelated wakes should not re-run it while any same-cohort GPU worker can
    still unblock it.
    """

    reason: str
    worker_name: str = ""
    component: str = ""
    config_fingerprint: str = ""
    gpu_id: str = ""
    not_before_at: float = 0.0
    active_count: int = 0


@dataclass(order=True)
class _QueueItem:
    """PriorityQueue item.  Ordering: (priority_class, arrival_time,
    task_id).  Lower tuple compares first → higher scheduling priority.

    ``handle`` is excluded from comparison (dataclass ``compare=False``) so
    two items with identical priority keys do not require ``TaskHandle``
    to be orderable.
    """

    priority_class: int
    arrival_time: float
    task_id: str
    handle: Any = field(compare=False)


@dataclass
class TaskHandle:
    """Carrier owned by the producer and observed by the Supervisor drain.

    ``run`` is an async callable the caller supplies: it executes exactly
    one attempt at the per-task Core Loop.  Return values:
      - ``SKIP_THIS_CYCLE`` → re-enqueue via ``submit(..., reason="retry_after_skip")``
      - anything else       → treated as terminal (success or FAILED)

    Fields mirror what Plan requires for priority classification.

    ``dead_worker_retry_count`` is lifted from the per-task Core Loop
    body so that the retry budget persists across Supervisor drain
    re-entries (Plan — ``worker_dead`` / ``grpc_error``
    are the only violation types that consume a counted retry slot).

    Lifecycle fields (Plan + gateway test contract):
      - ``_done_future``: terminal-state signal.  Single source of truth
        for "this task has reached SUCCEEDED / FAILED / CANCELLED".  The
        producer (``_build_task_handle``) creates the Future; the
        per-attempt runner (``_run_task``) sets the result in its
        ``finally``; ``GatewayHTTPService._wait_for_terminal`` returns
        this Future directly.
      - ``_current_inner``: the asyncio.Task running the current attempt,
        populated inside ``handle.run()`` each time the drain invokes it.
        ``cancel()`` routes cancellation here.
      - ``_cancel_requested``: set to True by ``cancel()``.  If no inner
        exists yet (cancel arrives before drain has invoked ``run``),
        the next ``handle.run()`` short-circuits by raising
        ``CancelledError`` and cancelling ``_done_future``.
    """

    task_id: str
    arrival_time: float
    is_backfill: bool = False
    component: str = ""
    config_fingerprint: str = ""
    has_progress_hint: bool = False
    run: Callable[[], Awaitable[Any]] | None = None
    dead_worker_retry_count: int = 0
    _done_future: asyncio.Future[Any] | None = field(default=None, repr=False)
    _current_inner: asyncio.Task | None = field(default=None, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)
    _supervisor_queued: bool = field(default=False, repr=False)
    _drive_active: bool = field(default=False, repr=False)
    _last_skip_at: float = field(default=0.0, repr=False)
    _not_before_at: float = field(default=0.0, repr=False)
    _replacement_reenqueue: bool = field(default=False, repr=False)
    _drive_started_at: float = field(default=0.0, repr=False)
    _precomputed_dispatch_plan: Any | None = field(default=None, repr=False)
    _precomputed_campaign_hints: dict[str, Any] = field(
        default_factory=dict, repr=False
    )
    _precomputed_input_size: float = field(default=0.0, repr=False)

    def cancel(self) -> None:
        """Propagate cancellation into the current attempt (if running)
        and mark the handle so the next ``handle.run()`` invocation
        short-circuits.  Matches the old ``_task_tasks[task_id].cancel()``
        contract that ``_cancel_task_impl`` relies on."""
        self._cancel_requested = True
        inner = self._current_inner
        if inner is not None and not inner.done():
            inner.cancel()


class SchedulingSupervisor:
    """Outer-loop scheduler.  See module docstring for full contract.

    Lifecycle:
      1. Construct the Supervisor (no side effects).
      2. ``await start()`` — spawns the drain task.
      3. Producers call ``submit(task, reason=...)`` whenever they want the
         per-task runner executed.
      4. Event sources call ``notify_wake(trigger)`` to shrink response
         latency (no-op when the queue is already draining).
      5. ``await stop()`` to tear down.
    """

    def __init__(
        self,
        *,
        periodic_tick_sec: float = 2.0,
        jitter_max_sec: float = 0.5,
        next_event_provider: Callable[[], float | None] | None = None,
        available_dispatch_front_slots_provider: Callable[[], int] | None = None,
        worker_front_has_capacity_provider: Callable[[str], bool] | None = None,
        batch_plan_provider: Callable[
            [list[TaskHandle]],
            Awaitable[tuple[list[TaskHandle], list[TaskHandle]]],
        ]
        | None = None,
        strict_task_token_dedup: bool = False,
    ) -> None:
        self._retry_queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._wake_event = asyncio.Event()
        self._wake_trigger_counts: Counter[str] = Counter()
        self._last_wake_summary_at: float = 0.0
        self._periodic_tick_sec = periodic_tick_sec
        self._jitter_max_sec = jitter_max_sec
        self._next_event_provider: Callable[[], float | None] | None = (
            next_event_provider
        )
        self._available_dispatch_front_slots_provider = (
            available_dispatch_front_slots_provider
        )
        self._worker_front_has_capacity_provider = worker_front_has_capacity_provider
        self._batch_plan_provider = batch_plan_provider
        self._strict_task_token_dedup = bool(strict_task_token_dedup)
        self._blocked_by_worker: dict[str, list[_QueueItem]] = {}
        self._blocked_by_front_cohort: dict[tuple[str, str], list[_QueueItem]] = {}
        self._worker_front_log_state: dict[tuple[str, str], dict[str, Any]] = {}
        self._queued_task_ids: set[str] = set()
        self._active_task_ids: set[str] = set()
        self._active_handles_by_task_id: dict[str, list[TaskHandle]] = {}

        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._drive_one_tasks: set[asyncio.Task] = set()

    def _mark_waiting(self, handle: TaskHandle) -> bool:
        """Mark *handle* as queued/blocked if no copy is already pending.

        ``submit()`` is not the only place that can place a handle back into
        the supervisor: SKIP and typed deferrals intentionally requeue from
        inside ``_drive_one`` without firing a self-wake.  Those internal
        paths need the same dedupe discipline as external submit, otherwise a
        single fixed-point task can accumulate many queue copies and re-enter
        the Core Loop hundreds of times before one real dispatch succeeds.
        """
        task_id = str(getattr(handle, "task_id", "") or "")
        active_duplicate = self._task_id_has_live_active(task_id)
        handle_drive_active = self._handle_is_live_active(handle)
        if (
            task_id in self._queued_task_ids
            or active_duplicate
            or getattr(handle, "_supervisor_queued", False)
            or handle_drive_active
        ):
            _LOG.debug(
                "[scheduler-sup] dedupe internal deferral task=%s queued=%s active=%s",
                handle.task_id,
                getattr(handle, "_supervisor_queued", False)
                or task_id in self._queued_task_ids,
                handle_drive_active or active_duplicate,
            )
            return False
        handle._supervisor_queued = True
        if task_id:
            self._queued_task_ids.add(task_id)
        return True


    async def start(self) -> None:
        """Spawn the drain loop.  Idempotent."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the drain loop + gather in-flight per-handle driver Tasks.

        Plan Fix-1 (M3-D7) — drain loop cancel  ``_drive_one``
        tasks  cancel + gather (return_exceptions=True)  orphan
        .  Graceful shutdown : _done_future callbacks  shutdown
            resolve, VRAM cleanup  _drive_one finally
          .  Idempotent.
        """
        self._running = False
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            self._wake_event.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        drivers = list(self._drive_one_tasks)
        for drv in drivers:
            if not drv.done():
                drv.cancel()
        if drivers:
            await asyncio.gather(*drivers, return_exceptions=True)
        self._drive_one_tasks.clear()


    def submit(self, handle: TaskHandle, reason: str = "submit") -> None:
        """Enqueue *handle* and fire the wake event atomically.

        Producer sites — **external producers plus explicit supervisor-owned
        retry paths** ( P1 invariant: direct ``_retry_queue.put_nowait``
        from outside the Supervisor is forbidden).  Internal retry paths may
        also re-enter through ``submit()`` so duplicate ownership checks and
        queue state transitions stay centralized.

        Audited external producer sites in the current codebase:
          - ``reason="arrival"``                — http_server task enqueue
            (``_submit_task_impl``).
          - ``reason="eviction_reenqueue"``     — EvictionCoordinator
            backfill re-enqueue after eviction.
          - ``reason="retry_after_skip"``       — (internal) Supervisor
            SKIP path.  The handle is re-submitted with ``_not_before_at``
            set, so the wake is harmless: the drain observes the queued
            handle, then defers it until the micro-backoff expires instead
            of busy-spinning.

        Planned / aspirational producer sites in the plan docstring
        that **do not yet have call-sites** (Phase 3 M3-D9 finding):
          - ``reason="stage_completed"``  — stage completion → dependent
            task enqueue (external orchestrator responsibility; not
            implemented at this Supervisor boundary).
          - ``reason="profile_drift"``    — aspirational drift-triggered
            task requeue.  The
            ProfileDriftEvent callback registry (``planner.py:226``,
            ``global_planner.py:2276``, ``campaign_scheduler.py:1419``)
            processes drift events but currently **does not** route
            through ``submit()``; re-planning happens via invalidation,
            wake, and the Planner's fresh-snapshot mechanism on the next
            outer-loop entry.
            Future work: wire drift callbacks to ``submit(...,
            reason="profile_drift")`` once drift-triggered re-enqueue
            semantics are formalized.
        """
        done = getattr(handle, "_done_future", None)
        if done is not None and done.done():
            _LOG.debug(
                "[scheduler-sup] drop terminal submit task=%s reason=%s",
                handle.task_id,
                reason,
            )
            return
        if getattr(handle, "_cancel_requested", False):
            _LOG.debug(
                "[scheduler-sup] drop cancelled submit task=%s reason=%s",
                handle.task_id,
                reason,
            )
            return
        task_id = str(getattr(handle, "task_id", "") or "")
        replacement_reenqueue = str(reason or "") in _REPLACEMENT_REENQUEUE_REASONS
        queued_duplicate = task_id in self._queued_task_ids or getattr(
            handle, "_supervisor_queued", False
        )
        active_duplicate = self._task_id_has_live_active(task_id)
        handle_drive_active = self._handle_is_live_active(handle)
        if (
            queued_duplicate
            or handle_drive_active
            or (active_duplicate and not replacement_reenqueue)
        ):
            _LOG.debug(
                "[scheduler-sup] dedupe submit task=%s reason=%s queued=%s active=%s",
                handle.task_id,
                reason,
                queued_duplicate,
                handle_drive_active or active_duplicate,
            )
            return
        priority_class = self._classify_priority(handle)
        item = _QueueItem(
            priority_class=priority_class,
            arrival_time=handle.arrival_time,
            task_id=handle.task_id,
            handle=handle,
        )
        handle._supervisor_queued = True
        handle._replacement_reenqueue = replacement_reenqueue
        if task_id:
            self._queued_task_ids.add(task_id)
        self._retry_queue.put_nowait(item)
        _LOG.debug(
            "[scheduler-sup] submit task=%s priority=%d reason=%s",
            handle.task_id,
            priority_class,
            reason,
        )
        self.notify_wake(trigger=f"submit({reason})")

    def notify_wake(self, trigger: str) -> None:
        """Fire the wake event.  Idempotent (re-setting is a no-op)."""
        self._wake_trigger_counts[trigger] += 1
        _LOG.debug("[scheduler-sup] wake trigger=%s", trigger)
        self._wake_event.set()

    def _maybe_log_wake_summary(self) -> None:
        now = time.time()
        if now - self._last_wake_summary_at < 60.0:
            return
        self._last_wake_summary_at = now
        if self._wake_trigger_counts:
            _LOG.info(
                "[scheduler-sup] wake-trigger-summary %s",
                dict(sorted(self._wake_trigger_counts.items(), key=lambda kv: -kv[1])),
            )

    def _retry_queue_items(self) -> list[_QueueItem]:
        try:
            return list(getattr(self._retry_queue, "_queue", []) or [])
        except Exception:
            return []

    def _handle_is_live_active(self, handle: TaskHandle) -> bool:
        inner = getattr(handle, "_current_inner", None)
        if inner is not None:
            return not inner.done()
        if not getattr(handle, "_drive_active", False):
            return False
        started_at = handle._drive_started_at
        return (
            started_at <= 0.0 or (time.time() - started_at) <= _ACTIVE_LAUNCH_GRACE_SEC
        )

    def _task_id_has_live_active(self, task_id: str) -> bool:
        if not task_id:
            return False
        handles = list(self._active_handles_by_task_id.get(task_id, []) or [])
        if not handles:
            self._active_task_ids.discard(task_id)
            return False
        live: list[TaskHandle] = []
        for active_handle in handles:
            if self._handle_is_live_active(active_handle):
                live.append(active_handle)
            else:
                active_handle._drive_active = False
                active_handle._drive_started_at = 0.0
        if live:
            self._active_handles_by_task_id[task_id] = live
            self._active_task_ids.add(task_id)
            return True
        self._active_handles_by_task_id.pop(task_id, None)
        self._active_task_ids.discard(task_id)
        return False

    def _register_active_handle(self, handle: TaskHandle) -> None:
        task_id = str(getattr(handle, "task_id", "") or "")
        if not task_id:
            return
        handles = self._active_handles_by_task_id.setdefault(task_id, [])
        if all(existing is not handle for existing in handles):
            handles.append(handle)
        self._active_task_ids.add(task_id)
        handle._drive_active = True
        handle._drive_started_at = time.time()

    def _unregister_active_handle(self, handle: TaskHandle) -> None:
        task_id = str(getattr(handle, "task_id", "") or "")
        handle._drive_active = False
        handle._drive_started_at = 0.0
        if not task_id:
            return
        handles = self._active_handles_by_task_id.get(task_id)
        if handles is not None:
            kept = [existing for existing in handles if existing is not handle]
            if kept:
                self._active_handles_by_task_id[task_id] = kept
            else:
                self._active_handles_by_task_id.pop(task_id, None)
        if not self._active_handles_by_task_id.get(task_id):
            self._active_task_ids.discard(task_id)

    def _handle_live_location(self, handle: TaskHandle) -> str:
        """Return where this exact handle is able to make progress."""
        for item in self._retry_queue_items():
            if item.handle is handle:
                return "queued"
        for items in self._blocked_by_worker.values():
            for item in items:
                if item.handle is handle:
                    return "blocked_worker"
        for items in self._blocked_by_front_cohort.values():
            for item in items:
                if item.handle is handle:
                    return "blocked_front"
        inner = getattr(handle, "_current_inner", None)
        if inner is not None and not inner.done():
            return "active"
        if self._handle_is_live_active(handle):
            return "active"
        return ""

    def _task_id_waiter_location(self, task_id: str) -> str:
        """Return where any queued/blocked waiter for *task_id* lives."""
        if not task_id:
            return ""
        for item in self._retry_queue_items():
            if str(getattr(item, "task_id", "") or "") == task_id:
                return "queued"
        for items in self._blocked_by_worker.values():
            for item in items:
                if str(getattr(item, "task_id", "") or "") == task_id:
                    return "blocked_worker"
        for items in self._blocked_by_front_cohort.values():
            for item in items:
                if str(getattr(item, "task_id", "") or "") == task_id:
                    return "blocked_front"
        return ""

    def _remove_task_id_waiters_except(self, task_id: str, keep: TaskHandle) -> int:
        """Drop stale queued/blocked waiters for *task_id* except *keep*."""
        if not task_id:
            return 0
        removed = 0
        queue = self._retry_queue_items()
        if queue:
            kept_queue: list[_QueueItem] = []
            for item in queue:
                if (
                    str(getattr(item, "task_id", "") or "") == task_id
                    and item.handle is not keep
                ):
                    removed += 1
                    item.handle._supervisor_queued = False
                    continue
                kept_queue.append(item)
            if removed:
                with contextlib.suppress(Exception):
                    raw_queue = vars(self._retry_queue)["_queue"]
                    raw_queue[:] = kept_queue
                    heapq.heapify(raw_queue)
        for key in list(self._blocked_by_worker):
            items = self._blocked_by_worker.get(key, [])
            kept = []
            for item in items:
                if (
                    str(getattr(item, "task_id", "") or "") == task_id
                    and item.handle is not keep
                ):
                    removed += 1
                    item.handle._supervisor_queued = False
                    continue
                kept.append(item)
            if kept:
                self._blocked_by_worker[key] = kept
            else:
                self._blocked_by_worker.pop(key, None)
        for key in list(self._blocked_by_front_cohort):
            items = self._blocked_by_front_cohort.get(key, [])
            kept = []
            for item in items:
                if (
                    str(getattr(item, "task_id", "") or "") == task_id
                    and item.handle is not keep
                ):
                    removed += 1
                    item.handle._supervisor_queued = False
                    continue
                kept.append(item)
            if kept:
                self._blocked_by_front_cohort[key] = kept
            else:
                self._blocked_by_front_cohort.pop(key, None)
        if removed:
            self._queued_task_ids.discard(task_id)
        return removed

    def _has_stale_ownership_flags(self, handle: TaskHandle) -> bool:
        task_id = str(getattr(handle, "task_id", "") or "")
        return bool(
            task_id in self._queued_task_ids
            or self._task_id_has_live_active(task_id)
            or getattr(handle, "_supervisor_queued", False)
            or self._handle_is_live_active(handle)
        )

    def _clear_stale_ownership_flags(
        self,
        handle: TaskHandle,
        *,
        clear_active: bool = False,
    ) -> None:
        task_id = str(getattr(handle, "task_id", "") or "")
        if task_id:
            self._queued_task_ids.discard(task_id)
            if clear_active:
                self._active_task_ids.discard(task_id)
        handle._supervisor_queued = False
        if clear_active:
            handle._drive_active = False
            handle._drive_started_at = 0.0
            self._unregister_active_handle(handle)
        handle._replacement_reenqueue = False

    def is_handle_owned(self, handle: TaskHandle) -> bool:
        """Return True when the supervisor can still drive *handle*.

        Older code treated bookkeeping flags as ownership.  That is too
        optimistic for liveness: a stale ``_supervisor_queued`` flag can make
        ``submit()`` dedupe the only retry handle even though no concrete
        queue, blocked bucket, or driver task exists.  This predicate is now
        grounded in actual supervisor containers / live driver state.
        """
        return bool(self._handle_live_location(handle))

    def ensure_handle_requeued_or_woken(
        self,
        handle: TaskHandle,
        *,
        reason: str = "eviction_reenqueue",
    ) -> str:
        """Ensure a no-inflight retry handle remains live.

        Returns a small action string for diagnostics:
          - ``terminal`` / ``cancelled``: nothing was queued.
          - ``woken_*``: the supervisor already has a concrete queue,
            blocked, or active owner, so only a wake was needed.
          - ``recovered_stale_owned``: stale ownership flags were cleared
            and the handle was newly enqueued.
          - ``submitted``: an orphan handle was newly enqueued.
          - ``dropped``: ``submit()`` declined the handle after a race.
        """
        done = getattr(handle, "_done_future", None)
        if done is not None and done.done():
            _LOG.debug(
                "[scheduler-sup] drop terminal requeue task=%s reason=%s",
                handle.task_id,
                reason,
            )
            return "terminal"
        if getattr(handle, "_cancel_requested", False):
            _LOG.debug(
                "[scheduler-sup] drop cancelled requeue task=%s reason=%s",
                handle.task_id,
                reason,
            )
            return "cancelled"
        task_id = str(getattr(handle, "task_id", "") or "")
        location = self._handle_live_location(handle)
        if location:
            handle._not_before_at = 0.0
            self.notify_wake(trigger=f"{reason}:owned")
            _LOG.debug(
                "[scheduler-sup] wake owned requeue task=%s reason=%s location=%s",
                handle.task_id,
                reason,
                location,
            )
            return f"woken_{location}"

        if (
            self._strict_task_token_dedup
            and str(reason or "") in _REPLACEMENT_REENQUEUE_REASONS
        ):
            waiter_location = self._task_id_waiter_location(task_id)
            if waiter_location:
                handle._supervisor_queued = False
                handle._replacement_reenqueue = False
                self.notify_wake(trigger=f"{reason}:dedup_waiter")
                _LOG.debug(
                    "[scheduler-sup] collapse duplicate replacement task=%s "
                    "reason=%s location=%s",
                    handle.task_id,
                    reason,
                    waiter_location,
                )
                return f"woken_{waiter_location}"

        removed_waiters = self._remove_task_id_waiters_except(task_id, handle)
        had_stale_flags = self._has_stale_ownership_flags(handle) or removed_waiters > 0
        if had_stale_flags:
            self._clear_stale_ownership_flags(
                handle,
                clear_active=str(reason or "") not in _REPLACEMENT_REENQUEUE_REASONS,
            )
        self.submit(handle, reason=reason)
        if self._handle_live_location(handle):
            if had_stale_flags:
                _LOG.info(
                    "[scheduler-sup] recovered stale-owned requeue task=%s "
                    "reason=%s removed_waiters=%d",
                    handle.task_id,
                    reason,
                    removed_waiters,
                )
                return "recovered_stale_owned"
            return "submitted"
        return "dropped"

    def notify_worker_front_capacity(self, worker_name: str) -> None:
        """Requeue handles blocked on a worker-front capacity event.

        A worker-front release is GPU-worker local, but a blocked task can be
        replanned to a sibling worker of the same component/config fingerprint.
        The worker name itself does not carry a fingerprint, so this wake
        considers exact worker waiters plus same-component cohort waiters, then
        releases only the capacity represented by this event.
        """
        worker = str(worker_name or "").strip()
        if not worker:
            return
        if self._worker_front_has_capacity_provider is None:
            release_limit = 1
        else:
            release_limit = 1 if self._worker_front_has_capacity(worker) else 0
        items_with_origin: list[tuple[tuple[str, str] | None, _QueueItem]] = [
            (None, item) for item in self._blocked_by_worker.pop(worker, [])
        ]
        component = self._component_from_worker_name(worker)
        cohort_keys: list[tuple[str, str]] = []
        if component:
            cohort_keys = [
                key
                for key in list(self._blocked_by_front_cohort)
                if key[0] == component
            ]
        for key in cohort_keys:
            items_with_origin.extend(
                (key, item) for item in self._blocked_by_front_cohort.pop(key, [])
            )
        if not items_with_origin:
            return
        items_with_origin.sort(key=lambda pair: pair[1])
        seen_task_ids: set[str] = set()
        unblocked = 0
        retained: list[tuple[tuple[str, str] | None, _QueueItem]] = []
        for origin, item in items_with_origin:
            task_id = str(getattr(item.handle, "task_id", "") or "")
            if task_id:
                if task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
            done = getattr(item.handle, "_done_future", None)
            if done is not None and done.done():
                if task_id:
                    self._queued_task_ids.discard(task_id)
                item.handle._supervisor_queued = False
                continue
            if getattr(item.handle, "_cancel_requested", False):
                if task_id:
                    self._queued_task_ids.discard(task_id)
                item.handle._supervisor_queued = False
                continue
            if unblocked >= release_limit:
                retained.append((origin, item))
                continue
            self._retry_queue.put_nowait(item)
            unblocked += 1
        for origin, item in retained:
            if origin is None:
                self._blocked_by_worker.setdefault(worker, []).append(item)
            else:
                self._blocked_by_front_cohort.setdefault(origin, []).append(item)
        _LOG.debug(
            "[scheduler-sup] worker-front capacity worker=%s component=%s "
            "cohorts=%d unblocked=%d retained=%d",
            worker,
            component,
            len(cohort_keys),
            unblocked,
            len(retained),
        )
        if unblocked:
            self.notify_wake(trigger=f"worker_front_capacity({worker})")


    @staticmethod
    def _classify_priority(handle: TaskHandle) -> int:
        """Priority class.  Lower = higher priority.

        Mapping (plan):
          - 0 = primary (campaign WSJF head)
          - 1 = backfill
          - 2 = stage_failed retry (progress hint present)
        """
        if handle.has_progress_hint:
            return 2
        if handle.is_backfill:
            return 1
        return 0

    @staticmethod
    def _component_from_worker_name(worker_name: str) -> str:
        worker = str(worker_name or "").strip()
        marker = "-gpu"
        if marker in worker:
            return worker.rsplit(marker, 1)[0]
        return worker

    async def _run(self) -> None:
        """Main drain loop.  See class docstring for contract.

        Pattern (plan , with fire-and-forget per-handle execution):
            while self._running:
                pending = drain();
                if pending empty: wait_for_wake_or_tick(); continue
                for handle in pending:
                    launch handle.run() as independent Task
                wait_for_wake_or_tick()

        Each handle.run() is launched as an independent asyncio.Task
        (``_drive_one``) so concurrent per-task attempts execute in
        parallel rather than serializing on the drain coroutine.  SKIP
        semantics: the per-handle driver observes ``SKIP_THIS_CYCLE``
        as ``handle.run`` return value and re-enqueues via ``submit``.
        """
        while self._running:
            self._maybe_log_wake_summary()
            self._release_ready_worker_front_blocked()
            pending = self._drain_pending()
            if not pending:
                await self._wait_for_wake_or_tick()
                continue
            pending = await self._apply_batch_planning(pending)
            if not pending:
                await self._wait_for_wake_or_tick()
                continue
            for handle in pending:
                if handle.run is None:
                    self._unregister_active_handle(handle)
                    continue
                task = asyncio.create_task(self._drive_one(handle))
                self._drive_one_tasks.add(task)
            await self._wait_for_wake_or_tick()

    async def _apply_batch_planning(
        self, pending: list[TaskHandle]
    ) -> list[TaskHandle]:
        provider = self._batch_plan_provider
        if provider is None:
            return pending
        try:
            launchable, deferred = await provider(pending)
        except Exception:
            _LOG.warning(
                "[scheduler-sup] batch planning provider failed; falling back "
                "to per-handle runners",
                exc_info=True,
            )
            return pending
        launch_ids = {id(handle) for handle in launchable}
        defer_ids = {id(handle) for handle in deferred}
        if not launch_ids and not defer_ids:
            return pending
        unknown = [
            handle
            for handle in pending
            if id(handle) not in launch_ids and id(handle) not in defer_ids
        ]
        now = time.time()
        for handle in deferred:
            self._unregister_active_handle(handle)
            deferral = getattr(handle, "_batch_planning_retry_deferral", None)
            if isinstance(deferral, RetryDeferral):
                try:
                    delattr(handle, "_batch_planning_retry_deferral")
                except Exception:
                    handle._batch_planning_retry_deferral = None
                self._defer_retry(handle, deferral)
            else:
                handle._last_skip_at = now
                handle._not_before_at = max(
                    handle._not_before_at,
                    now + _SKIP_BACKOFF_SEC,
                )
                self._requeue_internal(handle, reason="batch_planning_skip")
        for handle in list(launchable) + unknown:
            if hasattr(handle, "_batch_planning_retry_deferral"):
                try:
                    delattr(handle, "_batch_planning_retry_deferral")
                except Exception:
                    handle._batch_planning_retry_deferral = None
        if deferred:
            _LOG.debug(
                "[scheduler-sup] batch planning launch=%d defer=%d fallback=%d",
                len(launchable),
                len(deferred),
                len(unknown),
            )
        return list(launchable) + unknown

    def _requeue_internal(self, handle: TaskHandle, *, reason: str) -> None:
        if not self._mark_waiting(handle):
            return
        priority_class = self._classify_priority(handle)
        item = _QueueItem(
            priority_class=priority_class,
            arrival_time=handle.arrival_time,
            task_id=handle.task_id,
            handle=handle,
        )
        self._retry_queue.put_nowait(item)
        _LOG.debug(
            "[scheduler-sup] internal requeue task=%s priority=%d reason=%s",
            handle.task_id,
            priority_class,
            reason,
        )

    async def _drive_one(self, handle: TaskHandle) -> None:
        """Per-handle driver Task: awaits ``handle.run()`` and handles
        SKIP re-enqueue.  Exception isolation so one handle's error
        never tears down the drain loop.

        Registers itself as ``handle._current_inner`` so
        ``handle.cancel()`` (called from ``_cancel_task_impl``) can
        propagate CancelledError into the currently-running
        ``handle.run()`` → ``_run_task`` coroutine via the drive Task.

        Plan Fix-1 (M3-D7) — also registers itself in
        ``self._drive_one_tasks`` so ``stop()`` can gather and cancel
        all in-flight drivers during graceful shutdown.

        Plan busy-loop prevention — queue-level ``_not_before_at``
        normally prevents early driver creation.  Keep this local sleep as
        a defensive fallback for direct invocations that bypass
        ``_drain_pending``.
        """
        elapsed = time.time() - handle._last_skip_at
        if 0 < elapsed < _SKIP_BACKOFF_SEC:
            try:
                await asyncio.sleep(_SKIP_BACKOFF_SEC - elapsed)
            except asyncio.CancelledError as _cancelled:
                return

        current = asyncio.current_task()
        handle._current_inner = current
        if current is not None:
            self._drive_one_tasks.add(current)
        try:
            run = handle.run
            if run is None:
                return
            result = await run()
        except asyncio.CancelledError as _cancelled:
            return
        except Exception as exc:
            _LOG.error(
                "[scheduler-sup] task=%s runner raised: %s",
                handle.task_id,
                exc,
                exc_info=True,
            )
            return
        finally:
            handle._current_inner = None
            if current is not None:
                self._drive_one_tasks.discard(current)
            self._unregister_active_handle(handle)
        if isinstance(result, RetryDeferral):
            self._defer_retry(handle, result)
            return

        if result is SKIP_THIS_CYCLE:
            handle._last_skip_at = time.time()
            handle._not_before_at = handle._last_skip_at + _SKIP_BACKOFF_SEC
            self.submit(handle, reason="retry_after_skip")
            _LOG.debug(
                "[scheduler-sup] deferred task=%s (SKIP, wake after backoff)",
                handle.task_id,
            )

    def _defer_retry(self, handle: TaskHandle, deferral: RetryDeferral) -> None:
        priority_class = self._classify_priority(handle)
        item = _QueueItem(
            priority_class=priority_class,
            arrival_time=handle.arrival_time,
            task_id=handle.task_id,
            handle=handle,
        )
        if deferral.not_before_at > 0.0:
            handle._not_before_at = max(handle._not_before_at, deferral.not_before_at)
        reason = str(deferral.reason or "").strip()
        worker = str(deferral.worker_name or "").strip()
        if reason == "worker_front" and worker:
            if self._worker_front_has_capacity(worker):
                if not self._mark_waiting(handle):
                    return
                self._retry_queue.put_nowait(item)
                _LOG.debug(
                    "[scheduler-sup] worker-front capacity already available "
                    "worker=%s task=%s priority=%d",
                    worker,
                    handle.task_id,
                    priority_class,
                )
                self.notify_wake(trigger=f"worker_front_capacity_ready({worker})")
                return
            retained_plan = getattr(handle, "_precomputed_dispatch_plan", None)
            retained_worker = str(
                getattr(retained_plan, "target_worker_name", "") or ""
            ).strip()
            component = ""
            if retained_worker != worker:
                component = (
                    str(deferral.component or "").strip()
                    or str(getattr(handle, "component", "") or "").strip()
                    or self._component_from_worker_name(worker)
                )
            config_fp = (
                str(deferral.config_fingerprint or "").strip()
                or str(getattr(handle, "config_fingerprint", "") or "").strip()
            )
            if component:
                if not self._mark_waiting(handle):
                    return
                self._blocked_by_front_cohort.setdefault(
                    (component, config_fp),
                    [],
                ).append(item)
            else:
                if not self._mark_waiting(handle):
                    return
                self._blocked_by_worker.setdefault(worker, []).append(item)
            self._log_worker_front_deferral(worker, deferral, item)
            return
        if not self._mark_waiting(handle):
            return
        self._retry_queue.put_nowait(item)
        self.notify_wake(trigger=f"deferred({reason or 'unknown'})")
        _LOG.debug(
            "[scheduler-sup] deferred task=%s priority=%d reason=%s",
            handle.task_id,
            priority_class,
            reason or "unknown",
        )

    def _worker_front_has_capacity(self, worker: str) -> bool:
        provider = self._worker_front_has_capacity_provider
        if provider is None:
            return False
        try:
            return bool(provider(worker))
        except Exception as exc:
            _LOG.debug(
                "[scheduler-sup] worker-front capacity provider failed worker=%s: %s",
                worker,
                exc,
            )
            return True

    def _dispatch_front_slots_available_count(self) -> int:
        provider = self._available_dispatch_front_slots_provider
        if provider is None:
            return 0
        try:
            return max(0, int(provider() or 0))
        except Exception as exc:
            _LOG.debug(
                "[scheduler-sup] dispatch-front slot provider failed: %s",
                exc,
            )
            return 1

    def _dispatch_front_slot_available(self) -> bool:
        return self._dispatch_front_slots_available_count() > 0

    def _requeue_worker_front_items(
        self,
        items: list[_QueueItem],
        *,
        limit: int | None = None,
    ) -> tuple[int, list[_QueueItem]]:
        items = sorted(items)
        seen_task_ids: set[str] = set()
        unblocked = 0
        retained: list[_QueueItem] = []
        for item in items:
            task_id = str(getattr(item.handle, "task_id", "") or "")
            if task_id:
                if task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
            done = getattr(item.handle, "_done_future", None)
            if done is not None and done.done():
                if task_id:
                    self._queued_task_ids.discard(task_id)
                item.handle._supervisor_queued = False
                continue
            if getattr(item.handle, "_cancel_requested", False):
                if task_id:
                    self._queued_task_ids.discard(task_id)
                item.handle._supervisor_queued = False
                continue
            if limit is not None and unblocked >= limit:
                retained.append(item)
                continue
            self._retry_queue.put_nowait(item)
            unblocked += 1
        return unblocked, retained

    def _release_ready_worker_front_blocked(self) -> int:
        """Periodically revalidate worker-front deferrals.

        Capacity notifications are edge-triggered by worker completions.  A
        handle can otherwise remain parked after the last relevant worker has
        already become idle, leaving no future completion event to release it.
        The periodic sweep is deliberately conservative: exact-worker waiters
        require that worker to have capacity, while component/config cohort
        waiters are requeued up to the observed dispatch-front slot budget so
        the normal planner/validator path can pick currently feasible workers
        without recreating a wake storm.
        """
        released = 0
        for worker in list(self._blocked_by_worker):
            if not self._worker_front_has_capacity(worker):
                continue
            count, retained = self._requeue_worker_front_items(
                self._blocked_by_worker.pop(worker, []),
                limit=1,
            )
            released += count
            if retained:
                self._blocked_by_worker.setdefault(worker, []).extend(retained)

        available_slots = self._dispatch_front_slots_available_count()
        if self._blocked_by_front_cohort and available_slots > 0:
            cohort_items: list[tuple[tuple[str, str], _QueueItem]] = []
            for key in list(self._blocked_by_front_cohort):
                cohort_items.extend(
                    (key, item) for item in self._blocked_by_front_cohort.pop(key, [])
                )
            cohort_items.sort(key=lambda pair: pair[1])
            requeue_items = [item for _key, item in cohort_items]
            count, retained_items = self._requeue_worker_front_items(
                requeue_items,
                limit=available_slots,
            )
            released += count
            retained_ids = {id(item) for item in retained_items}
            for key, item in cohort_items:
                if id(item) in retained_ids:
                    self._blocked_by_front_cohort.setdefault(key, []).append(item)

        if released:
            _LOG.debug(
                "[scheduler-sup] released worker-front blocked handles=%d",
                released,
            )
        return released

    def _log_worker_front_deferral(
        self,
        worker: str,
        deferral: RetryDeferral,
        item: _QueueItem,
    ) -> None:
        key = (worker, str(deferral.reason or "worker_front"))
        state = self._worker_front_log_state.setdefault(
            key,
            {
                "last": 0.0,
                "count": 0,
                "max_backlog": 0,
                "gpu": str(deferral.gpu_id or ""),
            },
        )
        state["count"] = _as_int(state.get("count", 0)) + 1
        with contextlib.suppress(Exception):
            state["max_backlog"] = max(
                _as_int(state.get("max_backlog", 0)),
                deferral.active_count,
            )
        if deferral.gpu_id:
            state["gpu"] = str(deferral.gpu_id)
        now = time.time()
        count = _as_int(state.get("count", 0))
        last = _as_float(state.get("last", 0.0))
        if last <= 0.0:
            state["last"] = now
            last = now
        if count < 100 and now - last < 10.0:
            return
        component = (
            str(deferral.component or "").strip()
            or str(getattr(item.handle, "component", "") or "").strip()
            or self._component_from_worker_name(worker)
        )
        config_fp = (
            str(deferral.config_fingerprint or "").strip()
            or str(getattr(item.handle, "config_fingerprint", "") or "").strip()
        )
        blocked = len(self._blocked_by_worker.get(worker, []))
        if component:
            blocked += len(
                self._blocked_by_front_cohort.get((component, config_fp), [])
            )
        _LOG.info(
            "[scheduler-sup] worker_front deferred worker=%s component=%s "
            "config_fp=%s gpu=%s "
            "suppressed_count=%d max_backlog=%s blocked_handles=%d",
            worker,
            component,
            config_fp,
            state.get("gpu", ""),
            count,
            state.get("max_backlog", 0),
            blocked,
        )
        state["count"] = 0
        state["last"] = now
        state["max_backlog"] = 0

    def _drain_pending(self) -> list[TaskHandle]:
        """Pop all pending handles (priority order, non-blocking).

        Plan — ``_drain_pending`` pops
        the entire ready queue per wake cycle, with no per-cycle batch
        cap.  Items are already PriorityQueue-ordered, so the returned
        list respects (priority_class, arrival_time, task_id).

        Outer-loop bound: ``campaign_timeout_sec`` at the per-task
        level (Plan Hybrid Bounded-Inner / Time-Bounded-
        Outer).  Inner bound: constraint lattice fixed-point.  No
        intermediate per-cycle batch cap is specified by plan.
        """
        out: list[TaskHandle] = []
        deferred: list[_QueueItem] = []
        now = time.time()
        while True:
            try:
                item = self._retry_queue.get_nowait()
                not_before = item.handle._not_before_at
                if not_before > now:
                    deferred.append(item)
                    continue
                task_id = str(getattr(item.handle, "task_id", "") or "")
                handle_drive_active = self._handle_is_live_active(item.handle)
                active_duplicate = self._task_id_has_live_active(task_id)
                if handle_drive_active or active_duplicate:
                    if getattr(item.handle, "_replacement_reenqueue", False):
                        deferred.append(item)
                        continue
                    item.handle._supervisor_queued = False
                    if task_id:
                        self._queued_task_ids.discard(task_id)
                    continue
                item.handle._supervisor_queued = False
                if task_id:
                    self._queued_task_ids.discard(task_id)
                self._register_active_handle(item.handle)
                out.append(item.handle)
            except asyncio.QueueEmpty as _empty:
                break
        for item in deferred:
            self._retry_queue.put_nowait(item)
        return out

    def _next_retry_eligible_at(self) -> float | None:
        try:
            items = list(getattr(self._retry_queue, "_queue", []) or [])
        except Exception:
            return None
        now = time.time()
        next_at: float | None = None
        for item in items:
            ts = item.handle._not_before_at
            if ts <= now:
                return now
            if next_at is None or ts < next_at:
                next_at = ts
        return next_at

    async def _wait_for_wake_or_tick(self) -> None:
        """Block until a wake trigger fires OR the jittered periodic tick
        expires.  ``_wake_event.clear()`` runs in finally to guarantee a
        fresh wait on the next iteration.

        Jitter (plan): desynchronize multiple supervisor instances —
        does not affect intra-batch fairness.

        Plan fix — Dynamic periodic tick timeout:
        Compute ``timeout`` as min(next_predicted_completion - now,
        periodic_tick_sec) when a next-event provider is wired and
        returns a non-None value.  Cold-start / low-activity 
         10s    event   wait   
        .  Provider  /  None /   (> periodic_tick_sec)
          static periodic_tick_sec fallback.  Floor 0.1s  race
         (scenario   event  sub-ms timestamp ).
        Jitter   timeout    .
        """
        base = self._periodic_tick_sec
        if self._next_event_provider is not None:
            try:
                next_at = self._next_event_provider()
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "next_at = None",
                    exc_info=True,
                )
                next_at = None
            if next_at is not None:
                delta = next_at - time.time()
                if 0.0 < delta < base:
                    base = max(0.1, delta)
        next_retry = self._next_retry_eligible_at()
        if next_retry is not None:
            delta = next_retry - time.time()
            if delta <= 0.0:
                base = 0.1
            elif delta < base:
                base = max(0.1, delta)
        timeout = base + random.uniform(0.0, self._jitter_max_sec)
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as _timeout:
            self._wake_trigger_counts["periodic_tick"] += 1
            return
        finally:
            self._wake_event.clear()
