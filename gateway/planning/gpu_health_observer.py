"""GpuHealthObserver — NVML recovery-channel bridge (Plan P0-CRITICAL).

Without this module, ``_gpu_reset_required`` / ``_rma_qualifying_gpus``
are add-only sets: Xid 64/95 marks a GPU, and the GPU becomes permanently
infeasible (no recovery path).  This observer polls NVML field ID 230
(``NVML_FI_DEV_GET_GPU_RECOVERY_ACTION``) and clears the reset-required
flag on those GPUs once the driver reports ``RECOVERY_ACTION_NONE`` — or
when an operator explicitly calls ``admin_confirm_reset`` /
``admin_confirm_rma_complete``.

Plan recovery triggers (all three wired):

  1. **NVML auto-detect** — field ID 230 reads ``NONE`` AND elapsed time
     since mandatory-reset request ≥ ``min_settle_time_sec`` (default 30 s).
     Driver ≥ r570 is required; older drivers fall through to trigger 3.
  2. **Operator API** — ``admin_confirm_reset(gpu_id)`` flips the reset
     flag explicitly.  ``admin_confirm_rma_complete(gpu_id)`` additionally
     clears the RMA-qualifying set for Xid 64.
  3. **force_clear_after_sec override** — only when the operator
     explicitly sets this config; default is ``None`` (disabled).  Plain
     wall-clock timeout without NVML verification is unsafe, so this
     path is opt-in.

``max_quarantine_sec`` (3600 s) is the upper bound on NVML auto-clear
elapsed; complex condition is
``min_settle_time_sec ≤ elapsed ≤ max_quarantine_sec AND recovery_action == NONE``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

_LOG = logging.getLogger(__name__)


NVML_FI_DEV_GET_GPU_RECOVERY_ACTION = 230
NVML_GPU_RECOVERY_ACTION_NONE = 0x0
NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED = 0x1
NVML_GPU_RECOVERY_ACTION_NODE_REBOOT_REQUIRED = 0x2
NVML_GPU_RECOVERY_ACTION_DRAIN_P2P = 0x3
NVML_GPU_RECOVERY_ACTION_DRAIN_AND_RESET = 0x4


class GpuHealthObserver:
    """Background task bridging NVML / operator recovery signals to
    ``ConstraintTracker`` state.

    Runs in a loop on ``poll_interval_sec`` (default 30 s).  For every
    GPU in ``tracker._gpu_reset_required``, evaluates the three recovery
    triggers and calls ``GlobalPlanner.on_reset_completed`` /
    ``admin_confirm_rma_complete`` as appropriate.

    The observer is **passive** for correctness: lazy expiry in
    ``ConstraintTracker.is_gpu_feasible_for_task`` remains the fallback,
    so a failed observer only degrades wake latency (GPU stays
    infeasible for its NVML / operator-driven clear; no false positive).
    """

    def __init__(
        self,
        global_planner: Any,
        *,
        poll_interval_sec: float = 30.0,
        min_settle_time_sec: float = 30.0,
        max_quarantine_sec: float = 3600.0,
        force_clear_after_sec: Optional[float] = None,
    ) -> None:
        self._global_planner = global_planner
        self._poll_interval_sec = float(poll_interval_sec)
        self._min_settle_time_sec = float(min_settle_time_sec)
        self._max_quarantine_sec = float(max_quarantine_sec)
        self._force_clear_after_sec = force_clear_after_sec

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._nvml_available: Optional[bool] = None


    async def start(self) -> None:
        """Spawn the polling task.  Idempotent."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the polling task.  Idempotent."""
        self._running = False
        task = self._loop_task
        self._loop_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


    def on_xid_event(
        self, gpu_id: str, xid: int, *, source: str = "nvml_stats",
    ) -> None:
        """Plan — emit a structured
        ``gpu_unhealthy`` ConstraintViolation so the Planner's
        ConstraintTracker records the Xid quarantine.  Called by the
        supervisor stats loop / NVML polling path whenever an Xid
        event is observed.  The resulting violation carries the Xid
        code in ``cuda_xid`` so ``_resolve_xid_ttl`` can apply the
        correct TTL bucket.
        """
        from .contracts import ConstraintViolation, ViolationType, ViolationSource

        violation = ConstraintViolation(
            violation_type=ViolationType.GPU_UNHEALTHY,
            gpu_id=str(gpu_id),
            worker_name="",
            source=ViolationSource.EXTERNAL if source != "nvml_stats" else ViolationSource.SCHEDULER,
            cuda_xid=int(xid),
        )
        try:
            self._global_planner.incorporate_constraint(violation)
        except Exception as exc:
            _LOG.warning(
                "[gpu-health] on_xid_event for gpu=%s xid=%d failed: %s",
                gpu_id, xid, exc,
            )

    def admin_confirm_reset(self, gpu_id: str) -> None:
        """Operator signal: GPU reset completed (e.g., ``nvidia-smi -r``).

        Clears ``_gpu_reset_required`` but NOT ``_rma_qualifying_gpus`` —
        RMA candidates require a separate RMA confirmation because the
        hardware itself may still be defective even after a successful
        reset.
        """
        self._global_planner.on_reset_completed(str(gpu_id))
        _LOG.info(
            "[gpu-health] GPU %s reset confirmed (source=operator_reset)", gpu_id,
        )

    def on_preempted(
        self, gpu_id: str, *, preempt_until: Optional[float] = None,
    ) -> None:
        """Plan — emit ``preempted`` when
        a spot-VM revocation / SIGTERM / external orchestrator
        pre-empts the GPU.  ``preempt_until`` (optional wall-clock
        time) is recorded in ``memory_guard_details`` so the
        ConstraintTracker can set the exclusion expiry."""
        from .contracts import ConstraintViolation, ViolationType, ViolationSource

        details: Dict[str, Any] = {}
        if preempt_until is not None:
            details["preempt_until"] = float(preempt_until)
        violation = ConstraintViolation(
            violation_type=ViolationType.PREEMPTED,
            gpu_id=str(gpu_id),
            worker_name="",
            source=ViolationSource.EXTERNAL,
            memory_guard_details=details or None,
        )
        try:
            self._global_planner.incorporate_constraint(violation)
        except Exception as exc:
            _LOG.warning(
                "[gpu-health] on_preempted for gpu=%s failed: %s", gpu_id, exc,
            )

    def on_host_resource_exhausted(
        self, worker_name: str, *, detail: str = "",
    ) -> None:
        """Plan — emit
        ``host_resource_exhausted`` (disk full, RAM exhausted, fd
        leak).  Intended to be called by a host-level watchdog or
        the supervisor's container-creation path when OS-level
        exhaustion is observed.
        """
        from .contracts import ConstraintViolation, ViolationType, ViolationSource

        violation = ConstraintViolation(
            violation_type=ViolationType.HOST_RESOURCE_EXHAUSTED,
            gpu_id="",
            worker_name=str(worker_name),
            source=ViolationSource.EXTERNAL,
            memory_guard_details={"detail": detail} if detail else None,
        )
        try:
            self._global_planner.incorporate_constraint(violation)
        except Exception as exc:
            _LOG.warning(
                "[gpu-health] on_host_resource_exhausted for worker=%s failed: %s",
                worker_name, exc,
            )

    def on_stage_failed(
        self,
        *,
        plan: Any,
        progress_info: Dict[str, Any],
    ) -> None:
        """Plan — emit ``stage_failed``
        when the adapter surfaces partial-progress metadata (checkpoint
        stage completions).  The Planner's ConstraintTracker
        preserves ``progress_info`` so next ``plan()`` can resume."""
        from .contracts import ConstraintViolation, ViolationType, ViolationSource

        violation = ConstraintViolation(
            violation_type=ViolationType.STAGE_FAILED,
            gpu_id=getattr(plan, "target_gpu_id", ""),
            worker_name=getattr(plan, "target_worker_name", ""),
            source=ViolationSource.WORKER,
            progress_info=dict(progress_info or {}),
            failed_plan=plan,
        )
        try:
            self._global_planner.incorporate_constraint(violation)
        except Exception as exc:
            _LOG.warning(
                "[gpu-health] on_stage_failed for task=%s failed: %s",
                getattr(plan, "task_id", ""), exc,
            )

    def on_scheduler_internal_error(self, detail: str = "") -> None:
        """Plan — emit
        ``scheduler_internal_error`` on ConstraintTracker inconsistency
        / asyncio deadlock / lattice invariant violation.  Fires a
        P0 observability alert as side-effect."""
        from .contracts import ConstraintViolation, ViolationType, ViolationSource

        violation = ConstraintViolation(
            violation_type=ViolationType.SCHEDULER_INTERNAL_ERROR,
            gpu_id="",
            worker_name="",
            source=ViolationSource.SCHEDULER,
            memory_guard_details={"detail": detail} if detail else None,
        )
        try:
            self._global_planner.incorporate_constraint(violation)
        except Exception as exc:
            _LOG.error(
                "[gpu-health] on_scheduler_internal_error raised: %s", exc,
            )

    def admin_confirm_rma_complete(self, gpu_id: str) -> None:
        """Operator signal: RMA / hardware replacement / Field Diagnostic
        clean.  Clears both the reset-required flag and the
        RMA-qualifying set so the GPU becomes re-admissible."""
        self._global_planner.admin_confirm_rma_complete(str(gpu_id))
        _LOG.info(
            "[gpu-health] GPU %s RMA confirmed (source=operator_rma)", gpu_id,
        )


    async def _run(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval_sec)
                self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOG.error(
                    "[gpu-health] scan crashed: %s (continuing)", exc, exc_info=True,
                )

    def _scan_once(self) -> None:
        tracker = getattr(self._global_planner, "_constraint_tracker", None)
        if tracker is None:
            return
        now = time.time()
        pending = list(tracker._gpu_reset_required)
        for gpu_id in pending:
            requested_at = tracker._gpu_reset_requested_at.get(gpu_id, now)
            elapsed = now - requested_at

            if (
                elapsed >= self._min_settle_time_sec
                and elapsed <= self._max_quarantine_sec
            ):
                action = self._get_recovery_action_nvml(gpu_id)
                if action == NVML_GPU_RECOVERY_ACTION_NONE:
                    self._global_planner.on_reset_completed(gpu_id)
                    _LOG.info(
                        "[gpu-health] GPU %s reset completed (source=nvml, elapsed=%.0fs)",
                        gpu_id, elapsed,
                    )
                    continue

            if (
                self._force_clear_after_sec is not None
                and elapsed > self._force_clear_after_sec
            ):
                _LOG.warning(
                    "[gpu-health] GPU %s force-cleared after %.0fs (operator override; "
                    "NVML recovery action not verified)",
                    gpu_id, elapsed,
                )
                self._global_planner.on_reset_completed(gpu_id)

    def _get_recovery_action_nvml(self, gpu_id: str) -> int:
        """Query ``NVML_FI_DEV_GET_GPU_RECOVERY_ACTION`` (field ID 230).

        Returns the numeric recovery-action code on success, or a
        sentinel non-NONE value on failure so the auto-clear branch does
        NOT trigger.  Driver < r570 or missing pynvml ⇒ returns
        ``RECOVERY_ACTION_GPU_RESET_REQUIRED`` (keeps the GPU quarantined
        until operator confirmation).
        """
        if self._nvml_available is False:
            return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
        try:
            import pynvml
        except ImportError:
            self._nvml_available = False
            _LOG.debug(
                "[gpu-health] pynvml not installed — NVML auto-clear disabled",
            )
            return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
        try:
            if self._nvml_available is None:
                pynvml.nvmlInit()
                self._nvml_available = True
            try:
                index = int(gpu_id)
            except (TypeError, ValueError):
                return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                _LOG.warning(
                    '[silent-except] %s swallowed an exception; body=%s',
                    __name__, 'return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED', exc_info=True,
                )
                return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
            try:
                field_values = pynvml.nvmlDeviceGetFieldValues(
                    handle, [NVML_FI_DEV_GET_GPU_RECOVERY_ACTION],
                )
            except Exception:
                return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
            if not field_values:
                return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
            fv = field_values[0]
            if getattr(fv, "nvmlReturn", 0) != 0:
                return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
            return int(fv.value.ui)
        except Exception as exc:
            _LOG.warning(
                "[gpu-health] NVML query failed for gpu=%s: %s", gpu_id, exc,
            )
            return NVML_GPU_RECOVERY_ACTION_GPU_RESET_REQUIRED
