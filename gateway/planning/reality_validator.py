"""Reality Validator — checks if a DispatchPlan is feasible against live state.

Plan's Core Loop:
  Planner: plan(task) -> DispatchPlan
  Validator: validate_and_dispatch(plan)
      |-- feasible -> dispatch -> done
      +-- gap detected -> ConstraintViolation -> Planner re_plan

Internal modules:
  WorkerResolver     — plan's target_worker_name -> supervisor state lookup
  ReservationManager — dispatch-front backlog check + dispatch_pending reservation
  AdmissionGate      — VRAM / host-RAM feasibility pre-check
  WorkerActivator    — cold-start container creation if plan.needs_cold_start
  GapDetector        — final plan assumptions vs live state check
  Dispatcher         — build WorkerSelection from plan, query Signal, execute_plan (gRPC)
  EvictionCoordinator — classify dispatch errors into ConstraintViolation, eviction decisions
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
import time
from asyncio import CancelledError
from collections.abc import Mapping
from typing import Any

from .contracts import (
    ConstraintViolation,
    DispatchPlan,
    ViolationSource,
    ViolationType,
)

_LOG = logging.getLogger(__name__)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_numerical_failure(exc: BaseException) -> bool:
    """Plan — detect NaN / Inf / fp16 overflow
    exceptions via message content.  The adapter layer raises typed
    ArithmeticError subclasses; some third-party code raises generic
    ValueError/OverflowError with "NaN" / "Inf" / "overflow" in the
    message.  This helper accepts both shapes."""
    msg = str(exc).lower()
    return "nan" in msg or "inf" in msg or "overflow" in msg or "numerical" in msg


class RealityValidator:
    """Reality Validator — checks if a DispatchPlan is feasible against live state.

    Bridges the GlobalPlanner's proactive plan with the actual system state.
    Each validate_and_dispatch call runs the 7-module pipeline:
      WorkerResolver -> ReservationManager -> AdmissionGate -> WorkerActivator
      -> GapDetector -> Dispatcher (-> EvictionCoordinator on failure)

    Returns (result, None) on success, (None, violation) on gap detection.
    """

    GRPC_RETRY_DELAY_SEC: float = 0.5

    def __init__(
        self,
        gateway_service: Any,
        global_planner: Any,
        *,
        grpc_retry_delay_sec: float = 0.5,
    ) -> None:
        self._gateway = gateway_service
        self._global_planner = global_planner
        self._grpc_retry_delay_sec = _safe_float(grpc_retry_delay_sec, 0.5)

    def _cp_begin(self) -> int:
        begin = getattr(self._gateway, "_cp_begin", None)
        if not callable(begin):
            return 0
        return _safe_int(begin())

    def _cp_record(
        self,
        phase: str,
        start_ns: int,
        *,
        active_wall: bool = False,
        **attrs: Any,
    ) -> None:
        record = getattr(self._gateway, "_cp_record", None)
        if not callable(record) or not start_ns:
            return
        try:
            record(phase, start_ns, active_wall=active_wall, **attrs)
        except Exception:
            return

    def _force_full_wall_after_temporal_miss(
        self,
        plan: DispatchPlan,
        *,
        reason: str,
    ) -> None:
        """Fail missing or corrupt temporal evidence closed without retry changes."""
        timelines = self._global_planner.campaign_scheduler._timelines
        force_full_wall = getattr(timelines, "force_full_wall_for_task", None)
        if callable(force_full_wall) and force_full_wall(plan.task_id):
            _LOG.info(
                "[reality-validator] temporal fallback reason=%s task=%s "
                "now uses full-wall reservation",
                reason,
                plan.task_id,
            )

    @staticmethod
    def _log_temporal_replan(plan: DispatchPlan, *, reason: str) -> None:
        """Record a stale dispatch plan without mutating its temporal model."""
        _LOG.info(
            "[reality-validator] temporal replan reason=%s task=%s "
            "reservation preserved",
            reason,
            plan.task_id,
        )

    def _validation_interval(
        self,
        plan: DispatchPlan,
        *,
        config_fingerprint: str = "",
        pre_activation: bool = False,
    ) -> tuple[float, float, bool, bool, float]:
        """Return the tight actual VRAM interval and fail-closed state."""
        timelines = self._global_planner.campaign_scheduler._timelines
        now = time.time()
        planned_start = max(
            now,
            _safe_float(getattr(plan, "planned_start_time", 0.0)),
        )
        actual_start = now
        start = actual_start
        duration = max(
            0.0,
            _safe_float(plan.predicted_latency_sec) - max(0.0, planned_start - now),
        )
        find_entry = getattr(timelines, "find_entry", None)
        entry = find_entry(plan.task_id) if callable(find_entry) else None
        temporal_mode = (
            getattr(timelines, "_vram_reservation_model_name", "full_wall")
            == "temporal_peak_interval"
        )
        task_uses_full_wall = getattr(
            timelines,
            "task_uses_full_wall_reservation",
            None,
        )
        already_full_wall = callable(task_uses_full_wall) and task_uses_full_wall(
            plan.task_id
        )
        if not temporal_mode or already_full_wall:
            return start, duration, False, False, actual_start
        if entry is None:
            self._force_full_wall_after_temporal_miss(
                plan,
                reason="missing_temporal_entry",
            )
            return start, duration, False, True, actual_start
        identity_matches = bool(
            str(getattr(entry, "gpu_id", "")) == str(plan.target_gpu_id)
            and str(getattr(entry, "component", "")) == str(plan.component)
            and str(getattr(entry, "config_fingerprint", ""))
            == str(config_fingerprint or "")
        )
        if not identity_matches:
            self._log_temporal_replan(plan, reason="identity_mismatch")
            return start, duration, False, True, actual_start
        if getattr(entry, "is_completed", False) or not getattr(
            entry,
            "is_predicted",
            False,
        ):
            self._log_temporal_replan(plan, reason="entry_state_mismatch")
            return start, duration, False, True, actual_start
        planned_wall_duration = _safe_float(
            getattr(entry, "predicted_end_time", 0.0)
        ) - _safe_float(getattr(entry, "start_time", 0.0))
        if planned_wall_duration > 0.0:
            duration = planned_wall_duration
        temporal_valid = getattr(entry, "_has_valid_temporal_reservation", None)
        if (
            getattr(entry, "is_predicted", False)
            and callable(temporal_valid)
            and temporal_valid()
        ):
            if pre_activation:
                return (
                    _safe_float(getattr(entry, "start_time", actual_start)),
                    duration,
                    True,
                    False,
                    actual_start,
                )
            return actual_start, duration, False, False, actual_start
        temporal_fields = (
            getattr(entry, "allow_vram_mb", None),
            getattr(entry, "peak_start_time", None),
            getattr(entry, "peak_end_time", None),
        )
        if all(value is None for value in temporal_fields):
            self._force_full_wall_after_temporal_miss(
                plan,
                reason="no_temporal_evidence",
            )
            return start, duration, False, False, actual_start
        if any(value is None for value in temporal_fields):
            self._force_full_wall_after_temporal_miss(
                plan,
                reason="corrupt_temporal_fields",
            )
            return start, duration, False, True, actual_start
        temporal_geometry = getattr(
            entry,
            "_has_temporal_reservation_geometry",
            None,
        )
        if callable(temporal_geometry) and temporal_geometry():
            self._log_temporal_replan(plan, reason="timing_mismatch")
            return start, duration, False, True, actual_start
        self._force_full_wall_after_temporal_miss(
            plan,
            reason="corrupt_temporal_geometry",
        )
        return start, duration, False, True, actual_start


    async def validate_and_dispatch(
        self,
        plan: DispatchPlan,
        record: Any,
        payload: dict[str, Any],
        *,
        task_campaign_id: str | None = None,
        task_req: Any = None,
        campaign_hints: dict[str, Any] | None = None,
        normalized_workload_features: dict[str, Any] | None = None,
        normalized_execution_overrides: dict[str, Any] | None = None,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        timeout_s: int = 300,
        planner_intent: Any = None,
    ) -> tuple[Any | None, ConstraintViolation | None]:
        """Core validation + dispatch flow.

        Returns (result, None) on success, (None, violation) on gap detection.
        """

        global_planner = self._global_planner or getattr(
            getattr(self._gateway, "_planner", None),
            "_global_planner",
            None,
        )
        reciprocal_payload: dict[str, Any] | None = None
        if global_planner is not None and bool(
            getattr(global_planner, "reciprocal_interference_correction", False)
        ):
            raw_reciprocal = (getattr(plan, "worker_metadata", {}) or {}).get(
                "reciprocal_interference"
            )
            try:
                if not isinstance(raw_reciprocal, Mapping) or not raw_reciprocal:
                    raise ValueError("reciprocal handoff metadata missing")
                reciprocal_payload = dict(raw_reciprocal)
                global_planner.validate_reciprocal_dispatch_reservation(
                    plan.task_id,
                    plan.component,
                    plan.target_gpu_id,
                    reciprocal_payload,
                )
            except Exception as exc:
                _LOG.error(
                    "[reality-validator] reciprocal preflight rejected task=%s: %s",
                    plan.task_id,
                    exc,
                )
                return None, ConstraintViolation(
                    violation_type=ViolationType.SCHEDULER_INTERNAL_ERROR,
                    gpu_id=plan.target_gpu_id,
                    worker_name=plan.target_worker_name,
                    source=ViolationSource.SCHEDULER,
                    plan_snapshot_id=plan.plan_snapshot_id,
                    failed_plan=plan,
                )

        _cp_start = self._cp_begin()
        worker_state, violation = self._resolve_worker(plan)
        self._cp_record(
            "dispatch_worker_resolve",
            _cp_start,
            active_wall=True,
            component=plan.component,
            gpu_id=plan.target_gpu_id,
        )
        if violation:
            return None, violation

        _cp_start = self._cp_begin()
        reservation, violation = await self._acquire_front_checked_reservation(
            plan,
            worker_state,
        )
        self._cp_record(
            "dispatch_reservation",
            _cp_start,
            active_wall=True,
            component=plan.component,
            gpu_id=plan.target_gpu_id,
            result="violation" if violation else "reserved",
        )
        if violation:
            return None, violation

        try:
            _cp_start = self._cp_begin()
            violation = await self._check_admission(
                plan,
                worker_state,
                config_fingerprint=config_fingerprint,
            )
            self._cp_record(
                "dispatch_admission_check",
                _cp_start,
                active_wall=True,
                component=plan.component,
                gpu_id=plan.target_gpu_id,
                result="violation" if violation else "ok",
            )
            if violation:
                self._release_reservation(reservation)
                return None, violation

            intrinsic_signal_for_activator = None
            if task_req is not None:
                _cp_start = self._cp_begin()
                intrinsic_signal_for_activator = self._query_signal(plan, task_req)
                self._cp_record(
                    "dispatch_activation_signal_query",
                    _cp_start,
                    active_wall=True,
                    component=plan.component,
                    gpu_id=plan.target_gpu_id,
                )
            _cp_start = self._cp_begin()
            activated_worker, was_cold_start, violation = await self._activate_worker(
                plan,
                worker_state,
                record,
                reservation,
                intrinsic_signal=intrinsic_signal_for_activator,
                planner_intent=planner_intent,
                campaign_hints=campaign_hints,
            )
            self._cp_record(
                "dispatch_worker_activation",
                _cp_start,
                active_wall=False,
                component=plan.component,
                gpu_id=plan.target_gpu_id,
                result="violation" if violation else "ok",
                was_cold_start=bool(was_cold_start),
            )
            if violation:
                self._release_reservation(reservation)
                return None, violation

            _cp_start = self._cp_begin()
            violation = self._detect_gap(
                plan,
                config_fingerprint=config_fingerprint,
            )
            self._cp_record(
                "dispatch_gap_detect",
                _cp_start,
                active_wall=True,
                component=plan.component,
                gpu_id=plan.target_gpu_id,
                result="violation" if violation else "ok",
            )
            if violation:
                self._release_reservation(reservation)
                return None, violation
        except BaseException:
            self._release_reservation(reservation)
            raise

        dispatched_ok = False
        try:
            result = await self._dispatch(
                plan,
                worker_state,
                record,
                payload,
                task_campaign_id=task_campaign_id,
                task_req=task_req,
                campaign_hints=campaign_hints,
                normalized_workload_features=normalized_workload_features,
                normalized_execution_overrides=normalized_execution_overrides,
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                timeout_s=timeout_s,
                planner_intent=planner_intent,
                reservation=reservation,
                activated_worker=activated_worker,
                was_cold_start=was_cold_start,
                reciprocal_interference=reciprocal_payload,
            )
            dispatched_ok = True
            self._clear_post_dispatch_markers(
                plan,
                activated_worker,
            )
            return result, None
        except BaseException:
            return self._handle_dispatch_failure(
                sys.exc_info()[1],
                plan,
                reservation,
            )
        finally:
            if not dispatched_ok:
                self._release_reservation(reservation)

    def _clear_post_dispatch_markers(
        self,
        plan: DispatchPlan,
        activated_worker: Any,
    ) -> None:
        """Shared successful-dispatch recovery, used by both admission paths."""
        tracker = self._get_constraint_tracker()
        if tracker is None:
            return
        activated_name = (
            str(getattr(activated_worker, "worker_name", "") or "").strip()
            if activated_worker is not None
            else ""
        )
        names_to_clear = {
            name for name in (plan.target_worker_name, activated_name) if name
        }
        if (
            activated_name
            and plan.target_worker_name
            and activated_name != plan.target_worker_name
        ):
            _LOG.warning(
                "[marker-clear] canonical name mismatch plan=%s "
                "dispatched=%s — clearing both",
                plan.target_worker_name,
                activated_name,
            )
        for name in names_to_clear:
            tracker.reset_transient_grpc_failures(name)
            if hasattr(tracker, "clear_worker_cold_marker"):
                tracker.clear_worker_cold_marker(name)
            clear_unreachable = getattr(tracker, "clear_worker_unreachable", None)
            if callable(clear_unreachable):
                clear_unreachable(name)
        clear_activation = getattr(tracker, "clear_activation_exclusion", None)
        if callable(clear_activation):
            clear_activation(plan.component, plan.target_gpu_id)

    def _handle_dispatch_failure(
        self,
        error: BaseException | None,
        plan: DispatchPlan,
        reservation: Any,
    ) -> tuple[None, ConstraintViolation]:
        if isinstance(error, CancelledError):
            self._release_reservation(reservation)
            raise error
        if not isinstance(error, Exception):
            self._release_reservation(reservation)
            raise error or RuntimeError("dispatch failed without an exception")

        from ..http_server import _DispatchRetrySignal

        classified_error: Exception = error
        if isinstance(error, _DispatchRetrySignal):
            if error.constraint_violation:
                cv = error.constraint_violation
                if not cv.correlation_id and error.failed_worker_name:
                    cv.correlation_id = (
                        f"{error.failed_worker_name}:{_safe_int(time.time())}"
                    )
                if not cv.plan_snapshot_id:
                    cv.plan_snapshot_id = plan.plan_snapshot_id
                if cv.failed_plan is None:
                    cv.failed_plan = plan
                return None, cv
            cause = getattr(error, "cause", None)
            if isinstance(cause, Exception):
                classified_error = cause
        return None, self._classify_error(classified_error, plan)


    def _detect_gap(
        self,
        plan: DispatchPlan,
        *,
        config_fingerprint: str = "",
    ) -> ConstraintViolation | None:
        """Compare plan's ConstraintAssumptions vs live state.

        Checks:
        - VRAM gap: assumed_available_vram_mb vs timeline available_vram_at()
        - Planned occupancy gap: assumed_gpu_active_count vs timeline
          planned occupancy entries
        - Worker ready gap: assumed_worker_ready vs worker.ready
        """
        assumptions = plan.constraint_assumptions
        if not assumptions:
            return None

        cs = self._global_planner.campaign_scheduler
        tl = cs._timelines.get(plan.target_gpu_id)
        if not tl:
            return None

        actual_vram = tl.available_vram_at(exclude_task_id=plan.task_id)
        temporal_mode = (
            getattr(cs._timelines, "_vram_reservation_model_name", "full_wall")
            == "temporal_peak_interval"
        )
        temporal_gap = False
        planned_start = 0.0
        duration = max(0.0, _safe_float(plan.predicted_latency_sec))
        actual_start = max(
            time.time(),
            _safe_float(getattr(plan, "planned_start_time", 0.0)),
        )
        interval_fit = getattr(cs._timelines, "candidate_interval_fits", None)
        if temporal_mode and callable(interval_fit) and plan.vram_budget_mb > 0:
            (
                planned_start,
                duration,
                use_planned_envelope,
                envelope_miss,
                actual_start,
            ) = self._validation_interval(
                plan,
                config_fingerprint=config_fingerprint,
            )
            temporal_gap = envelope_miss or not bool(
                interval_fit(
                    plan.target_gpu_id,
                    planned_start,
                    planned_start + duration,
                    plan.vram_budget_mb,
                    0.0,
                    exclude_task_id=plan.task_id,
                    candidate_component=plan.component,
                    candidate_config_fingerprint=config_fingerprint,
                    candidate_use_planned_envelope=use_planned_envelope,
                    candidate_peak_margin_sec=0.0,
                )
            )
        if temporal_gap or (
            not temporal_mode
            and assumptions.assumed_available_vram_mb > 0
            and plan.vram_budget_mb > 0
            and actual_vram < plan.vram_budget_mb
        ):
            _LOG.info(
                "[gap-detector] VRAM gap on GPU %s: assumed=%.0f actual=%.0f needed=%d",
                plan.target_gpu_id,
                assumptions.assumed_available_vram_mb,
                actual_vram,
                plan.vram_budget_mb,
            )
            return ConstraintViolation(
                violation_type="vram_insufficient",
                gpu_id=plan.target_gpu_id,
                worker_name=plan.target_worker_name,
                requested_vram_mb=plan.vram_budget_mb,
                available_vram_mb=_safe_int(actual_vram),
                failed_plan=plan,
            )

        actual_ram = cs._timelines.available_host_ram_at(
            exclude_task_id=plan.task_id,
        )
        ram_budget_mb = _safe_int(getattr(plan, "ram_budget_mb", 0))
        temporal_ram_gap = bool(
            temporal_mode
            and callable(interval_fit)
            and ram_budget_mb > 0
            and not interval_fit(
                plan.target_gpu_id,
                actual_start,
                actual_start + duration,
                0.0,
                ram_budget_mb,
                exclude_task_id=plan.task_id,
            )
        )
        if temporal_ram_gap or (
            getattr(assumptions, "assumed_available_host_ram_mb", 0.0) > 0
            and ram_budget_mb > 0
            and actual_ram < ram_budget_mb
        ):
            _LOG.info(
                "[gap-detector] host RAM gap: assumed=%.0f actual=%.0f needed=%d",
                assumptions.assumed_available_host_ram_mb,
                actual_ram,
                plan.ram_budget_mb,
            )
            return ConstraintViolation(
                violation_type="host_ram_saturated",
                gpu_id=plan.target_gpu_id,
                worker_name=plan.target_worker_name,
                host_mem_available_mib=_safe_int(actual_ram),
                host_mem_threshold_mib=_safe_int(plan.ram_budget_mb),
                failed_plan=plan,
            )

        actual_active = (
            tl.planned_occupancy_count()
            if hasattr(tl, "planned_occupancy_count")
            else sum(1 for e in tl.active_entries)
        )
        if (
            assumptions.assumed_gpu_active_count >= 0
            and actual_active > assumptions.assumed_gpu_active_count + 1
        ):
            _LOG.info(
                "[gap-detector] Planned occupancy gap on GPU %s: assumed=%d actual=%d",
                plan.target_gpu_id,
                assumptions.assumed_gpu_active_count,
                actual_active,
            )

        if assumptions.assumed_worker_ready and not plan.needs_cold_start:
            sup = getattr(cs, "_supervisor", None)
            if sup:
                st = self._lookup_worker_state_by_plan(plan)
                if st and not (st.ready and st.addr):
                    _LOG.info(
                        "[gap-detector] Worker not ready: %s (assumed ready)",
                        plan.target_worker_name,
                    )
                    return ConstraintViolation(
                        violation_type="worker_not_ready",
                        gpu_id=plan.target_gpu_id,
                        worker_name=plan.target_worker_name,
                        failed_plan=plan,
                    )


        return None


    def _resolve_worker(
        self,
        plan: DispatchPlan,
    ) -> tuple[Any | None, ConstraintViolation | None]:
        """Verify plan's target worker via supervisor state lookup.

        Returns (worker_state, None) on success, (None, violation) on failure.
        worker_state is the supervisor WorkerState or a lightweight proxy.
        """
        cs = self._global_planner.campaign_scheduler
        sup = getattr(cs, "_supervisor", None)
        if not sup:
            return None, None

        st = self._lookup_worker_state_by_plan(plan)
        if st is None:
            if plan.needs_cold_start:
                return None, None
            return None, ConstraintViolation(
                violation_type="worker_not_ready",
                gpu_id=plan.target_gpu_id,
                worker_name=plan.target_worker_name,
                failed_plan=plan,
            )

        lifecycle = getattr(st, "lifecycle_state", "")
        if lifecycle == "killed":
            return None, ConstraintViolation(
                violation_type="worker_dead",
                gpu_id=plan.target_gpu_id,
                worker_name=plan.target_worker_name,
                failed_plan=plan,
            )

        return st, None


    async def _check_admission(
        self,
        plan: DispatchPlan,
        worker_state: Any,
        *,
        config_fingerprint: str = "",
    ) -> ConstraintViolation | None:
        """Plan step 3 — AdmissionGate.

        Fail-fast pre-check BEFORE WorkerActivator pays the cold-start
        cost.  Verifies:

          1. VRAM feasibility — projected VRAM at now ≥ plan.vram_budget_mb.
             Mirrors the ``is_gpu_feasible_for_task`` step 5 check but
             fires earlier so cold-start work is skipped on obviously
             blocked GPUs.  Fires ``vram_insufficient``.

        Plan fix: active-count hard-gate (previously Gate 2
        based on ``slowdown_fallback_max_active``) removed.  Interference
        co-location  Planner D2 EFT   (``Σ (sd − 1) × overlap``
        + cold-start fallback ``slowdown_cold_start_default``, Roofline
        upper bound)   .  Validator   active-count
          , physical saturation  Planner  ordering
          deprioritize .  GPU health (NVML
        unhealthy / quarantine / RMA)  ``ConstraintTracker.is_gpu_feasible_for_task``
         Planner   Validator  VRAM 
        pre-check.
        """
        cs = self._global_planner.campaign_scheduler
        tl = cs._timelines.get(plan.target_gpu_id)
        if tl is None:
            return None
        interval_fit = getattr(cs._timelines, "candidate_interval_fits", None)
        validation_actual_start = max(
            time.time(),
            _safe_float(getattr(plan, "planned_start_time", 0.0)),
        )
        validation_duration = max(0.0, _safe_float(plan.predicted_latency_sec))

        if plan.vram_budget_mb > 0:
            if callable(interval_fit):
                (
                    planned_start,
                    duration,
                    use_planned_envelope,
                    envelope_miss,
                    validation_actual_start,
                ) = self._validation_interval(
                    plan,
                    config_fingerprint=config_fingerprint,
                    pre_activation=bool(
                        plan.needs_cold_start
                        or not getattr(worker_state, "ready", False)
                        or not getattr(worker_state, "addr", "")
                    ),
                )
                validation_duration = duration
                if envelope_miss or not interval_fit(
                    plan.target_gpu_id,
                    planned_start,
                    planned_start + duration,
                    plan.vram_budget_mb,
                    0.0,
                    exclude_task_id=plan.task_id,
                    candidate_component=plan.component,
                    candidate_config_fingerprint=config_fingerprint,
                    candidate_use_planned_envelope=use_planned_envelope,
                    candidate_peak_margin_sec=0.0,
                ):
                    avail_mb_int = _safe_int(
                        tl.available_vram_at(exclude_task_id=plan.task_id)
                    )
                    return ConstraintViolation(
                        violation_type="vram_insufficient",
                        gpu_id=plan.target_gpu_id,
                        worker_name=plan.target_worker_name,
                        requested_vram_mb=_safe_int(plan.vram_budget_mb),
                        available_vram_mb=avail_mb_int,
                        failed_plan=plan,
                    )
            else:
                available = getattr(tl, "available_vram_at", None)
                avail_mb = (
                    available(exclude_task_id=plan.task_id)
                    if callable(available)
                    else None
                )
                avail_mb_int = _safe_int(avail_mb, -1)
                if avail_mb is not None and avail_mb_int < plan.vram_budget_mb:
                    return ConstraintViolation(
                        violation_type="vram_insufficient",
                        gpu_id=plan.target_gpu_id,
                        worker_name=plan.target_worker_name,
                        requested_vram_mb=_safe_int(plan.vram_budget_mb),
                        available_vram_mb=avail_mb_int,
                        failed_plan=plan,
                    )

        ram_budget_mb = _safe_int(getattr(plan, "ram_budget_mb", 0))
        if ram_budget_mb > 0:
            avail_ram = _safe_int(
                cs._timelines.available_host_ram_at(
                    exclude_task_id=plan.task_id,
                )
            )
            temporal_ram_gap = bool(
                getattr(
                    cs._timelines,
                    "_vram_reservation_model_name",
                    "full_wall",
                )
                == "temporal_peak_interval"
                and callable(interval_fit)
                and not interval_fit(
                    plan.target_gpu_id,
                    validation_actual_start,
                    validation_actual_start + validation_duration,
                    0.0,
                    ram_budget_mb,
                    exclude_task_id=plan.task_id,
                )
            )
            if temporal_ram_gap or avail_ram < ram_budget_mb:
                return ConstraintViolation(
                    violation_type="host_ram_saturated",
                    gpu_id=plan.target_gpu_id,
                    worker_name=plan.target_worker_name,
                    host_mem_available_mib=avail_ram,
                    host_mem_threshold_mib=ram_budget_mb,
                    failed_plan=plan,
                )

        return None


    async def _activate_worker(
        self,
        plan: DispatchPlan,
        worker_state: Any,
        record: Any,
        reservation: Any,
        *,
        intrinsic_signal: Any = None,
        planner_intent: Any = None,
        campaign_hints: dict[str, Any] | None = None,
    ) -> tuple[Any, bool, ConstraintViolation | None]:
        """Plan fix (e) — Stage 4 full cold-start lifecycle
        owner.  Returns (activated_worker, was_cold_start, violation).
        ``was_cold_start=True`` when this invocation actually performed
        cold-start activation (worker was not ready on entry or
        plan.needs_cold_start=True) — used downstream by
        ``on_task_dispatched`` to render the init-phase entry in the
        timeline (Bug 1 fix: without propagating this, dispatch.py
        sees ``worker.ready=True`` post-Stage-4 and incorrectly flags
        the task as warm).
        """
        worker = self._build_worker_selection(plan, worker_state)
        was_cold_start = False

        if not bool(worker.ready) or plan.needs_cold_start:
            activate_fn = getattr(self._gateway, "_activate_selected_worker", None)
            if activate_fn is None:
                return (
                    None,
                    False,
                    ConstraintViolation(
                        violation_type="activation_failed",
                        gpu_id=plan.target_gpu_id,
                        worker_name=plan.target_worker_name,
                        failed_plan=plan,
                    ),
                )
            try:
                worker, did_activate = await activate_fn(
                    record=record,
                    worker=worker,
                    reservation=reservation,
                    preferred_worker_addr=plan.target_worker_addr,
                    preferred_gpu_ids=[plan.target_gpu_id],
                    campaign_id=plan.campaign_id,
                    intrinsic_signal=intrinsic_signal,
                    planner_intent=planner_intent,
                )
                was_cold_start = bool(did_activate)
            except Exception as exc:
                if "dispatch-front reservation denied" in str(exc):
                    return (
                        None,
                        False,
                        ConstraintViolation(
                            violation_type="worker_queue_saturated",
                            gpu_id=plan.target_gpu_id,
                            worker_name=plan.target_worker_name,
                            active_count=1,
                            failed_plan=plan,
                        ),
                    )
                _LOG.warning(
                    "[worker-activator] Cold-start failed for %s on GPU %s: %s",
                    plan.target_worker_name,
                    plan.target_gpu_id,
                    exc,
                )
                return (
                    None,
                    False,
                    ConstraintViolation(
                        violation_type="activation_failed",
                        gpu_id=plan.target_gpu_id,
                        worker_name=plan.target_worker_name,
                        failed_plan=plan,
                    ),
                )

        stage_fn = getattr(self._gateway, "_stage_selected_worker", None)
        if stage_fn is not None:
            try:
                hints = campaign_hints or {}
                selected_context_applied = (
                    "post_selection_retry" if hints.get("reasons") else None
                )
                await stage_fn(
                    record,
                    worker,
                    preferred_gpu_ids=[plan.target_gpu_id],
                    selected_context_applied=selected_context_applied,
                )
            except Exception:
                _LOG.warning(
                    "[worker-activator] _stage_selected_worker failed",
                    exc_info=True,
                )

        return worker, was_cold_start, None


    def _dispatch_front_lock_key(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> str:
        if worker_state is None:
            worker_state = self._lookup_worker_state_by_plan(plan)
        worker_name = str(plan.target_worker_name or "").strip()
        if not worker_name and worker_state is not None:
            spec = getattr(worker_state, "spec", None)
            worker_name = str(getattr(spec, "name", "") or "").strip()
        worker_addr = str(plan.target_worker_addr or "").strip()
        if not worker_addr and worker_state is not None:
            worker_addr = str(getattr(worker_state, "addr", "") or "").strip()
        return worker_addr or worker_name or f"{plan.component}:{plan.target_gpu_id}"

    def _dispatch_front_lock(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> asyncio.Lock:
        locks = getattr(self._gateway, "_dispatch_front_locks", None)
        if locks is None:
            locks = {}
            self._gateway._dispatch_front_locks = locks
        key = self._dispatch_front_lock_key(plan, worker_state)
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock

    def _lookup_worker_state_by_plan(self, plan: DispatchPlan) -> Any:
        sup = getattr(self._gateway, "supervisor", None)
        states = getattr(sup, "states", {}) if sup is not None else {}
        worker_name = str(plan.target_worker_name or "").strip()
        if worker_name and worker_name in states:
            return states.get(worker_name)
        worker_addr = str(plan.target_worker_addr or "").strip()
        if worker_addr:
            for st in states.values():
                if str(getattr(st, "addr", "") or "").strip() == worker_addr:
                    return st
        component = str(plan.component or "").strip().lower()
        gpu_id = str(plan.target_gpu_id or "").strip()
        if component and gpu_id:
            for st in states.values():
                spec = getattr(st, "spec", None)
                spec_component = (
                    str(getattr(spec, "component", "") or "").strip().lower()
                )
                spec_gpus = [
                    str(g).strip() for g in list(getattr(spec, "gpus", []) or [])
                ]
                if spec_component == component and gpu_id in spec_gpus:
                    return st
        return None

    def _resolved_worker_name(self, plan: DispatchPlan, worker_state: Any) -> str:
        worker_name = str(plan.target_worker_name or "").strip()
        if worker_name:
            return worker_name
        if worker_state is None:
            worker_state = self._lookup_worker_state_by_plan(plan)
        if worker_state is not None:
            spec = getattr(worker_state, "spec", None)
            worker_name = str(getattr(spec, "name", "") or "").strip()
            if worker_name:
                return worker_name
        return ""

    def _pre_reservation_dispatch_front_violation(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> ConstraintViolation | None:
        """Check hidden worker-front backlog before taking this task's reservation."""
        from .contracts import ViolationType

        if worker_state is None:
            worker_state = self._lookup_worker_state_by_plan(plan)

        cap = _safe_int(getattr(self._gateway, "dispatch_backlog_per_worker", 1))
        if cap <= 0:
            return None
        backlog_fn = getattr(self._gateway, "_dispatch_front_backlog", None)
        backlog = _safe_int(backlog_fn(worker_state) if callable(backlog_fn) else 0)
        lease_count_fn = getattr(
            self._gateway, "_front_slot_lease_count_for_plan", None
        )
        if callable(lease_count_fn):
            backlog += _safe_int(
                lease_count_fn(
                    plan,
                    worker_state,
                    exclude_task_id=str(plan.task_id or ""),
                )
            )
        if backlog >= cap:
            _LOG.debug(
                "[dispatch-front] worker_queue_saturated pre-reservation "
                "worker=%s gpu=%s hidden_backlog=%d cap=%d task=%s",
                self._resolved_worker_name(plan, worker_state)
                or plan.target_worker_addr,
                plan.target_gpu_id,
                backlog,
                cap,
                plan.task_id,
            )
            return ConstraintViolation(
                violation_type=ViolationType.WORKER_QUEUE_SATURATED,
                gpu_id=str(plan.target_gpu_id or ""),
                worker_name=self._resolved_worker_name(plan, worker_state),
                active_count=backlog,
                failed_plan=plan,
            )
        return None

    async def _acquire_front_checked_reservation(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> tuple[Any, ConstraintViolation | None]:
        """Atomically check dispatch-front backlog and reserve a worker slot."""
        lock = self._dispatch_front_lock(plan, worker_state)
        async with lock:
            release_lease = getattr(
                self._gateway,
                "_release_front_slot_lease_for_task",
                None,
            )
            try:
                violation = self._pre_reservation_dispatch_front_violation(
                    plan,
                    worker_state,
                )
                if violation:
                    if callable(release_lease):
                        release_lease(str(plan.task_id or ""))
                    return None, violation
                reservation = self._acquire_reservation(plan, worker_state)
                if callable(release_lease):
                    release_lease(str(plan.task_id or ""))
                return reservation, None
            except RuntimeError:
                return self._handle_reservation_failure(
                    plan,
                    worker_state,
                    release_lease,
                    sys.exc_info()[1],
                )

    def _handle_reservation_failure(
        self,
        plan: DispatchPlan,
        worker_state: Any,
        release_lease: Any,
        error: BaseException | None,
    ) -> tuple[Any, ConstraintViolation | None]:
        if callable(release_lease):
            release_lease(str(plan.task_id or ""))
        if not isinstance(error, RuntimeError):
            raise error or RuntimeError("reservation failed without an exception")
        if "dispatch-front reservation denied" not in str(error):
            raise error
        from .contracts import ViolationType

        return None, ConstraintViolation(
            violation_type=ViolationType.WORKER_QUEUE_SATURATED,
            gpu_id=str(plan.target_gpu_id or ""),
            worker_name=self._resolved_worker_name(plan, worker_state),
            active_count=1,
            failed_plan=plan,
        )

    def _release_reservation(self, reservation: Any) -> None:
        """Release a pre-acquired reservation on the error path.

        Mirrors plan  — try/finally reservation cleanup
        around cold-start / admission / gap-detection failures.  Safe to
        call with ``None`` or a reservation object that already released.
        """
        if reservation is None:
            return
        release = getattr(reservation, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                _LOG.warning(
                    "[reality-validator] reservation.release failed",
                    exc_info=True,
                )

    def _acquire_reservation(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> Any:
        """Acquire dispatch_pending reservation.

        Wraps the _AdmissionReservation lifecycle — bump counter on acquire,
        release on failure. Uses the gateway's existing reservation mechanism.
        """
        from ..http_server import _AdmissionReservation

        bump_fn = getattr(self._gateway, "_bump_dispatch_pending", None)
        release_fn = getattr(self._gateway, "_release_dispatch_pending", None)
        adjust_fn = getattr(self._gateway, "_adjust_worker_inflight_by_name", None)

        if worker_state is None:
            worker_state = self._lookup_worker_state_by_plan(plan)
        worker_name = self._resolved_worker_name(plan, worker_state)
        worker_addr = plan.target_worker_addr or (
            getattr(worker_state, "addr", None) if worker_state else None
        )

        if worker_addr:
            return _AdmissionReservation.reserve_ready(
                worker_name=worker_name,
                worker_addr=worker_addr,
                adjust_worker_inflight_by_name=adjust_fn,
                bump_dispatch_pending=bump_fn,
                release_dispatch_pending=release_fn,
            )
        else:
            return _AdmissionReservation.reserve_cold(
                worker_name=worker_name,
                adjust_worker_inflight_by_name=adjust_fn,
                bump_dispatch_pending=bump_fn,
                release_dispatch_pending=release_fn,
            )


    async def _dispatch(
        self,
        plan: DispatchPlan,
        worker_state: Any,
        record: Any,
        payload: dict[str, Any],
        *,
        task_campaign_id: str | None = None,
        task_req: Any = None,
        campaign_hints: dict[str, Any] | None = None,
        normalized_workload_features: dict[str, Any] | None = None,
        normalized_execution_overrides: dict[str, Any] | None = None,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        timeout_s: int = 300,
        planner_intent: Any = None,
        reservation: Any = None,
        activated_worker: Any = None,
        was_cold_start: bool = False,
        reciprocal_interference: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch via WorkerSelection built from plan + supervisor state.

        Core Loop (plan line 298-307):
          Planner: plan(task) → DispatchPlan
          Validator: validate_and_dispatch(plan) → dispatch via execute_plan

        Builds WorkerSelection directly from the DispatchPlan's target
        worker info and supervisor state.  The Planner has already made
        the scheduling decision; the Validator resolves and dispatches.
        """
        from .planner import PlannerTaskRequest

        payload = dict(payload)

        if activated_worker is not None:
            worker = activated_worker
        else:
            worker = self._build_worker_selection(plan, worker_state)
        self._attach_plan_metadata_to_worker(worker, plan)

        if reservation is None:
            reservation = self._acquire_reservation(plan, worker_state)

        if task_req is None:
            task_req = PlannerTaskRequest(
                component=plan.component,
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                workload_features=dict(normalized_workload_features or {}),
                execution_overrides=dict(normalized_execution_overrides or {}),
                campaign_id=task_campaign_id or plan.campaign_id,
                preferred_gpu_ids=[plan.target_gpu_id] if plan.target_gpu_id else [],
                scheduling_hints=campaign_hints or {},
                timeout_s=timeout_s,
            )
        else:
            task_req = dataclasses.replace(
                task_req,
                workload_features=dict(task_req.workload_features),
                execution_overrides=dict(task_req.execution_overrides),
                scheduling_hints=dict(task_req.scheduling_hints),
                preferred_gpu_ids=(
                    [plan.target_gpu_id]
                    if plan.target_gpu_id
                    else list(task_req.preferred_gpu_ids or [])
                ),
            )

        intrinsic_signal = self._query_signal(plan, task_req)

        dispatcher = getattr(self._gateway, "_dispatcher", None)
        if dispatcher is None:
            reservation.release()
            raise RuntimeError("DispatchService not available on gateway")

        try:
            dynamic_batch_context = dict(
                (plan.worker_metadata or {}).get("dynamic_batch") or {}
            )
            execution_profile = (plan.worker_metadata or {}).get(
                "dynamic_batch_profile"
            )
            if isinstance(execution_profile, dict):
                dynamic_batch_context["execution_profile"] = dict(execution_profile)
            resp, evaluation = await dispatcher.execute_plan(
                request=task_req,
                record=record,
                worker=worker,
                reservation=reservation,
                intrinsic_signal=intrinsic_signal,
                payload=payload,
                was_cold_start=was_cold_start,
                reciprocal_interference=reciprocal_interference,
                dynamic_batch_context=dynamic_batch_context,
            )
            return resp
        except BaseException:
            reservation.release()
            raise

    def _attach_plan_metadata_to_worker(self, worker: Any, plan: DispatchPlan) -> None:
        scheduler_metadata = {
            key: dict(value)
            for key in ("phase_scheduler", "heft_scheduler")
            if isinstance(value := (plan.worker_metadata or {}).get(key), dict)
        }
        if not scheduler_metadata:
            return
        placement = getattr(worker, "placement", None)
        if placement is None:
            return
        intent = dict(getattr(placement, "planner_intent", None) or {})
        intent.update(scheduler_metadata)
        object.__setattr__(placement, "planner_intent", intent)

    def _build_worker_selection(
        self,
        plan: DispatchPlan,
        worker_state: Any,
    ) -> Any:
        """Build WorkerSelection from DispatchPlan + supervisor state.

        WorkerResolver output: translates plan's target_worker_name/addr
        into a concrete WorkerSelection that execute_plan can consume.
        """
        from .scheduler import WorkerSelection

        addr = plan.target_worker_addr or ""
        ready = not plan.needs_cold_start
        gpu_ids = [plan.target_gpu_id] if plan.target_gpu_id else []
        max_concurrency = 1
        priority = 100

        if worker_state is not None:
            addr = addr or str(getattr(worker_state, "addr", "") or "")
            ready = bool(getattr(worker_state, "ready", False)) and bool(addr)
            assigned_gpus = getattr(worker_state, "assigned_gpus", None)
            if assigned_gpus:
                gpu_ids = [str(g) for g in assigned_gpus]
            state_max = _safe_int(getattr(worker_state, "max_concurrency", 0))
            if state_max > 0:
                max_concurrency = state_max
            else:
                max_concurrency = max(
                    1,
                    _safe_int(
                        getattr(self._gateway, "max_inflight_per_worker", 1),
                        1,
                    ),
                )

        meta = plan.worker_metadata or {}
        return WorkerSelection.from_mapping(
            {
                "addr": addr,
                "ready": ready,
                "gpu_ids": gpu_ids,
                "worker_name": plan.target_worker_name,
                "max_concurrency": max_concurrency,
                "priority": priority,
                "resident_baseline_snapshot": meta.get(
                    "resident_baseline_snapshot", {}
                ),
                "estimator_worker_context": meta.get("estimator_worker_context", {}),
            }
        )

    def _query_signal(self, plan: DispatchPlan, task_req: Any) -> Any:
        """Query Signal service for intrinsic data.

        Returns SignalResult used by execute_plan for cold-start context
        and predictive memory admission.  Falls back to a minimal stub
        if Signal is unavailable.
        """
        signal_service = getattr(self._gateway, "_signal_service", None)
        if signal_service is not None:
            try:
                from ..signals.contracts import PlannerIntent

                return signal_service.query(
                    component=plan.component,
                    config_fingerprint=task_req.config_fingerprint,
                    input_fingerprint=task_req.input_fingerprint,
                    workload_features=task_req.workload_features,
                    execution_overrides=task_req.execution_overrides,
                    campaign_id=plan.campaign_id or "",
                    planner_intent=PlannerIntent(),
                )
            except Exception:
                _LOG.warning("[dispatcher] Signal query failed, using stub")

        return self._build_stub_signal(plan)

    @staticmethod
    def _build_stub_signal(plan: DispatchPlan) -> Any:
        """Build minimal SignalResult stub when Signal is unavailable."""
        from ..signals.contracts import (
            MemorySignal,
            PlannerIntent,
            RuntimeSignal,
            SignalArtifacts,
            SignalBundle,
            SignalProvenance,
            SignalResult,
        )

        return SignalResult(
            bundle=SignalBundle(
                runtime=RuntimeSignal(
                    estimate_sec=plan.predicted_latency_sec,
                    upper_sec=(
                        plan.predicted_latency_sec * 1.5
                        if plan.predicted_latency_sec
                        else None
                    ),
                    guard_margin_sec=None,
                ),
                memory=MemorySignal(
                    active_estimate_mib=_safe_float(plan.vram_budget_mb)
                    if plan.vram_budget_mb
                    else None,
                    active_upper_mib=_safe_float(plan.vram_budget_mb)
                    if plan.vram_budget_mb
                    else None,
                    resident_estimate_mib=None,
                    resident_upper_mib=None,
                    total_upper_bound_mib=None,
                    memory_basis="global_planner",
                    peak_fidelity="gp_upper",
                    guard_margin_mib=None,
                ),
                provenance=SignalProvenance(),
                artifacts=SignalArtifacts(),
                planner_intent=PlannerIntent(),
            ),
            worker_context_applied=False,
        )


    def _classify_error(
        self,
        exc: Exception,
        plan: DispatchPlan,
    ) -> ConstraintViolation:
        """Classify a dispatch exception into a structured ConstraintViolation.

        Maps gRPC errors, memory admission errors, and dispatch failures
        to the appropriate violation type for the Planner's re_plan.  Every
        returned violation carries the correlation-window key
        (``correlation_id``) and the owning plan's snapshot id so that
        ``ConstraintTracker.incorporate`` can merge root-cause fallouts
        (Plan Blocker 2).  Mirrors the ``_DispatchRetrySignal`` path in
        ``validate_and_dispatch`` to keep both branches symmetric.
        """
        from .dispatch import MemoryAdmissionError

        def _finalize(
            violation_type: str,
            *,
            memory_guard_details: dict | None = None,
            grpc_code: int | None = None,
        ) -> ConstraintViolation:
            corr = (
                f"{plan.target_worker_name}:{_safe_int(time.time())}"
                if plan.target_worker_name
                else ""
            )
            return ConstraintViolation(
                violation_type=violation_type,
                gpu_id=plan.target_gpu_id,
                worker_name=plan.target_worker_name,
                correlation_id=corr,
                plan_snapshot_id=plan.plan_snapshot_id,
                grpc_code=grpc_code,
                memory_guard_details=memory_guard_details,
                failed_plan=plan,
            )

        if isinstance(exc, MemoryAdmissionError):
            return _finalize(
                "memory_admission_rejected",
                memory_guard_details={
                    "upper_bound_mib": getattr(
                        getattr(exc, "admission", None),
                        "memory_upper_bound_mib",
                        None,
                    ),
                    "safe_limit_mib": getattr(
                        getattr(exc, "admission", None),
                        "mem_safe_limit_mib",
                        None,
                    ),
                },
            )

        try:
            import grpc

            if isinstance(exc, grpc.aio.AioRpcError):
                code = exc.code()
                grpc_code = int(code.value[0]) if hasattr(code, "value") else None
                if code in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                ):
                    vtype = self._classify_grpc_lifecycle(
                        code,
                        plan.target_worker_name,
                    )
                else:
                    vtype = "grpc_error"
                return _finalize(vtype, grpc_code=grpc_code)
        except ImportError:
            pass

        if isinstance(
            exc, (ArithmeticError, ValueError, OverflowError)
        ) and _is_numerical_failure(exc):
            return _finalize("numerical_failure")

        if isinstance(exc, OSError):
            return _finalize("host_resource_exhausted")

        _LOG.warning(
            "[dispatch-classifier] post-activation %s: %s",
            type(exc).__name__,
            exc,
        )
        return _finalize("grpc_error")

    def _classify_grpc_lifecycle(
        self,
        code: Any,
        worker_name: str,
    ) -> str:
        """Plan (fix  ) —
        route UNAVAILABLE / DEADLINE_EXCEEDED to one of
        (``grpc_error``, ``worker_unreachable``, ``worker_dead``) using
        supervisor health-check as the **sole** worker_dead signal.

        Rules (Plan , fix):
            UNAVAILABLE:
              - ready=True (or unknown) → grpc_error (transient; gateway retry guard owns pacing)
              - ready=False             → worker_dead
            DEADLINE_EXCEEDED:
              - ready=True  → worker_unreachable (hang-but-alive)
              - ready=False → worker_dead

        fix rationale: prior "N-consecutive UNAVAILABLE →
        worker_dead ()" promotion is removed.  Task failure,
        VRAM-observation-miss evictions, kernel-level kills, and
        guard-MEASURE releases are structurally decoupled from worker
        lifecycle — counting gRPC failures is not a valid worker-sanity
        signal.  The ``_worker_transient_failures`` counter is kept
        for observability only (not used to escalate classification).
        """
        import grpc as _grpc

        sup = getattr(self._gateway, "supervisor", None)
        supervisor_ready: bool | None = None
        if sup is not None and worker_name:
            st = sup.states.get(worker_name)
            if st is not None:
                supervisor_ready = bool(getattr(st, "ready", False))

        tracker = self._get_constraint_tracker()

        if supervisor_ready is not None and not supervisor_ready:
            return "worker_dead"

        if tracker is not None and worker_name:
            try:
                tracker.record_transient_grpc_failure(worker_name)
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )
        if code == _grpc.StatusCode.DEADLINE_EXCEEDED and supervisor_ready:
            return "worker_unreachable"
        return "grpc_error"

    def _get_constraint_tracker(self) -> Any | None:
        """Resolve the Planner's ConstraintTracker (may be absent in
        unit-test shims that construct only a RealityValidator)."""
        planner = getattr(self._gateway, "_planner", None)
        gp = getattr(planner, "_global_planner", None) if planner else None
        return getattr(gp, "_constraint_tracker", None) if gp else None

    def select_for_eviction(
        self,
        candidates: list,
        gpu_id: str,
        needed_mb: int,
    ) -> list:
        """Scenario-based eviction: protect workers with predicted tasks.

        Delegates to PlannerService.select_for_eviction, providing access
        to GlobalPlanner's timeline information.
        """
        planner = getattr(self._gateway, "_planner", None)
        if planner:
            return planner.select_for_eviction(candidates, gpu_id, needed_mb)
        return candidates

    def select_for_recovery(
        self,
        candidates: list,
        resource_tracker: Any,
    ) -> list:
        """Scenario-based recovery with VRAM capacity check.

        Delegates to PlannerService.select_for_recovery, providing access
        to GlobalPlanner's timeline information.
        """
        planner = getattr(self._gateway, "_planner", None)
        if planner:
            return planner.select_for_recovery(candidates, resource_tracker)
        return []
