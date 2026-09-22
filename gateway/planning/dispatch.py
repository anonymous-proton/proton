import itertools
import logging
from collections.abc import Callable, Mapping
from typing import Any

from modelworker import modelworker_pb2 as pb

from .planner import PlannerTaskRequest
from .scheduler import WorkerSelection

_LOG = logging.getLogger(__name__)
_BATCH_ID_COUNTER = itertools.count(1)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class DispatchService:
    """
    Stage 2 Pipeline: Interprets the plan generated from the PlannerService, ensures
    target cold-workers are active, performs final scope-aware memory admission against real
    telemetry, and performs the blocking VRAM acquisition.
    """

    def __init__(self, gateway_service: Any):
        self._gateway = gateway_service
        self._on_inference_start: Callable[..., Any] | None = None

    def _cp_begin(self) -> int:
        begin = getattr(self._gateway, "_cp_begin", None)
        if not callable(begin):
            return 0
        try:
            value = begin()
            return int(value) if isinstance(value, (int, float, str)) else 0
        except Exception:
            return 0

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

    async def execute_plan(
        self,
        request: PlannerTaskRequest,
        record: Any,
        worker: WorkerSelection,
        reservation: Any,
        intrinsic_signal: Any,
        payload: dict[str, Any],
        *,
        was_cold_start: bool = False,
        reciprocal_interference: dict[str, Any] | None = None,
        dynamic_batch_context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Takes the chosen worker/reservation from Planner and proceeds through admission
        and gRPC. Returns the InferBatch output if successful.
        """
        worker_name = str(worker.worker_name or "").strip()
        selected_context_applied = (
            "post_selection_retry" if request.scheduling_hints.get("reasons") else None
        )
        was_cold = bool(was_cold_start)

        _cp_start = self._cp_begin()
        await self._gateway._stage_selected_worker(
            record,
            worker,
            preferred_gpu_ids=request.preferred_gpu_ids,
            selected_context_applied=selected_context_applied,
        )
        self._cp_record(
            "dispatch_stage_selected_worker",
            _cp_start,
            active_wall=True,
            component=request.component,
            worker_name=worker_name,
        )

        if not bool(worker.ready):
            _LOG.info(
                "[task %s] Worker %s not ready at Stage 6 (Stage 4→6 race); "
                "surfacing worker_not_ready.",
                record.task_id,
                worker_name,
            )
            from ..http_server import _DispatchRetrySignal
            from .contracts import ConstraintViolation

            raise _DispatchRetrySignal(
                failed_worker_addr=str(worker.addr or "").strip(),
                failed_worker_name=worker_name,
                cause=RuntimeError(
                    "worker_not_ready: Stage 6 dispatch-time worker.ready=False"
                ),
                constraint_violation=ConstraintViolation(
                    violation_type="worker_not_ready",
                    gpu_id=(
                        str(request.preferred_gpu_ids[0])
                        if request.preferred_gpu_ids
                        else ""
                    ),
                    worker_name=worker_name,
                ),
            )

        effective_overrides = dict(request.execution_overrides)
        batch_meta = (
            dict(dynamic_batch_context)
            if isinstance(dynamic_batch_context, dict)
            else {}
        )
        selected_k = _safe_int(batch_meta.get("selected_batch_size"))
        if selected_k > 0:
            effective_overrides["batch_size"] = selected_k
        _cp_start = self._cp_begin()
        evaluation = self._gateway._evaluate_selected_worker_dispatch(
            record=record,
            selection=worker,
            workload_features=request.workload_features,
            execution_overrides=effective_overrides,
            config_fingerprint=request.config_fingerprint,
            input_fingerprint=request.input_fingerprint,
            selected_context_applied=selected_context_applied,
            planner_intent=intrinsic_signal.bundle.planner_intent,
            execution_profile=(
                batch_meta.get("execution_profile")
                if isinstance(batch_meta.get("execution_profile"), Mapping)
                else None
            ),
        )
        self._cp_record(
            "dispatch_predictive_evaluation",
            _cp_start,
            active_wall=True,
            component=request.component,
            worker_name=worker_name,
        )
        admission = evaluation.admission

        if admission.memory_fits is not None and not admission.memory_fits:
            raise MemoryAdmissionError(admission, evaluation, worker)

        resolved_overrides = dict(evaluation.execution_overrides)
        if selected_k > 0:
            final_k = _safe_int(resolved_overrides.get("batch_size"))
            if final_k != selected_k:
                raise MemoryAdmissionError(admission, evaluation, worker)
            logical_n = _safe_int(batch_meta.get("logical_batch_size"))
            payload["_dynamic_batch"] = {
                "batch_size_arg": str(batch_meta.get("batch_size_arg") or ""),
                "logical_n": logical_n,
            }
            dynamic_batch_context = {
                "execution_batch_size": selected_k,
                "final_admission_batch_size": final_k,
                "logical_batch_size": logical_n,
                "batch_phase": str(batch_meta.get("phase") or ""),
                "batch_policy": str(batch_meta.get("policy") or ""),
                "fallback_reason": str(batch_meta.get("fallback_reason") or ""),
            }
            if isinstance(batch_meta.get("execution_profile"), Mapping):
                dynamic_batch_context["execution_profile"] = dict(
                    batch_meta["execution_profile"]
                )
        elif batch_meta:
            dynamic_batch_context = {
                "execution_batch_size": None,
                "final_admission_batch_size": None,
                "logical_batch_size": _safe_int(batch_meta.get("logical_batch_size")),
                "batch_phase": str(batch_meta.get("phase") or "exact"),
                "batch_policy": str(batch_meta.get("policy") or "fixed_n"),
                "fallback_reason": str(batch_meta.get("fallback_reason") or ""),
            }
        if resolved_overrides:
            payload["execution_overrides"] = dict(resolved_overrides)

        addr = str(worker.addr or "").strip()
        if not addr:
            raise ValueError("Worker address is missing after activation")

        import json

        request_item_type = pb.__dict__["RequestItem"]
        req_item = request_item_type(
            request_id=record.task_id,
            payload_json=json.dumps(payload).encode("utf-8"),
        )
        batch = pb.BatchRequest(batch_id=next(_BATCH_ID_COUNTER), requests=[req_item])


        is_backfill = request.scheduling_hints.get("is_backfill", False)


        _cp_start = self._cp_begin()
        await self._gateway._apply_dispatch_evaluation(
            record,
            worker,
            evaluation,
            preferred_gpu_ids=request.preferred_gpu_ids,
        )
        self._cp_record(
            "dispatch_apply_evaluation",
            _cp_start,
            active_wall=True,
            component=request.component,
            worker_name=worker_name,
        )

        return await self._gateway._dispatch_selected_worker(
            record=record,
            worker=worker,
            reservation=reservation,
            batch=batch,
            timeout_s=request.timeout_s,
            campaign_id=request.campaign_id,
            is_backfill=is_backfill,
            was_cold_start=was_cold,
            reciprocal_interference=reciprocal_interference,
            dynamic_batch_context=dynamic_batch_context,
        ), evaluation


class MemoryAdmissionError(Exception):
    def __init__(self, admission: Any, evaluation: Any, worker: WorkerSelection):
        self.admission = admission
        self.evaluation = evaluation
        self.worker = worker
        super().__init__("Worker rejected by predictive memory admission")
