"""HTTP JSON API for the Gateway service.

This provides a simpler HTTP interface for the Nextflow gw executor plugin,
which submits tasks via HTTP instead of gRPC for implementation simplicity.

Endpoints:
    POST /api/v1/job/submit      - Submit task job
    GET  /api/v1/job/{job_id}    - Get unified job status
    POST /api/v1/job/cancel/{job_id} - Cancel unified job
    GET  /api/v1/profile/runs     - Query indexed profiling runs
    GET  /api/v1/profile/runs/{run_key} - Inspect one indexed run
    GET  /api/v1/campaigns        - Campaign queue/list read model
    GET  /api/v1/campaigns/{campaign_id} - Campaign detail read model
    GET  /api/v1/telemetry/health - Inspect task telemetry ingestion health
    GET  /api/v1/model/versions   - Query model versions
    GET  /api/v1/workers          - List worker lifecycle states
    GET  /api/v1/debug/tasks      - List active tasks in memory
    GET  /health                  - Health check
"""

from __future__ import annotations

import asyncio
import contextlib
from asyncio import CancelledError
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import grpc
from aiohttp import web

from modelworker import modelworker_pb2 as _pb
from modelworker import modelworker_pb2_grpc as pb_grpc
from modelworker.memory_observer import (
    BootstrapMemorySummary,
    DispatchMemoryWindow,
    dispatch_memory_summary,
)
from modelworker.nextflow_contract import ContractError, validate_task_payload

from .admission import AdmissionDecision, evaluate_admission
from .extraction.command_extraction import _host_to_container_path
from .extraction.component_features import extract_fan_out as _extract_fan_out
from .extraction.component_features import (
    extract_features as _extract_component_features,
)
from .extraction.component_features import (
    filter_dynamic_batch_config as _filter_dynamic_batch_config,
)
from .extraction.component_features import logical_batch_size as _logical_batch_size
from .gateway_identity import (
    gateway_payload_from_mapping,
    normalize_gateway_git_commit,
    normalize_gateway_instance_id,
)
from .jobs.job_service import JobService
from .planning.constraint_tracker import PlanningExhausted
from .planning.dispatch import DispatchService
from .planning.pipeline_dag import PipelineDAG
from .planning.planner import PlannerService, PlannerTaskRequest
from .planning.scheduler import WorkerSelection
from .profiling.campaign_views import get_campaign_detail as build_campaign_detail
from .profiling.campaign_views import list_campaigns as build_campaign_queue
from .profiling.component_model_view import (
    build_component_model_payload,
    choose_effective_model_view,
)
from .profiling.constants import (
    PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_DESCRIPTION,
    PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_NAME,
)
from .profiling.control_plane_overhead import (
    SNAPSHOT_KEY as CONTROL_PLANE_OVERHEAD_SNAPSHOT_KEY,
)
from .profiling.control_plane_overhead import (
    ControlPlaneOverheadRecorder,
)
from .profiling.index_models import AxisFilter, RunQuery
from .profiling.index_store import RunIndexStore
from .profiling.promotion_gate import evaluate_promotion_gate
from .profiling.run_identity import canonicalize_axes
from .profiling.runtime_regime import bucket_batch_size, normalize_worker_context
from .profiling.telemetry_health import build_task_telemetry_health_report
from .signals import (
    PlannerIntent,
    SignalArtifacts,
    SignalBundle,
    SignalResult,
    SignalService,
)

pb = cast(Any, _pb)

_LOG = logging.getLogger(__name__)
_DEFAULT_INDEX_DB = (
    Path(__file__).resolve().parents[1] / ".gateway" / "profile_runs.sqlite3"
).resolve()
_DEFAULT_MEM_SAFE_LIMIT_MIB = 0.9 * 24576.0

TASK_STATE_UNSPECIFIED = 0
TASK_STATE_SUBMITTED = 1
TASK_STATE_RUNNING = 2
TASK_STATE_SUCCEEDED = 3
TASK_STATE_FAILED = 4
TASK_STATE_CANCELLED = 5

STATE_NAMES = {
    TASK_STATE_UNSPECIFIED: "UNSPECIFIED",
    TASK_STATE_SUBMITTED: "SUBMITTED",
    TASK_STATE_RUNNING: "RUNNING",
    TASK_STATE_SUCCEEDED: "SUCCEEDED",
    TASK_STATE_FAILED: "FAILED",
    TASK_STATE_CANCELLED: "CANCELLED",
}


@dataclass
class TaskRecord:
    task_id: str
    nf_task_id: str
    component: str
    workdir: str
    state: int
    campaign_id: str = ""
    run_name: str = ""
    submitter: str = ""
    campaign_metadata_version: int = 0
    ok: bool = False
    exit_code: int = 0
    message: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    worker_timing_us: dict[str, int] | None = None
    dispatch_worker_addr: str = ""
    dispatch_gpu_ids: list[str] = field(default_factory=list)
    dispatch_worker_name: str = ""
    dispatch_resident_memory_mib: float | None = None
    dispatch_resident_memory_source: str = ""
    dispatch_resident_baseline_collected_at: float | None = None
    dispatch_resident_baseline_lifecycle_token: str = field(default_factory=str)
    dispatch_resident_baseline_state: str = ""
    worker_generation_token: str = field(default_factory=str)
    run_ordinal_in_generation: int | None = None
    is_first_real_run: bool = False
    generation_token_stale: bool = False
    workload_features: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = field(default_factory=str)
    input_fingerprint: str = field(default_factory=str)
    decision_payload: dict[str, Any] = field(default_factory=dict)
    signal_runtime_sec: float | None = None
    signal_runtime_upper_sec: float | None = None
    dispatch_vram_budget_mb: int = 0
    dispatch_ram_budget_mb: int = 0
    dispatch_planned_start_time: float = 0.0
    workflow_completion_expected: bool = False
    _feature_extraction_cache_key: str = field(default="", repr=False)
    _feature_extraction_done: bool = field(default=False, repr=False)
    _cached_extracted_workload_features: dict[str, Any] = field(
        default_factory=dict, repr=False
    )
    _cached_extracted_config_features: dict[str, str] = field(
        default_factory=dict, repr=False
    )
    _handle_context: dict[str, Any] = field(default_factory=dict, repr=False)
    _dynamic_batch_context: dict[str, Any] = field(default_factory=dict, repr=False)
    _campaign_submit_registered: bool = field(default=False, repr=False)


class _APIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)


class _TaskDispatchFailure(Exception):
    pass


class _DispatchRetrySignal(Exception):
    """Signal carrying a structured ConstraintViolation back to the Core Loop.

     removed the Validator's ``redirect_gpu_id`` override — the Planner
    is the sole placement decision-maker.  The attribute remains for
    backward-compatibility with existing call sites (always empty) so that
    importers do not need a coordinated update in Phase 3 of the
    refactoring; Phase 4 / subsequent cleanup removes it entirely along
    with its (already-dead) callers.
    """

    def __init__(
        self,
        *,
        failed_worker_addr: str,
        failed_worker_name: str,
        cause: Exception,
        constraint_violation: Any = None,
    ) -> None:
        super().__init__(str(cause))
        self.failed_worker_addr = str(failed_worker_addr or "").strip()
        self.failed_worker_name = str(failed_worker_name or "").strip()
        self.cause = cause
        self.redirect_gpu_id = ""
        self.constraint_violation = constraint_violation


class WorkerSelector(Protocol):
    def __call__(
        self,
        component: str,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        *,
        schedule_hint: Mapping[str, Any] | None = None,
        include_not_ready: bool = False,
        preferred_worker_name: str | None = None,
        campaign_id: str | None = None,
        intrinsic_signal: Any | None = None,
        planner_intent: PlannerIntent | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class _DispatchEvaluation:
    selection: WorkerSelection
    signal: SignalResult
    admission: AdmissionDecision
    selected_context_applied: str
    execution_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class _AdmissionReservation:
    worker_name: str
    bump_dispatch_pending: Callable[[str], Any] | None = None
    release_dispatch_pending: Callable[..., None] | None = None
    adjust_worker_inflight_by_name: Callable[[str, int], None] | None = None
    worker_addr: str = ""
    state: str = "released"
    cold_pending_reserved: bool = False
    front_reserved: bool = False

    def _release_dispatch_token(self, key: str) -> None:
        if not self.release_dispatch_pending:
            return
        try:
            self.release_dispatch_pending(key, worker_name=self.worker_name)
        except TypeError:
            self.release_dispatch_pending(key)

    @classmethod
    def reserve_cold(
        cls,
        *,
        worker_name: str,
        adjust_worker_inflight_by_name: Callable[[str, int], None] | None = None,
        bump_dispatch_pending: Callable[[str], Any] | None = None,
        release_dispatch_pending: Callable[[str], None] | None = None,
    ) -> _AdmissionReservation:
        normalized_name = str(worker_name or "").strip()
        cold_pending_reserved = False
        front_reserved = False
        if normalized_name and bump_dispatch_pending:
            bumped = bump_dispatch_pending(normalized_name)
            if bumped is False:
                raise RuntimeError("dispatch-front reservation denied")
            cold_pending_reserved = True
            front_reserved = True
        elif normalized_name and adjust_worker_inflight_by_name:
            adjust_worker_inflight_by_name(normalized_name, 1)
            front_reserved = True
        return cls(
            worker_name=normalized_name,
            bump_dispatch_pending=bump_dispatch_pending,
            release_dispatch_pending=release_dispatch_pending,
            adjust_worker_inflight_by_name=adjust_worker_inflight_by_name,
            state="cold_name_reserved",
            cold_pending_reserved=cold_pending_reserved,
            front_reserved=front_reserved,
        )

    @classmethod
    def reserve_ready(
        cls,
        *,
        worker_name: str,
        worker_addr: str,
        adjust_worker_inflight_by_name: Callable[[str, int], None] | None = None,
        bump_dispatch_pending: Callable[[str], Any] | None = None,
        release_dispatch_pending: Callable[[str], None] | None = None,
    ) -> _AdmissionReservation:
        normalized_addr = str(worker_addr or "").strip()
        if not normalized_addr:
            raise ValueError("worker address missing")
        front_reserved = False
        if bump_dispatch_pending:
            bumped = bump_dispatch_pending(normalized_addr)
            if bumped is False:
                raise RuntimeError("dispatch-front reservation denied")
            front_reserved = True
        return cls(
            worker_name=str(worker_name or "").strip(),
            worker_addr=normalized_addr,
            bump_dispatch_pending=bump_dispatch_pending,
            release_dispatch_pending=release_dispatch_pending,
            adjust_worker_inflight_by_name=adjust_worker_inflight_by_name,
            state="ready_addr_reserved",
            front_reserved=front_reserved,
        )

    def promote_to_addr(self, worker_addr: str) -> None:
        if self.state != "cold_name_reserved":
            raise RuntimeError(f"cannot promote reservation from state {self.state}")
        normalized_addr = str(worker_addr or "").strip()
        if not normalized_addr:
            raise ValueError("worker address missing")

        if self.front_reserved and self.cold_pending_reserved:
            self.worker_addr = normalized_addr
            self.state = "ready_addr_reserved"
            return

        bumped = False
        try:
            if self.bump_dispatch_pending:
                bump_result = self.bump_dispatch_pending(normalized_addr)
                if bump_result is False:
                    raise RuntimeError("dispatch-front reservation denied")
                bumped = True
            if (
                self.cold_pending_reserved
                and self.worker_name
                and self.release_dispatch_pending
            ):
                self._release_dispatch_token(self.worker_name)
            elif self.worker_name and self.adjust_worker_inflight_by_name:
                self.adjust_worker_inflight_by_name(self.worker_name, -1)
        except Exception:
            if bumped and self.release_dispatch_pending:
                self._release_dispatch_token(normalized_addr)
            raise
        self.worker_addr = normalized_addr
        self.state = "ready_addr_reserved"
        self.front_reserved = True

    def retarget_ready_addr(self, worker_addr: str) -> None:
        if self.state != "ready_addr_reserved":
            raise RuntimeError(f"cannot retarget reservation from state {self.state}")
        normalized_addr = str(worker_addr or "").strip()
        if not normalized_addr:
            raise ValueError("worker address missing")
        old_addr = str(self.worker_addr or "").strip()
        if old_addr == normalized_addr:
            return
        bumped = False
        try:
            if self.bump_dispatch_pending:
                bump_result = self.bump_dispatch_pending(normalized_addr)
                if bump_result is False:
                    raise RuntimeError("dispatch-front reservation denied")
                bumped = True
            if old_addr and self.release_dispatch_pending:
                self._release_dispatch_token(old_addr)
            elif self.worker_name and self.release_dispatch_pending:
                self._release_dispatch_token(self.worker_name)
        except Exception:
            if bumped and self.release_dispatch_pending:
                self._release_dispatch_token(normalized_addr)
            raise
        self.worker_addr = normalized_addr
        self.front_reserved = True

    def consume_on_task_start(
        self, on_task_start: Callable[..., None] | None, *args, **kwargs
    ) -> None:
        if self.state != "ready_addr_reserved":
            raise RuntimeError(f"cannot consume reservation from state {self.state}")
        if not self.worker_addr:
            raise RuntimeError("ready reservation missing worker address")
        if on_task_start:
            callback_kwargs = dict(kwargs)
            callback_kwargs.setdefault("worker_name", self.worker_name)
            on_task_start(self.worker_addr, *args, **callback_kwargs)
        elif self.release_dispatch_pending:
            self._release_dispatch_token(self.worker_addr)
        self.state = "started_consumed"
        self.front_reserved = False

    def release(self) -> None:
        if self.state == "released":
            return
        if self.state == "cold_name_reserved":
            if (
                self.front_reserved
                and self.cold_pending_reserved
                and self.worker_name
                and self.release_dispatch_pending
            ):
                self._release_dispatch_token(self.worker_name)
            elif (
                self.front_reserved
                and self.worker_name
                and self.adjust_worker_inflight_by_name
            ):
                self.adjust_worker_inflight_by_name(self.worker_name, -1)
        elif self.state == "ready_addr_reserved" and (
            self.front_reserved and self.worker_addr and self.release_dispatch_pending
        ):
            self._release_dispatch_token(self.worker_addr)
        self.state = "released"
        self.front_reserved = False


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except Exception:
        return False


def _normalize_env(env: Any) -> dict[str, str]:
    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise ValueError("env must be a string->string map")
    out: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str):
            raise ValueError("env keys must be strings")
        if not isinstance(value, str):
            raise ValueError("env values must be strings")
        out[key] = value
    return out


def _normalize_timeout_s(value: Any, default: int = 300) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError("timeout_s must be an integer") from exc
    if parsed < 0:
        raise ValueError("timeout_s must be >= 0")
    return parsed


def _normalize_optional_positive_int(value: Any, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer when provided") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer when provided")
    return parsed


def _normalize_optional_float(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a float when provided") from exc
    return parsed


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _normalize_argv(argv: Any) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise ValueError("argv must be a non-empty list of strings")
    out: list[str] = []
    for arg in argv:
        if not isinstance(arg, str) or not arg:
            raise ValueError("argv must be a non-empty list of strings")
        out.append(_host_to_container_path(arg))
    return out


def _normalize_tool_cwd(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("tool_cwd must be a string when provided")
    return _host_to_container_path(value)


def _normalize_profiling(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("profiling must be an object when provided")
    if value.get("level") is not None:
        raise ValueError("profiling.level is not supported in canonical runtime path")

    run_id = value.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("profiling.run_id must be a string when provided")

    output_dir = value.get("output_dir")
    if output_dir is not None and (
        not isinstance(output_dir, str) or not output_dir.strip()
    ):
        raise ValueError(
            "profiling.output_dir must be a non-empty string when provided"
        )

    trace_input_path = value.get("trace_input_path")
    if trace_input_path is not None:
        if not isinstance(trace_input_path, str) or not trace_input_path.strip():
            raise ValueError(
                "profiling.trace_input_path must be a non-empty string when provided"
            )
        raise ValueError(
            "profiling.trace_input_path is not supported in canonical runtime path"
        )

    include_preprocess = value.get("include_preprocess", False)
    if not isinstance(include_preprocess, bool):
        raise ValueError("profiling.include_preprocess must be a boolean when provided")
    if include_preprocess:
        raise ValueError("profiling.include_preprocess=true is not supported")

    normalized: dict[str, Any] = {"include_preprocess": include_preprocess}
    if run_id is not None:
        normalized["run_id"] = run_id
    if output_dir is not None:
        normalized["output_dir"] = output_dir
    return normalized


def _normalize_optional_identity_field(value: Any, field: str) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when provided")
    token = str(value).strip()
    if field == "campaign_id" and token == "__unknown__":
        raise ValueError("campaign_id cannot be __unknown__")
    return token


def _extract_worker_timing(timing: Any) -> dict[str, int] | None:
    if timing is None:
        return None
    payload = {
        "queue_delay_us": int(timing.queue_delay_us),
        "prepare_us": int(timing.prepare_us),
        "execute_us": int(timing.execute_us),
        "finalize_us": int(timing.finalize_us),
        "total_us": int(timing.total_us),
    }
    if any(value != 0 for value in payload.values()):
        return payload
    return None


def _stable_fingerprint(payload: Mapping[str, Any], *, prefix: str) -> str:
    try:
        raw = json.dumps(
            dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
    except Exception:
        raw = repr(dict(payload))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def _normalize_resident_baseline_snapshot(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(payload or {}) if isinstance(payload, Mapping) else {}
    state = str(raw.get("resident_baseline_state") or "").strip().lower()
    resident_memory_mib = _to_float(raw.get("resident_memory_mib"))
    resident_memory_source = str(raw.get("resident_memory_source") or "").strip()
    if state != "ready":
        resident_memory_mib = None
        resident_memory_source = ""
    return {
        "resident_memory_mib": resident_memory_mib,
        "resident_memory_source": resident_memory_source,
        "resident_baseline_collected_at": _to_float(
            raw.get("resident_baseline_collected_at")
        ),
        "resident_baseline_lifecycle_token": str(
            raw.get("resident_baseline_lifecycle_token") or ""
        ).strip(),
        "resident_baseline_state": state,
    }


def _copy_dispatch_resident_baseline(
    record: TaskRecord, worker: WorkerSelection | Mapping[str, Any]
) -> None:
    selection = _normalize_worker_selection(worker)
    snapshot = _normalize_resident_baseline_snapshot(
        selection.resident_baseline_snapshot
    )
    record.dispatch_resident_memory_mib = snapshot["resident_memory_mib"]
    record.dispatch_resident_memory_source = str(
        snapshot["resident_memory_source"] or ""
    )
    record.dispatch_resident_baseline_collected_at = _to_float(
        snapshot["resident_baseline_collected_at"]
    )
    record.dispatch_resident_baseline_lifecycle_token = str(
        snapshot["resident_baseline_lifecycle_token"] or ""
    ).strip()
    record.dispatch_resident_baseline_state = str(
        snapshot["resident_baseline_state"] or ""
    ).strip()


def _dedupe_reasons(values: Sequence[Any] | None) -> list[str]:
    return list(
        dict.fromkeys(
            str(item).strip() for item in list(values or []) if str(item).strip()
        )
    )


def _normalize_worker_selection(
    worker: WorkerSelection | Mapping[str, Any],
) -> WorkerSelection:
    if isinstance(worker, WorkerSelection):
        return worker
    if isinstance(worker, Mapping):
        return WorkerSelection.from_mapping(worker)
    raise TypeError(f"unsupported worker selection payload: {type(worker)!r}")


def _worker_uses_serialized_execute(worker_state: Any) -> bool:
    """Return True when a worker can run only one execute stage at a time."""
    cap = 0
    try:
        cap = int(
            getattr(getattr(worker_state, "caps", None), "max_inflight_batches", 0) or 0
        )
    except Exception:
        cap = 0
    configured = int(getattr(worker_state, "max_concurrency", 0) or 0)
    return configured == 1 or cap == 1


def _worker_execution_identity(worker_state: Any) -> dict[str, str]:
    """Return known actor/MPS identity, leaving unavailable fields empty."""
    if worker_state is None:
        return {}
    worker_server = dict(
        getattr(getattr(worker_state, "spec", None), "worker_server", None) or {}
    )
    configured_mps = worker_server.get("cuda_mps")
    mps_enabled = True if configured_mps is None else configured_mps is True
    return {
        "mps_mode": "enabled" if mps_enabled else "disabled",
        "worker_backend": "persistent_actor",
        "actor_model": "shared_cuda_ipc",
        "adapter_version": str(
            getattr(getattr(worker_state, "caps", None), "model_version", "") or ""
        ),
    }


def _selected_worker_context(
    worker: WorkerSelection | Mapping[str, Any],
) -> dict[str, Any]:
    selection = _normalize_worker_selection(worker)
    raw = dict(selection.estimator_worker_context)
    return normalize_worker_context(
        {
            "worker_addr": str(raw.get("worker_addr") or selection.addr or "").strip(),
            "worker_name": str(
                raw.get("worker_name") or selection.worker_name or ""
            ).strip(),
            "gpu_ids": [
                str(item).strip()
                for item in list(raw.get("gpu_ids") or selection.gpu_ids or [])
                if str(item).strip()
            ],
            "adapter": str(raw.get("adapter") or "").strip(),
            "image": str(raw.get("image") or "").strip(),
            "model_version": str(raw.get("model_version") or "").strip(),
            "active_request_count": _to_int(raw.get("active_request_count")) or 1,
            "queue_depth": _to_int(raw.get("queue_depth")) or 0,
            "dispatch_group_size": _to_int(raw.get("dispatch_group_size")) or 1,
            "co_location_signature": str(
                raw.get("co_location_signature") or ""
            ).strip(),
            "hardware_software": str(raw.get("hardware_software") or "").strip(),
            "residency_state": str(raw.get("residency_state") or "").strip(),
        }
    )


def _placement_payload(
    *,
    selection: WorkerSelection,
    selected_worker_context: Mapping[str, Any],
    selected_context_applied: str,
    extra_reasons: Sequence[str] | None = None,
    admission_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    placement = selection.placement.as_dict()
    placement["selected_worker_context"] = dict(selected_worker_context)
    placement["selected_context_applied"] = str(selected_context_applied or "")
    placement["reasons"] = _dedupe_reasons(
        list(placement.get("reasons") or []) + list(extra_reasons or [])
    )
    if admission_history:
        placement["admission_history"] = [
            dict(item) for item in admission_history if isinstance(item, Mapping)
        ]
    return placement


def _decision_payload_from_evaluation(
    *,
    evaluation: _DispatchEvaluation,
    extra_reasons: Sequence[str] | None = None,
    admission_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_worker_context = _selected_worker_context(evaluation.selection)
    return {
        "version": 5,
        "placement": _placement_payload(
            selection=evaluation.selection,
            selected_worker_context=selected_worker_context,
            selected_context_applied=evaluation.selected_context_applied,
            extra_reasons=extra_reasons,
            admission_history=admission_history,
        ),
        "signal": evaluation.signal.as_dict(),
        "admission": evaluation.admission.as_dict(),
        "actual": {},
    }


def _mapping_payload(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _batch_observation_context(
    record: Any,
    telemetry: Mapping[str, Any],
    *,
    succeeded: bool | None = None,
) -> dict[str, Any]:
    attempt = _mapping_payload(getattr(record, "_dynamic_batch_context", None))
    selected = _to_int(attempt.get("execution_batch_size"))
    final_admission = _to_int(attempt.get("final_admission_batch_size"))
    argument_applied = _to_int(telemetry.get("dynamic_batch_argument_applied_k"))
    consumed = _to_int(telemetry.get("dynamic_batch_consumed_k"))
    logical_n = _to_int(
        telemetry.get("dynamic_batch_logical_n") or attempt.get("logical_batch_size")
    )
    component = str(getattr(record, "component", "") or "").strip().lower()
    agreement = (
        selected is not None
        and final_admission == selected
        and argument_applied == selected
        and logical_n is not None
        and 1 <= selected <= logical_n
        and (component != "mmseqs2" or consumed == selected)
    )
    return {
        "batch_phase": str(attempt.get("batch_phase") or ""),
        "batch_policy": str(attempt.get("batch_policy") or ""),
        "fallback_reason": str(attempt.get("fallback_reason") or ""),
        "execution_batch_size": selected,
        "final_admission_batch_size": final_admission,
        "argument_applied_batch_size": argument_applied,
        "consumed_batch_size": consumed,
        "logical_batch_size": logical_n,
        "group_count": _to_int(telemetry.get("dynamic_batch_group_count")),
        "companion_eligible": bool(
            agreement
            and (
                succeeded
                if succeeded is not None
                else STATE_NAMES.get(int(getattr(record, "state", 0) or 0))
                == "SUCCEEDED"
            )
        ),
    }


def _decode_json_mapping(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        if isinstance(value, bytes):
            payload = json.loads(value.decode("utf-8"))
        else:
            payload = json.loads(str(value))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _extract_task_telemetry(
    payload_json: bytes,
    *,
    bootstrap_memory_summary_json: bytes = b"",
    dispatch_memory_window_json: bytes = b"",
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mean_gpu_util_percent": None,
        "peak_vram_mib": None,
        "active_vram_mib": None,
        "vram_memory_basis": "",
        "vram_memory_measurement": "",
        "vram_memory_qc_keep": False,
        "vram_memory_attribution": "",
        "actor_id": None,
        "actor_generation": None,
        "actor_pid": None,
        "actor_resident_owner": "",
        "actor_resident_vram_mib": None,
        "actor_resident_host_mib": None,
        "shared_resident_vram_mib": None,
        "shared_resident_host_mib": None,
        "vram_source_complete": False,
        "host_source_complete": False,
        "device_pre_vram_mib": None,
        "device_peak_vram_mib": None,
        "device_post_vram_mib": None,
        "device_unattributed_pre_vram_mib": None,
        "device_unattributed_peak_vram_mib": None,
        "device_vram_measurement": "",
        "host_peak_memory_mib": None,
        "host_active_memory_mib": None,
        "host_resident_memory_mib": None,
        "host_resident_memory_source": "",
        "host_resident_baseline_collected_at": None,
        "host_resident_baseline_lifecycle_token": "",
        "host_resident_baseline_state": "",
        "host_memory_basis": "",
        "host_memory_measurement": "",
        "host_memory_qc_keep": False,
        "host_memory_attribution": "",
        "synthetic_supervisor_activation_mib": None,
        "host_dispatch_memory_window": {},
        "host_bootstrap_memory_summary": {},
        "peak_memory_mib": None,
        "active_memory_mib": None,
        "active_memory_scope": "",
        "active_memory_measurement": "",
        "memory_basis": "",
        "peak_fidelity": "",
        "total_upper_bound_mib": None,
        "resident_memory_mib": None,
        "resident_memory_source": "",
        "resident_baseline_collected_at": None,
        "resident_baseline_lifecycle_token": "",
        "resident_baseline_state": "",
        "memory_qc_keep": False,
        "concurrent_execute_overlap": False,
        "telemetry_wall_clock_sec": None,
        "bootstrap_memory_summary": {},
        "dispatch_memory_window": {},
        "dynamic_batch_argument_applied_k": None,
        "dynamic_batch_consumed_k": None,
        "dynamic_batch_logical_n": None,
        "dynamic_batch_group_count": None,
    }
    payload = _decode_json_mapping(payload_json)
    util_keys = [
        "mean_gpu_util_percent",
        "gpu_util_percent",
        "mean_temporal_util_percent",
        "gpu_util",
        "util_percent",
    ]
    mem_keys = [
        "peak_memory_mib",
        "peak_mem_mib",
        "gpu_peak_memory_mib",
        "peak_mem",
    ]
    active_mem_keys = [
        "active_memory_mib",
        "active_mem_mib",
    ]
    wall_keys = [
        "telemetry_wall_clock_sec",
        "wall_clock_sec",
        "elapsed_sec",
    ]

    def _first(keys: list[str]) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                _LOG.debug("Ignoring non-numeric profile value %r: %s", value, exc)
                continue
            if parsed != parsed:
                continue
            return parsed
        return None

    active_memory_mib = _first(active_mem_keys)
    legacy_peak_memory_mib = _first(mem_keys)
    if legacy_peak_memory_mib is None:
        legacy_peak_memory_mib = active_memory_mib
    active_vram_mib = _to_float(payload.get("active_vram_mib"))
    peak_vram_mib = _to_float(payload.get("peak_vram_mib"))
    bootstrap_payload = _decode_json_mapping(bootstrap_memory_summary_json)
    dispatch_payload = _decode_json_mapping(dispatch_memory_window_json)
    if bootstrap_payload:
        summary["bootstrap_memory_summary"] = dict(
            BootstrapMemorySummary.from_mapping(bootstrap_payload).as_dict()
        )
    if dispatch_payload:
        dispatch_window = DispatchMemoryWindow.from_mapping(dispatch_payload)
        dispatch_summary = dispatch_memory_summary(dispatch_window)
        if active_vram_mib is None:
            active_vram_mib = _to_float(dispatch_summary.get("active_memory_mib"))
        if peak_vram_mib is None:
            peak_vram_mib = _to_float(dispatch_summary.get("peak_memory_mib"))
        summary.update(
            {
                **dispatch_summary,
                "active_memory_scope": str(
                    dispatch_summary.get("memory_basis") or ""
                ).strip(),
                "dispatch_memory_window": dispatch_window.as_dict(),
            }
        )
    if active_vram_mib is None:
        active_vram_mib = active_memory_mib
    if peak_vram_mib is None:
        peak_vram_mib = legacy_peak_memory_mib
    host_dispatch_window = _decode_json_mapping(
        payload.get("host_dispatch_memory_window")
    )
    host_bootstrap_summary = _decode_json_mapping(
        payload.get("host_bootstrap_memory_summary")
    )
    summary.update(
        {
            "mean_gpu_util_percent": _first(util_keys),
            "peak_vram_mib": peak_vram_mib,
            "active_vram_mib": active_vram_mib,
            "vram_memory_basis": str(
                payload.get("vram_memory_basis")
                or summary.get("memory_basis")
                or payload.get("memory_basis")
                or payload.get("active_memory_scope")
                or ""
            ).strip(),
            "vram_memory_measurement": str(
                payload.get("vram_memory_measurement")
                or summary.get("active_memory_measurement")
                or payload.get("active_memory_measurement")
                or ""
            ).strip(),
            "vram_memory_qc_keep": _to_bool(
                payload.get("vram_memory_qc_keep")
                if "vram_memory_qc_keep" in payload
                else summary.get("memory_qc_keep")
                if dispatch_payload
                else payload.get("memory_qc_keep"),
                default=False,
            ),
            "vram_memory_attribution": str(
                payload.get("vram_memory_attribution")
                or summary.get("memory_attribution")
                or ""
            ).strip(),
            "host_peak_memory_mib": _to_float(payload.get("host_peak_memory_mib")),
            "host_active_memory_mib": _to_float(payload.get("host_active_memory_mib")),
            "host_resident_memory_mib": _to_float(
                payload.get("host_resident_memory_mib")
            ),
            "host_resident_memory_source": str(
                payload.get("host_resident_memory_source") or ""
            ).strip(),
            "host_resident_baseline_collected_at": _to_float(
                payload.get("host_resident_baseline_collected_at")
            ),
            "host_resident_baseline_lifecycle_token": str(
                payload.get("host_resident_baseline_lifecycle_token") or ""
            ).strip(),
            "host_resident_baseline_state": str(
                payload.get("host_resident_baseline_state") or ""
            ).strip(),
            "host_memory_basis": str(payload.get("host_memory_basis") or "").strip(),
            "host_memory_measurement": str(
                payload.get("host_memory_measurement") or ""
            ).strip(),
            "host_memory_qc_keep": _to_bool(
                payload.get("host_memory_qc_keep"), default=False
            ),
            "host_memory_attribution": str(
                payload.get("host_memory_attribution") or ""
            ).strip(),
            "actor_id": _to_int(payload.get("actor_id")),
            "actor_generation": _to_int(payload.get("actor_generation")),
            "actor_pid": _to_int(payload.get("actor_pid")),
            "actor_resident_owner": str(
                payload.get("actor_resident_owner") or ""
            ).strip(),
            "actor_resident_vram_mib": _to_float(
                payload.get("actor_resident_vram_mib")
            ),
            "actor_resident_host_mib": _to_float(
                payload.get("actor_resident_host_mib")
            ),
            "shared_resident_vram_mib": _to_float(
                payload.get("shared_resident_vram_mib")
            ),
            "shared_resident_host_mib": _to_float(
                payload.get("shared_resident_host_mib")
            ),
            "vram_source_complete": _to_bool(
                payload.get("vram_source_complete"), default=False
            ),
            "host_source_complete": _to_bool(
                payload.get("host_source_complete"), default=False
            ),
            "device_pre_vram_mib": _to_float(payload.get("device_pre_vram_mib")),
            "device_peak_vram_mib": _to_float(payload.get("device_peak_vram_mib")),
            "device_post_vram_mib": _to_float(payload.get("device_post_vram_mib")),
            "device_unattributed_pre_vram_mib": _to_float(
                payload.get("device_unattributed_pre_vram_mib")
            ),
            "device_unattributed_peak_vram_mib": _to_float(
                payload.get("device_unattributed_peak_vram_mib")
            ),
            "device_vram_measurement": str(
                payload.get("device_vram_measurement") or ""
            ).strip(),
            "synthetic_supervisor_activation_mib": _to_float(
                payload.get("synthetic_supervisor_activation_mib")
            ),
            "host_dispatch_memory_window": host_dispatch_window,
            "host_bootstrap_memory_summary": host_bootstrap_summary,
            "peak_memory_mib": peak_vram_mib,
            "active_memory_mib": active_vram_mib,
            "active_memory_scope": str(
                summary.get("active_memory_scope")
                or payload.get("active_memory_scope")
                or payload.get("vram_memory_basis")
                or ""
            ).strip(),
            "active_memory_measurement": str(
                summary.get("active_memory_measurement")
                or payload.get("active_memory_measurement")
                or payload.get("vram_memory_measurement")
                or ""
            ).strip(),
            "memory_basis": str(
                summary.get("memory_basis")
                or payload.get("memory_basis")
                or payload.get("active_memory_scope")
                or payload.get("vram_memory_basis")
                or ""
            ).strip(),
            "peak_fidelity": str(
                summary.get("peak_fidelity") or payload.get("peak_fidelity") or ""
            ).strip(),
            "resident_memory_mib": _to_float(payload.get("resident_memory_mib")),
            "resident_memory_source": str(
                payload.get("resident_memory_source") or ""
            ).strip(),
            "resident_baseline_collected_at": _to_float(
                payload.get("resident_baseline_collected_at")
            ),
            "resident_baseline_lifecycle_token": str(
                payload.get("resident_baseline_lifecycle_token") or ""
            ).strip(),
            "resident_baseline_state": str(
                payload.get("resident_baseline_state") or ""
            ).strip(),
            "memory_qc_keep": (
                summary.get("memory_qc_keep")
                if dispatch_payload
                else _to_bool(
                    payload.get("memory_qc_keep")
                    if "memory_qc_keep" in payload
                    else payload.get("vram_memory_qc_keep"),
                    default=False,
                )
            ),
            "concurrent_execute_overlap": _to_bool(
                summary.get("concurrent_execute_overlap")
                if dispatch_payload
                else payload.get("concurrent_execute_overlap"),
                default=bool(summary.get("concurrent_execute_overlap")),
            ),
            "telemetry_wall_clock_sec": _first(wall_keys),
            "dynamic_batch_argument_applied_k": _to_int(
                payload.get("dynamic_batch_argument_applied_k")
            ),
            "dynamic_batch_consumed_k": _to_int(
                payload.get("dynamic_batch_consumed_k")
            ),
            "dynamic_batch_logical_n": _to_int(payload.get("dynamic_batch_logical_n")),
            "dynamic_batch_group_count": _to_int(
                payload.get("dynamic_batch_group_count")
            ),
        }
    )
    return summary


class GatewayHTTPService:
    supervisor: Any
    """HTTP service that manages task submission and polling."""

    WORKER_NOT_READY_RETRY_DELAY_SEC: float = 0.1
    ACTIVE_CANCEL_RPC_TIMEOUT_SEC: float = 10.0

    def __init__(
        self,
        worker_selector: WorkerSelector,
        work_root: Path | None = None,
        *,
        max_inflight_per_worker: int = 1,
        ensure_worker_ready: Callable[..., Awaitable[None]] | None = None,
        worker_capabilities_resolver: Callable[..., dict[str, Any]] | None = None,
        worker_inventory_provider: Callable[[], list[dict[str, Any]]] | None = None,
        profiling_index_db_path: Path | None = None,
        on_task_start: Callable[..., None] | None = None,
        on_task_end: Callable[..., Awaitable[None]] | None = None,
        resource_tracker: Any | None = None,
        check_worker_status: Callable[[str], str | None] | None = None,
        evict_fn: Callable[[str, int], Awaitable[int]] | None = None,
        adjust_worker_inflight_by_name: Callable[[str, int], None] | None = None,
        bump_dispatch_pending: Callable[[str], None] | None = None,
        release_dispatch_pending: Callable[[str], None] | None = None,
        gateway_identity: Mapping[str, Any] | None = None,
        pipeline_dag: PipelineDAG | None = None,
        max_task_retries: int = 0,
        scheduler_policy: str = "campaign_fifo",
        adapter_retry_backoff: Mapping[str, Any] | None = None,
        dispatch_backlog_per_worker: int = 1,
        dispatch_start_grace_sec: float = 1.0,
        control_plane_overhead_enabled: bool = False,
        control_plane_overhead_top_k: int = 32,
    ) -> None:
        self._scheduler_policy = scheduler_policy
        self.worker_selector = worker_selector
        self._ensure_worker_ready = ensure_worker_ready
        self._worker_capabilities_resolver = worker_capabilities_resolver
        self._worker_inventory_provider = worker_inventory_provider
        self.work_root = work_root.resolve() if work_root else None
        self.max_inflight_per_worker = max(1, int(max_inflight_per_worker))
        self._tasks: dict[str, TaskRecord] = {}
        self._handles: dict[str, Any] = {}
        self._active_regular_tasks: set[str] = set()
        self._core_loop_active_task_ids: set[str] = set()
        self._core_loop_active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._record_by_campaign_nf: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()
        self._core_loop_init_lock = asyncio.Lock()
        self._worker_limits: dict[str, asyncio.Semaphore] = {}
        self.dispatch_backlog_per_worker = max(0, int(dispatch_backlog_per_worker))
        self.dispatch_start_grace_sec = max(0.0, float(dispatch_start_grace_sec))
        self._front_slot_lease_lock = threading.RLock()
        self._front_slot_leases: dict[str, dict[str, float]] = {}
        self._front_slot_lease_ttl_s = 0.0
        self._admission_sem = asyncio.Semaphore(16)
        self._channel_pool: dict[str, grpc.aio.Channel] = {}
        self._on_task_start = on_task_start
        self._on_task_end = on_task_end
        self.resource_tracker = resource_tracker
        self._check_worker_status = check_worker_status
        self.evict_fn = evict_fn
        self._adjust_worker_inflight_by_name = adjust_worker_inflight_by_name
        self._bump_dispatch_pending = bump_dispatch_pending
        self._release_dispatch_pending = release_dispatch_pending
        self._campaign_identity_metrics: dict[str, int] = {
            "accepted": 0,
            "unassigned": 0,
            "rejected": 0,
        }
        self._gateway_identity = gateway_payload_from_mapping(gateway_identity)
        if profiling_index_db_path is None:
            index_path = (
                _DEFAULT_INDEX_DB.parent / f"profile_runs_{uuid.uuid4().hex}.sqlite3"
            ).resolve()
        else:
            index_path = Path(profiling_index_db_path).expanduser().resolve()
        self._run_index = RunIndexStore(index_path)
        self._sqlite_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sqlite-"
        )
        self._signal_service = SignalService(run_index=self._run_index)
        self._control_plane_overhead = ControlPlaneOverheadRecorder(
            enabled=bool(control_plane_overhead_enabled),
            top_k=int(control_plane_overhead_top_k),
        )
        self._control_plane_overhead_enabled = bool(
            self._control_plane_overhead.enabled,
        )
        self._jobs = JobService(
            submit_task=self._submit_task_impl,
            get_task=self._get_task_impl,
            cancel_task=self._cancel_task_impl,
        )
        self._planner = PlannerService(
            self, pipeline_dag=pipeline_dag, scheduler_policy=self._scheduler_policy
        )
        self._dispatcher = DispatchService(self)
        self._dispatcher._on_inference_start = self._on_inference_start
        self._max_task_retries = max(0, int(max_task_retries))
        _retry_backoff_cfg = dict(adapter_retry_backoff or {})
        self._adapter_retry_backoff_enabled = bool(
            _retry_backoff_cfg.get("enabled", False),
        )
        self._adapter_retry_backoff_base_s = max(
            0.0,
            float(_retry_backoff_cfg.get("base_s", 0.0)),
        )
        self._adapter_retry_backoff_factor = max(
            1.0,
            float(_retry_backoff_cfg.get("factor", 1.0)),
        )
        self._adapter_retry_backoff_max_s = max(
            self._adapter_retry_backoff_base_s,
            float(_retry_backoff_cfg.get("max_s", self._adapter_retry_backoff_base_s)),
        )
        self._delayed_retry_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._reality_validator = None
        self._last_safety_net_gc: float = (
            0.0
        )
        self._sweeper_started: bool = (
            False
        )
        self._gpu_health_observer = None
        self._scheduling_supervisor = None
        self._event_handler = None

    def _compute_max_refinements(self) -> int:
        """Plan — dynamic inner-loop
        bound derived from the constraint lattice size.

            max = gpus × 3 + workers × 2 + components × gpus
                  + min(50, active_correlations_count() + 10)

        Per Plan (), no fallback
        constant is allowed; raises if the scenario is not initialised.
        The correlation term scales with live root-cause merges bounded
        at 50 so a burst of correlation fallouts during a storm does
        not explode the inner-loop budget.
        """
        scenario = None
        planner = getattr(self, "_planner", None)
        if planner is not None:
            cs = getattr(planner, "campaign_scheduler", None)
            if cs is not None:
                scenario = getattr(cs, "_timelines", None)
        if scenario is None:
            raise RuntimeError(
                "compute_max_refinements: scenario unavailable "
                "(plan  forbids fallback constants)"
            )
        try:
            gpus = len(list(scenario.gpu_ids)) or 1
        except Exception:
            gpus = 1
        sup = getattr(self, "supervisor", None)
        workers = len(getattr(sup, "states", {})) if sup is not None else gpus
        components = (
            max(1, len(getattr(planner, "_components", []) or [])) if planner else 1
        )
        active_corr = 0
        gp = getattr(planner, "_global_planner", None) if planner else None
        ct = getattr(gp, "_constraint_tracker", None) if gp else None
        if ct is not None:
            try:
                active_corr = int(ct.active_correlations_count())
            except Exception:
                active_corr = 0
        correlation_term = min(50, active_corr + 10)
        return gpus * 3 + workers * 2 + components * gpus + correlation_term

    def _cold_start_activation_mb(self, addr: str) -> int:
        """Weight-proportional activation estimate when GP data is unavailable.

        Prevents unlimited concurrent dispatch on cold start by reserving
        ``memory_reserved_mb * cold_start_activation_ratio`` as estimated
        activation VRAM.  Once GP data accumulates, the GP prediction takes
        priority and this fallback is never reached.
        """
        supervisor = getattr(self, "supervisor", None)
        if not supervisor:
            return 0
        ratio = getattr(supervisor, "cold_start_activation_ratio", 1.0)
        for st in supervisor.states.values():
            memory_reserved_mb = int(getattr(st, "memory_reserved_mb", 0) or 0)
            if getattr(st, "addr", "") == addr and memory_reserved_mb > 0:
                return int(memory_reserved_mb * ratio)
        return 0

    def _supports_active_cancel(self, component: str, *, worker_name: str = "") -> bool:
        """Return whether active CancelBatch is safe for this worker/component.

        The persistent-actor backend hard-cancellable for every adapter
        (``WorkerEngine.supports_cancel_batch`` is a constant True), so the
        capability does not depend on the worker's runtime state.  A cold /
        unresolved worker must therefore default to cancel-safe instead of
        returning False, otherwise pending backfill on a cold GPU can never
        use the cooperative-cancel tier and starves.
        """
        resolver = self._worker_capabilities_resolver
        if resolver is None:
            return True
        try:
            caps = resolver(str(component or ""), str(worker_name or ""))
        except Exception as exc:
            if not isinstance(exc, TypeError):
                return True
            try:
                caps = resolver(str(component or ""))
            except Exception:
                return True
        if not isinstance(caps, Mapping):
            return True
        return _to_bool(caps.get("supports_cancel_batch"), default=True)

    def _on_inference_start(
        self,
        task_id: str,
        component: str,
        gpu_id: str,
        campaign_id: str = "",
        was_cold_start: bool = False,
        input_size: float = 0.0,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        worker_name: str = "",
        active_cancel_safe: bool = False,
        reciprocal_interference: Mapping[str, Any] | None = None,
        reciprocal_event_id: str = "",
        gpu_model: str = "",
        mps_mode: str = "",
        worker_backend: str = "",
        actor_model: str = "",
        adapter_version: str = "",
    ) -> dict[str, Any] | None:
        """Atomically transition one launch-ready prediction to exec-in."""
        _LOG.info("[exec_in] %s %s on GPU %s", task_id[:12], component, gpu_id)
        scheduler = self._planner.campaign_scheduler
        cq = scheduler._campaign_queues.get(campaign_id)
        campaign_snapshot = None
        if cq is not None:
            campaign_snapshot = (
                cq,
                int(cq._active_count),
                int(cq._pending_count),
                int(cq._completed_count),
                set(cq._active_ids),
                set(cq._pending_ids),
                set(cq._dispatched_components),
                dict(scheduler._pending_pre_inits),
                set(scheduler._fired_pre_inits),
            )
        global_planner = getattr(self._planner, "_global_planner", None)
        preparation = None
        reciprocal_enabled = bool(
            getattr(global_planner, "reciprocal_interference_correction", False)
        )
        if reciprocal_enabled:
            if (
                not isinstance(reciprocal_interference, Mapping)
                or not reciprocal_interference
            ):
                raise ValueError("reciprocal handoff metadata missing")
            reciprocal_payload = dict(reciprocal_interference)
        else:
            reciprocal_payload = None
        if reciprocal_payload is not None:
            prepare = getattr(global_planner, "prepare_reciprocal_dispatch_start", None)
            if not callable(prepare):
                raise RuntimeError("reciprocal planner prepare hook missing")
            preparation = cast(
                Any,
                prepare(
                    task_id,
                    reciprocal_event_id,
                    component,
                    gpu_id,
                    reciprocal_payload,
                ),
            )
            if preparation.duplicate:
                return {"duplicate": True, "preparation": preparation}
        tracker = getattr(global_planner, "_constraint_tracker", None)
        tracker_primary_snapshot = (
            set(getattr(tracker, "_primary_dispatched_components", set()))
            if tracker is not None
            else None
        )
        rollback = {
            "preparation": preparation,
            "campaign_snapshot": campaign_snapshot,
            "tracker": tracker,
            "tracker_primary_snapshot": tracker_primary_snapshot,
            "planner_event": None,
        }
        try:
            is_backfill = False
            if cq:
                is_backfill = (
                    not scheduler._is_primary(campaign_id)
                    and len(scheduler._ordered_campaigns()) > 1
                )
            scheduler.on_task_dispatched(
                task_id=task_id,
                campaign_id=campaign_id,
                component=component,
                gpu_id=gpu_id,
                input_size=float(input_size or 0.0),
                config_fingerprint=str(config_fingerprint or "").strip(),
                is_backfill=is_backfill,
                was_cold_start=was_cold_start,
                worker_name=str(worker_name or "").strip(),
                active_cancel_safe=bool(active_cancel_safe),
            )
            if preparation is not None and reciprocal_payload is not None:
                _cp_start = self._cp_begin()
                commit = getattr(
                    global_planner, "commit_reciprocal_dispatch_start", None
                )
                if not callable(commit):
                    raise RuntimeError("reciprocal planner commit hook missing")
                applied_incumbents, applied_total_sec = cast(
                    tuple[int, float],
                    commit(
                        preparation,
                        reciprocal_payload,
                        config_fingerprint=config_fingerprint,
                        input_fingerprint=input_fingerprint,
                        gpu_model=gpu_model,
                        mps_mode=mps_mode,
                        worker_backend=worker_backend,
                        actor_model=actor_model,
                        adapter_version=adapter_version,
                    ),
                )
                self._cp_record(
                    "dispatch_reciprocal_reprice",
                    _cp_start,
                    component=component,
                    gpu_id=gpu_id,
                    applied_incumbents=applied_incumbents,
                    applied_total_sec=applied_total_sec,
                    version_retried=bool(preparation.version_retried),
                    scanned_entries=int(
                        reciprocal_payload.get("commit_scanned_entries", 0)
                    ),
                    temporal_segments=int(
                        reciprocal_payload.get("commit_temporal_segments", 0)
                    ),
                    interference_lookups=int(
                        reciprocal_payload.get("commit_interference_lookups", 0)
                    ),
                )
            predicted_vram = scheduler._predict_vram(
                component,
                float(input_size or 0.0),
                gpu_id,
                config_fingerprint=str(config_fingerprint or "").strip(),
            )
            self._planner.on_task_dispatched_event(gpu_id, component, predicted_vram)
            rollback["planner_event"] = (gpu_id, component, predicted_vram)
            return rollback
        except Exception:
            self._rollback_inference_start(rollback)
            _LOG.error(
                "[exec_in] on_task_dispatched failed for %s — aborting dispatch",
                task_id[:12],
                exc_info=True,
            )
            raise

    def _rollback_inference_start(self, rollback: Mapping[str, Any] | None) -> None:
        """Restore pre-exec-in scheduler/timeline/resource state."""
        if not rollback or rollback.get("duplicate"):
            return
        planner_event = rollback.get("planner_event")
        if planner_event:
            with contextlib.suppress(Exception):
                self._planner.on_task_complete_event(*planner_event)
        preparation = rollback.get("preparation")
        global_planner = getattr(self._planner, "_global_planner", None)
        if preparation is not None and global_planner is not None:
            restore = getattr(
                global_planner, "rollback_reciprocal_dispatch_start", None
            )
            if callable(restore):
                restore(preparation)
        snapshot = cast(Any, rollback.get("campaign_snapshot"))
        if snapshot is not None:
            (
                cq,
                cq._active_count,
                cq._pending_count,
                cq._completed_count,
                active_ids,
                pending_ids,
                dispatched_components,
                pending_pre_inits,
                fired_pre_inits,
            ) = snapshot
            cq._active_ids = active_ids
            cq._pending_ids = pending_ids
            cq._dispatched_components = dispatched_components
            scheduler = self._planner.campaign_scheduler
            scheduler._pending_pre_inits = pending_pre_inits
            scheduler._fired_pre_inits = fired_pre_inits
        tracker = rollback.get("tracker")
        tracker_primary_snapshot = rollback.get("tracker_primary_snapshot")
        if tracker is not None and tracker_primary_snapshot is not None:
            tracker._primary_dispatched_components = tracker_primary_snapshot

    def _complete_campaign_task(
        self,
        *,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
        execution_attempt_id: str = "",
        from_eviction: bool = False,
    ) -> bool:
        """Apply one attempt-scoped exit and reconcile its affected GPU."""
        global_planner = getattr(self._planner, "_global_planner", None)
        is_current = getattr(global_planner, "is_reciprocal_attempt_current", None)
        if callable(is_current) and not is_current(
            task_id,
            execution_attempt_id,
            gpu_id,
        ):
            _LOG.info(
                "[task-exit] stale attempt ignored task=%s event=%s gpu=%s",
                task_id[:12],
                execution_attempt_id,
                gpu_id,
            )
            return False
        if self._event_handler is not None and not from_eviction:
            self._event_handler.on_task_complete(
                task_id=task_id,
                campaign_id=campaign_id,
                component=component,
                gpu_id=gpu_id,
            )
        else:
            self._planner.campaign_scheduler.on_task_complete(
                task_id=task_id,
                campaign_id=campaign_id,
                component=component,
                gpu_id=gpu_id,
                from_eviction=from_eviction,
            )
        reconcile = getattr(global_planner, "reconcile_reciprocal_event", None)
        if bool(
            getattr(
                global_planner,
                "reciprocal_interference_correction",
                False,
            )
        ) and callable(reconcile):
            reconcile([gpu_id], "task_exit")
        finish = getattr(global_planner, "finish_reciprocal_attempt", None)
        if callable(finish) and execution_attempt_id:
            finish(task_id, execution_attempt_id, gpu_id)
        return True

    def register_routes(self, app: web.Application) -> None:
        app.router.add_post("/api/v1/job/submit", self.submit_job)
        app.router.add_get("/api/v1/job/{job_id}", self.get_job)
        app.router.add_post("/api/v1/job/cancel/{job_id}", self.cancel_job)
        app.router.add_post("/api/v1/campaign/complete", self.mark_campaign_complete)
        app.router.add_get("/api/v1/profile/runs", self.get_profile_runs)
        app.router.add_get("/api/v1/profile/runs/{run_key}", self.get_profile_run)
        app.router.add_get("/api/v1/campaigns", self.get_campaigns)
        app.router.add_get("/api/v1/campaigns/{campaign_id}", self.get_campaign_detail)
        app.router.add_get("/api/v1/telemetry/health", self.get_telemetry_health)
        app.router.add_get("/api/v1/model/versions", self.get_model_versions)
        app.router.add_get("/api/v1/model/components", self.get_model_components)
        app.router.add_get(
            "/api/v1/model/components/{component}", self.get_model_component
        )
        app.router.add_get(
            "/api/v1/model/components/{component}/versions",
            self.get_model_component_versions,
        )
        app.router.add_get("/api/v1/workers", self.get_workers)
        app.router.add_get("/api/v1/signals", self.get_signals)
        app.router.add_get("/api/v1/debug/tasks", self.get_debug_tasks)
        app.router.add_get("/api/v1/profile/export", self.get_profile_export)
        app.router.add_post(
            "/api/v1/admin/runtime_state/snapshot",
            self.admin_snapshot_runtime_state,
        )
        app.router.add_post(
            "/api/v1/admin/control_plane_overhead/snapshot",
            self.admin_snapshot_control_plane_overhead,
        )
        app.router.add_get("/health", self.health)
        app.router.add_post(
            "/api/v1/admin/gpu/{gpu_id}/confirm_reset",
            self.admin_confirm_reset,
        )
        app.router.add_post(
            "/api/v1/admin/gpu/{gpu_id}/confirm_rma",
            self.admin_confirm_rma,
        )

    async def admin_confirm_reset(self, request: web.Request) -> web.Response:
        """Operator-driven GPU reset confirmation (Plan 
        Trigger 2).  Clears ``_gpu_reset_required`` on the target GPU;
        Xid 64 RMA-qualifying GPUs additionally require
        ``admin_confirm_rma``.
        """
        gpu_id = request.match_info.get("gpu_id", "")
        gp = getattr(self._planner, "_global_planner", None)
        if gp is None or not hasattr(gp, "on_reset_completed"):
            return web.json_response(
                {"ok": False, "error": "GlobalPlanner not initialized"},
                status=503,
            )
        try:
            gp.on_reset_completed(str(gpu_id))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {"ok": True, "gpu_id": gpu_id, "action": "reset_confirmed"}
        )

    async def admin_confirm_rma(self, request: web.Request) -> web.Response:
        """Operator-driven RMA / Field-Diagnostic completion signal
        (Plan Trigger 2 +  P0-SAFETY 2-signal
        invariant).  Required in addition to ``confirm_reset`` for
        Xid 64 (row-remapping failure) GPUs."""
        gpu_id = request.match_info.get("gpu_id", "")
        gp = getattr(self._planner, "_global_planner", None)
        if gp is None or not hasattr(gp, "admin_confirm_rma_complete"):
            return web.json_response(
                {"ok": False, "error": "GlobalPlanner not initialized"},
                status=503,
            )
        try:
            gp.admin_confirm_rma_complete(str(gpu_id))
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {"ok": True, "gpu_id": gpu_id, "action": "rma_confirmed"}
        )

    _GRPC_CHANNEL_OPTIONS = (
        ("grpc.keepalive_time_ms", 2**31 - 1),
        ("grpc.keepalive_permit_without_calls", 0),
    )

    def _get_channel(self, addr: str) -> grpc.aio.Channel:
        channel = self._channel_pool.get(addr)
        if channel is None:
            channel = grpc.aio.insecure_channel(
                addr, options=self._GRPC_CHANNEL_OPTIONS
            )
            self._channel_pool[addr] = channel
        return channel

    def invalidate_channel(self, addr: str) -> None:
        """Remove a cached gRPC channel (called when worker addr changes)."""
        channel = self._channel_pool.pop(addr, None)
        if channel is not None:
            asyncio.create_task(channel.close())

    def _resolve_submit_argv(
        self,
        body: Mapping[str, Any],
    ) -> tuple[list[str], str | None]:
        try:
            argv = _normalize_argv(body.get("argv"))
            tool_cwd = _normalize_tool_cwd(body.get("tool_cwd"))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return argv, tool_cwd

    def _record_campaign_identity_metric(self, key: str) -> None:
        if key not in self._campaign_identity_metrics:
            return
        self._campaign_identity_metrics[key] = (
            int(self._campaign_identity_metrics.get(key, 0)) + 1
        )

    @staticmethod
    def _campaign_metadata_version_for_identity(campaign_id: str) -> int:
        return 1 if str(campaign_id or "").strip() else 0

    def _register_campaign_fan_out(
        self,
        component: str,
        output_sample_count: int | None,
        campaign_id: str | None,
    ) -> None:
        """Register campaign-local fan-out and refresh scheduler DAG context.

        ``CampaignQueue.dag_context`` is a snapshot.  If we only update
        ``PipelineDAG`` after the queue has been created, completion checks can
        keep seeing ``fan_out_map[component] is None`` forever and a finished
        Nextflow campaign remains primary.  Keep the two structures in sync at
        every fan-out registration site.
        """
        if output_sample_count is None:
            return
        try:
            osc = int(output_sample_count)
        except (TypeError, ValueError):
            return
        if osc < 1:
            return
        comp = str(component or "").strip().lower()
        if not comp:
            return
        pipeline_dag = getattr(getattr(self, "_planner", None), "pipeline_dag", None)
        if pipeline_dag is None:
            raise RuntimeError("Gateway planner is missing pipeline_dag")
        pipeline_dag.register_fan_out(comp, osc, campaign_id=campaign_id)
        cid = str(campaign_id or "").strip()
        if not cid:
            return
        campaign_scheduler = getattr(
            getattr(self, "_planner", None),
            "campaign_scheduler",
            None,
        )
        if campaign_scheduler is not None and hasattr(
            campaign_scheduler, "update_dag_context"
        ):
            campaign_scheduler.update_dag_context(
                cid,
                pipeline_dag.get_dag_context(campaign_id),
            )

    async def _submit_task_impl(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        component = str(body.get("component") or "").strip().lower()
        workdir = str(body.get("workdir") or "")
        nf_task_id = str(body.get("nf_task_id") or "")

        if not component:
            raise ValueError("component is required")
        if not workdir:
            raise ValueError("workdir is required")
        if not os.path.isabs(workdir):
            raise ValueError("workdir must be an absolute path")

        workdir_path = Path(workdir).resolve()
        if self.work_root and not _is_within(self.work_root, workdir_path):
            raise ValueError("workdir not under work_root")

        try:
            env = _normalize_env(body.get("env", {}))
            timeout_s = _normalize_timeout_s(body.get("timeout_s", 300), default=300)
            profiling = _normalize_profiling(body.get("profiling"))
            workload_features_raw = body.get("workload_features", {})
            if workload_features_raw is None:
                workload_features_raw = {}
            if not isinstance(workload_features_raw, Mapping):
                raise ValueError("workload_features must be an object when provided")
            workload_features = canonicalize_axes(dict(workload_features_raw))
            config_fingerprint = str(body.get("config_fingerprint") or "").strip()
            input_fingerprint = str(body.get("input_fingerprint") or "").strip()
            workflow_completion_expected = bool(
                body.get("workflow_completion_expected", False)
            )
            campaign_id = _normalize_optional_identity_field(
                body.get("campaign_id"), "campaign_id"
            )
            run_name = _normalize_optional_identity_field(
                body.get("run_name"), "run_name"
            )
            submitter = _normalize_optional_identity_field(
                body.get("submitter"), "submitter"
            )
            campaign_metadata_version = self._campaign_metadata_version_for_identity(
                campaign_id
            )
            if campaign_metadata_version >= 1:
                self._record_campaign_identity_metric("accepted")
            else:
                self._record_campaign_identity_metric("unassigned")
            execution_overrides_raw = body.get("execution_overrides", {})
            if execution_overrides_raw is None:
                execution_overrides_raw = {}
            if not isinstance(execution_overrides_raw, Mapping):
                raise ValueError("execution_overrides must be an object when provided")
            execution_overrides: dict[str, Any] = {}
            override_batch_size = _normalize_optional_positive_int(
                execution_overrides_raw.get("batch_size"),
                "execution_overrides.batch_size",
            )
            if override_batch_size is not None:
                execution_overrides["batch_size"] = int(override_batch_size)
            preferred_worker_addr = (
                str(body.get("preferred_worker_addr") or "").strip() or None
            )
            raw_preferred_gpu_ids = body.get("preferred_gpu_ids") or []
            if raw_preferred_gpu_ids is None:
                raw_preferred_gpu_ids = []
            if not isinstance(raw_preferred_gpu_ids, list):
                raise ValueError("preferred_gpu_ids must be a list of strings")
            preferred_gpu_ids = [
                str(item).strip() for item in raw_preferred_gpu_ids if str(item).strip()
            ]
        except ValueError as exc:
            self._record_campaign_identity_metric("rejected")
            raise ValueError(str(exc)) from exc

        argv, tool_cwd = self._resolve_submit_argv(body)

        process_name = str(body.get("process_name") or "").strip() or None
        dag_topology_raw = body.get("dag_topology")
        if isinstance(dag_topology_raw, dict) and dag_topology_raw:
            self._planner.pipeline_dag.register_topology(dag_topology_raw)
        if process_name and component:
            self._planner.pipeline_dag.register_process_component(
                process_name, component
            )
        _raw_osc = workload_features.get("output_sample_count")
        output_sample_count: int | None = None
        if _raw_osc is not None:
            try:
                _osc_int = int(_raw_osc)
                if _osc_int > 0:
                    output_sample_count = _osc_int
            except (ValueError, TypeError):
                output_sample_count = None
        self._register_campaign_fan_out(
            component,
            output_sample_count,
            campaign_id,
        )

        _supersede_key: tuple[str, str] | None = (
            (str(campaign_id), str(nf_task_id))
            if (campaign_id and nf_task_id)
            else None
        )
        task_id = str(uuid.uuid4())
        record = TaskRecord(
            task_id=task_id,
            nf_task_id=nf_task_id,
            component=component,
            workdir=str(workdir_path),
            state=TASK_STATE_SUBMITTED,
            created_at=time.time(),
            updated_at=time.time(),
            campaign_id=campaign_id,
            run_name=run_name,
            submitter=submitter,
            campaign_metadata_version=campaign_metadata_version,
            workload_features=workload_features,
            config_fingerprint=config_fingerprint,
            input_fingerprint=input_fingerprint,
            workflow_completion_expected=workflow_completion_expected,
        )

        _cleanup_prev_task_id: str | None = None
        async with self._lock:
            if _supersede_key is not None:
                _prev_task_id = self._record_by_campaign_nf.get(_supersede_key)
                _prev_record = self._tasks.get(_prev_task_id) if _prev_task_id else None
                if _prev_record is not None and _prev_task_id is None:
                    raise RuntimeError("task index returned a record without a task id")
                _prev_task_id_str = str(_prev_task_id or "")
                if _prev_record is not None and self._is_same_submit_attempt(
                    _prev_record,
                    workdir_path,
                ):
                    _LOG.info(
                        "[] idempotent submit replay task=%s campaign=%s nf_task=%s",
                        _prev_task_id_str[:8],
                        (campaign_id or "")[:8],
                        nf_task_id,
                    )
                    return {
                        "task_id": _prev_task_id_str,
                        "state": STATE_NAMES.get(
                            int(
                                getattr(_prev_record, "state", TASK_STATE_SUBMITTED)
                                or TASK_STATE_SUBMITTED
                            ),
                            "SUBMITTED",
                        ),
                    }
                if _prev_record is not None and int(
                    getattr(_prev_record, "state", 0) or 0
                ) in (
                    TASK_STATE_SUBMITTED,
                    TASK_STATE_RUNNING,
                ):
                    _LOG.warning(
                        "[] live nf_task_id collision preserved prev_task=%s "
                        "campaign=%s nf_task=%s prev_workdir=%s new_workdir=%s",
                        _prev_task_id_str[:8],
                        (campaign_id or "")[:8],
                        nf_task_id,
                        getattr(_prev_record, "workdir", ""),
                        str(workdir_path),
                    )
                elif _prev_task_id:
                    _cleanup_prev_task_id = _prev_task_id

            self._tasks[task_id] = record
            self._active_regular_tasks.add(task_id)
            if _supersede_key is not None:
                self._record_by_campaign_nf[_supersede_key] = task_id
            try:
                cs = getattr(
                    getattr(self, "_planner", None),
                    "campaign_scheduler",
                    None,
                )
                if cs is not None and hasattr(cs, "note_task_submitted"):
                    cs.note_task_submitted(record)
            except Exception:
                _LOG.warning(
                    "[patch-v4-04] note_task_submitted failed "
                    "(task_id=%s campaign_id=%s)",
                    task_id,
                    campaign_id,
                    exc_info=True,
                )

        if _cleanup_prev_task_id:
            await self._cleanup_superseded_record(
                _cleanup_prev_task_id,
                reason="nf_retry_supersede",
            )

        if campaign_metadata_version >= 1:
            try:
                await self._run_index_call(
                    self._run_index.upsert_campaign_registry,
                    campaign_id=campaign_id,
                    run_name=run_name,
                    submitter=submitter,
                    gateway_instance_id=str(
                        self._gateway_identity.get("instance_id") or ""
                    ),
                    gateway_bind_addr=str(
                        self._gateway_identity.get("bind_addr") or ""
                    ),
                    gateway_git_commit=str(
                        self._gateway_identity.get("git_commit") or ""
                    ),
                    gateway_started_at=_to_float(
                        self._gateway_identity.get("started_at")
                    ),
                    status="waiting",
                    created_at=record.created_at,
                    last_seen_at=record.updated_at,
                    event_ts=record.updated_at,
                )
            except Exception as exc:
                _LOG.warning(
                    "[task %s] non-fatal campaign registry persistence failure "
                    "during submit: %s",
                    task_id,
                    exc,
                )

        await self._ensure_core_loop_components()
        handle = self._build_task_handle(
            record,
            argv=argv,
            tool_cwd=tool_cwd,
            env=env,
            timeout_s=timeout_s,
            profiling=profiling,
            workload_features=workload_features,
            config_fingerprint=config_fingerprint,
            input_fingerprint=input_fingerprint,
            execution_overrides=execution_overrides,
            preferred_worker_addr=preferred_worker_addr,
            preferred_gpu_ids=preferred_gpu_ids,
            output_sample_count=output_sample_count,
            process_name=process_name,
            is_backfill_hint=False,
        )
        if self._scheduling_supervisor is None:
            raise RuntimeError(
                "SchedulingSupervisor invariant violation: ensure_core_loop_components "
                "did not produce a supervisor before arrival-site submit",
            )
        self._scheduling_supervisor.submit(handle, reason="arrival")

        return {
            "task_id": task_id,
            "state": STATE_NAMES[record.state],
        }

    async def _get_task_impl(self, task_id: str) -> Mapping[str, Any] | None:
        normalized = str(task_id or "").strip()
        if not normalized:
            raise ValueError("task_id is required")
        async with self._lock:
            record = self._tasks.get(normalized)

        if not record:
            return None

        return {
            "task_id": record.task_id,
            "nf_task_id": record.nf_task_id,
            "component": record.component,
            "campaign_id": record.campaign_id,
            "run_name": record.run_name,
            "submitter": record.submitter,
            "state": STATE_NAMES[record.state],
            "ok": record.ok,
            "exit_code": record.exit_code,
            "message": record.message,
            "worker_timing_us": record.worker_timing_us,
            "dispatch": {
                "worker_addr": record.dispatch_worker_addr,
                "gpu_ids": list(record.dispatch_gpu_ids),
                "worker_name": record.dispatch_worker_name,
                "worker_generation_token": record.worker_generation_token,
                "run_ordinal_in_generation": record.run_ordinal_in_generation,
                "is_first_real_run": bool(record.is_first_real_run),
            },
            "workflow_completion_expected": bool(record.workflow_completion_expected),
        }

    COOPERATIVE_CANCEL_REASONS = frozenset(
        {
            "backfill_eviction",
            "drift_eviction",
            "mcpse_eviction",
            "primary_protection",
            "worker_killed_reenqueue",
        }
    )

    async def _cancel_task_impl(self, task_id: str, *, reason: str = "") -> bool:
        _LOG.info(
            "[cancel_task] task=%s reason=%r", task_id[:12] if task_id else "?", reason
        )
        normalized = str(task_id or "").strip()
        if not normalized:
            raise ValueError("task_id is required")
        async with self._lock:
            record = self._tasks.get(normalized)
            handle = self._handles.get(normalized)

        if not record:
            return False

        if record.state in {
            TASK_STATE_SUCCEEDED,
            TASK_STATE_FAILED,
            TASK_STATE_CANCELLED,
        }:
            return True

        cooperative_cancel = reason in self.COOPERATIVE_CANCEL_REASONS
        inner = getattr(handle, "_current_inner", None) if handle is not None else None
        had_active_driver = inner is not None and not inner.done()
        if cooperative_cancel and had_active_driver and reason:
            record._cancel_reason = reason

        _batch_id = getattr(record, "_dispatched_batch_id", None)
        _worker_addr = getattr(record, "_dispatched_worker_addr", None)
        _has_inflight_worker = bool(_batch_id is not None and _worker_addr)
        _worker_cancel_accepted = False
        _worker_cancel_error = ""
        if _batch_id is not None and _worker_addr:
            try:
                _ch = self._get_channel(_worker_addr)
                _stub = pb_grpc.ModelWorkerStub(_ch)
                _cancel_resp = await _stub.CancelBatch(
                    pb.CancelBatchRequest(batch_id=_batch_id),
                    timeout=self.ACTIVE_CANCEL_RPC_TIMEOUT_SEC,
                )
                _worker_cancel_accepted = bool(getattr(_cancel_resp, "ok", False))
                if not _worker_cancel_accepted:
                    _worker_cancel_error = str(
                        getattr(_cancel_resp, "message", "") or "worker rejected cancel"
                    )
            except Exception as exc:
                _worker_cancel_error = str(exc)

        if cooperative_cancel:
            if had_active_driver:
                if _has_inflight_worker and not _worker_cancel_accepted:
                    if _worker_cancel_error == "batch not found":
                        _LOG.info(
                            "[cancel_task] cooperative cancel outcome pending "
                            "task=%s reason=%s batch=%s worker=%s",
                            normalized[:12],
                            reason,
                            str(_batch_id or "")[:12],
                            _worker_addr,
                        )
                        return True
                    if getattr(record, "_cancel_reason", "") == reason:
                        record._cancel_reason = ""
                    _LOG.warning(
                        "[cancel_task] active cooperative cancel rejected "
                        "task=%s reason=%s batch=%s worker=%s error=%s",
                        normalized[:12],
                        reason,
                        str(_batch_id or "")[:12],
                        _worker_addr,
                        _worker_cancel_error or "adapter is not cancel-safe",
                    )
                    return False
                if record.state in {
                    TASK_STATE_SUCCEEDED,
                    TASK_STATE_FAILED,
                    TASK_STATE_CANCELLED,
                }:
                    if getattr(record, "_cancel_reason", "") == reason:
                        record._cancel_reason = ""
                    return True
                if handle is None:
                    raise RuntimeError("active driver has no task handle")
                handle.cancel()
                return True

            if not had_active_driver:
                if _has_inflight_worker:
                    _LOG.warning(
                        "[cancel_task] stale in-flight marker without active "
                        "driver task=%s reason=%s batch=%s worker=%s",
                        normalized[:12],
                        reason,
                        str(_batch_id or "")[:12],
                        _worker_addr,
                    )
                    self._clear_record_dispatch_inflight(record)
                self._detach_precomputed_plan_from_handle(
                    handle,
                    reason="cooperative_pre_run_cancel",
                )
                if handle is None:
                    handle = self._rebuild_task_handle_from_record(record)
                if handle is None:
                    if reason:
                        record._cancel_reason = ""
                    _LOG.error(
                        "[cancel_task] no-inflight cooperative cancel has no "
                        "TaskHandle/context task=%s reason=%s",
                        normalized[:12],
                        reason,
                    )
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=(
                            "cooperative cancel could not requeue missing "
                            f"task handle/context: {reason}"
                        ),
                    )
                    async with self._lock:
                        self._active_regular_tasks.discard(normalized)
                    return False
                record._cancel_reason = ""
                if handle is None:
                    raise RuntimeError("requeue handle vanished after reconstruction")
                handle._cancel_requested = False
                handle._not_before_at = 0.0
                await self._mark_record_pending_after_skip(record)
                supervisor = getattr(self, "_scheduling_supervisor", None)
                if supervisor is None:
                    ensure_core = getattr(
                        self,
                        "_ensure_core_loop_components",
                        None,
                    )
                    if callable(ensure_core):
                        try:
                            await cast(Callable[[], Awaitable[Any]], ensure_core)()
                        except Exception:
                            _LOG.warning(
                                "[cancel_task] failed to initialize "
                                "SchedulingSupervisor for no-inflight "
                                "cooperative cancel task=%s reason=%s",
                                normalized[:12],
                                reason,
                                exc_info=True,
                            )
                    supervisor = getattr(self, "_scheduling_supervisor", None)
                if supervisor is None:
                    _LOG.error(
                        "[cancel_task] no-inflight cooperative cancel has no "
                        "SchedulingSupervisor task=%s reason=%s",
                        normalized[:12],
                        reason,
                    )
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=(
                            "cooperative cancel could not requeue without "
                            f"SchedulingSupervisor: {reason}"
                        ),
                    )
                    async with self._lock:
                        self._active_regular_tasks.discard(normalized)
                    return False
                ensure = getattr(
                    supervisor,
                    "ensure_handle_requeued_or_woken",
                    None,
                )
                if callable(ensure):
                    action = ensure(handle, reason="eviction_reenqueue")
                else:
                    supervisor.submit(handle, reason="eviction_reenqueue")
                    action = "submitted_legacy"
                log_fn = (
                    _LOG.info
                    if action
                    in {
                        "recovered_stale_owned",
                        "submitted",
                        "dropped",
                    }
                    else _LOG.debug
                )
                log_fn(
                    "[cancel_task] no-inflight cooperative cancel "
                    "task=%s reason=%s action=%s",
                    normalized[:12],
                    reason,
                    action,
                )
                return True

        if reason:
            record._cancel_reason = reason
        if handle is not None:
            self._detach_precomputed_plan_from_handle(
                handle,
                reason="terminal_pre_run_cancel",
            )
            handle.cancel()

        _LOG.warning(
            "[cancel_task] setting CANCELLED for task=%s (reason was empty)",
            normalized[:12],
        )
        await self._set_state(
            record,
            TASK_STATE_CANCELLED,
            ok=False,
            exit_code=1,
            message="cancelled",
        )
        async with self._lock:
            self._active_regular_tasks.discard(normalized)
        return True

    async def health(self, request: web.Request) -> web.Response:
        """GET /health - Health check endpoint."""
        return web.json_response(
            {"status": "ok", "gateway": dict(self._gateway_identity)}
        )

    def _cp_begin(self) -> int:
        if not getattr(self, "_control_plane_overhead_enabled", False):
            return 0
        try:
            return self._control_plane_overhead.begin()
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
        if not start_ns or not getattr(self, "_control_plane_overhead_enabled", False):
            return
        try:
            self._control_plane_overhead.record_since(
                phase,
                start_ns,
                active_wall=active_wall,
                **attrs,
            )
        except Exception:
            return

    def _cp_count(self, counter: str, value: int = 1, **attrs: Any) -> None:
        if not getattr(self, "_control_plane_overhead_enabled", False):
            return
        try:
            self._control_plane_overhead.increment(counter, value, **attrs)
        except Exception:
            return

    def _cp_record_reciprocal_scoring(self, plan: Any) -> None:
        if not getattr(self, "_control_plane_overhead_enabled", False):
            return
        reciprocal = (getattr(plan, "worker_metadata", {}) or {}).get(
            "reciprocal_interference",
            {},
        )
        if not isinstance(reciprocal, Mapping):
            return
        try:
            elapsed_ns = max(0, int(reciprocal.get("scoring_elapsed_ns", 0)))
            candidate_count = max(
                0,
                int(reciprocal.get("scoring_candidate_count", 0)),
            )
            affected = max(0, int(reciprocal.get("affected_incumbents", 0)))
            scanned_entries = max(0, int(reciprocal.get("scanned_entries", 0)))
            temporal_segments = max(0, int(reciprocal.get("temporal_segments", 0)))
            interference_lookups = max(
                0,
                int(reciprocal.get("interference_lookups", 0)),
            )
        except (TypeError, ValueError):
            return
        self._control_plane_overhead.record_elapsed_ns(
            "reciprocal_scoring",
            elapsed_ns,
            attrs={
                "component": getattr(plan, "component", ""),
                "candidate_count": candidate_count,
                "affected_incumbents": affected,
                "scanned_entries": scanned_entries,
                "temporal_segments": temporal_segments,
                "interference_lookups": interference_lookups,
            },
        )
        self._cp_count("reciprocal_scoring_candidates", candidate_count)
        self._cp_count("reciprocal_affected_incumbents", affected)
        self._cp_count("reciprocal_scanned_entries", scanned_entries)
        self._cp_count("reciprocal_temporal_segments", temporal_segments)
        self._cp_count("reciprocal_interference_lookups", interference_lookups)

    async def get_workers(self, request: web.Request) -> web.Response:
        """GET /api/v1/workers - Debug worker lifecycle snapshot.

        Enriches each worker with signal-derived fields:
        - co_located_components: currently co-executing components on the worker
        - active_task_count: number of tasks tracked by latency tracker
        - workload_class: inferred workload class for the worker's component
        """
        _ = request
        if self._worker_inventory_provider is None:
            return web.json_response({"workers": []})
        try:
            workers = list(self._worker_inventory_provider())
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        for w in workers:
            addr = str(w.get("addr") or "")
            component = str(w.get("component") or "")
            if addr:
                co_located = self._signal_service.get_worker_co_located_components(addr)
                tracker = self._signal_service._latency_trackers.get(addr)
                w["co_located_components"] = sorted(co_located)
                w["signal_active_task_count"] = tracker.active_count if tracker else 0
            else:
                w["co_located_components"] = []
                w["signal_active_task_count"] = 0
            if component:
                w["workload_class"] = self._signal_service.workload_classifier.classify(
                    component
                )
            else:
                w["workload_class"] = "unknown"

        return web.json_response({"workers": workers})

    async def _run_index_call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """ fix — route every ``self._run_index.<method>``
        invocation through the dedicated single-worker SQLite executor.

        Replaces ``self._run_index_call(self._run_index.<method>, ...)``
        which used the default 24-worker pool and triggered fcntl
        contention on concurrent SQLite access ( deadlock root
        cause).  The dedicated pool serialises all SQLite syscalls —
        under burst the queue grows but no two threads ever hit
        ``sqlite3.connect`` simultaneously, so the deadlock is
        structurally impossible.

         cache (campaigns/runs/signals 2 s TTL) keeps query rate
        low enough for a single thread to absorb steady-state load
        (~2-3 queries/s including write paths).
        """
        executor = getattr(self, "_sqlite_executor", None)
        if executor is None:
            return await asyncio.to_thread(fn, *args, **kwargs)

        def _invoke() -> Any:
            return fn(*args, **kwargs)

        future = executor.submit(_invoke)
        try:
            while not future.done():
                await asyncio.sleep(0.01)
            return future.result()
        except CancelledError as _cancelled:
            future.cancel()
            raise

    async def get_signals(self, request: web.Request) -> web.Response:
        """GET /api/v1/signals - SignalService observability snapshot.

         fix — split into **light** (default) + **heavy**
        (opt-in via ``?include=heavy``) sections.   observed that the
        full ``signal_service.as_dict()`` build (interference matrix +
        latency_trackers history + resource_profiles GP posterior dump
        + init_profiles) was the dominant Cholesky/serialize cost when
        polled regularly, blocking the asyncio loop in bursts and
        leaving GPU 0/3 idle while drift cascade ran on GPU 1/2.

        Light path (default — what ops console + bench monitoring need):
            campaign_scheduler.campaigns, scheduling_scenario,
            pipeline_dag, snapshot_at.

        Heavy path (?include=heavy — for debugging / paper analysis):
            adds activation_peaks, latency_trackers, interference,
            workload_profiles, resource_profiles, init_profiles.

        Both paths are wrapped in ``asyncio.to_thread`` to keep build +
        serialize off the main loop, and 2 s response cache absorbs
        repeated polls.
        """
        query = request.rel_url.query
        try:
            include_heavy = self._parse_bool_query(
                query.get("include_heavy"),
                default=False,
            ) or (str(query.get("include", "")).lower() == "heavy")
            include_retired_raw = str(query.get("include_retired", "")).strip().lower()
            if include_retired_raw in {"all", "full", "unbounded"}:
                include_retired = True
                retired_limit: int | None = None
            else:
                include_retired = self._parse_bool_query(
                    query.get("include_retired"),
                    default=True,
                )
                retired_limit_raw = (
                    str(query.get("retired_limit", "128")).strip().lower()
                )
                if retired_limit_raw in {"all", "full", "unbounded"}:
                    retired_limit = None
                else:
                    retired_limit = max(0, min(10000, int(retired_limit_raw or "128")))
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

        cache_key = (
            "heavy" if include_heavy else "light",
            bool(include_retired),
            retired_limit if retired_limit is not None else "all",
        )
        cached = self._signals_cache_get(cache_key)
        if cached is not None:
            return web.json_response(cached)

        try:
            payload = await asyncio.to_thread(
                self._build_signals_payload,
                include_heavy,
                include_retired,
                retired_limit,
            )
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
        self._signals_cache_put(cache_key, payload)
        return web.json_response(payload)

    def _build_signals_payload(
        self,
        include_heavy: bool,
        include_retired: bool = True,
        retired_limit: int | None = 128,
    ) -> dict[str, Any]:
        """Sync builder — runs in ThreadPoolExecutor via asyncio.to_thread.

        Light vs heavy split keeps the polling-friendly default
        sub-millisecond on the main loop (only the cache lookup is
        async; the build itself is offloaded).
        """
        import time as _time_mod

        payload: dict[str, Any] = {"snapshot_at": _time_mod.time()}
        try:
            payload["campaign_scheduler"] = self._planner.campaign_scheduler.as_dict(
                include_retired=include_retired,
                retired_limit=retired_limit,
            )
        except Exception:
            payload["campaign_scheduler"] = None
        try:
            payload["scheduling_scenario"] = self._planner.placement_ctx.summarize()
        except Exception:
            payload["scheduling_scenario"] = None
        try:
            pdag = getattr(self._planner, "pipeline_dag", None)
            payload["pipeline_dag"] = pdag.to_dict() if pdag is not None else {}
        except Exception:
            payload["pipeline_dag"] = {}

        if include_heavy:
            try:
                heavy = self._signal_service.as_dict()
                for k, v in heavy.items():
                    if k not in payload:
                        payload[k] = v
            except Exception:
                _LOG.debug("[signals] heavy payload unavailable", exc_info=True)
            try:
                supervisor = getattr(self, "supervisor", None)
                init = getattr(supervisor, "init_tracker", None) if supervisor else None
                payload["init_profiles"] = init.as_dict() if init else {}
            except Exception:
                payload["init_profiles"] = {}
        return payload


    _SIGNALS_CACHE_TTL_SEC: float = 2.0

    def _signals_cache_get(self, key: tuple) -> dict[str, Any] | None:
        cache = getattr(self, "_signals_cache", None)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if (time.time() - ts) > self._SIGNALS_CACHE_TTL_SEC:
            cache.pop(key, None)
            return None
        return payload

    def _signals_cache_put(self, key: tuple, payload: dict[str, Any]) -> None:
        cache = getattr(self, "_signals_cache", None)
        if cache is None:
            cache = {}
            self._signals_cache = cache
        cache[key] = (time.time(), payload)

    async def get_profile_export(self, request: web.Request) -> web.Response:
        """GET /api/v1/profile/export — baseline-only profile export.

        Aggregates ResourceProfileRegistry GP posterior data into the
        input-length-agnostic peak/mean statistics used by
        ``configs/workers.profile.yaml``.  Read-only — proton scheduling
        is not affected.

        Output schema (matches tools/export_workers_profile.py):
            {
              "snapshot_at": <unix_ts>,
              "components": {
                "<component>": {
                  "n_observations": int,
                  "weight_max_mb": int,        # μ + 1.96σ peak across input_size bucket
                  "activation_max_mb": int,
                  "runtime_max_sec": float,    # μ + 1.96σ peak latency (gateway GP)
                  "interference_solo_vram_mb": Optional[int],
                  "input_size_buckets_observed": [float, ...],
                }
              }
            }

        For each component:
        - vram_max    = max over (config, gpu, observed input_size) of
                        (vram_gp.predict(x_star) μ + 1.96·√σ²)
        - runtime_max = max over (config, gpu, observed input_size) of
                        (latency_gp.predict(x_star) μ + 1.96·√σ²)

        weight/activation split: when interference registry has a
        ``solo_vram`` baseline for the component, use it as the weight
        floor and (vram_max − solo_vram) as the activation peak.
        Otherwise fall back to a 60/40 split (heuristic; baselines all
        consume `total = weight + activation` so the split is neutral
        for scheduling correctness).

        Note: The exported ``runtime_max_sec`` here is the gateway GP
        posterior peak (μ + 1.96σ).  ``configs/workers.profile.yaml``
        uses a *different* convention for the slurm baseline:
        ``2 × observed_max(realtime)`` from NF trace.tsv (backfill
        literature F=2 convention; Tsafrir TPDS 2007).  These two
        values are NOT the same and the YAML is curated by hand from
        offline analysis of NF traces; ``init_max_sec`` and
        ``runtime_mean_sec`` were retired .
        """
        _ = request
        try:
            import math as _math
            import time as _time_mod

            registry = self._signal_service.resource_profiles
            interference = getattr(self._signal_service, "_interference_registry", None)

            payload_components: dict[str, Any] = {}
            for component, profile in registry._profiles.items():
                vram_peaks: list[float] = []
                ram_peaks: list[float] = []
                latency_peaks: list[float] = []
                input_buckets: set[float] = set()
                n_obs = 0

                for _fp, config_profile in profile._config_baselines.items():
                    for _gpu_id, baseline in config_profile._gpu_baselines.items():
                        vram_gp = baseline.vram_gp
                        ram_gp = baseline.ram_gp
                        latency_gp = baseline.latency_gp
                        for x in vram_gp._xs:
                            input_buckets.add(round(float(x), 1))
                        for x in ram_gp._xs:
                            input_buckets.add(round(float(x), 1))
                        for x in latency_gp._xs:
                            input_buckets.add(round(float(x), 1))
                        n_obs += vram_gp.n + ram_gp.n + latency_gp.n
                        for x in sorted(input_buckets):
                            try:
                                v_mu, v_var = vram_gp.predict(x)
                                v_upper = v_mu + 1.96 * _math.sqrt(max(v_var, 0.0))
                                if v_upper > 0:
                                    vram_peaks.append(v_upper)
                            except Exception:
                                _LOG.debug(
                                    "[profile-export] VRAM prediction failed",
                                    exc_info=True,
                                )
                            try:
                                r_mu, r_var = ram_gp.predict(x)
                                r_upper = r_mu + 1.96 * _math.sqrt(max(r_var, 0.0))
                                if r_upper > 0:
                                    ram_peaks.append(r_upper)
                            except Exception:
                                _LOG.debug(
                                    "[profile-export] RAM prediction failed",
                                    exc_info=True,
                                )
                            try:
                                l_mu, l_var = latency_gp.predict(x)
                                l_upper = l_mu + 1.96 * _math.sqrt(max(l_var, 0.0))
                                if l_upper > 0:
                                    latency_peaks.append(l_upper)
                            except Exception:
                                _LOG.debug(
                                    "[profile-export] latency prediction failed",
                                    exc_info=True,
                                )

                if not vram_peaks and not latency_peaks:
                    continue

                vram_max = int(max(vram_peaks)) if vram_peaks else 0
                runtime_max = float(max(latency_peaks)) if latency_peaks else 0.0

                solo_vram_mb: int | None = None
                if interference is not None:
                    try:
                        solo = None
                        for fp_iter in profile._config_baselines:
                            try:
                                v = interference.get_solo_vram_baseline(
                                    component, fp=fp_iter
                                )
                            except Exception:
                                v = None
                            if v is not None and v > 0:
                                solo = v if solo is None else max(solo, v)
                        if solo is not None:
                            solo_vram_mb = int(solo)
                    except Exception:
                        solo_vram_mb = None

                weight_max_mb = 0
                supervisor = getattr(self, "supervisor", None)
                if supervisor is not None and hasattr(
                    supervisor, "get_component_weight"
                ):
                    try:
                        weight_max_mb = int(
                            supervisor.get_component_weight(component) or 0
                        )
                    except Exception:
                        weight_max_mb = 0
                if weight_max_mb > 0:
                    activation_max_mb = vram_max
                elif solo_vram_mb is not None and solo_vram_mb < vram_max:
                    weight_max_mb = solo_vram_mb
                    activation_max_mb = max(0, vram_max - solo_vram_mb)
                else:
                    weight_max_mb = int(vram_max * 0.6)
                    activation_max_mb = max(0, vram_max - weight_max_mb)

                payload_components[component] = {
                    "n_observations": n_obs,
                    "weight_max_mb": weight_max_mb,
                    "activation_max_mb": activation_max_mb,
                    "ram_max_mb": int(max(ram_peaks)) if ram_peaks else 0,
                    "total_vram_mb": vram_max,
                    "runtime_max_sec": runtime_max,
                    "interference_solo_vram_mb": solo_vram_mb,
                    "input_size_buckets_observed": sorted(input_buckets),
                }

            return web.json_response(
                {
                    "snapshot_at": _time_mod.time(),
                    "components": payload_components,
                }
            )
        except Exception as e:
            return web.json_response(
                {"error": f"profile export failed: {type(e).__name__}: {e}"},
                status=500,
            )

    def build_runtime_state_snapshot(self) -> dict[str, Any]:
        supervisor = getattr(self, "supervisor", None)
        init_profile = getattr(supervisor, "init_tracker", None) if supervisor else None
        planner = getattr(self, "_planner", None)
        export_planner_state = getattr(planner, "export_runtime_state", None)
        planner_state = export_planner_state() if callable(export_planner_state) else {}
        return {
            "schema": "runtime_state_v1",
            "exported_at_wall": time.time(),
            "gateway_identity": dict(getattr(self, "_gateway_identity", {}) or {}),
            "signal_service": self._signal_service.export_runtime_state(),
            "init_profile": (
                init_profile.export_state()
                if init_profile is not None and hasattr(init_profile, "export_state")
                else {}
            ),
            "supervisor": (
                supervisor.export_runtime_state()
                if supervisor is not None
                and hasattr(supervisor, "export_runtime_state")
                else {}
            ),
            "planner": planner_state,
        }

    async def save_runtime_state_snapshot(self) -> dict[str, Any]:
        payload = self.build_runtime_state_snapshot()
        await self._run_index_call(
            self._run_index.save_signal_snapshot,
            self._signal_service._RUNTIME_SNAPSHOT_KEY,
            payload,
        )
        _LOG.info("[runtime-state] Saved runtime_state_v1 snapshot")
        return payload

    async def admin_snapshot_runtime_state(self, request: web.Request) -> web.Response:
        """POST /api/v1/admin/runtime_state/snapshot — persist exact runtime state."""
        _ = request
        try:
            payload = await self.save_runtime_state_snapshot()
        except Exception as exc:
            _LOG.warning(
                "[runtime-state] admin snapshot failed: %s", exc, exc_info=True
            )
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {
                "ok": True,
                "key": self._signal_service._RUNTIME_SNAPSHOT_KEY,
                "schema": payload.get("schema"),
                "exported_at_wall": payload.get("exported_at_wall"),
            }
        )

    def build_control_plane_overhead_snapshot(self) -> dict[str, Any]:
        return self._control_plane_overhead.export_state()

    async def save_control_plane_overhead_snapshot(self) -> dict[str, Any] | None:
        if not getattr(self, "_control_plane_overhead_enabled", False):
            return None
        payload = self.build_control_plane_overhead_snapshot()
        await self._run_index_call(
            self._run_index.save_signal_snapshot,
            CONTROL_PLANE_OVERHEAD_SNAPSHOT_KEY,
            payload,
        )
        _LOG.info("[control-plane-overhead] Saved control_plane_overhead_v1 snapshot")
        return payload

    async def admin_snapshot_control_plane_overhead(
        self, request: web.Request
    ) -> web.Response:
        """POST /api/v1/admin/control_plane_overhead/snapshot."""
        _ = request
        if not getattr(self, "_control_plane_overhead_enabled", False):
            return web.json_response(
                {
                    "ok": True,
                    "enabled": False,
                    "saved": False,
                    "key": CONTROL_PLANE_OVERHEAD_SNAPSHOT_KEY,
                }
            )
        try:
            payload = await self.save_control_plane_overhead_snapshot()
        except Exception as exc:
            _LOG.warning(
                "[control-plane-overhead] admin snapshot failed: %s",
                exc,
                exc_info=True,
            )
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response(
            {
                "ok": True,
                "enabled": True,
                "saved": payload is not None,
                "key": CONTROL_PLANE_OVERHEAD_SNAPSHOT_KEY,
                "schema": (payload or {}).get("schema"),
                "exported_at_wall": (payload or {}).get("exported_at_wall"),
            }
        )

    async def get_debug_tasks(self, request: web.Request) -> web.Response:
        """GET /api/v1/debug/tasks - Inspect in-memory task records."""
        _ = request
        async with self._lock:
            tasks = []
            for _tid, rec in self._tasks.items():
                tasks.append(
                    {
                        "task_id": rec.task_id,
                        "nf_task_id": rec.nf_task_id,
                        "component": rec.component,
                        "campaign_id": rec.campaign_id,
                        "run_name": rec.run_name,
                        "submitter": rec.submitter,
                        "state": STATE_NAMES[rec.state],
                        "created_at": rec.created_at,
                        "updated_at": rec.updated_at,
                        "dispatch_worker_addr": rec.dispatch_worker_addr,
                        "dispatch_worker_name": rec.dispatch_worker_name,
                        "message": rec.message,
                    }
                )
        return web.json_response(
            {
                "tasks": tasks,
                "campaign_identity_metrics": dict(self._campaign_identity_metrics),
            }
        )

    async def _snapshot_active_tasks_for_campaigns(self) -> list[dict[str, Any]]:
        async with self._lock:
            items: list[dict[str, Any]] = []
            for rec in self._tasks.values():
                items.append(
                    {
                        "task_id": rec.task_id,
                        "run_id": rec.nf_task_id,
                        "campaign_id": rec.campaign_id,
                        "run_name": rec.run_name,
                        "submitter": rec.submitter,
                        "campaign_metadata_version": rec.campaign_metadata_version,
                        "component": rec.component,
                        "state": STATE_NAMES.get(rec.state, "UNSPECIFIED"),
                        "created_at": rec.created_at,
                        "updated_at": rec.updated_at,
                        "is_dispatched": bool(
                            str(getattr(rec, "_dispatched_gpu_id", "") or "").strip()
                        ),
                        "gateway": dict(self._gateway_identity),
                    }
                )
        return items

    async def get_campaigns(self, request: web.Request) -> web.Response:
        query = request.rel_url.query
        try:
            limit = _normalize_optional_positive_int(query.get("limit"), "limit") or 200
            if limit > 1000:
                raise ValueError("limit must be <= 1000")
            offset_raw = query.get("offset", "0")
            offset = int(offset_raw)
            if offset < 0:
                raise ValueError("offset must be an integer >= 0")
            include_unassigned = self._parse_bool_query(
                query.get("include_unassigned"), default=False
            )
            gateway_filters = self._parse_gateway_filters(query)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        try:
            active_rows = await self._snapshot_active_tasks_for_campaigns()
            cache_key = (
                int(limit),
                int(offset),
                bool(include_unassigned),
                tuple(
                    sorted(
                        (k, tuple(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in gateway_filters.items()
                    )
                ),
            )
            payload = self._campaigns_cache_get(cache_key)
            if payload is None:
                payload = await self._run_index_call(
                    build_campaign_queue,
                    run_index=self._run_index,
                    active_tasks=active_rows,
                    limit=limit,
                    offset=offset,
                    include_unassigned=include_unassigned,
                    **gateway_filters,
                )
                self._campaigns_cache_put(cache_key, payload)
        except Exception as exc:
            _LOG.exception("campaign queue query failed: %s", exc)
            return web.json_response(
                {"error": "campaign queue query failed"}, status=500
            )
        return web.json_response(payload)

    async def get_campaign_detail(self, request: web.Request) -> web.Response:
        campaign_id = str(request.match_info.get("campaign_id") or "").strip()
        if not campaign_id:
            return web.json_response({"error": "campaign_id is required"}, status=400)
        try:
            gateway_filters = self._parse_gateway_filters(request.rel_url.query)
            active_rows = await self._snapshot_active_tasks_for_campaigns()
            payload = await self._run_index_call(
                build_campaign_detail,
                run_index=self._run_index,
                active_tasks=active_rows,
                campaign_id=campaign_id,
                **gateway_filters,
            )
        except Exception as exc:
            _LOG.exception("campaign detail query failed: %s", exc)
            return web.json_response(
                {"error": "campaign detail query failed"}, status=500
            )
        if payload is None:
            return web.json_response(
                {"error": "campaign not found", "campaign_id": campaign_id}, status=404
            )
        return web.json_response(payload)


    _CAMPAIGNS_CACHE_TTL_SEC: float = 2.0

    def _campaigns_cache_get(self, key: tuple) -> dict[str, Any] | None:
        """Return cached campaigns payload if fresh, else None.

        Cache is per-server instance (``self._campaigns_cache``).  Stale
        entries (age > TTL) are evicted on read.  No size cap needed —
        keyspace bounded by (limit, offset, include_unassigned, filter
        tuple) and ops console issues only ~3 distinct keys in steady
        state.
        """
        cache = getattr(self, "_campaigns_cache", None)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if (time.time() - ts) > self._CAMPAIGNS_CACHE_TTL_SEC:
            cache.pop(key, None)
            return None
        return payload

    def _campaigns_cache_put(self, key: tuple, payload: dict[str, Any]) -> None:
        """Store campaigns payload with current timestamp."""
        cache = getattr(self, "_campaigns_cache", None)
        if cache is None:
            cache = {}
            self._campaigns_cache = cache
        cache[key] = (time.time(), payload)

    @staticmethod
    def _parse_sort(sort_raw: Any) -> tuple[str, bool]:
        raw = str(sort_raw or "finished_at").strip()
        if not raw:
            return "finished_at", True
        descending = True
        field = raw
        if raw.startswith("-"):
            descending = True
            field = raw[1:]
        elif ":" in raw:
            lhs, rhs = raw.split(":", 1)
            field = lhs.strip()
            direction = rhs.strip().lower()
            if direction in {"asc", "ascending"}:
                descending = False
            elif direction in {"desc", "descending", ""}:
                descending = True
            else:
                raise ValueError("sort direction must be asc or desc")
        return field, descending

    def _build_run_query(
        self,
        source: Mapping[str, Any],
        *,
        axis_keys: list[str] | None = None,
        axis_values: list[str] | None = None,
    ) -> RunQuery:
        limit = _normalize_optional_positive_int(source.get("limit"), "limit") or 200
        if limit > 200:
            limit = 200
        offset_raw = source.get("offset", 0)
        try:
            offset = int(offset_raw)
        except Exception as exc:
            raise ValueError("offset must be an integer >= 0") from exc
        if offset < 0:
            raise ValueError("offset must be an integer >= 0")

        sort_field, descending = self._parse_sort(source.get("sort"))
        from_ts = _normalize_optional_float(source.get("from_ts"), "from_ts")
        to_ts = _normalize_optional_float(source.get("to_ts"), "to_ts")

        key_list = [
            str(item).strip() for item in list(axis_keys or []) if str(item).strip()
        ]
        val_list = [
            str(item).strip() for item in list(axis_values or []) if str(item).strip()
        ]
        if (key_list or val_list) and len(key_list) != len(val_list):
            raise ValueError("axis_key and axis_value must have the same item count")
        gateway_filters = self._parse_gateway_filters(source)

        return RunQuery(
            run_source=str(source.get("run_source") or "").strip().lower() or None,
            component=str(source.get("component") or "").strip().lower() or None,
            level=str(source.get("level") or "").strip().lower() or None,
            campaign_id=str(source.get("campaign_id") or "").strip() or None,
            cell_schema_id=str(source.get("cell_schema_id") or "").strip() or None,
            state=str(source.get("state") or "").strip().upper() or None,
            sample_id=str(source.get("sample_id") or "").strip() or None,
            input_batch_size=_normalize_optional_positive_int(
                source.get("input_batch_size"), "input_batch_size"
            ),
            output_sample_count=_normalize_optional_positive_int(
                source.get("output_sample_count"), "output_sample_count"
            ),
            from_ts=from_ts,
            to_ts=to_ts,
            worker_name=str(source.get("worker_name") or "").strip() or None,
            gpu_id=str(source.get("gpu_id") or "").strip() or None,
            include_gateway_instance_ids=gateway_filters.get(
                "include_gateway_instance_ids", []
            ),
            exclude_gateway_instance_ids=gateway_filters.get(
                "exclude_gateway_instance_ids", []
            ),
            include_gateway_git_commits=gateway_filters.get(
                "include_gateway_git_commits", []
            ),
            exclude_gateway_git_commits=gateway_filters.get(
                "exclude_gateway_git_commits", []
            ),
            axis_filters=[
                AxisFilter(key=key, value=value)
                for key, value in zip(key_list, val_list, strict=False)
            ],
            limit=limit,
            offset=offset,
            sort=sort_field,
            descending=descending,
        )

    async def get_profile_runs(self, request: web.Request) -> web.Response:
        query = request.rel_url.query
        axis_keys = [str(item).strip() for item in query.getall("axis_key", [])]
        axis_values = [str(item).strip() for item in query.getall("axis_value", [])]
        try:
            run_query = self._build_run_query(
                query, axis_keys=axis_keys, axis_values=axis_values
            )
            cache_key = self._run_query_cache_key(run_query)
            cached = self._runs_cache_get(cache_key)
            if cached is not None:
                return web.Response(body=cached, content_type="application/json")

            rows, total = await self._run_index_call(
                self._run_index.query_runs, run_query
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

        body = await asyncio.to_thread(
            self._serialize_profile_runs_response,
            rows,
            total,
            run_query.limit,
            run_query.offset,
        )
        self._runs_cache_put(cache_key, body)
        return web.Response(body=body, content_type="application/json")

    @staticmethod
    def _serialize_profile_runs_response(
        rows: Sequence[Any],
        total: int,
        limit: int,
        offset: int,
    ) -> bytes:
        payload = {
            "runs": [row.as_dict() for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        return json.dumps(payload).encode("utf-8")


    _RUNS_CACHE_TTL_SEC: float = 2.0

    @staticmethod
    def _run_query_cache_key(rq: Any) -> tuple:
        """Stable hashable key from a ``RunQuery``.  Fields that
        change per call (e.g., ``axis_filters`` list) are normalised
        into tuples; lists into sorted tuples.  Mirrors the way
        ``RunIndexStore.query_runs`` consumes the query.
        """

        def _list(x: Any) -> tuple:
            return (
                tuple(sorted(x)) if isinstance(x, (list, tuple)) else (x,) if x else ()
            )

        return (
            getattr(rq, "run_source", None),
            getattr(rq, "component", None),
            getattr(rq, "level", None),
            getattr(rq, "campaign_id", None),
            getattr(rq, "cell_schema_id", None),
            getattr(rq, "state", None),
            getattr(rq, "sample_id", None),
            getattr(rq, "input_batch_size", None),
            getattr(rq, "output_sample_count", None),
            getattr(rq, "from_ts", None),
            getattr(rq, "to_ts", None),
            getattr(rq, "worker_name", None),
            getattr(rq, "gpu_id", None),
            _list(getattr(rq, "include_gateway_instance_ids", []) or []),
            _list(getattr(rq, "exclude_gateway_instance_ids", []) or []),
            _list(getattr(rq, "include_gateway_git_commits", []) or []),
            _list(getattr(rq, "exclude_gateway_git_commits", []) or []),
            tuple(
                (getattr(af, "key", None), getattr(af, "value", None))
                for af in (getattr(rq, "axis_filters", []) or [])
            ),
            getattr(rq, "limit", None),
            getattr(rq, "offset", None),
        )

    def _runs_cache_get(self, key: tuple) -> bytes | None:
        cache = getattr(self, "_runs_cache", None)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if (time.time() - ts) > self._RUNS_CACHE_TTL_SEC:
            cache.pop(key, None)
            return None
        return payload

    def _runs_cache_put(self, key: tuple, payload: bytes) -> None:
        cache = getattr(self, "_runs_cache", None)
        if cache is None:
            cache = {}
            self._runs_cache = cache
        cache[key] = (time.time(), payload)
        if len(cache) > 64:
            oldest = min(cache.items(), key=lambda kv: kv[1][0])
            cache.pop(oldest[0], None)

    async def get_profile_run(self, request: web.Request) -> web.Response:
        run_key = str(request.match_info.get("run_key") or "").strip()
        if not run_key:
            return web.json_response({"error": "run_key is required"}, status=400)
        run = await self._run_index_call(self._run_index.get_run, run_key)
        if run is None:
            return web.json_response(
                {"error": "run not found", "run_key": run_key}, status=404
            )
        return web.json_response(run.as_dict())

    @staticmethod
    def _parse_bool_query(value: Any, *, default: bool) -> bool:
        if value is None:
            return bool(default)
        raw = str(value).strip().lower()
        if not raw:
            return bool(default)
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError("boolean query value must be one of [true,false,1,0,yes,no]")

    @staticmethod
    def _parse_repeated_query_values(query: Mapping[str, Any], key: str) -> list[str]:
        values = (
            cast(Any, query).getall(key, [])
            if hasattr(query, "getall")
            else [query.get(key)]
        )
        out: list[str] = []
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            for token in text.split(","):
                item = str(token or "").strip()
                if item:
                    out.append(item)
        return list(dict.fromkeys(out))

    def _parse_gateway_filters(self, query: Mapping[str, Any]) -> dict[str, list[str]]:
        return {
            "include_gateway_instance_ids": [
                normalize_gateway_instance_id(item)
                for item in self._parse_repeated_query_values(
                    query, "include_gateway_instance_id"
                )
            ],
            "exclude_gateway_instance_ids": [
                normalize_gateway_instance_id(item)
                for item in self._parse_repeated_query_values(
                    query, "exclude_gateway_instance_id"
                )
            ],
            "include_gateway_git_commits": [
                normalize_gateway_git_commit(item)
                for item in self._parse_repeated_query_values(
                    query, "include_gateway_git_commit"
                )
            ],
            "exclude_gateway_git_commits": [
                normalize_gateway_git_commit(item)
                for item in self._parse_repeated_query_values(
                    query, "exclude_gateway_git_commit"
                )
            ],
        }

    @staticmethod
    def _parse_component_tokens(query: Mapping[str, Any]) -> list[str]:
        out: list[str] = []
        for key in ("component", "components"):
            values = (
                cast(Any, query).getall(key, [])
                if hasattr(query, "getall")
                else [query.get(key)]
            )
            for raw in values:
                text = str(raw or "").strip()
                if not text:
                    continue
                for token in text.split(","):
                    item = str(token or "").strip().lower()
                    if item:
                        out.append(item)
        return list(dict.fromkeys(out))

    async def get_telemetry_health(self, request: web.Request) -> web.Response:
        query = request.rel_url.query
        components = self._parse_component_tokens(query)
        campaign_id = str(query.get("campaign_id") or "").strip() or None

        try:
            window_hours_raw = query.get("window_hours")
            if window_hours_raw in (None, ""):
                window_hours = 24
            else:
                window_hours = int(window_hours_raw)
            if window_hours < 1 or window_hours > (24 * 30):
                raise ValueError("window_hours must be between 1 and 720")
            include_all_time = self._parse_bool_query(
                query.get("include_all_time"), default=True
            )
            payload = await self._run_index_call(
                build_task_telemetry_health_report,
                run_index=self._run_index,
                components=components,
                campaign_id=campaign_id,
                window_hours=window_hours,
                include_all_time=include_all_time,
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            _LOG.exception("telemetry health query failed: %s", exc)
            return web.json_response(
                {"error": "telemetry health query failed"}, status=500
            )

        return web.json_response(payload)

    async def get_model_versions(self, request: web.Request) -> web.Response:
        query = request.rel_url.query
        try:
            limit = _normalize_optional_positive_int(query.get("limit"), "limit") or 50
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if limit > 500:
            return web.json_response({"error": "limit must be <= 500"}, status=400)

        rows = await self._run_index_call(self._run_index.list_model_versions)
        versions: list[dict[str, Any]] = []
        for row in rows:
            versions.append(dict(row))
        return web.json_response({"count": len(versions), "versions": versions[:limit]})

    @staticmethod
    def _component_model_selection_rule() -> dict[str, Any]:
        return {
            "name": PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_NAME,
            "description": PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_DESCRIPTION,
        }

    async def get_model_components(self, request: web.Request) -> web.Response:
        query = request.rel_url.query
        try:
            limit = _normalize_optional_positive_int(query.get("limit"), "limit") or 200
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if limit > 2000:
            return web.json_response({"error": "limit must be <= 2000"}, status=400)
        component_filter = str(query.get("component") or "").strip().lower()
        status_filter = str(query.get("status") or "").strip().upper()
        latest_list, active_list = await asyncio.gather(
            self._run_index_call(self._run_index.list_latest_component_models),
            self._run_index_call(self._run_index.list_active_component_models),
        )
        latest_rows = {
            str(row.get("component") or "").strip().lower(): row
            for row in latest_list
            if str(row.get("component") or "").strip()
        }
        active_rows = {
            str(row.get("component") or "").strip().lower(): row
            for row in active_list
            if str(row.get("component") or "").strip()
        }
        components = sorted(set(latest_rows.keys()) | set(active_rows.keys()))
        views: list[dict[str, Any]] = []
        for component in components:
            latest_view = (
                build_component_model_payload(latest_rows[component])
                if component in latest_rows
                else None
            )
            active_view = (
                build_component_model_payload(active_rows[component])
                if component in active_rows
                else None
            )
            promotion_gate = evaluate_promotion_gate(latest_view)
            effective_view = choose_effective_model_view(
                latest_model=latest_view,
                active_model=active_view,
                promotion_gate=promotion_gate,
            )

            view = {
                "component": component,
                "latest_model": latest_view,
                "active_model": active_view,
                "effective_model": effective_view,
                "promotion_gate": promotion_gate,
                "latest_status": str((latest_view or {}).get("status") or ""),
                "active_status": str((active_view or {}).get("status") or ""),
                "effective_status": str((effective_view or {}).get("status") or ""),
            }
            if (
                component_filter
                and str(view.get("component") or "") != component_filter
            ):
                continue
            filter_status = str(view.get("effective_status") or "")
            if status_filter and filter_status != status_filter:
                continue
            views.append(view)
        views.sort(
            key=lambda item: (
                1 if str(item.get("effective_status") or "") == "BLOCKED" else 0,
                _to_int((item.get("latest_model") or {}).get("priority_rank")) or 10**9,
                str(item.get("component") or ""),
            )
        )
        return web.json_response(
            {
                "count": min(len(views), int(limit)),
                "total": len(views),
                "components": views[:limit],
            }
        )

    async def get_model_component(self, request: web.Request) -> web.Response:
        component = str(request.match_info.get("component") or "").strip().lower()
        if not component:
            return web.json_response({"error": "component is required"}, status=400)
        versions, active_row = await asyncio.gather(
            self._run_index_call(
                self._run_index.list_component_model_versions,
                component=component,
                limit=1,
            ),
            self._run_index_call(
                self._run_index.get_active_component_model, component=component
            ),
        )
        if not versions and active_row is None:
            return web.json_response(
                {"error": f"component model not found: {component}"}, status=404
            )
        latest = versions[0] if versions else None
        latest_view = (
            build_component_model_payload(latest) if latest is not None else None
        )
        active_view = (
            build_component_model_payload(active_row)
            if isinstance(active_row, Mapping)
            else None
        )
        promotion_gate = evaluate_promotion_gate(latest_view)
        effective_view = choose_effective_model_view(
            latest_model=latest_view,
            active_model=active_view,
            promotion_gate=promotion_gate,
        )
        return web.json_response(
            {
                "component": component,
                "latest": latest_view,
                "active": active_view,
                "effective": effective_view,
                "promotion_gate": promotion_gate,
                "selection_rule": self._component_model_selection_rule(),
            }
        )

    async def get_model_component_versions(self, request: web.Request) -> web.Response:
        component = str(request.match_info.get("component") or "").strip().lower()
        if not component:
            return web.json_response({"error": "component is required"}, status=400)
        query = request.rel_url.query
        try:
            limit = _normalize_optional_positive_int(query.get("limit"), "limit") or 200
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if limit > 5000:
            return web.json_response({"error": "limit must be <= 5000"}, status=400)
        rows = await self._run_index_call(
            self._run_index.list_component_model_versions,
            component=component,
            limit=limit,
        )
        views = [build_component_model_payload(row) for row in rows]
        return web.json_response(
            {
                "component": component,
                "count": len(views),
                "versions": views,
            }
        )

    async def submit_job(self, request: web.Request) -> web.Response:
        """POST /api/v1/job/submit"""
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"invalid json: {exc}"}, status=400)
        if not isinstance(body, Mapping):
            return web.json_response(
                {"error": "request body must be an object"}, status=400
            )

        try:
            kind = JobService.parse_kind(body.get("kind"))
            payload = body.get("payload") or {}
            if not isinstance(payload, Mapping):
                raise ValueError("payload must be an object")
            _cp_start = self._cp_begin()
            record = await self._jobs.submit(kind=kind, payload=payload)
            self._cp_record(
                "submit_enqueue",
                _cp_start,
                active_wall=True,
                kind=getattr(kind, "value", str(kind)),
                component=str(payload.get("component") or ""),
            )
        except _APIError as exc:
            return web.json_response({"error": str(exc)}, status=exc.status)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        return web.json_response(
            {
                "job_id": record.job_id,
                "kind": record.kind.value,
                "state": record.state.value,
            }
        )

    async def get_job(self, request: web.Request) -> web.Response:
        """GET /api/v1/job/{job_id}"""
        job_id = str(request.match_info.get("job_id", "")).strip()
        if not job_id:
            return web.json_response({"error": "job_id is required"}, status=400)
        record = await self._jobs.get(job_id)
        if record is None:
            return web.json_response(
                {"error": "job not found", "job_id": job_id}, status=404
            )
        return web.json_response(record.as_response())

    async def cancel_job(self, request: web.Request) -> web.Response:
        """POST /api/v1/job/cancel/{job_id}"""
        job_id = str(request.match_info.get("job_id", "")).strip()
        if not job_id:
            return web.json_response({"error": "job_id is required"}, status=400)
        cancelled = await self._jobs.cancel(job_id)
        if not cancelled:
            return web.json_response(
                {"ok": False, "message": "job not found"}, status=404
            )
        return web.json_response({"ok": True, "message": "cancelled"})

    async def mark_campaign_complete(self, request: web.Request) -> web.Response:
        """POST /api/v1/campaign/complete — Nextflow workflow terminal signal.

        Task-level submissions alone cannot prove workflow completion because
        Nextflow may be between stages with no gateway-visible work.  This
        endpoint lets Nextflow close the campaign explicitly.
        """
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": f"invalid json: {exc}"}, status=400
            )
        if not isinstance(body, Mapping):
            return web.json_response(
                {"ok": False, "error": "request body must be an object"}, status=400
            )

        campaign_id = str(body.get("campaign_id") or "").strip()
        run_name = str(body.get("run_name") or "").strip()
        status = str(body.get("status") or "").strip().lower()
        ok = bool(body.get("ok", False))
        if status in {"succeeded", "success", "completed", "complete", "ok"}:
            ok = True
        elif status in {"failed", "error", "cancelled", "canceled"}:
            ok = False
        message = str(body.get("message") or "").strip()
        if not campaign_id and not run_name:
            return web.json_response(
                {"ok": False, "error": "campaign_id or run_name is required"},
                status=400,
            )
        cs = getattr(getattr(self, "_planner", None), "campaign_scheduler", None)
        if cs is None or not hasattr(cs, "mark_external_completion"):
            return web.json_response(
                {"ok": False, "error": "campaign scheduler not initialized"},
                status=503,
            )
        matched = cs.mark_external_completion(
            campaign_id=campaign_id,
            run_name=run_name,
            ok=ok,
            message=message,
        )
        _LOG.info(
            "[campaign-complete] campaign_id=%s run_name=%s ok=%s matched=%s",
            campaign_id or "<none>",
            run_name or "<none>",
            ok,
            matched,
        )
        return web.json_response(
            {
                "ok": True,
                "campaign_id": campaign_id,
                "run_name": run_name,
                "workflow_ok": ok,
                "matched_campaigns": matched,
            }
        )

    async def _cleanup_superseded_record(
        self,
        prev_task_id: str,
        reason: str = "nf_retry_supersede",
    ) -> None:
        """Plan fix — single-record cleanup for NF retry
        submit-path dedup.

        Affects ONLY the task identified by ``prev_task_id``.  Other
        records / nf_task_ids / campaigns / timeline entries / VRAM
        reservations / pre-init schedules are preserved untouched.

        Invariants honoured:

        * **VRAM Lifecycle ()** — push/drain count parity is kept
          by draining the *one* prev record's ``_pending_vram_releases``
          via ``on_task_complete_event`` (the documented sole release
          path).  No double-drain — list is cleared after iteration.
        * **Timeline state ( Management Regression Checklist)**
          — only this prev_task_id's entries are removed (cross-GPU
          ``remove_all_entries`` operates on a single task_id).  Other
          tasks' active / predicted entries remain intact.
        * **Campaign live counters** — supersede is terminal for the
          previous record, so its pending/active contribution must be
          removed even when ``CampaignQueue`` is connected to gateway
          tasks.  ``pending_tasks`` / ``active_tasks`` are live counters,
          not scan-derived properties.
        * **Idempotent** — a second call after ``_tasks.pop(prev_task_id)``
          returns immediately (record lookup yields None).

        Order of operations chosen to minimize cross-cutting damage:

          1. VRAM drain (before state transition; on_task_complete may
             also do counter cleanup but does not drain VRAM).
          2. Timeline removal (free EFT projection of this task only).
          3. Pending pre-init removal (entries triggered BY this
             task_id; identified by ``key[2] == prev_task_id``).
          4. Counter / active_ids decrement for the superseded record.
          5. Terminal ``_set_state(CANCELLED)`` — fires wake trigger.
          6. ``dispatch_worker_addr`` cleared so scan-based
             ``active_tasks`` property excludes this record immediately.
          7. ``_tasks`` / ``_active_regular_tasks`` pop.

        Race-safety: each step is best-effort (try/except) so a partial
        failure leaves the rest of the cleanup intact.  ``_run_task``
        finally block remains idempotent w.r.t. an already-CANCELLED
        record (it observes the terminal state and skips its own
        on_task_complete bump).
        """
        async with self._lock:
            prev_record = self._tasks.get(prev_task_id)
        if prev_record is None:
            return

        prev_handle = getattr(self, "_handles", {}).get(prev_task_id)
        if prev_handle is not None:
            self._detach_precomputed_plan_from_handle(
                prev_handle,
                reason="nf_retry_supersede",
            )

        pending = getattr(prev_record, "_pending_vram_releases", None)
        if pending:
            for _gid, _comp, _vram in list(pending):
                try:
                    self._planner.on_task_complete_event(_gid, _comp, _vram)
                except Exception:
                    _LOG.warning(
                        "[] vram release failed for prev_task=%s gpu=%s",
                        prev_task_id[:8],
                        _gid,
                        exc_info=True,
                    )
            with contextlib.suppress(Exception):
                pending.clear()

        try:
            scenario = self._planner.campaign_scheduler._timelines
            scenario.remove_all_entries(prev_task_id)
        except Exception:
            _LOG.warning(
                "[] timeline cleanup failed for prev_task=%s",
                prev_task_id[:8],
                exc_info=True,
            )

        try:
            cs = self._planner.campaign_scheduler
            stale_keys = [
                k for k in cs._pending_pre_inits if len(k) >= 3 and k[2] == prev_task_id
            ]
            for k in stale_keys:
                cs._pending_pre_inits.pop(k, None)
                cs._fired_pre_inits.discard((k[0], k[1]))
        except Exception:
            _LOG.warning(
                "[] pre-init cleanup failed for prev_task=%s",
                prev_task_id[:8],
                exc_info=True,
            )

        try:
            cq = None
            cs = self._planner.campaign_scheduler
            if prev_record.campaign_id:
                cq = cs._campaign_queues.get(prev_record.campaign_id)
            if cq is not None:
                if prev_record.state == TASK_STATE_SUBMITTED:
                    if prev_task_id in getattr(cq, "_pending_ids", set()):
                        cq._pending_ids.discard(prev_task_id)
                        cq._pending_count = max(0, cq._pending_count - 1)
                    elif cq._pending_count > 0:
                        cq._pending_count = max(0, cq._pending_count - 1)
                elif prev_record.state == TASK_STATE_RUNNING:
                    if prev_task_id in getattr(cq, "_active_ids", set()):
                        cq._active_count = max(0, cq._active_count - 1)
                else:
                    if prev_task_id in getattr(cq, "_active_ids", set()):
                        cq._active_count = max(0, cq._active_count - 1)
                cq._pending_ids.discard(prev_task_id)
                cq._active_ids.discard(prev_task_id)
        except Exception:
            _LOG.warning(
                "[] counter cleanup failed for prev_task=%s",
                prev_task_id[:8],
                exc_info=True,
            )

        if prev_record.state in (TASK_STATE_SUBMITTED, TASK_STATE_RUNNING):
            try:
                await self._set_state(
                    prev_record,
                    TASK_STATE_CANCELLED,
                    ok=False,
                    exit_code=0,
                    message=f"superseded ({reason})",
                )
            except Exception:
                _LOG.warning(
                    "[] _set_state(CANCELLED) failed for prev_task=%s",
                    prev_task_id[:8],
                    exc_info=True,
                )

        with contextlib.suppress(Exception):
            prev_record.dispatch_worker_addr = ""

        async with self._lock:
            self._tasks.pop(prev_task_id, None)
            self._active_regular_tasks.discard(prev_task_id)

        _LOG.info(
            "[] superseded prev_task=%s reason=%s campaign=%s nf_task=%s",
            prev_task_id[:8],
            reason,
            (prev_record.campaign_id or "")[:8] if prev_record.campaign_id else "",
            prev_record.nf_task_id or "",
        )

    @staticmethod
    def _is_same_submit_attempt(record: Any, workdir_path: Path) -> bool:
        """Return True when a submit with the same NF identity is an HTTP
        replay of the exact same Nextflow workdir.

        ``job_client`` retries POST on transient TCP/read failures.  If the
        first POST reached the gateway but the response was lost, the second
        POST carries the same ``nf_task_id`` and the same workdir.  That is
        an idempotency replay and must return the existing task id, not
        supersede an already-running attempt.
        """
        try:
            prev = Path(str(getattr(record, "workdir", "") or "")).resolve()
        except Exception:
            return False
        try:
            current = Path(workdir_path).resolve()
        except Exception:
            current = workdir_path
        return prev == current

    async def _mark_record_pending_for_scheduler(
        self,
        record: TaskRecord,
        *,
        message: str = "waiting for scheduler wake",
    ) -> None:
        """Restore SUBMITTED liveness for a non-terminal scheduler retry.

        ``_run_task`` switches the HTTP TaskRecord to RUNNING before
        planning so the caller can observe that the gateway accepted the
        request.  If that attempt does not produce a terminal result
        (SKIP, cooperative eviction requeue, adapter retry), the record is
        pending again from Nextflow's perspective.  Leaving it as RUNNING
        with stale dispatch metadata makes live-counter reconcile count it
        as active even after the worker/timeline has been released.
        """
        task_id = str(getattr(record, "task_id", "") or "").strip()
        if not task_id:
            return
        if int(getattr(record, "state", 0) or 0) in (
            TASK_STATE_SUCCEEDED,
            TASK_STATE_FAILED,
            TASK_STATE_CANCELLED,
        ):
            return
        async with self._lock:
            record.state = TASK_STATE_SUBMITTED
            record.updated_at = time.time()
            record.ok = False
            record.exit_code = 0
            record.message = str(message or "waiting for scheduler wake")
            record.dispatch_worker_addr = ""
            record.dispatch_worker_name = ""
            record.dispatch_gpu_ids = []
            record._dispatched_gpu_id = ""
            self._clear_record_dispatch_inflight(record)

        try:
            cs = self._planner.campaign_scheduler
            cq = cs._campaign_queues.get(str(record.campaign_id or ""))
            if cq is None:
                return
            if task_id in getattr(cq, "_active_ids", set()):
                cq._active_ids.discard(task_id)
            cq._pending_ids.add(task_id)
            cq._pending_count = len(cq._pending_ids)
            cq._active_count = len(cq._active_ids)
            cq._is_empty_slow_cache = None
        except Exception:
            _LOG.debug(
                "[skip-cleanup] failed to restore pending state for %s",
                task_id[:12],
                exc_info=True,
            )

    async def _mark_record_pending_after_skip(self, record: TaskRecord) -> None:
        """Restore SUBMITTED liveness after a scheduler SKIP attempt."""
        await self._mark_record_pending_for_scheduler(
            record,
            message="waiting for scheduler wake",
        )

    @staticmethod
    def _clear_record_dispatch_inflight(record: TaskRecord) -> None:
        """Clear private worker-RPC markers after an RPC is no longer live."""
        record._dispatched_batch_id = None
        record._dispatched_worker_addr = ""

    async def _release_core_loop_owner(
        self,
        task_id: str,
        owner_task: asyncio.Task[Any] | None,
    ) -> None:
        """Release the per-task Core Loop entry guard if we still own it."""
        task_id = str(task_id or "").strip()
        if not task_id:
            return
        async with self._lock:
            owners = getattr(self, "_core_loop_active_tasks", None)
            current_owner = owners.get(task_id) if isinstance(owners, dict) else None
            if (
                current_owner is not None
                and owner_task is not None
                and current_owner is not owner_task
            ):
                return
            self._core_loop_active_task_ids.discard(task_id)
            if isinstance(owners, dict):
                owners.pop(task_id, None)

    async def _set_state(
        self,
        record: TaskRecord,
        state: int,
        *,
        ok: bool | None = None,
        exit_code: int | None = None,
        message: str | None = None,
        worker_timing_us: dict[str, int] | None = None,
    ) -> None:
        event_ts = time.time()
        async with self._lock:
            record.state = state
            record.updated_at = event_ts
            if state in (TASK_STATE_SUCCEEDED, TASK_STATE_FAILED, TASK_STATE_CANCELLED):
                dispatched_gpu = str(
                    getattr(record, "_dispatched_gpu_id", "") or ""
                ).strip()
                pending_releases = getattr(record, "_pending_vram_releases", [])
                if dispatched_gpu or pending_releases:
                    record._terminal_callback_pending = True
            if ok is not None:
                record.ok = ok
            if exit_code is not None:
                record.exit_code = int(exit_code)
            if message is not None:
                record.message = message
            if worker_timing_us is not None:
                record.worker_timing_us = dict(worker_timing_us)

        if (
            state != TASK_STATE_RUNNING
            and int(record.campaign_metadata_version or 0) >= 1
            and str(record.campaign_id or "").strip()
        ):
            try:
                await self._run_index_call(
                    self._run_index.mark_campaign_task_state,
                    campaign_id=str(record.campaign_id or ""),
                    run_name=str(record.run_name or ""),
                    submitter=str(record.submitter or ""),
                    gateway_instance_id=str(
                        self._gateway_identity.get("instance_id") or ""
                    ),
                    gateway_bind_addr=str(
                        self._gateway_identity.get("bind_addr") or ""
                    ),
                    gateway_git_commit=str(
                        self._gateway_identity.get("git_commit") or ""
                    ),
                    gateway_started_at=_to_float(
                        self._gateway_identity.get("started_at")
                    ),
                    task_state=STATE_NAMES.get(state, "UNSPECIFIED"),
                    event_ts=event_ts,
                    created_at=record.created_at,
                )
            except Exception as exc:
                _LOG.warning(
                    "[task %s] non-fatal campaign registry persistence failure "
                    "during state=%s: %s",
                    record.task_id,
                    STATE_NAMES.get(state, "UNSPECIFIED"),
                    exc,
                )

    def _evaluate_selected_worker_dispatch(
        self,
        *,
        record: TaskRecord,
        selection: WorkerSelection,
        workload_features: Mapping[str, Any] | None = None,
        execution_overrides: Mapping[str, Any] | None = None,
        config_fingerprint: str,
        input_fingerprint: str,
        selected_context_applied: str | None = None,
        planner_intent: PlannerIntent | None = None,
        execution_profile: Mapping[str, Any] | None = None,
    ) -> _DispatchEvaluation:
        normalized_component = str(record.component or "").strip().lower()
        normalized_cfg = str(config_fingerprint or "").strip()
        normalized_inp = str(input_fingerprint or "").strip()
        normalized_workload = (
            dict(workload_features or {})
            if isinstance(workload_features, Mapping)
            else {}
        )
        normalized_overrides = (
            dict(execution_overrides or {})
            if isinstance(execution_overrides, Mapping)
            else {}
        )
        effective_planner_intent = planner_intent or PlannerIntent()
        worker_context = _selected_worker_context(selection)
        prediction_source = str(selected_context_applied or "").strip()
        if not prediction_source:
            prediction_source = (
                "post_selection" if worker_context else "missing_worker_context"
            )

        def _query(overrides: Mapping[str, Any]) -> SignalResult:
            return self._signal_service.query(
                component=normalized_component,
                config_fingerprint=normalized_cfg,
                input_fingerprint=normalized_inp,
                workload_features=normalized_workload,
                execution_overrides=overrides,
                worker_context=worker_context,
                campaign_id=str(record.campaign_id or "").strip(),
                planner_intent=effective_planner_intent,
                execution_profile=execution_profile,
            )

        signal = _query(normalized_overrides)
        resident_baseline_snapshot = _normalize_resident_baseline_snapshot(
            selection.resident_baseline_snapshot
        )
        worker_activation_mib = float(
            getattr(selection, "current_activation_mib", 0.0) or 0.0
        )
        admission = evaluate_admission(
            signal=signal.bundle,
            workload_features=normalized_workload,
            execution_overrides=normalized_overrides,
            resident_baseline_mib=_to_float(
                resident_baseline_snapshot.get("resident_memory_mib")
            ),
            default_mem_safe_limit_mib=_DEFAULT_MEM_SAFE_LIMIT_MIB,
            planner_intent=effective_planner_intent,
            current_activation_mib=worker_activation_mib,
        )
        requested_batch_size = (
            _to_int(normalized_overrides.get("batch_size"))
            or _to_int(normalized_workload.get("input_batch_size"))
            or 1
        )
        initial_query_context = _mapping_payload(signal.bundle.artifacts.query_context)
        initial_execution_context = _mapping_payload(
            initial_query_context.get("execution_envelope")
        )
        initial_batch_bucket = str(
            initial_execution_context.get("effective_batch_bucket") or ""
        )
        final_overrides = dict(normalized_overrides)
        requery_applied = False
        if (
            admission.batch_size_applied is not None
            and admission.batch_size_applied > 0
        ):
            final_overrides["batch_size"] = int(admission.batch_size_applied)
            if (
                execution_profile is None
                and bucket_batch_size(admission.batch_size_applied)
                != initial_batch_bucket
            ):
                requery_applied = True
                signal = _query(final_overrides)
                admission = evaluate_admission(
                    signal=signal.bundle,
                    workload_features=normalized_workload,
                    execution_overrides=final_overrides,
                    resident_baseline_mib=_to_float(
                        resident_baseline_snapshot.get("resident_memory_mib")
                    ),
                    default_mem_safe_limit_mib=_DEFAULT_MEM_SAFE_LIMIT_MIB,
                    planner_intent=effective_planner_intent,
                    current_activation_mib=worker_activation_mib,
                )
                if (
                    admission.batch_size_applied is not None
                    and admission.batch_size_applied > 0
                ):
                    final_overrides["batch_size"] = int(admission.batch_size_applied)

        query_context_payload = _mapping_payload(signal.bundle.artifacts.query_context)
        final_execution_context = _mapping_payload(
            query_context_payload.get("execution_envelope")
        )
        query_context_payload["batch_resolution"] = {
            "requested_batch_size": int(requested_batch_size),
            "initial_effective_batch_bucket": initial_batch_bucket,
            "final_effective_batch_bucket": str(
                final_execution_context.get("effective_batch_bucket") or ""
            ),
            "batch_size_safe_cap": admission.batch_size_safe_cap,
            "batch_size_applied": admission.batch_size_applied,
            "requery_applied": bool(requery_applied),
        }
        signal = SignalResult(
            bundle=SignalBundle(
                runtime=signal.bundle.runtime,
                memory=signal.bundle.memory,
                provenance=signal.bundle.provenance,
                artifacts=SignalArtifacts(
                    query_context=query_context_payload,
                    execution_envelope=dict(signal.bundle.artifacts.execution_envelope),
                    replica_baseline=dict(signal.bundle.artifacts.replica_baseline),
                    corrections=dict(signal.bundle.artifacts.corrections),
                    guards=dict(signal.bundle.artifacts.guards),
                ),
                planner_intent=signal.bundle.planner_intent,
            ),
            worker_context_applied=signal.worker_context_applied,
            worker_context=dict(signal.worker_context),
        )
        return _DispatchEvaluation(
            selection=selection,
            signal=signal,
            admission=admission,
            selected_context_applied=prediction_source,
            execution_overrides=final_overrides,
        )

    def _select_worker(
        self,
        component: str,
        *,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        schedule_hint: Mapping[str, Any] | None = None,
        include_not_ready: bool = False,
        preferred_worker_name: str | None = None,
        campaign_id: str | None = None,
        intrinsic_signal: Any | None = None,
        planner_intent: PlannerIntent | None = None,
    ) -> WorkerSelection:
        normalized_hint = (
            dict(schedule_hint or {}) if schedule_hint is not None else None
        )
        effective_include_not_ready = bool(include_not_ready or preferred_worker_name)
        try:
            raw_selection = self.worker_selector(
                component,
                preferred_worker_addr,
                preferred_gpu_ids,
                schedule_hint=normalized_hint,
                include_not_ready=effective_include_not_ready,
                preferred_worker_name=preferred_worker_name,
                campaign_id=campaign_id,
                intrinsic_signal=intrinsic_signal,
                planner_intent=planner_intent,
            )
        except TypeError as exc:
            if "intrinsic_signal" not in str(exc) and "planner_intent" not in str(exc):
                raise
            raw_selection = self.worker_selector(
                component,
                preferred_worker_addr,
                preferred_gpu_ids,
                schedule_hint=normalized_hint,
                include_not_ready=effective_include_not_ready,
                preferred_worker_name=preferred_worker_name,
                campaign_id=campaign_id,
            )
        return _normalize_worker_selection(raw_selection)

    async def _stage_selected_worker(
        self,
        record: TaskRecord,
        worker: WorkerSelection | Mapping[str, Any],
        *,
        preferred_gpu_ids: list[str] | None = None,
        selected_context_applied: str | None = None,
    ) -> None:
        selection = _normalize_worker_selection(worker)
        worker_context = _selected_worker_context(selection)
        prediction_source = str(selected_context_applied or "").strip()
        if not prediction_source:
            prediction_source = (
                "post_selection" if worker_context else "missing_worker_context"
            )
        decision_payload = {
            "version": 5,
            "placement": _placement_payload(
                selection=selection,
                selected_worker_context=worker_context,
                selected_context_applied=prediction_source,
            ),
            "signal": {},
            "admission": {},
            "actual": {},
        }
        async with self._lock:
            record.dispatch_worker_addr = str(selection.addr or "").strip()
            record.dispatch_gpu_ids = [
                str(item).strip()
                for item in list(selection.gpu_ids or preferred_gpu_ids or [])
                if str(item).strip()
            ]
            record.dispatch_worker_name = str(selection.worker_name or "").strip()
            _copy_dispatch_resident_baseline(record, selection)
            record.worker_generation_token = str(
                getattr(selection, "worker_generation_token", "")
                or record.dispatch_resident_baseline_lifecycle_token
                or ""
            ).strip()
            record.run_ordinal_in_generation = None
            record.is_first_real_run = False
            record.decision_payload = decision_payload

    async def _apply_dispatch_evaluation(
        self,
        record: TaskRecord,
        worker: WorkerSelection | Mapping[str, Any],
        evaluation: _DispatchEvaluation,
        *,
        preferred_gpu_ids: list[str] | None = None,
    ) -> None:
        selection = _normalize_worker_selection(worker)
        async with self._lock:
            record.dispatch_worker_addr = str(selection.addr or "").strip()
            record.dispatch_gpu_ids = [
                str(item).strip()
                for item in list(selection.gpu_ids or preferred_gpu_ids or [])
                if str(item).strip()
            ]
            record.dispatch_worker_name = str(selection.worker_name or "").strip()
            _copy_dispatch_resident_baseline(record, selection)
            record.worker_generation_token = str(
                getattr(selection, "worker_generation_token", "")
                or record.dispatch_resident_baseline_lifecycle_token
                or ""
            ).strip()
            record.decision_payload = _decision_payload_from_evaluation(
                evaluation=evaluation
            )
            record.signal_runtime_sec = evaluation.signal.bundle.runtime.estimate_sec
            record.signal_runtime_upper_sec = evaluation.signal.bundle.runtime.upper_sec

    async def _apply_dispatch_attribution(
        self,
        record: TaskRecord,
        attribution: Any,
    ) -> None:
        token = str(getattr(attribution, "worker_generation_token", "") or "").strip()
        ordinal = _to_int(getattr(attribution, "run_ordinal_in_generation", None))
        is_first = _to_bool(
            getattr(attribution, "is_first_real_run", False), default=False
        )
        async with self._lock:
            if token:
                record.worker_generation_token = token
            if ordinal is not None and ordinal > 0:
                record.run_ordinal_in_generation = int(ordinal)
            record.is_first_real_run = bool(is_first)
            if token and getattr(self, "supervisor", None):
                addr = str(record.dispatch_worker_addr or "").strip()
                for _st in self.supervisor.states.values():
                    if _st.addr == addr:
                        current_token = str(
                            getattr(_st, "resident_baseline_lifecycle_token", "") or ""
                        ).strip()
                        if current_token and current_token != token:
                            record.generation_token_stale = True
                        break


    async def _activate_selected_worker(
        self,
        *,
        record: TaskRecord,
        worker: WorkerSelection,
        reservation: _AdmissionReservation,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        campaign_id: str | None = None,
        intrinsic_signal: Any | None = None,
        planner_intent: PlannerIntent | None = None,
    ) -> tuple[WorkerSelection, bool]:
        """Plan fix — returns ``(WorkerSelection, did_activate)``.

        ``did_activate`` is the supervisor's authoritative signal
        indicating whether a real cold-start was performed during this
        call (True) or the worker was already warm + the ensure call was
        idempotent (False).  Propagates up to
        ``RealityValidator._activate_worker`` → ``was_cold_start`` so the
        ``CampaignScheduler.on_task_dispatched`` init-entry creation
        branch fires only when the supervisor actually executed init
        work.  This replaces the proxy signals (``plan.needs_cold_start``
        + ``worker.ready``) whose staleness produced phantom init
        timeline entries under race conditions (fix/f failures
        observed in bench15/16/17).
        """
        worker_name = str(worker.worker_name or reservation.worker_name or "").strip()
        if self._ensure_worker_ready is None:
            reservation.release()
            raise _TaskDispatchFailure(
                f"Worker {worker_name} is not ready and cannot be activated"
            )
        try:
            did_activate = bool(
                await self._ensure_worker_ready(
                    record.component,
                    preferred_worker_addr,
                    preferred_gpu_ids,
                    preferred_worker_name=worker_name,
                    campaign_id=campaign_id,
                )
            )
            async with self._admission_sem:
                activated_worker = self._select_worker(
                    record.component,
                    preferred_worker_addr=preferred_worker_addr,
                    preferred_gpu_ids=preferred_gpu_ids,
                    preferred_worker_name=worker_name,
                    campaign_id=campaign_id,
                    intrinsic_signal=intrinsic_signal,
                    planner_intent=planner_intent,
                )
                activated_worker_name = str(activated_worker.worker_name or "").strip()
                if (
                    worker_name
                    and activated_worker_name
                    and activated_worker_name != worker_name
                ):
                    raise RuntimeError(
                        f"activation returned worker '{activated_worker_name}' for reserved worker '{worker_name}'"
                    )
                if not bool(activated_worker.ready):
                    raise RuntimeError(
                        f"Worker {worker_name} is not ready after activation"
                    )

                gpu_key = str(preferred_gpu_ids[0]) if preferred_gpu_ids else ""
                try:
                    init_mu, init_sigma2 = self.supervisor.init_tracker.predict(
                        record.component,
                        gpu_key,
                    )
                    init_sigma = (
                        (float(init_sigma2) ** 0.5)
                        if init_sigma2 and init_sigma2 > 0
                        else 0.0
                    )
                    wait_timeout_s = max(1.0, float(init_mu) + 4.0 * init_sigma)
                except Exception:
                    wait_timeout_s = 30.0
                try:
                    await self.supervisor.wait_for_ready(
                        worker_name,
                        component=record.component,
                        timeout_s=wait_timeout_s,
                    )
                except TimeoutError as exc:
                    raise RuntimeError(
                        f"Worker {worker_name} not ready within "
                        f"{wait_timeout_s:.1f}s (post-activation re-check)"
                    ) from exc

                activated_addr = str(activated_worker.addr or "").strip()
                if not activated_addr:
                    raise RuntimeError("worker address missing")
                if getattr(reservation, "state", "") == "cold_name_reserved":
                    reservation.promote_to_addr(activated_addr)
                elif getattr(reservation, "state", "") == "ready_addr_reserved":
                    if (
                        str(getattr(reservation, "worker_addr", "") or "").strip()
                        != activated_addr
                    ):
                        reservation.retarget_ready_addr(activated_addr)
                else:
                    raise RuntimeError(
                        f"cannot use reservation from state {getattr(reservation, 'state', '')}"
                    )
            return activated_worker, did_activate
        except CancelledError as _cancelled:
            reservation.release()
            raise
        except Exception as exc:
            reservation.release()
            raise _TaskDispatchFailure(
                f"Failed to activate worker {worker_name}: {exc}"
            ) from exc

    def _lookup_worker_state_for_dispatch(
        self,
        *,
        worker_name: str = "",
        worker_addr: str = "",
    ) -> Any | None:
        sup = getattr(self, "supervisor", None)
        if sup is None:
            return None
        states = getattr(sup, "states", {}) or {}
        if worker_name and worker_name in states:
            return states.get(worker_name)
        if worker_addr:
            for st in states.values():
                if str(getattr(st, "addr", "") or "").strip() == worker_addr:
                    return st
        return None

    def _front_slot_lease_state(self) -> tuple[Any, dict[str, dict[str, float]]]:
        lock = getattr(self, "_front_slot_lease_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._front_slot_lease_lock = lock
        leases = getattr(self, "_front_slot_leases", None)
        if leases is None:
            leases = {}
            self._front_slot_leases = leases
        return lock, leases

    def _front_slot_lease_key(
        self,
        *,
        worker_name: str = "",
        worker_addr: str = "",
        worker_state: Any = None,
    ) -> str:
        name = str(worker_name or "").strip()
        if name:
            return f"name:{name}"
        if worker_state is not None:
            spec = getattr(worker_state, "spec", None)
            state_name = str(getattr(spec, "name", "") or "").strip()
            if state_name:
                return f"name:{state_name}"
        addr = str(worker_addr or "").strip()
        if addr:
            st = self._lookup_worker_state_for_dispatch(worker_addr=addr)
            if st is not None:
                spec = getattr(st, "spec", None)
                state_name = str(getattr(spec, "name", "") or "").strip()
                if state_name:
                    return f"name:{state_name}"
            return f"addr:{addr}"
        return ""

    def _prune_front_slot_leases_locked(
        self,
        leases: dict[str, dict[str, float]],
        *,
        now: float | None = None,
    ) -> None:
        ttl = float(getattr(self, "_front_slot_lease_ttl_s", 0.0))
        if ttl <= 0:
            return
        now = time.time() if now is None else float(now)
        empty: list[str] = []
        for key, by_task in list(leases.items()):
            for task_id, ts in list(by_task.items()):
                if now - float(ts or 0.0) >= ttl:
                    by_task.pop(task_id, None)
            if not by_task:
                empty.append(key)
        for key in empty:
            leases.pop(key, None)

    def _front_slot_lease_count(
        self,
        *,
        worker_name: str = "",
        worker_addr: str = "",
        worker_state: Any = None,
        exclude_task_id: str = "",
    ) -> int:
        key = self._front_slot_lease_key(
            worker_name=worker_name,
            worker_addr=worker_addr,
            worker_state=worker_state,
        )
        if not key:
            return 0
        lock, leases = self._front_slot_lease_state()
        exclude = str(exclude_task_id or "").strip()
        with lock:
            self._prune_front_slot_leases_locked(leases)
            by_task = leases.get(key, {})
            return sum(1 for task_id in by_task if task_id != exclude)

    def _front_slot_lease_count_for_plan(
        self,
        plan: Any,
        worker_state: Any = None,
        *,
        exclude_task_id: str = "",
    ) -> int:
        return self._front_slot_lease_count(
            worker_name=str(getattr(plan, "target_worker_name", "") or ""),
            worker_addr=str(getattr(plan, "target_worker_addr", "") or ""),
            worker_state=worker_state,
            exclude_task_id=exclude_task_id,
        )

    def _try_claim_front_slot_lease(
        self,
        *,
        task_id: str,
        plan: Any,
        worker_state: Any = None,
    ) -> bool:
        cap = int(getattr(self, "dispatch_backlog_per_worker", 1) or 0)
        if cap <= 0:
            return True
        task_id = str(task_id or "").strip()
        if not task_id:
            return True
        worker_name = str(getattr(plan, "target_worker_name", "") or "")
        worker_addr = str(getattr(plan, "target_worker_addr", "") or "")
        if worker_state is None:
            worker_state = self._lookup_worker_state_for_dispatch(
                worker_name=worker_name,
                worker_addr=worker_addr,
            )
        key = self._front_slot_lease_key(
            worker_name=worker_name,
            worker_addr=worker_addr,
            worker_state=worker_state,
        )
        if not key:
            return True
        lock, leases = self._front_slot_lease_state()
        with lock:
            self._prune_front_slot_leases_locked(leases)
            by_task = leases.setdefault(key, {})
            if task_id in by_task:
                by_task[task_id] = time.time()
                return True
            backlog = self._dispatch_front_backlog(worker_state)
            if backlog + len(by_task) >= cap:
                return False
            by_task[task_id] = time.time()
            return True

    def _release_front_slot_lease_for_task(self, task_id: str) -> None:
        task_id = str(task_id or "").strip()
        if not task_id:
            return
        lock, leases = self._front_slot_lease_state()
        with lock:
            empty: list[str] = []
            for key, by_task in list(leases.items()):
                by_task.pop(task_id, None)
                if not by_task:
                    empty.append(key)
            for key in empty:
                leases.pop(key, None)

    @staticmethod
    def _dispatch_front_backlog(
        st: Any,
        *,
        current_reservation_tokens: int = 0,
    ) -> int:
        """Return scheduler-hidden queued/not-yet-executing dispatch count.

        Running execute slots are intentionally excluded: parallel execution
        is allowed.  The cap applies only to work that has left gateway
        scheduling but has not reached execute yet.

        ``current_reservation_tokens`` is used by the post-reservation
        validation path.  The current task's token lives only in the local
        ``dispatch_pending`` counter at that point; worker/RPC queue counters
        describe pre-existing work and must not be decremented.  Subtracting
        from the aggregate backlog would hide an older worker-side waiter and
        allow an effective front depth of two under cap=1.
        """
        if st is None:
            return 0
        dispatch_pending = int(getattr(st, "dispatch_pending", 0) or 0)
        state = str(getattr(st, "state", "") or "").strip().lower()
        if (
            dispatch_pending > 0
            and state != "starting"
            and int(getattr(st, "queue_gw_inflight", 0) or 0) == 0
            and int(getattr(st, "queue_in_queue", 0) or 0) == 0
            and int(getattr(st, "queue_prepare_inflight", 0) or 0) == 0
            and int(getattr(st, "queue_prepared_queue", 0) or 0) == 0
            and int(getattr(st, "queue_execute_inflight", 0) or 0) == 0
            and int(getattr(st, "queue_finalize_inflight", 0) or 0) == 0
        ):
            updated_at = float(getattr(st, "dispatch_pending_updated_at", 0.0) or 0.0)
            age = time.time() - updated_at if updated_at > 0.0 else 0.0
            if updated_at > 0.0 and age >= 2.0:
                dispatch_pending = 0
        if current_reservation_tokens > 0:
            dispatch_pending = max(
                0,
                dispatch_pending - int(current_reservation_tokens),
            )
        worker_not_executing = (
            int(getattr(st, "queue_in_queue", 0) or 0)
            + int(getattr(st, "queue_prepare_inflight", 0) or 0)
            + int(getattr(st, "queue_prepared_queue", 0) or 0)
        )
        return max(0, dispatch_pending, worker_not_executing)

    def _available_dispatch_front_slots(self) -> int:
        cap = int(getattr(self, "dispatch_backlog_per_worker", 1) or 0)
        if cap <= 0:
            return 1
        sup = getattr(self, "supervisor", None)
        if sup is None:
            return 1
        reconcile = getattr(sup, "reconcile_dispatch_front_capacity", None)
        if callable(reconcile):
            try:
                reconcile()
            except Exception:
                _LOG.debug(
                    "[dispatch-front] capacity reconciliation failed",
                    exc_info=True,
                )
        states = getattr(sup, "states", {}) or {}
        slots = 0
        for worker_name, st in states.items():
            state = str(getattr(st, "state", "") or "").strip().lower()
            lifecycle = str(getattr(st, "lifecycle_state", "") or "").strip().lower()
            if state not in {"ready", "starting"} and lifecycle not in {
                "ready",
                "starting",
            }:
                continue
            leased = self._front_slot_lease_count(
                worker_name=str(worker_name or ""),
                worker_addr=str(getattr(st, "addr", "") or ""),
                worker_state=st,
            )
            slots += max(0, cap - self._dispatch_front_backlog(st) - leased)
        return slots

    def _worker_front_has_capacity(self, worker_name: str) -> bool:
        cap = int(getattr(self, "dispatch_backlog_per_worker", 1) or 0)
        if cap <= 0:
            return True
        sup = getattr(self, "supervisor", None)
        if sup is None:
            return True
        states = getattr(sup, "states", {}) or {}
        st = states.get(str(worker_name or ""))
        if st is None:
            return True
        reconcile = getattr(sup, "reconcile_dispatch_front_capacity", None)
        if callable(reconcile):
            try:
                reconcile(str(worker_name or ""))
            except Exception:
                _LOG.debug(
                    "[dispatch-front] worker capacity reconciliation failed worker=%s",
                    worker_name,
                    exc_info=True,
                )
        state = str(getattr(st, "state", "") or "").strip().lower()
        lifecycle = str(getattr(st, "lifecycle_state", "") or "").strip().lower()
        if state not in {"ready", "starting"} and lifecycle not in {
            "ready",
            "starting",
        }:
            return False
        leased = self._front_slot_lease_count(
            worker_name=str(worker_name or ""),
            worker_addr=str(getattr(st, "addr", "") or ""),
            worker_state=st,
        )
        return self._dispatch_front_backlog(st) + leased < cap

    def _dispatch_front_violation(
        self,
        *,
        record: TaskRecord,
        worker_name: str,
        worker_addr: str,
        gpu_id: str,
        was_cold_start: bool = False,
    ) -> Any | None:
        from .planning.contracts import ConstraintViolation, ViolationType

        cap = int(getattr(self, "dispatch_backlog_per_worker", 1) or 0)
        if cap <= 0:
            return None
        sup = getattr(self, "supervisor", None)
        reconcile = (
            getattr(sup, "reconcile_dispatch_front_capacity", None)
            if sup is not None
            else None
        )
        if callable(reconcile):
            try:
                reconcile(str(worker_name or ""))
            except Exception:
                _LOG.debug(
                    "[dispatch-front] pre-dispatch reconciliation failed worker=%s",
                    worker_name or worker_addr,
                    exc_info=True,
                )
        st = self._lookup_worker_state_for_dispatch(
            worker_name=worker_name,
            worker_addr=worker_addr,
        )
        backlog = self._dispatch_front_backlog(st)
        preexisting_backlog = self._dispatch_front_backlog(
            st,
            current_reservation_tokens=1,
        )
        if preexisting_backlog >= cap:
            _LOG.debug(
                "[dispatch-front] worker_queue_saturated worker=%s gpu=%s "
                "hidden_backlog=%d preexisting=%d cap=%d task=%s",
                worker_name or worker_addr,
                gpu_id,
                backlog,
                preexisting_backlog,
                cap,
                record.task_id,
            )
            return ConstraintViolation(
                violation_type=ViolationType.WORKER_QUEUE_SATURATED,
                gpu_id=str(gpu_id or ""),
                worker_name=worker_name,
                active_count=backlog,
            )
        return None

    def _worker_queue_saturated_retry_deferral(
        self,
        *,
        record: TaskRecord,
        violation: Any,
        config_fingerprint: str,
    ) -> Any:
        from .planning.scheduling_supervisor import RetryDeferral

        worker_name = str(getattr(violation, "worker_name", "") or "")
        gpu_id = str(getattr(violation, "gpu_id", "") or "")
        active_count = int(getattr(violation, "active_count", 0) or 0)
        return RetryDeferral(
            reason="worker_front",
            worker_name=worker_name,
            component=str(record.component or "").strip(),
            config_fingerprint=str(config_fingerprint or "").strip(),
            gpu_id=gpu_id,
            active_count=active_count,
        )

    def _worker_not_ready_retry_deferral(self) -> Any:
        from .planning.scheduling_supervisor import RetryDeferral

        return RetryDeferral(
            reason="worker_not_ready",
            not_before_at=time.time() + self.WORKER_NOT_READY_RETRY_DELAY_SEC,
        )

    async def _notify_task_end(
        self,
        addr: str,
        *,
        activation_mb: int = 0,
        activation_is_guarded: bool = False,
        worker_name: str = "",
    ) -> None:
        if not self._on_task_end:
            return
        try:
            await self._on_task_end(
                addr,
                activation_mb=activation_mb,
                activation_is_guarded=activation_is_guarded,
                worker_name=worker_name,
            )
        except TypeError:
            await self._on_task_end(
                addr,
                activation_mb=activation_mb,
                activation_is_guarded=activation_is_guarded,
            )

    async def _dispatch_selected_worker(
        self,
        *,
        record: TaskRecord,
        worker: WorkerSelection,
        reservation: _AdmissionReservation,
        batch: Any,
        timeout_s: int,
        campaign_id: str | None = None,
        is_backfill: bool = False,
        was_cold_start: bool = False,
        reciprocal_interference: Mapping[str, Any] | None = None,
        dynamic_batch_context: Mapping[str, Any] | None = None,
    ) -> Any:
        addr = str(worker.addr or "").strip()
        if not addr:
            raise _TaskDispatchFailure("worker address missing")
        worker_name = str(worker.worker_name or "").strip()
        gpu_ids = [
            str(item).strip()
            for item in list(worker.gpu_ids or record.dispatch_gpu_ids or [])
            if str(item).strip()
        ]
        gid = gpu_ids[0] if gpu_ids else ""

        latency_batch_context: dict[str, Any] | None = None
        if isinstance(dynamic_batch_context, Mapping):
            record._dynamic_batch_context = dict(dynamic_batch_context)
        self._release_front_slot_lease_for_task(record.task_id)
        _front_violation = self._dispatch_front_violation(
            record=record,
            worker_name=worker_name,
            worker_addr=addr,
            gpu_id=gid,
            was_cold_start=was_cold_start,
        )
        if _front_violation is not None:
            reservation.release()
            raise _DispatchRetrySignal(
                failed_worker_addr=addr,
                failed_worker_name=worker_name,
                cause=RuntimeError("worker dispatch-front backlog saturated"),
                constraint_violation=_front_violation,
            )
        limit_raw = worker.max_concurrency or self.max_inflight_per_worker
        try:
            limit = max(1, int(limit_raw))
        except Exception:
            limit = self.max_inflight_per_worker
        sem = self._worker_limits.setdefault(addr, asyncio.Semaphore(limit))

        component = str(record.component or "").strip().lower()
        config_fp = str(record.config_fingerprint or "").strip()
        features = dict(record.workload_features or {})

        activation_mb = int(getattr(record, "dispatch_vram_budget_mb", 0) or 0)
        host_ram_mb = int(getattr(record, "dispatch_ram_budget_mb", 0) or 0)
        activation_is_guarded = True
        input_size = 0.0

        try:
            from .signals.resource_profile import ResourceProfileRegistry

            input_size = ResourceProfileRegistry.extract_input_size(
                features, component=component
            )
            if activation_mb > 0:
                gp_sufficient = (
                    self._signal_service.resource_profiles.is_vram_sufficient(
                        component,
                        config_fp or "__default__",
                        required_relative_error=0.20,
                    )
                )
                if gp_sufficient:
                    activation_is_guarded = False
                _LOG.info(
                    "[gp-activation] %s cfg=%s input=%.0f: %d MB "
                    "(planner budget, %s guard)",
                    component,
                    config_fp,
                    input_size,
                    activation_mb,
                    "strict" if not activation_is_guarded else "lenient",
                )
            else:
                gp_pred = self._signal_service.resource_profiles.predict_vram(
                    component,
                    config_fp or "__default__",
                    input_size=input_size,
                )
                if gp_pred is not None and gp_pred > 0:
                    activation_mb = int(gp_pred)
                    gp_sufficient = (
                        self._signal_service.resource_profiles.is_vram_sufficient(
                            component,
                            config_fp or "__default__",
                            required_relative_error=0.20,
                        )
                    )
                    if gp_sufficient:
                        activation_is_guarded = False
                        _LOG.info(
                            "[gp-activation] %s cfg=%s input=%.0f: %d MB "
                            "(legacy GP-mean fallback, confident, strict guard)",
                            component,
                            config_fp,
                            input_size,
                            activation_mb,
                        )
                    else:
                        _LOG.info(
                            "[gp-activation] %s cfg=%s input=%.0f: %d MB "
                            "(legacy μ+zσ fallback, low confidence, lenient guard)",
                            component,
                            config_fp,
                            input_size,
                            activation_mb,
                        )
                else:
                    activation_mb = self._cold_start_activation_mb(addr)
                    _LOG.info(
                        "[gp-activation] %s cfg=%s: no GP data, cold-start fallback "
                        "activation_mb=%d (weight×%.1f, guarded)",
                        component,
                        config_fp,
                        activation_mb,
                        getattr(
                            getattr(self, "supervisor", None),
                            "cold_start_activation_ratio",
                            1.0,
                        ),
                    )
        except Exception:
            activation_mb = self._cold_start_activation_mb(addr)
            _LOG.debug(
                "[gp-activation] %s: GP exception, cold-start fallback activation_mb=%d",
                component,
                activation_mb,
            )

        if host_ram_mb <= 0:
            try:
                ram_pred = self._signal_service.resource_profiles.predict_ram_upper(
                    component,
                    config_fp or "__default__",
                    input_size=input_size,
                    z=getattr(
                        self._planner.campaign_scheduler,
                        "resource_upper_z",
                        1.645,
                    ),
                )
                if ram_pred is None or ram_pred <= 0:
                    ram_pred = self._signal_service.resource_profiles.predict_ram(
                        component,
                        config_fp or "__default__",
                        input_size=input_size,
                    )
                if ram_pred is not None and ram_pred > 0:
                    host_ram_mb = int(ram_pred)
            except Exception:
                host_ram_mb = 0

        priority = int(_to_int(worker.priority) or 100)
        started_addr = ""
        vram_acquired = False
        host_ram_acquired = False
        reciprocal_exec_rollback: dict[str, Any] | None = None
        rpc_started = False
        drop_latency_observation = False

        if self.resource_tracker and activation_mb > 0 and gpu_ids:
            gid = gpu_ids[0]
            total_mb = self.resource_tracker.total_vram.get(gid, 24576)
            if activation_mb > total_mb:
                _LOG.warning(
                    "[gp-activation-cap] %s gpu=%s: capped %d MB → %d MB "
                    "(exceeds GPU total)",
                    component,
                    gid,
                    activation_mb,
                    total_mb,
                )
                activation_mb = total_mb

        try:
            async with sem:
                gid = gpu_ids[0] if gpu_ids else ""
                cs = getattr(
                    getattr(self, "_planner", None), "_campaign_scheduler", None
                )
                cancel_fn = getattr(self, "_cancel_task_impl", None)
                _sup = getattr(self, "supervisor", None)
                idle_freeable_fn = _sup.estimate_idle_freeable if _sup else None

                if self.resource_tracker and (
                    activation_mb > 0
                    or host_ram_mb > 0
                    or getattr(self.resource_tracker, "_host_ram_min_available_mib", 0)
                    > 0
                ):
                    def _next_completion() -> float | None:
                        try:
                            tl = getattr(cs, "_timelines", None) if cs else None
                            if tl is not None and hasattr(
                                tl,
                                "next_predicted_completion",
                            ):
                                return tl.next_predicted_completion()
                        except Exception:
                            return None
                        return None

                    from .supervisor import HostRAMSaturatedError as _HostRAM

                    try:
                        _acq_result = await self.resource_tracker.acquire_activation(
                            gpu_ids,
                            activation_mb,
                            host_ram_mb=host_ram_mb,
                            priority=priority,
                            campaign_id=str(campaign_id or ""),
                            is_backfill=is_backfill,
                            task_id=str(record.task_id or ""),
                            required_duration_sec=max(
                                0.0,
                                _to_float(record.signal_runtime_upper_sec) or 0.0,
                            ),
                            evict_fn=self.evict_fn,
                            cancel_backfill_fn=cancel_fn,
                            campaign_scheduler=cs,
                            idle_freeable_fn=idle_freeable_fn,
                            next_predicted_completion_fn=_next_completion,
                        )
                    except _HostRAM as _hr_err:
                        from .planning.contracts import (
                            ConstraintViolation as _CV,
                        )

                        raise _DispatchRetrySignal(
                            failed_worker_addr=addr,
                            failed_worker_name=worker_name,
                            cause=_hr_err,
                            constraint_violation=_CV(
                                violation_type="host_ram_saturated",
                                gpu_id=str(gid or _hr_err.gpu_id or ""),
                                worker_name=worker_name,
                                host_mem_available_mib=_hr_err.mem_available_mib,
                                host_mem_threshold_mib=_hr_err.threshold_mib,
                            ),
                        ) from _hr_err
                    if type(_acq_result) is int:
                        activation_mb = _acq_result
                        vram_acquired = activation_mb > 0
                        host_ram_acquired = True
                    else:
                        vram_acquired = bool(_acq_result) and activation_mb > 0
                        host_ram_acquired = bool(_acq_result) and (
                            host_ram_mb > 0
                            or getattr(
                                self.resource_tracker,
                                "_host_ram_min_available_mib",
                                0,
                            )
                            > 0
                        )
                if activation_mb > 0 and not vram_acquired:
                    from .planning.contracts import ConstraintViolation as _CV

                    _avail = 0
                    try:
                        if self.resource_tracker is None:
                            raise RuntimeError("resource tracker unavailable")
                        _avail = int(
                            self.resource_tracker.get_available_memory(gid or "")
                        )
                    except Exception:
                        _avail = 0
                    raise _DispatchRetrySignal(
                        failed_worker_addr=addr,
                        failed_worker_name=worker_name,
                        cause=RuntimeError(
                            "VRAM admission stall — shortfall beyond idle-freeing",
                        ),
                        constraint_violation=_CV(
                            violation_type="vram_insufficient",
                            gpu_id=str(gid or ""),
                            worker_name=worker_name,
                            requested_vram_mb=int(activation_mb),
                            available_vram_mb=_avail,
                        ),
                    )

                if self.resource_tracker and cs and gid:
                    (
                        compute_ok,
                        compute_gid,
                    ) = await self.resource_tracker.acquire_compute_slot(
                        gpu_id=gid,
                        component=component,
                        campaign_scheduler=cs,
                        is_backfill=is_backfill,
                        cancel_backfill_fn=cancel_fn,
                        campaign_id=str(campaign_id or ""),
                    )
                    if not compute_ok:
                        from .planning.contracts import ConstraintViolation as _CV

                        raise _DispatchRetrySignal(
                            failed_worker_addr=addr,
                            failed_worker_name=worker_name,
                            cause=RuntimeError("planned GPU blocked — re-scheduling"),
                            constraint_violation=_CV(
                                violation_type="compute_saturated",
                                gpu_id=str(gid or ""),
                                worker_name=worker_name,
                            ),
                        )

                _worker_context = dict(worker.estimator_worker_context or {})
                _worker_state = self.supervisor.states.get(str(worker_name or ""))
                _worker_context.update(
                    {
                        key: value
                        for key, value in _worker_execution_identity(
                            _worker_state
                        ).items()
                        if value
                    }
                )
                _gpu_models = getattr(
                    getattr(self._planner, "_global_planner", None),
                    "_gpu_models",
                    {},
                )
                if not isinstance(_gpu_models, Mapping):
                    _gpu_models = {}
                _gpu_model = str(
                    _worker_context.get("gpu_model")
                    or _gpu_models.get(str(gid), "")
                    or ""
                )
                _reciprocal_event_id = (
                    f"{str(record.task_id or '').strip()}:{batch.batch_id}"
                )
                reciprocal_exec_rollback = self._on_inference_start(
                    task_id=str(record.task_id or "").strip(),
                    component=component,
                    gpu_id=gid,
                    campaign_id=str(campaign_id or ""),
                    was_cold_start=was_cold_start,
                    input_size=float(input_size or 0.0),
                    config_fingerprint=config_fp,
                    input_fingerprint=str(
                        getattr(record, "input_fingerprint", "") or ""
                    ),
                    worker_name=str(worker_name or "").strip(),
                    active_cancel_safe=self._supports_active_cancel(
                        component,
                        worker_name=str(worker_name or "").strip(),
                    ),
                    reciprocal_interference=reciprocal_interference,
                    reciprocal_event_id=_reciprocal_event_id,
                    gpu_model=_gpu_model,
                    mps_mode=str(_worker_context.get("mps_mode") or ""),
                    worker_backend=str(_worker_context.get("worker_backend") or ""),
                    actor_model=str(_worker_context.get("actor_model") or ""),
                    adapter_version=str(
                        _worker_context.get("adapter_version")
                        or _worker_context.get("model_version")
                        or ""
                    ),
                )
                record._reciprocal_event_id = _reciprocal_event_id
                record._dispatched_gpu_id = gid
                record._dispatched_batch_id = batch.batch_id
                record._dispatched_worker_addr = addr
                predicted_vram = self._planner.campaign_scheduler._predict_vram(
                    component,
                    float(input_size or 0.0),
                    gid,
                    config_fingerprint=config_fp,
                )
                if not hasattr(record, "_pending_vram_releases"):
                    record._pending_vram_releases = []
                record._pending_vram_releases.append((gid, component, predicted_vram))

                reservation.consume_on_task_start(
                    self._on_task_start,
                    activation_mb=activation_mb,
                    activation_is_guarded=activation_is_guarded,
                )
                started_addr = addr

                task_id = str(record.task_id or "").strip()
                if task_id:
                    _len_parts = []
                    for _k in sorted(features.keys()):
                        _v = features.get(_k)
                        if _v is not None and _v != "" and _v != 0:
                            try:
                                if float(_v) != 0:
                                    _len_parts.append(f"{_k}={_v}")
                            except (TypeError, ValueError):
                                continue
                    registry_key = (
                        component,
                        config_fp,
                        "|".join(_len_parts) or "default",
                    )
                    await asyncio.to_thread(
                        self._signal_service.latency_on_task_enter,
                        addr,
                        task_id=task_id,
                        component=component,
                        registry_key=registry_key,
                        workload_features=features,
                        config_fingerprint=config_fp,
                        campaign_id=str(campaign_id or ""),
                        gpu_ids=gpu_ids,
                        input_fingerprint=str(
                            getattr(record, "input_fingerprint", "") or ""
                        ),
                        gpu_model=_gpu_model,
                        mps_mode=str(_worker_context.get("mps_mode") or ""),
                        worker_backend=str(_worker_context.get("worker_backend") or ""),
                        actor_model=str(_worker_context.get("actor_model") or ""),
                        adapter_version=str(
                            _worker_context.get("adapter_version")
                            or _worker_context.get("model_version")
                            or ""
                        ),
                    )

                channel = self._get_channel(addr)
                stub = pb_grpc.ModelWorkerStub(channel)
                effective_timeout = timeout_s
                if effective_timeout and getattr(self, "supervisor", None):
                    for _st in self.supervisor.states.values():
                        if _st.addr == addr:
                            _depth = max(
                                0, _st.queue_execute_inflight + _st.queue_in_queue
                            )
                            if _worker_uses_serialized_execute(_st) and _depth > 0:
                                effective_timeout = effective_timeout * (_depth + 1)
                            break
                rpc_started = True
                response = await stub.InferBatch(
                    batch,
                    timeout=effective_timeout or None,
                )
                if (
                    isinstance(dynamic_batch_context, Mapping)
                    and len(response.responses) == 1
                ):
                    response_item = response.responses[0]
                    worker_telemetry = _extract_task_telemetry(
                        response_item.payload_json
                    )
                    candidate = _batch_observation_context(
                        record,
                        worker_telemetry,
                        succeeded=bool(response_item.ok),
                    )
                    if candidate.get("companion_eligible"):
                        latency_batch_context = candidate
                return response
        except BaseException as exc:
            if reciprocal_exec_rollback is not None and not rpc_started:
                self._rollback_inference_start(reciprocal_exec_rollback)
                reciprocal_exec_rollback = None
            if isinstance(exc, CancelledError):
                current_handle = self._handles.get(record.task_id)
                if (
                    rpc_started
                    and not self._closing
                    and not getattr(record, "_cancel_reason", "")
                    and not getattr(current_handle, "_cancel_requested", True)
                ):
                    drop_latency_observation = True
                    raise _DispatchRetrySignal(
                        failed_worker_addr=addr,
                        failed_worker_name=worker_name,
                        cause=RuntimeError("worker RPC transport cancelled"),
                        constraint_violation=None,
                    ) from exc
                reservation.release()
                raise
            if isinstance(exc, _TaskDispatchFailure):
                reservation.release()
                raise
            if (
                not rpc_started
                and isinstance(exc, Exception)
                and not isinstance(exc, _DispatchRetrySignal)
            ):
                from .planning.contracts import (
                    ConstraintViolation as _CV,
                )
                from .planning.contracts import ViolationSource as _VS
                from .planning.contracts import ViolationType as _VT

                reservation.release()
                raise _DispatchRetrySignal(
                    failed_worker_addr=addr,
                    failed_worker_name=worker_name,
                    cause=exc,
                    constraint_violation=_CV(
                        violation_type=_VT.SCHEDULER_INTERNAL_ERROR,
                        gpu_id=str(gid or ""),
                        worker_name=worker_name,
                        source=_VS.SCHEDULER,
                    ),
                ) from exc
            if isinstance(exc, grpc.aio.AioRpcError):
                retryable = exc.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNKNOWN,
                )
                drop_latency_observation = True
                if retryable:
                    channel_to_close = self._channel_pool.pop(addr, None)
                    if channel_to_close is not None:
                        with contextlib.suppress(Exception):
                            await channel_to_close.close()
                    raise _DispatchRetrySignal(
                        failed_worker_addr=addr,
                        failed_worker_name=worker_name,
                        cause=exc,
                        constraint_violation=None,
                    ) from exc
                raise
            if isinstance(exc, Exception) and not started_addr:
                reservation.release()
            raise
        finally:

            async def _cleanup():
                if started_addr:
                    task_id = str(record.task_id or "").strip()
                    if task_id:
                        _cancel_reason = getattr(record, "_cancel_reason", "") or ""
                        try:
                            if _cancel_reason or drop_latency_observation:
                                await asyncio.to_thread(
                                    self._signal_service.latency_drop_task,
                                    started_addr,
                                    task_id,
                                )
                            else:
                                await asyncio.to_thread(
                                    self._signal_service.latency_on_task_exit,
                                    started_addr,
                                    task_id,
                                    latency_batch_context,
                                )
                        except Exception:
                            _LOG.exception("Error in latency tracking during cleanup")
                if self.resource_tracker and (vram_acquired or host_ram_acquired):
                    try:
                        await self.resource_tracker.release_activation(
                            gpu_ids,
                            activation_mb if vram_acquired else 0,
                            host_ram_mb=host_ram_mb if host_ram_acquired else 0,
                        )
                    except Exception:
                        _LOG.exception("Error in release_activation during cleanup")
                if started_addr and self._on_task_end:
                    try:
                        await self._notify_task_end(
                            started_addr,
                            activation_mb=activation_mb,
                            activation_is_guarded=activation_is_guarded,
                            worker_name=worker_name,
                        )
                    except Exception:
                        _LOG.exception("Error in _notify_task_end during cleanup")
                    self._clear_record_dispatch_inflight(record)

            try:
                await asyncio.shield(_cleanup())
            except CancelledError as _cancelled:
                raise

    async def _ensure_core_loop_components(self) -> None:
        """Lazy-init of the 3-layer architecture components that the
        producer path needs before submitting a TaskHandle to the
        SchedulingSupervisor.  Idempotent via the ``is None`` guards:
        callable from every producer site, does work exactly once per
        gateway lifecycle (Plan , ,
        ).

        Post-condition: ``self._scheduling_supervisor is not None`` AND
        ``self._reality_validator is not None``.  Producer sites rely
        on this invariant; a ``None`` observed after this call is an
        invariant violation (``RuntimeError``), not a graceful
        degradation path.  Tests must call ``await service.close()``
        to tear down the drain / observers for clean event-loop
        shutdown — the HEAD-era "pre-wire RealityValidator → skip all
        background tasks" shortcut is intentionally removed.
        """
        async with self._core_loop_init_lock:
            await self._ensure_core_loop_components_locked()

    async def _ensure_core_loop_components_locked(self) -> None:
        try:
            self._planner._connect_supervisor_refs()
        except Exception as _cs_exc:
            raise RuntimeError(
                "GlobalPlanner wiring failed during core-loop initialization"
            ) from _cs_exc
        _global_planner = getattr(self._planner, "_global_planner", None)
        if _global_planner is None:
            raise RuntimeError(
                "GlobalPlanner not initialized; cannot start Supervisor",
            )

        if self._reality_validator is None:
            from .planning.reality_validator import RealityValidator

            self._reality_validator = RealityValidator(self, _global_planner)
            try:
                sup = getattr(self, "supervisor", None)
                intf_reg = getattr(self._signal_service, "_interference_registry", None)
                if (
                    sup is not None
                    and intf_reg is not None
                    and hasattr(intf_reg, "set_maturity_observations_per_dim")
                ):
                    intf_reg.set_maturity_observations_per_dim(
                        int(getattr(sup, "gp_maturity_observations_per_dim", 10)),
                    )
                if (
                    sup is not None
                    and intf_reg is not None
                    and hasattr(intf_reg, "set_slowdown_prior_means")
                ):
                    intf_reg.set_slowdown_prior_means(
                        self_slowdown=float(
                            getattr(
                                sup,
                                "self_slowdown_cold_start_default",
                                2.0,
                            )
                        ),
                        pairwise_slowdown=float(
                            getattr(
                                sup,
                                "pairwise_slowdown_cold_start",
                                1.3,
                            )
                        ),
                    )
                if (
                    sup is not None
                    and intf_reg is not None
                    and hasattr(intf_reg, "set_self_interference_prior_coeff")
                ):
                    coeff = float(getattr(sup, "self_interference_prior", 1.0))
                    intf_reg.set_self_interference_prior_coeff(coeff)
            except Exception as _mat_exc:
                _LOG.warning(
                    "[core-loop] maturity-threshold wiring skipped: %s",
                    _mat_exc,
                )
            if not self._sweeper_started:
                tracker = getattr(_global_planner, "_constraint_tracker", None)
                if tracker is not None and hasattr(tracker, "start_sweeper"):
                    try:
                        await tracker.start_sweeper()
                        self._sweeper_started = True
                    except Exception as _sw_exc:
                        _LOG.warning(
                            "[core-loop] ConstraintTracker sweeper start failed: %s",
                            _sw_exc,
                        )

        if self._scheduling_supervisor is None:
            from .planning.scheduling_supervisor import SchedulingSupervisor

            _cfg = getattr(self, "_runtime_config", {}) or {}

            def _next_event_provider() -> float | None:
                try:
                    cs = getattr(_global_planner, "campaign_scheduler", None)
                    tl = getattr(cs, "_timelines", None) if cs else None
                    reconcile = getattr(
                        _global_planner, "reconcile_reciprocal_event", None
                    )
                    if (
                        bool(
                            getattr(
                                _global_planner,
                                "reciprocal_interference_correction",
                                False,
                            )
                        )
                        and callable(reconcile)
                        and tl is not None
                    ):
                        reconcile(list(getattr(tl, "gpu_ids", ())), "periodic_tick")
                    if tl is not None and hasattr(
                        tl,
                        "next_predicted_completion",
                    ):
                        return tl.next_predicted_completion()
                except Exception:
                    return None
                return None

            sup = SchedulingSupervisor(
                periodic_tick_sec=float(_cfg.get("periodic_tick_sec", 2.0)),
                jitter_max_sec=float(_cfg.get("jitter_max_sec", 0.5)),
                next_event_provider=_next_event_provider,
                available_dispatch_front_slots_provider=(
                    self._available_dispatch_front_slots
                ),
                worker_front_has_capacity_provider=(self._worker_front_has_capacity),
                batch_plan_provider=self._batch_plan_ready_handles,
                strict_task_token_dedup=bool(
                    _cfg.get("proton_naive_scheduler_token_dedup", False)
                ),
            )
            await sup.start()
            self._scheduling_supervisor = sup
            _supervisor = getattr(self, "supervisor", None)
            if _supervisor is not None:
                _supervisor.dispatch_front_capacity_hook = (
                    sup.notify_worker_front_capacity
                )
            tracker = getattr(_global_planner, "_constraint_tracker", None)
            if tracker is not None and hasattr(tracker, "set_wake_hook"):
                tracker.set_wake_hook(sup.notify_wake)
            if hasattr(_global_planner, "attach_supervisor_wake"):
                _global_planner.attach_supervisor_wake(sup)
            _cs_obj = getattr(_global_planner, "campaign_scheduler", None)
            if _cs_obj is not None and hasattr(_cs_obj, "attach_supervisor_wake"):
                _cs_obj.attach_supervisor_wake(sup)

        if self._gpu_health_observer is None:
            try:
                from .planning.gpu_health_observer import GpuHealthObserver

                _cfg = getattr(self, "_runtime_config", {}) or {}
                gho = GpuHealthObserver(
                    _global_planner,
                    poll_interval_sec=float(
                        _cfg.get("gpu_health_poll_interval_sec", 30.0),
                    ),
                    min_settle_time_sec=float(
                        _cfg.get("min_settle_time_sec", 30.0),
                    ),
                    max_quarantine_sec=float(
                        _cfg.get("max_quarantine_sec", 3600.0),
                    ),
                    force_clear_after_sec=_cfg.get("force_clear_after_sec", None),
                )
                await gho.start()
                self._gpu_health_observer = gho
            except Exception as _gho_exc:
                _LOG.warning(
                    "[core-loop] GpuHealthObserver start failed: %s",
                    _gho_exc,
                )

        if self._event_handler is None:
            try:
                from .planning.event_handler import EventHandler

                _cs_obj = getattr(_global_planner, "campaign_scheduler", None)
                if _cs_obj is not None:
                    _cfg = getattr(self, "_runtime_config", {}) or {}
                    _gp_cfg = (
                        _cfg.get("global_planner", {}) if isinstance(_cfg, dict) else {}
                    )
                    if not isinstance(_gp_cfg, dict):
                        _gp_cfg = {}
                    eh = EventHandler(
                        self,
                        _global_planner,
                        _cs_obj,
                        safety_net_gc_interval_sec=float(
                            _cfg.get("safety_net_gc_interval_sec", 60.0),
                        ),
                        snapshot_max_age_sec=float(
                            _cfg.get("snapshot_max_age_sec", 10.0),
                        ),
                        pre_init_fire_interval_sec=float(
                            _cfg.get("pre_init_fire_interval_sec", 1.0),
                        ),
                        eviction_grace=bool(_gp_cfg.get("dynamic_grace", False)),
                        drift_eviction=bool(_gp_cfg.get("drift_eviction", False)),
                    )
                    await eh.start()
                    self._event_handler = eh
            except Exception as _eh_exc:
                _LOG.warning(
                    "[core-loop] EventHandler start failed: %s",
                    _eh_exc,
                )

    def _build_task_handle(
        self,
        record: TaskRecord,
        *,
        argv: list,
        tool_cwd,
        env,
        timeout_s,
        profiling,
        workload_features=None,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        execution_overrides=None,
        preferred_worker_addr=None,
        preferred_gpu_ids=None,
        output_sample_count=None,
        process_name=None,
        is_backfill_hint: bool = False,
    ):
        """Plan — construct a TaskHandle whose
        ``run`` closure executes exactly one per-task Core Loop attempt.
        The Supervisor drain is the SOLE executor: the drain's
        ``_drive_one`` Task is the single asyncio.Task running the
        attempt — ``handle.run`` directly awaits ``_run_task`` without
        wrapping.  Producer sites therefore never spawn an independent
        per-attempt Task that bypasses the drain (plan-literal
        invariant).

        ``_done_future`` is the canonical terminal-state signal.  It is
        created here and carried on the TaskHandle; ``_run_task``'s
        ``finally`` block sets the result (or exception).  Callers
        await the terminal via ``GatewayHTTPService._wait_for_terminal``
        which returns the same Future — no parallel bookkeeping dict.
        """
        self._remember_task_handle_context(
            record,
            argv=argv,
            tool_cwd=tool_cwd,
            env=env,
            timeout_s=timeout_s,
            profiling=profiling,
            workload_features=workload_features,
            config_fingerprint=config_fingerprint,
            input_fingerprint=input_fingerprint,
            execution_overrides=execution_overrides,
            preferred_worker_addr=preferred_worker_addr,
            preferred_gpu_ids=preferred_gpu_ids,
            output_sample_count=output_sample_count,
            process_name=process_name,
            is_backfill_hint=is_backfill_hint,
        )
        from .planning.scheduling_supervisor import (
            TaskHandle,
        )

        loop = asyncio.get_event_loop()
        existing_handle = self._handles.get(record.task_id)
        existing_future = getattr(existing_handle, "_done_future", None)
        if existing_future is not None and not existing_future.done():
            try:
                same_loop = existing_future.get_loop() is loop
            except Exception:
                same_loop = False
            done_future = existing_future if same_loop else loop.create_future()
        else:
            done_future = loop.create_future()

        handle = TaskHandle(
            task_id=record.task_id,
            arrival_time=getattr(record, "arrival_time", time.time()),
            is_backfill=bool(is_backfill_hint),
            component=str(record.component or "").strip(),
            config_fingerprint=str(config_fingerprint or "").strip(),
            has_progress_hint=False,
            _done_future=done_future,
        )

        async def _run():
            if handle._cancel_requested:
                if not done_future.done():
                    done_future.cancel()
                raise asyncio.CancelledError(
                    f"handle cancelled before drain pick-up: {record.task_id}"
                )
            return await self._run_task(
                handle,
                record,
                argv=argv,
                tool_cwd=tool_cwd,
                env=env,
                timeout_s=timeout_s,
                profiling=profiling,
                workload_features=workload_features,
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                execution_overrides=execution_overrides,
                preferred_worker_addr=preferred_worker_addr,
                preferred_gpu_ids=preferred_gpu_ids,
                output_sample_count=output_sample_count,
                process_name=process_name,
            )

        handle.run = _run
        self._handles[record.task_id] = handle
        return handle

    def _remember_task_handle_context(
        self,
        record: TaskRecord,
        *,
        argv: list,
        tool_cwd,
        env,
        timeout_s,
        profiling,
        workload_features=None,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        execution_overrides=None,
        preferred_worker_addr=None,
        preferred_gpu_ids=None,
        output_sample_count=None,
        process_name=None,
        is_backfill_hint: bool = False,
    ) -> None:
        """Store the latest handle construction inputs for liveness repair."""
        record._handle_context = {
            "argv": list(argv or []),
            "tool_cwd": tool_cwd,
            "env": dict(env or {}),
            "timeout_s": timeout_s,
            "profiling": dict(profiling or {}),
            "workload_features": (
                dict(workload_features or {})
                if isinstance(workload_features, Mapping)
                else workload_features
            ),
            "config_fingerprint": str(config_fingerprint or ""),
            "input_fingerprint": str(input_fingerprint or ""),
            "execution_overrides": (
                dict(execution_overrides or {})
                if isinstance(execution_overrides, Mapping)
                else execution_overrides
            ),
            "preferred_worker_addr": preferred_worker_addr,
            "preferred_gpu_ids": (
                list(preferred_gpu_ids or []) if preferred_gpu_ids is not None else None
            ),
            "output_sample_count": output_sample_count,
            "process_name": process_name,
            "is_backfill_hint": bool(is_backfill_hint),
        }

    def _rebuild_task_handle_from_record(self, record: TaskRecord):
        """Recreate a missing TaskHandle from the last known submit context."""
        context = getattr(record, "_handle_context", None)
        if not isinstance(context, Mapping):
            return None
        argv = context.get("argv")
        if not isinstance(argv, list) or not argv:
            return None
        try:
            return self._build_task_handle(
                record,
                argv=list(argv),
                tool_cwd=context.get("tool_cwd"),
                env=dict(context.get("env") or {}),
                timeout_s=context.get("timeout_s", 300),
                profiling=dict(context.get("profiling") or {}),
                workload_features=context.get("workload_features"),
                config_fingerprint=str(context.get("config_fingerprint") or ""),
                input_fingerprint=str(context.get("input_fingerprint") or ""),
                execution_overrides=context.get("execution_overrides"),
                preferred_worker_addr=context.get("preferred_worker_addr"),
                preferred_gpu_ids=context.get("preferred_gpu_ids"),
                output_sample_count=context.get("output_sample_count"),
                process_name=context.get("process_name"),
                is_backfill_hint=bool(context.get("is_backfill_hint", False)),
            )
        except Exception as exc:
            _LOG.error(
                "[cancel_task] failed to rebuild TaskHandle task=%s: %s",
                getattr(record, "task_id", "<unknown>"),
                exc,
                exc_info=True,
            )
            return None

    def _rollback_batch_plan_predictions(
        self,
        task_ids: Sequence[str],
        *,
        reason: str,
        include_dispatching: bool = False,
    ) -> int:
        """Drop predicted entries for batch plans that will not launch.

        Detached batch solve keeps speculative entries out of the live
        timeline, but legacy/fallback planners and validation failure paths
        may still leave predicted entries behind.  Parked handles are not
        dispatching, so the default rollback removes only pure predictions.
        Validation failures are different: the validator may have marked the
        prediction dispatching before discovering a real runtime gap, and
        that failed attempt must also be removed or PBBC/self-concurrency
        gates count phantom work and can block later wakes.
        """
        timelines = getattr(
            getattr(getattr(self, "_planner", None), "campaign_scheduler", None),
            "_timelines",
            None,
        )
        remove_fn = getattr(timelines, "remove_predicted_entries_for_task", None)
        if not callable(remove_fn):
            return 0

        removed_total = 0
        for task_id in task_ids:
            task_key = str(task_id or "")
            if not task_key:
                continue
            try:
                removed = cast(Callable[..., Any], remove_fn)(
                    task_key,
                    include_dispatching=bool(include_dispatching),
                )
            except TypeError:
                removed = cast(Callable[..., Any], remove_fn)(task_key)
            with contextlib.suppress(TypeError, ValueError):
                removed_total += int(removed or 0)
        if removed_total:
            _LOG.info(
                "[batch-plan] rolled back %d predicted entries (%s)",
                removed_total,
                reason,
            )
        return removed_total

    async def _solve_batch_admissible_detached(
        self,
        global_planner: Any,
        tasks: list[dict[str, Any]],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Run a batch solve without committing speculative predictions."""
        _cp_start = self._cp_begin()
        try:
            solve_result = global_planner.solve_admissible(
                tasks,
                commit_predictions=False,
            )
            if hasattr(solve_result, "__await__"):
                result = await solve_result
            else:
                result = solve_result
        except TypeError as exc:
            if "commit_predictions" not in str(exc):
                raise
            solve_result = global_planner.solve_admissible(tasks)
            if hasattr(solve_result, "__await__"):
                result = await solve_result
            else:
                result = solve_result
        finally:
            self._cp_record(
                "planner_solve",
                _cp_start,
                active_wall=True,
                mode="batch",
                task_count=len(tasks),
            )
        for plan in result[0]:
            self._cp_record_reciprocal_scoring(plan)
        return result

    def _commit_batch_plan_prediction(
        self,
        plan: Any,
        item: Mapping[str, Any],
        *,
        dispatching: bool = True,
    ) -> bool:
        """Commit one batch plan, optionally as a pure parked prediction."""
        global_planner = getattr(self._planner, "_global_planner", None)
        commit = getattr(global_planner, "commit_dispatch_plan_prediction", None)
        if not callable(commit):
            return False
        commit(
            plan,
            input_size=float(item.get("input_size", 0.0) or 0.0),
        )
        if not dispatching:
            timelines = getattr(
                getattr(self._planner, "campaign_scheduler", None),
                "_timelines",
                None,
            )
            get_prediction = getattr(timelines, "get_predicted_entry", None)
            entry = (
                cast(Any, get_prediction(str(getattr(plan, "task_id", "") or "")))
                if callable(get_prediction)
                else None
            )
            if entry is None:
                return False
            entry.is_dispatching = False
        return True

    @staticmethod
    def _attach_batch_plan_to_handle(
        handle: Any,
        plan: Any,
        item: Mapping[str, Any],
    ) -> None:
        if hasattr(handle, "_batch_planning_retry_deferral"):
            try:
                delattr(handle, "_batch_planning_retry_deferral")
            except Exception:
                handle._batch_planning_retry_deferral = None
        handle._precomputed_dispatch_plan = plan
        handle._precomputed_campaign_hints = dict(item.get("campaign_hints") or {})
        handle._precomputed_input_size = float(item.get("input_size", 0.0) or 0.0)
        handle.is_backfill = bool(
            handle._precomputed_campaign_hints.get("is_backfill"),
        )

    def _detach_precomputed_plan_from_handle(
        self,
        handle: Any,
        *,
        reason: str,
    ) -> int:
        if handle is None:
            return 0
        plan = getattr(handle, "_precomputed_dispatch_plan", None)
        task_id = str(
            getattr(plan, "task_id", "") or getattr(handle, "task_id", "") or ""
        )
        removed = 0
        if task_id:
            removed = self._rollback_batch_plan_predictions(
                [task_id],
                reason=reason,
                include_dispatching=True,
            )
            self._release_front_slot_lease_for_task(task_id)
        handle._precomputed_dispatch_plan = None
        handle._precomputed_campaign_hints = {}
        handle._precomputed_input_size = 0.0
        return removed

    def _commit_and_attach_batch_plan(
        self,
        handle: Any,
        plan: Any,
        item: Mapping[str, Any],
        *,
        front_parked: bool = False,
    ) -> bool:
        task_id = str(getattr(plan, "task_id", "") or "")
        committed = self._commit_batch_plan_prediction(
            plan,
            item,
            dispatching=not front_parked,
        )
        if front_parked and not committed:
            return False
        try:
            self._attach_batch_plan_to_handle(handle, plan, item)
        except BaseException:
            self._rollback_batch_plan_predictions(
                [task_id],
                reason="handle_attach_failure",
                include_dispatching=True,
            )
            self._release_front_slot_lease_for_task(task_id)
            raise
        return True

    def _batch_plan_front_slots_for_worker(
        self,
        plan: Any,
    ) -> tuple[int | None, Any | None, bool]:
        """Return front-slot capacity for a batch-planned worker.

        ``None`` means unbounded/unknown and should fail open.  The boolean
        indicates a warm-plan readiness mismatch that should become a short
        worker_not_ready retry instead of launching into validation just to
        roll back there.
        """
        cap = int(getattr(self, "dispatch_backlog_per_worker", 1) or 0)
        if cap <= 0:
            return None, None, False
        worker_name = str(getattr(plan, "target_worker_name", "") or "").strip()
        worker_addr = str(getattr(plan, "target_worker_addr", "") or "").strip()
        if not worker_name and not worker_addr:
            return None, None, False

        sup = getattr(self, "supervisor", None)
        if sup is None:
            return None, None, False
        reconcile = getattr(sup, "reconcile_dispatch_front_capacity", None)
        if callable(reconcile) and worker_name:
            try:
                reconcile(worker_name)
            except Exception:
                _LOG.debug(
                    "[batch-plan] front reconciliation failed worker=%s",
                    worker_name,
                    exc_info=True,
                )

        st = self._lookup_worker_state_for_dispatch(
            worker_name=worker_name,
            worker_addr=worker_addr,
        )
        needs_cold_start = bool(getattr(plan, "needs_cold_start", False))
        if st is None:
            if needs_cold_start:
                leased = self._front_slot_lease_count(
                    worker_name=worker_name,
                    worker_addr=worker_addr,
                )
                return max(0, cap - leased), None, False
            return 0, None, True

        is_ready = bool(getattr(st, "ready", False)) and bool(
            str(getattr(st, "addr", "") or "").strip()
        )
        if not needs_cold_start and not is_ready:
            return 0, st, True
        leased = self._front_slot_lease_count(
            worker_name=worker_name,
            worker_addr=worker_addr,
            worker_state=st,
        )
        slots = max(0, cap - self._dispatch_front_backlog(st) - leased)
        return slots, st, False

    def _batch_plan_retry_deferral(
        self,
        *,
        reason: str,
        handle: Any,
        item: Mapping[str, Any],
        plan: Any,
        active_count: int = 0,
    ) -> Any:
        from .planning.scheduling_supervisor import RetryDeferral

        worker_name = str(getattr(plan, "target_worker_name", "") or "").strip()
        component = str(
            (item.get("task") or {}).get("component")
            or getattr(handle, "component", "")
            or "",
        ).strip()
        config_fp = str(getattr(handle, "config_fingerprint", "") or "").strip()
        gpu_id = str(getattr(plan, "target_gpu_id", "") or "").strip()
        not_before_at = 0.0
        if reason == "worker_not_ready":
            not_before_at = time.time() + self.WORKER_NOT_READY_RETRY_DELAY_SEC
        return RetryDeferral(
            reason=reason,
            worker_name=worker_name,
            component=component,
            config_fingerprint=config_fp,
            gpu_id=gpu_id,
            not_before_at=not_before_at,
            active_count=int(active_count or 0),
        )

    def _batch_plan_admit_front_slot(
        self,
        *,
        handle: Any,
        item: Mapping[str, Any],
        plan: Any,
        front_budget: dict[str, int | None],
    ) -> bool:
        task_id = str(getattr(handle, "task_id", "") or "")
        worker_name = str(getattr(plan, "target_worker_name", "") or "").strip()
        worker_key = worker_name or str(getattr(plan, "target_worker_addr", "") or "")
        task_lease_count = self._front_slot_lease_count_for_plan(
            plan,
            exclude_task_id="",
        ) - self._front_slot_lease_count_for_plan(
            plan,
            exclude_task_id=task_id,
        )
        if not worker_key:
            return True
        if worker_key not in front_budget:
            slots, _st, not_ready = self._batch_plan_front_slots_for_worker(plan)
            if not_ready:
                self._cp_count("batch_front_slot_reject", 1, reason="worker_not_ready")
                handle._batch_planning_retry_deferral = self._batch_plan_retry_deferral(
                    reason="worker_not_ready", handle=handle, item=item, plan=plan
                )
                return False
            front_budget[worker_key] = slots
        if task_lease_count > 0:
            return True
        remaining = front_budget.get(worker_key)
        if remaining is None:
            return True
        if remaining <= 0:
            self._cp_count("batch_front_slot_reject", 1, reason="worker_front")
            handle._batch_planning_retry_deferral = self._batch_plan_retry_deferral(
                reason="worker_front",
                handle=handle,
                item=item,
                plan=plan,
                active_count=1,
            )
            return False
        if not self._try_claim_front_slot_lease(
            task_id=task_id,
            plan=plan,
        ):
            self._cp_count("batch_front_slot_reject", 1, reason="lease_busy")
            handle._batch_planning_retry_deferral = self._batch_plan_retry_deferral(
                reason="worker_front",
                handle=handle,
                item=item,
                plan=plan,
                active_count=1,
            )
            return False
        front_budget[worker_key] = remaining - 1
        return True

    def _partition_front_retained_plans(
        self,
        prepared: list[tuple[Any, dict[str, Any]]],
    ) -> tuple[list[tuple[Any, dict[str, Any], Any]], list[tuple[Any, dict[str, Any]]]]:
        timelines = getattr(
            getattr(self._planner, "campaign_scheduler", None),
            "_timelines",
            None,
        )
        get_prediction = getattr(timelines, "get_predicted_entry", None)
        retained: list[tuple[Any, dict[str, Any], Any]] = []
        fresh: list[tuple[Any, dict[str, Any]]] = []
        for handle, item in prepared:
            task_id = str(getattr(handle, "task_id", "") or "")
            plan = getattr(handle, "_precomputed_dispatch_plan", None)
            entry = (
                cast(Any, get_prediction(task_id))
                if plan is not None and callable(get_prediction)
                else None
            )
            plan_matches = (
                entry is not None
                and str(getattr(plan, "task_id", "") or "") == task_id
                and str(getattr(plan, "target_gpu_id", "") or "")
                == str(getattr(entry, "gpu_id", "") or "")
                and str(getattr(plan, "target_worker_name", "") or "")
                == str(getattr(entry, "worker_name", "") or "")
            )
            entry_is_current = plan_matches and not any(
                bool(getattr(entry, name, False))
                for name in (
                    "is_dispatching",
                    "is_completed",
                    "prediction_stale",
                    "is_invalidated",
                    "is_evict_masked",
                )
            )
            if entry_is_current:
                retained.append((handle, item, plan))
                continue
            if plan is not None:
                self._detach_precomputed_plan_from_handle(
                    handle,
                    reason="front_retained_stale",
                )
            fresh.append((handle, item))

        return retained, fresh

    def _reuse_front_retained_plans(
        self,
        retained: list[tuple[Any, dict[str, Any], Any]],
        front_budget: dict[str, int | None],
    ) -> tuple[list[Any], list[Any]]:
        timelines = getattr(
            getattr(self._planner, "campaign_scheduler", None),
            "_timelines",
            None,
        )
        mark_dispatching = getattr(
            timelines,
            "mark_predicted_entry_dispatching",
            None,
        )
        launch: list[Any] = []
        defer: list[Any] = []
        for handle, item, plan in retained:
            if self._batch_plan_admit_front_slot(
                handle=handle,
                item=item,
                plan=plan,
                front_budget=front_budget,
            ):
                if callable(mark_dispatching):
                    mark_dispatching(
                        str(getattr(handle, "task_id", "") or ""),
                        component=str(getattr(plan, "component", "") or ""),
                        gpu_id=str(getattr(plan, "target_gpu_id", "") or ""),
                    )
                launch.append(handle)
                self._cp_count("batch_front_plan_reused", 1)
                continue
            deferral = getattr(handle, "_batch_planning_retry_deferral", None)
            if str(getattr(deferral, "reason", "") or "") != "worker_front":
                self._detach_precomputed_plan_from_handle(
                    handle,
                    reason="front_retained_no_longer_waiting",
                )
            defer.append(handle)
        return launch, defer

    def _retain_front_deferred_plan(
        self,
        handle: Any,
        item: Mapping[str, Any],
        plan: Any,
        *,
        prediction_already_committed: bool = False,
    ) -> bool:
        deferral = getattr(handle, "_batch_planning_retry_deferral", None)
        if str(getattr(deferral, "reason", "") or "") != "worker_front":
            return False
        if prediction_already_committed:
            timelines = getattr(
                getattr(self._planner, "campaign_scheduler", None),
                "_timelines",
                None,
            )
            park = getattr(timelines, "park_predicted_entry", None)
            entry = (
                cast(Any, park)(
                    str(getattr(plan, "task_id", "") or ""),
                    component=str(getattr(plan, "component", "") or ""),
                    gpu_id=str(getattr(plan, "target_gpu_id", "") or ""),
                    worker_name=str(getattr(plan, "target_worker_name", "") or ""),
                )
                if callable(park)
                else None
            )
            if entry is None:
                return False
            self._attach_batch_plan_to_handle(handle, plan, item)
        elif not self._commit_and_attach_batch_plan(
            handle,
            plan,
            item,
            front_parked=True,
        ):
            return False
        handle._batch_planning_retry_deferral = deferral
        self._cp_count("batch_front_plan_retained", 1)
        return True

    async def _batch_plan_ready_handles(self, handles: list) -> tuple[list, list]:
        """Plan one supervisor wake as a batch and attach plans to handles."""
        if not handles:
            return handles, []
        _cp_total_start = self._cp_begin()
        _global_planner = getattr(self._planner, "_global_planner", None)
        if _global_planner is None or not hasattr(_global_planner, "solve_admissible"):
            self._cp_record(
                "batch_planning_total",
                _cp_total_start,
                ready_count=len(handles),
                outcome="fallback_no_global_planner",
            )
            return handles, []

        prepared: list[tuple[Any, dict[str, Any]]] = []
        fallback: list[Any] = []
        _cp_prepare_start = self._cp_begin()
        for handle in handles:
            item = self._prepare_batch_planning_item(handle)
            if item is None:
                fallback.append(handle)
            else:
                prepared.append((handle, item))
        self._cp_record(
            "batch_planning_prepare",
            _cp_prepare_start,
            active_wall=True,
            ready_count=len(handles),
            prepared_count=len(prepared),
            fallback_count=len(fallback),
        )
        if not prepared:
            self._cp_record(
                "batch_planning_total",
                _cp_total_start,
                ready_count=len(handles),
                outcome="fallback_no_prepared",
            )
            return handles, []

        retained, fresh = self._partition_front_retained_plans(prepared)
        launch: list[Any] = list(fallback)
        component_exhausted: list[Any] = []
        defer_map: dict[int, Any] = {}
        remaining: list[tuple[Any, dict[str, Any]]] = fresh
        plans: list[Any] = []
        round_count = 0
        front_budget: dict[str, int | None] = {}
        ready_slack_s = float(
            (getattr(self, "_runtime_config", {}) or {}).get(
                "batch_plan_ready_slack_s",
                0.25,
            )
        )
        retained_launch, retained_defer = self._reuse_front_retained_plans(
            retained,
            front_budget,
        )
        launch.extend(retained_launch)
        defer_map.update((id(handle), handle) for handle in retained_defer)

        while remaining:
            round_count += 1
            try:
                self._planner.prepare_for_planning()
                tasks = [dict(item["task"]) for _, item in remaining]
                plans, skipped = await self._solve_batch_admissible_detached(
                    _global_planner,
                    tasks,
                )
            except Exception:
                fallback_remaining = [handle for handle, _item in remaining]
                for handle in fallback_remaining:
                    self._detach_precomputed_plan_from_handle(
                        handle,
                        reason="solve_exception",
                    )
                launch.extend(component_exhausted)
                launch.extend(fallback_remaining)
                defer = list(defer_map.values())
                _LOG.warning(
                    "[batch-plan] solve_admissible failed for fresh work; "
                    "preserving retained plans and falling back to per-handle planning",
                    exc_info=True,
                )
                self._cp_count("batch_planning_exception", 1, phase="solve")
                self._cp_record(
                    "batch_planning_total",
                    _cp_total_start,
                    ready_count=len(handles),
                    launch_count=len(launch),
                    defer_count=len(defer),
                    outcome="solve_exception",
                    rounds=round_count,
                )
                return launch, defer

            plan_by_task_id = {str(plan.task_id): plan for plan in plans}
            next_remaining: list[tuple[Any, dict[str, Any]]] = []
            future_start_pending: list[tuple[Any, dict[str, Any], Any, float]] = []
            launched_this_round = 0
            now = time.time()
            _cp_handoff_start = self._cp_begin()
            for handle, item in remaining:
                task_id = str(getattr(handle, "task_id", "") or "")
                plan = plan_by_task_id.get(task_id)
                if plan is None:
                    self._rollback_batch_plan_predictions(
                        [task_id],
                        reason="not_launchable",
                    )
                    exc = skipped.get(task_id)
                    violation = getattr(exc, "violation", None)
                    if (
                        violation is not None
                        and getattr(violation, "violation_type", "")
                        == "component_exhausted"
                    ):
                        component_exhausted.append(handle)
                    else:
                        defer_map[id(handle)] = handle
                        self._cp_count(
                            "batch_deferral",
                            1,
                            reason=str(
                                getattr(violation, "violation_type", "not_launchable")
                                if violation
                                else "not_launchable"
                            ),
                        )
                    continue
                planned_start = float(getattr(plan, "planned_start_time", 0.0) or 0.0)
                needs_cold_start = bool(getattr(plan, "needs_cold_start", False))
                if not needs_cold_start and planned_start > now + max(
                    0.0, ready_slack_s
                ):
                    future_start_pending.append(
                        (
                            handle,
                            item,
                            plan,
                            planned_start,
                        )
                    )
                    continue
                if not self._batch_plan_admit_front_slot(
                    handle=handle,
                    item=item,
                    plan=plan,
                    front_budget=front_budget,
                ):
                    if not self._retain_front_deferred_plan(handle, item, plan):
                        self._rollback_batch_plan_predictions(
                            [task_id],
                            reason="front_budget",
                        )
                    defer_map[id(handle)] = handle
                    self._cp_count("batch_deferral", 1, reason="front_budget")
                    continue
                self._commit_and_attach_batch_plan(handle, plan, item)
                launch.append(handle)
                launched_this_round += 1

            if future_start_pending:
                timelines = None
                try:
                    timelines = getattr(
                        getattr(self._planner, "campaign_scheduler", None),
                        "_timelines",
                        None,
                    )
                except Exception:
                    timelines = None

                launch_target_gpus: set[str] = set()
                for launched_handle in launch:
                    launched_plan = getattr(
                        launched_handle,
                        "_precomputed_dispatch_plan",
                        None,
                    )
                    launched_gpu = str(
                        getattr(launched_plan, "target_gpu_id", "") or "",
                    )
                    if launched_gpu:
                        launch_target_gpus.add(launched_gpu)

                def _target_gpu_has_runtime_occupancy(
                    plan: Any,
                    timelines: Any = timelines,
                    launch_target_gpus: set[str] = launch_target_gpus,
                ) -> bool:
                    if timelines is None:
                        return True
                    gpu_id = str(getattr(plan, "target_gpu_id", "") or "")
                    if not gpu_id:
                        return True
                    if gpu_id in launch_target_gpus:
                        return True
                    try:
                        tl = timelines.get(gpu_id)
                    except Exception:
                        return True
                    if tl is None:
                        return False
                    for entry in getattr(tl, "active_entries", []) or []:
                        if getattr(entry, "is_invalidated", False):
                            continue
                        if getattr(entry, "is_evict_masked", False):
                            continue
                        if getattr(entry, "is_completed", False):
                            continue
                        if bool(getattr(entry, "is_predicted", False)) and not bool(
                            getattr(entry, "is_dispatching", False)
                        ):
                            continue
                        return True
                    return False

                earliest_idle_start_by_gpu: dict[str, float] = {}
                for _handle, _item, plan, planned_start in future_start_pending:
                    gpu_id = str(getattr(plan, "target_gpu_id", "") or "")
                    if not gpu_id:
                        continue
                    if _target_gpu_has_runtime_occupancy(plan):
                        continue
                    previous = earliest_idle_start_by_gpu.get(gpu_id)
                    if previous is None or planned_start < previous:
                        earliest_idle_start_by_gpu[gpu_id] = planned_start

                if earliest_idle_start_by_gpu:
                    shifted = 0
                    rollback_future_task_ids: list[str] = []
                    for handle, item, plan, planned_start in future_start_pending:
                        gpu_id = str(getattr(plan, "target_gpu_id", "") or "")
                        earliest_start = earliest_idle_start_by_gpu.get(gpu_id)
                        if (
                            earliest_start is not None
                            and planned_start
                            <= earliest_start + max(0.0, ready_slack_s)
                        ):
                            if not self._batch_plan_admit_front_slot(
                                handle=handle,
                                item=item,
                                plan=plan,
                                front_budget=front_budget,
                            ):
                                if not self._retain_front_deferred_plan(
                                    handle,
                                    item,
                                    plan,
                                ):
                                    rollback_future_task_ids.append(
                                        str(getattr(handle, "task_id", "") or ""),
                                    )
                                defer_map[id(handle)] = handle
                                self._cp_count(
                                    "batch_deferral", 1, reason="front_budget_future"
                                )
                                continue
                            self._commit_and_attach_batch_plan(handle, plan, item)
                            launch.append(handle)
                            launched_this_round += 1
                            shifted += 1
                        else:
                            rollback_future_task_ids.append(
                                str(getattr(handle, "task_id", "") or ""),
                            )
                            next_remaining.append((handle, item))
                    if rollback_future_task_ids:
                        self._rollback_batch_plan_predictions(
                            rollback_future_task_ids,
                            reason="future_start",
                        )
                    if shifted:
                        max_delta = max(
                            max(0.0, start - now)
                            for start in earliest_idle_start_by_gpu.values()
                        )
                        _LOG.info(
                            "[batch-plan] shifted idle-gpu future wave "
                            "launch=%d gpu_count=%d max_start_delta=%.3fs",
                            shifted,
                            len(earliest_idle_start_by_gpu),
                            max_delta,
                        )
                else:
                    self._rollback_batch_plan_predictions(
                        [
                            str(getattr(handle, "task_id", "") or "")
                            for handle, _item, _plan, _planned_start in future_start_pending
                        ],
                        reason="future_start",
                    )
                    next_remaining.extend(
                        (handle, item)
                        for handle, item, _plan, _planned_start in future_start_pending
                    )

            self._cp_record(
                "batch_planning_handoff_filter",
                _cp_handoff_start,
                active_wall=True,
                round=round_count,
                remaining_count=len(remaining),
                launched_this_round=launched_this_round,
                future_start_count=len(future_start_pending),
                deferred_count=len(defer_map),
            )

            if not next_remaining:
                break
            if launched_this_round <= 0:
                for handle, _ in next_remaining:
                    defer_map[id(handle)] = handle
                break
            remaining = next_remaining

        defer = list(defer_map.values())

        launch.extend(component_exhausted)
        if plans or defer:
            _LOG.debug(
                "[batch-plan] ready=%d launch=%d deferred=%d fallback=%d rounds=%d",
                len(handles),
                len(launch),
                len(defer),
                len(fallback) + len(component_exhausted),
                round_count,
            )
        self._cp_count("batch_launch", len(launch))
        self._cp_count("batch_defer", len(defer))
        self._cp_record(
            "batch_planning_total",
            _cp_total_start,
            ready_count=len(handles),
            launch_count=len(launch),
            defer_count=len(defer),
            fallback_count=len(fallback) + len(component_exhausted),
            rounds=round_count,
            outcome="ok",
        )
        return launch, defer

    def _prepare_batch_planning_item(self, handle: Any) -> dict[str, Any] | None:
        task_id = str(getattr(handle, "task_id", "") or "").strip()
        if not task_id:
            return None
        record = self._tasks.get(task_id)
        if record is None:
            return None
        done = getattr(handle, "_done_future", None)
        if done is not None and done.done():
            return None
        if getattr(handle, "_cancel_requested", False):
            return None

        context = getattr(record, "_handle_context", None)
        if not isinstance(context, Mapping):
            return None
        argv = context.get("argv")
        if not isinstance(argv, list) or not argv:
            return None

        workload_features = context.get("workload_features")
        normalized_workload_features = canonicalize_axes(
            dict(workload_features or {})
            if isinstance(workload_features, Mapping)
            else {}
        )
        payload: dict[str, Any] = {
            "mode": "nextflow_task",
            "workdir": _host_to_container_path(record.workdir),
            "argv": list(argv),
            "env": dict(context.get("env") or {}),
        }
        tool_cwd = context.get("tool_cwd")
        profiling = context.get("profiling")
        if tool_cwd:
            payload["tool_cwd"] = tool_cwd
        if profiling:
            payload["profiling"] = profiling

        extracted_length, extracted_config = self._extract_component_features_cached(
            record,
            payload,
        )
        if extracted_length:
            merged = {**extracted_length, **normalized_workload_features}
            normalized_workload_features = canonicalize_axes(merged)
            record.workload_features = dict(normalized_workload_features)
            if isinstance(record._handle_context, dict):
                record._handle_context["workload_features"] = dict(
                    normalized_workload_features,
                )
        config_fingerprint = str(context.get("config_fingerprint") or "")
        if extracted_config and not config_fingerprint:
            config_fingerprint = "|".join(
                f"{k}={v}" for k, v in sorted(extracted_config.items())
            )
            record.config_fingerprint = config_fingerprint
            if isinstance(record._handle_context, dict):
                record._handle_context["config_fingerprint"] = config_fingerprint
            handle.config_fingerprint = config_fingerprint

        logical_batch_size = _logical_batch_size(
            record.component,
            payload,
            dict(normalized_workload_features),
        )

        try:
            from .signals.resource_profile import ResourceProfileRegistry

            _submit_features = dict(normalized_workload_features or {})
            if not _submit_features and record.workload_features:
                _submit_features = dict(record.workload_features)
            input_size = ResourceProfileRegistry.extract_input_size(
                _submit_features,
                component=str(record.component or "").strip().lower(),
            )
        except Exception:
            input_size = 0.0

        task_campaign_id = str(record.campaign_id or "").strip()
        if not task_campaign_id or not record.component:
            return None

        try:
            self._planner._connect_supervisor_refs()
            cs = self._planner.campaign_scheduler
            if not bool(getattr(record, "_campaign_submit_registered", False)):
                placement = cs.on_task_submit(
                    task_id=record.task_id,
                    campaign_id=task_campaign_id,
                    component=record.component,
                    input_size=float(input_size or 0.0),
                )
                campaign_hints = dict(placement.scheduling_hints)
                record._campaign_submit_registered = True
            else:
                cq = cs._get_or_create(task_campaign_id)
                is_primary = cs._is_primary(task_campaign_id)
                is_backfill = not is_primary and len(cs._ordered_campaigns()) > 1
                campaign_hints = {
                    "is_backfill": is_backfill,
                    "_campaign_arrival": cq.arrival_time,
                }
        except Exception:
            _LOG.debug(
                "[batch-plan] prepare failed for task=%s",
                task_id,
                exc_info=True,
            )
            return None

        handle.is_backfill = bool(campaign_hints.get("is_backfill"))
        return {
            "task": {
                "task_id": record.task_id,
                "campaign_id": task_campaign_id,
                "component": record.component,
                "input_size": float(input_size or 0.0),
                "logical_batch_size": int(logical_batch_size or 0),
                "execution_overrides": dict(context.get("execution_overrides") or {}),
                "config_fingerprint": config_fingerprint,
                "input_fingerprint": str(record.input_fingerprint or ""),
                "created_at": float(record.created_at or 0.0),
                "nf_task_id": str(record.nf_task_id or ""),
                "active_cancel_safe": self._supports_active_cancel(record.component),
                "is_backfill": bool(campaign_hints.get("is_backfill")),
            },
            "campaign_hints": campaign_hints,
            "input_size": float(input_size or 0.0),
        }

    async def _handle_cancel_reenqueue(
        self,
        record: TaskRecord,
        reason: str,
        dispatch_gpu_id: str,
        task_campaign_id: str,
        argv: list,
        tool_cwd,
        env,
        timeout_s,
        profiling,
        workload_features,
        config_fingerprint,
        input_fingerprint,
        execution_overrides,
        output_sample_count,
        process_name,
        *,
        is_backfill_hint: bool = False,
    ) -> bool:
        """Common handler for cancel re-enqueue (backfill eviction + cooperative cancel).

        Kills the timeline entry, completes the dispatch, and re-enqueues
        via a new _run_task coroutine.  Returns True (sets _backfill_reenqueued).
        """
        record._cancel_reason = ""
        _LOG.info("[task %s] cancel (%s) — re-enqueueing", record.task_id, reason)
        _evict_gpu = dispatch_gpu_id or getattr(record, "_dispatched_gpu_id", "")
        try:
            tl_entry = self._planner.campaign_scheduler._timelines.find_entry(
                record.task_id
            )
            if tl_entry:
                tl = self._planner.campaign_scheduler._timelines.get(tl_entry.gpu_id)
                if tl:
                    tl.kill(record.task_id)
            _evict_gpu = _evict_gpu or (tl_entry.gpu_id if tl_entry else "")
            self._complete_campaign_task(
                task_id=record.task_id,
                campaign_id=task_campaign_id or "",
                component=record.component,
                gpu_id=_evict_gpu,
                execution_attempt_id=str(
                    getattr(record, "_reciprocal_event_id", "") or ""
                ),
                from_eviction=True,
            )
        except Exception:
            _LOG.warning(
                "[task %s] cancel re-enqueue cleanup failed",
                record.task_id,
                exc_info=True,
            )
        await self._mark_record_pending_for_scheduler(
            record,
            message="waiting for scheduler wake",
        )
        handle = self._build_task_handle(
            record,
            argv=list(argv),
            tool_cwd=tool_cwd,
            env=env,
            timeout_s=timeout_s,
            profiling=profiling,
            workload_features=workload_features,
            config_fingerprint=config_fingerprint,
            input_fingerprint=input_fingerprint,
            execution_overrides=execution_overrides,
            output_sample_count=output_sample_count,
            process_name=process_name,
            is_backfill_hint=is_backfill_hint,
        )
        if self._scheduling_supervisor is None:
            raise RuntimeError(
                "SchedulingSupervisor invariant violation: eviction_reenqueue "
                "from within an active Core Loop attempt but supervisor is None",
            )
        action = self._submit_retry_handle_to_supervisor(
            handle,
            reason="eviction_reenqueue",
        )
        if not self._retry_supervisor_action_is_live(action):
            _LOG.warning(
                "[task %s] eviction reenqueue handle did not become live (action=%s)",
                record.task_id,
                action,
            )
            return False
        return True

    def _validation_fingerprint(
        self,
        plan: Any,
        *,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
    ) -> tuple[Any, ...]:
        """Snapshot of the validator's deterministic inputs for attempt dedup.

        Covers the pre-activation admission path (worker resolve, reservation,
        logical VRAM/RAM gate): plan identity, logical timeline generation, and
        the target worker's live state including the dispatch_pending front
        counter.  Only ``vram_insufficient`` / ``host_ram_saturated`` violations
        are memoized, so physical SMI/activation state is never part of a cached
        decision.
        """
        global_planner = getattr(self, "_global_planner", None) or getattr(
            getattr(self, "_planner", None), "_global_planner", None
        )
        cs = getattr(global_planner, "campaign_scheduler", None)
        timelines = getattr(cs, "_timelines", None)
        generation = timelines._scenario_generation() if timelines is not None else None
        worker_name = str(getattr(plan, "target_worker_name", "") or "")
        worker_state = None
        sup = getattr(self, "supervisor", None)
        if sup and worker_name:
            worker_state = getattr(sup, "states", {}).get(worker_name)
        if worker_state is not None:
            worker_snapshot: Any = (
                getattr(worker_state, "ready", None),
                getattr(worker_state, "addr", None),
                getattr(worker_state, "lifecycle_state", None),
                getattr(worker_state, "memory_reserved_mb", None),
                getattr(worker_state, "dispatch_pending", None),
                tuple(getattr(worker_state, "assigned_gpus", ()) or ()),
            )
        else:
            worker_snapshot = None
        reciprocal_meta = getattr(plan, "worker_metadata", {}) or {}
        try:
            reciprocal_digest = hashlib.sha256(
                json.dumps(reciprocal_meta, sort_keys=True, default=str).encode()
            ).hexdigest()
        except (TypeError, ValueError):
            reciprocal_digest = None
        return (
            str(getattr(plan, "task_id", "")),
            str(getattr(plan, "component", "")),
            str(getattr(plan, "target_gpu_id", "")),
            worker_name,
            str(config_fingerprint or ""),
            str(input_fingerprint or ""),
            getattr(plan, "vram_budget_mb", 0),
            getattr(plan, "ram_budget_mb", 0),
            getattr(plan, "planned_start_time", 0.0),
            getattr(plan, "predicted_latency_sec", 0.0),
            getattr(plan, "needs_cold_start", False),
            reciprocal_digest,
            generation,
            worker_snapshot,
        )

    async def _run_task(
        self,
        handle: Any,
        record: TaskRecord,
        *,
        argv: list[str],
        tool_cwd: str | None,
        env: dict[str, str],
        timeout_s: int,
        profiling: dict[str, Any],
        workload_features: Mapping[str, Any] | None = None,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        execution_overrides: Mapping[str, Any] | None = None,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        output_sample_count: int | None = None,
        process_name: str | None = None,
    ) -> None:
        _backfill_reenqueued = False
        _returned_skip = False
        _core_loop_owner = False
        _core_loop_owner_task: asyncio.Task[Any] | None = None
        _suppress_terminal_completion = False
        _launch_prediction_owned = False
        telemetry_summary: dict[str, Any] = {
            "mean_gpu_util_percent": None,
            "peak_memory_mib": None,
            "active_memory_mib": None,
            "memory_basis": "",
            "peak_fidelity": "",
            "active_memory_scope": "",
            "active_memory_measurement": "",
            "vram_memory_attribution": "",
            "resident_memory_mib": None,
            "resident_memory_source": "",
            "memory_qc_keep": False,
            "concurrent_execute_overlap": False,
            "telemetry_wall_clock_sec": None,
        }
        normalized_workload_features = canonicalize_axes(dict(workload_features or {}))
        normalized_execution_overrides: dict[str, Any] = (
            dict(execution_overrides or {})
            if isinstance(execution_overrides, Mapping)
            else {}
        )
        planner_intent = PlannerIntent()
        addr: str = ""
        task_campaign_id: str | None = None
        dispatch_gpu_id = ""
        try:
            async with self._lock:
                owners = getattr(self, "_core_loop_active_tasks", None)
                if owners is None:
                    owners = {}
                    self._core_loop_active_tasks = owners
                existing_owner = (
                    owners.get(record.task_id) if isinstance(owners, dict) else None
                )
                if existing_owner is not None and existing_owner.done():
                    self._core_loop_active_task_ids.discard(record.task_id)
                    owners.pop(record.task_id, None)
                    existing_owner = None
                if record.task_id in self._core_loop_active_task_ids:
                    _active_handle = self._handles.get(record.task_id)
                    _active_future = getattr(_active_handle, "_done_future", None)
                else:
                    self._core_loop_active_task_ids.add(record.task_id)
                    _current_task = asyncio.current_task()
                    if _current_task is not None and isinstance(owners, dict):
                        owners[record.task_id] = _current_task
                        _core_loop_owner_task = _current_task
                    _core_loop_owner = True
                    _active_handle = None
                    _active_future = None
            if not _core_loop_owner:
                _returned_skip = True
                _LOG.debug(
                    "[task %s] duplicate Core Loop attempt deferred",
                    record.task_id,
                )
                if (
                    _active_future is not None
                    and _active_future is not getattr(handle, "_done_future", None)
                    and not _active_future.done()
                ):
                    with contextlib.suppress(Exception):
                        await _active_future
                from .planning.scheduling_supervisor import RetryDeferral

                return RetryDeferral(
                    reason="core_loop_active",
                    not_before_at=time.time() + 0.25,
                )

            await self._set_state(
                record, TASK_STATE_RUNNING, ok=False, exit_code=0, message=""
            )

            container_workdir = _host_to_container_path(record.workdir)
            payload: dict[str, Any] = {
                "mode": "nextflow_task",
                "workdir": container_workdir,
                "argv": argv,
                "env": env,
            }
            _LOG.debug(
                "[task %s] Core Loop entry for %s (nf_id=%s)",
                record.task_id,
                record.component,
                record.nf_task_id,
            )

            if tool_cwd:
                payload["tool_cwd"] = tool_cwd
            if profiling:
                payload["profiling"] = profiling
            extracted_length, extracted_config = (
                self._extract_component_features_cached(
                    record,
                    payload,
                )
            )
            if extracted_length:
                merged = {**extracted_length, **normalized_workload_features}
                normalized_workload_features = canonicalize_axes(merged)
                record.workload_features = dict(normalized_workload_features)
                if isinstance(record._handle_context, dict):
                    record._handle_context["workload_features"] = dict(
                        normalized_workload_features,
                    )
            logical_batch_size = _logical_batch_size(
                record.component,
                payload,
                dict(normalized_workload_features),
            )

            if extracted_config and not config_fingerprint:
                config_fingerprint = "|".join(
                    f"{k}={v}" for k, v in sorted(extracted_config.items())
                )
                record.config_fingerprint = config_fingerprint
                if isinstance(record._handle_context, dict):
                    record._handle_context["config_fingerprint"] = config_fingerprint
                _LOG.debug(
                    "[component-features] Auto config_fp for %s: %s",
                    record.component,
                    config_fingerprint,
                )

            if output_sample_count is None:
                extracted_fan_out = _extract_fan_out(record.component, payload)
                if extracted_fan_out is not None:
                    output_sample_count = extracted_fan_out
                    _LOG.debug(
                        "[fan-out] Extracted output_sample_count=%d for %s from argv",
                        extracted_fan_out,
                        record.component,
                    )

            self._register_campaign_fan_out(
                record.component,
                output_sample_count,
                record.campaign_id,
            )

            if normalized_workload_features:
                payload["workload_features"] = dict(normalized_workload_features)
            if normalized_execution_overrides:
                payload["execution_overrides"] = dict(normalized_execution_overrides)
            if preferred_worker_addr:
                payload["preferred_worker_addr"] = str(preferred_worker_addr)
            if preferred_gpu_ids:
                payload["preferred_gpu_ids"] = [
                    str(item) for item in list(preferred_gpu_ids)
                ]

            try:
                payload = validate_task_payload(payload)
            except ContractError as exc:
                await self._set_state(
                    record,
                    TASK_STATE_FAILED,
                    ok=False,
                    exit_code=1,
                    message=str(exc),
                )
                return

            task_campaign_id = str(record.campaign_id or "").strip() or None
            max_dispatch_retries = self._compute_max_refinements()
            resp: Any

            _precomputed_dispatch_plan = getattr(
                handle,
                "_precomputed_dispatch_plan",
                None,
            )
            _precomputed_valid = _precomputed_dispatch_plan is not None and str(
                getattr(_precomputed_dispatch_plan, "task_id", "") or ""
            ) == str(record.task_id)

            self._planner._connect_supervisor_refs()

            try:
                from .signals.resource_profile import ResourceProfileRegistry

                _submit_features = dict(normalized_workload_features or {})
                if not _submit_features and record.workload_features:
                    _submit_features = dict(record.workload_features)
                _submit_input_size = ResourceProfileRegistry.extract_input_size(
                    _submit_features,
                    component=str(record.component or "").strip().lower(),
                )
            except Exception:
                _LOG.warning(
                    "[campaign-submit] extract_input_size failed — "
                    "WSJF remaining_est will degrade to FIFO for task %s",
                    record.task_id,
                    exc_info=True,
                )
                _submit_input_size = 0.0

            if _precomputed_valid:
                _submit_input_size = float(
                    getattr(handle, "_precomputed_input_size", 0.0) or 0.0,
                )
                campaign_hints = dict(
                    getattr(handle, "_precomputed_campaign_hints", {}) or {},
                )
                _dispatch_plan = _precomputed_dispatch_plan
                _launch_prediction_owned = True
                handle._precomputed_dispatch_plan = None
                handle._precomputed_campaign_hints = {}
                handle._precomputed_input_size = 0.0
            else:
                campaign_placement = self._planner.campaign_scheduler.on_task_submit(
                    task_id=record.task_id,
                    campaign_id=task_campaign_id or "",
                    component=record.component,
                    input_size=float(_submit_input_size or 0.0),
                )
                record._campaign_submit_registered = True
                campaign_hints = dict(campaign_placement.scheduling_hints)

            _global_planner = getattr(self._planner, "_global_planner", None)
            if _global_planner is None:
                await self._set_state(
                    record,
                    TASK_STATE_FAILED,
                    ok=False,
                    exit_code=1,
                    message="GlobalPlanner not initialized",
                )
                return

            if self._scheduling_supervisor is None or self._reality_validator is None:
                await self._ensure_core_loop_components()

            if not _precomputed_valid:
                self._planner.prepare_for_planning()

            try:
                if not _precomputed_valid:
                    _cp_plan_start = self._cp_begin()
                    _dynamic_plan_kwargs = (
                        {"execution_overrides": dict(execution_overrides or {})}
                        if callable(
                            getattr(
                                _global_planner,
                                "set_dynamic_batch_profiles",
                                None,
                            )
                        )
                        else {}
                    )
                    _dispatch_plan = _global_planner.plan(
                        task_id=record.task_id,
                        campaign_id=task_campaign_id or "",
                        component=record.component,
                        input_size=float(_submit_input_size or 0.0),
                        logical_batch_size=int(logical_batch_size or 0),
                        is_backfill=bool(campaign_hints.get("is_backfill")),
                        config_fingerprint=str(record.config_fingerprint or ""),
                        input_fingerprint=str(record.input_fingerprint or ""),
                        **_dynamic_plan_kwargs,
                    )
                    _global_planner.timelines.mark_predicted_entry_dispatching(
                        record.task_id,
                        component=record.component,
                        gpu_id=str(_dispatch_plan.target_gpu_id or ""),
                    )
                    _launch_prediction_owned = True
                    self._cp_record_reciprocal_scoring(_dispatch_plan)
                    self._cp_record(
                        "planner_solve",
                        _cp_plan_start,
                        active_wall=True,
                        mode="singleton",
                        task_count=1,
                        component=record.component,
                    )
                record.dispatch_vram_budget_mb = int(
                    getattr(_dispatch_plan, "vram_budget_mb", 0) or 0,
                )
                record.dispatch_ram_budget_mb = int(
                    getattr(_dispatch_plan, "ram_budget_mb", 0) or 0,
                )
                record.dispatch_planned_start_time = float(
                    getattr(_dispatch_plan, "planned_start_time", 0.0) or 0.0,
                )
            except PlanningExhausted as _pe_exc:
                _violation = getattr(_pe_exc, "violation", None)
                _is_component_exhausted = (
                    _violation is not None
                    and getattr(_violation, "violation_type", "")
                    == "component_exhausted"
                )
                if _is_component_exhausted:
                    _LOG.warning(
                        "[task %s] COMPONENT_EXHAUSTED at Phase 1 plan — "
                        "no feasible (component, gpu) pair permanently "
                        "excluded.  Operator intervention required.",
                        record.task_id,
                    )
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=f"PlanningExhausted: {_pe_exc}",
                    )
                    return
                _is_backfill_deadline = bool(
                    campaign_hints.get("is_backfill")
                ) and "no GPU admits it within primary deadline" in str(_pe_exc)
                _LOG.debug(
                    "[task %s] PlanningExhausted (transient) at Phase 1 "
                    "plan: %s — returning %s",
                    record.task_id,
                    _pe_exc,
                    "RetryDeferral(planning_transient)"
                    if _is_backfill_deadline
                    else "SKIP_THIS_CYCLE",
                )
                _returned_skip = True
                from .planning.scheduling_supervisor import (
                    SKIP_THIS_CYCLE as _SKIP,
                )
                from .planning.scheduling_supervisor import (
                    RetryDeferral as _RetryDeferral,
                )

                if _is_backfill_deadline:
                    _runtime_cfg = getattr(self, "_runtime_config", {}) or {}
                    _backoff_s = float(
                        _runtime_cfg.get("planning_transient_backoff_s", 2.0),
                    )
                    return _RetryDeferral(
                        reason="planning_transient",
                        not_before_at=time.time() + max(0.0, _backoff_s),
                    )
                return _SKIP

            task_req = PlannerTaskRequest(
                component=record.component,
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                workload_features=normalized_workload_features,
                execution_overrides=normalized_execution_overrides,
                campaign_id=task_campaign_id,
                preferred_worker_addr=preferred_worker_addr,
                preferred_gpu_ids=[_dispatch_plan.target_gpu_id]
                if _dispatch_plan.target_gpu_id
                else preferred_gpu_ids,
                scheduling_hints=campaign_hints,
                timeout_s=timeout_s,
                output_sample_count=output_sample_count,
                process_name=process_name,
            )

            try:
                _cs = self._planner.campaign_scheduler
                _timelines = getattr(_cs, "_timelines", None)
                if _timelines is not None and hasattr(
                    _timelines,
                    "gc_orphaned_predicted_entries",
                ):
                    _timelines.gc_orphaned_predicted_entries()
                now = time.time()
                last = getattr(self, "_last_safety_net_gc", 0.0)
                if (
                    now - last >= 60.0
                    and _timelines is not None
                    and hasattr(_timelines, "gc_stale_entries_safety_net")
                ):
                    _timelines.gc_stale_entries_safety_net(grace_period_sec=30.0)
                    self._last_safety_net_gc = now
            except Exception as _gc_exc:
                _LOG.debug("[core-loop] Layer-2/3 GC skipped: %s", _gc_exc)

            _constraint_view = None
            try:
                if _global_planner is not None and hasattr(
                    _global_planner,
                    "_constraint_tracker",
                ):
                    tracker_snapshot = getattr(
                        _global_planner._constraint_tracker,
                        "snapshot",
                        None,
                    )
                    if callable(tracker_snapshot):
                        _constraint_view = tracker_snapshot()
            except Exception as _cv_exc:
                _LOG.debug(
                    "[core-loop] ConstraintView snapshot skipped: %s",
                    _cv_exc,
                )
                _constraint_view = None

            handle.is_backfill = bool(campaign_hints.get("is_backfill"))

            addr = ""
            _validation_memo = getattr(self, "_validation_memo", None)
            if _validation_memo is None:
                _validation_memo = {}
                self._validation_memo = _validation_memo
            for _attempt in range(max_dispatch_retries):
                _cp_validation_start = self._cp_begin()
                _fp = self._validation_fingerprint(
                    _dispatch_plan,
                    config_fingerprint=config_fingerprint,
                    input_fingerprint=input_fingerprint,
                )
                _cached = _validation_memo.get(_fp)
                if _cached is not None:
                    result, violation = _cached
                else:
                    (
                        result,
                        violation,
                    ) = await self._reality_validator.validate_and_dispatch(
                        _dispatch_plan,
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
                    )
                    if (
                        result is None
                        and violation is not None
                        and str(getattr(violation, "violation_type", ""))
                        in ("vram_insufficient", "host_ram_saturated")
                    ):
                        if len(_validation_memo) > 4096:
                            _validation_memo.clear()
                        _validation_memo[_fp] = (result, violation)
                self._cp_record(
                    "dispatch_validation_admission",
                    _cp_validation_start,
                    component=record.component,
                    attempt=_attempt + 1,
                    result=(
                        "dispatched"
                        if result is not None
                        else str(
                            getattr(violation, "violation_type", "unknown")
                            if violation
                            else "unknown"
                        )
                    ),
                    gpu_id=str(
                        getattr(_dispatch_plan, "target_gpu_id", "")
                        or getattr(violation, "gpu_id", "")
                        if violation
                        else getattr(_dispatch_plan, "target_gpu_id", "")
                    ),
                )
                if result is not None:
                    _launch_prediction_owned = False
                    resp = result
                    dispatch_gpu_id = getattr(record, "_dispatched_gpu_id", "") or (
                        _dispatch_plan.target_gpu_id or ""
                    )
                    break

                _vtype = violation.violation_type if violation else "unknown"
                self._cp_count("dispatch_rollback", 1, reason=str(_vtype))
                _log_attempt = (
                    _LOG.debug if str(_vtype) == "worker_queue_saturated" else _LOG.info
                )
                _log_attempt(
                    "[task %s] Core Loop attempt %d/%d: %s on GPU %s",
                    record.task_id,
                    _attempt + 1,
                    max_dispatch_retries,
                    _vtype,
                    getattr(violation, "gpu_id", ""),
                )
                _worker_front_deferral = None
                _retained_prediction = False
                if str(_vtype) == "worker_queue_saturated":
                    _worker_front_deferral = (
                        self._worker_queue_saturated_retry_deferral(
                            record=record,
                            violation=violation,
                            config_fingerprint=str(config_fingerprint or "").strip(),
                        )
                    )
                    if _precomputed_valid and _launch_prediction_owned:
                        handle._batch_planning_retry_deferral = _worker_front_deferral
                        _retained_prediction = self._retain_front_deferred_plan(
                            handle,
                            {
                                "campaign_hints": campaign_hints,
                                "input_size": _submit_input_size,
                            },
                            _dispatch_plan,
                            prediction_already_committed=True,
                        )
                if not _retained_prediction:
                    self._rollback_batch_plan_predictions(
                        [record.task_id],
                        reason=f"validation_{_vtype}",
                        include_dispatching=True,
                    )
                _launch_prediction_owned = False

                _fail_gpu = getattr(record, "_dispatched_gpu_id", "")
                if _fail_gpu:
                    with contextlib.suppress(Exception):
                        self._complete_campaign_task(
                            task_id=record.task_id,
                            campaign_id=task_campaign_id or "",
                            component=record.component,
                            gpu_id=_fail_gpu,
                            execution_attempt_id=str(
                                getattr(record, "_reciprocal_event_id", "") or ""
                            ),
                            from_eviction=True,
                        )
                    dispatch_gpu_id = ""
                    record._dispatched_gpu_id = ""

                if _vtype == "scheduler_internal_error":
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message="scheduler internal dispatch invariant",
                    )
                    return

                _retry_not_before_at = 0.0
                _retry_reason = "constraint_updated"

                if violation and _vtype in ("worker_dead", "grpc_error"):
                    _was_guard_kill = False
                    _sup = getattr(self, "supervisor", None)
                    if _sup and violation.worker_name:
                        _wst = _sup.states.get(violation.worker_name)
                        if _wst and _wst.lifecycle_state == "killed":
                            _was_guard_kill = True
                    if not _was_guard_kill:
                        handle.dead_worker_retry_count += 1
                    addr = violation.worker_name or ""
                    if handle.dead_worker_retry_count >= max_dispatch_retries:
                        await self._set_state(
                            record,
                            TASK_STATE_FAILED,
                            ok=False,
                            exit_code=1,
                            message=f"Max dispatch retries ({max_dispatch_retries}) exhausted: {_vtype}",
                        )
                        return
                    backoff_s = min(
                        2 ** max(handle.dead_worker_retry_count, 1),
                        30,
                    )
                    _LOG.warning(
                        "[task %s] %s, deferring retry %d/%d by %ds",
                        record.task_id,
                        "Guard-kill recovery" if _was_guard_kill else _vtype,
                        handle.dead_worker_retry_count,
                        max_dispatch_retries,
                        backoff_s,
                    )
                    _retry_not_before_at = time.time() + max(0.0, backoff_s)
                    _retry_reason = str(_vtype)

                try:
                    if _constraint_view is not None:
                        _progress = _constraint_view.incorporate(violation)
                    else:
                        _progress = _global_planner.incorporate_constraint(violation)
                except PlanningExhausted as _ic_pe_exc:
                    _LOG.warning(
                        "[task %s] COMPONENT_EXHAUSTED raised during "
                        "incorporate_constraint: %s — operator "
                        "intervention required.",
                        record.task_id,
                        _ic_pe_exc,
                    )
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=f"PlanningExhausted: {_ic_pe_exc}",
                    )
                    return
                except Exception as _ic_exc:
                    _LOG.warning(
                        "[core-loop] incorporate_constraint raised: %s",
                        _ic_exc,
                    )
                    _progress = True
                if _progress is None:
                    _progress = True

                if str(_vtype) == "worker_not_ready":
                    _LOG.debug(
                        "[task %s] Core Loop observed worker_not_ready — "
                        "requeueing shortly for batch planning",
                        record.task_id,
                    )
                    _returned_skip = True
                    return self._worker_not_ready_retry_deferral()

                if not _progress:
                    if str(_vtype) == "worker_queue_saturated":
                        _deferral = _worker_front_deferral
                        _LOG.debug(
                            "[task %s] Core Loop fixed-point after %s — "
                            "deferring until worker-front capacity",
                            record.task_id,
                            _vtype,
                        )
                        _returned_skip = True
                        return _deferral
                    _LOG.debug(
                        "[task %s] Core Loop fixed-point after %s — "
                        "returning SKIP_THIS_CYCLE; Supervisor drain will "
                        "re-enqueue on next wake trigger",
                        record.task_id,
                        _vtype,
                    )
                    _returned_skip = True
                    from .planning.scheduling_supervisor import (
                        SKIP_THIS_CYCLE as _SKIP,
                    )

                    return _SKIP

                if str(_vtype) == "worker_queue_saturated":
                    _deferral = _worker_front_deferral
                    _LOG.debug(
                        "[task %s] Core Loop incorporated %s — deferring "
                        "until worker-front capacity",
                        record.task_id,
                        _vtype,
                    )
                    _returned_skip = True
                    return _deferral

                _LOG.debug(
                    "[task %s] Core Loop incorporated %s — requeueing for "
                    "batch planning",
                    record.task_id,
                    _vtype,
                )
                _returned_skip = True
                from .planning.scheduling_supervisor import (
                    RetryDeferral as _RetryDeferral,
                )

                return _RetryDeferral(
                    reason=_retry_reason,
                    not_before_at=_retry_not_before_at,
                )
            else:
                _LOG.info(
                    "[task %s] %d inner dispatch attempts exhausted; "
                    "returning SKIP_THIS_CYCLE for outer wake-driven retry",
                    record.task_id,
                    max_dispatch_retries,
                )
                _returned_skip = True
                from .planning.scheduling_supervisor import (
                    SKIP_THIS_CYCLE as _SKIP,
                )

                return _SKIP

            _LOG.info("[task %s] InferBatch returned ok=%s", record.task_id, resp.ok)

            timing_us = _extract_worker_timing(resp.timing)
            await self._apply_dispatch_attribution(
                record,
                getattr(resp, "attribution", None),
            )

            if not resp.ok:
                raise RuntimeError(resp.error_message or "worker batch failed")

            if not resp.responses:
                await self._set_state(
                    record,
                    TASK_STATE_FAILED,
                    ok=False,
                    exit_code=1,
                    message="worker response missing",
                    worker_timing_us=timing_us,
                )
                return

            item = resp.responses[0]
            telemetry_summary = _extract_task_telemetry(
                item.payload_json,
                bootstrap_memory_summary_json=bytes(
                    getattr(resp, "bootstrap_memory_summary_json", b"") or b""
                ),
                dispatch_memory_window_json=bytes(
                    getattr(resp, "dispatch_memory_window_json", b"") or b""
                ),
            )
            if not item.ok:
                raise RuntimeError(item.error_message or "worker item failed")

            if getattr(record, "_cancel_reason", "") and not bool(
                getattr(handle, "_cancel_requested", False)
            ):
                record._cancel_reason = ""
            await self._set_state(
                record,
                TASK_STATE_SUCCEEDED,
                ok=True,
                exit_code=0,
                message="",
                worker_timing_us=timing_us,
            )

        except CancelledError as _cancelled:
            cancel_reason = getattr(record, "_cancel_reason", "")
            if cancel_reason:
                _backfill_reenqueued = await self._handle_cancel_reenqueue(
                    record,
                    cancel_reason,
                    dispatch_gpu_id,
                    task_campaign_id,
                    argv,
                    tool_cwd,
                    env,
                    timeout_s,
                    profiling,
                    workload_features,
                    config_fingerprint,
                    input_fingerprint,
                    execution_overrides,
                    output_sample_count,
                    process_name,
                    is_backfill_hint=bool(handle.is_backfill),
                )
                if not _backfill_reenqueued:
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=f"cooperative cancel requeue failed: {cancel_reason}",
                    )
            else:
                _LOG.warning(
                    "[CancelledError] no cancel_reason — setting CANCELLED for task=%s",
                    record.task_id[:12],
                )
                await self._set_state(
                    record,
                    TASK_STATE_CANCELLED,
                    ok=False,
                    exit_code=1,
                    message="cancelled",
                )
        except Exception as exc:
            _cancel_reason = getattr(record, "_cancel_reason", "")
            if _cancel_reason:
                _backfill_reenqueued = await self._handle_cancel_reenqueue(
                    record,
                    f"cooperative:{_cancel_reason}",
                    dispatch_gpu_id,
                    task_campaign_id,
                    argv,
                    tool_cwd,
                    env,
                    timeout_s,
                    profiling,
                    workload_features,
                    config_fingerprint,
                    input_fingerprint,
                    execution_overrides,
                    output_sample_count,
                    process_name,
                    is_backfill_hint=bool(handle.is_backfill),
                )
                if not _backfill_reenqueued:
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=f"cooperative cancel requeue failed: {_cancel_reason}",
                    )
            else:
                attempt = getattr(record, "_retry_attempt", 0)
                from .extraction.component_features import has_unlimited_retries

                within_limit = (
                    attempt < self._max_task_retries
                    or has_unlimited_retries(record.component)
                )
                if within_limit and self._is_retryable(record.component, str(exc)):
                    record._retry_attempt = attempt + 1
                    retry_argv = self._adjust_argv_for_retry(
                        record.component, list(argv), attempt + 1
                    )
                    _max_display = (
                        "∞"
                        if has_unlimited_retries(record.component)
                        else str(self._max_task_retries)
                    )
                    _LOG.warning(
                        "[task %s] attempt %d/%s failed (%s), retrying",
                        record.task_id,
                        attempt + 1,
                        _max_display,
                        exc,
                    )
                    _retry_fail_gpu = dispatch_gpu_id or getattr(
                        record, "_dispatched_gpu_id", ""
                    )
                    try:
                        self._complete_campaign_task(
                            task_id=record.task_id,
                            campaign_id=task_campaign_id or "",
                            component=record.component,
                            gpu_id=_retry_fail_gpu,
                            execution_attempt_id=str(
                                getattr(record, "_reciprocal_event_id", "") or ""
                            ),
                            from_eviction=True,
                        )
                        if _retry_fail_gpu:
                            from .signals.resource_profile import (
                                ResourceProfileRegistry,
                            )

                            _retry_input_size = (
                                ResourceProfileRegistry.extract_input_size(
                                    dict(workload_features or {}),
                                    component=str(record.component or "")
                                    .strip()
                                    .lower(),
                                )
                            )
                            _rv = self._planner.campaign_scheduler._predict_vram(
                                record.component,
                                float(_retry_input_size or 0.0),
                                _retry_fail_gpu,
                                config_fingerprint=str(
                                    config_fingerprint or ""
                                ).strip(),
                            )
                    except Exception:
                        _LOG.debug(
                            "[task %s] retry VRAM refresh failed",
                            record.task_id,
                            exc_info=True,
                        )
                    await self._mark_record_pending_for_scheduler(
                        record,
                        message="waiting for scheduler wake",
                    )
                    _retry_handle = self._build_task_handle(
                        record,
                        argv=retry_argv,
                        tool_cwd=tool_cwd,
                        env=env,
                        timeout_s=timeout_s,
                        profiling=profiling,
                        workload_features=workload_features,
                        config_fingerprint=config_fingerprint,
                        input_fingerprint=input_fingerprint,
                        execution_overrides=execution_overrides,
                        output_sample_count=output_sample_count,
                        process_name=process_name,
                        is_backfill_hint=bool(
                            campaign_hints.get("is_backfill")
                            if campaign_hints
                            else False,
                        ),
                    )
                    if self._scheduling_supervisor is None:
                        raise RuntimeError(
                            "SchedulingSupervisor invariant violation: retry_after_adapter_failure "
                            "from within an active Core Loop attempt but supervisor is None",
                        ) from exc
                    _retry_action = self._schedule_adapter_retry_handle(
                        _retry_handle,
                        attempt=attempt + 1,
                    )
                    _backfill_reenqueued = self._retry_supervisor_action_is_live(
                        _retry_action,
                    )
                    if not _backfill_reenqueued:
                        await self._set_state(
                            record,
                            TASK_STATE_FAILED,
                            ok=False,
                            exit_code=1,
                            message=(
                                "retryable adapter failure could not requeue "
                                f"replacement handle (action={_retry_action})"
                            ),
                        )
                else:
                    _LOG.exception("task failed: %s", exc)
                    message = str(exc)
                    if addr and self._check_worker_status:
                        worker_error = await asyncio.to_thread(
                            self._check_worker_status, addr
                        )
                        if worker_error:
                            message = f"{worker_error}\n(original error: {message})"
                    await self._set_state(
                        record,
                        TASK_STATE_FAILED,
                        ok=False,
                        exit_code=1,
                        message=message,
                    )
        finally:
            if _launch_prediction_owned:
                self._rollback_batch_plan_predictions(
                    [record.task_id],
                    reason="launch_owner_exit",
                    include_dispatching=True,
                )
            self._release_front_slot_lease_for_task(record.task_id)
            _releases = getattr(record, "_pending_vram_releases", [])
            if _releases:
                record._pending_vram_releases = []
            for _rel_gpu, _rel_comp, _rel_vram in _releases:
                with contextlib.suppress(Exception):
                    self._planner.on_task_complete_event(_rel_gpu, _rel_comp, _rel_vram)

            if _core_loop_owner and (
                _backfill_reenqueued or _suppress_terminal_completion
            ):
                await self._release_core_loop_owner(
                    record.task_id,
                    _core_loop_owner_task,
                )
                _core_loop_owner = False

            if _suppress_terminal_completion:
                pass
            elif _backfill_reenqueued:
                try:
                    await self._persist_task_observation(
                        record,
                        workload_features=normalized_workload_features,
                        config_fingerprint=config_fingerprint,
                        input_fingerprint=input_fingerprint,
                        telemetry_summary=telemetry_summary,
                    )
                except Exception as exc:
                    _LOG.warning(
                        "[fix-B'] failed to persist re-enqueue task observation %s: %s",
                        record.task_id,
                        exc,
                    )
            elif _returned_skip:
                await self._mark_record_pending_after_skip(record)
            else:
                _final_gpu = dispatch_gpu_id or getattr(
                    record, "_dispatched_gpu_id", ""
                )
                try:
                    _LOG.info(
                        "[finally] on_task_complete %s gpu=%s",
                        record.task_id,
                        _final_gpu,
                    )
                    self._complete_campaign_task(
                        task_id=record.task_id,
                        campaign_id=task_campaign_id or "",
                        component=record.component,
                        gpu_id=_final_gpu,
                        execution_attempt_id=str(
                            getattr(record, "_reciprocal_event_id", "") or ""
                        ),
                    )
                    try:
                        from .signals.observation import (
                            GPObservation,
                            emit_observation,
                        )

                        _task_state = getattr(record, "state", "") or ""
                        _succeeded = str(
                            _task_state
                        ).upper() == "SUCCEEDED" and not bool(
                            getattr(record, "_backfill_reenqueued", False)
                        )
                        if _final_gpu:
                            obs = GPObservation.from_dispatch_result(
                                component=record.component,
                                gpu_id=_final_gpu,
                                actual_duration_sec=float(
                                    getattr(record, "duration_sec", 0.0) or 0.0
                                ),
                                actual_vram_mb=float(
                                    telemetry_summary.get("peak_memory_mib") or 0.0,
                                ),
                                predicted_duration_sec=0.0,
                                predicted_vram_mb=0.0,
                                succeeded=_succeeded,
                            )
                            emit_observation(self._signal_service, obs)
                    except Exception:
                        _LOG.debug(
                            "[finally] GPObservation emission failed",
                            exc_info=True,
                        )
                except Exception as _fin_exc:
                    _LOG.warning(
                        "[finally] on_task_complete failed for %s gpu=%s: %s",
                        record.task_id,
                        _final_gpu,
                        _fin_exc,
                    )
                try:
                    _cp_completion_start = self._cp_begin()
                    await self._persist_task_observation(
                        record,
                        workload_features=normalized_workload_features,
                        config_fingerprint=config_fingerprint,
                        input_fingerprint=input_fingerprint,
                        telemetry_summary=telemetry_summary,
                    )
                    self._cp_record(
                        "completion_bookkeeping",
                        _cp_completion_start,
                        active_wall=True,
                        component=record.component,
                        state=STATE_NAMES.get(int(record.state), "UNSPECIFIED"),
                    )
                except Exception as exc:
                    _LOG.warning(
                        "failed to persist task observation %s: %s", record.task_id, exc
                    )
                record._terminal_callback_pending = False
                if not _returned_skip:
                    await self._finalize_terminal_task_handle(record, handle)
            if _core_loop_owner:
                await self._release_core_loop_owner(
                    record.task_id,
                    _core_loop_owner_task,
                )

    def _wait_for_terminal(self, task_id: str) -> asyncio.Future[Any]:
        """Plan — caller-facing terminal-state
        awaitable.  Returns the TaskHandle's own ``_done_future``
        (single source of truth: no parallel dict).  Raises ``KeyError``
        for unknown ``task_id``.

        Tests use this in place of the HEAD-era
        ``await service._task_tasks[task_id]`` idiom.
        """
        handle = self._handles[task_id]
        fut = handle._done_future
        if fut is None:
            raise RuntimeError(
                f"TaskHandle for {task_id} has no _done_future — "
                "_build_task_handle invariant violated",
            )
        return fut

    async def _finalize_terminal_task_handle(
        self,
        record: TaskRecord,
        handle: Any,
    ) -> bool:
        """Resolve/pop the terminal handle only if it is still current.

        Retry/requeue paths replace ``self._handles[task_id]`` with a fresh
        handle before the old attempt reaches ``finally``.  The old attempt
        must never complete or pop that replacement handle; otherwise HTTP/NF
        waiters can lose the only live terminal future.
        """
        current = self._handles.get(record.task_id)
        if current is not handle:
            _LOG.debug(
                "[finally] skip terminal handle cleanup for %s; current handle "
                "was replaced",
                record.task_id,
            )
            return False
        async with self._lock:
            self._active_regular_tasks.discard(record.task_id)
        fut = getattr(handle, "_done_future", None)
        if fut is not None and not fut.done():
            fut.set_result(getattr(record, "state", None))
        if self._handles.get(record.task_id) is handle:
            self._handles.pop(record.task_id, None)
        return True

    async def close(self) -> None:
        """Shutdown lifecycle for the Supervisor / observers / event
        handler spun up by ``_ensure_core_loop_components``.  Tests
        call this from ``try/finally`` at the end of each
        ``asyncio.run(_run())`` driver so ``_cancel_all_tasks`` is not
        left to race with long-lived drain loops.

        Contract (Plan):
          - Stop drain + observer + event-handler (cancel + await).
          - Stop ConstraintTracker sweeper.
          - Cancel any still-pending ``_done_future`` so callers blocked
            on ``_wait_for_terminal`` unwind with ``CancelledError``.
          - Clear component references so a subsequent
            ``_ensure_core_loop_components`` can re-initialize.
        """
        self._closing = True
        delayed_retry_tasks = list(getattr(self, "_delayed_retry_tasks", set()))
        for task in delayed_retry_tasks:
            if not task.done():
                task.cancel()
        if delayed_retry_tasks:
            await asyncio.gather(*delayed_retry_tasks, return_exceptions=True)
            self._delayed_retry_tasks.clear()

        if self._scheduling_supervisor is not None:
            try:
                await self._scheduling_supervisor.stop()
            except Exception as exc:
                _LOG.warning("[close] supervisor.stop raised: %s", exc)
            self._scheduling_supervisor = None
        if self._gpu_health_observer is not None:
            try:
                await self._gpu_health_observer.stop()
            except Exception as exc:
                _LOG.warning("[close] gpu_health_observer.stop raised: %s", exc)
            self._gpu_health_observer = None
        if self._event_handler is not None:
            try:
                await self._event_handler.stop()
            except Exception as exc:
                _LOG.warning("[close] event_handler.stop raised: %s", exc)
            self._event_handler = None
        _gp = getattr(self._planner, "_global_planner", None)
        tracker = getattr(_gp, "_constraint_tracker", None) if _gp is not None else None
        if tracker is not None and hasattr(tracker, "stop_sweeper"):
            try:
                await tracker.stop_sweeper()
            except Exception as exc:
                _LOG.warning("[close] sweeper.stop raised: %s", exc)
        self._sweeper_started = False
        for _task_id, handle in list(self._handles.items()):
            self._detach_precomputed_plan_from_handle(
                handle,
                reason="service_close",
            )
            fut = getattr(handle, "_done_future", None)
            if fut is not None and not fut.done():
                fut.cancel()
            if handle._current_inner is not None and not handle._current_inner.done():
                handle._current_inner.cancel()
        self._handles.clear()
        self._reality_validator = None
        executor = getattr(self, "_sqlite_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _is_retryable(component: str, error_message: str) -> bool:
        """Check if a failure is retryable via the adapter's is_retryable_error hook.

        If the adapter doesn't declare the hook, all errors are retryable (default).
        """
        from .extraction.component_features import get_retryable_checker

        if str(error_message or "").startswith("dynamic_batch_contract_error:"):
            return False
        fn = get_retryable_checker(component)
        if fn is not None:
            try:
                return bool(fn(error_message))
            except Exception:
                _LOG.warning(
                    "retryable-error checker failed for component=%s",
                    component,
                    exc_info=True,
                )
        return True

    @staticmethod
    def _adjust_argv_for_retry(
        component: str, argv: list[str], attempt: int
    ) -> list[str]:
        """Component-specific argv adjustments on retry.

        Delegates to the adapter's ``adjust_argv_for_retry(argv, attempt)``
        function in ``workload_features.py`` if declared.  Otherwise returns
        argv unchanged.
        """
        from .extraction.component_features import get_retry_adjuster

        fn = get_retry_adjuster(component)
        if fn is not None:
            try:
                adjusted = fn(list(argv), attempt)
                if adjusted is not None:
                    return adjusted
            except Exception:
                _LOG.warning(
                    "retry argv adjuster failed for component=%s",
                    component,
                    exc_info=True,
                )
        return argv

    @staticmethod
    def _component_feature_cache_key(
        component: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Stable key for per-record workload feature extraction.

        The adapter extractor is pure with respect to component, argv, env,
        and workdir.  Retry paths that mutate argv/env naturally get a new key;
        ordinary scheduler deferrals for the same task reuse the cached result.
        """
        env_raw = payload.get("env") or {}
        env_items = []
        if isinstance(env_raw, Mapping):
            env_items = sorted((str(k), str(v)) for k, v in env_raw.items())
        key_payload = {
            "component": str(component or "").strip().lower(),
            "argv": [str(item) for item in list(payload.get("argv") or [])],
            "env": env_items,
            "workdir": str(payload.get("workdir") or ""),
        }
        encoded = json.dumps(
            key_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()

    def _extract_component_features_cached(
        self,
        record: TaskRecord,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Extract workload/config features once per task payload shape."""
        key = self._component_feature_cache_key(record.component, payload)
        if (
            bool(getattr(record, "_feature_extraction_done", False))
            and str(getattr(record, "_feature_extraction_cache_key", "") or "") == key
        ):
            return (
                dict(getattr(record, "_cached_extracted_workload_features", {}) or {}),
                dict(getattr(record, "_cached_extracted_config_features", {}) or {}),
            )

        length_features, config_features = _extract_component_features(
            record.component,
            dict(payload),
        )
        runtime_cfg = getattr(self, "_runtime_config", {}) or {}
        phase_cfg = (
            runtime_cfg.get("phase_scheduler", {})
            if isinstance(runtime_cfg, Mapping)
            else {}
        )
        config_features = _filter_dynamic_batch_config(
            record.component,
            config_features,
            active=(
                str(runtime_cfg.get("scheduler_implementation", "")).lower()
                in ("proton_heft", "proton_react")
                or (
                    str(runtime_cfg.get("scheduler_implementation", "")).lower()
                    == "proton_phase"
                    and not bool(phase_cfg.get("phase_shadow_mode", False))
                )
            ),
        )
        record._feature_extraction_cache_key = key
        record._feature_extraction_done = True
        record._cached_extracted_workload_features = dict(length_features or {})
        record._cached_extracted_config_features = {
            str(k): str(v) for k, v in dict(config_features or {}).items()
        }
        return (
            dict(record._cached_extracted_workload_features),
            dict(record._cached_extracted_config_features),
        )

    def _adapter_retry_delay_s(self, attempt: int) -> float:
        """Return adapter retry delay for a 1-based retry attempt."""
        if not getattr(self, "_adapter_retry_backoff_enabled", False):
            return 0.0
        try:
            attempt_idx = max(0, int(attempt) - 1)
            base_s = max(0.0, float(self._adapter_retry_backoff_base_s))
            factor = max(1.0, float(self._adapter_retry_backoff_factor))
            max_s = max(base_s, float(self._adapter_retry_backoff_max_s))
        except Exception:
            return 0.0
        return min(base_s * (factor**attempt_idx), max_s)

    @staticmethod
    def _retry_supervisor_action_is_live(action: str) -> bool:
        action = str(action or "")
        return action in {
            "submitted",
            "recovered_stale_owned",
            "submitted_legacy",
        } or action.startswith("woken_")

    def _submit_retry_handle_to_supervisor(self, handle: Any, *, reason: str) -> str:
        supervisor = getattr(self, "_scheduling_supervisor", None)
        if supervisor is None:
            raise RuntimeError(
                "SchedulingSupervisor invariant violation: retry submit requested "
                "but supervisor is None",
            )
        ensure = getattr(supervisor, "ensure_handle_requeued_or_woken", None)
        if callable(ensure):
            action = ensure(handle, reason=reason)
        else:
            supervisor.submit(handle, reason=reason)
            action = "submitted_legacy"
        if action in {"recovered_stale_owned", "submitted", "dropped"}:
            _LOG.info(
                "[task %s] retry handle supervisor action=%s reason=%s",
                getattr(handle, "task_id", "<unknown>"),
                action,
                reason,
            )
        return str(action)

    def _schedule_adapter_retry_handle(self, handle: Any, *, attempt: int) -> str:
        if self._scheduling_supervisor is None:
            raise RuntimeError(
                "SchedulingSupervisor invariant violation: retry_after_adapter_failure "
                "from within an active Core Loop attempt but supervisor is None",
            )
        delay_s = self._adapter_retry_delay_s(attempt)
        reason = "retry_after_adapter_failure"
        if delay_s > 0.0:
            handle._not_before_at = max(
                float(getattr(handle, "_not_before_at", 0.0) or 0.0),
                time.time() + delay_s,
            )
            _LOG.warning(
                "[task %s] scheduling adapter retry attempt %d in %.3fs",
                getattr(handle, "task_id", "<unknown>"),
                attempt,
                delay_s,
            )
        return self._submit_retry_handle_to_supervisor(handle, reason=reason)

    async def _persist_task_observation(
        self,
        record: TaskRecord,
        *,
        workload_features: Mapping[str, Any],
        config_fingerprint: str,
        input_fingerprint: str,
        telemetry_summary: Mapping[str, Any],
    ) -> None:
        def _safe_optional_positive_int(value: Any) -> int | None:
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except Exception:
                return None
            return parsed if parsed > 0 else None

        run_level = "level_a"
        timing_us = dict(record.worker_timing_us or {})
        runtime_sec: float | None = None
        try:
            execute_us = int(timing_us.get("execute_us", 0))
            if execute_us > 0:
                runtime_sec = float(execute_us) / 1_000_000.0
        except Exception:
            runtime_sec = None
        intrinsic_runtime_available = runtime_sec is not None
        qc_status = "" if intrinsic_runtime_available else "soft_drop"
        qc_reason = "" if intrinsic_runtime_available else "missing_execute_us"
        canonical_active_vram_mib = _to_float(telemetry_summary.get("active_vram_mib"))
        active_vram_mib = canonical_active_vram_mib
        if active_vram_mib is None:
            active_vram_mib = _to_float(telemetry_summary.get("active_memory_mib"))
        peak_vram_mib = _to_float(telemetry_summary.get("peak_vram_mib"))
        if peak_vram_mib is None:
            peak_vram_mib = _to_float(telemetry_summary.get("peak_memory_mib"))
        host_active_memory_mib = _to_float(
            telemetry_summary.get("host_active_memory_mib")
        )
        host_peak_memory_mib = _to_float(telemetry_summary.get("host_peak_memory_mib"))
        host_resident_memory_mib = _to_float(
            telemetry_summary.get("host_resident_memory_mib")
        )
        vram_memory_attribution = str(
            telemetry_summary.get("vram_memory_attribution") or ""
        ).strip()
        host_memory_attribution = str(
            telemetry_summary.get("host_memory_attribution") or ""
        ).strip()
        measured_resident_mib = _to_float(telemetry_summary.get("resident_memory_mib"))
        measured_resident_source = str(
            telemetry_summary.get("resident_memory_source") or ""
        ).strip()

        if (
            measured_resident_mib
            and measured_resident_mib > 0
            and measured_resident_source
        ):
            resident_baseline_snapshot = _normalize_resident_baseline_snapshot(
                {
                    "resident_memory_mib": measured_resident_mib,
                    "resident_memory_source": measured_resident_source,
                    "resident_baseline_collected_at": _to_float(
                        telemetry_summary.get("resident_baseline_collected_at")
                        or record.dispatch_resident_baseline_collected_at
                    ),
                    "resident_baseline_lifecycle_token": str(
                        telemetry_summary.get("resident_baseline_lifecycle_token")
                        or record.dispatch_resident_baseline_lifecycle_token
                        or ""
                    ).strip(),
                    "resident_baseline_state": str(
                        telemetry_summary.get("resident_baseline_state")
                        or record.dispatch_resident_baseline_state
                        or ""
                    ).strip(),
                }
            )
        else:
            resident_baseline_snapshot = _normalize_resident_baseline_snapshot(
                {
                    "resident_memory_mib": record.dispatch_resident_memory_mib,
                    "resident_memory_source": record.dispatch_resident_memory_source,
                    "resident_baseline_collected_at": record.dispatch_resident_baseline_collected_at,
                    "resident_baseline_lifecycle_token": record.dispatch_resident_baseline_lifecycle_token,
                    "resident_baseline_state": record.dispatch_resident_baseline_state,
                }
            )

        decision_payload = dict(record.decision_payload or {})
        if decision_payload:
            state_name = STATE_NAMES.get(int(record.state), "UNSPECIFIED")
            failure_outcome = (
                state_name if state_name in {"FAILED", "CANCELLED"} else ""
            )
            actual_payload = {
                "runtime_sec": _to_float(runtime_sec),
                "peak_memory_mib": peak_vram_mib,
                "active_memory_mib": active_vram_mib,
                "peak_vram_mib": peak_vram_mib,
                "active_vram_mib": active_vram_mib,
                "vram_memory_measurement": str(
                    telemetry_summary.get("vram_memory_measurement")
                    or telemetry_summary.get("active_memory_measurement")
                    or ""
                ).strip(),
                "vram_memory_qc_keep": _to_bool(
                    telemetry_summary.get("vram_memory_qc_keep"),
                    default=_to_bool(
                        telemetry_summary.get("memory_qc_keep"), default=False
                    ),
                ),
                "vram_memory_attribution": vram_memory_attribution,
                "host_peak_memory_mib": host_peak_memory_mib,
                "host_active_memory_mib": host_active_memory_mib,
                "host_resident_memory_mib": host_resident_memory_mib,
                "host_memory_measurement": str(
                    telemetry_summary.get("host_memory_measurement") or ""
                ).strip(),
                "host_memory_qc_keep": _to_bool(
                    telemetry_summary.get("host_memory_qc_keep"), default=False
                ),
                "host_memory_attribution": host_memory_attribution,
                "memory_basis": str(
                    telemetry_summary.get("memory_basis") or ""
                ).strip(),
                "active_memory_scope": str(
                    telemetry_summary.get("memory_basis")
                    or telemetry_summary.get("active_memory_scope")
                    or ""
                ).strip(),
                "peak_fidelity": str(
                    telemetry_summary.get("peak_fidelity") or ""
                ).strip(),
                "total_upper_bound_mib": _to_float(
                    telemetry_summary.get("total_upper_bound_mib")
                ),
                "active_memory_measurement": str(
                    telemetry_summary.get("active_memory_measurement") or ""
                ).strip(),
                "resident_memory_mib": _to_float(
                    resident_baseline_snapshot.get("resident_memory_mib")
                ),
                "resident_memory_source": str(
                    resident_baseline_snapshot.get("resident_memory_source") or ""
                ).strip(),
                "resident_baseline_collected_at": _to_float(
                    resident_baseline_snapshot.get("resident_baseline_collected_at")
                ),
                "resident_baseline_lifecycle_token": str(
                    resident_baseline_snapshot.get("resident_baseline_lifecycle_token")
                    or ""
                ).strip(),
                "resident_baseline_state": str(
                    resident_baseline_snapshot.get("resident_baseline_state") or ""
                ).strip(),
                "worker_generation_token": str(
                    record.worker_generation_token or ""
                ).strip(),
                "run_ordinal_in_generation": _to_int(record.run_ordinal_in_generation),
                "is_first_real_run": bool(record.is_first_real_run),
                "generation_token_stale": bool(record.generation_token_stale),
                "memory_qc_keep": _to_bool(
                    telemetry_summary.get("memory_qc_keep"), default=False
                ),
                "concurrent_execute_overlap": _to_bool(
                    telemetry_summary.get("concurrent_execute_overlap"),
                    default=False,
                ),
                "mean_gpu_util_percent": _to_float(
                    telemetry_summary.get("mean_gpu_util_percent")
                ),
                "telemetry_wall_clock_sec": _to_float(
                    telemetry_summary.get("telemetry_wall_clock_sec")
                ),
                "bootstrap_memory_summary": dict(
                    _mapping_payload(telemetry_summary.get("bootstrap_memory_summary"))
                ),
                "dispatch_memory_window": dict(
                    _mapping_payload(telemetry_summary.get("dispatch_memory_window"))
                ),
                "failure_outcome": failure_outcome,
            }
            current_actual = decision_payload.get("actual")
            merged_actual = (
                dict(current_actual) if isinstance(current_actual, Mapping) else {}
            )
            merged_actual.update(actual_payload)
            decision_payload["actual"] = merged_actual

        source_event_id = str(record.task_id or "").strip()
        run_key = (
            f"task:{source_event_id}" if source_event_id else f"task:{uuid.uuid4().hex}"
        )
        normalized_config_fp = str(config_fingerprint or "").strip()
        if not normalized_config_fp:
            normalized_config_fp = _stable_fingerprint(
                {
                    "component": record.component,
                    "profiling_level": run_level,
                    "argv": [],
                    "worker_name": record.dispatch_worker_name,
                },
                prefix="cfg",
            )
        normalized_input_fp = str(input_fingerprint or "").strip()
        if not normalized_input_fp:
            normalized_input_fp = _stable_fingerprint(
                dict(workload_features or {}),
                prefix="inp",
            )
        axes_payload = canonicalize_axes(dict(workload_features or {}))
        batch_context = _batch_observation_context(record, telemetry_summary)
        if batch_context.get("batch_phase") or batch_context.get("batch_policy"):
            axes_payload.update(
                {
                    "batch_phase": batch_context.get("batch_phase"),
                    "batch_policy": batch_context.get("batch_policy"),
                    "dynamic_batch_fallback_reason": batch_context.get(
                        "fallback_reason"
                    ),
                    "dynamic_batch_selected_k": batch_context.get(
                        "execution_batch_size"
                    ),
                    "dynamic_batch_final_admission_k": batch_context.get(
                        "final_admission_batch_size"
                    ),
                    "dynamic_batch_argument_applied_k": batch_context.get(
                        "argument_applied_batch_size"
                    ),
                    "dynamic_batch_consumed_k": batch_context.get(
                        "consumed_batch_size"
                    ),
                    "dynamic_batch_logical_n": batch_context.get("logical_batch_size"),
                    "dynamic_batch_group_count": batch_context.get("group_count"),
                }
            )
            axes_payload = canonicalize_axes(
                {
                    key: value
                    for key, value in axes_payload.items()
                    if value not in (None, "")
                }
            )

        cell_key = f"task:{record.component}:{normalized_input_fp}"
        run_index_kwargs = {
            "run_key": run_key,
            "run_source": "task",
            "source_event_id": source_event_id,
            "cell_schema_id": "legacy_v0",
            "cell_key": cell_key,
            "run_id": str(record.nf_task_id or record.task_id),
            "campaign_id": str(record.campaign_id or ""),
            "run_name": str(record.run_name or ""),
            "submitter": str(record.submitter or ""),
            "campaign_metadata_version": int(record.campaign_metadata_version or 0),
            "config_fingerprint": normalized_config_fp,
            "input_fingerprint": normalized_input_fp,
            "component": str(record.component or "").strip().lower(),
            "level": run_level,
            "state": STATE_NAMES.get(record.state, "UNSPECIFIED"),
            "sample_id": str(workload_features.get("sample_id") or ""),
            "input_batch_size": _safe_optional_positive_int(
                workload_features.get("input_batch_size")
            ),
            "output_sample_count": _safe_optional_positive_int(
                workload_features.get("output_sample_count")
            ),
            "dispatch_worker_name": str(record.dispatch_worker_name or ""),
            "dispatch_worker_addr": str(record.dispatch_worker_addr or ""),
            "dispatch_gpu_ids": list(record.dispatch_gpu_ids or []),
            "gateway_instance_id": str(self._gateway_identity.get("instance_id") or ""),
            "gateway_bind_addr": str(self._gateway_identity.get("bind_addr") or ""),
            "gateway_git_commit": str(self._gateway_identity.get("git_commit") or ""),
            "gateway_started_at": _to_float(self._gateway_identity.get("started_at")),
            "worker_timing_us": timing_us or None,
            "artifact_manifest": [],
            "runtime_sec": runtime_sec,
            "mean_gpu_util_percent": telemetry_summary.get("mean_gpu_util_percent"),
            "peak_memory_mib": peak_vram_mib,
            "active_memory_mib": active_vram_mib,
            "active_memory_scope": str(
                telemetry_summary.get("memory_basis")
                or telemetry_summary.get("active_memory_scope")
                or ""
            ).strip(),
            "active_memory_measurement": str(
                telemetry_summary.get("active_memory_measurement") or ""
            ).strip(),
            "peak_vram_mib": peak_vram_mib,
            "active_vram_mib": active_vram_mib,
            "vram_memory_measurement": str(
                telemetry_summary.get("vram_memory_measurement")
                or telemetry_summary.get("active_memory_measurement")
                or ""
            ).strip(),
            "vram_memory_qc_keep": _to_bool(
                telemetry_summary.get("vram_memory_qc_keep"),
                default=_to_bool(
                    telemetry_summary.get("memory_qc_keep"), default=False
                ),
            ),
            "vram_memory_attribution": vram_memory_attribution,
            "host_peak_memory_mib": host_peak_memory_mib,
            "host_active_memory_mib": host_active_memory_mib,
            "host_resident_memory_mib": host_resident_memory_mib,
            "host_memory_measurement": str(
                telemetry_summary.get("host_memory_measurement") or ""
            ).strip(),
            "host_memory_qc_keep": _to_bool(
                telemetry_summary.get("host_memory_qc_keep"), default=False
            ),
            "host_memory_attribution": host_memory_attribution,
            "resident_memory_mib": _to_float(
                resident_baseline_snapshot.get("resident_memory_mib")
            ),
            "resident_memory_source": str(
                resident_baseline_snapshot.get("resident_memory_source") or ""
            ).strip(),
            "resident_baseline_collected_at": _to_float(
                resident_baseline_snapshot.get("resident_baseline_collected_at")
            ),
            "resident_baseline_lifecycle_token": str(
                resident_baseline_snapshot.get("resident_baseline_lifecycle_token")
                or ""
            ).strip(),
            "resident_baseline_state": str(
                resident_baseline_snapshot.get("resident_baseline_state") or ""
            ).strip(),
            "worker_generation_token": str(
                record.worker_generation_token or ""
            ).strip(),
            "run_ordinal_in_generation": _to_int(record.run_ordinal_in_generation),
            "is_first_real_run": bool(record.is_first_real_run),
            "memory_qc_keep": _to_bool(
                telemetry_summary.get("memory_qc_keep"), default=False
            ),
            "concurrent_execute_overlap": _to_bool(
                telemetry_summary.get("concurrent_execute_overlap"),
                default=False,
            ),
            "telemetry_wall_clock_sec": telemetry_summary.get(
                "telemetry_wall_clock_sec"
            ),
            "telemetry_collected_at": time.time(),
            "bootstrap_memory_summary": _mapping_payload(
                telemetry_summary.get("bootstrap_memory_summary")
            ),
            "dispatch_memory_window": _mapping_payload(
                telemetry_summary.get("dispatch_memory_window")
            ),
            "scheduler_decision": decision_payload,
            "predicted_runtime_sec": _to_float(record.signal_runtime_sec),
            "predicted_p90_sec": _to_float(record.signal_runtime_upper_sec),
            "error": str(record.message or ""),
            "qc_status": qc_status,
            "qc_keep": intrinsic_runtime_available,
            "qc_reason": qc_reason,
            "created_at": record.created_at,
            "finished_at": record.updated_at,
            "axes": axes_payload,
        }

        def _write_run_index_records() -> None:
            self._run_index.upsert_run(**run_index_kwargs)
            if int(record.campaign_metadata_version or 0) >= 1:
                self._run_index.upsert_campaign_registry(
                    campaign_id=str(record.campaign_id or ""),
                    run_name=str(record.run_name or ""),
                    submitter=str(record.submitter or ""),
                    gateway_instance_id=str(
                        self._gateway_identity.get("instance_id") or ""
                    ),
                    gateway_bind_addr=str(
                        self._gateway_identity.get("bind_addr") or ""
                    ),
                    gateway_git_commit=str(
                        self._gateway_identity.get("git_commit") or ""
                    ),
                    gateway_started_at=_to_float(
                        self._gateway_identity.get("started_at")
                    ),
                    status=(
                        "completed"
                        if STATE_NAMES.get(record.state, "UNSPECIFIED") == "SUCCEEDED"
                        else "failed"
                        if STATE_NAMES.get(record.state, "UNSPECIFIED")
                        in {"FAILED", "CANCELLED"}
                        else "active"
                        if STATE_NAMES.get(record.state, "UNSPECIFIED")
                        in {"RUNNING", "SUBMITTED"}
                        else "unknown"
                    ),
                    last_seen_at=record.updated_at,
                    event_ts=record.updated_at,
                )
            self._run_index.link_job(
                job_id=source_event_id, kind="task", run_key=run_key
            )

        await self._run_index_call(_write_run_index_records)

        component = str(record.component or "").strip().lower()

        current_state = STATE_NAMES.get(record.state, "UNSPECIFIED")
        peak_activation_mb: float | None = None
        if current_state in {"SUCCEEDED", "FAILED", "CANCELLED", "RUNNING"}:
            worker_addr = str(record.dispatch_worker_addr or "").strip()
            task_id = str(record.task_id or "").strip()
            peak_activation_mb = (
                self._signal_service.consume_activation_peak(
                    worker_addr, task_id=task_id
                )
                if worker_addr
                else None
            )
            synthetic_supervisor_peak = _to_float(
                telemetry_summary.get("synthetic_supervisor_activation_mib")
            )
            if (
                (peak_activation_mb is None or peak_activation_mb <= 0)
                and synthetic_supervisor_peak is not None
                and synthetic_supervisor_peak > 0
                and worker_addr
            ):
                self._signal_service.record_activation_peak(
                    worker_addr,
                    synthetic_supervisor_peak,
                    exec_inflight=1,
                )
                peak_activation_mb = self._signal_service.consume_activation_peak(
                    worker_addr,
                    task_id=task_id,
                )

        worker_vram_qc_keep = _to_bool(
            telemetry_summary.get("vram_memory_qc_keep"),
            default=False,
        )
        temporal_window = _mapping_payload(
            telemetry_summary.get("dispatch_memory_window")
        )
        vram_window_flags = _mapping_payload(temporal_window.get("flags"))
        vram_increment_is_identifiable = not _to_bool(
            vram_window_flags.get("lazy_materialization_detected"),
            default=False,
        )
        vram_is_request_owned = vram_memory_attribution == "request_process_tree"
        worker_vram_valid = (
            canonical_active_vram_mib is not None
            and canonical_active_vram_mib > 0
            and worker_vram_qc_keep
            and vram_increment_is_identifiable
            and vram_is_request_owned
            and not record.generation_token_stale
        )
        from .signals.resource_profile import ResourceProfileRegistry

        input_size = ResourceProfileRegistry.extract_input_size(
            dict(workload_features or {}), component=component
        )
        temporal_low_vram = _to_float(temporal_window.get("low_vram"))
        temporal_start_ratio = _to_float(temporal_window.get("peak_start_ratio"))
        temporal_end_ratio = _to_float(temporal_window.get("peak_end_ratio"))
        if (
            current_state == "SUCCEEDED"
            and worker_vram_valid
            and temporal_low_vram is not None
            and temporal_start_ratio is not None
            and temporal_end_ratio is not None
        ):
            self._signal_service.record_temporal_vram(
                component,
                temporal_low_vram,
                temporal_start_ratio,
                temporal_end_ratio,
                config_fingerprint=str(config_fingerprint or "").strip(),
                input_size=input_size,
                runtime_sec=runtime_sec,
                memory_attribution=vram_memory_attribution,
            )
            timelines = getattr(
                getattr(self._planner, "campaign_scheduler", None),
                "_timelines",
                None,
            )
            invalidate_temporal_cache = getattr(
                timelines,
                "invalidate_temporal_profile_cache",
                None,
            )
            if callable(invalidate_temporal_cache):
                invalidate_temporal_cache()

        if (
            current_state in ("SUCCEEDED", "FAILED", "CANCELLED", "RUNNING")
            and component
        ):
            is_solo = not _to_bool(
                telemetry_summary.get("concurrent_execute_overlap"),
                default=False,
            )
            dispatch_window = telemetry_summary.get("dispatch_memory_window")
            concurrent_count = 1
            if isinstance(dispatch_window, dict):
                concurrent_count = max(
                    1,
                    int(
                        dispatch_window.get("max_active_request_count_during_window", 1)
                        or 1
                    ),
                )
            elif not is_solo:
                concurrent_count = 2

            if current_state == "SUCCEEDED":
                gpu_util = _to_float(telemetry_summary.get("mean_gpu_util_percent"))
                if gpu_util is None and self._worker_inventory_provider:
                    worker_addr = str(record.dispatch_worker_addr or "").strip()
                    for w in self._worker_inventory_provider() or []:
                        if str(w.get("addr", "")).strip() == worker_addr:
                            gpu_util = _to_float(w.get("gpu_util_percent"))
                            break
                active_mem = canonical_active_vram_mib if worker_vram_valid else None
                execute_us: float | None = None
                if timing_us and isinstance(timing_us, dict):
                    execute_us = _to_float(timing_us.get("execute_us"))
                self._signal_service.update_workload_profile(
                    component,
                    gpu_util_percent=gpu_util,
                    execute_us=execute_us,
                    active_memory_mib=active_mem,
                    is_solo=is_solo,
                    concurrent_task_count=concurrent_count,
                )
            vram_observation_mib: float | None = None
            vram_observation_is_solo = False
            progress_ratio: float | None = None
            if worker_vram_valid:
                vram_observation_mib = canonical_active_vram_mib
                vram_observation_is_solo = current_state == "SUCCEEDED" and is_solo
            elif peak_activation_mb and peak_activation_mb > 0:
                vram_observation_mib = peak_activation_mb
                vram_observation_is_solo = current_state == "SUCCEEDED" and is_solo
                if current_state != "SUCCEEDED":
                    actual_partial = float(
                        getattr(record, "duration_sec", 0.0) or runtime_sec or 0.0
                    )
                    if actual_partial > 0:
                        try:
                            cfg_fp_str = (
                                str(config_fingerprint or "").strip() or "__default__"
                            )
                            gp_full = (
                                self._signal_service.resource_profiles.predict_latency(
                                    component,
                                    cfg_fp_str,
                                    input_size=input_size,
                                )
                            )
                        except Exception:
                            gp_full = None
                        if gp_full is not None and gp_full > 0:
                            progress_ratio = actual_partial / float(gp_full)

            if vram_observation_mib is not None and vram_observation_mib > 0:
                if current_state != "SUCCEEDED":
                    dispatch_window_payload = _mapping_payload(
                        telemetry_summary.get("dispatch_memory_window")
                    )
                    raw_peak_vram = _to_float(
                        dispatch_window_payload.get("dispatch_peak_total_mib")
                    )
                    raw_pre_vram = _to_float(
                        dispatch_window_payload.get("dispatch_pre_quiescent_mib")
                    )
                    raw_active_vram = None
                    if raw_peak_vram is not None and raw_peak_vram > 0:
                        raw_active_vram = raw_peak_vram
                        if raw_pre_vram is not None:
                            raw_active_vram = max(
                                0.0,
                                float(raw_peak_vram) - float(raw_pre_vram),
                            )
                    if (
                        raw_active_vram is not None
                        and raw_active_vram > vram_observation_mib
                    ):
                        vram_observation_mib = raw_active_vram
                skip_vram_feedback = False
                uncertain_lower_bound_sample = (
                    not worker_vram_valid
                    or current_state != "SUCCEEDED"
                    or progress_ratio is not None
                )
                if uncertain_lower_bound_sample:
                    try:
                        prior_vram = (
                            self._signal_service.resource_profiles.predict_vram(
                                component,
                                str(config_fingerprint or "").strip(),
                                input_size=input_size,
                            )
                        )
                    except Exception:
                        prior_vram = None
                    if (
                        prior_vram is not None
                        and prior_vram > 0
                        and vram_observation_mib < prior_vram
                    ):
                        skip_vram_feedback = True
                        _LOG.info(
                            "[vram-feedback] skip lower-bound sample "
                            "component=%s observed=%.1f prior=%.1f "
                            "state=%s attribution=%s progress_ratio=%s",
                            component,
                            vram_observation_mib,
                            prior_vram,
                            current_state,
                            vram_memory_attribution or "",
                            (
                                f"{progress_ratio:.3f}"
                                if progress_ratio is not None
                                else ""
                            ),
                        )
                if not skip_vram_feedback:
                    self._signal_service.record_solo_vram(
                        component,
                        vram_observation_mib,
                        config_fingerprint=str(config_fingerprint or "").strip(),
                        input_size=input_size,
                        campaign_id=str(record.campaign_id or ""),
                        is_solo=vram_observation_is_solo,
                        progress_ratio=progress_ratio,
                        batch_context=batch_context,
                        memory_attribution=(
                            vram_memory_attribution
                            if worker_vram_valid
                            else "supervisor_activation_lower_bound"
                        ),
                    )
            active_ram_mib = _to_float(telemetry_summary.get("host_active_memory_mib"))
            if active_ram_mib is not None and active_ram_mib > 0:
                host_window = _mapping_payload(
                    telemetry_summary.get("host_dispatch_memory_window")
                )
                concurrent_count = 1
                if isinstance(host_window, Mapping):
                    try:
                        concurrent_count = max(
                            1,
                            int(
                                host_window.get(
                                    "max_active_request_count_during_window"
                                )
                                or host_window.get("active_request_count_at_start")
                                or 1
                            ),
                        )
                    except Exception:
                        concurrent_count = 1
                host_qc_keep = _to_bool(
                    telemetry_summary.get("host_memory_qc_keep"), default=False
                )
                host_flags = _mapping_payload(host_window.get("flags"))
                host_increment_is_identifiable = not _to_bool(
                    host_flags.get("lazy_materialization_detected"),
                    default=False,
                )
                host_is_request_owned = (
                    host_memory_attribution == "request_process_tree"
                )
                if (
                    host_qc_keep
                    and host_increment_is_identifiable
                    and host_is_request_owned
                    and not record.generation_token_stale
                ):
                    ram_is_solo = is_solo if current_state == "SUCCEEDED" else False
                    self._signal_service.record_solo_ram(
                        component,
                        active_ram_mib,
                        config_fingerprint=str(config_fingerprint or "").strip(),
                        input_size=input_size,
                        campaign_id=str(record.campaign_id or ""),
                        is_solo=ram_is_solo,
                        concurrent_task_count=concurrent_count,
                        batch_context=batch_context,
                        memory_attribution=host_memory_attribution,
                    )


GATEWAY_SERVICE_APP_KEY: web.AppKey[GatewayHTTPService] = web.AppKey(
    "gateway_service",
    GatewayHTTPService,
)


def create_app(
    worker_selector: WorkerSelector,
    work_root: Path | None = None,
    max_inflight_per_worker: int = 1,
    ensure_worker_ready: Callable[
        [str, str | None, str | None, list[str] | None], Awaitable[None]
    ]
    | None = None,
    worker_capabilities_resolver: Callable[[str, str | None], dict[str, Any]]
    | None = None,
    worker_inventory_provider: Callable[[], list[dict[str, Any]]] | None = None,
    profiling_index_db_path: Path | None = None,
    on_task_start: Callable[..., None] | None = None,
    on_task_end: Callable[[str], Awaitable[None]] | None = None,
    resource_tracker: Any | None = None,
    check_worker_status: Callable[[str], str | None] | None = None,
    evict_fn: Callable[[str, int], Awaitable[int]] | None = None,
    adjust_worker_inflight_by_name: Callable[[str, int], None] | None = None,
    bump_dispatch_pending: Callable[[str], None] | None = None,
    release_dispatch_pending: Callable[[str], None] | None = None,
    gateway_identity: Mapping[str, Any] | None = None,
    pipeline_dag: PipelineDAG | None = None,
    max_task_retries: int = 0,
    scheduler_policy: str = "campaign_fifo",
    adapter_retry_backoff: Mapping[str, Any] | None = None,
    dispatch_backlog_per_worker: int = 1,
    dispatch_start_grace_sec: float = 1.0,
    control_plane_overhead_enabled: bool = False,
    control_plane_overhead_top_k: int = 32,
) -> web.Application:
    """Create the aiohttp application with routes."""
    service = GatewayHTTPService(
        worker_selector=worker_selector,
        work_root=work_root,
        max_inflight_per_worker=max_inflight_per_worker,
        ensure_worker_ready=ensure_worker_ready,
        worker_capabilities_resolver=worker_capabilities_resolver,
        worker_inventory_provider=worker_inventory_provider,
        profiling_index_db_path=profiling_index_db_path,
        on_task_start=on_task_start,
        on_task_end=on_task_end,
        resource_tracker=resource_tracker,
        check_worker_status=check_worker_status,
        evict_fn=evict_fn,
        adjust_worker_inflight_by_name=adjust_worker_inflight_by_name,
        bump_dispatch_pending=bump_dispatch_pending,
        release_dispatch_pending=release_dispatch_pending,
        gateway_identity=gateway_identity,
        pipeline_dag=pipeline_dag,
        max_task_retries=max_task_retries,
        scheduler_policy=scheduler_policy,
        adapter_retry_backoff=adapter_retry_backoff,
        dispatch_backlog_per_worker=dispatch_backlog_per_worker,
        dispatch_start_grace_sec=dispatch_start_grace_sec,
        control_plane_overhead_enabled=control_plane_overhead_enabled,
        control_plane_overhead_top_k=control_plane_overhead_top_k,
    )

    middlewares = []
    if control_plane_overhead_enabled:

        @web.middleware
        async def _control_plane_observability_middleware(
            request: web.Request,
            handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
        ) -> web.StreamResponse:
            path = str(request.path or "")
            observe = path.startswith(
                (
                    "/api/v1/signals",
                    "/api/v1/workers",
                    "/api/v1/profile",
                    "/api/v1/campaigns",
                    "/api/v1/telemetry",
                )
            )
            _cp_start = service._cp_begin() if observe else 0
            try:
                return await handler(request)
            finally:
                if observe:
                    service._cp_record(
                        "observability_endpoint",
                        _cp_start,
                        method=request.method,
                        path=path,
                    )

        middlewares.append(_control_plane_observability_middleware)

    app = web.Application(middlewares=middlewares)
    service.register_routes(app)

    app[GATEWAY_SERVICE_APP_KEY] = service

    return app


async def serve(
    host: str,
    port: int,
    *,
    worker_selector: WorkerSelector,
    work_root: Path | None,
    max_inflight_per_worker: int,
    ensure_worker_ready: Callable[
        [str, str | None, str | None, list[str] | None], Awaitable[None]
    ]
    | None = None,
    worker_capabilities_resolver: Callable[[str, str | None], dict[str, Any]]
    | None = None,
    worker_inventory_provider: Callable[[], list[dict[str, Any]]] | None = None,
    access_log: logging.Logger | None = None,
) -> None:
    """Start the HTTP server."""
    app = create_app(
        worker_selector=worker_selector,
        work_root=work_root,
        max_inflight_per_worker=max_inflight_per_worker,
        ensure_worker_ready=ensure_worker_ready,
        worker_capabilities_resolver=worker_capabilities_resolver,
        worker_inventory_provider=worker_inventory_provider,
    )
    runner = web.AppRunner(app, access_log=access_log)
    await runner.setup()
    site = web.TCPSite(runner, host, port, backlog=65535)
    await site.start()
    _LOG.info("Gateway HTTP server listening on http://%s:%d", host, port)

    try:
        while True:
            await asyncio.sleep(3600)
    except CancelledError as _cancelled:
        return
    finally:
        await runner.cleanup()
