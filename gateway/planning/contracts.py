"""Data contracts for the 3-layer scheduling architecture.

Global Planner → Reality Validator → Event Handler

These dataclasses define the interfaces between the three components:
- ``DispatchPlan``: Planner's output — a fully-specified proactive plan
- ``ConstraintViolation``: Validator's feedback — gap between plan and reality
- ``ConstraintAssumptions``: What the Planner assumed (Validator validates these)
- ``PreInitAction``: Scheduled pre-init trigger for downstream stages

Plan version reference:  (ConstraintViolation 15 types, canonical Cu/Co for
backfill, interference formula `Σ (sd-1) × overlap`, `gpu_redirect` removed per
 architectural principle, `_would_slow_primary` removed per ).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any




class ViolationType:
    """Canonical violation type names — Plan Type Semantics.

    Exhaustively covers all Validator failure scenarios (WorkerResolver →
    ReservationManager → AdmissionGate → WorkerActivator → GapDetector →
    Dispatcher).  Categories:

    (a) Resource contention: vram_insufficient, compute_saturated,
        worker_queue_saturated, host_ram_saturated, memory_admission_rejected,
        host_resource_exhausted, preempted
    (b) Worker lifecycle: worker_dead, worker_not_ready, worker_unreachable,
        activation_failed, grpc_error
    (c) Hardware / execution integrity: gpu_unhealthy, numerical_failure,
        stage_failed
    (d) Scheduler meta: component_exhausted, scheduler_internal_error

    NOTE: ``gpu_redirect`` was removed in Plan (acquire_compute_slot is
    planned-GPU-only, no Validator re-routing).  ``backfill_admission_stale``
    was removed in Plan (Dispatcher cannot make admission decisions).
    ``host_ram_saturated`` was added  — admission gate's host-RAM
    axis (K8s `--eviction-hard=memory.available<N` parity; Slurm `MemSpecLimit`
    parity).  Distinct from ``host_resource_exhausted`` (which is worker-
    specific permanent failure) and from ``compute_saturated`` (GPU-level only):
    host_ram_saturated is cluster-wide transient pressure derived from
    /proc/meminfo MemAvailable, retry-able on next wake.
    """

    VRAM_INSUFFICIENT = "vram_insufficient"
    COMPUTE_SATURATED = "compute_saturated"
    WORKER_QUEUE_SATURATED = "worker_queue_saturated"
    HOST_RAM_SATURATED = "host_ram_saturated"
    MEMORY_ADMISSION_REJECTED = "memory_admission_rejected"
    HOST_RESOURCE_EXHAUSTED = "host_resource_exhausted"
    PREEMPTED = "preempted"

    WORKER_DEAD = "worker_dead"
    WORKER_NOT_READY = "worker_not_ready"
    WORKER_UNREACHABLE = "worker_unreachable"
    ACTIVATION_FAILED = "activation_failed"
    GRPC_ERROR = "grpc_error"

    GPU_UNHEALTHY = "gpu_unhealthy"
    NUMERICAL_FAILURE = "numerical_failure"
    STAGE_FAILED = "stage_failed"

    COMPONENT_EXHAUSTED = "component_exhausted"
    SCHEDULER_INTERNAL_ERROR = "scheduler_internal_error"

    ALL = frozenset(
        {
            VRAM_INSUFFICIENT,
            COMPUTE_SATURATED,
            WORKER_QUEUE_SATURATED,
            HOST_RAM_SATURATED,
            MEMORY_ADMISSION_REJECTED,
            HOST_RESOURCE_EXHAUSTED,
            PREEMPTED,
            WORKER_DEAD,
            WORKER_NOT_READY,
            WORKER_UNREACHABLE,
            ACTIVATION_FAILED,
            GRPC_ERROR,
            GPU_UNHEALTHY,
            NUMERICAL_FAILURE,
            STAGE_FAILED,
            COMPONENT_EXHAUSTED,
            SCHEDULER_INTERNAL_ERROR,
        }
    )


class ViolationSource:
    """Where the violation originated — used for correlation-window merging."""

    WORKER = "worker"
    SCHEDULER = "scheduler"
    NETWORK = "network"
    EXTERNAL = "external"




@dataclass
class PreInitAction:
    """A scheduled pre-init trigger for a downstream pipeline stage.

    Created by the Planner's solve() loop when placing a task that has
    DAG successors.  The Validator fires this trigger at ``trigger_at``.
    """

    component: str
    target_gpu_id: str
    trigger_at: float
    init_duration_sec: float
    weight_vram_mb: float




@dataclass
class ConstraintAssumptions:
    """What the Planner assumed when producing a DispatchPlan.

    The Reality Validator checks each assumption against live state.
    Any mismatch produces a ConstraintViolation.
    """

    assumed_available_vram_mb: float
    assumed_gpu_active_count: int
    assumed_worker_ready: bool
    assumed_interference_slowdown: float
    assumed_available_host_ram_mb: float = 0.0




@dataclass
class DispatchPlan:
    """Planner output: a fully-specified proactive scheduling plan.

    The Reality Validator checks this plan against live system state.
    If feasible → dispatch.  If not → ConstraintViolation feedback.

    All values derived from GP observations — no arbitrary constants.
    """

    task_id: str
    campaign_id: str
    component: str

    target_gpu_id: str

    target_worker_name: str
    target_worker_addr: str | None

    vram_budget_mb: int
    predicted_latency_sec: float

    is_backfill: bool
    needs_cold_start: bool
    ram_budget_mb: int = 0
    planned_start_time: float = (
        0.0
    )

    pre_init_schedule: list[PreInitAction] = field(default_factory=list)

    constraint_assumptions: ConstraintAssumptions | None = None

    worker_metadata: dict[str, Any] = field(default_factory=dict)

    plan_snapshot_id: str = ""




@dataclass
class ConstraintViolation:
    """Structured feedback from Reality Validator to Global Planner.

    16 violation types (plan ,  post-cleanup +
     host_ram_saturated) — exhaustively covers all Validator
    failure scenarios.  Each type has a well-defined semantic boundary
    and Planner action (see ``GlobalPlanner.incorporate_constraint``).

    Correlation fields (Blocker 2 — VRAM-guard-kill triple-observation
    ambiguity): one root cause may emit multiple violations (in-flight
    gRPC break, health-check fail, admission reject on retry).  Fields
    below allow Planner to merge fallouts under the same correlation_id.

    Use ``ViolationType`` string constants for ``violation_type`` where
    possible.
    """

    violation_type: str
    gpu_id: str
    worker_name: str = ""

    source: str = ViolationSource.WORKER
    correlation_id: str = ""
    plan_snapshot_id: str = ""
    grpc_code: int | None = None
    cuda_xid: int | None = None

    requested_vram_mb: int | None = None
    available_vram_mb: int | None = None

    active_count: int | None = None

    host_mem_available_mib: int | None = None
    host_mem_threshold_mib: int | None = None

    memory_guard_details: dict[str, Any] | None = None

    progress_info: dict[str, Any] | None = None

    failed_plan: DispatchPlan | None = None

    timestamp: float = field(default_factory=_time.time)


__all__ = [
    "PreInitAction",
    "ConstraintAssumptions",
    "DispatchPlan",
    "ConstraintViolation",
    "ViolationType",
    "ViolationSource",
]
