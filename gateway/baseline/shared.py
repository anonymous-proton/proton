"""
Shared infrastructure for baseline schedulers (fifo / k8s / slurm).

Surface area is intentionally minimal — proton's 3-layer architecture
(SchedulingScenario / ConstraintTracker / RealityValidator) is NOT
used here.  Baselines see only:
  * WorkerSupervisor (worker process lifecycle, gRPC channel pool)
  * Static workers.profile.yaml (peak / mean estimate from a previous
    proton bench's GP database)
  * Task submission queue (the same queue that http_server feeds)

This module provides:
  * `WorkersProfile` / `WorkersProfileEntry` — typed loader for
    `configs/workers.profile.yaml`
  * `TaskSubmission` — minimal task descriptor seen by baselines
  * `DispatchDecision` — what a baseline emits per dispatch
  * `BaselineSchedulerProtocol` — what every baseline must implement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml


@dataclass(frozen=True)
class WorkersProfileEntry:
    """Static peak estimate per component.

    Generated from observed nf-baseline trace.tsv (Nextflow ``realtime``
    column, which is wall-clock and therefore already includes container
    init + model load + inference).  Aggregated across all observed
    workload groups — Group A (synthetic) + Group B (CASP) folded in.

    Field semantics:

    weight_max_mb     — peak weight VRAM (μ + 1.96σ over all observed
                        input_sizes, all groups).  k8s residency-aware
                        accounting: charged once per (component, gpu_id),
                        reused across concurrent inferences.
    activation_max_mb — peak activation VRAM.  k8s charges this per
                        concurrent inference (transient, released on
                        task completion).
    runtime_max_sec   — ``2 × observed_max(realtime)`` from nf-baseline
                        traces (k8s + slurm union).  Used by slurm_style
                        as the ``--time`` reservation window — backfill
                        literature convention F=2 (Tsafrir et al.
                        TPDS 2007; Lawson-Smirni-Puiu Perf. Eval. 2005;
                        Mu'alem-Feitelson TPDS 2001).  Already includes
                        init since NF ``realtime`` is wall-clock end-to-
                        end.  k8s_style + fifo do NOT consume this.
    ram_max_mb       — peak host RAM increment attributable to one task.
                        Used by proton-k8s host-RAM feasibility checking.
    """

    component: str
    weight_max_mb: int
    activation_max_mb: int
    runtime_max_sec: float
    ram_max_mb: int = 0

    @property
    def total_vram_mb(self) -> int:
        """Total VRAM footprint a baseline must reserve to admit a task
        of this component (weight is held for the worker lifetime,
        activation overlaps the inference window)."""
        return self.weight_max_mb + self.activation_max_mb

    @property
    def slurm_time_sec(self) -> float:
        """Slurm ``--time`` upstream semantics — wall-clock upper bound.

        Equals ``runtime_max_sec`` (= 2 × observed_max from NF traces,
        which already includes container/model-load overhead since NF
        ``realtime`` is end-to-end wall-clock).  k8s_style + fifo do
        NOT consume this property."""
        return self.runtime_max_sec


@dataclass(frozen=True)
class WorkersProfile:
    """Typed view of `configs/workers.profile.yaml`.

    Loaded once at scheduler init.  Baselines treat this as immutable
    for the lifetime of the gateway (mirrors Slurm's `--time` being
    fixed at submit + K8s' `resources.requests` being immutable for
    the pod lifetime).
    """

    components: Dict[str, WorkersProfileEntry]

    @classmethod
    def load(cls, path: Path) -> "WorkersProfile":
        with path.open() as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict) or "components" not in raw:
            raise ValueError(
                f"workers.profile.yaml at {path} missing top-level "
                f"'components' key"
            )
        components: Dict[str, WorkersProfileEntry] = {}
        for comp_name, fields_raw in raw["components"].items():
            components[comp_name] = WorkersProfileEntry(
                component=comp_name,
                weight_max_mb=int(fields_raw["weight_max_mb"]),
                activation_max_mb=int(fields_raw["activation_max_mb"]),
                runtime_max_sec=float(fields_raw["runtime_max_sec"]),
                ram_max_mb=int(fields_raw.get("ram_max_mb", 0) or 0),
            )
        return cls(components=components)

    def get(self, component: str) -> WorkersProfileEntry:
        if component not in self.components:
            raise KeyError(
                f"Component {component!r} not in workers.profile.yaml. "
                f"Available: {sorted(self.components)}"
            )
        return self.components[component]


@dataclass
class TaskSubmission:
    """Minimal task descriptor seen by baselines.

    Baselines never see DAG / b-rank / GP posterior / fan-out.  Only
    these fields are exposed (mirrors what an upstream task-level
    scheduler would receive).
    """

    task_id: str
    component: str
    input_size: float
    submit_time: float
    arrival_time: float
    workload_features: Dict[str, Any] = field(default_factory=dict)
    account: str = ""
    fairshare: float = 0.0


@dataclass
class DispatchDecision:
    """What a baseline emits per dispatch attempt.

    The gateway's dispatch path then turns this into a gRPC InferBatch
    call against the chosen worker — same path that proton uses, just
    without proton's RealityValidator stages.
    """

    task_id: str
    target_gpu_id: str
    target_worker_name: str
    component: str
    is_dispatchable: bool
    reason: str = ""


class BaselineSchedulerProtocol(Protocol):
    """Common interface every baseline must satisfy.

    Deliberately tiny — no `incorporate_constraint`, no `re_plan`, no
    scenario state queries.  Baselines are expected to hold their own
    minimal state internally.
    """

    name: str

    def submit(self, task: TaskSubmission) -> None:
        """Add a task to the baseline's internal queue."""
        ...

    def step(self, now: float) -> List[DispatchDecision]:
        """Run one scheduling cycle.  Returns the list of dispatches
        that should be attempted this cycle (may be empty)."""
        ...

    def on_dispatch_success(self, task_id: str, gpu_id: str, now: float) -> None:
        """Notify the baseline that a dispatch attempt succeeded.
        Baselines update their internal accounting (e.g. k8s
        `requested_vram_mb`, slurm reservation freeze)."""
        ...

    def on_dispatch_failure(self, task_id: str, reason: str, now: float) -> None:
        """Notify the baseline that a dispatch attempt failed (gRPC
        error, OOM, worker unavailable).  Baselines decide how to
        re-enqueue / give up — proton's ConstraintTracker is NOT
        consulted."""
        ...

    def on_task_complete(self, task_id: str, now: float) -> None:
        """Notify the baseline that a task finished (success or
        failure terminal).  Baselines update their accounting."""
        ...
