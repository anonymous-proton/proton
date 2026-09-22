"""
Baseline schedulers (fifo / k8s / slurm) — algorithm-only ports of
upstream task-level schedulers.

Each baseline is a faithful port of the upstream scheduling algorithm
with `file:line` citations next to each ported block.  None of them
use `SchedulingScenario` / `ConstraintTracker` / `RealityValidator`
(proton's 3-layer architecture) — only `WorkerSupervisor` + gRPC
dispatch + `workers.yaml` are shared.

Selection happens at startup via the runtime config file:

  configs/runtime.proton.yaml  -> gateway.planning.global_planner.GlobalPlanner
  configs/runtime.fifo.yaml    -> gateway.baseline.fifo.scheduler.FifoBaselineScheduler
  configs/runtime.k8s.yaml     -> gateway.baseline.k8s.scheduler.K8sBaselineScheduler
  configs/runtime.slurm.yaml   -> gateway.baseline.slurm.scheduler.SlurmBaselineScheduler

See `_workspace/plan_baseline_schedulers.md` for full design rationale.
"""

from gateway.baseline.shared import (
    BaselineSchedulerProtocol,
    DispatchDecision,
    TaskSubmission,
    WorkersProfile,
    WorkersProfileEntry,
)

__all__ = [
    "BaselineSchedulerProtocol",
    "DispatchDecision",
    "TaskSubmission",
    "WorkersProfile",
    "WorkersProfileEntry",
]
