"""Data classes for task-to-worker dispatch (Plan).

Plan removed the
entire ``SchedulingPolicy`` hierarchy from production code — HEFT min-EFT
in ``GlobalPlanner.solve()`` is the sole placement authority.  This
module retains only the **data classes** required by the 3-layer
architecture:

    - ``SchedulingContext``   — request-level metadata carrier
    - ``PlacementDecision``   — per-task placement record
    - ``WorkerSelection``     — Planner → Validator hand-off

Historical ``SchedulingPolicy``/``SchedulerRegistry`` + 10
subclasses have been deleted per plan  row for
``gateway/planning/scheduler.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..signals.contracts import IntrinsicSignalSummary, PlannerIntent



@dataclass(frozen=True)
class SchedulingContext:
    """Immutable snapshot of request-level metadata available at scheduling time."""

    component: str = ""
    preferred_gpu_ids: Optional[List[str]] = None
    preferred_worker_addr: Optional[str] = None
    hints: Dict[str, Any] = field(default_factory=dict)
    campaign_id: Optional[str] = None
    intrinsic_signal: Optional["IntrinsicSignalSummary"] = None
    planner_intent: Optional["PlannerIntent"] = None


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class PlacementDecision:
    version: int = 1
    chosen_worker_name: str = ""
    chosen_worker_addr: str = ""
    chosen_gpu_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    intrinsic_signal: Optional[Dict[str, Any]] = None
    planner_intent: Optional[Dict[str, Any]] = None

    def chosen_worker(self) -> Dict[str, Any]:
        return {
            "name": str(self.chosen_worker_name or ""),
            "addr": str(self.chosen_worker_addr or ""),
            "gpu_ids": [str(item) for item in self.chosen_gpu_ids if str(item)],
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "chosen_worker": self.chosen_worker(),
            "reasons": [str(item) for item in self.reasons if str(item)],
            "intrinsic_signal": dict(self.intrinsic_signal or {}),
            "planner_intent": dict(self.planner_intent or {}),
        }


@dataclass(frozen=True)
class WorkerSelection:
    addr: str
    ready: bool
    gpu_ids: tuple[str, ...]
    worker_name: str
    resident_baseline_snapshot: Dict[str, Any]
    max_concurrency: int
    priority: int
    estimator_worker_context: Dict[str, Any]
    placement: PlacementDecision
    worker_addr: Optional[str] = None
    worker_pid: Optional[int] = None
    current_activation_mib: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorkerSelection":
        placement_payload = _mapping(payload.get("placement"))
        legacy_decision = _mapping(payload.get("scheduler_decision"))
        chosen_worker = _mapping(placement_payload.get("chosen_worker"))
        if not chosen_worker:
            chosen_worker = _mapping(legacy_decision.get("chosen_worker"))
        reasons = [
            str(item).strip()
            for item in list(placement_payload.get("reasons") or legacy_decision.get("reasons") or [])
            if str(item).strip()
        ]
        placement = PlacementDecision(
            version=int(placement_payload.get("version") or legacy_decision.get("version") or 1),
            chosen_worker_name=str(chosen_worker.get("name") or payload.get("worker_name") or "").strip(),
            chosen_worker_addr=str(chosen_worker.get("addr") or payload.get("addr") or "").strip(),
            chosen_gpu_ids=tuple(
                str(item).strip()
                for item in list(chosen_worker.get("gpu_ids") or payload.get("gpu_ids") or [])
                if str(item).strip()
            ),
            reasons=tuple(dict.fromkeys(reasons)),
            intrinsic_signal=_mapping(placement_payload.get("intrinsic_signal")),
            planner_intent=_mapping(placement_payload.get("planner_intent")),
        )
        worker_pid = payload.get("worker_pid")
        return cls(
            addr=str(payload.get("addr") or "").strip(),
            ready=bool(payload.get("ready", True)),
            gpu_ids=tuple(
                str(item).strip()
                for item in list(payload.get("gpu_ids") or [])
                if str(item).strip()
            ),
            worker_name=str(payload.get("worker_name") or placement.chosen_worker_name or "").strip(),
            resident_baseline_snapshot=_mapping(payload.get("resident_baseline_snapshot")),
            worker_addr=str(payload.get("worker_addr") or "").strip(),
            worker_pid=int(worker_pid) if worker_pid is not None else None,
            max_concurrency=max(1, int(payload.get("max_concurrency") or 1)),
            priority=int(payload.get("priority") or 100),
            estimator_worker_context=_mapping(payload.get("estimator_worker_context")),
            placement=placement,
        )

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "addr": str(self.addr or ""),
            "ready": bool(self.ready),
            "gpu_ids": [str(item) for item in self.gpu_ids if str(item)],
            "worker_name": str(self.worker_name or ""),
            "resident_baseline_snapshot": dict(self.resident_baseline_snapshot),
            "worker_addr": str(self.worker_addr or ""),
            "worker_pid": int(self.worker_pid) if self.worker_pid is not None else None,
            "max_concurrency": int(self.max_concurrency),
            "priority": int(self.priority),
            "estimator_worker_context": dict(self.estimator_worker_context),
            "placement": self.placement.as_dict(),
        }


__all__ = [
    "SchedulingContext",
    "PlacementDecision",
    "WorkerSelection",
]
