"""
FIFO baseline scheduler — strict arrival-order priority queue, no
backfill, no DAG awareness, no campaign awareness.

Worst-of-all-baselines reference point.  Paper's lower-bound.

Decision per cycle:
  1. Sort pending tasks by (arrival_time, task_id).
  2. For the head task, pick the GPU with highest available_vram that
     fits `weight_max + activation_max` from workers.profile.yaml.
     Ties: lowest GPU id (deterministic).
  3. If no GPU fits, SKIP this task this cycle.  Other tasks are NOT
     considered — strict FIFO; this is how the paper's contribution
     surface widens (head-of-line block).
  4. Failure path: caller re-enqueues at the back of the queue via
     on_dispatch_failure + a fresh submit() with the original
     TaskSubmission.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set

from gateway.baseline.shared import (
    DispatchDecision,
    TaskSubmission,
    WorkersProfile,
)


@dataclass
class _GpuSlot:
    """Per-GPU bookkeeping (mirrors NodeInfo cache, minimal)."""

    gpu_id: str
    capacity_vram_mb: int
    requested_vram_mb: int = 0
    bound_tasks: Set[str] = field(default_factory=set)

    @property
    def available_vram_mb(self) -> int:
        return self.capacity_vram_mb - self.requested_vram_mb


class FifoBaselineScheduler:
    """Strict FIFO priority queue, no backfill."""

    name = "fifo"

    def __init__(
        self,
        profile: WorkersProfile,
        gpu_capacity_mb: Dict[str, int],
        worker_name_format: str = "{component}-gpu{gpu_id}",
    ) -> None:
        self._profile = profile
        self._worker_name_format = worker_name_format
        self._slots: Dict[str, _GpuSlot] = {
            gpu_id: _GpuSlot(gpu_id=gpu_id, capacity_vram_mb=cap)
            for gpu_id, cap in sorted(gpu_capacity_mb.items())
        }
        self._pending: Deque[TaskSubmission] = deque()
        self._inflight: Dict[str, TaskSubmission] = {}

    def submit(self, task: TaskSubmission) -> None:
        self._pending.append(task)

    def step(self, now: float) -> List[DispatchDecision]:
        """Strict FIFO: examine ONLY the head task each cycle."""
        if not self._pending:
            return []
        ordered = sorted(self._pending, key=lambda t: (t.arrival_time, t.task_id))
        self._pending = deque(ordered)
        head = self._pending[0]
        entry = self._profile.get(head.component)
        candidates = [
            slot for slot in self._slots.values()
            if slot.available_vram_mb >= entry.total_vram_mb
        ]
        if not candidates:
            return []
        best = max(candidates, key=lambda s: (s.available_vram_mb, -int(s.gpu_id)))
        self._pending.popleft()
        worker_name = self._worker_name_format.format(
            component=head.component, gpu_id=best.gpu_id
        )
        return [DispatchDecision(
            task_id=head.task_id,
            target_gpu_id=best.gpu_id,
            target_worker_name=worker_name,
            component=head.component,
            is_dispatchable=True,
        )]

    def on_dispatch_success(self, task_id: str, gpu_id: str, now: float) -> None:
        if task_id in self._inflight:
            return
        sub = self._just_dispatched.pop(task_id, None)
        if sub is None:
            return
        slot = self._slots[gpu_id]
        entry = self._profile.get(sub.component)
        slot.requested_vram_mb += entry.total_vram_mb
        slot.bound_tasks.add(task_id)
        self._inflight[task_id] = sub

    def on_dispatch_failure(self, task_id: str, reason: str, now: float) -> None:
        sub = self._inflight.pop(task_id, None)
        self._just_dispatched.pop(task_id, None)
        if sub is not None:
            self._release_slot(task_id, sub)

    def on_task_complete(self, task_id: str, now: float) -> None:
        sub = self._inflight.pop(task_id, None)
        if sub is not None:
            self._release_slot(task_id, sub)


    def remember_dispatched(self, task: TaskSubmission) -> None:
        """The gateway calls this immediately after step() picks a task
        but before on_dispatch_success.  Keeps the side-channel small
        (just the in-flight dispatches awaiting the success/failure
        callback)."""
        self._just_dispatched[task.task_id] = task

    @property
    def _just_dispatched(self) -> Dict[str, TaskSubmission]:
        if not hasattr(self, "_just_dispatched_cache"):
            self._just_dispatched_cache: Dict[str, TaskSubmission] = {}
        return self._just_dispatched_cache

    def _release_slot(self, task_id: str, sub: TaskSubmission) -> None:
        for slot in self._slots.values():
            if task_id in slot.bound_tasks:
                slot.bound_tasks.discard(task_id)
                entry = self._profile.get(sub.component)
                slot.requested_vram_mb = max(0, slot.requested_vram_mb - entry.total_vram_mb)
                return
