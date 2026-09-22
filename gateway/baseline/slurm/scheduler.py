"""
Slurm-style baseline scheduler — algorithm-only port of the
sched/backfill plugin from SchedMD/slurm @ slurm-23-11-11-1.

Upstream pin: SchedMD/slurm @ slurm-23-11-11-1
              (commit upstream)

Per-block citations follow the form:
  Port of <repo>/<file>:<line-range> @ <tag>

Algorithmic shape (simplified to match the paper's contribution
surface — interference-free, single-partition, single-QoS, no het
jobs, no preemption):

  1. **Priority queue**   (priority_multifactor.c::_get_priority_internal:
                           weight_age * (now - accrue_time) / max_age)
     Older job → higher priority.  Linear age with 1.0 cap after
     max_age elapsed.  We linearize without the cap (timescales here
     are short relative to default Slurm `priority_max_age=1 week`).

  2. **`_start_job`**     (backfill.c:3213-3290 _start_job)
     Try the head-of-queue first.  If a GPU has enough VRAM RIGHT NOW
     (taking already-running tasks into account), dispatch.

  3. **Reservation**       (backfill.c:1741-3210 _attempt_backfill,
                           specifically the node_space update path)
     If head can't start now, find the GPU whose currently-running
     task finishes soonest and reserve that GPU at that time.  The
     reservation is recorded as a `ReservationSlot`.

  4. **EASY backfill**     (backfill.c:_attempt_backfill the inner
                           per-pending-job loop + `_test_resv_overlap`
                           backfill.c:3496-3526)
     Iterate lower-priority pending jobs.  Each is admitted iff:
       a. Some GPU has enough VRAM right now, AND
       b. Its `[now, now + runtime_max]` window does NOT overlap the
          head's reservation on the same GPU (= "doesn't push back the
          highest-priority job").  `runtime_max = μ + 1.96σ`
          (workers.profile.yaml::runtime_max_sec) — exact match for
          Slurm's user-supplied `--time` upper bound.
     This is the EASY (Extensible Argonne Scheduling System) variant
     — the one Slurm's default `sched/backfill` plugin implements.

Concurrent execution: multiple tasks can bind to the same `(component,
gpu_id)` worker simultaneously; per-GPU `_running` list accumulates
all bound `_RunningTask` records, and `available_vram_mb` =
`capacity - Σ running.vram_mb`.  Mirrors Slurm's `gres-shard` semantics
where a single GPU can host multiple concurrent jobs as long as
total declared GRES does not exceed the GPU's shard count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from gateway.baseline.shared import (
    DispatchDecision,
    TaskSubmission,
    WorkersProfile,
)
from gateway.baseline.k8s.scheduler import _read_mem_available_mib
from gateway.baseline.slurm.reservation import ReservationSlot


@dataclass
class _RunningTask:
    """Per-bound-task record on a `_GpuSlot`.

    Mirrors slurm/.../job_mgr.c::job_record_t (subset) in that we keep
    the start time + expected runtime to drive the reservation
    lookahead.
    """

    task_id: str
    component: str
    start_time: float
    expected_release_time: float
    vram_mb: int
    ram_mb: int = 0


@dataclass
class _GpuSlot:
    """Per-GPU bookkeeping (one entry per GRES-gpu in cluster).

    Maps to slurm/src/common/gres.c GRES gpu node tracking — we keep
    the running tasks list so `_predicted_release_time` can compute
    "earliest time this GPU has free VRAM" without reaching into
    Slurm's full select_nodes() machinery.
    """

    gpu_id: str
    capacity_vram_mb: int
    running: List[_RunningTask] = field(default_factory=list)

    @property
    def used_vram_mb(self) -> int:
        return sum(t.vram_mb for t in self.running)

    @property
    def available_vram_mb(self) -> int:
        return max(0, self.capacity_vram_mb - self.used_vram_mb)

    def earliest_release_time(self, now: float) -> Optional[float]:
        """When does this GPU first free enough VRAM to host another
        task of any size?  Returns the soonest expected_release_time
        among currently running tasks, or None if no tasks running."""
        future = [t.expected_release_time for t in self.running if t.expected_release_time > now]
        return min(future) if future else None

    def projected_available_vram_at(self, target_time: float) -> int:
        """Capacity − VRAM still in use at target_time."""
        held = sum(t.vram_mb for t in self.running if t.expected_release_time > target_time)
        return self.capacity_vram_mb - held


@dataclass
class _Job:
    """In-queue job record (mirrors job_record_t subset).

    `expected_runtime` is the **deterministic upper bound** ``--time``
    (= workers.profile.yaml::runtime_max_sec = 2 × observed_max from NF
    trace.tsv).  Backfill literature convention F=2 (Tsafrir et al.
    TPDS 2007; Lawson-Smirni-Puiu Perf. Eval. 2005; Mu'alem-Feitelson
    TPDS 2001) — F=2 captures essentially all of EASY backfill's
    performance improvement vs perfect estimates while leaving slack
    for tail variance.  Slurm ``--time`` semantics: jobs declare an
    upper bound at submit, sched/backfill projects release using that
    bound, and a job exceeding it is killed by slurmd.
    """

    submission: TaskSubmission
    expected_runtime: float
    vram_required_mb: int
    ram_required_mb: int = 0

    @property
    def task_id(self) -> str:
        return self.submission.task_id

    @property
    def component(self) -> str:
        return self.submission.component

    @property
    def submit_time(self) -> float:
        return self.submission.submit_time


class SlurmBaselineScheduler:
    """Age-based priority queue + EASY backfill + reservation slots."""

    name = "slurm"

    def __init__(
        self,
        profile: WorkersProfile,
        gpu_capacity_mb: Dict[str, int],
        worker_name_format: str = "{component}-gpu{gpu_id}",
        age_factor: float = 1.0,
        fairshare_weight: float = 10.0,
        easy_backfill: bool = True,
        whole_gpu_exclusive: bool = False,
        total_host_ram_mb: int = 0,
        host_ram_min_available_mb: int = 0,
    ) -> None:
        self._profile = profile
        self._worker_name_format = worker_name_format
        self._age_factor = age_factor
        self._fairshare_weight = fairshare_weight
        self._easy_backfill = easy_backfill
        self._whole_gpu_exclusive = whole_gpu_exclusive
        self._total_host_ram_mb = int(total_host_ram_mb or 0)
        self._host_ram_min_available_mb = int(host_ram_min_available_mb or 0)
        self._host_requested_ram_mb = 0
        self._slots: Dict[str, _GpuSlot] = {
            gpu_id: _GpuSlot(gpu_id=gpu_id, capacity_vram_mb=cap)
            for gpu_id, cap in sorted(gpu_capacity_mb.items())
        }
        self._queue: Deque[_Job] = deque()
        self._reservations: List[ReservationSlot] = []
        self._inflight: Dict[str, _Job] = {}
        self._just_dispatched: Dict[str, _Job] = {}


    def submit(self, task: TaskSubmission) -> None:
        entry = self._profile.get(task.component)
        self._queue.append(_Job(
            submission=task,
            expected_runtime=entry.slurm_time_sec,
            vram_required_mb=entry.total_vram_mb,
            ram_required_mb=int(getattr(entry, "ram_max_mb", 0) or 0),
        ))

    def step(self, now: float) -> List[DispatchDecision]:
        """One scheduling cycle (`_attempt_backfill` epoch).

        1. Sort by Slurm priority (oldest job first).
        2. Try to start head-of-queue immediately (`_start_job`).
        3. If head can't start now, plan a reservation for it and try
           EASY backfill on the rest.
        """
        if not self._queue:
            return []

        self._reservations = [r for r in self._reservations if r.end_time > now]

        ordered = sorted(
            self._queue,
            key=lambda j: (-self._slurm_priority(j, now), j.submission.arrival_time, j.task_id),
        )
        self._queue = deque(ordered)

        decisions: List[DispatchDecision] = []
        head = self._queue[0]

        gpu_for_head = self._find_gpu_with_capacity_now(head, now)
        if gpu_for_head is not None:
            decisions.append(self._build_decision(head, gpu_for_head))
            self._queue.popleft()
            self._commit_binding(head, gpu_for_head, now)
            return decisions

        if not self._easy_backfill:
            return []

        head_resv = self._make_reservation_for_head(head, now)
        if head_resv is None:
            return []

        for job in list(self._queue)[1:]:
            gpu_for_bf = self._find_gpu_for_easy_backfill(job, head_resv, now)
            if gpu_for_bf is None:
                continue
            decisions.append(self._build_decision(job, gpu_for_bf))
            try:
                self._queue.remove(job)
            except ValueError:
                pass
            self._commit_binding(job, gpu_for_bf, now)

        return decisions

    def _commit_binding(self, job: "_Job", gpu_id: str, now: float) -> None:
        """F1 () — immediate-binding NodeInfo commit.

        Adds the dispatched task to ``slot.running`` and ``_inflight``
        synchronously within step(), mirroring Slurm's select_nodes
        post-success path running synchronously inside the scheduling
        cycle.  Idempotent — safe to call multiple times for the same
        task_id (the second call short-circuits via _inflight).
        """
        if job.task_id in self._inflight:
            return
        self._slots[gpu_id].running.append(_RunningTask(
            task_id=job.task_id,
            component=job.component,
            start_time=now,
            expected_release_time=now + job.expected_runtime,
            vram_mb=job.vram_required_mb,
            ram_mb=job.ram_required_mb,
        ))
        self._inflight[job.task_id] = job
        self._host_requested_ram_mb += max(0, int(job.ram_required_mb or 0))
        self._reservations = [r for r in self._reservations if r.job_id != job.task_id]

    def on_dispatch_success(self, task_id: str, gpu_id: str, now: float) -> None:
        """Job started → record running task.  Mirrors
        `select_nodes` post-success path (job_mgr.c::launch_job).
        """
        if task_id in self._inflight:
            return
        job = self._just_dispatched.pop(task_id, None)
        if job is None:
            return
        self._slots[gpu_id].running.append(_RunningTask(
            task_id=task_id,
            component=job.component,
            start_time=now,
            expected_release_time=now + job.expected_runtime,
            vram_mb=job.vram_required_mb,
            ram_mb=job.ram_required_mb,
        ))
        self._inflight[task_id] = job
        self._host_requested_ram_mb += max(0, int(job.ram_required_mb or 0))
        self._reservations = [r for r in self._reservations if r.job_id != task_id]

    def on_dispatch_failure(self, task_id: str, reason: str, now: float) -> None:
        """Bind error → remove from running set, caller re-submits."""
        job = self._inflight.pop(task_id, None)
        if job is not None:
            self._host_requested_ram_mb = max(
                0,
                self._host_requested_ram_mb - max(0, int(job.ram_required_mb or 0)),
            )
        self._just_dispatched.pop(task_id, None)
        for slot in self._slots.values():
            slot.running = [t for t in slot.running if t.task_id != task_id]
        self._reservations = [r for r in self._reservations if r.job_id != task_id]

    def on_task_complete(self, task_id: str, now: float) -> None:
        """Job finished → release VRAM."""
        job = self._inflight.pop(task_id, None)
        if job is not None:
            self._host_requested_ram_mb = max(
                0,
                self._host_requested_ram_mb - max(0, int(job.ram_required_mb or 0)),
            )
        for slot in self._slots.values():
            slot.running = [t for t in slot.running if t.task_id != task_id]
        self._reservations = [r for r in self._reservations if r.job_id != task_id]

    def describe_head_blockage(self, now: float) -> Optional[Dict[str, object]]:
        if not self._queue:
            return None
        ordered = sorted(
            self._queue,
            key=lambda j: (-self._slurm_priority(j, now), j.submission.arrival_time, j.task_id),
        )
        head = ordered[0]
        if self._find_gpu_with_capacity_now(head, now) is not None:
            return None
        shortfalls: Dict[str, int] = {}
        for gpu_id, slot in sorted(self._slots.items(), key=lambda item: int(item[0])):
            if self._whole_gpu_exclusive and len(slot.running) > 0:
                shortfall = max(1, head.vram_required_mb - slot.available_vram_mb)
            else:
                shortfall = max(0, head.vram_required_mb - slot.available_vram_mb)
            if shortfall > 0:
                shortfalls[gpu_id] = shortfall
        host_ram_shortfall = 0
        if not self._fits_host_ram(head):
            host_ram_shortfall = max(
                0,
                int(head.ram_required_mb or 0) - self._available_host_ram_mb(),
            )
        if not shortfalls and host_ram_shortfall <= 0:
            return None
        mode = (
            "reservation_blocked"
            if self._make_reservation_for_head(head, now) is not None
            else "immediate_fit_blocked"
        )
        self._reservations = [r for r in self._reservations if r.job_id != head.task_id]
        payload = {
            "task_id": head.task_id,
            "component": head.component,
            "mode": mode,
        }
        if shortfalls:
            payload["gpu_shortfalls_mb"] = shortfalls
        if host_ram_shortfall > 0:
            payload["host_ram_shortfall_mb"] = host_ram_shortfall
        return payload

    def debug_snapshot(self) -> Dict[str, object]:
        """Bounded read-only state for baseline live debugging."""
        import time
        now = time.time()
        try:
            blockage = self.describe_head_blockage(now)
        except Exception as exc:
            blockage = {"error": str(exc)}
        slots: Dict[str, object] = {}
        for gpu_id, slot in sorted(self._slots.items(), key=lambda item: int(item[0])):
            slots[gpu_id] = {
                "used_vram_mb": int(slot.used_vram_mb),
                "available_vram_mb": int(slot.available_vram_mb),
                "running": len(slot.running),
                "running_components": [
                    t.component for t in slot.running[:50]
                ],
            }
        pending_head = self._queue[0].task_id if self._queue else ""
        return {
            "name": self.name,
            "pending": len(self._queue),
            "pending_head": pending_head,
            "inflight": len(self._inflight),
            "just_dispatched": len(self._just_dispatched),
            "reservations": len(self._reservations),
            "host_requested_ram_mb": int(self._host_requested_ram_mb),
            "available_host_ram_mb": int(self._available_host_ram_mb()),
            "head_blockage": blockage,
            "slots": slots,
        }

    def on_worker_killed(self, component: str, gpu_ids: List[str]) -> None:
        """Worker container kill / evict → drain in-flight + drop reservations.

        Whole-GPU exclusive mode keeps no persistent residency state on
        the slot (each slot's ``running`` list is the only artefact of a
        live task), so on a guard-MEASURE OOM kill or external evict
        the next task on that GPU naturally pays the full cold-start
        cost via ``slurm_time_sec`` (= ``2 × observed_max(realtime)``,
        which already includes init + exec end-to-end).  This handler
        clears any in-flight bookkeeping for ``(component, gpu_id)``
        pairs so the scheduler doesn't keep ghost reservations for a
        dead worker.  Symmetric with the K8s baseline's
        ``on_worker_killed``.
        """
        gpu_set = {str(g) for g in gpu_ids}
        for slot in self._slots.values():
            if slot.gpu_id not in gpu_set:
                continue
            killed_ids = [
                t.task_id for t in slot.running if t.component == component
            ]
            slot.running = [
                t for t in slot.running
                if not (t.component == component and t.task_id in killed_ids)
            ]
            for tid in killed_ids:
                job = self._inflight.pop(tid, None)
                if job is not None:
                    self._host_requested_ram_mb = max(
                        0,
                        self._host_requested_ram_mb - max(0, int(job.ram_required_mb or 0)),
                    )
                self._just_dispatched.pop(tid, None)
        if killed_ids := [
            tid for tid, job in self._inflight.items()
            if job.submission.component == component
        ]:
            self._reservations = [
                r for r in self._reservations if r.job_id not in killed_ids
            ]

    def remember_dispatched(self, task: TaskSubmission) -> None:
        """Side-channel for reconstructing the dispatched job's runtime
        estimate at on_dispatch_success time (without keeping a
        scheduler-wide submission map).  Mirrors ``submit()`` —
        ``slurm_time_sec`` covers init + exec per Slurm ``--time``
        upstream semantics."""
        entry = self._profile.get(task.component)
        self._just_dispatched[task.task_id] = _Job(
            submission=task,
            expected_runtime=entry.slurm_time_sec,
            vram_required_mb=entry.total_vram_mb,
            ram_required_mb=int(getattr(entry, "ram_max_mb", 0) or 0),
        )


    def _slurm_priority(self, job: _Job, now: float) -> float:
        """Multifactor priority: ``age × W_age + fairshare × W_fair``.

        Port of slurm/src/plugins/priority/multifactor/
        priority_multifactor.c:2133-2148 weight_age * diff / max_age.

        We linearize age (no max_age cap — paper's claim is "older job →
        higher priority", cap matters only at week-scale timescales,
        default `priority_max_age=1 week`).  Fairshare term mirrors
        Slurm's ``PriorityWeightFairshare × fairshare_factor`` (real
        slurmctld uses Fair-Tree priority/multifactor formula); the
        baseline reads the per-task ``fairshare`` value populated by
        the gateway from the SBATCH_ACCOUNT/account routing path.
        Higher fairshare → higher priority — matches the slurmdbd setup
        (c1=300 / c2=200 / c3=100 in slurmdbd_setup.sh).  ``0.0``
        fairshare degrades to pure-age (legacy single-user).
        """
        age_term = self._age_factor * (now - job.submit_time)
        fair_term = self._fairshare_weight * float(
            getattr(job.submission, "fairshare", 0.0) or 0.0
        )
        return age_term + fair_term

    def _find_gpu_with_capacity_now(self, job: _Job, now: float) -> Optional[str]:
        """Equivalent of `_start_job` immediate-fit path.

        Returns the gpu_id with enough free VRAM to host `job` right
        now.  When ``whole_gpu_exclusive=True`` (matches `--gres=gpu:1`),
        only fully-idle slots are eligible regardless of VRAM headroom.
        Otherwise fall back to gres-shard sum-VRAM accounting.  Choose
        the GPU with most free VRAM among eligible (Slurm select/cons_tres
        spread bias); tie-break: lowest gpu_id.
        """
        if self._whole_gpu_exclusive:
            candidates = [
                slot for slot in self._slots.values()
                if len(slot.running) == 0
            ]
        else:
            candidates = [
                slot for slot in self._slots.values()
                if slot.available_vram_mb >= job.vram_required_mb
            ]
        if not candidates:
            return None
        if not self._fits_host_ram(job):
            return None
        end_time = now + job.expected_runtime
        free = [
            slot for slot in candidates
            if not self._has_blocking_reservation(slot.gpu_id, now, end_time, job)
        ]
        if not free:
            return None
        chosen = max(free, key=lambda s: (s.available_vram_mb, -int(s.gpu_id)))
        return chosen.gpu_id

    def _make_reservation_for_head(
        self, head: _Job, now: float,
    ) -> Optional[ReservationSlot]:
        """Find when/where head can start; create a ReservationSlot
        and append to the node_space.

        Picks the GPU whose earliest release time + free VRAM at that
        time admits head's request.  Mirrors backfill.c node_space
        update path where future avail_bitmap is computed by accounting
        for currently-running jobs' end times.
        """
        if not self._fits_host_ram(head):
            return None
        best: Optional[Tuple[str, float]] = None
        for slot in self._slots.values():
            release_time = slot.earliest_release_time(now)
            if release_time is None:
                continue
            projected = slot.projected_available_vram_at(release_time)
            if projected < head.vram_required_mb:
                continue
            end_at = release_time + head.expected_runtime
            if self._has_blocking_reservation(slot.gpu_id, release_time, end_at, head):
                continue
            if best is None or release_time < best[1]:
                best = (slot.gpu_id, release_time)

        if best is None:
            return None
        gpu_id, release_time = best
        resv = ReservationSlot(
            gpu_id=gpu_id,
            start_time=release_time,
            end_time=release_time + head.expected_runtime,
            vram_mb=head.vram_required_mb,
            job_id=head.task_id,
        )
        self._reservations.append(resv)
        return resv

    def _find_gpu_for_easy_backfill(
        self, job: _Job, head_resv: ReservationSlot, now: float,
    ) -> Optional[str]:
        """EASY backfill admission test.

        A lower-priority job may run NOW iff:
          (a) some GPU has enough VRAM right now, AND
          (b) its `[now, now + expected_runtime]` window does NOT
              push back any existing reservation (head_resv or any
              other).

        Mirrors `_attempt_backfill` per-pending-job inner loop +
        `_test_resv_overlap` (backfill.c:3496-3526).
        """
        end_time = now + job.expected_runtime

        if self._whole_gpu_exclusive:
            candidates = [
                slot for slot in self._slots.values()
                if len(slot.running) == 0
            ]
        else:
            candidates = [
                slot for slot in self._slots.values()
                if slot.available_vram_mb >= job.vram_required_mb
            ]
        if not candidates:
            return None
        if not self._fits_host_ram(job):
            return None

        feasible = []
        for slot in candidates:
            if self._has_blocking_reservation(slot.gpu_id, now, end_time, job):
                continue
            feasible.append(slot)
        if not feasible:
            return None

        chosen = max(feasible, key=lambda s: (s.available_vram_mb, -int(s.gpu_id)))
        return chosen.gpu_id

    def _available_host_ram_mb(self) -> int:
        if self._total_host_ram_mb <= 0:
            return 1 << 30
        safe_capacity = max(0, self._total_host_ram_mb - self._host_ram_min_available_mb)
        observed_available = max(0, _read_mem_available_mib() - self._host_ram_min_available_mb)
        external_used = max(
            0,
            safe_capacity - observed_available - max(0, self._host_requested_ram_mb),
        )
        return max(0, safe_capacity - external_used - self._host_requested_ram_mb)

    def _fits_host_ram(self, job: _Job) -> bool:
        needed = max(0, int(getattr(job, "ram_required_mb", 0) or 0))
        if needed <= 0:
            return True
        return needed <= self._available_host_ram_mb()

    def _has_blocking_reservation(
        self,
        gpu_id: str,
        candidate_start: float,
        candidate_end: float,
        candidate_job: _Job,
    ) -> bool:
        """Port of backfill.c:3496-3526 _test_resv_overlap.

        Returns True iff there exists a reservation on `gpu_id` whose
        time window overlaps [candidate_start, candidate_end] AND
        admitting `candidate_job` would force that reservation's
        resource demand into infeasibility.

        For our single-resource model: an overlap is blocking iff the
        reserved VRAM + the candidate's VRAM would exceed capacity at
        the reservation's start_time.
        """
        slot = self._slots[gpu_id]
        for resv in self._reservations:
            if resv.gpu_id != gpu_id:
                continue
            if resv.job_id == candidate_job.task_id:
                continue
            if not resv.overlaps(candidate_start, candidate_end):
                continue
            still_running = sum(
                t.vram_mb for t in slot.running
                if t.expected_release_time > resv.start_time
            )
            held_by_other_resv = sum(
                r.vram_mb for r in self._reservations
                if r.gpu_id == gpu_id
                and r.job_id != resv.job_id
                and r.job_id != candidate_job.task_id
                and r.start_time <= resv.start_time < r.end_time
            )
            candidate_held_at_resv = (
                candidate_job.vram_required_mb
                if candidate_end > resv.start_time
                else 0
            )
            need_for_resv = resv.vram_mb
            free_at_resv = (
                slot.capacity_vram_mb
                - still_running
                - held_by_other_resv
                - candidate_held_at_resv
            )
            if free_at_resv < need_for_resv:
                return True
        return False

    def _build_decision(self, job: _Job, gpu_id: str) -> DispatchDecision:
        worker_name = self._worker_name_format.format(
            component=job.component, gpu_id=gpu_id,
        )
        return DispatchDecision(
            task_id=job.task_id,
            target_gpu_id=gpu_id,
            target_worker_name=worker_name,
            component=job.component,
            is_dispatchable=True,
        )
