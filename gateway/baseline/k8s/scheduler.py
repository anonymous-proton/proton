"""
K8s-style baseline scheduler — algorithm-only port of vanilla
kube-scheduler v1.30.4's Filter → Score → Bind framework.

Upstream pin: kubernetes/kubernetes @ v1.30.4
              (commit upstream)
GPU resource semantics: public-project/gpushare-scheduler-extender
                        @ upstream

Per-block citations follow the form:
  Port of <repo>/<file>:<line-range> @ <tag>

Vanilla kube-scheduler **has no backfill** — bin-packing scores exist
but no lookahead reservation.  This gap is preserved verbatim (it IS
the paper's evidence: idle GPU gap accumulation).

Concurrent execution: the K8s scheduler natively supports multiple
pods on the same node (multiple TaskSubmission instances bound to the
same `(component, gpu_id)` worker).  VRAM accounting accumulates
through `_NodeInfo.requested_vram_mb` (mirrors k8s
`pkg/scheduler/framework/types.go::NodeInfo.Requested`).  This matches
upstream behaviour where `nodeInfo.Requested` is the sum of bound
pods' resource requests.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from gateway.baseline.shared import (
    DispatchDecision,
    TaskSubmission,
    WorkersProfile,
    WorkersProfileEntry,
)


_MAX_NODE_SCORE: int = 100


_PYTORCH_OVERHEAD_PER_TASK_MB: int = 800


def _read_mem_available_mib() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1 << 30


@dataclass
class _NodeInfo:
    """Per-GPU bookkeeping with PROTON model-residency accounting.

    Port of kubernetes/pkg/scheduler/framework/types.go::NodeInfo
    @ v1.30.4, ADAPTED FOR PROTON-INTERNAL USE: vanilla nf-k8s charges
    every pod its full ``ext.gpu_mib = ceil((weight + activation + 800)
    / 1024)`` because each pod brings its own model load (no residency).
    PROTON has persistent workers that load the weight once per
    (component, gpu_id) pair, so this scheduler ports the kube algorithm
    onto the proton residency model:

      requested_vram_mb = Σ_{component loaded here} weight_max_mb
                       + Σ_{concurrent infer in flight} activation_max_mb

    Field mapping:
      gpu_id                    ↔ k8s NodeInfo.node.Name
      capacity_vram_mb          ↔ k8s NodeInfo.Allocatable.<resource>
      loaded_components         — components whose weight is resident here.
                                  Charged once on first dispatch per
                                  (component, gpu).
      activation_count          — per-component count of in-flight
                                  inferences.  Charged activation_mb each.
      bound_tasks               ↔ k8s NodeInfo.Pods (which task → which
                                  component, for release accounting).
    """

    gpu_id: str
    capacity_vram_mb: int
    loaded_components: Dict[str, int] = field(default_factory=dict)
    activation_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    activation_mb: Dict[str, int] = field(default_factory=dict)
    bound_tasks: Dict[str, str] = field(default_factory=dict)

    @property
    def requested_vram_mb(self) -> int:
        weight = sum(self.loaded_components.values())
        active = sum(self.activation_count[c] * self.activation_mb.get(c, 0)
                     for c in self.activation_count)
        return weight + active

    @property
    def available_vram_mb(self) -> int:
        return self.capacity_vram_mb - self.requested_vram_mb

    def projected_request_for(self, entry: WorkersProfileEntry) -> int:
        """Hypothetical resource demand if a new task of ``entry``'s
        component were admitted here.  Adds activation_max_mb (+ a
        per-task overhead pad) always; adds weight_max_mb only if the
        component is not already resident.

        Per-task activation pad (``_PYTORCH_OVERHEAD_PER_TASK_MB``)
        compensates for the gap between ``workers.profile.yaml``'s
        observed peak activation and **PyTorch's reserved-memory
        bookkeeping** (caching allocator + cuDNN workspace + alignment
        fragmentation).  This overhead **scales with concurrent task
        count** — each running inference holds its own caching
        allocator state — so the pad is charged per activation rather
        than once at weight load time.  Without this, k8s baseline
        admits enough diffdock concurrent tasks to trigger CUDA OOM
        on RTX-3090 24 GiB GPUs (observed).
        """
        delta = entry.activation_max_mb + _PYTORCH_OVERHEAD_PER_TASK_MB
        if entry.component not in self.loaded_components:
            delta += entry.weight_max_mb
        return delta


class K8sBaselineScheduler:
    """Filter → Score → Bind chain over a fixed GPU set.

    Per cycle the scheduler pops a single task from the FIFO queue
    (vanilla kube-scheduler `pkg/scheduler/queue/scheduling_queue.go
    ::SchedulingQueue.Pop` returns one PodInfo at a time), runs the
    Filter plugin (NodeResourcesFit), scores feasible nodes
    (LeastRequestedPriority), and binds to the highest-scoring node
    (defaultbinder).
    """

    name = "k8s"

    def __init__(
        self,
        profile: WorkersProfile,
        gpu_capacity_mb: Dict[str, int],
        worker_name_format: str = "{component}-gpu{gpu_id}",
        total_host_ram_mb: int = 0,
        host_ram_min_available_mb: int = 0,
    ) -> None:
        self._profile = profile
        self._worker_name_format = worker_name_format
        self._total_host_ram_mb = int(total_host_ram_mb or 0)
        self._host_ram_min_available_mb = int(host_ram_min_available_mb or 0)
        self._host_requested_ram_mb = 0
        self._nodes: Dict[str, _NodeInfo] = {
            gpu_id: _NodeInfo(gpu_id=gpu_id, capacity_vram_mb=cap)
            for gpu_id, cap in sorted(gpu_capacity_mb.items())
        }
        self._pending: Deque[TaskSubmission] = deque()
        self._inflight: Dict[str, TaskSubmission] = {}
        self._just_dispatched: Dict[str, TaskSubmission] = {}


    def submit(self, task: TaskSubmission) -> None:
        self._pending.append(task)

    def step(self, now: float) -> List[DispatchDecision]:
        """One scheduling cycle.

        Vanilla kube-scheduler processes pods one-at-a-time
        (`pkg/scheduler/scheduler.go::scheduleOne`); each cycle pops
        the head of the priority queue, runs Filter+Score+Bind, then
        returns.  We mirror that here — even if multiple tasks are
        pending, only one dispatch decision is emitted per call.
        """
        if not self._pending:
            return []
        ordered = sorted(
            self._pending,
            key=lambda t: (-float(t.fairshare or 0.0), t.arrival_time, t.task_id),
        )
        self._pending = deque(ordered)
        head = self._pending[0]
        entry = self._profile.get(head.component)

        feasible = [
            n for n in self._nodes.values()
            if self._filter(n, entry) and self._fits_host_ram(entry)
        ]
        if not feasible:
            return []

        scored: List[Tuple[int, _NodeInfo]] = [
            (self._least_requested_score(n, entry), n) for n in feasible
        ]
        best_score = max(score for score, _ in scored)

        ties = [n for score, n in scored if score == best_score]
        chosen = min(ties, key=lambda n: int(n.gpu_id))

        self._pending.popleft()
        worker_name = self._worker_name_format.format(
            component=head.component, gpu_id=chosen.gpu_id
        )

        if head.component not in chosen.loaded_components:
            chosen.loaded_components[head.component] = entry.weight_max_mb
        chosen.activation_count[head.component] += 1
        chosen.activation_mb[head.component] = entry.activation_max_mb
        chosen.bound_tasks[head.task_id] = head.component
        self._host_requested_ram_mb += max(0, int(entry.ram_max_mb or 0))
        self._inflight[head.task_id] = head

        return [DispatchDecision(
            task_id=head.task_id,
            target_gpu_id=chosen.gpu_id,
            target_worker_name=worker_name,
            component=head.component,
            is_dispatchable=True,
        )]

    def on_dispatch_success(self, task_id: str, gpu_id: str, now: float) -> None:
        """Bind committed → update NodeInfo with PROTON residency model.

        Mirrors `pkg/scheduler/cache/cache.go::AddPod`, but updates
        loaded_components (weight charged once) and activation_count
        (per-task) instead of a single `Requested` counter.
        """
        if task_id in self._inflight:
            return
        sub = self._just_dispatched.pop(task_id, None)
        if sub is None:
            return
        node = self._nodes[gpu_id]
        entry = self._profile.get(sub.component)
        if sub.component not in node.loaded_components:
            node.loaded_components[sub.component] = entry.weight_max_mb
        node.activation_count[sub.component] += 1
        node.activation_mb[sub.component] = entry.activation_max_mb
        node.bound_tasks[task_id] = sub.component
        self._inflight[task_id] = sub

    def on_dispatch_failure(self, task_id: str, reason: str, now: float) -> None:
        """Bind error → caller re-enqueues; we release any partial state.

        Mirrors `pkg/scheduler/scheduler.go::schedulingCycle` post-bind
        error path which calls `recordSchedulingFailure` and the pod
        ends up back in the unschedulable queue.
        """
        sub = self._inflight.pop(task_id, None)
        self._just_dispatched.pop(task_id, None)
        if sub is not None:
            self._release_node(task_id, sub)

    def on_task_complete(self, task_id: str, now: float) -> None:
        """Pod terminated → kubelet PLEG → cache removes the pod.

        Mirrors `pkg/scheduler/cache/cache.go::RemovePod` which
        decrements NodeInfo.Requested by the pod's resource request.
        """
        sub = self._inflight.pop(task_id, None)
        if sub is not None:
            self._release_node(task_id, sub)

    def describe_head_blockage(self, now: float) -> Optional[Dict[str, object]]:
        _ = now
        if not self._pending:
            return None
        ordered = sorted(
            self._pending,
            key=lambda t: (-float(t.fairshare or 0.0), t.arrival_time, t.task_id),
        )
        head = ordered[0]
        entry = self._profile.get(head.component)
        feasible = [
            n for n in self._nodes.values()
            if self._filter(n, entry) and self._fits_host_ram(entry)
        ]
        if feasible:
            return None
        shortfalls: Dict[str, int] = {}
        for gpu_id, node in sorted(self._nodes.items(), key=lambda item: int(item[0])):
            delta = node.projected_request_for(entry)
            shortfall = max(0, delta - node.available_vram_mb)
            if shortfall > 0:
                shortfalls[gpu_id] = shortfall
        host_ram_shortfall = 0
        if not self._fits_host_ram(entry):
            host_ram_shortfall = max(
                0,
                int(entry.ram_max_mb or 0) - self._available_host_ram_mb(),
            )
        if not shortfalls and host_ram_shortfall <= 0:
            return None
        payload = {
            "task_id": head.task_id,
            "component": head.component,
        }
        if shortfalls:
            payload["gpu_shortfalls_mb"] = shortfalls
        if host_ram_shortfall > 0:
            payload["host_ram_shortfall_mb"] = host_ram_shortfall
        return payload

    def debug_snapshot(self) -> Dict[str, object]:
        """Bounded read-only state for baseline live debugging."""
        now = 0.0
        try:
            import time
            now = time.time()
        except Exception:
            pass
        try:
            blockage = self.describe_head_blockage(now)
        except Exception as exc:
            blockage = {"error": str(exc)}
        nodes: Dict[str, object] = {}
        for gpu_id, node in sorted(self._nodes.items(), key=lambda item: int(item[0])):
            nodes[gpu_id] = {
                "requested_vram_mb": int(node.requested_vram_mb),
                "available_vram_mb": int(node.available_vram_mb),
                "loaded_components": dict(node.loaded_components),
                "activation_count": dict(node.activation_count),
                "bound_tasks": len(node.bound_tasks),
            }
        pending_head = ""
        if self._pending:
            ordered = sorted(
                self._pending,
                key=lambda t: (-float(t.fairshare or 0.0), t.arrival_time, t.task_id),
            )
            pending_head = ordered[0].task_id
        return {
            "name": self.name,
            "pending": len(self._pending),
            "pending_head": pending_head,
            "inflight": len(self._inflight),
            "just_dispatched": len(self._just_dispatched),
            "host_requested_ram_mb": int(self._host_requested_ram_mb),
            "available_host_ram_mb": int(self._available_host_ram_mb()),
            "head_blockage": blockage,
            "nodes": nodes,
        }

    def on_worker_killed(self, component: str, gpu_ids: List[str]) -> None:
        """Worker container kill / evict → weight loss + re-init cost.

        kubelet OOM-kill / pod eviction (NodeMemoryPressure / preemption
        / manual delete) terminates the container holding the model
        weights.  The next pod to land on the same node must re-pull
        + re-load (cold-start cost re-incurred).  PROTON baselines
        model this by clearing the ``loaded_components`` entry so that
        ``projected_request_for(entry)`` charges weight_max_mb again
        on the next dispatch (matches ``on_dispatch_success`` first-
        time semantics).  Also drains in-flight bound tasks for the
        component on those GPUs (their next attempt re-binds after
        re-init).
        """
        for gpu_id in gpu_ids:
            node = self._nodes.get(str(gpu_id))
            if node is None:
                continue
            node.loaded_components.pop(component, None)
            killed = [
                tid for tid, comp in node.bound_tasks.items() if comp == component
            ]
            for tid in killed:
                node.bound_tasks.pop(tid, None)
                self._inflight.pop(tid, None)
                self._just_dispatched.pop(tid, None)
                try:
                    self._host_requested_ram_mb = max(
                        0,
                        self._host_requested_ram_mb
                        - max(0, int(self._profile.get(component).ram_max_mb or 0)),
                    )
                except Exception:
                    pass
            node.activation_count[component] = 0

    def remember_dispatched(self, task: TaskSubmission) -> None:
        """Side-channel — caller invokes between step() and
        on_dispatch_success so we can reconstruct the bound resource
        request without keeping every TaskSubmission's component
        in another global map."""
        self._just_dispatched[task.task_id] = task


    @staticmethod
    def _filter(node: _NodeInfo, entry: WorkersProfileEntry) -> bool:
        """Port of pkg/scheduler/framework/plugins/noderesources/fit.go:
        421-503 fitsRequest @ v1.30.4, ADAPTED FOR PROTON RESIDENCY:
        weight charged once per (component, gpu) (already loaded → 0),
        activation charged per task.  Hypothetical resource demand =
        ``node.projected_request_for(entry)``.

        The check is:
          projected_request > (capacity - currently_requested)
        """
        delta = node.projected_request_for(entry)
        if delta <= 0:
            return True
        return delta <= node.available_vram_mb

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

    def _fits_host_ram(self, entry: WorkersProfileEntry) -> bool:
        needed = max(0, int(getattr(entry, "ram_max_mb", 0) or 0))
        if needed <= 0:
            return True
        return needed <= self._available_host_ram_mb()

    @staticmethod
    def _least_requested_score(node: _NodeInfo, entry: WorkersProfileEntry) -> int:
        """Port of pkg/scheduler/framework/plugins/noderesources/
        least_allocated.go:30-61 leastResourceScorer + 52-61
        leastRequestedScore @ v1.30.4, ADAPTED FOR PROTON RESIDENCY.

        Score formula:
            post_request = currently_requested + delta(entry)
            score = ((capacity - post_request) * MaxNodeScore) / capacity
        Where ``delta(entry)`` adds activation_max only (weight already
        loaded) when the component is resident, else weight + activation.
        """
        if node.capacity_vram_mb == 0:
            return 0
        delta = node.projected_request_for(entry)
        post_request = node.requested_vram_mb + delta
        if post_request > node.capacity_vram_mb:
            return 0
        return ((node.capacity_vram_mb - post_request) * _MAX_NODE_SCORE) // node.capacity_vram_mb


    def _release_node(self, task_id: str, sub: TaskSubmission) -> None:
        """Release activation slot.  Weight stays loaded (residency
        invariant) — kube has no analogue to model unload, but PROTON
        workers keep weight resident across tasks until worker restart.
        """
        for node in self._nodes.values():
            if task_id in node.bound_tasks:
                comp = node.bound_tasks.pop(task_id)
                node.activation_count[comp] = max(0, node.activation_count[comp] - 1)
                self._host_requested_ram_mb = max(
                    0,
                    self._host_requested_ram_mb
                    - max(0, int(self._profile.get(comp).ram_max_mb or 0)),
                )
                return
