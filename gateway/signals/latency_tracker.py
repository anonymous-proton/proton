"""Per-worker latency tracking with co-location context.

Analogous to activation peak tracking (VRAM), this module tracks per-task
execution latency.  Each worker gets a ``WorkerLatencyTracker`` that records
task enter/exit events and produces ``LatencyObservation`` objects upon
completion.  Observations carry co-location context (which other components
shared the GPU during execution) so the interference model can learn pairwise
slowdown factors.

Key accuracy feature: ``solo_fraction`` tracks what fraction of a task's
lifetime was spent as the sole executor.  This lets the interference model
reject partially-concurrent observations that would contaminate baselines.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import ReciprocalInterferenceQuery


@dataclass(frozen=True)
class LatencyTaskIdentity:
    """Task-instance identity retained in every GPU composition segment."""

    task_id: str
    component: str
    config_fingerprint: str = ""
    input_fingerprint: str = ""
    gpu_ids: tuple[str, ...] = ()
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""


@dataclass(frozen=True)
class LatencyCompositionSegment:
    start_mono: float
    end_mono: float
    neighbors: tuple[LatencyTaskIdentity, ...] = ()

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_mono - self.start_mono)


@dataclass
class LatencySlot:
    """Tracks one executing task's latency state on a worker."""

    task_id: str
    component: str
    registry_key: tuple[str, str, str]
    workload_features: dict[str, Any]
    config_fingerprint: str

    start_mono: float
    start_wall: float

    co_located_components: frozenset[str]
    co_located_task_count: int

    campaign_id: str = ""

    gpu_ids: list[str] = field(default_factory=list)
    input_fingerprint: str = ""
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""

    max_concurrent: int = 1
    was_ever_solo: bool = False

    _solo_time_acc: float = 0.0
    _last_transition_mono: float = 0.0
    _currently_solo: bool = False
    _segment_start_mono: float = 0.0
    _current_neighbors: tuple[LatencyTaskIdentity, ...] = ()
    segments: list[LatencyCompositionSegment] = field(default_factory=list)

    def identity(self) -> LatencyTaskIdentity:
        return LatencyTaskIdentity(
            task_id=self.task_id,
            component=self.component,
            config_fingerprint=self.config_fingerprint,
            input_fingerprint=self.input_fingerprint,
            gpu_ids=tuple(self.gpu_ids),
            gpu_model=self.gpu_model,
            mps_mode=self.mps_mode,
            worker_backend=self.worker_backend,
            actor_model=self.actor_model,
            adapter_version=self.adapter_version,
        )


@dataclass
class LatencyObservation:
    """Completed latency observation ready for recording."""

    task_id: str
    component: str
    registry_key: tuple[str, str, str]
    workload_features: dict[str, Any]
    config_fingerprint: str

    duration_sec: float

    was_solo_throughout: bool
    was_ever_solo: bool
    co_located_components: frozenset[str]
    max_concurrent: int

    campaign_id: str = ""
    solo_fraction: float = 0.0

    gpu_ids: list[str] = field(default_factory=list)
    input_fingerprint: str = ""
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""
    composition_segments: tuple[LatencyCompositionSegment, ...] = ()
    correction_provenance: tuple[str, ...] = ()
    correction_applied: bool = False
    direct_slowdown_factor: float | None = None
    direct_uncertainty: float = 0.0
    direct_query: ReciprocalInterferenceQuery | None = None


class WorkerLatencyTracker:
    """Manages per-task latency tracking for one worker.

    Thread-safety: all methods are called from the asyncio event loop
    (single-threaded), so no locking is needed.

    Solo fraction tracking: every time the concurrency level changes
    (task enters or exits), we snapshot the elapsed time and accumulate
    solo time for each slot that was the sole executor.
    """

    __slots__ = ("_slots",)

    def __init__(self) -> None:
        self._slots: dict[str, LatencySlot] = {}

    @property
    def active_count(self) -> int:
        return len(self._slots)

    @property
    def active_components(self) -> frozenset[str]:
        return frozenset(s.component for s in self._slots.values())

    def active_components_on_gpus(
        self, gpu_ids: Sequence[str] | None = None
    ) -> frozenset[str]:
        """Components of live slots, optionally scoped to a GPU set.

        ``gpu_ids`` empty/None keeps the legacy whole-worker view.
        A slot counts only when its GPU set intersects the requested
        GPUs — a live task on another GPU of a multi-GPU worker must not
        be reported as a co-location neighbor for this candidate.
        """
        if not gpu_ids:
            return self.active_components
        wanted = {str(item) for item in gpu_ids if str(item)}
        if not wanted:
            return self.active_components
        return frozenset(
            slot.component
            for slot in self._slots.values()
            if wanted & {str(g) for g in slot.gpu_ids}
        )

    def active_component_concurrency(
        self,
        component: str,
        gpu_ids: Sequence[str] | None = None,
    ) -> int:
        """Count live slots of *component*, optionally GPU-scoped.

        Used as the live-incumbent part of the projected self-interference
        concurrency degree N (the candidate adds +1 at the call site).
        """
        if not gpu_ids:
            return sum(1 for s in self._slots.values() if s.component == component)
        wanted = {str(item) for item in gpu_ids if str(item)}
        if not wanted:
            return sum(1 for s in self._slots.values() if s.component == component)
        return sum(
            1
            for s in self._slots.values()
            if s.component == component and wanted & {str(g) for g in s.gpu_ids}
        )

    def on_task_enter(
        self,
        *,
        task_id: str,
        component: str,
        registry_key: tuple[str, str, str],
        workload_features: dict[str, Any],
        config_fingerprint: str,
        campaign_id: str = "",
        gpu_co_located: frozenset[str] = frozenset(),
        gpu_ids: Sequence[str] | None = None,
        gpu_id: str | None = None,
        input_fingerprint: str = "",
        gpu_model: str = "",
        mps_mode: str = "",
        worker_backend: str = "",
        actor_model: str = "",
        adapter_version: str = "",
        co_located_tasks: Sequence[LatencyTaskIdentity] | None = None,
        now_mono: float | None = None,
    ) -> None:
        """Record a task start; duplicate IDs and GPU aliases fail closed."""
        if task_id in self._slots:
            raise ValueError(f"duplicate active latency task_id: {task_id}")
        normalized_gpu_ids = [str(item) for item in (gpu_ids or ())]
        if len(normalized_gpu_ids) != len(set(normalized_gpu_ids)):
            raise ValueError(
                f"duplicate gpu_ids for task {task_id}: {normalized_gpu_ids}"
            )
        if gpu_id is not None:
            if normalized_gpu_ids and normalized_gpu_ids != [str(gpu_id)]:
                raise ValueError(
                    f"gpu_id/gpu_ids mismatch for task {task_id}: "
                    f"{gpu_id!r} vs {normalized_gpu_ids!r}"
                )
            normalized_gpu_ids = [str(gpu_id)]
        now_mono = time.monotonic() if now_mono is None else now_mono

        self._snapshot_solo_time(now_mono)
        worker_peers = tuple(s.identity() for s in self._slots.values())
        if co_located_tasks is None:
            synthetic = tuple(
                LatencyTaskIdentity(task_id="", component=component)
                for component in sorted(gpu_co_located)
                if component not in {peer.component for peer in worker_peers}
            )
            peers = worker_peers + synthetic
        else:
            peers = tuple(co_located_tasks)
        co_located = frozenset(peer.component for peer in peers)
        n_others = len(peers)
        is_solo = n_others == 0

        slot = LatencySlot(
            task_id=task_id,
            component=component,
            registry_key=registry_key,
            workload_features=dict(workload_features),
            config_fingerprint=config_fingerprint,
            campaign_id=campaign_id,
            gpu_ids=normalized_gpu_ids,
            input_fingerprint=str(input_fingerprint or registry_key[2] or ""),
            gpu_model=str(gpu_model or ""),
            mps_mode=str(mps_mode or ""),
            worker_backend=str(worker_backend or ""),
            actor_model=str(actor_model or ""),
            adapter_version=str(adapter_version or ""),
            start_mono=now_mono,
            start_wall=time.time(),
            co_located_components=co_located,
            co_located_task_count=n_others,
            max_concurrent=n_others + 1,
            was_ever_solo=is_solo,
            _solo_time_acc=0.0,
            _last_transition_mono=now_mono,
            _currently_solo=is_solo,
            _segment_start_mono=now_mono,
            _current_neighbors=peers,
        )
        self._slots[task_id] = slot
        self._transition_all(now_mono)
        self.set_task_neighbors(task_id, peers, now_mono=now_mono)

    def on_task_exit(
        self, task_id: str, *, now_mono: float | None = None
    ) -> LatencyObservation | None:
        """Record a task completing execution. Returns the observation or None."""
        slot = self._slots.get(task_id)
        if slot is None:
            return None

        now_mono = time.monotonic() if now_mono is None else now_mono
        self._close_segment(slot, now_mono)
        self._slots.pop(task_id)
        duration = now_mono - slot.start_mono

        if slot._currently_solo:
            slot._solo_time_acc += now_mono - slot._last_transition_mono

        self._transition_all(now_mono)

        was_solo_throughout = slot.was_ever_solo and slot.max_concurrent == 1
        solo_fraction = (slot._solo_time_acc / duration) if duration > 0 else 0.0

        exit_co_located = frozenset(s.component for s in self._slots.values())
        merged_co_located = slot.co_located_components | exit_co_located

        return LatencyObservation(
            task_id=slot.task_id,
            component=slot.component,
            registry_key=slot.registry_key,
            workload_features=slot.workload_features,
            config_fingerprint=slot.config_fingerprint,
            campaign_id=slot.campaign_id,
            gpu_ids=slot.gpu_ids,
            input_fingerprint=slot.input_fingerprint,
            gpu_model=slot.gpu_model,
            mps_mode=slot.mps_mode,
            worker_backend=slot.worker_backend,
            actor_model=slot.actor_model,
            adapter_version=slot.adapter_version,
            composition_segments=tuple(slot.segments),
            duration_sec=duration,
            was_solo_throughout=was_solo_throughout,
            was_ever_solo=slot.was_ever_solo,
            co_located_components=merged_co_located,
            max_concurrent=slot.max_concurrent,
            solo_fraction=round(min(1.0, max(0.0, solo_fraction)), 4),
        )

    def set_task_neighbors(
        self,
        task_id: str,
        neighbors: Sequence[LatencyTaskIdentity],
        *,
        now_mono: float | None = None,
    ) -> None:
        """Close the prior segment and start the supplied composition."""
        slot = self._slots.get(task_id)
        if slot is None:
            raise KeyError(f"unknown active latency task_id: {task_id}")
        now = time.monotonic() if now_mono is None else now_mono
        normalized = tuple(
            sorted(neighbors, key=lambda item: (item.task_id, item.component))
        )
        if normalized == slot._current_neighbors:
            return
        self._close_segment(slot, now)
        slot._segment_start_mono = now
        slot._current_neighbors = normalized
        slot.co_located_components = slot.co_located_components | frozenset(
            item.component for item in normalized
        )
        slot.max_concurrent = max(slot.max_concurrent, len(normalized) + 1)
        slot._currently_solo = not normalized
        if not normalized:
            slot.was_ever_solo = True
        slot._last_transition_mono = now

    def drop_task(self, task_id: str, *, now_mono: float | None = None) -> bool:
        """Remove a cancelled slot while still closing survivor segments."""
        slot = self._slots.get(task_id)
        if slot is None:
            return False
        now = time.monotonic() if now_mono is None else now_mono
        self._close_segment(slot, now)
        self._slots.pop(task_id)
        self._transition_all(now)
        return True

    def clear(self) -> None:
        """Clear all slots (e.g. on worker restart)."""
        self._slots.clear()

    def _transition_all(self, now_mono: float) -> None:
        identities = {task_id: slot.identity() for task_id, slot in self._slots.items()}
        for task_id in tuple(self._slots):
            self.set_task_neighbors(
                task_id,
                [
                    identity
                    for peer_id, identity in identities.items()
                    if peer_id != task_id
                ],
                now_mono=now_mono,
            )

    @staticmethod
    def _close_segment(slot: LatencySlot, now_mono: float) -> None:
        if now_mono < slot._segment_start_mono:
            raise ValueError("latency segment time moved backwards")
        if now_mono > slot._segment_start_mono:
            slot.segments.append(
                LatencyCompositionSegment(
                    start_mono=slot._segment_start_mono,
                    end_mono=now_mono,
                    neighbors=slot._current_neighbors,
                )
            )

    def _snapshot_solo_time(self, now_mono: float) -> None:
        """Accumulate solo time for any slot that is currently the sole executor."""
        for s in self._slots.values():
            if s._currently_solo:
                s._solo_time_acc += now_mono - s._last_transition_mono
            s._last_transition_mono = now_mono


__all__ = [
    "LatencyCompositionSegment",
    "LatencyObservation",
    "LatencySlot",
    "LatencyTaskIdentity",
    "WorkerLatencyTracker",
]
