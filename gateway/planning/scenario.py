"""GPU Timeline: projected per-GPU usage model for lookahead scheduling.

Each GPU maintains a timeline of running and projected tasks.  The
``CampaignScheduler`` uses these timelines to answer questions like:

- "When will GPU g have enough free VRAM for task t?"
- "If I place task t on GPU g now, when will it finish?"
- "Which GPU will have the most free VRAM after currently running tasks complete?"

Predictions come from the per-(component, gpu_id) GPEstimate in SignalService.
When predictions are invalidated by a drift event, affected timeline entries
are marked stale and reverted to conservative bounds.

See <docs>  for design rationale.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any

from .vram_reservation import (
    VramReservation,
    VramReservationModel,
    create_vram_reservation_model,
)

_LOG = logging.getLogger(__name__)


def _profile_interval_fit(name: str):
    """Accumulate wall-time + call count for an interval-fit method.

    Temporal mode's planner_solve is dominated by these; this profiles
    them so the breakdown surfaces in the log (periodic every 5000 calls).
    """

    def deco(fn):
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                prof = getattr(self, "_interval_fit_prof", None)
                if prof is not None and name in prof:
                    prof[name]["wall"] += time.perf_counter() - t0
                    prof[name]["calls"] += 1
                    if prof[name]["calls"] % 5000 == 0:
                        _LOG.warning(
                            "[interval-fit-profile] %s calls=%d wall=%.2fs",
                            name,
                            prof[name]["calls"],
                            prof[name]["wall"],
                        )

        return wrapper

    return deco


TEMPORAL_PEAK_MARGIN_RATIO_MEAN = 0.3247337736552777
TEMPORAL_PEAK_MARGIN_RATIO_STD = 0.5406760084212469

_VRAM_FREEING_METRICS = frozenset({"vram", "interference_vram"})


def _checked_float(value: Any) -> float:
    """Preserve strict conversion while making the failure boundary explicit."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise


def _checked_int(value: Any) -> int:
    """Preserve strict conversion while making the failure boundary explicit."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise


def _temporal_peak_margin_sec(
    peak_start_time: float,
    peak_end_time: float,
    resource_z: float,
) -> float:
    peak_duration = max(0.0, peak_end_time - peak_start_time)
    ratio = TEMPORAL_PEAK_MARGIN_RATIO_MEAN + max(0.0, resource_z) * (
        TEMPORAL_PEAK_MARGIN_RATIO_STD
    )
    return peak_duration * ratio


def _range_max_used(
    prefix: tuple[float, ...],
    boundaries: tuple[float, ...],
    lo: float,
    hi: float,
) -> float:
    """Max incumbent usage over [lo, hi) from a sorted prefix-sum."""
    if lo >= hi:
        return 0.0
    idx = bisect_right(boundaries, lo) - 1
    best = prefix[idx] if idx >= 0 else 0.0
    k_lo = bisect_right(boundaries, lo)
    k_hi = bisect_left(boundaries, hi)
    if k_lo < k_hi:
        window = prefix[k_lo:k_hi]
        if window:
            best = max(best, max(window))
    return best


def _merge_sorted_boundaries(
    cached: tuple[tuple[float, str], ...],
    new: tuple[tuple[float, str], ...],
) -> tuple[tuple[float, str], ...]:
    """Insert sorted *new* boundaries into sorted *cached* without re-sorting."""
    if not new:
        return cached
    merged = list(cached)
    for boundary in new:
        idx = bisect_left(merged, boundary)
        if idx >= len(merged) or merged[idx] != boundary:
            merged.insert(idx, boundary)
    return tuple(merged)


@dataclass(frozen=True)
class _IncumbentInterval:
    start_time: float
    end_time: float | None
    amount_mb: float
    task_id: str
    is_predicted: bool


@dataclass(frozen=True)
class _ScenarioEventIndex:
    generation: tuple[Any, ...]
    vram_by_gpu: dict[str, tuple[_IncumbentInterval, ...]]
    ram_intervals: tuple[_IncumbentInterval, ...]
    candidate_boundaries: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class _IncumbentSweep:
    generation: tuple[Any, ...]
    gpu_id: str
    vram_boundaries: tuple[float, ...]
    vram_prefix: tuple[float, ...]
    ram_boundaries: tuple[float, ...]
    ram_prefix: tuple[float, ...]
    predicted_task_ids: frozenset[str]


@dataclass
class TimelineEntry:
    """A single task occupying GPU time and VRAM."""

    task_id: str
    component: str
    campaign_id: str
    gpu_id: str
    start_time: float
    predicted_end_time: float
    predicted_vram_mb: float
    allow_vram_mb: float | None = None
    peak_start_time: float | None = None
    peak_end_time: float | None = None
    launch_uncertainty_sec: float = field(default=0.0, repr=False)
    predicted_ram_mb: float = 0.0
    is_backfill: bool = False
    is_predicted: bool = False
    is_init: bool = False
    prediction_stale: bool = False
    is_invalidated: bool = (
        False
    )
    deadline_uncertain: bool = False
    input_size: float = 0.0
    config_fingerprint: str = ""
    logical_batch_size: int = 0
    execution_batch_size: int = 0
    completed_at: float | None = (
        None
    )
    killed_at: float | None = None
    worker_name: str = ""
    active_cancel_safe: bool = False
    is_dispatching: bool = False
    was_primary_at_dispatch: bool = False
    is_evict_masked: bool = False
    execution_attempt_id: str = ""
    total_base_work_sec: float | None = None
    remaining_base_work_sec: float | None = None
    last_accounted_mono: float | None = None
    current_reciprocal_multiplier: float = 1.0
    reciprocal_evidence_epoch: int = 0
    reciprocal_cancel_at: float | None = None
    input_fingerprint: str = ""
    gpu_model: str = ""
    mps_mode: str = ""
    worker_backend: str = ""
    actor_model: str = ""
    adapter_version: str = ""

    @property
    def reciprocal_base_duration_sec(self) -> float | None:
        """Compatibility alias for the single authoritative base-work total."""
        return self.total_base_work_sec

    @reciprocal_base_duration_sec.setter
    def reciprocal_base_duration_sec(self, value: float | None) -> None:
        previous_total = self.total_base_work_sec
        self.total_base_work_sec = (
            None if value is None else max(0.0, _checked_float(value))
        )
        if value is None:
            self.remaining_base_work_sec = None
        elif self.remaining_base_work_sec is None or (
            self.last_accounted_mono is None
            and self.remaining_base_work_sec == previous_total
        ):
            self.remaining_base_work_sec = self.total_base_work_sec

    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None or self.killed_at is not None

    @property
    def is_killed(self) -> bool:
        return self.killed_at is not None

    def _has_temporal_reservation_geometry(self) -> bool:
        allow = self.allow_vram_mb
        peak_start = self.peak_start_time
        peak_end = self.peak_end_time
        return bool(
            allow is not None
            and peak_start is not None
            and peak_end is not None
            and self.start_time <= peak_start < peak_end
            and allow >= 0
            and allow < self.predicted_vram_mb
        )

    def _has_valid_temporal_reservation(self) -> bool:
        return bool(
            not self.is_completed
            and self._has_temporal_reservation_geometry()
            and self.peak_end_time is not None
            and self.peak_end_time <= self.predicted_end_time
        )

    def vram_reservation_end_time(self) -> float:
        return self.predicted_end_time

    def temporal_peak_reservation_start_time(self) -> float | None:
        if not self._has_valid_temporal_reservation():
            return None
        assert self.peak_start_time is not None
        return self.peak_start_time

    def temporal_peak_reservation_end_time(self) -> float | None:
        if not self._has_valid_temporal_reservation():
            return None
        assert self.peak_end_time is not None
        if not self.is_predicted:
            return self.peak_end_time
        return min(
            self.predicted_end_time,
            self.peak_end_time + max(0.0, _checked_float(self.launch_uncertainty_sec)),
        )

    def reserved_vram_at(self, at_time: float) -> float:
        """Return this entry's step-function VRAM reservation."""
        if not self._has_valid_temporal_reservation():
            return self.predicted_vram_mb
        assert self.allow_vram_mb is not None
        peak_start = self.temporal_peak_reservation_start_time()
        peak_end = self.temporal_peak_reservation_end_time()
        assert peak_start is not None and peak_end is not None
        return (
            self.predicted_vram_mb
            if peak_start <= at_time < peak_end
            else self.allow_vram_mb
        )

    @property
    def predicted_duration(self) -> float:
        return max(0.0, self.predicted_end_time - self.start_time)

    @property
    def elapsed(self) -> float:
        end = self.killed_at or self.completed_at
        if end is not None:
            return max(0.0, end - self.start_time)
        return max(0.0, time.time() - self.start_time)

    @property
    def remaining(self) -> float:
        if self.completed_at is not None or self.killed_at is not None:
            return 0.0
        return max(0.0, self.predicted_end_time - time.time())

    def as_dict(self) -> dict[str, Any]:
        temporal_geometry = self._has_temporal_reservation_geometry()
        return {
            "task_id": self.task_id,
            "component": self.component,
            "campaign_id": self.campaign_id,
            "gpu_id": self.gpu_id,
            "start_time": round(self.start_time, 3),
            "predicted_end_time": round(self.predicted_end_time, 3),
            "predicted_vram_mb": round(self.predicted_vram_mb, 1),
            "allow_vram_mb": (
                round(self.allow_vram_mb, 1)
                if temporal_geometry and self.allow_vram_mb is not None
                else None
            ),
            "peak_start_time": (self.peak_start_time if temporal_geometry else None),
            "peak_end_time": self.peak_end_time if temporal_geometry else None,
            "predicted_ram_mb": round(self.predicted_ram_mb, 1),
            "is_backfill": self.is_backfill,
            "is_predicted": self.is_predicted,
            "is_init": self.is_init,
            "prediction_stale": self.prediction_stale,
            "is_invalidated": self.is_invalidated,
            "completed": self.is_completed,
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "is_killed": self.is_killed,
            "killed_at": round(self.killed_at, 3) if self.killed_at else None,
            "elapsed_sec": round(self.elapsed, 2),
            "remaining_sec": round(self.remaining, 2),
            "worker_name": self.worker_name,
            "logical_batch_size": self.logical_batch_size,
            "execution_batch_size": self.execution_batch_size,
            "active_cancel_safe": bool(self.active_cancel_safe),
            "execution_attempt_id": self.execution_attempt_id,
            "total_base_work_sec": self.total_base_work_sec,
            "remaining_base_work_sec": self.remaining_base_work_sec,
            "current_reciprocal_multiplier": self.current_reciprocal_multiplier,
            "reciprocal_evidence_epoch": self.reciprocal_evidence_epoch,
            "reciprocal_cancel_at": self.reciprocal_cancel_at,
        }


class GpuTimeline:
    """Projected timeline for a single GPU.

    Maintains a list of ``TimelineEntry`` objects representing currently
    running and projected tasks.  Entries are removed when tasks complete
    (via ``remove``) or when predictions are invalidated (projected entries
    removed by ``invalidate_predictions``).
    """

    def __init__(self, gpu_id: str, total_vram_mb: float = 0.0) -> None:
        self.gpu_id = gpu_id
        self.total_vram_mb = total_vram_mb
        self.reserved_weights_mb: float = 0.0
        self.idle_weights_mb: float = 0.0
        self._entries: list[TimelineEntry] = []
        self._retired_entries: list[TimelineEntry] = []
        self._entries_by_component: dict[str, list[TimelineEntry]] = {}
        self._entry_ref_ids: set[int] = set()
        self._component_versions: dict[str, int] = {}
        self._component_invalidated_versions: dict[str, int] = {}
        self._state_version: int = 0
        self._vram_at_cache: dict[tuple[int, int], float] = {}
        self._temporal_boundary_buckets: set[int] = set()
        self._active_entries_cache: list[TimelineEntry] | None = None

    def _bump_state_version(self, added_entry: TimelineEntry | None = None) -> None:
        """Invalidate timeline caches, extending boundaries for a pure append."""
        self._state_version += 1
        self._vram_at_cache.clear()
        self._active_entries_cache = None
        if added_entry is not None:
            if added_entry._has_valid_temporal_reservation():
                self._temporal_boundary_buckets.update(
                    _checked_int(boundary)
                    for boundary in (
                        added_entry.start_time,
                        added_entry.temporal_peak_reservation_start_time(),
                        added_entry.temporal_peak_reservation_end_time(),
                        added_entry.vram_reservation_end_time(),
                    )
                    if boundary is not None
                )
            return
        self._temporal_boundary_buckets = {
            _checked_int(boundary)
            for entry in self._entries
            if entry._has_valid_temporal_reservation()
            for boundary in (
                entry.start_time,
                entry.temporal_peak_reservation_start_time(),
                entry.temporal_peak_reservation_end_time(),
                entry.vram_reservation_end_time(),
            )
            if boundary is not None
        }

    def _touch_component(self, component: str) -> None:
        self._component_versions[component] = (
            self._component_versions.get(component, 0) + 1
        )

    def _index_entry(self, entry: TimelineEntry) -> None:
        self._entries_by_component.setdefault(entry.component, []).append(entry)
        self._entry_ref_ids.add(id(entry))

    def _unindex_entry(self, entry: TimelineEntry) -> None:
        self._entry_ref_ids.discard(id(entry))
        component_entries = self._entries_by_component.get(entry.component)
        if component_entries is not None:
            with contextlib.suppress(ValueError):
                component_entries.remove(entry)
            if not component_entries:
                self._entries_by_component.pop(entry.component, None)
        self._touch_component(entry.component)

    @staticmethod
    def _is_active_index_entry(entry: TimelineEntry) -> bool:
        return not entry.is_completed and not getattr(entry, "is_invalidated", False)

    def _rebuild_component_index(self) -> None:
        self._entries_by_component.clear()
        self._entry_ref_ids.clear()
        for entry in self._entries:
            self._index_entry(entry)
        for component in self._entries_by_component:
            self._touch_component(component)

    def _ensure_entry_indexes_current(self) -> None:
        """Repair indexes if a test/legacy path mutates ``_entries`` directly."""
        if len(self._entry_ref_ids) != len(self._entries):
            self._rebuild_component_index()

    @property
    def entries(self) -> list[TimelineEntry]:
        return list(self._entries) + list(self._retired_entries)

    @property
    def active_entries(self) -> list[TimelineEntry]:
        """Only entries that are still running (not completed)."""
        cached = self._active_entries_cache
        if cached is not None and len(self._entry_ref_ids) == len(self._entries):
            return cached
        self._ensure_entry_indexes_current()
        cached = [e for e in self._entries if self._is_active_index_entry(e)]
        self._active_entries_cache = cached
        return cached

    @property
    def active_count(self) -> int:
        return len(self.active_entries)

    @staticmethod
    def is_planned_occupancy_entry(
        entry: TimelineEntry,
        at_time: float | None = None,
    ) -> bool:
        """True when an entry occupies planned GPU capacity.

        Planned occupancy is the scheduler's projection contract: predicted
        entries occupy their projected window, while actual dispatched entries
        occupy capacity until an explicit completion/kill event.  Actual tasks
        must not disappear just because their predicted end time was too
        optimistic.
        """
        t = time.time() if at_time is None else _checked_float(at_time)
        if (
            entry.is_completed
            or getattr(entry, "is_invalidated", False)
            or getattr(entry, "is_evict_masked", False)
        ):
            return False
        if not getattr(entry, "is_predicted", False):
            return True
        return (
            _checked_float(getattr(entry, "start_time", 0.0) or 0.0)
            <= t
            < _checked_float(getattr(entry, "predicted_end_time", 0.0) or 0.0)
        )

    @staticmethod
    def _is_vram_reservation_entry(
        entry: TimelineEntry,
        at_time: float,
    ) -> bool:
        if entry.is_completed or getattr(entry, "is_evict_masked", False):
            return False
        if not entry.is_predicted:
            return True
        return entry.vram_reservation_end_time() > at_time

    def planned_occupancy_entries(
        self,
        at_time: float | None = None,
    ) -> list[TimelineEntry]:
        """Entries that count toward planned occupancy at *at_time*."""
        t = time.time() if at_time is None else _checked_float(at_time)
        return [e for e in self._entries if self.is_planned_occupancy_entry(e, t)]

    def planned_occupancy_count(self, at_time: float | None = None) -> int:
        return len(self.planned_occupancy_entries(at_time))


    def add(self, entry: TimelineEntry) -> None:
        """Record a task placement on this GPU."""
        if entry.is_completed:
            self._retired_entries.append(entry)
            self._bump_state_version()
            return
        self._entries.append(entry)
        self._index_entry(entry)
        self._touch_component(entry.component)
        self._bump_state_version(added_entry=entry)

    def _retire_entry(self, entry: TimelineEntry) -> None:
        """Move a terminal entry out of scheduler state into ops history."""
        try:
            self._entries.remove(entry)
        except ValueError:
            return
        self._unindex_entry(entry)
        self._retired_entries.append(entry)
        self._bump_state_version()

    def complete(self, task_id: str) -> TimelineEntry | None:
        """Mark a task as completed — keeps it in timeline for visual history.

        Completed entries stay for the full gateway session (the ops
        dashboard needs run history for the timeline chart).
        If the entry was already killed, skip (don't overwrite killed state).
        Predicted (projected) entries are never completed — they represent
        future plans, not actual work.  Completing them would prevent
        ``prune_stale_predicted`` from cleaning them up.

        With exec_concurrency > 1, a killed entry and an active entry
        for the same task_id can coexist (from eviction + re-dispatch).
        We must skip killed entries and complete the ACTIVE one.
        """
        active_match: TimelineEntry | None = None
        for e in self._entries:
            if e.task_id == task_id:
                if e.is_predicted:
                    continue
                if e.killed_at is not None:
                    continue
                if e.completed_at is not None:
                    continue
                active_match = e
                break
        if active_match is not None:
            active_match.completed_at = time.time()
            active_match.predicted_end_time = active_match.completed_at
            self._retire_entry(active_match)
            return active_match
        for e in self._retired_entries:
            if e.task_id == task_id and not e.is_predicted:
                if e.killed_at is not None and e.completed_at is None:
                    e.completed_at = time.time()
                    e.predicted_end_time = e.killed_at
                    self._bump_state_version()
                return e
        return None

    def kill(self, task_id: str) -> TimelineEntry | None:
        """Mark a task as killed (worker guard-killed mid-task).

        Visually distinct from completed — the task did not finish normally.
        Prefers active (non-killed, non-completed) entries when multiple
        entries with the same task_id exist (eviction + re-dispatch).
        """
        for e in self._entries:
            if e.task_id == task_id and e.killed_at is None and e.completed_at is None:
                e.killed_at = time.time()
                e.predicted_end_time = e.killed_at
                self._retire_entry(e)
                return e
        return None

    def remove(self, task_id: str) -> TimelineEntry | None:
        """Immediately remove an active entry (e.g., cancelled task).

        Killed / completed entries are **preserved** for timeline visibility
        — they document the VRAM-prediction vs reality history and let ops
        dashboards render the full lifecycle (user directive: "killed 
           timeline   VRAM   
           ").
        """
        for i, e in enumerate(self._entries):
            if e.task_id == task_id and e.killed_at is None and e.completed_at is None:
                popped = self._entries.pop(i)
                self._unindex_entry(popped)
                self._bump_state_version()
                return popped
        return None

    def clear_projected(self) -> int:
        """Remove speculative projections while preserving launch ownership.

        Called before re-computing lookahead projections so stale
        predictions don't accumulate. Dispatching predictions remain owned
        by their launch attempt until commit or explicit rollback.
        """
        before = len(self._entries)
        self._entries = [
            e for e in self._entries if not e.is_predicted or e.is_dispatching
        ]
        if before != len(self._entries):
            self._rebuild_component_index()
            self._bump_state_version()
        return before - len(self._entries)

    def prune_predicted_component_backlog(
        self,
        component: str,
        *,
        keep_count: int,
        exclude_task_id: str = "",
        drop_order_task_ids: list[str] | None = None,
    ) -> int:
        """Cap pure predicted entries for ``component`` on this GPU.

        This is used by the learned self-concurrency gate to prevent rejected
        placements from being silently refilled by lookahead backlog.  Only
        pure predictions are removable: active entries and dispatch-in-flight
        predictions are preserved so the method cannot cancel real work or
        race the RealityValidator's predicted->active promotion.
        """
        target = max(0, _checked_int(keep_count))
        self._ensure_entry_indexes_current()
        candidates = [
            e
            for e in self._entries_by_component.get(component, [])
            if e.is_predicted
            and not e.is_completed
            and not e.is_dispatching
            and not getattr(e, "is_invalidated", False)
            and not getattr(e, "is_evict_masked", False)
            and (not exclude_task_id or e.task_id != exclude_task_id)
        ]
        if len(candidates) <= target:
            return 0

        excess = len(candidates) - target
        if drop_order_task_ids:
            rank = {
                str(task_id): idx for idx, task_id in enumerate(drop_order_task_ids)
            }
            candidates.sort(
                key=lambda e: (
                    rank.get(str(getattr(e, "task_id", "") or ""), len(rank)),
                ),
            )
        removed = 0
        for entry in candidates[:excess]:
            try:
                self._entries.remove(entry)
            except ValueError:
                continue
            self._unindex_entry(entry)
            removed += 1
        if removed:
            self._bump_state_version()
        return removed

    def invalidate_predictions(
        self,
        component: str,
        *,
        free_vram: bool = True,
    ) -> int:
        """Mark entries for *component* as stale after a drift event.

        Running entries get ``prediction_stale = True`` for diagnostics.
        Projected entries are lazily invalidated and masked out of
        planned-occupancy queries; low-frequency GC can compact them.

        ``free_vram`` controls whether VRAM reservations are also masked
        (``is_evict_masked``).  RAM/latency/interference drift changes only
        ``predicted_ram_mb``/``predicted_end_time``/interference state — it
        never mutates the VRAM step-function tuple — so VRAM entries must
        stay reserved to avoid a spurious VRAM re-solve.  VRAM-metric drift
        (``vram``/``interference_vram``) frees VRAM as before.

        Returns the number of entries affected.
        """
        affected = 0
        now = time.time()
        mutated = False
        component_version = self._component_versions.get(component, 0)
        if self._component_invalidated_versions.get(component) == component_version:
            return 0
        component_entries = self._entries_by_component.get(component, [])
        for e in component_entries:
            if id(e) not in self._entry_ref_ids:
                continue
            if e.component != component:
                continue
            if e.is_completed:
                continue
            affected += 1
            e.prediction_stale = True
            if (e.is_predicted or e.start_time > now) and not e.is_dispatching:
                e.is_invalidated = True
                if free_vram:
                    e.is_evict_masked = True
                mutated = True
        self._component_invalidated_versions[component] = component_version
        if mutated:
            self._bump_state_version()
        return affected


    def reserved_vram_at(
        self,
        at_time: float | None = None,
        *,
        exclude_task_id: str = "",
    ) -> float:
        """Total VRAM reserved by tasks predicted to still be running at *at_time*.

        Completed entries are excluded (they no longer hold VRAM).

        Plan fix — entries with ``is_evict_masked=True`` are
        also excluded.  ``_stochastic_eft_after_eviction`` flips this flag
        on backfills under MCPSE projection so the after-eviction VRAM
        budget reflects the freed slot.  Without this filter, MCPSE could
        not detect VRAM-induced infeasibility relief from eviction (the
        prior implementation flipped ``is_predicted=True`` for the same
        purpose, but ``reserved_vram_at`` never honoured ``is_predicted``
        — making the entire eviction projection a no-op).

        ``exclude_task_id`` is used only for the currently evaluated task's
        still-predicted entry.  Re-planning a pending task must not let that
        task's old reservation block itself, while other tasks' predictions
        must continue to reserve capacity.
        """
        t = at_time if at_time is not None else time.time()
        exclude_id = str(exclude_task_id or "")
        bucket = _checked_int(t)
        use_cache = not exclude_id and bucket not in self._temporal_boundary_buckets
        cache_key = (self._state_version, bucket)
        if use_cache:
            cached = self._vram_at_cache.get(cache_key)
            if cached is not None:
                return cached
        total = 0.0
        for e in self._entries:
            if getattr(e, "is_evict_masked", False):
                continue
            if (
                exclude_id
                and e.task_id == exclude_id
                and getattr(e, "is_predicted", False)
            ):
                continue
            if e.start_time <= t and self._is_vram_reservation_entry(e, t):
                total += e.reserved_vram_at(t)
        if use_cache:
            self._vram_at_cache[cache_key] = total
        return total

    def available_vram_at(
        self,
        at_time: float | None = None,
        *,
        exclude_task_id: str = "",
    ) -> float:
        """Estimated free VRAM at *at_time*."""
        if self.total_vram_mb <= 0:
            return 0.0
        reserved_w = _checked_float(getattr(self, "reserved_weights_mb", 0.0) or 0.0)
        idle_w = _checked_float(getattr(self, "idle_weights_mb", 0.0) or 0.0)
        non_idle_weight = max(0.0, reserved_w - idle_w)
        return max(
            0.0,
            self.total_vram_mb
            - non_idle_weight
            - self.reserved_vram_at(
                at_time,
                exclude_task_id=exclude_task_id,
            ),
        )

    def earliest_fit_time(
        self,
        needed_vram_mb: float,
        horizon_sec: float = 120.0,
        step_sec: float = 1.0,
        *,
        exclude_task_id: str = "",
    ) -> float | None:
        """Find the earliest time within *horizon_sec* when *needed_vram_mb* fits.

        Returns ``None`` if the task cannot fit within the horizon.
        Uses a simple step-scan over the timeline — acceptable for small
        entry counts (typically < 20 per GPU).
        """
        now = time.time()
        if (
            self.available_vram_at(now, exclude_task_id=exclude_task_id)
            >= needed_vram_mb
        ):
            return now
        t = now
        end = now + horizon_sec
        while t < end:
            t += step_sec
            if (
                self.available_vram_at(t, exclude_task_id=exclude_task_id)
                >= needed_vram_mb
            ):
                return t
        return None

    def backfill_entries(self) -> list[TimelineEntry]:
        """Return all backfill entries on this GPU."""
        return [e for e in self._entries if e.is_backfill]

    def entries_for_component(self, component: str) -> list[TimelineEntry]:
        self._ensure_entry_indexes_current()
        current = {id(e) for e in self._entries}
        return [
            e for e in self._entries_by_component.get(component, []) if id(e) in current
        ]

    def active_entries_for_component(self, component: str) -> list[TimelineEntry]:
        self._ensure_entry_indexes_current()
        return [
            e
            for e in self._entries_by_component.get(component, [])
            if self._is_active_index_entry(e)
        ]

    def entries_for_campaign(self, campaign_id: str) -> list[TimelineEntry]:
        return [e for e in self.entries if e.campaign_id == campaign_id]

    def active_entries_for_campaign(self, campaign_id: str) -> list[TimelineEntry]:
        self._ensure_entry_indexes_current()
        return [
            e
            for e in self._entries
            if e.campaign_id == campaign_id and self._is_active_index_entry(e)
        ]

    def has_stale_predictions(self) -> bool:
        return any(e.prediction_stale for e in self._entries)


    @staticmethod
    def _hide_from_ops_live_view(entry: TimelineEntry) -> bool:
        """Return True for planner-masked projections that only confuse ops.

        Drift invalidation marks pure projected entries as
        ``is_invalidated`` / ``is_evict_masked`` so all planner and
        reservation paths ignore them.  Keep that internal state intact, but
        do not serialize those projections as live timeline work.
        """
        return (
            bool(getattr(entry, "is_predicted", False))
            and not bool(getattr(entry, "is_dispatching", False))
            and bool(getattr(entry, "is_invalidated", False))
        )

    def as_dict(
        self,
        *,
        include_retired: bool = True,
        retired_limit: int | None = 128,
    ) -> dict[str, Any]:
        hidden_live_entries = [
            entry for entry in self._entries if self._hide_from_ops_live_view(entry)
        ]
        live_entries = [
            entry for entry in self._entries if not self._hide_from_ops_live_view(entry)
        ]
        retired_entries: list[TimelineEntry] = []
        if include_retired:
            retired_entries = list(self._retired_entries)
            if retired_limit is not None:
                limit = max(0, _checked_int(retired_limit))
                retired_entries = retired_entries[-limit:] if limit > 0 else []
        entries = live_entries + retired_entries
        retired_total = len(self._retired_entries)
        retired_included = len(retired_entries)
        return {
            "gpu_id": self.gpu_id,
            "total_vram_mb": round(self.total_vram_mb, 1),
            "current_reserved_mb": round(self.reserved_vram_at(), 1),
            "current_available_mb": round(self.available_vram_at(), 1),
            "entry_count": len(entries),
            "live_entry_count": len(live_entries),
            "hidden_invalidated_predicted_entry_count": len(hidden_live_entries),
            "retired_entry_count": retired_total,
            "retired_entries_included": retired_included,
            "retired_entries_truncated": max(0, retired_total - retired_included),
            "entries": [e.as_dict() for e in entries],
        }

    def __repr__(self) -> str:
        return (
            f"GpuTimeline(gpu={self.gpu_id}, "
            f"vram={self.total_vram_mb:.0f}MB, "
            f"entries={len(self._entries)})"
        )


class SchedulingScenario:
    """Collection of ``GpuTimeline`` objects — one per GPU in the pool.

    The ``CampaignScheduler`` owns a single ``SchedulingScenario`` and uses it
    for all lookahead queries.
    """

    def __init__(self) -> None:
        self._timelines: dict[str, GpuTimeline] = {}
        self._vram_reservation_model: VramReservationModel = (
            create_vram_reservation_model("full_wall")
        )
        self.total_host_ram_mb: float = 0.0
        self.host_ram_min_available_mb: float = 0.0
        self.host_ram_external_used_mb: float = 0.0
        self.idle_ram_mb: float = 0.0
        self._vram_reservation_model_name = "full_wall"
        self._vram_reservation_signal_service: Any = None
        self._vram_reservation_z: float = 1.96
        self._full_wall_fallback_task_ids: set[str] = set()
        self._scenario_resource_version: int = 0
        self._scenario_event_index: _ScenarioEventIndex | None = None
        self._candidate_fit_cache: dict[tuple[Any, ...], bool] = {}
        self._earliest_fit_cache: dict[tuple[Any, ...], float | None] = {}
        self._candidate_fit_temporal_profile_version: int = 0
        self._incumbent_sweep_cache: dict[tuple[Any, ...], _IncumbentSweep] = {}
        self._interval_fit_prof: dict[str, dict[str, float]] = {
            "candidate_interval_fits": {"wall": 0.0, "calls": 0},
            "earliest_dual_fit_time": {"wall": 0.0, "calls": 0},
        }
        self._task_entry_index: dict[str, TimelineEntry] = {}
        self._task_entry_index_gen: tuple[Any, ...] = ()

    def _invalidate_scenario_event_index(self) -> None:
        self._scenario_resource_version += 1
        self._scenario_event_index = None
        self._incumbent_sweep_cache.clear()
        self._candidate_fit_cache.clear()
        self._earliest_fit_cache.clear()

    def _scenario_generation(self) -> tuple[Any, ...]:
        return (
            self._scenario_resource_version,
            self._vram_reservation_model_name,
            self._vram_reservation_z,
            self._candidate_fit_temporal_profile_version,
            tuple(
                (gpu_id, timeline._state_version)
                for gpu_id, timeline in sorted(self._timelines.items())
            ),
        )

    def _entry_event_index_parts(
        self,
        entry: TimelineEntry,
    ) -> tuple[
        tuple[_IncumbentInterval, ...],
        _IncumbentInterval | None,
        tuple[tuple[float, str], ...],
    ]:
        entry_completed = entry.is_completed
        evict_masked = getattr(entry, "is_evict_masked", False)
        invalidated = getattr(entry, "is_invalidated", False)
        vram_skip = entry_completed or evict_masked
        ram_skip = entry_completed or invalidated or evict_masked
        if vram_skip and ram_skip:
            return (), None, ()
        entry_start = _checked_float(entry.start_time)
        task_id = str(entry.task_id)
        is_predicted = bool(entry.is_predicted)
        boundaries: set[tuple[float, str]] = set()
        vram_intervals: tuple[_IncumbentInterval, ...] = ()
        if not vram_skip:
            if entry._has_valid_temporal_reservation():
                peak_start = entry.temporal_peak_reservation_start_time()
                peak_end = entry.temporal_peak_reservation_end_time()
                if (
                    entry.allow_vram_mb is None
                    or peak_start is None
                    or peak_end is None
                ):
                    return (), None, ()
                entry_end = entry.vram_reservation_end_time() if is_predicted else None
                vram_intervals = (
                    _IncumbentInterval(
                        entry_start,
                        peak_start,
                        _checked_float(entry.allow_vram_mb),
                        task_id,
                        is_predicted,
                    ),
                    _IncumbentInterval(
                        peak_start,
                        peak_end,
                        _checked_float(entry.predicted_vram_mb),
                        task_id,
                        is_predicted,
                    ),
                    _IncumbentInterval(
                        peak_end,
                        entry_end,
                        _checked_float(entry.allow_vram_mb),
                        task_id,
                        is_predicted,
                    ),
                )
                if self._vram_reservation_model_name == "temporal_peak_interval":
                    boundaries.update(((peak_start, task_id), (peak_end, task_id)))
            else:
                entry_end = entry.vram_reservation_end_time() if is_predicted else None
                vram_intervals = (
                    _IncumbentInterval(
                        entry_start,
                        entry_end,
                        _checked_float(entry.predicted_vram_mb),
                        task_id,
                        is_predicted,
                    ),
                )
            if is_predicted:
                boundaries.update(
                    ((entry_start, task_id), (_checked_float(entry_end), task_id))
                )
        ram_interval = None
        if not ram_skip:
            ram_end = entry.predicted_end_time if is_predicted else None
            ram_interval = _IncumbentInterval(
                entry_start,
                ram_end,
                _checked_float(entry.predicted_ram_mb or 0.0),
                task_id,
                is_predicted,
            )
        return vram_intervals, ram_interval, tuple(sorted(boundaries))

    @staticmethod
    def _is_single_timeline_append_generation(
        before: tuple[Any, ...],
        after: tuple[Any, ...],
        gpu_id: str,
    ) -> bool:
        if before[:4] != after[:4]:
            return False
        before_timelines = dict(before[4])
        after_timelines = dict(after[4])
        if before_timelines.keys() != after_timelines.keys():
            return False
        target = str(gpu_id)
        return all(
            after_timelines[current_gpu]
            == before_version + (1 if current_gpu == target else 0)
            for current_gpu, before_version in before_timelines.items()
        )

    def _event_index_with_added_entry(
        self,
        cached: _ScenarioEventIndex,
        entry: TimelineEntry,
        generation: tuple[Any, ...],
    ) -> _ScenarioEventIndex | None:
        if generation != self._scenario_generation():
            return None
        vram_intervals, ram_interval, boundaries = self._entry_event_index_parts(entry)
        if ram_interval is None:
            return None
        ram_offset = 0
        found_entry = False
        for timeline in self._timelines.values():
            for active_entry in timeline.active_entries:
                if getattr(active_entry, "is_evict_masked", False):
                    continue
                if active_entry is entry:
                    found_entry = True
                    break
                ram_offset += 1
            if found_entry:
                break
        if not found_entry:
            return None
        vram_by_gpu = dict(cached.vram_by_gpu)
        gpu_id = str(entry.gpu_id)
        vram_by_gpu[gpu_id] = (
            *vram_by_gpu.get(gpu_id, ()),
            *vram_intervals,
        )
        extended = _ScenarioEventIndex(
            generation=generation,
            vram_by_gpu=vram_by_gpu,
            ram_intervals=(
                *cached.ram_intervals[:ram_offset],
                ram_interval,
                *cached.ram_intervals[ram_offset:],
            ),
            candidate_boundaries=_merge_sorted_boundaries(
                cached.candidate_boundaries,
                boundaries,
            ),
        )
        return extended if generation == self._scenario_generation() else None

    def _event_index_with_removed_entries(
        self,
        cached: _ScenarioEventIndex,
        removed_task_ids: set[str],
        generation: tuple[Any, ...],
    ) -> _ScenarioEventIndex | None:
        """Incrementally drop removed-task intervals from the cached index.

        Mirrors ``_event_index_with_added_entry`` for the remove direction.
        ``remove_predicted_entries_for_task(s)`` bumps timeline state
        versions, which would otherwise invalidate the whole event index
        and force a full rebuild on the next ``_event_index`` call.  Since
        removal only drops whole task-owned intervals, filter them out by
        ``task_id`` instead of rebuilding.
        """
        if generation != self._scenario_generation():
            return None
        removed = set(removed_task_ids)
        vram_by_gpu: dict[str, tuple[_IncumbentInterval, ...]] = {}
        for gpu_id, intervals in cached.vram_by_gpu.items():
            vram_by_gpu[gpu_id] = tuple(
                interval for interval in intervals if interval.task_id not in removed
            )
        ram_intervals = tuple(
            interval
            for interval in cached.ram_intervals
            if interval.task_id not in removed
        )
        candidate_boundaries = tuple(
            boundary
            for boundary in cached.candidate_boundaries
            if boundary[1] not in removed
        )
        extended = _ScenarioEventIndex(
            generation=generation,
            vram_by_gpu=vram_by_gpu,
            ram_intervals=ram_intervals,
            candidate_boundaries=candidate_boundaries,
        )
        return extended if generation == self._scenario_generation() else None

    def _event_index(self) -> _ScenarioEventIndex:
        generation = self._scenario_generation()
        cached = self._scenario_event_index
        if cached is not None and cached.generation == generation:
            return cached
        vram_by_gpu: dict[str, tuple[_IncumbentInterval, ...]] = {}
        ram_intervals: list[_IncumbentInterval] = []
        candidate_boundaries: set[tuple[float, str]] = set()
        for gpu_id, timeline in self._timelines.items():
            vram_intervals: list[_IncumbentInterval] = []
            for entry in timeline.active_entries:
                entry_vram, entry_ram, entry_boundaries = self._entry_event_index_parts(
                    entry
                )
                vram_intervals.extend(entry_vram)
                if entry_ram is not None:
                    ram_intervals.append(entry_ram)
                candidate_boundaries.update(entry_boundaries)
            vram_by_gpu[str(gpu_id)] = tuple(vram_intervals)
        cached = _ScenarioEventIndex(
            generation=generation,
            vram_by_gpu=vram_by_gpu,
            ram_intervals=tuple(ram_intervals),
            candidate_boundaries=tuple(sorted(candidate_boundaries)),
        )
        if generation != self._scenario_generation():
            return self._event_index()
        self._scenario_event_index = cached
        return cached

    @staticmethod
    def _add_indexed_interval(
        events: dict[float, float],
        interval: _IncumbentInterval,
        *,
        start_time: float,
        end_time: float,
    ) -> float:
        if (
            interval.amount_mb <= 0.0
            or (interval.end_time is not None and interval.end_time <= start_time)
            or interval.start_time >= end_time
        ):
            return 0.0
        if interval.start_time <= start_time and (
            interval.end_time is None or start_time < interval.end_time
        ):
            initial = interval.amount_mb
        else:
            initial = 0.0
            if start_time < interval.start_time < end_time:
                events[interval.start_time] = (
                    events.get(interval.start_time, 0.0) + interval.amount_mb
                )
        if interval.end_time is not None and start_time < interval.end_time < end_time:
            events[interval.end_time] = (
                events.get(interval.end_time, 0.0) - interval.amount_mb
            )
        return initial

    def _incumbent_sweep(self, gpu_id: str) -> _IncumbentSweep:
        """Per-GPU generation-scoped incumbent prefix-sum for range queries."""
        generation = self._scenario_generation()
        key = (generation, gpu_id)
        cached = self._incumbent_sweep_cache.get(key)
        if cached is not None:
            return cached
        event_index = self._event_index()
        events: dict[float, float] = {}
        for interval in event_index.vram_by_gpu.get(gpu_id, ()):
            if interval.amount_mb <= 0.0:
                continue
            events[interval.start_time] = (
                events.get(interval.start_time, 0.0) + interval.amount_mb
            )
            if interval.end_time is not None:
                events[interval.end_time] = (
                    events.get(interval.end_time, 0.0) - interval.amount_mb
                )
        vram_boundaries = tuple(sorted(events))
        vram_prefix: list[float] = []
        vram_acc = 0.0
        for boundary in vram_boundaries:
            vram_acc += events[boundary]
            vram_prefix.append(vram_acc)
        ram_events: dict[float, float] = {}
        for interval in event_index.ram_intervals:
            if interval.amount_mb <= 0.0:
                continue
            ram_events[interval.start_time] = (
                ram_events.get(interval.start_time, 0.0) + interval.amount_mb
            )
            if interval.end_time is not None:
                ram_events[interval.end_time] = (
                    ram_events.get(interval.end_time, 0.0) - interval.amount_mb
                )
        ram_boundaries = tuple(sorted(ram_events))
        ram_prefix: list[float] = []
        ram_acc = 0.0
        for boundary in ram_boundaries:
            ram_acc += ram_events[boundary]
            ram_prefix.append(ram_acc)
        predicted_task_ids = frozenset(
            task_id for (_boundary, task_id) in event_index.candidate_boundaries
        )
        sweep = _IncumbentSweep(
            generation=generation,
            gpu_id=gpu_id,
            vram_boundaries=vram_boundaries,
            vram_prefix=tuple(vram_prefix),
            ram_boundaries=ram_boundaries,
            ram_prefix=tuple(ram_prefix),
            predicted_task_ids=predicted_task_ids,
        )
        self._incumbent_sweep_cache[key] = sweep
        return sweep

    def invalidate_temporal_profile_cache(self) -> None:
        """Invalidate exact fit reuse after temporal profile observations change."""
        self._candidate_fit_temporal_profile_version += 1
        self._candidate_fit_cache.clear()
        self._earliest_fit_cache.clear()
        invalidate = getattr(
            self._vram_reservation_model, "invalidate_temporal_profile", None
        )
        if callable(invalidate):
            invalidate()

    def set_vram_reservation_model(
        self,
        name: str | None,
        *,
        signal_service: Any = None,
        resource_z: float = 1.96,
    ) -> None:
        self._vram_reservation_model = create_vram_reservation_model(
            name,
            signal_service=signal_service,
            resource_z=resource_z,
        )
        self._vram_reservation_model_name = self._vram_reservation_model.name
        self._vram_reservation_signal_service = signal_service
        parsed_z = _checked_float(resource_z)
        self._vram_reservation_z = (
            parsed_z if math.isfinite(parsed_z) and parsed_z > 0 else 1.96
        )
        self._full_wall_fallback_task_ids.clear()
        self._invalidate_scenario_event_index()

    def force_full_wall_for_task(self, task_id: str) -> bool:
        """Fail one unstable temporal candidate closed to the full-wall model."""
        task_key = str(task_id or "")
        if (
            not task_key
            or self._vram_reservation_model_name != "temporal_peak_interval"
        ):
            return False
        changed = task_key not in self._full_wall_fallback_task_ids
        self._full_wall_fallback_task_ids.add(task_key)
        for timeline in self._timelines.values():
            mutated = False
            for entry in timeline._entries:
                if entry.task_id != task_key:
                    continue
                if any(
                    value is not None
                    for value in (
                        entry.allow_vram_mb,
                        entry.peak_start_time,
                        entry.peak_end_time,
                    )
                ):
                    entry.allow_vram_mb = None
                    entry.peak_start_time = None
                    entry.peak_end_time = None
                    entry.launch_uncertainty_sec = 0.0
                    mutated = True
            if mutated:
                timeline._bump_state_version()
                changed = True
        if changed:
            self._invalidate_scenario_event_index()
        return changed

    def task_uses_full_wall_reservation(self, task_id: str) -> bool:
        return (
            self._vram_reservation_model_name != "temporal_peak_interval"
            or str(task_id or "") in self._full_wall_fallback_task_ids
        )

    def get_or_create(self, gpu_id: str, total_vram_mb: float = 0.0) -> GpuTimeline:
        tl = self._timelines.get(gpu_id)
        if tl is None:
            tl = GpuTimeline(gpu_id, total_vram_mb)
            self._timelines[gpu_id] = tl
        elif total_vram_mb > 0 and tl.total_vram_mb <= 0:
            tl.total_vram_mb = total_vram_mb
            self._invalidate_scenario_event_index()
        return tl

    def get(self, gpu_id: str) -> GpuTimeline | None:
        return self._timelines.get(gpu_id)

    @property
    def gpu_ids(self) -> list[str]:
        return list(self._timelines.keys())

    def sync_vram_totals(self, vram_totals: dict[str, int]) -> None:
        """Update total VRAM from ``ResourceAdmissionTracker.total_vram``."""
        changed = False
        for gpu_id, total in vram_totals.items():
            tl = self.get_or_create(gpu_id, _checked_float(total))
            parsed = _checked_float(total)
            if tl.total_vram_mb != parsed:
                tl.total_vram_mb = parsed
                changed = True
        if changed:
            self._invalidate_scenario_event_index()

    def sync_reserved_weights(self, reserved_weights: dict[str, int]) -> None:
        """Update reserved weights (resident workers) from ResourceAdmissionTracker."""
        for gpu_id, weight in reserved_weights.items():
            tl = self.get_or_create(gpu_id)
            tl.reserved_weights_mb = _checked_float(weight)
            tl._bump_state_version()

    def sync_idle_weights(self, idle_weights: dict[str, int]) -> None:
        """Update idle weights (freeable resident workers) from ResourceAdmissionTracker."""
        for gpu_id, weight in idle_weights.items():
            tl = self.get_or_create(gpu_id)
            tl.idle_weights_mb = _checked_float(weight)
            tl._bump_state_version()

    def sync_idle_ram(self, idle_ram: float) -> None:
        """Update global idle worker RAM capacity."""
        parsed = _checked_float(idle_ram)
        if self.idle_ram_mb != parsed:
            self.idle_ram_mb = parsed
            self._invalidate_scenario_event_index()

    def sync_host_ram(
        self,
        *,
        total_ram_mb: float | None = None,
        min_available_mb: float | None = None,
        mem_available_mb: float | None = None,
        active_reserved_mb: float | None = None,
    ) -> None:
        """Update host-wide CPU RAM planning capacity.

        ``min_available_mb`` mirrors the supervisor's MemAvailable guard.
        If total RAM is unavailable, host RAM planning is disabled
        fail-open and the runtime guard remains authoritative.
        """
        if total_ram_mb is not None and total_ram_mb > 0:
            self.total_host_ram_mb = _checked_float(total_ram_mb)
        if min_available_mb is not None and min_available_mb >= 0:
            self.host_ram_min_available_mb = _checked_float(min_available_mb)
        if mem_available_mb is not None and self.total_host_ram_mb > 0:
            safe_capacity = max(
                0.0, self.total_host_ram_mb - self.host_ram_min_available_mb
            )
            observed_available = max(
                0.0, _checked_float(mem_available_mb) - self.host_ram_min_available_mb
            )
            current_reserved = (
                _checked_float(active_reserved_mb)
                if active_reserved_mb is not None
                else self.reserved_host_ram_at()
            )
            self.host_ram_external_used_mb = max(
                0.0,
                safe_capacity - observed_available - max(0.0, current_reserved),
            )
        self._invalidate_scenario_event_index()

    def reserved_host_ram_at(
        self,
        at_time: float | None = None,
        *,
        exclude_task_id: str = "",
    ) -> float:
        """Host-wide active CPU RAM reserved by planned occupancy."""
        total = 0.0
        t = time.time() if at_time is None else _checked_float(at_time)
        exclude_id = str(exclude_task_id or "")
        for tl in self._timelines.values():
            for e in tl.active_entries:
                if not tl.is_planned_occupancy_entry(e, t):
                    continue
                if (
                    exclude_id
                    and e.task_id == exclude_id
                    and getattr(e, "is_predicted", False)
                ):
                    continue
                if e.start_time <= t:
                    total += _checked_float(getattr(e, "predicted_ram_mb", 0.0) or 0.0)
        return total

    def available_host_ram_at(
        self,
        at_time: float | None = None,
        *,
        exclude_task_id: str = "",
    ) -> float:
        """Estimated host RAM available to new active task reservations."""
        if self.total_host_ram_mb <= 0:
            return _checked_float("inf")
        safe_capacity = max(
            0.0, self.total_host_ram_mb - self.host_ram_min_available_mb
        )
        external_used = max(
            0.0,
            self.host_ram_external_used_mb
            - _checked_float(getattr(self, "idle_ram_mb", 0.0) or 0.0),
        )
        return max(
            0.0,
            safe_capacity
            - external_used
            - self.reserved_host_ram_at(
                at_time,
                exclude_task_id=exclude_task_id,
            ),
        )

    def _candidate_vram_reservation(
        self,
        *,
        gpu_id: str,
        start_time: float,
        end_time: float,
        needed_vram_mb: float,
        exclude_task_id: str = "",
        component: str = "",
        config_fingerprint: str = "",
        input_size: float | None = None,
        use_planned_envelope: bool = False,
        force_full_wall: bool = False,
    ) -> VramReservation:
        """Resolve one candidate's temporal shape without inserting it."""
        if (
            force_full_wall
            or self._vram_reservation_model_name != "temporal_peak_interval"
            or exclude_task_id in self._full_wall_fallback_task_ids
        ):
            return VramReservation()
        entry = self.find_entry(exclude_task_id) if exclude_task_id else None
        requested_component = str(component or "")
        requested_config = str(config_fingerprint or "")
        entry_component = str(getattr(entry, "component", "") or "")
        entry_config = str(getattr(entry, "config_fingerprint", "") or "")
        candidate_input_size = (
            _checked_float(input_size)
            if input_size is not None
            else _checked_float(getattr(entry, "input_size", 0.0) or 0.0)
        )
        entry_matches_candidate = (
            not requested_component or requested_component == entry_component
        ) and (not requested_config or requested_config == entry_config)
        if (
            entry is not None
            and entry_matches_candidate
            and not entry.is_completed
            and str(entry.gpu_id) == str(gpu_id)
            and entry._has_valid_temporal_reservation()
        ):
            entry_low = entry.allow_vram_mb
            entry_peak_start = entry.peak_start_time
            entry_peak_end = entry.peak_end_time
            if use_planned_envelope:
                return VramReservation(
                    allow_vram_mb=entry_low,
                    peak_start_time=entry_peak_start,
                    peak_end_time=entry_peak_end,
                )
            if (
                end_time > start_time
                and entry_low is not None
                and entry_peak_start is not None
                and entry_peak_end is not None
            ):
                peak_start = start_time + (entry_peak_start - entry.start_time)
                peak_end = start_time + (entry_peak_end - entry.start_time)
                if start_time < peak_start < peak_end < end_time:
                    return VramReservation(
                        allow_vram_mb=entry_low,
                        peak_start_time=peak_start,
                        peak_end_time=peak_end,
                    )
                return VramReservation()
        candidate_component = requested_component or entry_component
        if not candidate_component:
            return VramReservation()
        candidate_config = requested_config
        if not candidate_config and (
            not requested_component or requested_component == entry_component
        ):
            candidate_config = entry_config
        return self._vram_reservation_model.reservation_for(
            component=candidate_component,
            config_fingerprint=candidate_config,
            gpu_id=str(gpu_id),
            start_time=start_time,
            predicted_end_time=end_time,
            predicted_vram_mb=needed_vram_mb,
            input_size=candidate_input_size,
        )

    def _fast_interval_fits(
        self,
        sweep: _IncumbentSweep,
        gpu_id: str,
        start: float,
        end: float,
        needed_vram_mb: float,
        needed_ram_mb: float,
        candidate_temporal: bool,
        candidate_low: float | None,
        candidate_peak_reservation_start: float | None,
        candidate_peak_reservation_end: float | None,
        vram_capacity: float,
        host_capacity: float,
    ) -> bool:
        """Range-max check of the incumbent sweep; same hard-capacity gates."""
        vram_boundaries = sweep.vram_boundaries
        vram_prefix = sweep.vram_prefix
        if (
            candidate_temporal
            and candidate_low is not None
            and candidate_peak_reservation_start is not None
            and candidate_peak_reservation_end is not None
        ):
            peak_lo = max(start, candidate_peak_reservation_start)
            peak_hi = min(end, candidate_peak_reservation_end)
            if peak_lo < peak_hi:
                used = _range_max_used(vram_prefix, vram_boundaries, peak_lo, peak_hi)
                if max(0.0, vram_capacity - used) < needed_vram_mb:
                    return False
            left_hi = min(end, peak_lo)
            if start < left_hi:
                used = _range_max_used(vram_prefix, vram_boundaries, start, left_hi)
                if max(0.0, vram_capacity - used) < candidate_low:
                    return False
            right_lo = max(start, peak_hi)
            if right_lo < end:
                used = _range_max_used(vram_prefix, vram_boundaries, right_lo, end)
                if max(0.0, vram_capacity - used) < candidate_low:
                    return False
        else:
            used = _range_max_used(vram_prefix, vram_boundaries, start, end)
            if max(0.0, vram_capacity - used) < needed_vram_mb:
                return False
        if needed_ram_mb > 0.0:
            used = _range_max_used(sweep.ram_prefix, sweep.ram_boundaries, start, end)
            if max(0.0, host_capacity - used) < needed_ram_mb:
                return False
        return True

    @_profile_interval_fit("candidate_interval_fits")
    def candidate_interval_fits(
        self,
        gpu_id: str,
        start_time: float,
        end_time: float,
        needed_vram_mb: float,
        needed_ram_mb: float = 0.0,
        *,
        exclude_task_id: str = "",
        candidate_component: str = "",
        candidate_config_fingerprint: str = "",
        candidate_input_size: float | None = None,
        candidate_wall_end_time: float | None = None,
        candidate_use_planned_envelope: bool = False,
        candidate_force_full_wall: bool = False,
        candidate_peak_margin_sec: float | None = None,
        _captured_now: float | None = None,
    ) -> bool:
        """Check one occupied interval using the candidate's full wall shape."""
        start = _checked_float(start_time)
        end = _checked_float(end_time)
        cache_key = (
            self._scenario_generation(),
            str(gpu_id),
            start,
            end,
            _checked_float(needed_vram_mb),
            _checked_float(needed_ram_mb),
            str(exclude_task_id or ""),
            str(candidate_component or ""),
            str(candidate_config_fingerprint or ""),
            (
                _checked_float(candidate_input_size)
                if candidate_input_size is not None
                else None
            ),
            (
                _checked_float(candidate_wall_end_time)
                if candidate_wall_end_time is not None
                else None
            ),
            bool(candidate_use_planned_envelope),
            bool(candidate_force_full_wall),
            str(exclude_task_id or "") in self._full_wall_fallback_task_ids,
            (
                _checked_float(candidate_peak_margin_sec)
                if candidate_peak_margin_sec is not None
                else None
            ),
            _checked_float(_captured_now) if _captured_now is not None else None,
        )
        cache_enabled = _captured_now is not None
        cached_fit = self._candidate_fit_cache.get(cache_key) if cache_enabled else None
        if cache_key[0] != self._scenario_generation():
            return self.candidate_interval_fits(
                gpu_id,
                start_time,
                end_time,
                needed_vram_mb,
                needed_ram_mb,
                exclude_task_id=exclude_task_id,
                candidate_component=candidate_component,
                candidate_config_fingerprint=candidate_config_fingerprint,
                candidate_input_size=candidate_input_size,
                candidate_wall_end_time=candidate_wall_end_time,
                candidate_use_planned_envelope=candidate_use_planned_envelope,
                candidate_force_full_wall=candidate_force_full_wall,
                candidate_peak_margin_sec=candidate_peak_margin_sec,
                _captured_now=_captured_now,
            )
        if cached_fit is not None:
            return cached_fit
        planned_entry = (
            self.find_entry(exclude_task_id)
            if candidate_use_planned_envelope and exclude_task_id
            else None
        )
        if (
            planned_entry is not None
            and planned_entry.is_predicted
            and planned_entry._has_valid_temporal_reservation()
        ):
            start = planned_entry.start_time
            end = planned_entry.predicted_end_time
        if end <= start:
            if cache_enabled:
                self._candidate_fit_cache[cache_key] = False
            return False
        candidate_wall_end = end
        if candidate_wall_end_time is not None:
            requested_wall_end = _checked_float(candidate_wall_end_time)
            if requested_wall_end >= end:
                candidate_wall_end = requested_wall_end
        timeline = self.get(str(gpu_id))
        if timeline is None:
            if cache_enabled:
                self._candidate_fit_cache[cache_key] = False
            return False
        temporal_mode = self._vram_reservation_model_name == "temporal_peak_interval"
        candidate_reservation = self._candidate_vram_reservation(
            gpu_id=str(gpu_id),
            start_time=start,
            end_time=candidate_wall_end,
            needed_vram_mb=needed_vram_mb,
            exclude_task_id=exclude_task_id,
            component=candidate_component,
            config_fingerprint=candidate_config_fingerprint,
            input_size=candidate_input_size,
            use_planned_envelope=candidate_use_planned_envelope,
            force_full_wall=candidate_force_full_wall,
        )
        candidate_low = candidate_reservation.allow_vram_mb
        candidate_peak_start = candidate_reservation.peak_start_time
        candidate_peak_end = candidate_reservation.peak_end_time
        candidate_temporal = bool(
            temporal_mode
            and candidate_low is not None
            and candidate_peak_start is not None
            and candidate_peak_end is not None
            and start <= candidate_peak_start < candidate_peak_end <= candidate_wall_end
            and 0 <= candidate_low < needed_vram_mb
        )
        peak_margin = 0.0
        if candidate_temporal:
            if planned_entry is not None:
                peak_margin = max(
                    0.0,
                    _checked_float(planned_entry.launch_uncertainty_sec),
                )
            elif candidate_peak_margin_sec is not None:
                peak_margin = max(0.0, _checked_float(candidate_peak_margin_sec))
            else:
                assert candidate_peak_start is not None
                assert candidate_peak_end is not None
                peak_margin = _temporal_peak_margin_sec(
                    candidate_peak_start,
                    candidate_peak_end,
                    self._vram_reservation_z,
                )
        reservation_end = end
        candidate_peak_reservation_start = candidate_peak_start
        candidate_peak_reservation_end = (
            min(end, candidate_peak_end + peak_margin)
            if candidate_peak_end is not None
            else None
        )
        vram_events: dict[float, float] = {}
        ram_events: dict[float, float] = {}
        vram_initial = 0.0
        ram_initial = 0.0

        exclude_id = str(exclude_task_id or "")
        non_idle_weight = max(
            0.0,
            _checked_float(timeline.reserved_weights_mb)
            - _checked_float(timeline.idle_weights_mb),
        )
        vram_capacity = max(0.0, timeline.total_vram_mb - non_idle_weight)
        host_capacity = _checked_float("inf")
        if self.total_host_ram_mb > 0:
            host_capacity = max(
                0.0,
                self.total_host_ram_mb
                - self.host_ram_min_available_mb
                - max(0.0, self.host_ram_external_used_mb - self.idle_ram_mb),
            )
        sweep = self._incumbent_sweep(str(gpu_id))
        if (not exclude_id) or (exclude_id not in sweep.predicted_task_ids):
            fits = self._fast_interval_fits(
                sweep=sweep,
                gpu_id=str(gpu_id),
                start=start,
                end=end,
                needed_vram_mb=needed_vram_mb,
                needed_ram_mb=needed_ram_mb,
                candidate_temporal=candidate_temporal,
                candidate_low=candidate_low,
                candidate_peak_reservation_start=candidate_peak_reservation_start,
                candidate_peak_reservation_end=candidate_peak_reservation_end,
                vram_capacity=vram_capacity,
                host_capacity=host_capacity,
            )
            if cache_enabled:
                if len(self._candidate_fit_cache) > 4096:
                    self._candidate_fit_cache.clear()
                self._candidate_fit_cache[cache_key] = fits
            return fits

        event_index = self._event_index()
        for interval in event_index.vram_by_gpu.get(str(gpu_id), ()):
            if exclude_id and interval.task_id == exclude_id and interval.is_predicted:
                continue
            vram_initial += self._add_indexed_interval(
                vram_events,
                interval,
                start_time=start,
                end_time=reservation_end,
            )

        if needed_ram_mb > 0.0:
            for interval in event_index.ram_intervals:
                if (
                    exclude_id
                    and interval.task_id == exclude_id
                    and interval.is_predicted
                ):
                    continue
                ram_initial += self._add_indexed_interval(
                    ram_events,
                    interval,
                    start_time=start,
                    end_time=reservation_end,
                )

        boundaries = {
            start,
            reservation_end,
            *vram_events,
            *ram_events,
        }
        if candidate_temporal:
            for boundary in (
                candidate_peak_reservation_start,
                candidate_peak_reservation_end,
            ):
                if boundary is not None and start < boundary < reservation_end:
                    boundaries.add(boundary)
        used_vram = vram_initial
        used_ram = ram_initial
        for segment_start in sorted(boundaries)[:-1]:
            if segment_start > start:
                used_vram += vram_events.get(segment_start, 0.0)
                used_ram += ram_events.get(segment_start, 0.0)
            candidate_vram = needed_vram_mb
            if (
                candidate_temporal
                and candidate_low is not None
                and candidate_peak_reservation_start is not None
                and candidate_peak_reservation_end is not None
                and not candidate_peak_reservation_start
                <= segment_start
                < candidate_peak_reservation_end
            ):
                candidate_vram = candidate_low
            candidate_ram = needed_ram_mb if segment_start < end else 0.0
            if (
                max(0.0, vram_capacity - used_vram) < candidate_vram
                or max(0.0, host_capacity - used_ram) < candidate_ram
            ):
                if cache_enabled:
                    if len(self._candidate_fit_cache) > 4096:
                        self._candidate_fit_cache.clear()
                    self._candidate_fit_cache[cache_key] = False
                return False
        if cache_enabled:
            if len(self._candidate_fit_cache) > 4096:
                self._candidate_fit_cache.clear()
            self._candidate_fit_cache[cache_key] = True
        return True

    @_profile_interval_fit("earliest_dual_fit_time")
    def earliest_dual_fit_time(
        self,
        gpu_id: str,
        needed_vram_mb: float,
        needed_ram_mb: float,
        horizon_sec: float = 120.0,
        step_sec: float = 1.0,
        *,
        exclude_task_id: str = "",
        required_duration_sec: float = 0.0,
        candidate_component: str = "",
        candidate_config_fingerprint: str = "",
        candidate_input_size: float | None = None,
        candidate_force_full_wall: bool = False,
        earliest_start_time: float | None = None,
        _captured_now: float | None = None,
    ) -> float | None:
        """Find the earliest start whose complete candidate interval fits."""
        now = (
            _checked_float(_captured_now) if _captured_now is not None else time.time()
        )
        earliest_start = max(
            now,
            _checked_float(earliest_start_time)
            if earliest_start_time is not None
            else now,
        )
        horizon_end = earliest_start + max(0.0, _checked_float(horizon_sec))
        duration = max(0.0, _checked_float(required_duration_sec or 0.0))
        generation = self._scenario_generation()
        cache_key = (
            generation,
            str(gpu_id),
            _checked_float(needed_vram_mb),
            _checked_float(needed_ram_mb),
            _checked_float(horizon_sec),
            _checked_float(step_sec),
            str(exclude_task_id or ""),
            duration,
            str(candidate_component or ""),
            str(candidate_config_fingerprint or ""),
            (
                _checked_float(candidate_input_size)
                if candidate_input_size is not None
                else None
            ),
            bool(candidate_force_full_wall),
            str(exclude_task_id or "") in self._full_wall_fallback_task_ids,
            earliest_start,
            now,
        )
        cache_enabled = _captured_now is not None

        def _retry() -> float | None:
            return self.earliest_dual_fit_time(
                gpu_id,
                needed_vram_mb,
                needed_ram_mb,
                horizon_sec,
                step_sec,
                exclude_task_id=exclude_task_id,
                required_duration_sec=required_duration_sec,
                candidate_component=candidate_component,
                candidate_config_fingerprint=candidate_config_fingerprint,
                candidate_input_size=candidate_input_size,
                candidate_force_full_wall=candidate_force_full_wall,
                earliest_start_time=earliest_start_time,
                _captured_now=_captured_now,
            )

        if generation != self._scenario_generation():
            return _retry()
        if cache_enabled and cache_key in self._earliest_fit_cache:
            return self._earliest_fit_cache[cache_key]
        target = self.get(str(gpu_id))
        if target is None:
            if cache_enabled:
                self._earliest_fit_cache[cache_key] = None
            return None
        temporal_mode = self._vram_reservation_model_name == "temporal_peak_interval"
        candidate_times = {earliest_start}
        event_end = horizon_end + duration
        probe_duration = max(duration, 0.01)
        probe_reservation = self._candidate_vram_reservation(
            gpu_id=str(gpu_id),
            start_time=now,
            end_time=now + probe_duration,
            needed_vram_mb=needed_vram_mb,
            exclude_task_id=exclude_task_id,
            component=candidate_component,
            config_fingerprint=candidate_config_fingerprint,
            input_size=candidate_input_size,
            force_full_wall=candidate_force_full_wall,
        )
        probe_temporal = bool(
            temporal_mode
            and probe_reservation.allow_vram_mb is not None
            and probe_reservation.peak_start_time is not None
            and probe_reservation.peak_end_time is not None
        )
        probe_peak_margin = (
            _temporal_peak_margin_sec(
                probe_reservation.peak_start_time,
                probe_reservation.peak_end_time,
                self._vram_reservation_z,
            )
            if probe_temporal
            and probe_reservation.peak_start_time is not None
            and probe_reservation.peak_end_time is not None
            else 0.0
        )
        candidate_offsets = [
            boundary - now
            for boundary in (
                probe_reservation.peak_start_time,
                (
                    min(
                        now + probe_duration,
                        probe_reservation.peak_end_time + probe_peak_margin,
                    )
                    if probe_temporal and probe_reservation.peak_end_time is not None
                    else probe_reservation.peak_end_time
                ),
            )
            if boundary is not None
        ]
        event_index = self._event_index()
        for event_time, event_task_id in event_index.candidate_boundaries:
            if event_task_id == exclude_task_id:
                continue
            if now < event_time <= event_end:
                candidate_times.add(event_time)
            for offset in candidate_offsets:
                aligned_start = event_time - offset
                if earliest_start < aligned_start <= horizon_end:
                    candidate_times.add(aligned_start)
        for candidate_start in sorted(candidate_times):
            if candidate_start < earliest_start or candidate_start > horizon_end:
                continue
            if self.candidate_interval_fits(
                gpu_id,
                candidate_start,
                candidate_start + probe_duration,
                needed_vram_mb,
                needed_ram_mb,
                exclude_task_id=exclude_task_id,
                candidate_component=candidate_component,
                candidate_config_fingerprint=candidate_config_fingerprint,
                candidate_input_size=candidate_input_size,
                candidate_force_full_wall=candidate_force_full_wall,
                _captured_now=now,
            ):
                if generation != self._scenario_generation():
                    return _retry()
                if cache_enabled:
                    if len(self._earliest_fit_cache) > 4096:
                        self._earliest_fit_cache.clear()
                    self._earliest_fit_cache[cache_key] = candidate_start
                return candidate_start
        if generation != self._scenario_generation():
            return _retry()
        if cache_enabled:
            if len(self._earliest_fit_cache) > 4096:
                self._earliest_fit_cache.clear()
            self._earliest_fit_cache[cache_key] = None
        return None

    def next_predicted_completion(
        self,
        now: float | None = None,
    ) -> float | None:
        """Plan fix — return the earliest future
        ``predicted_end_time`` across all timelines, or ``None`` if no
        future predicted completion exists.

        Used by ``SchedulingSupervisor._wait_for_wake_or_tick`` to set
        a dynamic timeout until the next anticipated state-change event
        (task completion).  Replaces the static ``periodic_tick_sec``
        upper bound when a closer event is known.

        Iterates all GPU timelines' active entries (active = not
        completed) and takes the minimum of ``predicted_end_time > now``.
        Entries whose predicted_end_time has already passed are excluded
        — they represent either stale predictions (Layer-2 GC candidate)
        or in-progress dispatch; neither should set a sub-zero timeout.

        Complexity: O(Σ |active_entries|) per call.  For a 4-GPU, ~40
        task scenario: negligible (<100 iterations).
        """
        t_now = time.time() if now is None else _checked_float(now)
        next_t: float | None = None
        for tl in self._timelines.values():
            for entry in tl.active_entries:
                candidate = entry.predicted_end_time
                if candidate > t_now and (next_t is None or candidate < next_t):
                    next_t = candidate
        return next_t

    def _prepare_entry_reservation(self, entry: TimelineEntry) -> None:
        if entry.task_id in self._full_wall_fallback_task_ids:
            entry.allow_vram_mb = None
            entry.peak_start_time = None
            entry.peak_end_time = None
            entry.launch_uncertainty_sec = 0.0
        elif (
            not entry.is_init
            and entry.allow_vram_mb is None
            and entry.predicted_vram_mb > 0
        ):
            reservation = self._vram_reservation_model.reservation_for(
                component=entry.component,
                config_fingerprint=entry.config_fingerprint,
                gpu_id=entry.gpu_id,
                start_time=entry.start_time,
                predicted_end_time=entry.predicted_end_time,
                predicted_vram_mb=entry.predicted_vram_mb,
                input_size=entry.input_size,
            )
            entry.allow_vram_mb = reservation.allow_vram_mb
            entry.peak_start_time = reservation.peak_start_time
            entry.peak_end_time = reservation.peak_end_time

    def add_entry(
        self,
        entry: TimelineEntry,
        *,
        _reservation_prepared: bool = False,
    ) -> None:
        """Add a timeline entry, with universal predicted-entry dedup.

        Plan fix invariant: " task_id   timeline
           predicted entry  ".  ``add_predicted_entry``
        already enforces this at its entry point, but several internal
        projection sites — ``campaign_scheduler.on_task_submit`` /
        ``on_task_dispatched`` / ``_project_downstream`` /
        ``_schedule_downstream_pre_init`` — call ``add_entry`` directly
        with real ``task_id`` values (or structured derivatives), and
        those paths bypassed the dedup.  Retries / re-plans then piled
        duplicates into the timeline with future ``start_time`` values,
        which Layer-2 GC could not reap until ``start_time < now``
        — producing the "x17 / x25 / x19 …" accumulation observed in
        the ops dashboard.

        Policy (Plan fix extension):
          - Only predicted entries (``is_predicted=True``) trigger dedup.
          - Only non-completed, non-dispatching duplicates are dropped
            (Validator's dispatch-in-flight entries are sacrosanct —
            their try/finally owns them).
          - Empty ``task_id`` skips dedup (never a coherent identity).
          - Non-predicted entries (active dispatch records, completed
            init phase, killed fallbacks) pass through unchanged —
            those are not projections and have independent lifecycle.
        """
        if not _reservation_prepared:
            self._prepare_entry_reservation(entry)
        cached_index = self._scenario_event_index
        generation_before = self._scenario_generation()
        target_timeline = self._timelines.get(str(entry.gpu_id))
        can_extend_index = (
            cached_index is not None
            and cached_index.generation == generation_before
            and target_timeline is not None
        )
        dedup_mutated = False
        if entry.is_predicted and entry.task_id and not entry.is_dispatching:
            for tl in self._timelines.values():
                to_drop = [
                    e
                    for e in tl._entries
                    if e.task_id == entry.task_id
                    and e.is_predicted
                    and not e.is_completed
                    and not e.is_dispatching
                ]
                for e in to_drop:
                    with contextlib.suppress(ValueError):
                        tl._entries.remove(e)
                if to_drop:
                    dedup_mutated = True
                    tl._rebuild_component_index()
                    tl._bump_state_version()
        tl = self.get_or_create(entry.gpu_id)
        tl.add(entry)
        generation_after = self._scenario_generation()
        if (
            can_extend_index
            and not dedup_mutated
            and cached_index is not None
            and self._is_single_timeline_append_generation(
                generation_before,
                generation_after,
                str(entry.gpu_id),
            )
        ):
            extended = self._event_index_with_added_entry(
                cached_index,
                entry,
                generation_after,
            )
            if extended is not None:
                self._scenario_event_index = extended

    def complete_entry(self, task_id: str) -> TimelineEntry | None:
        """Mark a task as completed (keeps it visible for retention period).

        Must check ALL timelines before returning a fallback.
        GpuTimeline.complete() returns killed entries as fallback when no
        active entry is found on that GPU.  If we return early on the
        fallback, we miss the active entry on a different GPU — causing
        orphaned active entries that inflate active_count → deadlock.
        """
        self._full_wall_fallback_task_ids.discard(str(task_id or ""))
        fallback: TimelineEntry | None = None
        for tl in self._timelines.values():
            entry = tl.complete(task_id)
            if entry is not None:
                if entry.completed_at is not None and entry.killed_at is None:
                    return entry
                if fallback is None:
                    fallback = entry
        return fallback

    def kill_entries_for_worker(
        self,
        component: str,
        gpu_ids: list[str],
    ) -> list[TimelineEntry]:
        """Mark all active entries for *component* on *gpu_ids* as killed.

        Called when a worker is guard-killed. Returns the list of killed
        entries so the caller can update campaign queue stats.
        """
        killed: list[TimelineEntry] = []
        for gpu_id in gpu_ids:
            tl = self._timelines.get(gpu_id)
            if not tl:
                continue
            for e in list(tl.active_entries):
                if (
                    e.component == component
                    and not e.is_completed
                    and not e.is_predicted
                ):
                    e.killed_at = time.time()
                    e.predicted_end_time = e.killed_at
                    killed.append(e)
                    tl._retire_entry(e)
        return killed

    def find_entry(self, task_id: str) -> TimelineEntry | None:
        """Find an entry by task_id across all GPUs.

        Prefers active (non-completed) entries over completed/killed ones,
        so that eviction handlers find the currently running entry rather
        than a stale killed entry from a previous dispatch attempt.

        Uses a lazily-rebuilt task_id index (rebuilt once per scenario
        generation, then O(1) per lookup) so the candidate fit path's
        frequent lookups don't re-scan every timeline entry.
        """
        gen = self._scenario_generation()
        if gen != self._task_entry_index_gen:
            idx: dict[str, TimelineEntry] = {}
            for tl in self._timelines.values():
                for e in tl.active_entries:
                    idx.setdefault(e.task_id, e)
            for tl in self._timelines.values():
                for e in tl._retired_entries:
                    idx.setdefault(e.task_id, e)
            self._task_entry_index = idx
            self._task_entry_index_gen = gen
        return self._task_entry_index.get(task_id)

    def prune_stale_predicted(self, max_age_sec: float = 120.0) -> int:
        """Remove predicted entries whose predicted_end_time has long passed.

        Predicted entries are projections of future work.  If they haven't
        been replaced by an actual entry within ``max_age_sec`` after their
        ``predicted_end_time``, they are stale (the task was likely blocked,
        cancelled, or never dispatched).
        """
        import time as _time

        now = _time.time()
        removed = 0
        for tl in self._timelines.values():
            before = len(tl._entries)
            tl._entries = [
                e
                for e in tl._entries
                if not (
                    e.is_predicted
                    and not e.is_completed
                    and not e.is_dispatching
                    and e.predicted_end_time + max_age_sec < now
                )
            ]
            after = len(tl._entries)
            removed += before - after
            if before != after:
                tl._rebuild_component_index()
                tl._bump_state_version()
        return removed

    def remove_all_entries(self, task_id: str) -> int:
        """Remove non-terminal entries with this task_id across all GPUs.

        **Both killed AND completed entries are preserved** so that the
        ops timeline retains the full lifecycle history (e.g. a task that
        ran on GPU 1, hit CUDA OOM, and was retried on GPU 0 — the GPU 1
        run must remain visible as a killed/completed bar so the operator
        can see where the failure happened).  Without preserving
        completed entries, ``on_task_dispatched`` re-dispatch path wipes
        the prior GPU's history when the same ``task_id`` lands on a new
        GPU after an in-flight failure / cancel.

        Only **predicted/projected** entries (``is_predicted=True``) and
        active entries that never started (no ``start_time`` advance) are
        removed.  Returns count removed.
        """
        removed = 0
        for tl in self._timelines.values():
            before = len(tl._entries)
            tl._entries = [
                e
                for e in tl._entries
                if not (
                    e.task_id == task_id
                    and e.killed_at is None
                    and e.completed_at is None
                )
            ]
            after = len(tl._entries)
            removed += before - after
            if before != after:
                tl._rebuild_component_index()
                tl._bump_state_version()
        return removed

    def mark_entry_killed(
        self,
        task_id: str,
    ) -> TimelineEntry | None:
        """MCPSE eviction / cancel mark — preserve active entries on the
        timeline for ops-dashboard visualisation (red X badge) instead of
        stripping them.

        Two behaviours depending on the entry's state:

          * Active entry (``is_predicted=False``, ``killed_at=None``,
            ``completed_at=None``) → set ``killed_at`` + freeze
            ``predicted_end_time`` so the Gantt rendering shows the
            task terminating where the eviction landed.  The entry
            stays in the timeline.
          * Predicted entry (``is_predicted=True``) → remove outright.
            These never ran, so a "killed predicted" bar would be
            misleading — it would suggest wasted compute where none
            happened.

        This matches the user directive ("MCPSE evict / cancel  task 
        ops  killed     timeline entry   
        kill  ") and lets operators visually distinguish
        real eviction events from plan-time rescheduling.

        Returns the killed TimelineEntry (if any), else None.
        """
        killed: TimelineEntry | None = None
        now = time.time()
        for tl in self._timelines.values():
            before = len(tl._entries)
            tl._entries = [
                e for e in tl._entries if not (e.task_id == task_id and e.is_predicted)
            ]
            mutated = before != len(tl._entries)
            if mutated:
                tl._rebuild_component_index()
                tl._bump_state_version()
            if killed is None:
                for e in list(tl._entries):
                    if (
                        e.task_id == task_id
                        and not e.is_predicted
                        and e.killed_at is None
                        and e.completed_at is None
                    ):
                        e.killed_at = now
                        e.predicted_end_time = now
                        killed = e
                        tl._retire_entry(e)
                        break
        return killed

    def restore_killed_entry(
        self,
        task_id: str,
        *,
        killed_at_token: float,
        predicted_end_time: float,
    ) -> bool:
        """Undo ``mark_entry_killed`` when its cancel task failed.

        The token guards against racing with a later dispatch or terminal
        transition for the same task_id.  Only the exact killed marker created
        by the failed cancel attempt is restored.
        """
        for tl in self._timelines.values():
            for e in list(tl._retired_entries):
                if e.task_id != task_id:
                    continue
                if e.completed_at is not None:
                    continue
                if (
                    abs(
                        _checked_float(e.killed_at or 0.0)
                        - _checked_float(killed_at_token or 0.0)
                    )
                    > 1e-6
                ):
                    continue
                if (
                    abs(
                        _checked_float(e.predicted_end_time or 0.0)
                        - _checked_float(killed_at_token or 0.0)
                    )
                    > 1e-6
                ):
                    continue
                try:
                    tl._retired_entries.remove(e)
                except ValueError:
                    return False
                e.killed_at = None
                e.predicted_end_time = _checked_float(predicted_end_time or 0.0)
                tl._entries.append(e)
                tl._index_entry(e)
                tl._bump_state_version()
                return True
        return False

    def remove_entry(
        self,
        task_id: str,
        *,
        missing_ok: bool = False,
    ) -> TimelineEntry | None:
        """Immediately remove an **active** (non-killed, non-completed)
        entry.  Killed / completed entries are preserved for timeline
        visibility so VRAM prediction vs actual usage can be compared
        visually on the ops dashboard.

        Two-pass to avoid destructive side effects: tl.remove() pops
        from the list, so calling it on every timeline would delete
        killed entries on other GPUs as collateral.  Instead, search
        first (non-destructive), then pop only from the chosen timeline.

        Args ( P0-RUNTIME):
            missing_ok: When True, silently return None if no active
                entry is found.  Enables idempotent double-removal.
                Presence of killed/completed entries with the same
                task_id does NOT count as "found" — they must stay.
        """
        active_tl = None
        for tl in self._timelines.values():
            for e in tl.active_entries:
                if e.task_id == task_id:
                    active_tl = tl
                    break
            if active_tl is not None:
                break

        if active_tl is not None:
            return active_tl.remove(task_id)
        if missing_ok:
            return None
        raise KeyError(
            f"Predicted entry for task_id={task_id} not found",
        )


    def add_predicted_entry(
        self,
        task_id: str,
        component: str,
        gpu_id: str,
        worker_name: str,
        start_time: float,
        predicted_end_time: float,
        predicted_vram_mb: float = 0.0,
        predicted_ram_mb: float = 0.0,
        campaign_id: str = "",
        is_backfill: bool = False,
        is_dispatching: bool = False,
        input_size: float = 0.0,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        gpu_model: str = "",
        mps_mode: str = "",
        worker_backend: str = "",
        actor_model: str = "",
        adapter_version: str = "",
        was_primary_at_dispatch: bool = False,
        logical_batch_size: int = 0,
        execution_batch_size: int = 0,
    ) -> TimelineEntry:
        """Register a Planner ``DispatchPlan`` as a predicted timeline entry.

        Plan fix — **pre-insertion dedup**: remove any
        existing entry (predicted, not completed, not dispatching) for
        the same ``task_id`` before adding.  Prevents duplicate
        predicted entries on re-plan (Planner may call this same API
        for the same task_id across multiple solve() iterations when
        an earlier attempt's plan failed pre-dispatch).  Without this,
        stale projections accumulate in the timeline until Layer-2 GC
        catches them, polluting EST computation.

        The entry lives in ``predicted_entries`` (``is_predicted=True``) until
        either (a) ``promote_predicted_to_active`` — dispatch succeeded, or
        (b) a Layer-1/2/3 GC sweep removes it.  Next ``solve()`` iterations
        see this entry and compute EST off the updated timeline.
        """
        for tl in self._timelines.values():
            to_drop = [
                e
                for e in tl._entries
                if e.task_id == task_id
                and e.is_predicted
                and not e.is_completed
                and not e.is_dispatching
            ]
            if not to_drop:
                continue
            existing = to_drop[0]
            identical = (
                str(existing.gpu_id) == str(gpu_id)
                and abs(existing.start_time - _checked_float(start_time)) < 1e-6
                and abs(
                    existing.predicted_end_time - _checked_float(predicted_end_time)
                )
                < 1e-6
            )
            if identical:
                existing.prediction_stale = False
                existing.is_invalidated = False
                return existing
            for e in to_drop:
                with contextlib.suppress(ValueError):
                    tl._entries.remove(e)
            tl._rebuild_component_index()
            tl._bump_state_version()

        entry = TimelineEntry(
            task_id=task_id,
            component=component,
            campaign_id=campaign_id,
            gpu_id=str(gpu_id),
            start_time=_checked_float(start_time),
            predicted_end_time=_checked_float(predicted_end_time),
            predicted_vram_mb=_checked_float(predicted_vram_mb),
            predicted_ram_mb=_checked_float(predicted_ram_mb),
            is_backfill=bool(is_backfill),
            is_predicted=True,
            is_dispatching=bool(is_dispatching),
            input_size=_checked_float(input_size),
            config_fingerprint=str(config_fingerprint or ""),
            logical_batch_size=max(0, _checked_int(logical_batch_size or 0)),
            execution_batch_size=max(0, _checked_int(execution_batch_size or 0)),
            input_fingerprint=str(input_fingerprint or ""),
            gpu_model=str(gpu_model or ""),
            mps_mode=str(mps_mode or ""),
            worker_backend=str(worker_backend or ""),
            actor_model=str(actor_model or ""),
            adapter_version=str(adapter_version or ""),
            worker_name=worker_name,
            was_primary_at_dispatch=bool(was_primary_at_dispatch),
        )
        self._prepare_entry_reservation(entry)
        duration = max(0.0, entry.predicted_end_time - entry.start_time)
        if entry._has_valid_temporal_reservation() and duration > 0.0:
            assert entry.peak_start_time is not None
            assert entry.peak_end_time is not None
            entry.launch_uncertainty_sec = _temporal_peak_margin_sec(
                entry.peak_start_time,
                entry.peak_end_time,
                self._vram_reservation_z,
            )
            if entry.total_base_work_sec is None:
                entry.reciprocal_base_duration_sec = duration
        self.add_entry(entry, _reservation_prepared=True)
        return entry

    def get_predicted_entry(self, task_id: str) -> TimelineEntry | None:
        """Find the predicted entry for *task_id* across all GPUs.

        Returns a reference (not a copy).  Validator uses this to flip
        ``is_dispatching`` before/after gRPC InferBatch.
        """
        for tl in self._timelines.values():
            for e in tl._entries:
                if e.task_id == task_id and e.is_predicted and not e.is_completed:
                    return e
        return None

    def park_predicted_entry(
        self,
        task_id: str,
        *,
        component: str = "",
        gpu_id: str = "",
        worker_name: str = "",
    ) -> TimelineEntry | None:
        """Release a dispatch claim while retaining its exact prediction."""
        entry = self.get_predicted_entry(str(task_id))
        if entry is None or entry.is_completed:
            return None
        if (
            (component and entry.component != str(component))
            or (gpu_id and entry.gpu_id != str(gpu_id))
            or (worker_name and entry.worker_name != str(worker_name))
        ):
            return None
        if entry.prediction_stale or entry.is_invalidated or entry.is_evict_masked:
            return None
        if entry.is_dispatching:
            entry.is_dispatching = False
            timeline = self._timelines.get(entry.gpu_id)
            if timeline is not None:
                timeline._bump_state_version()
        return entry

    def mark_predicted_entry_dispatching(
        self,
        task_id: str,
        *,
        component: str = "",
        gpu_id: str = "",
    ) -> TimelineEntry:
        """Claim the one live prediction at the final launch handoff."""
        matches: list[tuple[GpuTimeline, TimelineEntry]] = []
        for timeline in self._timelines.values():
            matches.extend(
                (timeline, entry)
                for entry in timeline._entries
                if entry.task_id == str(task_id)
                and entry.is_predicted
                and not entry.is_completed
            )
        if len(matches) != 1:
            raise ValueError("launch prediction reservation missing or duplicate")
        timeline, entry = matches[0]
        if (component and entry.component != str(component)) or (
            gpu_id and entry.gpu_id != str(gpu_id)
        ):
            raise ValueError("launch prediction reservation topology mismatch")
        if not entry.is_dispatching:
            entry.is_dispatching = True
            timeline._bump_state_version()
        return entry

    def remove_predicted_entries_for_task(
        self,
        task_id: str,
        *,
        include_dispatching: bool = False,
    ) -> int:
        """Remove pure predicted entries for one task.

        Batch planning may create a future reservation before the HTTP
        handoff decides whether the handle is launchable in this wake.  If
        the handle is parked, that reservation must be rolled back
        immediately; otherwise PBBC/self-concurrency gates can count work
        that no worker will ever run.  Dispatch-in-flight predictions stay
        owned by ``RealityValidator`` unless explicitly requested.
        """
        task_key = str(task_id or "")
        if not task_key:
            return 0
        cached_index = self._scenario_event_index
        generation_before = self._scenario_generation()
        removed = 0
        for tl in self._timelines.values():
            to_drop = [
                e
                for e in tl._entries
                if e.task_id == task_key
                and e.is_predicted
                and not e.is_completed
                and (include_dispatching or not e.is_dispatching)
            ]
            if not to_drop:
                continue
            for entry in to_drop:
                try:
                    tl._entries.remove(entry)
                    removed += 1
                except ValueError:
                    pass
            tl._rebuild_component_index()
            tl._bump_state_version()
        if (
            removed
            and cached_index is not None
            and cached_index.generation == generation_before
        ):
            generation_after = self._scenario_generation()
            extended = self._event_index_with_removed_entries(
                cached_index,
                {task_key},
                generation_after,
            )
            if extended is not None:
                self._scenario_event_index = extended
        return removed

    def remove_predicted_entries_for_tasks(
        self,
        task_ids: list[str],
        *,
        include_dispatching: bool = False,
    ) -> int:
        """Remove a detached solve's predictions with one rebuild per GPU."""
        task_keys = {str(task_id or "") for task_id in task_ids}
        task_keys.discard("")
        if not task_keys:
            return 0
        cached_index = self._scenario_event_index
        generation_before = self._scenario_generation()
        removed = 0
        for timeline in self._timelines.values():
            retained: list[TimelineEntry] = []
            removed_here = 0
            for entry in timeline._entries:
                should_remove = (
                    entry.task_id in task_keys
                    and entry.is_predicted
                    and not entry.is_completed
                    and (include_dispatching or not entry.is_dispatching)
                )
                if should_remove:
                    removed_here += 1
                else:
                    retained.append(entry)
            if not removed_here:
                continue
            timeline._entries = retained
            timeline._rebuild_component_index()
            timeline._bump_state_version()
            removed += removed_here
        if (
            removed
            and cached_index is not None
            and cached_index.generation == generation_before
        ):
            generation_after = self._scenario_generation()
            extended = self._event_index_with_removed_entries(
                cached_index,
                task_keys,
                generation_after,
            )
            if extended is not None:
                self._scenario_event_index = extended
        return removed

    def get_predicted_entry_refs(self) -> list[TimelineEntry]:
        """All predicted (non-completed) entries across GPUs — references."""
        refs: list[TimelineEntry] = []
        for tl in self._timelines.values():
            for e in tl._entries:
                if e.is_predicted and not e.is_completed:
                    refs.append(e)
        return refs

    def promote_predicted_to_active(
        self,
        task_id: str,
    ) -> TimelineEntry | None:
        """Atomic predicted → active transition on dispatch success.

        Clears ``is_predicted`` and ``is_dispatching``, rewrites
        ``start_time`` to wall-clock now so EST/elapsed reflect the
        actual dispatch instant.  Returns the promoted entry or None.
        """
        entry = self.get_predicted_entry(task_id)
        if entry is None:
            return None
        now = time.time()
        old_start = entry.start_time
        duration = max(0.0, entry.predicted_end_time - old_start)
        temporal_shift = now - old_start
        if entry.peak_start_time is not None:
            entry.peak_start_time += temporal_shift
        if entry.peak_end_time is not None:
            entry.peak_end_time += temporal_shift
        entry.is_predicted = False
        entry.is_dispatching = False
        entry.launch_uncertainty_sec = 0.0
        entry.is_invalidated = False
        entry.is_evict_masked = False
        entry.start_time = now
        entry.predicted_end_time = now + duration
        if entry.total_base_work_sec is None:
            entry.total_base_work_sec = duration
        entry.remaining_base_work_sec = entry.total_base_work_sec
        entry.last_accounted_mono = time.monotonic()
        entry.current_reciprocal_multiplier = 1.0
        tl = self._timelines.get(str(entry.gpu_id))
        if tl is not None:
            tl._bump_state_version()
        return entry


    def remove_predicted_entries_for_worker(self, worker_name: str) -> int:
        """Layer-1 GC — worker dead / unreachable → drop its predicted entries."""
        if not worker_name:
            return 0
        removed = 0
        for tl in self._timelines.values():
            before = len(tl._entries)
            tl._entries = [
                e
                for e in tl._entries
                if not (
                    e.is_predicted
                    and not e.is_dispatching
                    and e.worker_name == worker_name
                )
            ]
            after = len(tl._entries)
            removed += before - after
            if before != after:
                tl._rebuild_component_index()
                tl._bump_state_version()
        return removed

    def remove_predicted_entries_for_gpu(self, gpu_id: str) -> int:
        """Layer-1 GC — GPU hang / Xid 79 / permanent failure."""
        tl = self._timelines.get(str(gpu_id))
        if tl is None:
            return 0
        before = len(tl._entries)
        tl._entries = [e for e in tl._entries if not e.is_predicted or e.is_dispatching]
        removed = before - len(tl._entries)
        if removed:
            tl._rebuild_component_index()
            tl._bump_state_version()
        return removed

    def prune_predicted_component_backlog(
        self,
        component: str,
        gpu_id: str,
        *,
        keep_count: int,
        exclude_task_id: str = "",
        drop_order_task_ids: list[str] | None = None,
    ) -> int:
        """Cap pure predicted backlog for ``component`` on one GPU."""
        tl = self._timelines.get(str(gpu_id))
        if tl is None:
            return 0
        return tl.prune_predicted_component_backlog(
            component,
            keep_count=keep_count,
            exclude_task_id=exclude_task_id,
            drop_order_task_ids=drop_order_task_ids,
        )


    def gc_orphaned_predicted_entries(self) -> int:
        """Reap predicted entries whose ``start_time`` has passed AND which
        are not currently being dispatched (``is_dispatching`` False).

        These are re-plan leftovers: a prior ``plan()`` iteration registered
        the entry, the plan failed (e.g., VRAM admission reject), a new
        plan replaced it — but the original entry was never removed.  After
        ``start_time`` the entry pollutes EST for subsequent tasks.

        Safety: active dispatches (``is_dispatching=True``) are preserved;
        the Validator's try/finally resets the flag so Layer-2 can catch
        truly-abandoned entries on the *next* outer loop.
        """
        now = time.time()
        removed = 0
        for tl in self._timelines.values():
            stale_ids = {
                id(e)
                for e in tl.active_entries
                if (e.is_predicted and not e.is_dispatching and e.start_time < now)
            }
            if stale_ids:
                before = len(tl._entries)
                tl._entries = [e for e in tl._entries if id(e) not in stale_ids]
                after = len(tl._entries)
                removed += before - after
                tl._rebuild_component_index()
                tl._bump_state_version()
        if removed:
            _LOG.debug(
                "[scenario] gc_orphaned_predicted_entries removed=%d",
                removed,
            )
        return removed


    def gc_stale_entries_safety_net(
        self,
        grace_period_sec: float = 30.0,
    ) -> int:
        """Layer-3 safety net.  Only reaps predicted entries whose
        ``predicted_end_time`` is already ``grace_period_sec`` in the past.

        Normal path (Layer 1 + 2) should catch everything first; a non-zero
        return here indicates an event-channel gap (worker_killed /
        eviction_complete / gpu_unhealthy / task_cancel) and a P0-level
        observability alert should fire.
        """
        now = time.time()
        removed = 0
        for tl in self._timelines.values():
            before = len(tl._entries)
            tl._entries = [
                e
                for e in tl._entries
                if not (
                    e.is_predicted
                    and not e.is_completed
                    and not e.is_dispatching
                    and e.predicted_end_time + grace_period_sec < now
                )
            ]
            after = len(tl._entries)
            removed += before - after
            if before != after:
                tl._rebuild_component_index()
                tl._bump_state_version()
        if removed:
            _LOG.warning(
                "[scenario] safety-net GC removed %d stale predicted entries "
                "(event channels may be broken — investigate)",
                removed,
            )
        return removed

    def reconcile_actual_entries(
        self,
        *,
        task_states: dict[str, int],
        task_updated_at: dict[str, float] | None = None,
        task_completion_pending: set[str] | None = None,
        state_succeeded: int,
        state_failed: int,
        state_cancelled: int,
        state_running: int,
        state_submitted: int,
        grace_period_sec: float = 30.0,
        terminal_state_grace_sec: float = 2.0,
    ) -> int:
        """Reconcile non-predicted active entries against gateway task state.

        This is a safety net for missed terminal callbacks. Predicted-entry
        GC handles plan-time leftovers; this method handles actual timeline
        entries that would otherwise linger forever and block planned
        occupancy after the corresponding TaskRecord has already gone
        terminal or disappeared.

        Policy:
          - ``SUCCEEDED`` -> mark completed
          - ``FAILED`` / ``CANCELLED`` -> mark killed
          - missing or non-running/submitted state only reconciles after
            ``predicted_end_time + grace_period_sec`` to avoid racing a slow
            state transition
        """
        now = time.time()
        updated_at = task_updated_at or {}
        completion_pending = task_completion_pending or set()
        reconciled = 0
        for tl in self._timelines.values():
            for entry in list(tl.active_entries):
                if entry.is_predicted:
                    continue
                if entry.is_init:
                    continue
                if entry.task_id in completion_pending:
                    continue
                state = task_states.get(entry.task_id)
                if state == state_succeeded:
                    if (
                        terminal_state_grace_sec > 0.0
                        and now
                        - _checked_float(updated_at.get(entry.task_id, 0.0) or 0.0)
                        < terminal_state_grace_sec
                    ):
                        continue
                    entry.completed_at = now
                    entry.predicted_end_time = now
                    self._full_wall_fallback_task_ids.discard(entry.task_id)
                    reconciled += 1
                    tl._retire_entry(entry)
                    continue
                if state in (state_failed, state_cancelled):
                    if (
                        terminal_state_grace_sec > 0.0
                        and now
                        - _checked_float(updated_at.get(entry.task_id, 0.0) or 0.0)
                        < terminal_state_grace_sec
                    ):
                        continue
                    entry.killed_at = now
                    entry.predicted_end_time = now
                    self._full_wall_fallback_task_ids.discard(entry.task_id)
                    reconciled += 1
                    tl._retire_entry(entry)
                    continue
                if state in (state_running, state_submitted):
                    continue
                if entry.predicted_end_time + grace_period_sec >= now:
                    continue
                entry.killed_at = now
                entry.predicted_end_time = now
                self._full_wall_fallback_task_ids.discard(entry.task_id)
                reconciled += 1
                tl._retire_entry(entry)
        if reconciled:
            _LOG.warning(
                "[scenario] reconciled %d orphaned actual timeline entries",
                reconciled,
            )
        return reconciled

    def clear_all_projected(self) -> int:
        """Remove all projected entries across all GPUs."""
        total = 0
        for tl in self._timelines.values():
            total += tl.clear_projected()
        return total

    def invalidate_component(
        self,
        component: str,
        *,
        metrics: set[str] | None = None,
    ) -> int:
        """Invalidate predictions for *component* across all GPUs.

        ``metrics`` (optional) selects which reservation dimensions are
        freed.  VRAM is freed only for VRAM-metric drift
        (``vram``/``interference_vram``); RAM/latency/interference drift
        frees occupancy/RAM but keeps the VRAM step-function (which those
        metrics never mutate).  ``metrics=None`` keeps the legacy behavior
        of freeing VRAM (fail-closed to the previous semantics).
        """
        free_vram = True
        if metrics is not None:
            free_vram = any(str(metric) in _VRAM_FREEING_METRICS for metric in metrics)
        total = 0
        for tl in self._timelines.values():
            if component not in tl._entries_by_component:
                continue
            total += tl.invalidate_predictions(component, free_vram=free_vram)
        return total

    def best_gpu_for(
        self,
        needed_vram_mb: float,
        exclude: set | None = None,
        prefer_gpu: str | None = None,
    ) -> str | None:
        """Return a gpu_id that can fit *needed_vram_mb*.

        Prefer *prefer_gpu* (e.g. a downstream task's parent GPU) when it has
        enough VRAM, so a campaign DAG chain stays co-located and free GPUs
        remain available for backfill.  Otherwise fall back to the GPU with
        the most available VRAM.
        """
        if prefer_gpu is not None:
            tl = self._timelines.get(str(prefer_gpu))
            if tl is not None and tl.available_vram_at() >= needed_vram_mb:
                return str(prefer_gpu)
        best_id: str | None = None
        best_avail = -1.0
        for gpu_id, tl in self._timelines.items():
            if exclude and gpu_id in exclude:
                continue
            avail = tl.available_vram_at()
            if avail >= needed_vram_mb and avail > best_avail:
                best_id = gpu_id
                best_avail = avail
        return best_id

    def available_vram_at(
        self,
        gpu_id: str,
        at_time: float | None = None,
        *,
        exclude_task_id: str = "",
    ) -> float:
        """Convenience wrapper for per-GPU VRAM availability."""
        tl = self._timelines.get(str(gpu_id))
        return (
            tl.available_vram_at(
                at_time,
                exclude_task_id=exclude_task_id,
            )
            if tl is not None
            else 0.0
        )

    def as_dict(
        self,
        *,
        include_retired: bool = True,
        retired_limit: int | None = 128,
    ) -> dict[str, Any]:
        return {
            gpu_id: tl.as_dict(
                include_retired=include_retired,
                retired_limit=retired_limit,
            )
            for gpu_id, tl in sorted(self._timelines.items())
        }

    def __len__(self) -> int:
        return len(self._timelines)
