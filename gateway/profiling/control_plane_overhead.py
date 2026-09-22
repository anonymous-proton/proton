"""Flag-gated control-plane overhead recorder.

The recorder is intentionally inert unless explicitly enabled for a run.
It stores only in-memory aggregates and is flushed into ``signal_snapshots``
at shutdown or through the admin snapshot endpoint.
"""

from __future__ import annotations

import heapq
import threading
import time
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


SNAPSHOT_KEY = "control_plane_overhead_v1"


class ControlPlaneOverheadRecorder:
    """Small in-memory phase timer for one benchmark run."""

    def __init__(self, *, enabled: bool = False, top_k: int = 32) -> None:
        self.enabled = bool(enabled)
        self._top_k = max(0, int(top_k))
        self._created_at_wall = time.time()
        self._perf_anchor_ns = time.perf_counter_ns()
        self._wall_anchor_ns = time.time_ns()
        self._lock = threading.Lock()
        self._phases: dict[str, _PhaseStats] = defaultdict(_PhaseStats)
        self._active_phases: dict[str, _IntervalUnion] = defaultdict(_IntervalUnion)
        self._active_global = _IntervalUnion()
        self._slow: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        self._slow_seq = 0
        self._counters: dict[str, int] = defaultdict(int)

    def begin(self) -> int:
        if not self.enabled:
            return 0
        return time.perf_counter_ns()

    def record_since(
        self,
        phase: str,
        start_ns: int,
        *,
        active_wall: bool = False,
        **attrs: Any,
    ) -> None:
        if not self.enabled or not start_ns:
            return
        try:
            end_ns = time.perf_counter_ns()
            self.record_elapsed_ns(
                phase,
                max(0, end_ns - int(start_ns)),
                attrs=attrs,
                active_interval_ns=(int(start_ns), end_ns) if active_wall else None,
            )
        except Exception:
            return

    def record_elapsed_ns(
        self,
        phase: str,
        elapsed_ns: int,
        *,
        attrs: Mapping[str, Any] | None = None,
        active_interval_ns: tuple[int, int] | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            key = str(phase or "").strip() or "unknown"
            elapsed = max(0, int(elapsed_ns))
            with self._lock:
                self._phases[key].add(elapsed)
                if active_interval_ns is not None:
                    start_ns, end_ns = active_interval_ns
                    if end_ns > start_ns:
                        start_i = int(start_ns)
                        end_i = int(end_ns)
                        self._active_phases[key].add(start_i, end_i)
                        self._active_global.add(start_i, end_i)
                self._maybe_add_slow_sample_locked(key, elapsed, attrs or {})
        except Exception:
            return

    def _maybe_add_slow_sample_locked(
        self,
        key: str,
        elapsed_ns: int,
        attrs: Mapping[str, Any],
    ) -> None:
        if self._top_k <= 0:
            return
        bucket = self._slow[key]
        if len(bucket) >= self._top_k and elapsed_ns <= bucket[0][0]:
            return
        self._slow_seq += 1
        item = (int(elapsed_ns), int(self._slow_seq), _safe_attrs(attrs))
        if len(bucket) < self._top_k:
            heapq.heappush(bucket, item)
            return
        heapq.heapreplace(bucket, item)

    def increment(self, counter: str, value: int = 1, **attrs: Any) -> None:
        if not self.enabled:
            return
        try:
            name = str(counter or "").strip() or "unknown"
            with self._lock:
                self._counters[name] += int(value)
                if attrs:
                    for key, attr_value in attrs.items():
                        self._counters[
                            f"{name}.{key}.{_counter_value(attr_value)}"
                        ] += int(value)
        except Exception:
            return

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            phases = {
                key: values.summary() for key, values in sorted(self._phases.items())
            }
            active_phases = {
                key: values.summary()
                for key, values in sorted(self._active_phases.items())
            }
            phase_intervals_ns: dict[str, list[tuple[int, int]]] = {
                key: self._perf_intervals_to_wall(values.intervals)
                for key, values in sorted(self._active_phases.items())
            }
            global_intervals_ns = self._perf_intervals_to_wall(
                self._active_global.intervals
            )
            global_total_ns = self._active_global.total_ns
            phase_active_total_ns = {
                key: values.total_ns for key, values in self._active_phases.items()
            }
            active_union_ms = _ms(global_total_ns)
            slow = {key: list(values) for key, values in self._slow.items()}
            counters = dict(self._counters)
        phase_interval_durations = {
            key: _interval_total_ns(intervals)
            for key, intervals in phase_intervals_ns.items()
        }
        global_interval_duration = _interval_total_ns(global_intervals_ns)
        validation = {
            "control_plane_active_global_interval_ns": {
                "perf_total_ns": global_total_ns,
                "wall_total_ns": global_interval_duration,
                "interval_total_delta_ns": global_interval_duration - global_total_ns,
                "count": len(global_intervals_ns),
            },
            "control_plane_active_phase_interval_ns": {},
        }
        for key, intervals in phase_intervals_ns.items():
            perf_total_ns = phase_active_total_ns.get(key, 0)
            validation["control_plane_active_phase_interval_ns"][key] = {
                "perf_total_ns": perf_total_ns,
                "wall_total_ns": phase_interval_durations[key],
                "interval_total_delta_ns": phase_interval_durations[key]
                - perf_total_ns,
                "count": len(intervals),
            }
        return {
            "schema": SNAPSHOT_KEY,
            "enabled": self.enabled,
            "created_at_wall": self._created_at_wall,
            "exported_at_wall": time.time(),
            "perf_anchor_ns": self._perf_anchor_ns,
            "wall_anchor_ns": self._wall_anchor_ns,
            "phases": phases,
            "active_wall_phases": active_phases,
            "control_plane_active_intervals_wall_ns": global_intervals_ns,
            "control_plane_active_phase_intervals_wall_ns": phase_intervals_ns,
            "control_plane_active_interval_validation": validation,
            "slow_samples": {
                key: [
                    {
                        "elapsed_ms": round(float(elapsed_ns) / 1_000_000.0, 6),
                        "attrs": dict(attrs),
                    }
                    for elapsed_ns, _seq, attrs in sorted(values, reverse=True)
                ]
                for key, values in sorted(slow.items())
            },
            "counters": counters,
            "total_control_plane_sec": round(
                sum(
                    float(summary.get("total_ms") or 0.0) for summary in phases.values()
                )
                / 1000.0,
                6,
            ),
            "total_active_wall_sec": round(
                float(active_union_ms) / 1000.0,
                6,
            ),
        }

    def _to_wall_ns(self, perf_ns: int) -> int:
        return self._wall_anchor_ns + int(perf_ns) - self._perf_anchor_ns

    def _perf_intervals_to_wall(
        self, intervals: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return [
            (self._to_wall_ns(start), self._to_wall_ns(end)) for start, end in intervals
        ]


def _safe_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def _counter_value(value: Any) -> str:
    text = str(value).strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)[
        :80
    ]


class _PhaseStats:
    __slots__ = ("count", "total_ns", "min_ns", "max_ns")

    def __init__(self) -> None:
        self.count = 0
        self.total_ns = 0
        self.min_ns = 0
        self.max_ns = 0

    def add(self, elapsed_ns: int) -> None:
        elapsed = max(0, int(elapsed_ns))
        if self.count == 0:
            self.min_ns = elapsed
            self.max_ns = elapsed
        else:
            if elapsed < self.min_ns:
                self.min_ns = elapsed
            if elapsed > self.max_ns:
                self.max_ns = elapsed
        self.count += 1
        self.total_ns += elapsed

    def summary(self) -> dict[str, Any]:
        if self.count <= 0:
            return {
                "count": 0,
                "total_ms": 0.0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": 0.0,
            }
        return {
            "count": self.count,
            "total_ms": _ms(self.total_ns),
            "mean_ms": _ms(self.total_ns / self.count),
            "min_ms": _ms(self.min_ns),
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": _ms(self.max_ns),
        }


class _IntervalUnion:
    __slots__ = ("intervals", "raw_ns", "total_ns", "count")

    def __init__(self) -> None:
        self.intervals: list[tuple[int, int]] = []
        self.raw_ns = 0
        self.total_ns = 0
        self.count = 0

    def add(self, start_ns: int, end_ns: int) -> None:
        start = int(start_ns)
        end = int(end_ns)
        if end <= start:
            return
        self.count += 1
        self.raw_ns += end - start

        intervals = self.intervals
        if not intervals:
            intervals.append((start, end))
            self.total_ns += end - start
            return

        last_start, last_end = intervals[-1]
        if start >= last_end:
            intervals.append((start, end))
            self.total_ns += end - start
            return
        if start >= last_start:
            if end > last_end:
                intervals[-1] = (last_start, end)
                self.total_ns += end - last_end
            return

        idx = bisect_left(intervals, (start, end))
        if idx > 0 and intervals[idx - 1][1] >= start:
            idx -= 1
            start = min(start, intervals[idx][0])
            end = max(end, intervals[idx][1])
            self.total_ns -= intervals[idx][1] - intervals[idx][0]
            del intervals[idx]

        while idx < len(intervals) and intervals[idx][0] <= end:
            start = min(start, intervals[idx][0])
            end = max(end, intervals[idx][1])
            self.total_ns -= intervals[idx][1] - intervals[idx][0]
            del intervals[idx]

        intervals.insert(idx, (start, end))
        self.total_ns += end - start

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "raw_ms": _ms(self.raw_ns),
            "union_ms": _ms(self.total_ns),
            "merged_interval_count": len(self.intervals),
        }


def _ms(ns: float) -> float:
    return round(float(ns) / 1_000_000.0, 6)


def _interval_total_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(max(0, int(end) - int(start)) for start, end in intervals)
