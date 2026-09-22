from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence



def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return bool(int(value))
        except (TypeError, ValueError, OverflowError):
            return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


MEMORY_SOURCE_PROCESS_TREE = "process_tree"
MEMORY_SOURCE_HOST_PROCESS_TREE = "host_process_tree"
MEMORY_SOURCE_DEVICE_WIDE = "device_wide"
MEMORY_MEASUREMENT_PROCESS_TREE_PEAK = "nvml_process_tree_peak"
MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK = "process_tree_rss_peak"
MEMORY_ATTRIBUTION_NONE = ""
MEMORY_ATTRIBUTION_N_WAY_OVERLAP_SHARE = "n_way_overlap_share"
MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE = "request_process_tree"
MEMORY_BASIS_TOTAL_EXECUTION_PEAK = "total_execution_peak"
MEMORY_BASIS_INCREMENT_OVER_RESIDENT = "increment_over_resident"
PEAK_FIDELITY_FULL = "full_peak"
PEAK_FIDELITY_INCREMENTAL = "incremental_peak"
PEAK_FIDELITY_OBSERVED = "observed_peak"


@dataclass(frozen=True)
class MemorySource:
    kind: str = ""
    process_private: bool = False
    peak_capable: bool = False
    sampled: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind or ""),
            "process_private": bool(self.process_private),
            "peak_capable": bool(self.peak_capable),
            "sampled": bool(self.sampled),
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "MemorySource":
        raw = _mapping(payload)
        return cls(
            kind=str(raw.get("kind") or ""),
            process_private=_to_bool(raw.get("process_private"), default=False),
            peak_capable=_to_bool(raw.get("peak_capable"), default=False),
            sampled=_to_bool(raw.get("sampled"), default=False),
        )


@dataclass(frozen=True)
class MemoryObservationFlags:
    lazy_materialization_detected: bool = False
    allocator_reuse_likely: bool = False
    overlap_ambiguous: bool = False
    sampling_incomplete: bool = False
    source_ambiguous: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lazy_materialization_detected": bool(self.lazy_materialization_detected),
            "allocator_reuse_likely": bool(self.allocator_reuse_likely),
            "overlap_ambiguous": bool(self.overlap_ambiguous),
            "sampling_incomplete": bool(self.sampling_incomplete),
            "source_ambiguous": bool(self.source_ambiguous),
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "MemoryObservationFlags":
        raw = _mapping(payload)
        return cls(
            lazy_materialization_detected=_to_bool(
                raw.get("lazy_materialization_detected"), default=False
            ),
            allocator_reuse_likely=_to_bool(
                raw.get("allocator_reuse_likely"), default=False
            ),
            overlap_ambiguous=_to_bool(raw.get("overlap_ambiguous"), default=False),
            sampling_incomplete=_to_bool(raw.get("sampling_incomplete"), default=False),
            source_ambiguous=_to_bool(raw.get("source_ambiguous"), default=False),
        )


@dataclass(frozen=True)
class BootstrapMemorySample:
    phase: str
    collected_at: Optional[float]
    total_mib: Optional[float]
    source: MemorySource = field(default_factory=MemorySource)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": str(self.phase or ""),
            "collected_at": self.collected_at,
            "total_mib": self.total_mib,
            "source": self.source.as_dict(),
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "BootstrapMemorySample":
        raw = _mapping(payload)
        return cls(
            phase=str(raw.get("phase") or ""),
            collected_at=_to_float(raw.get("collected_at")),
            total_mib=_to_float(raw.get("total_mib")),
            source=MemorySource.from_mapping(raw.get("source")),
        )


@dataclass(frozen=True)
class BootstrapMemorySummary:
    worker_generation_token: str = ""
    bringup_total_ms: Optional[float] = None
    bringup_peak_total_mib: Optional[float] = None
    ready_quiescent_floor_mib: Optional[float] = None
    source: MemorySource = field(default_factory=MemorySource)
    flags: MemoryObservationFlags = field(default_factory=MemoryObservationFlags)
    samples: tuple[BootstrapMemorySample, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "worker_generation_token": str(self.worker_generation_token or ""),
            "bringup_total_ms": self.bringup_total_ms,
            "bringup_peak_total_mib": self.bringup_peak_total_mib,
            "ready_quiescent_floor_mib": self.ready_quiescent_floor_mib,
            "source": self.source.as_dict(),
            "flags": self.flags.as_dict(),
            "samples": [item.as_dict() for item in self.samples],
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "BootstrapMemorySummary":
        raw = _mapping(payload)
        samples_raw = raw.get("samples")
        samples: tuple[BootstrapMemorySample, ...] = ()
        if isinstance(samples_raw, Sequence) and not isinstance(
            samples_raw, (str, bytes)
        ):
            samples = tuple(
                BootstrapMemorySample.from_mapping(item) for item in samples_raw
            )
        return cls(
            worker_generation_token=str(raw.get("worker_generation_token") or ""),
            bringup_total_ms=_to_float(raw.get("bringup_total_ms")),
            bringup_peak_total_mib=_to_float(raw.get("bringup_peak_total_mib")),
            ready_quiescent_floor_mib=_to_float(raw.get("ready_quiescent_floor_mib")),
            source=MemorySource.from_mapping(raw.get("source")),
            flags=MemoryObservationFlags.from_mapping(raw.get("flags")),
            samples=samples,
        )


@dataclass(frozen=True)
class DispatchMemoryWindow:
    worker_generation_token: str = ""
    low_vram: Optional[float] = None
    peak_start_ratio: Optional[float] = None
    peak_end_ratio: Optional[float] = None
    run_ordinal_in_generation: Optional[int] = None
    is_first_real_run: bool = False
    dispatch_pre_quiescent_mib: Optional[float] = None
    dispatch_peak_total_mib: Optional[float] = None
    dispatch_post_quiescent_mib: Optional[float] = None
    active_request_count_at_start: Optional[int] = None
    max_active_request_count_during_window: Optional[int] = None
    effective_dispatch_batch_size: Optional[int] = None
    dispatch_group_size: Optional[int] = None
    source: MemorySource = field(default_factory=MemorySource)
    flags: MemoryObservationFlags = field(default_factory=MemoryObservationFlags)
    wall_duration_sec: Optional[float] = None
    started_at_epoch: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "worker_generation_token": str(self.worker_generation_token or ""),
            "low_vram": self.low_vram,
            "peak_start_ratio": self.peak_start_ratio,
            "peak_end_ratio": self.peak_end_ratio,
            "run_ordinal_in_generation": self.run_ordinal_in_generation,
            "is_first_real_run": bool(self.is_first_real_run),
            "dispatch_pre_quiescent_mib": self.dispatch_pre_quiescent_mib,
            "dispatch_peak_total_mib": self.dispatch_peak_total_mib,
            "dispatch_post_quiescent_mib": self.dispatch_post_quiescent_mib,
            "active_request_count_at_start": self.active_request_count_at_start,
            "max_active_request_count_during_window": self.max_active_request_count_during_window,
            "effective_dispatch_batch_size": self.effective_dispatch_batch_size,
            "dispatch_group_size": self.dispatch_group_size,
            "source": self.source.as_dict(),
            "flags": self.flags.as_dict(),
            "wall_duration_sec": self.wall_duration_sec,
            "started_at_epoch": self.started_at_epoch,
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> "DispatchMemoryWindow":
        raw = _mapping(payload)
        return cls(
            worker_generation_token=str(raw.get("worker_generation_token") or ""),
            low_vram=_to_float(raw.get("low_vram")),
            peak_start_ratio=_to_float(raw.get("peak_start_ratio")),
            peak_end_ratio=_to_float(raw.get("peak_end_ratio")),
            run_ordinal_in_generation=_to_int(raw.get("run_ordinal_in_generation")),
            is_first_real_run=_to_bool(raw.get("is_first_real_run"), default=False),
            dispatch_pre_quiescent_mib=_to_float(raw.get("dispatch_pre_quiescent_mib")),
            dispatch_peak_total_mib=_to_float(raw.get("dispatch_peak_total_mib")),
            dispatch_post_quiescent_mib=_to_float(
                raw.get("dispatch_post_quiescent_mib")
            ),
            active_request_count_at_start=_to_int(
                raw.get("active_request_count_at_start")
            ),
            max_active_request_count_during_window=_to_int(
                raw.get("max_active_request_count_during_window")
            ),
            effective_dispatch_batch_size=_to_int(
                raw.get("effective_dispatch_batch_size")
            ),
            dispatch_group_size=_to_int(raw.get("dispatch_group_size")),
            source=MemorySource.from_mapping(raw.get("source")),
            flags=MemoryObservationFlags.from_mapping(raw.get("flags")),
            wall_duration_sec=_to_float(raw.get("wall_duration_sec")),
            started_at_epoch=_to_float(raw.get("started_at_epoch")),
        )


def summarize_temporal_vram_samples(
    samples: Sequence[tuple[float, float]],
    *,
    wall_start: float | None,
    wall_end: float | None,
    resident_mib: float | None,
    complete: bool,
) -> dict[str, float] | None:
    """Reduce bounded request-owned samples to the three temporal fields.

    The raw sequence stays request-local.  Any missing/invalid evidence returns
    ``None`` so callers retain the existing full-wall peak reservation.
    """
    if not complete or len(samples) < 2:
        return None
    start = _to_float(wall_start)
    end = _to_float(wall_end)
    resident = _to_float(resident_mib)
    if start is None or end is None or resident is None or end <= start:
        return None
    ordered = list(samples)
    active: list[tuple[float, float]] = []
    previous_ts = -math.inf
    for timestamp, total_mib in ordered:
        ts = _to_float(timestamp)
        total = _to_float(total_mib)
        if ts is None or total is None or ts < start or ts > end or ts < previous_ts:
            return None
        value = total - resident
        if value < 0 or not math.isfinite(value):
            return None
        active.append((ts, value))
        previous_ts = ts
    sorted_values = sorted(value for _, value in active)
    gaps = [
        (sorted_values[index + 1] - sorted_values[index], index)
        for index in range(len(sorted_values) - 1)
        if sorted_values[index + 1] > sorted_values[index]
    ]
    if not gaps:
        return None
    largest_gap, split_index = max(gaps, key=lambda item: item[0])
    other_gaps = [gap for gap, index in gaps if index != split_index]
    second_gap = max(other_gaps, default=0.0)
    if second_gap > 0 and largest_gap <= 2.0 * second_gap:
        return None
    lower_values = sorted_values[: split_index + 1]
    upper_values = sorted_values[split_index + 1 :]
    if len(lower_values) < 2 or len(upper_values) < 2:
        return None
    threshold = (sorted_values[split_index] + sorted_values[split_index + 1]) / 2.0
    low = max(lower_values)
    peak = max(upper_values)
    if not math.isfinite(low) or not math.isfinite(peak) or peak <= low:
        return None
    high = [(ts, value) for ts, value in active if value > threshold]
    if len(high) < 2:
        return None
    wall = end - start
    first_ratio = (high[0][0] - start) / wall
    last_ratio = (high[-1][0] - start) / wall
    if not 0.0 < first_ratio < last_ratio < 1.0:
        return None
    return {
        "low_vram": low,
        "peak_start_ratio": first_ratio,
        "peak_end_ratio": last_ratio,
    }


def infer_memory_basis(window: DispatchMemoryWindow) -> tuple[str, str]:
    peak = _to_float(window.dispatch_peak_total_mib)
    if peak is None:
        return (MEMORY_BASIS_TOTAL_EXECUTION_PEAK, PEAK_FIDELITY_OBSERVED)
    if (
        window.flags.lazy_materialization_detected
        or window.flags.allocator_reuse_likely
        or window.flags.overlap_ambiguous
        or window.dispatch_pre_quiescent_mib is None
    ):
        return (MEMORY_BASIS_TOTAL_EXECUTION_PEAK, PEAK_FIDELITY_FULL)
    pre = _to_float(window.dispatch_pre_quiescent_mib)
    if pre is None:
        return (MEMORY_BASIS_TOTAL_EXECUTION_PEAK, PEAK_FIDELITY_FULL)
    increment = peak - pre
    if increment <= 0:
        return (MEMORY_BASIS_TOTAL_EXECUTION_PEAK, PEAK_FIDELITY_FULL)
    return (MEMORY_BASIS_INCREMENT_OVER_RESIDENT, PEAK_FIDELITY_INCREMENTAL)


def active_memory_estimate_mib(window: DispatchMemoryWindow) -> Optional[float]:
    basis, _ = infer_memory_basis(window)
    peak = _to_float(window.dispatch_peak_total_mib)
    if peak is None:
        return None
    if basis == MEMORY_BASIS_TOTAL_EXECUTION_PEAK:
        return peak
    pre = _to_float(window.dispatch_pre_quiescent_mib)
    if pre is None:
        return peak
    return max(0.0, peak - pre)


def peak_memory_mib(window: DispatchMemoryWindow) -> Optional[float]:
    return _to_float(window.dispatch_peak_total_mib)


def source_measurement_name(window: DispatchMemoryWindow) -> str:
    kind = str(window.source.kind or "").strip().lower()
    if kind == MEMORY_SOURCE_DEVICE_WIDE:
        return "device_wide_peak"
    if kind == MEMORY_SOURCE_HOST_PROCESS_TREE:
        return MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK
    return MEMORY_MEASUREMENT_PROCESS_TREE_PEAK


def dispatch_memory_summary(window: DispatchMemoryWindow) -> Dict[str, Any]:
    memory_basis, peak_fidelity = infer_memory_basis(window)
    active_memory_mib = active_memory_estimate_mib(window)
    peak_total_mib = peak_memory_mib(window)
    flags = window.flags
    is_concurrent = bool(flags.overlap_ambiguous)
    memory_attribution = MEMORY_ATTRIBUTION_NONE
    base_qc_ok = bool(
        window.source.sampled
        and active_memory_mib is not None
        and not flags.sampling_incomplete
        and not flags.source_ambiguous
    )
    if is_concurrent:
        active_count = max(
            1,
            window.max_active_request_count_during_window
            or window.active_request_count_at_start
            or 1,
        )
        peak_total = _to_float(window.dispatch_peak_total_mib)
        pre_total = _to_float(window.dispatch_pre_quiescent_mib)
        if peak_total is not None and pre_total is not None:
            active_memory_mib = max(0.0, peak_total - pre_total)
            peak_total_mib = active_memory_mib
        if active_memory_mib is not None:
            active_memory_mib = active_memory_mib / active_count
        if peak_total_mib is not None:
            peak_total_mib = peak_total_mib / active_count
        memory_qc_keep = False
        if base_qc_ok and active_memory_mib is not None:
            memory_attribution = MEMORY_ATTRIBUTION_N_WAY_OVERLAP_SHARE
    elif base_qc_ok:
        memory_qc_keep = True
    else:
        memory_qc_keep = False
    if (
        not memory_qc_keep
        and memory_attribution != MEMORY_ATTRIBUTION_N_WAY_OVERLAP_SHARE
    ):
        active_memory_mib = None
        peak_total_mib = None
    return {
        "active_memory_mib": active_memory_mib,
        "peak_memory_mib": peak_total_mib,
        "active_memory_measurement": source_measurement_name(window),
        "memory_basis": memory_basis,
        "peak_fidelity": peak_fidelity,
        "total_upper_bound_mib": peak_total_mib,
        "memory_qc_keep": memory_qc_keep,
        "memory_attribution": memory_attribution,
        "memory_qc_concurrent": is_concurrent,
        "concurrent_execute_overlap": is_concurrent,
    }


__all__ = [
    "BootstrapMemorySample",
    "BootstrapMemorySummary",
    "DispatchMemoryWindow",
    "MEMORY_BASIS_INCREMENT_OVER_RESIDENT",
    "MEMORY_BASIS_TOTAL_EXECUTION_PEAK",
    "MEMORY_ATTRIBUTION_NONE",
    "MEMORY_ATTRIBUTION_N_WAY_OVERLAP_SHARE",
    "MEMORY_MEASUREMENT_PROCESS_TREE_PEAK",
    "MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK",
    "MEMORY_SOURCE_DEVICE_WIDE",
    "MEMORY_SOURCE_HOST_PROCESS_TREE",
    "MEMORY_SOURCE_PROCESS_TREE",
    "MemoryObservationFlags",
    "MemorySource",
    "PEAK_FIDELITY_FULL",
    "PEAK_FIDELITY_INCREMENTAL",
    "PEAK_FIDELITY_OBSERVED",
    "active_memory_estimate_mib",
    "dispatch_memory_summary",
    "infer_memory_basis",
    "peak_memory_mib",
    "source_measurement_name",
    "summarize_temporal_vram_samples",
]
