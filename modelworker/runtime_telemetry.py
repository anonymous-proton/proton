from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .memory_observer import (
    MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE,
    MEMORY_BASIS_TOTAL_EXECUTION_PEAK,
    MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK,
    MEMORY_MEASUREMENT_PROCESS_TREE_PEAK,
    MEMORY_SOURCE_HOST_PROCESS_TREE,
    MEMORY_SOURCE_PROCESS_TREE,
    PEAK_FIDELITY_OBSERVED,
    BootstrapMemorySample,
    BootstrapMemorySummary,
    DispatchMemoryWindow,
    MemoryObservationFlags,
    MemorySource,
    dispatch_memory_summary,
    peak_memory_mib,
    summarize_temporal_vram_samples,
)

RESIDENT_MEMORY_SOURCE_REPLICA_BASELINE_CACHE = "replica_baseline_cache"
RESIDENT_BASELINE_STATE_MISSING = "missing"
RESIDENT_BASELINE_STATE_READY = "ready"
RESIDENT_BASELINE_STATE_STALE = "stale"
RESIDENT_BASELINE_STATE_INVALIDATED = "invalidated"
BYTES_PER_MIB = 1024.0 * 1024.0
_TEMPORAL_SAMPLE_LIMIT = 16_384

PidTreeProvider = Callable[[int], set[int]]
ProcessMemoryProvider = Callable[[Sequence[int]], Mapping[int, int]]
DeviceMemoryProvider = Callable[[], int]
RootPidProvider = Callable[[], int]

_LOG = logging.getLogger(__name__)


def mib_from_bytes(value: int | None) -> float | None:
    if value is None:
        return None
    return value / BYTES_PER_MIB


def normalize_pid(value: Any) -> int | None:
    try:
        pid = int(value)
    except Exception:
        return None
    return pid if pid > 0 else None


def normalize_gpu_id(value: Any) -> str:
    return str(value or "").strip()


def _normalize_gpu_expected_env(value: str) -> bool | None:
    token = str(value or "").strip().lower()
    if not token:
        return None
    return token not in {"none", "void", "no", "off", "-1"}


def _gpu_expected_from_env() -> bool:
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is None:
            continue
        parsed = _normalize_gpu_expected_env(value)
        if parsed is not None:
            return parsed
    return False


def _discover_pid_tree(root_pid: int) -> set[int]:
    normalized_root = normalize_pid(root_pid)
    if normalized_root is None:
        raise RuntimeError("worker root pid is invalid")
    children_by_parent: dict[int, set[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = normalize_pid(entry.name)
        if pid is None:
            continue
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                parent_pid: int | None = None
                for line in handle:
                    if line.startswith("PPid:"):
                        parent_pid = normalize_pid(line.split(":", 1)[1].strip())
                        break
        except OSError:
            continue
        if parent_pid is None:
            continue
        children_by_parent.setdefault(parent_pid, set()).add(pid)
    seen: set[int] = {normalized_root}
    stack = [normalized_root]
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, set()):
            if child in seen:
                continue
            seen.add(child)
            stack.append(child)
    return seen


def _read_process_rss_bytes(pid: int) -> int | None:
    normalized = normalize_pid(pid)
    if normalized is None:
        return None
    try:
        with open(f"/proc/{normalized}/status", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                kib = int(parts[1])
                return max(0, kib) * 1024
    except (OSError, ValueError):
        return None
    return None


def _resolve_host_rss_bytes(tracked_pids: Sequence[int]) -> Mapping[int, int]:
    out: dict[int, int] = {}
    for pid in tracked_pids:
        normalized = normalize_pid(pid)
        if normalized is None:
            continue
        value = _read_process_rss_bytes(normalized)
        if value is not None:
            out[normalized] = value
    return out


@dataclass(frozen=True)
class ResidentBaselineSnapshot:
    resident_memory_mib: float | None = None
    resident_memory_source: str = ""
    resident_baseline_collected_at: float | None = None
    resident_baseline_lifecycle_token: str = ""
    resident_baseline_state: str = RESIDENT_BASELINE_STATE_MISSING

    def is_ready(self) -> bool:
        return (
            self.resident_baseline_state == RESIDENT_BASELINE_STATE_READY
            and self.resident_memory_mib is not None
            and bool(self.resident_memory_source)
        )

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "resident_memory_mib": None,
            "resident_memory_source": "",
            "resident_baseline_collected_at": self.resident_baseline_collected_at,
            "resident_baseline_lifecycle_token": self.resident_baseline_lifecycle_token,
            "resident_baseline_state": self.resident_baseline_state,
        }
        if self.is_ready():
            payload["resident_memory_mib"] = self.resident_memory_mib
            payload["resident_memory_source"] = self.resident_memory_source
        return payload


@dataclass(frozen=True)
class MemoryTelemetry:
    active_vram_mib: float | None
    peak_vram_mib: float | None = None
    vram_memory_basis: str = MEMORY_BASIS_TOTAL_EXECUTION_PEAK
    vram_memory_measurement: str = MEMORY_MEASUREMENT_PROCESS_TREE_PEAK
    vram_memory_qc_keep: bool = False
    vram_memory_attribution: str = ""
    vram_resident_baseline: ResidentBaselineSnapshot = field(
        default_factory=ResidentBaselineSnapshot
    )
    shared_vram_resident_baseline: ResidentBaselineSnapshot = field(
        default_factory=ResidentBaselineSnapshot
    )
    host_active_memory_mib: float | None = None
    host_peak_memory_mib: float | None = None
    host_memory_basis: str = MEMORY_BASIS_TOTAL_EXECUTION_PEAK
    host_memory_measurement: str = MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK
    host_memory_qc_keep: bool = False
    host_memory_attribution: str = ""
    host_resident_baseline: ResidentBaselineSnapshot = field(
        default_factory=ResidentBaselineSnapshot
    )
    shared_host_resident_baseline: ResidentBaselineSnapshot = field(
        default_factory=ResidentBaselineSnapshot
    )
    concurrent_execute_overlap: bool = False
    bootstrap_memory_summary: BootstrapMemorySummary = field(
        default_factory=BootstrapMemorySummary
    )
    dispatch_memory_window: DispatchMemoryWindow = field(
        default_factory=DispatchMemoryWindow
    )
    host_bootstrap_memory_summary: BootstrapMemorySummary = field(
        default_factory=BootstrapMemorySummary
    )
    host_dispatch_memory_window: DispatchMemoryWindow = field(
        default_factory=DispatchMemoryWindow
    )
    peak_fidelity: str = PEAK_FIDELITY_OBSERVED
    host_peak_fidelity: str = PEAK_FIDELITY_OBSERVED
    actor_id: int | None = None
    actor_generation: int | None = None
    actor_pid: int | None = None
    actor_resident_owner: str = ""
    vram_source_complete: bool = False
    host_source_complete: bool = False
    device_pre_vram_mib: float | None = None
    device_peak_vram_mib: float | None = None
    device_post_vram_mib: float | None = None

    @property
    def active_memory_mib(self) -> float | None:
        return self.active_vram_mib

    @property
    def peak_memory_mib(self) -> float | None:
        return self.peak_vram_mib

    @property
    def memory_basis(self) -> str:
        return self.vram_memory_basis

    @property
    def active_memory_measurement(self) -> str:
        return self.vram_memory_measurement

    @property
    def memory_qc_keep(self) -> bool:
        return self.vram_memory_qc_keep

    @property
    def resident_baseline(self) -> ResidentBaselineSnapshot:
        return self.vram_resident_baseline

    @property
    def resident_memory_mib(self) -> float | None:
        return self.vram_resident_baseline.as_payload()["resident_memory_mib"]

    @property
    def resident_memory_source(self) -> str:
        return str(self.vram_resident_baseline.as_payload()["resident_memory_source"])

    @property
    def host_resident_memory_mib(self) -> float | None:
        return self.host_resident_baseline.as_payload()["resident_memory_mib"]

    def as_payload(self) -> dict[str, Any]:
        shared_vram = self.shared_vram_resident_baseline.resident_memory_mib
        actor_pre_vram = self.dispatch_memory_window.dispatch_pre_quiescent_mib
        device_unattributed_pre = None
        if self.device_pre_vram_mib is not None and shared_vram is not None:
            device_unattributed_pre = max(
                0.0,
                self.device_pre_vram_mib - shared_vram - (actor_pre_vram or 0.0),
            )
        device_unattributed_peak = None
        if (
            self.device_peak_vram_mib is not None
            and shared_vram is not None
            and self.peak_vram_mib is not None
        ):
            device_unattributed_peak = max(
                0.0,
                self.device_peak_vram_mib - shared_vram - self.peak_vram_mib,
            )
        payload = {
            "active_vram_mib": self.active_vram_mib,
            "peak_vram_mib": self.peak_vram_mib,
            "vram_memory_basis": self.vram_memory_basis,
            "vram_memory_measurement": self.vram_memory_measurement,
            "vram_memory_qc_keep": bool(self.vram_memory_qc_keep),
            "vram_memory_attribution": self.vram_memory_attribution,
            "host_active_memory_mib": self.host_active_memory_mib,
            "host_peak_memory_mib": self.host_peak_memory_mib,
            "host_memory_basis": self.host_memory_basis,
            "host_memory_measurement": self.host_memory_measurement,
            "host_memory_qc_keep": bool(self.host_memory_qc_keep),
            "host_memory_attribution": self.host_memory_attribution,
            "host_resident_memory_mib": self.host_resident_memory_mib,
            "host_resident_memory_source": str(
                self.host_resident_baseline.as_payload()["resident_memory_source"]
            ),
            "host_resident_baseline_collected_at": self.host_resident_baseline.as_payload()[
                "resident_baseline_collected_at"
            ],
            "host_resident_baseline_lifecycle_token": self.host_resident_baseline.as_payload()[
                "resident_baseline_lifecycle_token"
            ],
            "host_resident_baseline_state": self.host_resident_baseline.as_payload()[
                "resident_baseline_state"
            ],
            "host_dispatch_memory_window": self.host_dispatch_memory_window.as_dict(),
            "host_bootstrap_memory_summary": self.host_bootstrap_memory_summary.as_dict(),
            "host_peak_fidelity": self.host_peak_fidelity,
            "actor_id": self.actor_id,
            "actor_generation": self.actor_generation,
            "actor_pid": self.actor_pid,
            "actor_resident_owner": self.actor_resident_owner,
            "actor_resident_vram_mib": self.dispatch_memory_window.dispatch_pre_quiescent_mib,
            "actor_resident_host_mib": self.host_dispatch_memory_window.dispatch_pre_quiescent_mib,
            "shared_resident_vram_mib": self.shared_vram_resident_baseline.resident_memory_mib,
            "shared_resident_host_mib": self.shared_host_resident_baseline.resident_memory_mib,
            "vram_source_complete": bool(self.vram_source_complete),
            "host_source_complete": bool(self.host_source_complete),
            "device_pre_vram_mib": self.device_pre_vram_mib,
            "device_peak_vram_mib": self.device_peak_vram_mib,
            "device_post_vram_mib": self.device_post_vram_mib,
            "device_unattributed_pre_vram_mib": device_unattributed_pre,
            "device_unattributed_peak_vram_mib": device_unattributed_peak,
            "device_vram_measurement": "nvml_device_used",
            "active_memory_mib": self.active_vram_mib,
            "peak_memory_mib": self.peak_vram_mib,
            "memory_basis": self.vram_memory_basis,
            "active_memory_scope": self.vram_memory_basis,
            "active_memory_measurement": self.vram_memory_measurement,
            "peak_fidelity": self.peak_fidelity,
            "memory_qc_keep": bool(self.vram_memory_qc_keep),
            "concurrent_execute_overlap": bool(self.concurrent_execute_overlap),
        }
        payload.update(self.vram_resident_baseline.as_payload())
        return payload


@dataclass(frozen=True)
class DispatchAttribution:
    worker_generation_token: str = ""
    run_ordinal_in_generation: int = 0
    is_first_real_run: bool = False


@dataclass(eq=False)
class _MemoryChannelState:
    pid_tree_complete: bool = True
    sampling_complete: bool = True
    owned_pid_observed: bool = False
    peak_bytes: int | None = None
    pre_bytes: int | None = None
    post_bytes: int | None = None
    samples_seen: int = 0


@dataclass(eq=False)
class _DispatchSamplingState:
    root_pid: int
    started_at: float = field(default_factory=time.monotonic)
    started_at_epoch: float = field(default_factory=time.time)
    ended_at: float | None = None
    attribution: DispatchAttribution = field(default_factory=DispatchAttribution)
    memory_attribution: str = ""
    actor_id: int | None = None
    actor_generation: int | None = None
    actor_resident_owner: str = ""
    effective_dispatch_batch_size: int | None = None
    dispatch_group_size: int | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    sample_task: asyncio.Task[None] | None = None
    concurrent_execute_overlap: bool = False
    vram: _MemoryChannelState = field(default_factory=_MemoryChannelState)
    host: _MemoryChannelState = field(default_factory=_MemoryChannelState)
    active_request_count_at_start: int | None = None
    max_active_request_count_during_window: int | None = None
    device_pre_bytes: int | None = None
    device_peak_bytes: int | None = None
    device_post_bytes: int | None = None
    vram_samples: list[tuple[float, float]] = field(default_factory=list)
    vram_sample_buffer_full: bool = False


class WorkerTelemetryCollector:
    def __init__(
        self,
        *,
        root_pid_provider: RootPidProvider | None = None,
        pid_tree_provider: PidTreeProvider | None = None,
        process_memory_provider: ProcessMemoryProvider | None = None,
        vram_memory_provider: ProcessMemoryProvider | None = None,
        host_memory_provider: ProcessMemoryProvider | None = None,
        device_memory_provider: DeviceMemoryProvider | None = None,
        sample_interval_s: float = 0.01,
        lifecycle_id_prefix: str = "",
    ) -> None:
        self._root_pid_provider = root_pid_provider or os.getpid
        self._pid_tree_provider = pid_tree_provider or _discover_pid_tree
        self._vram_memory_provider = vram_memory_provider or process_memory_provider
        self._host_memory_provider = host_memory_provider or _resolve_host_rss_bytes
        self._device_memory_provider = device_memory_provider
        self._sample_interval_s = max(sample_interval_s, 0.001)
        self._startup_probed = False
        self._vram_enabled = self._vram_memory_provider is not None
        self._host_enabled = self._host_memory_provider is not None
        self._nvml: Any = None
        self._nvml_handles: list[Any] = []
        self._lock = asyncio.Lock()
        self._active_states: set[_DispatchSamplingState] = set()
        self._quiescent_refresh_task: asyncio.Task[None] | None = None
        self._lifecycle_id_prefix = (
            str(lifecycle_id_prefix or "").strip() or f"pid-{os.getpid()}"
        )
        self._lifecycle_generation = 0
        self._model_version = ""
        self._lifecycle_token = self._build_lifecycle_token()
        self._dispatch_attempt_count = 0
        self._vram_resident_baseline = ResidentBaselineSnapshot(
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_MISSING,
        )
        self._host_resident_baseline = ResidentBaselineSnapshot(
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_MISSING,
        )
        self._actor_vram_resident_mib: dict[tuple[int, int], float] = {}
        self._actor_host_resident_mib: dict[tuple[int, int], float] = {}
        self._bootstrap_started_at: float | None = None
        self._vram_bootstrap_samples: list[BootstrapMemorySample] = []
        self._host_bootstrap_samples: list[BootstrapMemorySample] = []
        self._vram_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
        )
        self._host_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
        )

    def _build_lifecycle_token(self) -> str:
        version = str(self._model_version or "unknown").strip() or "unknown"
        return f"{self._lifecycle_id_prefix}:{version}:{self._lifecycle_generation}"

    def _vram_memory_source(self, *, sampled: bool | None = None) -> MemorySource:
        return MemorySource(
            kind=MEMORY_SOURCE_PROCESS_TREE,
            process_private=True,
            peak_capable=True,
            sampled=bool(self._vram_enabled if sampled is None else sampled),
        )

    def _host_memory_source(self, *, sampled: bool | None = None) -> MemorySource:
        return MemorySource(
            kind=MEMORY_SOURCE_HOST_PROCESS_TREE,
            process_private=True,
            peak_capable=True,
            sampled=bool(self._host_enabled if sampled is None else sampled),
        )

    async def startup_probe(self) -> None:
        if self._startup_probed:
            return
        self._startup_probed = True
        if self._vram_memory_provider is not None:
            self._vram_enabled = True
            return
        gpu_expected = _gpu_expected_from_env()
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            handles = [
                nvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(int(nvml.nvmlDeviceGetCount()))
            ]
        except Exception as exc:
            if gpu_expected:
                raise RuntimeError(
                    f"NVML startup probe failed on GPU-backed worker: {exc}"
                ) from exc
            self._vram_enabled = False
            self._nvml = None
            self._nvml_handles = []
            return
        if not handles:
            try:
                nvml.nvmlShutdown()
            except Exception:
                _LOG.debug("NVML shutdown after empty probe failed", exc_info=True)
            if gpu_expected:
                raise RuntimeError(
                    "NVML startup probe found no visible GPUs on GPU-backed worker"
                )
            self._vram_enabled = False
            return
        self._nvml = nvml
        self._nvml_handles = handles
        self._vram_enabled = True

    def begin_lifecycle(self, *, model_version: str) -> None:
        self._lifecycle_generation += 1
        self._model_version = str(model_version or "").strip() or "unknown"
        self._lifecycle_token = self._build_lifecycle_token()
        self._dispatch_attempt_count = 0
        self._bootstrap_started_at = time.time()
        self._vram_bootstrap_samples = []
        self._host_bootstrap_samples = []
        self._actor_vram_resident_mib.clear()
        self._actor_host_resident_mib.clear()
        self._vram_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
            source=self._vram_memory_source(sampled=self._vram_enabled),
        )
        self._host_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
            source=self._host_memory_source(sampled=self._host_enabled),
        )
        self.invalidate_resident_baseline()

    def invalidate_resident_baseline(self) -> None:
        if self._quiescent_refresh_task is not None:
            self._quiescent_refresh_task.cancel()
            self._quiescent_refresh_task = None
        self._vram_resident_baseline = ResidentBaselineSnapshot(
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_INVALIDATED,
        )
        self._host_resident_baseline = ResidentBaselineSnapshot(
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_INVALIDATED,
        )

    def mark_resident_baseline_stale(self) -> None:
        snapshot = self._vram_resident_baseline
        self._vram_resident_baseline = ResidentBaselineSnapshot(
            resident_memory_mib=snapshot.resident_memory_mib,
            resident_memory_source=snapshot.resident_memory_source,
            resident_baseline_collected_at=snapshot.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_STALE,
        )
        host_snapshot = self._host_resident_baseline
        self._host_resident_baseline = ResidentBaselineSnapshot(
            resident_memory_mib=host_snapshot.resident_memory_mib,
            resident_memory_source=host_snapshot.resident_memory_source,
            resident_baseline_collected_at=host_snapshot.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_STALE,
        )

    async def shutdown(self) -> None:
        refresh_task = self._quiescent_refresh_task
        if refresh_task is not None:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
            self._quiescent_refresh_task = None
        async with self._lock:
            active_states = list(self._active_states)
        for state in active_states:
            state.stop_event.set()
            if state.sample_task is not None:
                await asyncio.gather(state.sample_task, return_exceptions=True)
        async with self._lock:
            self._active_states.clear()
        self.invalidate_resident_baseline()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                _LOG.debug("NVML shutdown failed", exc_info=True)
            self._nvml = None
            self._nvml_handles = []

    def get_resident_baseline_snapshot(self) -> ResidentBaselineSnapshot:
        return ResidentBaselineSnapshot(
            resident_memory_mib=self._vram_resident_baseline.resident_memory_mib,
            resident_memory_source=self._vram_resident_baseline.resident_memory_source,
            resident_baseline_collected_at=self._vram_resident_baseline.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=self._vram_resident_baseline.resident_baseline_lifecycle_token,
            resident_baseline_state=self._vram_resident_baseline.resident_baseline_state,
        )

    def get_host_resident_baseline_snapshot(self) -> ResidentBaselineSnapshot:
        return ResidentBaselineSnapshot(
            resident_memory_mib=self._host_resident_baseline.resident_memory_mib,
            resident_memory_source=self._host_resident_baseline.resident_memory_source,
            resident_baseline_collected_at=self._host_resident_baseline.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=self._host_resident_baseline.resident_baseline_lifecycle_token,
            resident_baseline_state=self._host_resident_baseline.resident_baseline_state,
        )

    def forget_actor(self, actor_id: int, generation: int) -> None:
        key = (actor_id, generation)
        self._actor_vram_resident_mib.pop(key, None)
        self._actor_host_resident_mib.pop(key, None)

    def _record_actor_resident(self, state: _DispatchSamplingState) -> None:
        if (
            state.memory_attribution != MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE
            or state.actor_id is None
            or state.actor_generation is None
        ):
            return
        key = (state.actor_id, state.actor_generation)
        vram_pre = mib_from_bytes(state.vram.pre_bytes)
        if state.vram.owned_pid_observed and vram_pre is not None:
            self._actor_vram_resident_mib[key] = vram_pre
        host_pre = mib_from_bytes(state.host.pre_bytes)
        if state.host.owned_pid_observed and host_pre is not None:
            self._actor_host_resident_mib[key] = host_pre

    def _combined_resident_baseline(
        self,
        shared: ResidentBaselineSnapshot,
        actor_values: Mapping[tuple[int, int], float],
    ) -> ResidentBaselineSnapshot:
        if not actor_values:
            return shared
        shared_mib = shared.resident_memory_mib if shared.is_ready() else 0.0
        return ResidentBaselineSnapshot(
            resident_memory_mib=(shared_mib or 0.0) + sum(actor_values.values()),
            resident_memory_source="parent_plus_actor_idle",
            resident_baseline_collected_at=time.time(),
            resident_baseline_lifecycle_token=self._lifecycle_token,
            resident_baseline_state=RESIDENT_BASELINE_STATE_READY,
        )

    def current_bootstrap_summary(self) -> BootstrapMemorySummary:
        return BootstrapMemorySummary.from_mapping(
            self._vram_bootstrap_summary.as_dict()
        )

    def current_host_bootstrap_summary(self) -> BootstrapMemorySummary:
        return BootstrapMemorySummary.from_mapping(
            self._host_bootstrap_summary.as_dict()
        )

    def get_dispatch_attempt_count(self) -> int:
        return self._dispatch_attempt_count

    async def note_real_dispatch_attempt(self) -> DispatchAttribution:
        async with self._lock:
            self._dispatch_attempt_count += 1
            ordinal = self._dispatch_attempt_count
            token = str(self._lifecycle_token or "").strip()
        return DispatchAttribution(
            worker_generation_token=token,
            run_ordinal_in_generation=ordinal,
            is_first_real_run=(ordinal == 1),
        )

    async def record_bootstrap_sample(self, phase: str) -> BootstrapMemorySample:
        collected_at = time.time()
        root_pid = normalize_pid(self._root_pid_provider())
        vram_total_mib = None
        host_total_mib = None
        active = bool(self._active_states)
        if root_pid is not None and not active:
            if self._vram_enabled:
                vram_total_mib = mib_from_bytes(
                    self._sample_root_process_bytes(root_pid, channel="vram")
                )
            if self._host_enabled:
                host_total_mib = mib_from_bytes(
                    self._sample_root_process_bytes(root_pid, channel="host")
                )
        vram_sample = BootstrapMemorySample(
            phase=str(phase or "").strip(),
            collected_at=collected_at,
            total_mib=vram_total_mib,
            source=self._vram_memory_source(
                sampled=self._vram_enabled and vram_total_mib is not None
            ),
        )
        host_sample = BootstrapMemorySample(
            phase=str(phase or "").strip(),
            collected_at=collected_at,
            total_mib=host_total_mib,
            source=self._host_memory_source(
                sampled=self._host_enabled and host_total_mib is not None
            ),
        )
        self._vram_bootstrap_samples.append(vram_sample)
        self._host_bootstrap_samples.append(host_sample)
        return vram_sample

    async def finalize_bootstrap_summary(self) -> BootstrapMemorySummary:
        vram_samples = tuple(self._vram_bootstrap_samples)
        host_samples = tuple(self._host_bootstrap_samples)
        vram_peak_total_mib = max(
            (
                sample.total_mib
                for sample in vram_samples
                if sample.total_mib is not None
            ),
            default=None,
        )
        host_peak_total_mib = max(
            (
                sample.total_mib
                for sample in host_samples
                if sample.total_mib is not None
            ),
            default=None,
        )
        vram_ready_floor_mib = vram_samples[-1].total_mib if vram_samples else None
        host_ready_floor_mib = host_samples[-1].total_mib if host_samples else None
        vram_baseline_snapshot = self.get_resident_baseline_snapshot()
        host_baseline_snapshot = self.get_host_resident_baseline_snapshot()
        if vram_ready_floor_mib is None and vram_baseline_snapshot.is_ready():
            vram_ready_floor_mib = vram_baseline_snapshot.resident_memory_mib
        if host_ready_floor_mib is None and host_baseline_snapshot.is_ready():
            host_ready_floor_mib = host_baseline_snapshot.resident_memory_mib
        end_ts = (
            vram_samples[-1].collected_at
            if vram_samples
            else host_samples[-1].collected_at
            if host_samples
            else time.time()
        )
        started_at = self._bootstrap_started_at
        bringup_total_ms = None
        if started_at is not None and end_ts is not None and end_ts >= started_at:
            bringup_total_ms = (end_ts - started_at) * 1000.0
        vram_sampling_incomplete = (not self._vram_enabled) or any(
            sample.total_mib is None for sample in vram_samples
        )
        host_sampling_incomplete = (not self._host_enabled) or any(
            sample.total_mib is None for sample in host_samples
        )
        self._vram_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
            bringup_total_ms=bringup_total_ms,
            bringup_peak_total_mib=vram_peak_total_mib,
            ready_quiescent_floor_mib=vram_ready_floor_mib,
            source=self._vram_memory_source(
                sampled=self._vram_enabled and bool(vram_samples)
            ),
            flags=MemoryObservationFlags(
                sampling_incomplete=vram_sampling_incomplete,
            ),
            samples=vram_samples,
        )
        self._host_bootstrap_summary = BootstrapMemorySummary(
            worker_generation_token=self._lifecycle_token,
            bringup_total_ms=bringup_total_ms,
            bringup_peak_total_mib=host_peak_total_mib,
            ready_quiescent_floor_mib=host_ready_floor_mib,
            source=self._host_memory_source(
                sampled=self._host_enabled and bool(host_samples)
            ),
            flags=MemoryObservationFlags(
                sampling_incomplete=host_sampling_incomplete,
            ),
            samples=host_samples,
        )
        return self.current_bootstrap_summary()

    def set_resident_baseline_snapshot(
        self, snapshot: ResidentBaselineSnapshot
    ) -> None:
        self._vram_resident_baseline = ResidentBaselineSnapshot(
            resident_memory_mib=snapshot.resident_memory_mib,
            resident_memory_source=snapshot.resident_memory_source,
            resident_baseline_collected_at=snapshot.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=(
                str(snapshot.resident_baseline_lifecycle_token or "").strip()
                or self._lifecycle_token
            ),
            resident_baseline_state=str(snapshot.resident_baseline_state or "").strip()
            or RESIDENT_BASELINE_STATE_MISSING,
        )

    def set_host_resident_baseline_snapshot(
        self, snapshot: ResidentBaselineSnapshot
    ) -> None:
        self._host_resident_baseline = ResidentBaselineSnapshot(
            resident_memory_mib=snapshot.resident_memory_mib,
            resident_memory_source=snapshot.resident_memory_source,
            resident_baseline_collected_at=snapshot.resident_baseline_collected_at,
            resident_baseline_lifecycle_token=(
                str(snapshot.resident_baseline_lifecycle_token or "").strip()
                or self._lifecycle_token
            ),
            resident_baseline_state=str(snapshot.resident_baseline_state or "").strip()
            or RESIDENT_BASELINE_STATE_MISSING,
        )

    async def refresh_resident_baseline(self) -> bool:
        if not (self._vram_enabled or self._host_enabled):
            return False
        async with self._lock:
            active_count = len(self._active_states)
            if active_count > 1:
                return False
            quality_state = (
                RESIDENT_BASELINE_STATE_READY
                if active_count == 0
                else RESIDENT_BASELINE_STATE_STALE
            )
        root_pid = normalize_pid(self._root_pid_provider())
        if root_pid is None:
            return False
        updated = False
        collected_at = time.time()
        if self._vram_enabled:
            vram_bytes = self._sample_root_process_bytes(root_pid, channel="vram")
            if vram_bytes is not None:
                self._vram_resident_baseline = ResidentBaselineSnapshot(
                    resident_memory_mib=mib_from_bytes(vram_bytes),
                    resident_memory_source=RESIDENT_MEMORY_SOURCE_REPLICA_BASELINE_CACHE,
                    resident_baseline_collected_at=collected_at,
                    resident_baseline_lifecycle_token=self._lifecycle_token,
                    resident_baseline_state=quality_state,
                )
                updated = True
        if self._host_enabled:
            host_bytes = self._sample_root_process_bytes(root_pid, channel="host")
            if host_bytes is not None:
                self._host_resident_baseline = ResidentBaselineSnapshot(
                    resident_memory_mib=mib_from_bytes(host_bytes),
                    resident_memory_source="host_process_tree_rss",
                    resident_baseline_collected_at=collected_at,
                    resident_baseline_lifecycle_token=self._lifecycle_token,
                    resident_baseline_state=quality_state,
                )
                updated = True
        return updated

    async def wait_for_pending_quiescent_refresh(self) -> None:
        task = self._quiescent_refresh_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def schedule_resident_baseline_refresh(self) -> None:
        if not (self._vram_enabled or self._host_enabled):
            return
        if (
            self._quiescent_refresh_task is not None
            and not self._quiescent_refresh_task.done()
        ):
            return
        self._quiescent_refresh_task = asyncio.create_task(
            self._run_quiescent_refresh()
        )

    async def _run_quiescent_refresh(self) -> None:
        try:
            await asyncio.sleep(0)
            await self.refresh_resident_baseline()
        finally:
            self._quiescent_refresh_task = None

    async def begin_dispatch(
        self,
        *,
        attribution: DispatchAttribution | None = None,
        effective_dispatch_batch_size: int | None = None,
        dispatch_group_size: int | None = None,
        root_pid: int | None = None,
        memory_attribution: str = "",
        actor_id: int | None = None,
        actor_generation: int | None = None,
        actor_resident_owner: str = "",
    ) -> _DispatchSamplingState:
        resolved_root = (
            normalize_pid(root_pid)
            or normalize_pid(self._root_pid_provider())
            or os.getpid()
        )
        state = _DispatchSamplingState(
            root_pid=resolved_root,
            attribution=attribution
            or DispatchAttribution(worker_generation_token=self._lifecycle_token),
            memory_attribution=str(memory_attribution or ""),
            actor_id=actor_id,
            actor_generation=actor_generation,
            actor_resident_owner=str(actor_resident_owner or ""),
            effective_dispatch_batch_size=effective_dispatch_batch_size,
            dispatch_group_size=dispatch_group_size,
        )
        if self._vram_enabled:
            state.vram.pre_bytes = self._sample_process_tree_bytes(
                resolved_root, state=state, channel="vram"
            )
            self._record_vram_sample(state, state.vram.pre_bytes)
        else:
            state.vram.pid_tree_complete = False
            state.vram.sampling_complete = False
        if self._host_enabled:
            state.host.pre_bytes = self._sample_process_tree_bytes(
                resolved_root, state=state, channel="host"
            )
        else:
            state.host.pid_tree_complete = False
            state.host.sampling_complete = False
        state.device_pre_bytes = self._resolve_device_used_bytes()
        state.device_peak_bytes = state.device_pre_bytes
        async with self._lock:
            self._active_states.add(state)
            active_count = len(self._active_states)
            state.active_request_count_at_start = active_count
            state.max_active_request_count_during_window = active_count
            if active_count > 1:
                state.concurrent_execute_overlap = True
                for active_state in self._active_states:
                    active_state.concurrent_execute_overlap = True
        if self._vram_enabled or self._host_enabled:
            state.sample_task = asyncio.create_task(self._run_dispatch_sampling(state))
        return state

    async def begin_execute(self) -> _DispatchSamplingState:
        return await self.begin_dispatch()

    async def end_dispatch(self, state: _DispatchSamplingState) -> MemoryTelemetry:
        state.stop_event.set()
        if state.sample_task is not None:
            await asyncio.gather(state.sample_task, return_exceptions=True)
        async with self._lock:
            self._active_states.discard(state)
        if self._vram_enabled:
            state.vram.post_bytes = self._sample_process_tree_bytes(
                state.root_pid, state=state, channel="vram"
            )
            self._record_vram_sample(state, state.vram.post_bytes)
        if self._host_enabled:
            state.host.post_bytes = self._sample_process_tree_bytes(
                state.root_pid, state=state, channel="host"
            )
        state.device_post_bytes = self._resolve_device_used_bytes()
        state.ended_at = time.monotonic()
        self._record_actor_resident(state)
        shared_vram_baseline = self.get_resident_baseline_snapshot()
        shared_host_baseline = self.get_host_resident_baseline_snapshot()
        combined_vram_baseline = self._combined_resident_baseline(
            shared_vram_baseline, self._actor_vram_resident_mib
        )
        combined_host_baseline = self._combined_resident_baseline(
            shared_host_baseline, self._actor_host_resident_mib
        )
        vram_window, vram_summary = self._build_dispatch_window(
            state=state,
            channel_state=state.vram,
            enabled=self._vram_enabled,
            baseline_snapshot=combined_vram_baseline,
            source=self._vram_memory_source(
                sampled=self._vram_enabled and state.vram.samples_seen > 0
            ),
        )
        host_window, host_summary = self._build_dispatch_window(
            state=state,
            channel_state=state.host,
            enabled=self._host_enabled,
            baseline_snapshot=combined_host_baseline,
            source=self._host_memory_source(
                sampled=self._host_enabled and state.host.samples_seen > 0
            ),
        )
        return MemoryTelemetry(
            active_vram_mib=vram_summary["active_memory_mib"],
            peak_vram_mib=vram_summary["peak_memory_mib"],
            vram_memory_basis=str(
                vram_summary["memory_basis"] or MEMORY_BASIS_TOTAL_EXECUTION_PEAK
            ),
            vram_memory_measurement=str(
                vram_summary["active_memory_measurement"]
                or MEMORY_MEASUREMENT_PROCESS_TREE_PEAK
            ),
            vram_memory_qc_keep=bool(vram_summary["memory_qc_keep"]),
            vram_memory_attribution=str(vram_summary.get("memory_attribution") or ""),
            vram_resident_baseline=combined_vram_baseline,
            shared_vram_resident_baseline=shared_vram_baseline,
            host_active_memory_mib=host_summary["active_memory_mib"],
            host_peak_memory_mib=host_summary["peak_memory_mib"],
            host_memory_basis=str(
                host_summary["memory_basis"] or MEMORY_BASIS_TOTAL_EXECUTION_PEAK
            ),
            host_memory_measurement=str(
                host_summary["active_memory_measurement"]
                or MEMORY_MEASUREMENT_HOST_PROCESS_TREE_RSS_PEAK
            ),
            host_memory_qc_keep=bool(host_summary["memory_qc_keep"]),
            host_memory_attribution=str(host_summary.get("memory_attribution") or ""),
            host_resident_baseline=combined_host_baseline,
            shared_host_resident_baseline=shared_host_baseline,
            concurrent_execute_overlap=bool(
                vram_summary["concurrent_execute_overlap"]
                or host_summary["concurrent_execute_overlap"]
            ),
            bootstrap_memory_summary=self.current_bootstrap_summary(),
            dispatch_memory_window=vram_window,
            host_bootstrap_memory_summary=self.current_host_bootstrap_summary(),
            host_dispatch_memory_window=host_window,
            peak_fidelity=str(vram_summary["peak_fidelity"] or PEAK_FIDELITY_OBSERVED),
            host_peak_fidelity=str(
                host_summary["peak_fidelity"] or PEAK_FIDELITY_OBSERVED
            ),
            actor_id=state.actor_id,
            actor_generation=state.actor_generation,
            actor_pid=state.root_pid if state.actor_id is not None else None,
            actor_resident_owner=state.actor_resident_owner,
            vram_source_complete=bool(
                not vram_window.flags.sampling_incomplete
                and not vram_window.flags.source_ambiguous
            ),
            host_source_complete=bool(
                not host_window.flags.sampling_incomplete
                and not host_window.flags.source_ambiguous
            ),
            device_pre_vram_mib=mib_from_bytes(state.device_pre_bytes),
            device_peak_vram_mib=mib_from_bytes(state.device_peak_bytes),
            device_post_vram_mib=mib_from_bytes(state.device_post_bytes),
        )

    def _build_dispatch_window(
        self,
        *,
        state: _DispatchSamplingState,
        channel_state: _MemoryChannelState,
        enabled: bool,
        baseline_snapshot: ResidentBaselineSnapshot,
        source: MemorySource,
    ) -> tuple[DispatchMemoryWindow, dict[str, Any]]:
        pre_mib = mib_from_bytes(channel_state.pre_bytes)
        post_mib = mib_from_bytes(channel_state.post_bytes)
        if pre_mib is None and baseline_snapshot.is_ready():
            pre_mib = baseline_snapshot.resident_memory_mib
        peak_total = peak_memory_mib(
            DispatchMemoryWindow(
                dispatch_peak_total_mib=mib_from_bytes(channel_state.peak_bytes),
            )
        )
        if peak_total is None:
            candidates = [value for value in (pre_mib, post_mib) if value is not None]
            peak_total = max(candidates) if candidates else None
        active_count = state.active_request_count_at_start or 1
        request_owned = (
            state.memory_attribution == MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE
        )
        attribution_count = 1 if request_owned else active_count
        lazy_materialization_detected = bool(
            state.attribution.is_first_real_run
            and attribution_count == 1
            and pre_mib is not None
            and post_mib is not None
            and post_mib > (pre_mib + 1.0)
        )
        temporal = None
        if channel_state is state.vram and enabled:
            temporal_resident_mib = (
                pre_mib if request_owned else baseline_snapshot.resident_memory_mib
            )
            temporal = summarize_temporal_vram_samples(
                state.vram_samples,
                wall_start=state.started_at,
                wall_end=state.ended_at,
                resident_mib=temporal_resident_mib,
                complete=not state.vram_sample_buffer_full
                and channel_state.sampling_complete
                and not lazy_materialization_detected,
            )
        flags = MemoryObservationFlags(
            lazy_materialization_detected=lazy_materialization_detected,
            overlap_ambiguous=bool(
                state.concurrent_execute_overlap and not request_owned
            ),
            sampling_incomplete=bool(
                (not enabled)
                or (not channel_state.sampling_complete)
                or channel_state.samples_seen <= 0
                or peak_total is None
            ),
            source_ambiguous=bool(
                not channel_state.pid_tree_complete
                or (request_owned and not channel_state.owned_pid_observed)
            ),
        )
        window = DispatchMemoryWindow(
            worker_generation_token=str(
                state.attribution.worker_generation_token or self._lifecycle_token
            ).strip(),
            run_ordinal_in_generation=(
                state.attribution.run_ordinal_in_generation or None
            ),
            is_first_real_run=bool(state.attribution.is_first_real_run),
            low_vram=temporal.get("low_vram") if temporal else None,
            peak_start_ratio=(temporal.get("peak_start_ratio") if temporal else None),
            peak_end_ratio=temporal.get("peak_end_ratio") if temporal else None,
            dispatch_pre_quiescent_mib=pre_mib,
            dispatch_peak_total_mib=peak_total,
            dispatch_post_quiescent_mib=post_mib,
            active_request_count_at_start=state.active_request_count_at_start,
            max_active_request_count_during_window=state.max_active_request_count_during_window,
            effective_dispatch_batch_size=state.effective_dispatch_batch_size,
            dispatch_group_size=state.dispatch_group_size,
            source=source,
            flags=flags,
            wall_duration_sec=(
                max(0.0, state.ended_at - state.started_at)
                if state.ended_at is not None and state.started_at is not None
                else None
            ),
            started_at_epoch=(
                float(state.started_at_epoch)
                if state.started_at_epoch is not None
                else None
            ),
        )
        summary = dispatch_memory_summary(window)
        summary["concurrent_execute_overlap"] = bool(state.concurrent_execute_overlap)
        if request_owned and summary["memory_qc_keep"]:
            summary["memory_attribution"] = MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE
        return window, summary

    async def end_execute(self, state: _DispatchSamplingState) -> MemoryTelemetry:
        return await self.end_dispatch(state)

    async def _run_dispatch_sampling(self, state: _DispatchSamplingState) -> None:
        await self._sample_dispatch_once(state)
        while not state.stop_event.is_set():
            try:
                await asyncio.wait_for(
                    state.stop_event.wait(), timeout=self._sample_interval_s
                )
                break
            except Exception as exc:
                if not isinstance(exc, asyncio.TimeoutError):
                    raise
                await self._sample_dispatch_once(state)

    async def _sample_dispatch_once(self, state: _DispatchSamplingState) -> None:
        if self._vram_enabled:
            self._sample_channel_once(state, state.vram, channel="vram")
        if self._host_enabled:
            self._sample_channel_once(state, state.host, channel="host")
        device_bytes = self._resolve_device_used_bytes()
        if device_bytes is not None and (
            state.device_peak_bytes is None or device_bytes > state.device_peak_bytes
        ):
            state.device_peak_bytes = device_bytes
        async with self._lock:
            state.max_active_request_count_during_window = max(
                state.max_active_request_count_during_window or 0,
                len(self._active_states),
            )

    def _record_vram_sample(
        self, state: _DispatchSamplingState, value: int | None
    ) -> None:
        if value is None:
            state.vram.sampling_complete = False
            return
        if len(state.vram_samples) >= _TEMPORAL_SAMPLE_LIMIT:
            state.vram_sample_buffer_full = True
            return
        sample_mib = mib_from_bytes(value)
        if sample_mib is None:
            state.vram.sampling_complete = False
            return
        state.vram_samples.append((time.monotonic(), sample_mib))

    def _sample_channel_once(
        self,
        state: _DispatchSamplingState,
        channel_state: _MemoryChannelState,
        *,
        channel: str,
    ) -> None:
        bytes_used = self._sample_process_tree_bytes(
            state.root_pid, state=state, channel=channel
        )
        if bytes_used is None:
            channel_state.sampling_complete = False
            return
        channel_state.samples_seen += 1
        if channel == "vram":
            self._record_vram_sample(state, bytes_used)
        if channel_state.peak_bytes is None or bytes_used > channel_state.peak_bytes:
            channel_state.peak_bytes = bytes_used

    def _sample_root_process_bytes(
        self,
        root_pid: int,
        *,
        channel: str,
    ) -> int | None:
        try:
            bytes_by_pid = self._resolve_process_bytes([root_pid], channel=channel)
        except Exception:
            return None
        value = bytes_by_pid.get(root_pid)
        if value is None:
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return None

    def _sample_process_tree_bytes(
        self,
        root_pid: int,
        *,
        state: _DispatchSamplingState | None = None,
        channel: str = "vram",
    ) -> int | None:
        tracked_pids: set[int]
        try:
            tracked_pids = {
                pid
                for pid in self._pid_tree_provider(root_pid)
                if normalize_pid(pid) is not None
            }
        except Exception:
            tracked_pids = {root_pid}
            if state is not None:
                self._channel_state(state, channel).pid_tree_complete = False
        if root_pid not in tracked_pids:
            tracked_pids.add(root_pid)
            if state is not None:
                self._channel_state(state, channel).pid_tree_complete = False
        try:
            bytes_by_pid = dict(
                self._resolve_process_bytes(sorted(tracked_pids), channel=channel)
            )
        except Exception:
            return None
        if state is not None and any(pid in bytes_by_pid for pid in tracked_pids):
            self._channel_state(state, channel).owned_pid_observed = True
        total_bytes = 0
        for pid in tracked_pids:
            value = bytes_by_pid.get(pid)
            if value is None:
                continue
            try:
                total_bytes += max(0, int(value))
            except Exception:
                continue
        return total_bytes

    @staticmethod
    def _channel_state(
        state: _DispatchSamplingState, channel: str
    ) -> _MemoryChannelState:
        return state.host if str(channel).strip().lower() == "host" else state.vram

    def _resolve_device_used_bytes(self) -> int | None:
        if self._device_memory_provider is not None:
            try:
                return max(0, int(self._device_memory_provider()))
            except (TypeError, ValueError, OverflowError, OSError):
                return None
        if self._nvml is None:
            return None
        total = 0
        try:
            for handle in self._nvml_handles:
                total += max(0, int(self._nvml.nvmlDeviceGetMemoryInfo(handle).used))
        except Exception:
            return None
        return total

    def _resolve_process_bytes(
        self, tracked_pids: Sequence[int], *, channel: str = "vram"
    ) -> Mapping[int, int]:
        if str(channel).strip().lower() == "host":
            if self._host_memory_provider is None:
                return {}
            return self._host_memory_provider(tracked_pids)
        if self._vram_memory_provider is not None:
            return self._vram_memory_provider(tracked_pids)
        if self._nvml is None:
            raise RuntimeError("NVML is not initialized")
        out: dict[int, int] = {}
        for handle in self._nvml_handles:
            for pid, bytes_used in self._query_device_process_bytes(handle).items():
                if pid not in tracked_pids:
                    continue
                out[pid] = out.get(pid, 0) + max(0, bytes_used)
        return out

    def _query_device_process_bytes(self, handle: Any) -> dict[int, int]:
        nvml = self._nvml
        if nvml is None:
            raise RuntimeError("NVML is not initialized")
        query_fns = [
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None),
        ]
        last_error: Exception | None = None
        for query_fn in query_fns:
            if query_fn is None:
                continue
            try:
                processes = query_fn(handle)
                break
            except Exception as exc:
                last_error = exc
                processes = None
                continue
        else:
            if last_error is not None:
                raise last_error
            processes = []
        out: dict[int, int] = {}
        for process in list(processes or []):
            pid = normalize_pid(getattr(process, "pid", None))
            if pid is None:
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used in (None, getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None)):
                continue
            try:
                out[pid] = out.get(pid, 0) + max(0, int(used))
            except Exception:
                continue
        return out


__all__ = [
    "DispatchAttribution",
    "MemoryTelemetry",
    "RESIDENT_BASELINE_STATE_INVALIDATED",
    "RESIDENT_BASELINE_STATE_MISSING",
    "RESIDENT_BASELINE_STATE_READY",
    "RESIDENT_BASELINE_STATE_STALE",
    "RESIDENT_MEMORY_SOURCE_REPLICA_BASELINE_CACHE",
    "ResidentBaselineSnapshot",
    "WorkerTelemetryCollector",
    "mib_from_bytes",
    "normalize_gpu_id",
    "normalize_pid",
]
