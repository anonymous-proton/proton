"""Gateway-side worker supervisor that manages model worker containers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import getpass
import io
import logging
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import grpc
import yaml

import docker

from gateway.gpu_capacity import gpu_vram_fallback_mib
from gateway.policies.eviction_policy import (
    EvictionPolicyRegistry,
    LruEvictionPolicy,
)
from gateway.policies.policy import (
    GpuPolicy,
    LeastMemoryPolicy,
    PolicyRegistry,
)
from gateway.policies.recovery_policy import (
    MruRecoveryPolicy,
    RecoveryPolicyRegistry,
)
from gateway.signals.init_tracker import InitProfile
from modelworker import modelworker_pb2 as pb
from modelworker import modelworker_pb2_grpc as pb_grpc
from modelworker.runtime_telemetry import (
    RESIDENT_BASELINE_STATE_INVALIDATED,
    RESIDENT_BASELINE_STATE_MISSING,
    RESIDENT_BASELINE_STATE_READY,
    RESIDENT_BASELINE_STATE_STALE,
)

_LOG = logging.getLogger(__name__)


def _live_smi_poll_disabled() -> bool:
    value = os.environ.get("GW_DISABLE_SMI_POLL", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _BlockingCallTimeout(TimeoutError):
    """Raised when a startup-only blocking Docker SDK call exceeds its bound."""


def _call_blocking_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_s: float,
    **kwargs: Any,
) -> Any:
    """Run a blocking call on a daemon thread and bound how long we wait.

    Startup stale-container cleanup is best-effort.  It must never wedge the
    gateway because Docker SDK internals, container removal, or a test double
    failed to return cleanly.  Avoid ``asyncio.to_thread`` here: ``asyncio.run``
    waits for the default executor during shutdown, so even a timed-out call can
    keep tests and startup cleanup stuck.
    """

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise _BlockingCallTimeout()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _read_process_rss_mib(pid: int | None) -> float | None:
    """Best-effort host RSS lookup for a live worker PID.

    Reads ``/proc/<pid>/status`` on the host and returns VmRSS in MiB.
    Fail-open: any parse/race error returns ``None`` so callers can fall
    back to cached resident baselines.
    """
    if not pid:
        return None
    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                rss_kib = int(parts[1])
                if rss_kib <= 0:
                    return None
                return float(rss_kib) / 1024.0
    except Exception:
        return None
    return None


_REPO_ROOT = Path(__file__).resolve().parents[1]

_ADMISSION_STALL_DEFAULT_TTL_SEC = 10.0


@dataclass
class WorkerSpec:
    component: str
    name: str
    image: str
    gpus: list[str]
    grpc_port: int
    adapter: str
    env: dict[str, str]
    worker_server: dict[str, object]
    command: str | None = None
    volumes: dict[str, dict[str, str]] | None = None
    gpu_count: int = 1
    shm_size: str | None = None
    priority: int = 100
    max_instances: int | None = None
    grace_period_s: int = 0
    preemptible: bool = True
    vram_violation_threshold: int = 1
    length_args: dict[str, Any] = field(default_factory=dict)
    config_args: dict[str, Any] = field(default_factory=dict)
    fan_out_args: dict[str, Any] = field(default_factory=dict)
    input_size_key: str = ""
    dynamic_batching: dict[str, object] = field(
        default_factory=lambda: {"enabled": False}
    )
    oom_score_adj: int = 1000


@dataclass
class _MpsDaemonState:
    gpu_id: str
    name: str
    pipe_dir: Path
    log_dir: Path
    container_id: str | None = None


@dataclass
class WorkerState:
    spec: WorkerSpec
    container_id: str | None = None
    container_name: str | None = None
    host_port: int | None = None
    addr: str | None = None
    status: str = "Cold"
    lifecycle_state: str = "cold"
    ready: bool = False
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_error: str = ""
    caps: pb.Capabilities | None = None
    log_task: asyncio.Task | None = None
    assigned_gpus: list[str] = field(default_factory=list)
    profile_reserved: bool = False
    last_used_at: float = 0.0
    queue_in_queue: int = 0
    queue_prepared_queue: int = 0
    queue_output_queue: int = 0
    queue_prepare_inflight: int = 0
    queue_execute_inflight: int = 0
    queue_finalize_inflight: int = 0
    queue_gw_inflight: int = 0
    dispatch_pending: int = 0
    dispatch_pending_updated_at: float = 0.0
    dispatch_pending_handoff_protected: bool = False
    dispatch_pending_handoff_started_at: float = 0.0
    _front_seen_execute_inflight: int = 0
    memory_reserved_mb: int = 0
    is_guarded: bool = False
    current_activation_mb: int = 0
    activation_guarded_count: int = (
        0
    )
    max_concurrency: int = 0
    gpu_util_percent: float = 0.0
    last_ready_at: float = 0.0
    host_pid: int | None = None
    actual_vram_mb: int = 0
    violation_count: int = (
        0
    )
    resident_memory_mib: float | None = None
    resident_memory_source: str = ""
    resident_baseline_collected_at: float | None = None
    resident_baseline_lifecycle_token: str = ""
    resident_baseline_state: str = RESIDENT_BASELINE_STATE_MISSING
    real_dispatch_count_in_generation: int = 0
    draining: bool = (
        False
    )
    recovery_suppressed: bool = False


@dataclass
class IdleEvictableResourceSnapshot:
    idle_ram_mib: float
    idle_weight_by_gpu: dict[str, int]
    idle_count_by_gpu: dict[str, int]


@dataclass
class SupervisorConfig:
    specs: list[WorkerSpec]
    host: str = "127.0.0.1"
    readiness_timeout_s: float = 240.0
    stream_logs: bool = True
    stats_interval_s: float = (
        0.1
    )
    stats_report_interval_s: float = (
        30.0
    )
    stats_live: bool = True
    gpu_allocation: dict[str, object] | None = None
    profiling_runtime: dict[str, object] | None = None
    eviction_policy: str = "lru"
    recovery_policy: str = "mru"
    eviction_grace_period_s: float = 5.0
    weight_oom_policy: str = "optimistic"
    cold_start_activation_ratio: float = 1.0
    max_primary_slowdown: float = 0.5
    slowdown_cold_start_default: float = 1.0
    pairwise_slowdown_cold_start: float = (
        1.3
    )
    self_slowdown_cold_start_default: float = 2.0
    self_concurrency_max_primary_cold_start: int = 1
    self_concurrency_max_backfill_cold_start: int = 2
    gp_maturity_observations_per_dim: int = 10
    concurrency_gate_safety_factor: float = 1.0
    disable_self_interference_gate: bool = False
    disable_pbbc_gate: bool = False
    disable_primary_slo_protection: bool = False
    disable_slowdown_cost: bool = False

    admission_host_ram_min_available_mib: int = 2048
    host_ram_leak_idle_grace_sec: float = 10.0


def _as_str_map(env: dict[str, object]) -> dict[str, str]:
    return {str(k): str(v) for k, v in env.items()}


def _normalize_optional_positive_int(value: Any, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be a positive integer when provided") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer when provided")
    return parsed


def _normalize_host_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _workload_cfg(w: dict[str, object]) -> dict[str, object]:
    return dict(w.get("workload") or {})


_BATCH_ARG_RE = re.compile(
    r"(?:--?[A-Za-z0-9][A-Za-z0-9_-]*|[A-Za-z_][A-Za-z0-9_.-]*)\Z"
)


def _parse_dynamic_batching(value: object, component: str) -> dict[str, object]:
    if value is None:
        return {"enabled": False}
    if not isinstance(value, Mapping):
        raise SystemExit(f"workers[{component}].dynamic_batching must be an object")
    raw = dict(value)
    unknown = set(raw) - {"enabled", "batch_size_arg"}
    if unknown:
        raise SystemExit(
            f"workers[{component}].dynamic_batching has unsupported keys: "
            f"{', '.join(sorted(str(key) for key in unknown))}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise SystemExit(
            f"workers[{component}].dynamic_batching.enabled must be a boolean"
        )
    arg = str(raw.get("batch_size_arg") or "").strip()
    if arg and not _BATCH_ARG_RE.fullmatch(arg):
        raise SystemExit(
            f"workers[{component}].dynamic_batching.batch_size_arg must be one canonical argument name"
        )
    if not enabled:
        return {"enabled": False, **({"batch_size_arg": arg} if arg else {})}
    if not arg:
        raise SystemExit(
            f"workers[{component}].dynamic_batching.batch_size_arg must be one canonical argument name"
        )
    return {"enabled": True, "batch_size_arg": arg}


def _parse_worker_specs(
    workers: list[dict[str, object]], component: str
) -> list[WorkerSpec]:
    specs: list[WorkerSpec] = []
    for w in workers:
        name = str(w.get("name") or component).strip()
        if not name:
            raise SystemExit(f"workers[{component}] name must not be empty")

        image = str(w.get("image") or "").strip()
        if not image:
            raise SystemExit(f"workers[{component}] image is required")

        adapter = str(w.get("adapter") or "").strip()
        if not adapter:
            raise SystemExit(f"workers[{component}] adapter is required")

        volumes_raw = w.get("volumes") or []
        volumes: dict[str, dict[str, str]] = {}
        if isinstance(volumes_raw, dict):
            items = []
            for host_path, cfg in volumes_raw.items():
                if isinstance(cfg, dict):
                    bind = cfg.get("bind")
                else:
                    bind = cfg
                items.append((host_path, bind))
        else:
            items = []
            for entry in list(volumes_raw):
                if not isinstance(entry, str):
                    raise SystemExit(
                        "volumes entries must be strings like /host:/container"
                    )
                if ":" not in entry:
                    raise SystemExit(
                        "volumes entries must be strings like /host:/container"
                    )
                host_path, bind = entry.split(":", 1)
                items.append((host_path, bind))

        for host_path, bind in items:
            host_path_resolved = _normalize_host_path(str(host_path))
            bind_path = Path(bind)
            if not bind_path.is_absolute():
                bind_resolved = str(_normalize_host_path(str(bind)))
            else:
                bind_resolved = str(bind)
            volumes[str(host_path_resolved)] = {"bind": bind_resolved, "mode": "rw"}
        specs.append(
            WorkerSpec(
                component=component,
                name=name,
                image=image,
                gpus=[str(x) for x in (w.get("gpus") or [])],
                grpc_port=int(w.get("grpc_port", 0)),
                adapter=adapter,
                env=_as_str_map(w.get("env") or {}),
                worker_server=dict(w.get("worker_server") or {}),
                command=str(w["command"]) if w.get("command") else None,
                volumes=volumes or None,
                gpu_count=int(w.get("gpu_count", 1)),
                shm_size=str(w.get("shm_size")).strip()
                if w.get("shm_size") is not None
                else None,
                priority=int(w.get("priority", 100)),
                max_instances=_normalize_optional_positive_int(
                    w.get("max_instances"), "max_instances"
                ),
                grace_period_s=int(w.get("grace_period_s", 0)),
                preemptible=bool(w.get("preemptible", True)),
                vram_violation_threshold=int(w.get("vram_violation_threshold", 1)),
                length_args=dict(_workload_cfg(w).get("length_args") or {}),
                config_args=dict(_workload_cfg(w).get("config_args") or {}),
                fan_out_args=dict(_workload_cfg(w).get("fan_out_args") or {}),
                oom_score_adj=int(w.get("oom_score_adj", 1000)),
                input_size_key=str(_workload_cfg(w).get("input_size_key") or ""),
                dynamic_batching=_parse_dynamic_batching(
                    w.get("dynamic_batching"), component
                ),
            )
        )
    return specs


def _deep_merge_dict(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_shared_worker_defaults(
    raw_shared: object,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if raw_shared is None:
        return {}, {}
    if not isinstance(raw_shared, Mapping):
        raise SystemExit("shared must be an object when provided")

    shared_obj = dict(raw_shared)
    defaults_obj = shared_obj.get("workers")
    if defaults_obj is None:
        defaults_obj = {k: v for k, v in shared_obj.items() if k != "components"}
    if not isinstance(defaults_obj, Mapping):
        raise SystemExit("shared.workers must be an object when provided")

    component_defaults_raw = shared_obj.get("components") or {}
    if not isinstance(component_defaults_raw, Mapping):
        raise SystemExit("shared.components must be an object when provided")

    component_defaults: dict[str, dict[str, object]] = {}
    for raw_component, raw_cfg in component_defaults_raw.items():
        component = str(raw_component).strip().lower()
        if not component:
            continue
        if not isinstance(raw_cfg, Mapping):
            raise SystemExit(f"shared.components.{component} must be an object")
        component_defaults[component] = dict(raw_cfg)

    return dict(defaults_obj), component_defaults


def _parse_profiling_runtime(
    raw: object,
    *,
    gpu_allocation: Mapping[str, object] | None,
) -> dict[str, object]:
    cfg = dict(raw or {}) if isinstance(raw, Mapping) else {}
    isolation_mode = str(cfg.get("isolation_mode", "strict")).strip().lower()
    parallelism_mode = str(cfg.get("parallelism_mode", "per_gpu")).strip().lower()
    contention_policy = str(cfg.get("contention_policy", "fail_fast")).strip().lower()
    worker_pool_model = str(cfg.get("worker_pool_model", "unified")).strip().lower()
    worker_name_format = str(
        cfg.get("worker_name_format", "{component}-gpu{gpu_id}")
    ).strip()
    _default_db = str(
        Path(__file__).resolve().parents[1] / ".index" / "profile_runs.sqlite3"
    )
    index_db_path = str(cfg.get("index_db_path") or _default_db).strip()
    try:
        profile_max_inflight_per_worker = int(
            cfg.get("profile_max_inflight_per_worker", 0)
        )
    except Exception as exc:
        raise SystemExit(
            "profiling_runtime.profile_max_inflight_per_worker must be a non-negative integer (0 = unlimited)"
        ) from exc

    if isolation_mode not in {"strict"}:
        raise SystemExit("profiling_runtime.isolation_mode must be 'strict'")
    if parallelism_mode not in {"per_gpu"}:
        raise SystemExit("profiling_runtime.parallelism_mode must be 'per_gpu'")
    if contention_policy not in {"fail_fast"}:
        raise SystemExit("profiling_runtime.contention_policy must be 'fail_fast'")
    if worker_pool_model not in {"unified"}:
        raise SystemExit("profiling_runtime.worker_pool_model must be 'unified'")
    if "{component}" not in worker_name_format or "{gpu_id}" not in worker_name_format:
        raise SystemExit(
            "profiling_runtime.worker_name_format must include '{component}' and '{gpu_id}' placeholders"
        )

    pool = [str(x) for x in list((gpu_allocation or {}).get("pool") or [])]
    if parallelism_mode == "per_gpu" and not pool:
        raise SystemExit(
            "profiling_runtime.parallelism_mode=per_gpu requires gpu_allocation.pool"
        )
    if profile_max_inflight_per_worker < 0:
        raise SystemExit(
            "profiling_runtime.profile_max_inflight_per_worker must be a non-negative integer (0 = unlimited)"
        )

    parsed: dict[str, object] = dict(cfg)
    parsed.update(
        {
            "isolation_mode": isolation_mode,
            "parallelism_mode": parallelism_mode,
            "contention_policy": contention_policy,
            "worker_pool_model": worker_pool_model,
            "worker_name_format": worker_name_format,
            "gpu_pool": pool,
            "index_db_path": index_db_path,
            "profile_max_inflight_per_worker": profile_max_inflight_per_worker,
            "scheduler_policy": str(
                cfg.get("scheduler_policy", "campaign_fifo")
            ).strip(),
            "eviction_policy": str(cfg.get("eviction_policy", "lru")).strip().lower(),
            "recovery_policy": str(cfg.get("recovery_policy", "mru")).strip().lower(),
            "startup_penalty": float(cfg.get("startup_penalty", 2.0)),
            "weight_oom_policy": str(cfg.get("weight_oom_policy", "optimistic"))
            .strip()
            .lower(),
        }
    )
    return parsed


def _merge_configs(runtime_path: str, worker_path: str) -> dict[str, Any]:
    with open(runtime_path, encoding="utf-8") as rf:
        runtime_doc = yaml.safe_load(rf) or {}
    with open(worker_path, encoding="utf-8") as wf:
        worker_doc = yaml.safe_load(wf) or {}

    merged = {**worker_doc, **runtime_doc}
    return merged


def load_specs(
    runtime_path: str, worker_path: str, components: list[str]
) -> tuple[list[WorkerSpec], dict[str, object] | None, dict[str, object]]:
    doc = _merge_configs(runtime_path, worker_path)

    requested = [c.strip().lower() for c in components if c.strip()]
    if not requested:
        raise SystemExit("No component requested")

    specs: list[WorkerSpec] = []

    workers_list = list(doc.get("workers") or [])
    shared_worker_defaults, shared_component_defaults = _parse_shared_worker_defaults(
        doc.get("shared")
    )
    gpu_allocation = doc.get("gpu_allocation")
    profiling_runtime = _parse_profiling_runtime(
        doc.get("profiling_runtime"),
        gpu_allocation=gpu_allocation if isinstance(gpu_allocation, Mapping) else None,
    )
    raw_shared = doc.get("shared")
    if isinstance(raw_shared, Mapping):
        try:
            profiling_runtime["cold_init_latency_sec"] = float(
                raw_shared.get("cold_init_latency_sec", 10.0)
            )
        except (TypeError, ValueError):
            profiling_runtime["cold_init_latency_sec"] = 10.0
        try:
            profiling_runtime["cold_inference_latency_sec"] = float(
                raw_shared.get("cold_inference_latency_sec", 30.0)
            )
        except (TypeError, ValueError):
            profiling_runtime["cold_inference_latency_sec"] = 30.0
        try:
            profiling_runtime["self_interference_prior"] = float(
                raw_shared.get("self_interference_prior", 1.0)
            )
        except (TypeError, ValueError):
            profiling_runtime["self_interference_prior"] = 1.0
    else:
        profiling_runtime["cold_init_latency_sec"] = 10.0
        profiling_runtime["cold_inference_latency_sec"] = 30.0
        profiling_runtime["self_interference_prior"] = 1.0
    if not workers_list:
        raise SystemExit("workers.yaml must define a top-level workers list")

    for worker in workers_list:
        comp_raw = str(worker.get("component", "")).strip().lower()
        if not comp_raw:
            if len(requested) == 1:
                comp_raw = requested[0]
            else:
                raise SystemExit("workers entries must include component")
        if comp_raw not in requested:
            continue
        merged = _deep_merge_dict(
            shared_worker_defaults, shared_component_defaults.get(comp_raw, {})
        )
        merged = _deep_merge_dict(merged, dict(worker))
        specs.extend(_parse_worker_specs([merged], comp_raw))

    dynamic_by_component: dict[str, dict[str, object]] = {}
    for spec in specs:
        previous = dynamic_by_component.setdefault(
            spec.component, dict(spec.dynamic_batching)
        )
        if previous != spec.dynamic_batching:
            raise SystemExit(
                f"workers[{spec.component}] resolved dynamic_batching values must agree"
            )

    specs = _expand_specs_for_runtime(
        specs,
        gpu_allocation=gpu_allocation if isinstance(gpu_allocation, Mapping) else None,
        profiling_runtime=profiling_runtime,
    )

    return specs, gpu_allocation, profiling_runtime


def _expand_specs_for_runtime(
    specs: list[WorkerSpec],
    *,
    gpu_allocation: Mapping[str, object] | None,
    profiling_runtime: Mapping[str, object],
) -> list[WorkerSpec]:
    policy_name = (
        str((gpu_allocation or {}).get("policy", "least_memory_used")).strip().lower()
    )
    gpu_pool = [
        str(x).strip()
        for x in list((gpu_allocation or {}).get("pool") or [])
        if str(x).strip()
    ]
    worker_name_format = str(
        profiling_runtime.get("worker_name_format") or "{component}-gpu{gpu_id}"
    )
    if policy_name != "on_demand" or not gpu_pool:
        return list(specs)

    expanded: list[WorkerSpec] = []
    for spec in specs:
        if spec.gpus:
            expanded.append(spec)
            continue

        if int(spec.gpu_count or 0) != 1:
            raise SystemExit(
                f"on_demand requires gpu_count=1 for worker {spec.component}/{spec.name} "
                "(multi-GPU workers are not supported in unified lazy mode)"
            )

        for gpu_id in gpu_pool:
            clone = replace(spec)
            clone.name = worker_name_format.format(
                component=spec.component, gpu_id=gpu_id
            )
            clone.gpus = [gpu_id]
            clone.gpu_count = 0
            expanded.append(clone)
    return expanded


def discover_components(worker_path: str) -> list[str]:
    with open(worker_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    workers_list = list(doc.get("workers") or [])
    comps = {
        str(worker.get("component", "")).strip().lower()
        for worker in workers_list
        if str(worker.get("component", "")).strip()
    }
    return sorted(comps)


_port_lock = threading.Lock()
_recently_allocated: set[int] = set()


def pick_free_port(host: str) -> int:
    """Allocate a free port, guarding against concurrent duplicate allocation."""
    with _port_lock:
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, 0))
                port = int(sock.getsockname()[1])
            if port not in _recently_allocated:
                _recently_allocated.add(port)
                return port
        _recently_allocated.add(port)
        return port


def release_port(port: int) -> None:
    """Remove a port from the recently-allocated set after container binds it."""
    with _port_lock:
        _recently_allocated.discard(port)


@dataclass(order=True)
class AdmissionRequest:
    campaign_arrival: float
    priority: int
    timestamp: float
    needed_mb: int = field(compare=False)
    gpu_ids: list[str] = field(compare=False)
    future: asyncio.Future[Any] | None = field(default=None, compare=False)
    type: str = field(default="activation", compare=False)
    campaign_id: str = field(default="", compare=False)
    is_backfill: bool = field(default=False, compare=False)
    host_ram_mb: int = field(default=0, compare=False)
    task_id: str = field(default="", compare=False)
    required_duration_sec: float = field(default=0.0, compare=False)
    campaign_scheduler: Any = field(default=None, compare=False, repr=False)


@dataclass
class _ComputeWaiter:
    """FIFO waiter for GPU compute slot admission.

    Sort key matches AdmissionRequest: (is_backfill, campaign_arrival, priority, timestamp).
    """

    is_backfill: bool
    campaign_arrival: float
    priority: int
    timestamp: float
    event: asyncio.Event = field(default_factory=asyncio.Event)


def _read_mem_available_mib() -> int:
    """Read /proc/meminfo MemAvailable in MiB (kernel-cached, ~microseconds).

    Linux 3.14+ exposes ``MemAvailable`` as the kernel's estimate of
    memory that can be allocated to new applications without swapping —
    free + reclaimable cache/buffers − reserved.  This is the same signal
    K8s kubelet uses for ``--eviction-hard=memory.available<N``.

    Returns a large sentinel (1 PiB) on read failure (``/proc/meminfo``
    missing, parse error) — **fail-open** so that observability bugs
    do not block admission.  Read failures are exceedingly rare on Linux
    (the file is always present in any non-bare-metal-kernel build).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1 << 30


class HostRAMSaturatedError(Exception):
    """Host RAM admission gate rejected admission ().

    Raised by ``ResourceAdmissionTracker.acquire_activation`` when the real-time
    ``/proc/meminfo MemAvailable`` falls below
    ``admission_host_ram_min_available_mib``.  Distinct from:

      * ``MemoryAdmissionError`` — worker-specific runtime VRAM guard
        rejection (per-worker policy).
      * Plain ``False`` return from ``acquire_activation`` — GPU VRAM
        shortfall (cluster-wide, after idle eviction).

    K8s ``--eviction-hard=memory.available<N`` parity; Slurm
    ``MemSpecLimit`` parity.  Cluster-wide transient host RAM pressure
    — always retry-able via the wake-trigger / periodic_tick framework.
    Caller (http_server / reality_validator) catches and converts to
    ``ConstraintViolation(host_ram_saturated)``.
    """

    def __init__(
        self,
        mem_available_mib: int,
        threshold_mib: int,
        gpu_id: str = "",
        worker_name: str = "",
    ) -> None:
        self.mem_available_mib = int(mem_available_mib)
        self.threshold_mib = int(threshold_mib)
        self.gpu_id = str(gpu_id or "")
        self.worker_name = str(worker_name or "")
        super().__init__(
            f"host RAM saturated: MemAvailable={mem_available_mib} MiB "
            f"< threshold={threshold_mib} MiB"
        )


class ResourceAdmissionTracker:
    """Tracks dispatch-time resource admission for the gateway.

    The primary axis is still per-GPU VRAM accounting, but this tracker
    also owns the host-RAM admission gate because activation admission
    is where both resource checks meet.

    Uses two accounting sources:
      - reserved_vram : Gateway-managed bookkeeping (weight + activation reservations).
      - _smi_used_vram: Periodically sampled from nvidia-smi (memory.used).

    Effective available memory = total - max(reserved, smi_used)

    This floor-based approach ensures that VRAM consumed by processes outside
    the Gateway (other users, system daemons, etc.) is automatically reflected
    in admission decisions without requiring the external processes to cooperate.
    This is the same pattern used by production inference servers such as
    NVIDIA Triton and TensorRT-LLM.
    """

    SMI_POLL_INTERVAL_S: float = 0.2
    SMI_STALE_THRESHOLD_S: float = (
        0.3
    )

    VRAM_DIAG = bool(os.environ.get("GW_VRAM_DIAG", ""))

    def __init__(
        self,
        gpu_pool: list[str],
        *,
        host_ram_min_available_mib: int = 0,
    ) -> None:
        """
        Args:
            gpu_pool: List of GPU id strings managed by this gateway.
            host_ram_min_available_mib: Single cluster-wide threshold
                (MiB) for the host-RAM admission gate.  0 disables the
                gate; ``acquire_activation`` then never raises
                ``HostRAMSaturatedError``.  Wired from
                ``SupervisorConfig.admission_host_ram_min_available_mib``.
        """
        self.gpu_pool = gpu_pool
        self._host_ram_min_available_mib = int(host_ram_min_available_mib)
        self.total_vram: dict[str, int] = {}
        self.reserved_vram: dict[
            str, int
        ] = {}
        self.active_vram: dict[
            str, int
        ] = {}
        self._host_ram_reserved_mb: int = 0
        self._smi_used_vram: dict[
            str, int
        ] = {}
        self._smi_gpu_util: dict[
            str, int
        ] = {}
        self._smi_last_poll_at: float = 0.0
        self._last_runtime_activity_ts: float = time.time()
        self._lock = asyncio.Lock()
        self._diag_last_dump: float = 0.0
        self._cv = asyncio.Condition(self._lock)
        self._initialized = False
        self._queues: dict[str, list[AdmissionRequest]] = {gid: [] for gid in gpu_pool}
        self._campaign_arrivals: dict[
            str, float
        ] = {}
        self._loop_task: asyncio.Task | None = None
        self._smi_poll_task: asyncio.Task | None = None
        self._compute_waiters: dict[str, list[_ComputeWaiter]] = {
            gid: [] for gid in gpu_pool
        }
        self._estimate_idle_host_ram_freeable_fn: Any | None = None
        self._evict_idle_workers_for_host_ram_fn: Any | None = None

    def estimate_idle_host_ram_freeable(self) -> int:
        fn = self._estimate_idle_host_ram_freeable_fn
        if not callable(fn):
            return 0
        try:
            return max(0, int(fn() or 0))
        except Exception:
            _LOG.warning(
                "[admission-host-ram] estimate_idle_host_ram_freeable failed",
                exc_info=True,
            )
            return 0

    async def evict_idle_workers_for_host_ram(
        self,
        needed_mib: int,
        *,
        suppress_recovery: bool = False,
    ) -> int:
        fn = self._evict_idle_workers_for_host_ram_fn
        if not callable(fn):
            return 0
        try:
            freed = fn(int(needed_mib or 0))
            if asyncio.iscoroutine(freed):
                freed = await freed
            return max(0, int(freed or 0))
        except Exception:
            _LOG.warning(
                "[admission-host-ram] evict_idle_workers_for_host_ram failed",
                exc_info=True,
            )
            return 0

    def _campaign_arrival_time(self, campaign_id: str) -> float:
        """Return first-seen time for a campaign, recording it if new.

        Empty campaign_id returns the current time so that unrelated requests
        never share a campaign arrival bucket.
        """
        if not campaign_id:
            return time.time()
        arrival = self._campaign_arrivals.get(campaign_id)
        if arrival is None:
            arrival = time.time()
            self._campaign_arrivals[campaign_id] = arrival
        return arrival

    def cleanup_campaign(self, campaign_id: str) -> None:
        """Remove a completed campaign from the arrival tracker."""
        self._campaign_arrivals.pop(campaign_id, None)

    @property
    def reserved_host_ram_mb(self) -> int:
        return int(self._host_ram_reserved_mb)

    async def pending_admission_count(self) -> int:
        """Return queued resource-admission waiters across GPUs.

        This includes both memory-admission requests and compute waiters.
        Host-RAM ledger recovery must treat any waiter as non-idle because
        clearing a reservation while an admission request is still alive can
        undercut the same owner/release semantics used for VRAM.
        """
        await self.ensure_initialized()
        async with self._lock:
            return int(
                sum(len(q) for q in self._queues.values())
                + sum(len(w) for w in self._compute_waiters.values())
            )

    async def clear_reserved_host_ram(self) -> int:
        """Force-clear leaked host RAM reservations.

        This is only safe when the caller has already established that no
        dispatch / execute activity and no resource-admission waiters remain
        anywhere in the cluster.  It is intended as a diagnostic recovery valve
        for ledger drift, not normal admission.
        """
        await self.ensure_initialized()
        async with self._lock:
            cleared = int(max(0, self._host_ram_reserved_mb))
            if cleared <= 0:
                return 0
            self._host_ram_reserved_mb = 0
            self._process_queues_locked()
            return cleared

    def _host_ram_shortfall_locked(
        self,
        needed_mb: int,
        mem_available_mib: int | None = None,
    ) -> int:
        mem_avail = int(
            _read_mem_available_mib()
            if mem_available_mib is None
            else mem_available_mib,
        )
        required = max(0, self._host_ram_min_available_mib)
        required += max(0, self._host_ram_reserved_mb)
        required += max(0, int(needed_mb or 0))
        return max(0, required - mem_avail)

    def _available_host_ram_locked(
        self,
        mem_available_mib: int | None = None,
    ) -> int:
        mem_avail = int(
            _read_mem_available_mib()
            if mem_available_mib is None
            else mem_available_mib,
        )
        usable = (
            mem_avail
            - max(0, self._host_ram_min_available_mib)
            - max(0, self._host_ram_reserved_mb)
        )
        return max(0, int(usable))

    async def _scheduling_loop(self):
        """Periodic loop to ensure stuck queues are processed."""
        while True:
            await asyncio.sleep(2.0)
            async with self._lock:
                self._process_queues_locked()

    async def _smi_poll_loop(self):
        """Periodically reads nvidia-smi memory.used and updates the external-usage floor.

        The floor prevents Gateway from overcommitting VRAM to processes it controls
        when another user (or daemon) is already consuming part of the GPU memory.
        On each poll we update _smi_used_vram and re-process pending admission queues
        in case freeing of external memory unblocked a waiting task.
        """
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return
        while True:
            await asyncio.sleep(self.SMI_POLL_INTERVAL_S)
            await self._do_smi_poll(nvidia_smi)

    async def _do_smi_poll(self, nvidia_smi: str | None = None) -> bool:
        """Execute one nvidia-smi query and update the floor. Returns True on success."""
        if _live_smi_poll_disabled():
            self._smi_last_poll_at = time.time()
            return False
        if nvidia_smi is None:
            nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return False
        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            updated: dict[str, int] = {}
            reader = csv.reader(io.StringIO(result.stdout))
            for row in reader:
                if not row or len(row) < 2:
                    continue
                idx, used = row[0].strip(), int(row[1].strip())
                gpu_util = int(row[2].strip()) if len(row) > 2 else 0
                if idx in self.gpu_pool:
                    updated[idx] = used
                    self._smi_gpu_util[idx] = gpu_util
            if updated:
                async with self._lock:
                    changed = False
                    for gid, used in updated.items():
                        old = self._smi_used_vram.get(gid, 0)
                        self._smi_used_vram[gid] = used
                        if used < old:
                            changed = True
                    self._smi_last_poll_at = time.time()
                    if changed:
                        self._process_queues_locked()
            return True
        except Exception as exc:
            _LOG.debug("[ResourceAdmissionTracker] smi poll error: %s", exc)
            return False

    async def _ensure_smi_fresh(self) -> None:
        """Called at admission time: if the cached smi data is stale, do a fresh poll.

        This closes the race window where an external container spikes GPU memory
        between two background poll cycles. The window is bounded by SMI_POLL_INTERVAL_S
        under normal conditions, but admission time is the only point where a stale
        reading can cause an actual OOM — so we refresh inline when needed.
        """
        age = time.time() - self._smi_last_poll_at
        if age > self.SMI_STALE_THRESHOLD_S:
            await self._do_smi_poll()

    @property
    def pool(self) -> list[str]:
        return self.gpu_pool

    def get_available_memory(self, gpu_id: str) -> int:
        """Returns the conservatively estimated available VRAM for gpu_id.

        Formula: Total - max(reserved, smi)

        - reserved: Gateway bookkeeping (weight + activation reservations).
        - smi: Actual GPU memory used (nvidia-smi).

        Using max() instead of sum avoids double-counting: between
        acquire_weight() and mark_active(), the same memory appears in
        both reserved (immediately) and smi (as the model loads).  The
        old formula ``reserved + max(0, smi - active)`` double-counted
        during this window, artificially reducing available VRAM and
        causing admission deadlocks.
        """
        total = self.total_vram.get(gpu_id, 0)
        reserved = self.reserved_vram.get(gpu_id, 0)
        smi = self._smi_used_vram.get(gpu_id, 0)
        effective_used = max(reserved, smi)
        return max(0, total - effective_used)

    def _diag_gpu(self, gpu_id: str, context: str = "") -> None:
        """Emit detailed VRAM accounting for a single GPU (GW_VRAM_DIAG=1)."""
        if not self.VRAM_DIAG:
            return
        total = self.total_vram.get(gpu_id, 0)
        reserved = self.reserved_vram.get(gpu_id, 0)
        active = self.active_vram.get(gpu_id, 0)
        smi = self._smi_used_vram.get(gpu_id, 0)
        external = max(0, smi - active)
        effective = reserved + external
        avail = max(0, total - effective)
        q = self._queues.get(gpu_id, [])
        q_info = [(r.type, r.needed_mb, r.campaign_id[:8]) for r in q[:5]]
        _LOG.warning(
            "[vram-diag] GPU %s %s: total=%d reserved=%d active=%d smi=%d "
            "external=%d effective_used=%d avail=%d queue=%d items=%s",
            gpu_id,
            context,
            total,
            reserved,
            active,
            smi,
            external,
            effective,
            avail,
            len(q),
            q_info,
        )

    def _diag_dump_all(self, context: str = "") -> None:
        """Dump VRAM state for all GPUs (rate-limited to every 10s)."""
        if not self.VRAM_DIAG:
            return
        now = time.time()
        if now - self._diag_last_dump < 10.0:
            return
        self._diag_last_dump = now
        for gid in self.gpu_pool:
            self._diag_gpu(gid, context)

    def get_queue_length(self, gpu_id: str) -> int:
        return len(self._queues.get(str(gpu_id), []))

    def gpus_with_pending_requests(self) -> set:
        """Return GPU IDs that have unsatisfied VRAM admission requests."""
        return {gpu_id for gpu_id, q in self._queues.items() if q}

    def get_max_pending_mb(self, gpu_id: str) -> int:
        """Return the largest needed_mb among pending requests for *gpu_id*."""
        q = self._queues.get(str(gpu_id), [])
        return max((req.needed_mb for req in q), default=0)

    async def ensure_initialized(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return

            nvidia_smi = (
                None if _live_smi_poll_disabled() else shutil.which("nvidia-smi")
            )
            if not nvidia_smi:
                for gpu_id in self.gpu_pool:
                    self.total_vram[gpu_id] = gpu_vram_fallback_mib()
                    self.reserved_vram[gpu_id] = 0
            else:
                try:
                    cmd = [
                        nvidia_smi,
                        "--query-gpu=index,memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                    result = await asyncio.to_thread(
                        subprocess.run,
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    reader = csv.reader(io.StringIO(result.stdout))
                    for row in reader:
                        idx, total = row[0].strip(), int(row[1].strip())
                        if idx in self.gpu_pool:
                            self.total_vram[idx] = total
                            self.reserved_vram[idx] = 0
                            self.active_vram[idx] = 0
                except Exception as e:
                    print(f"[ResourceAdmissionTracker] Failed to query total VRAM: {e}")
                    for gpu_id in self.gpu_pool:
                        self.total_vram[gpu_id] = gpu_vram_fallback_mib()
                        self.reserved_vram[gpu_id] = 0

            if self._loop_task is None:
                self._loop_task = asyncio.create_task(self._scheduling_loop())
            if self._smi_poll_task is None:
                self._smi_poll_task = asyncio.create_task(self._smi_poll_loop())
            self._initialized = True

    async def reserve_weight(self, gpu_ids: list[str], weight_mb: int):
        if not weight_mb:
            return
        await self.ensure_initialized()
        async with self._lock:
            for gid in gpu_ids:
                if gid in self.reserved_vram:
                    self.reserved_vram[gid] += weight_mb
            self._process_queues_locked()

    async def mark_active(self, gpu_ids: list[str], amount_mb: int):
        """Moves VRAM reservation from 'pending' to 'active' (expected in SMI)."""
        if not amount_mb:
            return
        async with self._lock:
            for gid in gpu_ids:
                if gid in self.active_vram:
                    self.active_vram[gid] += amount_mb
            self._process_queues_locked()

    async def release_weight(self, gpu_ids: list[str], weight_mb: int):
        if not weight_mb:
            return
        await self.ensure_initialized()
        async with self._lock:
            for gid in gpu_ids:
                if gid in self.reserved_vram:
                    before = self.reserved_vram[gid]
                    self.reserved_vram[gid] = max(
                        0, self.reserved_vram[gid] - weight_mb
                    )
                    if gid in self.active_vram:
                        self.active_vram[gid] = min(
                            self.active_vram[gid], self.reserved_vram[gid]
                        )
                    if self.VRAM_DIAG:
                        self._diag_gpu(
                            gid,
                            f"release_weight {weight_mb}MB (reserved {before}→{self.reserved_vram[gid]})",
                        )
            self._process_queues_locked()

    async def update_reservation(self, gpu_ids: list[str], old_mb: int, new_mb: int):
        if old_mb == new_mb:
            return
        async with self._lock:
            for gid in gpu_ids:
                if gid in self.reserved_vram:
                    self.reserved_vram[gid] = max(
                        0, self.reserved_vram[gid] - old_mb + new_mb
                    )
            self._process_queues_locked()

    def _request_interval_fit_locked(
        self,
        gpu_id: str,
        request: AdmissionRequest,
    ) -> tuple[int, bool, int] | None:
        scheduler = getattr(request, "campaign_scheduler", None)
        timelines = getattr(scheduler, "_timelines", None)
        if timelines is None:
            return None
        task_id = str(getattr(request, "task_id", "") or "")
        timeline = timelines.get(gpu_id)
        if not task_id or timeline is None:
            return None
        duration = max(
            0.0,
            float(getattr(request, "required_duration_sec", 0.0) or 0.0),
        )
        entry = timelines.find_entry(task_id)
        if entry is not None and not entry.is_completed:
            planned_wall_duration = max(
                0.0,
                entry.predicted_end_time - entry.start_time,
            )
            if planned_wall_duration > 0.0:
                duration = planned_wall_duration
        now = time.time()
        interval_fit = getattr(timelines, "candidate_interval_fits", None)
        use_tight_temporal = False
        use_planned_envelope = False
        fit_start = now
        temporal_mode = (
            getattr(timelines, "_vram_reservation_model_name", "full_wall")
            == "temporal_peak_interval"
        )
        task_uses_full_wall = getattr(
            timelines,
            "task_uses_full_wall_reservation",
            None,
        )
        already_full_wall = callable(task_uses_full_wall) and task_uses_full_wall(
            task_id
        )
        if temporal_mode and not already_full_wall:
            matching_prediction = bool(
                entry is not None
                and not entry.is_completed
                and entry.is_predicted
                and str(entry.gpu_id) == str(gpu_id)
            )
            temporal_valid = bool(
                matching_prediction
                and entry is not None
                and entry._has_valid_temporal_reservation()
            )
            pre_activation = False
            supervisor = getattr(scheduler, "_supervisor", None)
            worker_name = str(getattr(entry, "worker_name", "") or "")
            if temporal_valid and supervisor is not None and worker_name:
                state = (getattr(supervisor, "states", {}) or {}).get(worker_name)
                pre_activation = bool(
                    state is not None
                    and (
                        not getattr(state, "ready", False)
                        or not getattr(state, "addr", "")
                    )
                )
            use_planned_envelope = temporal_valid and pre_activation
            use_tight_temporal = temporal_valid and not pre_activation
            if use_planned_envelope and entry is not None:
                fit_start = entry.start_time
            if not temporal_valid:
                force_full_wall = getattr(
                    timelines,
                    "force_full_wall_for_task",
                    None,
                )
                if callable(force_full_wall):
                    force_full_wall(task_id)

        fit_now = bool(
            callable(interval_fit)
            and interval_fit(
                gpu_id,
                fit_start,
                fit_start + max(duration, 1e-9),
                request.needed_mb,
                0.0,
                exclude_task_id=task_id,
                candidate_use_planned_envelope=use_planned_envelope,
                candidate_peak_margin_sec=0.0 if use_tight_temporal else None,
            )
            and (
                request.host_ram_mb <= 0
                or interval_fit(
                    gpu_id,
                    now,
                    now + max(duration, 1e-9),
                    0.0,
                    request.host_ram_mb,
                    exclude_task_id=task_id,
                )
            )
        )
        candidate_required_now = int(request.needed_mb)
        reservation_for = getattr(timelines, "_candidate_vram_reservation", None)
        if callable(reservation_for):
            reservation = reservation_for(
                gpu_id=gpu_id,
                start_time=now,
                end_time=now + max(duration, 1e-9),
                needed_vram_mb=request.needed_mb,
                exclude_task_id=task_id,
            )
            low = getattr(reservation, "allow_vram_mb", None)
            peak_start = getattr(reservation, "peak_start_time", None)
            peak_end = getattr(reservation, "peak_end_time", None)
            if (
                isinstance(low, (int, float))
                and isinstance(peak_start, (int, float))
                and isinstance(peak_end, (int, float))
                and 0 <= low < request.needed_mb
                and now < peak_start < peak_end <= now + max(duration, 1e-9)
            ):
                candidate_required_now = (
                    request.needed_mb
                    if peak_start <= now < peak_end
                    else int(math.ceil(low))
                )
        planned_available = int(
            timeline.available_vram_at(now, exclude_task_id=task_id)
        )
        physical_available = max(
            0,
            int(self.total_vram.get(gpu_id, 0))
            - int(self._smi_used_vram.get(gpu_id, 0)),
        )
        fit_now = fit_now and physical_available >= int(request.needed_mb)
        return (
            min(planned_available, physical_available),
            fit_now,
            candidate_required_now,
        )

    def _request_capacity_locked(
        self,
        gid: str,
        req: AdmissionRequest,
        *,
        interval_fit: Any = None,
    ) -> tuple[int, bool, bool]:
        """Shared queue predicates for the ordinary grant path."""
        required_now = req.needed_mb
        if interval_fit is None:
            interval_fit = self._request_interval_fit_locked(gid, req)
        if interval_fit is None:
            available = self.get_available_memory(gid)
            fits = True
        else:
            available, fits, required_now = interval_fit
            if not fits:
                return available, False, False
        host_need = max(0, int(req.host_ram_mb or 0))
        host_ok = True
        if host_need > 0 or self._host_ram_min_available_mib > 0:
            host_ok = (
                self._host_ram_shortfall_locked(
                    host_need,
                    _read_mem_available_mib(),
                )
                <= 0
            )
        return available, fits, available >= required_now and host_ok

    def _grant_request_locked(self, gid: str, req: AdmissionRequest) -> None:
        self.reserved_vram[gid] += req.needed_mb
        if req.type == "activation" and gid in self.active_vram:
            self.active_vram[gid] += req.needed_mb
        self._host_ram_reserved_mb += max(0, int(req.host_ram_mb or 0))

    def _process_queues_locked(self):
        """Commit queued admissions in strict queue order.

        The Planner already decides primary/backfill ordering and GPU
        placement.  Runtime admission must serialize those decisions
        without adding a second backfill policy at commit time.
        """
        for gid in self.gpu_pool:
            q = self._queues[gid]
            if not q:
                continue

            idx = 0
            while idx < len(q):
                req = q[idx]
                available, fit_now, capacity_ok = self._request_capacity_locked(
                    gid, req
                )
                if not fit_now:
                    q.pop(idx)
                    if not req.future.done():
                        req.future.set_result(False)
                    continue

                if capacity_ok:
                    self._grant_request_locked(gid, req)

                    q.pop(idx)
                    if not req.future.done():
                        req.future.set_result(True)
                    continue
                else:
                    if self.VRAM_DIAG and idx == 0:
                        self._diag_gpu(
                            gid,
                            f"BLOCKED type={req.type} need={req.needed_mb}MB avail={available}MB",
                        )
                    break

    async def acquire_activation(
        self,
        gpu_ids: list[str],
        activation_mb: int,
        host_ram_mb: int = 0,
        priority: int = 100,
        campaign_id: str = "",
        is_backfill: bool = False,
        timeout: float | None = None,
        task_id: str = "",
        required_duration_sec: float = 0.0,
        evict_fn: Callable[[str, int], Awaitable[int]] | None = None,
        cancel_backfill_fn: Callable | None = None,
        campaign_scheduler: Any = None,
        idle_freeable_fn: Callable[[str], int] | None = None,
        next_predicted_completion_fn: Callable[[], float | None] | None = None,
    ) -> bool | int:
        await self.ensure_initialized()
        await self._ensure_smi_fresh()

        gpu_id = str(gpu_ids[0]) if gpu_ids else None
        if not gpu_id:
            return False
        host_ram_mb = max(0, int(host_ram_mb or 0))
        activation_mb = max(0, int(activation_mb or 0))

        future = asyncio.get_running_loop().create_future()
        arrival = self._campaign_arrival_time(campaign_id)
        req = AdmissionRequest(
            campaign_arrival=arrival,
            priority=priority,
            timestamp=time.time(),
            needed_mb=activation_mb,
            gpu_ids=gpu_ids,
            future=future,
            type="activation",
            campaign_id=campaign_id,
            is_backfill=is_backfill,
            host_ram_mb=host_ram_mb,
            task_id=str(task_id or ""),
            required_duration_sec=max(0.0, float(required_duration_sec or 0.0)),
            campaign_scheduler=campaign_scheduler,
        )

        async with self._lock:
            self._queues[gpu_id].append(req)
            self._process_queues_locked()

        def _admission_state_fingerprint() -> tuple[int, int, int]:
            reserved = int(self.reserved_vram.get(gpu_id, 0))
            active = int(self.active_vram.get(gpu_id, 0))
            qlen = len(self._queues.get(gpu_id, []))
            return (reserved, active, qlen)

        _entry_state = _admission_state_fingerprint()

        _retry_count = 0
        while not future.done():
            _retry_count += 1
            if self.VRAM_DIAG and _retry_count % 5 == 0:
                self._diag_gpu(
                    gpu_id,
                    f"acquire_activation retry={_retry_count} need={activation_mb}MB",
                )
                self._diag_dump_all(f"acquire_activation stall retry={_retry_count}")
            interval_fit = self._request_interval_fit_locked(gpu_id, req)
            available_for_request = (
                interval_fit[0]
                if interval_fit is not None
                else self.get_available_memory(gpu_id)
            )
            required_now = (
                interval_fit[2] if interval_fit is not None else activation_mb
            )
            shortfall = required_now - available_for_request
            host_shortfall = 0
            if host_ram_mb > 0 or self._host_ram_min_available_mib > 0:
                async with self._lock:
                    host_shortfall = self._host_ram_shortfall_locked(
                        host_ram_mb,
                        _read_mem_available_mib(),
                    )
            if shortfall > 0:
                near_top = False
                async with self._lock:
                    q = self._queues[gpu_id]
                    near_top = req in q[:1]

                if near_top:
                    idle_free = idle_freeable_fn(gpu_id) if idle_freeable_fn else 0
                    if evict_fn is not None and idle_free > 0:
                        await evict_fn(gpu_id, min(shortfall, idle_free))
                        async with self._lock:
                            self._process_queues_locked()
                        refreshed_fit = self._request_interval_fit_locked(gpu_id, req)
                        shortfall = (
                            refreshed_fit[2] - refreshed_fit[0]
                            if refreshed_fit is not None
                            else activation_mb - self.get_available_memory(gpu_id)
                        )
                    if shortfall <= 0:
                        pass
                    else:
                        now_ts = time.time()
                        elapsed = now_ts - req.timestamp
                        next_t: float | None = None
                        if next_predicted_completion_fn is not None:
                            try:
                                next_t = next_predicted_completion_fn()
                            except Exception:
                                next_t = None
                        if next_t is not None and next_t > now_ts:
                            stall_threshold = next_t - now_ts
                        else:
                            stall_threshold = _ADMISSION_STALL_DEFAULT_TTL_SEC
                        if elapsed >= stall_threshold:
                            _LOG.warning(
                                "[admission-stall] GPU %s: need %d MB, avail %d MB, "
                                "idle_free=%d MB, elapsed=%.1fs, threshold=%.1fs "
                                "(next_completion=%s) — surfacing to Core Loop",
                                gpu_id,
                                activation_mb,
                                self.get_available_memory(gpu_id),
                                idle_free,
                                elapsed,
                                stall_threshold,
                                f"{next_t - now_ts:.1f}s" if next_t else "none",
                            )
                            async with self._lock:
                                q = self._queues.get(gpu_id, [])
                                if req in q:
                                    q.remove(req)
                                self._process_queues_locked()
                            if not future.done():
                                future.set_result(False)
                            break
            elif host_shortfall > 0:
                near_top = False
                async with self._lock:
                    q = self._queues[gpu_id]
                    near_top = req in q[:1]
                if near_top:
                    elapsed = time.time() - req.timestamp
                    try:
                        freed_idle = await self.evict_idle_workers_for_host_ram(
                            host_shortfall,
                        )
                    except Exception:
                        freed_idle = 0
                        _LOG.warning(
                            "[admission-host-ram] idle host-RAM eviction failed",
                            exc_info=True,
                        )
                    if freed_idle > 0:
                        async with self._lock:
                            self._process_queues_locked()
                        host_shortfall = self._host_ram_shortfall_locked(
                            host_ram_mb,
                            _read_mem_available_mib(),
                        )
                    if (
                        host_shortfall > 0
                        and elapsed >= _ADMISSION_STALL_DEFAULT_TTL_SEC
                    ):
                        mem_avail = _read_mem_available_mib()
                        _LOG.warning(
                            "[admission-host-ram] reject gpu=%s "
                            "MemAvailable=%d MiB reserved=%d MiB "
                            "need=%d MiB threshold=%d MiB "
                            "(campaign=%s backfill=%s activation_mb=%d)",
                            gpu_id,
                            mem_avail,
                            self._host_ram_reserved_mb,
                            host_ram_mb,
                            self._host_ram_min_available_mib,
                            campaign_id or "<empty>",
                            is_backfill,
                            activation_mb,
                        )
                        async with self._lock:
                            q = self._queues.get(gpu_id, [])
                            if req in q:
                                q.remove(req)
                                self._process_queues_locked()
                        if not future.done():
                            future.set_exception(
                                HostRAMSaturatedError(
                                    mem_available_mib=mem_avail,
                                    threshold_mib=self._host_ram_min_available_mib,
                                    gpu_id=gpu_id,
                                ),
                            )
                        break
            else:
                async with self._lock:
                    self._process_queues_locked()

            try:
                wait_time = 2.0 if not future.done() else 0.1
                await asyncio.wait_for(asyncio.shield(future), timeout=wait_time)
            except asyncio.TimeoutError:
                if future.done():
                    break
                continue
            except asyncio.CancelledError:
                async with self._lock:
                    committed = False
                    if future.done() and not future.cancelled():
                        try:
                            committed = future.result() is True
                        except Exception:
                            committed = False
                    if committed:
                        self.reserved_vram[gpu_id] = max(
                            0,
                            self.reserved_vram.get(gpu_id, 0) - req.needed_mb,
                        )
                        if req.type == "activation" and gpu_id in self.active_vram:
                            self.active_vram[gpu_id] = max(
                                0,
                                self.active_vram[gpu_id] - req.needed_mb,
                            )
                        committed_host_mb = int(getattr(req, "host_ram_mb", 0) or 0)
                        if committed_host_mb > 0:
                            self._host_ram_reserved_mb = max(
                                0,
                                self._host_ram_reserved_mb - committed_host_mb,
                            )
                        self._process_queues_locked()
                    elif req in self._queues[gpu_id]:
                        self._queues[gpu_id].remove(req)
                        self._process_queues_locked()
                raise

        ok = await future
        if not ok:
            return ok
        return activation_mb if activation_mb > 0 else True

    @staticmethod
    def _estimate_backfill_freeable(gpu_id: str, campaign_scheduler: Any) -> int:
        """Estimate activation VRAM freeable by evicting backfill on *gpu_id*.

        Uses predicted_vram_mb from timeline entries — no side effects.
        """
        try:
            tl = campaign_scheduler._timelines.get(gpu_id)
            if not tl:
                return 0
            return int(
                sum(
                    e.predicted_vram_mb
                    for e in tl.active_entries
                    if e.is_backfill and not e.is_predicted
                )
            )
        except Exception:
            return 0

    async def acquire_compute_slot(
        self,
        gpu_id: str,
        component: str,
        campaign_scheduler: Any,
        is_backfill: bool = False,
        cancel_backfill_fn: Callable | None = None,
        campaign_id: str = "",
        global_planner: Any = None,
    ) -> tuple[bool, str]:
        """Compute admission gate — planned GPU physical availability only.

        3-layer architecture (Plan fix): the Global Planner
        has already selected the optimal GPU and has integrated interference
        into the D2 EFT formula (``Σ (sd-1) × overlap``).  This gate only
        validates **physical compute availability** of that planned GPU
        (active count + GPU health) — no interference re-evaluation.
        Re-running the same GP-based interference optimization here would be
        double-optimization and could conflict with the Planner's HEFT
        decision (no runtime signal exists to independently verify
        interference, unlike VRAM which has CUDA-runtime ground truth).

        If blocked → return (False, gpu_id) so the Core Loop feeds back
        ConstraintViolation(compute_saturated) → Planner re_plan.

        Returns (ok, gpu_id):
        - Backfill: check planned GPU → blocked → (False, "") → re-schedule
        - Primary: check planned GPU → blocked → (False, "") immediately
          (plan  — no autonomous eviction here; Planner's re_plan
          invokes ``try_evict_backfills_for`` via MCPSE).
        """
        if not gpu_id:
            return True, gpu_id

        def _would_slow(comp: str, gid: str, cid: str = "") -> bool:
            return False

        if is_backfill:
            if _would_slow(component, gpu_id, campaign_id):
                return False, ""
            return True, gpu_id

        if _would_slow(component, gpu_id, campaign_id):
            return False, ""
        return True, gpu_id

    def _notify_compute_waiters(self, gpu_id: str) -> None:
        """Wake all compute waiters for a GPU (called on task end + cancel)."""
        for w in self._compute_waiters.get(gpu_id, []):
            w.event.set()

    async def acquire_weight(
        self,
        gpu_ids: list[str],
        needed_mb: int,
        priority: int = 50,
        campaign_id: str = "",
        evict_fn: Callable[[str, int], Awaitable[int]] | None = None,
    ) -> None:
        if not gpu_ids or needed_mb <= 0:
            return
        await self.ensure_initialized()

        gpu_id = str(gpu_ids[0])
        future = asyncio.get_running_loop().create_future()
        arrival = self._campaign_arrival_time(campaign_id)
        req = AdmissionRequest(
            campaign_arrival=arrival,
            priority=priority,
            timestamp=time.time(),
            needed_mb=needed_mb,
            gpu_ids=gpu_ids,
            future=future,
            type="weight",
            campaign_id=campaign_id,
        )

        async with self._lock:
            self._queues[gpu_id].append(req)
            self._process_queues_locked()

        _retry_count = 0
        while not future.done():
            _retry_count += 1
            if self.VRAM_DIAG and _retry_count % 5 == 0:
                self._diag_gpu(
                    gpu_id, f"acquire_weight retry={_retry_count} need={needed_mb}MB"
                )
            if evict_fn:
                shortfall = needed_mb - self.get_available_memory(gpu_id)
                if shortfall > 0:
                    async with self._lock:
                        q = self._queues[gpu_id]
                        if req in q[:2]:
                            pass
                        else:
                            shortfall = 0

                    if shortfall > 0:
                        await evict_fn(gpu_id, shortfall)
                        async with self._lock:
                            self._process_queues_locked()
                else:
                    async with self._lock:
                        self._process_queues_locked()

            try:
                wait_time = 2.0 if not future.done() else 0.1
                await asyncio.wait_for(asyncio.shield(future), timeout=wait_time)
            except asyncio.TimeoutError:
                if future.done():
                    break
                continue
            except asyncio.CancelledError:
                async with self._lock:
                    committed = False
                    if future.done() and not future.cancelled():
                        try:
                            committed = future.result() is True
                        except Exception:
                            committed = False
                    if committed:
                        self.reserved_vram[gpu_id] = max(
                            0,
                            self.reserved_vram.get(gpu_id, 0) - req.needed_mb,
                        )
                        if req.type == "activation" and gpu_id in self.active_vram:
                            self.active_vram[gpu_id] = max(
                                0,
                                self.active_vram[gpu_id] - req.needed_mb,
                            )
                        self._process_queues_locked()
                    elif req in self._queues[gpu_id]:
                        self._queues[gpu_id].remove(req)
                        self._process_queues_locked()
                raise

    async def release_activation(
        self,
        gpu_ids: list[str],
        activation_mb: int,
        host_ram_mb: int = 0,
    ):
        if not activation_mb and not host_ram_mb:
            return
        await self.ensure_initialized()
        async with self._lock:
            for gid in gpu_ids:
                if gid in self.reserved_vram:
                    before = self.reserved_vram[gid]
                    self.reserved_vram[gid] = max(
                        0, self.reserved_vram[gid] - activation_mb
                    )
                    if gid in self.active_vram:
                        self.active_vram[gid] = max(
                            0, self.active_vram[gid] - activation_mb
                        )
                    if self.VRAM_DIAG:
                        self._diag_gpu(
                            gid,
                            f"release_activation {activation_mb}MB (reserved {before}→{self.reserved_vram[gid]})",
                        )
            if host_ram_mb > 0:
                self._host_ram_reserved_mb = max(
                    0,
                    self._host_ram_reserved_mb - int(host_ram_mb),
                )
            self._process_queues_locked()

    async def notify_waiters(self) -> None:
        async with self._cv:
            self._cv.notify_all()


def _log_background_task_failure(task: asyncio.Task[Any]) -> None:
    """Consume background task exceptions so they do not surface as
    ``Task exception was never retrieved`` warnings.

    Recovery / housekeeping loops intentionally spawn detached tasks to
    avoid blocking progress.  Those tasks still need exception harvest
    so operational failures show up as structured logs rather than raw
    asyncio warnings.
    """
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except RuntimeError as exc:
        message = str(exc)
        if "worker failed to become ready" in message:
            _LOG.warning(
                "[background-task] detached worker recovery failed: %s", message
            )
            return
        _LOG.exception("[background-task] detached task failed")
    except Exception:
        _LOG.exception("[background-task] detached task failed")


_NVIDIA_SHARED_DEVICE_NODES = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-modeset",
)


def docker_explicit_nvidia_devices(gpus: list[str]) -> list[str]:
    """Keep GPU device cgroup access visible to the low-level runtime."""
    nodes = [*_NVIDIA_SHARED_DEVICE_NODES]
    nodes.extend(f"/dev/nvidia{gpu}" for gpu in gpus if gpu.isdigit())
    return [f"{node}:{node}:rwm" for node in nodes if os.path.exists(node)]


def docker_device_request(gpus: list[str]):
    import docker.types

    return docker.types.DeviceRequest(device_ids=gpus, capabilities=[["gpu"]])


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in args)


def _append_flag(args: list[str], flag: str, value: object) -> None:
    if _has_flag(args, flag):
        return
    args.extend([flag, str(value)])


def worker_process_backend_enabled(spec: WorkerSpec) -> bool:
    worker_server = dict(spec.worker_server or {})
    return int(worker_server.get("execute_processes", 0)) >= 0


def worker_cuda_mps_enabled(spec: WorkerSpec) -> bool:
    configured = (spec.worker_server or {}).get("cuda_mps")
    return (
        worker_process_backend_enabled(spec)
        if configured is None
        else configured is True
    )


def build_worker_command(
    spec: WorkerSpec, host_port: int, max_concurrency: int = 0
) -> list[str]:
    if spec.command:
        cmd = shlex.split(spec.command)
    else:
        cmd = ["python", "-m", "modelworker.worker_server"]

    _append_flag(cmd, "--adapter", spec.adapter)
    _append_flag(cmd, "--addr", f"0.0.0.0:{host_port}")

    worker_server = dict(spec.worker_server or {})
    cfg_exec = int(worker_server.get("execute_concurrency") or 0)
    worker_server["execute_concurrency"] = cfg_exec if cfg_exec > 0 else max_concurrency
    worker_server.setdefault("execute_processes", 0)

    for key, flag in (
        ("prepare_concurrency", "--prepare-concurrency"),
        ("execute_concurrency", "--execute-concurrency"),
        ("finalize_concurrency", "--finalize-concurrency"),
        ("item_concurrency", "--item-concurrency"),
        ("execute_processes", "--execute-processes"),
    ):
        if key in worker_server and worker_server[key] is not None:
            _append_flag(cmd, flag, int(worker_server[key]))

    if worker_server.get("log_level") is not None:
        _append_flag(cmd, "--log-level", str(worker_server["log_level"]))

    return cmd


async def wait_ready(addr: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            async with grpc.aio.insecure_channel(addr) as channel:
                stub = pb_grpc.ModelWorkerStub(channel)
                resp = await stub.Health(pb.HealthRequest(readiness=True), timeout=10.0)
                if resp.ok:
                    return
        except Exception:
            await asyncio.sleep(0.5)
    raise TimeoutError(f"worker not ready: {addr}")


async def wait_alive(addr: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            async with grpc.aio.insecure_channel(addr) as channel:
                stub = pb_grpc.ModelWorkerStub(channel)
                resp = await stub.Health(
                    pb.HealthRequest(readiness=False), timeout=10.0
                )
                if resp.ok:
                    return
        except Exception:
            await asyncio.sleep(0.5)
    raise TimeoutError(f"worker not alive: {addr}")


async def get_caps(addr: str) -> pb.Capabilities:
    async with grpc.aio.insecure_channel(addr) as channel:
        stub = pb_grpc.ModelWorkerStub(channel)
        return await stub.GetCapabilities(pb.Empty(), timeout=120.0)


async def get_stats(addr: str, timeout_s: float = 4.0) -> pb.WorkerStats:
    async with grpc.aio.insecure_channel(addr) as channel:
        stub = pb_grpc.ModelWorkerStub(channel)
        return await stub.GetStats(pb.Empty(), timeout=timeout_s)


async def stream_logs(container, prefix: str) -> None:
    def _iter() -> None:
        for line in container.logs(stream=True, follow=True, tail=50):
            try:
                s = line.decode("utf-8", errors="replace").rstrip("\n")
            except Exception:
                s = str(line)
            print(f"[{prefix}] {s}", flush=True)

    await asyncio.to_thread(_iter)


class WorkerSupervisor:
    DISPATCH_FRONT_HANDOFF_ORPHAN_TIMEOUT_SEC = 60.0

    def __init__(self, cfg: SupervisorConfig):
        self.host_ram_leak_idle_grace_sec = float(
            getattr(cfg, "host_ram_leak_idle_grace_sec", 10.0)
        )
        self.specs = cfg.specs
        self.host = cfg.host
        self.readiness_timeout_s = cfg.readiness_timeout_s
        self.stats_interval_s = cfg.stats_interval_s
        self.stats_report_interval_s = cfg.stats_report_interval_s
        self._last_report_time: float = 0.0
        self.stream_logs_enabled = cfg.stream_logs
        self.stats_live = cfg.stats_live and not self.stream_logs_enabled
        self.dclient = docker.from_env()
        self._mps_daemons: dict[str, _MpsDaemonState] = {}
        self._mps_startup_wait_s = 2.0
        self.states: dict[str, WorkerState] = {
            self._state_key(s): WorkerState(spec=s) for s in cfg.specs
        }
        self._base_state_keys = {self._state_key(s) for s in cfg.specs}
        self._isolated_component: str | None = None
        self._isolated_parallelism: int = 0
        self._shutdown_event = asyncio.Event()
        self._last_runtime_activity_ts: float = time.time()
        self._idle_host_rss_cache: dict[int, tuple[float, int]] = {}
        self._idle_host_rss_cache_ttl_sec: float = 1.0
        self._spawn_locks: dict[object, asyncio.Lock] = {}
        self._stop = asyncio.Event()
        self._stats_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self._component_weight_cache: dict[str, int] = {}

        self._activation_peak_recorder: Any | None = None
        self._on_worker_killed: Any | None = None
        self._invalidate_channel_fn: Any | None = None
        self._eviction_handler: Any | None = None
        self._host_ram_eviction_handler: Any | None = None
        self._grace_period_fn: Any | None = None
        self._recovery_handler: Any | None = None
        self.dispatch_front_capacity_hook: Any | None = None
        from .policies.oom_guard import WeightOOMPolicy

        weight_policy_cls = WeightOOMPolicy.get(
            cfg.weight_oom_policy
        ) or WeightOOMPolicy.get("optimistic")
        self.weight_oom_guard = weight_policy_cls()
        self.cold_start_activation_ratio = cfg.cold_start_activation_ratio
        self.max_primary_slowdown = cfg.max_primary_slowdown
        self.slowdown_cold_start_default = cfg.slowdown_cold_start_default
        self.pairwise_slowdown_cold_start = cfg.pairwise_slowdown_cold_start
        self.self_slowdown_cold_start_default = cfg.self_slowdown_cold_start_default
        self.self_concurrency_max_primary_cold_start = (
            cfg.self_concurrency_max_primary_cold_start
        )
        self.self_concurrency_max_backfill_cold_start = (
            cfg.self_concurrency_max_backfill_cold_start
        )
        self.gp_maturity_observations_per_dim = cfg.gp_maturity_observations_per_dim
        self.concurrency_gate_safety_factor = cfg.concurrency_gate_safety_factor
        self.disable_self_interference_gate = cfg.disable_self_interference_gate
        self.disable_pbbc_gate = cfg.disable_pbbc_gate
        self.disable_primary_slo_protection = cfg.disable_primary_slo_protection
        self.disable_slowdown_cost = cfg.disable_slowdown_cost

        self.admission_host_ram_min_available_mib = int(
            cfg.admission_host_ram_min_available_mib
        )

        self.profiling_runtime = dict(cfg.profiling_runtime or {})
        self.eviction_grace_period_s = cfg.eviction_grace_period_s

        self.gpu_policy: GpuPolicy | None = None
        pool = []
        if cfg.gpu_allocation:
            pool = [str(x) for x in (cfg.gpu_allocation.get("pool") or [])]
            policy_name = str(cfg.gpu_allocation.get("policy", "least_memory_used"))

            policy_cls = PolicyRegistry.get(policy_name)
            if policy_cls:
                self.gpu_policy = policy_cls(pool)
            else:
                self.gpu_policy = LeastMemoryPolicy(pool)
                print(
                    f"Warning: Unknown policy '{policy_name}', using least_memory_used",
                    flush=True,
                )

        self.resource_tracker = ResourceAdmissionTracker(
            pool,
            host_ram_min_available_mib=self.admission_host_ram_min_available_mib,
        )
        self.resource_tracker._estimate_idle_host_ram_freeable_fn = (
            self.estimate_idle_host_ram_freeable
        )
        self.resource_tracker._evict_idle_workers_for_host_ram_fn = (
            self.evict_idle_workers_for_host_ram
        )
        _cold_init = 10.0
        if cfg.profiling_runtime:
            try:
                _cold_init = float(
                    cfg.profiling_runtime.get("cold_init_latency_sec", 10.0)
                )
            except (TypeError, ValueError):
                _cold_init = 10.0
        self.init_tracker = InitProfile(default_init_sec=_cold_init)
        self.cold_init_latency_sec = _cold_init
        _cold_infer = 30.0
        if cfg.profiling_runtime:
            try:
                _cold_infer = float(
                    cfg.profiling_runtime.get("cold_inference_latency_sec", 30.0)
                )
            except (TypeError, ValueError):
                _cold_infer = 30.0
        self.cold_inference_latency_sec = _cold_infer

        _self_intf_prior = 1.0
        if cfg.profiling_runtime:
            try:
                _self_intf_prior = float(
                    cfg.profiling_runtime.get("self_interference_prior", 1.0)
                )
            except (TypeError, ValueError):
                _self_intf_prior = 1.0
        if _self_intf_prior < 0.0:
            _self_intf_prior = 0.0
        self.self_interference_prior = _self_intf_prior

        policy_cls = EvictionPolicyRegistry.get(cfg.eviction_policy)
        if policy_cls:
            self.eviction_policy = policy_cls()
        else:
            self.eviction_policy = LruEvictionPolicy()
            print(
                f"Warning: Unknown eviction policy '{cfg.eviction_policy}', using lru",
                flush=True,
            )

        policy_cls = RecoveryPolicyRegistry.get(cfg.recovery_policy)
        if policy_cls:
            self.recovery_policy = policy_cls()
        else:
            self.recovery_policy = MruRecoveryPolicy()
            print(
                f"Warning: Unknown recovery policy '{cfg.recovery_policy}', using mru",
                flush=True,
            )

    def bootstrap_weights(self, run_index: Any) -> None:
        """Loads known component weights from the run index to prevent cold starts."""
        _LOG.info("[supervisor] Bootstrapping component weight cache from index...")
        try:
            query_resident = getattr(run_index, "query_latest_resident_baselines", None)
            records = (
                query_resident()
                if callable(query_resident)
                else run_index.query_observations(run_sources=["task"], limit=200)
            )
            seen_components = set()
            for rec in records:
                comp = str(rec.component or "").strip().lower()
                if not comp or comp in seen_components:
                    continue

                bootstrap = getattr(rec, "bootstrap_memory_summary", {}) or {}
                ready_floor = (
                    bootstrap.get("ready_quiescent_floor_mib")
                    if isinstance(bootstrap, Mapping)
                    else None
                )
                resident_source = str(
                    getattr(rec, "resident_memory_source", "") or ""
                ).strip()
                weight_mb = int(ready_floor or 0)
                if weight_mb <= 0 and resident_source != "parent_plus_actor_idle":
                    weight_mb = int(getattr(rec, "resident_memory_mib", 0) or 0)
                if weight_mb > 0:
                    self._update_weight_cache_direct(comp, weight_mb)
                    seen_components.add(comp)
                    _LOG.info(
                        "[supervisor] Bootstrapped weight for %s: %d MB",
                        comp,
                        weight_mb,
                    )
        except Exception as exc:
            _LOG.warning("[supervisor] weight bootstrap failed: %s", exc)

    def _update_weight_cache_direct(self, component: str, weight_mb: int) -> None:
        """Internal synchronous weight cache update."""
        if weight_mb > 0:
            self._component_weight_cache[str(component).strip().lower()] = int(
                weight_mb
            )

    def export_runtime_state(self) -> dict[str, Any]:
        now = time.time()
        worker_baselines: dict[str, dict[str, Any]] = {}
        for name, st in sorted(self.states.items()):
            collected_at = st.resident_baseline_collected_at
            try:
                collected_age = (
                    max(0.0, now - float(collected_at))
                    if collected_at is not None and float(collected_at) > 0
                    else None
                )
            except Exception:
                collected_age = None
            worker_baselines[str(name)] = {
                "component": st.spec.component,
                "gpu_ids": list(st.assigned_gpus or st.spec.gpus or []),
                "resident_memory_mib": st.resident_memory_mib,
                "resident_memory_source": st.resident_memory_source,
                "resident_baseline_collected_age_sec": collected_age,
                "resident_baseline_lifecycle_token": st.resident_baseline_lifecycle_token,
                "resident_baseline_state": st.resident_baseline_state,
                "real_dispatch_count_in_generation": int(
                    st.real_dispatch_count_in_generation
                ),
            }
        return {
            "schema": "supervisor_runtime_state_v1",
            "component_weight_cache": dict(
                sorted(self._component_weight_cache.items())
            ),
            "workers": worker_baselines,
        }

    def import_runtime_state(self, data: Mapping[str, Any]) -> dict[str, int]:
        if not isinstance(data, Mapping):
            return {"workers": 0, "component_weight_cache": 0}
        cache = data.get("component_weight_cache") or {}
        if isinstance(cache, Mapping):
            self._component_weight_cache = {
                str(comp).strip().lower(): int(weight)
                for comp, weight in cache.items()
                if str(comp).strip() and int(weight or 0) > 0
            }
        return {
            "workers": 0,
            "component_weight_cache": len(self._component_weight_cache),
        }

    def set_activation_peak_recorder(self, callback: Any) -> None:
        """Wire in SignalService.record_activation_peak so stats loop can feed GPU-measured peaks."""
        self._activation_peak_recorder = callback

    def get_component_weight(self, component: str) -> int:
        """Returns the cached resident weight (in MB) for a component. Returns 0 if unknown."""
        return self._component_weight_cache.get(str(component).strip().lower(), 0)

    async def update_component_weight(self, component: str, weight_mb: int) -> None:
        """Updates the cached resident weight for a component and proactively releases guards."""
        normalized_component = str(component).strip().lower()
        if weight_mb > 0:
            self._update_weight_cache_direct(normalized_component, weight_mb)

            async with self._lock:
                try:
                    for st in self.states.values():
                        if (
                            st.spec.component.strip().lower() == normalized_component
                            and st.is_guarded
                        ):
                            old_reserve = st.memory_reserved_mb
                            st.memory_reserved_mb = int(weight_mb)
                            st.is_guarded = False
                            _LOG.info(
                                "[guard-RELEASE-PROACTIVE] %s weight is now known (%d MB). Releasing guard.",
                                st.spec.name,
                                weight_mb,
                            )

                            if self.resource_tracker and st.assigned_gpus:
                                await self.resource_tracker.update_reservation(
                                    st.assigned_gpus, old_reserve, int(weight_mb)
                                )
                finally:
                    if self.resource_tracker:
                        await self.resource_tracker.notify_waiters()

    def _get_pending_gpu_usage(self) -> dict[str, int]:
        """Returns a map of GPU ID to pending memory usage from non-ready workers."""
        pending_usage: dict[str, int] = {}
        for st in self.states.values():
            if st.assigned_gpus and st.lifecycle_state == "starting":
                weight_mb = self.get_component_weight(st.spec.component)
                if weight_mb > 0:
                    for gpu_id in st.assigned_gpus:
                        pending_usage[gpu_id] = pending_usage.get(gpu_id, 0) + weight_mb
        return pending_usage

    def stop_event(self) -> asyncio.Event:
        return self._stop

    def request_stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _state_key(spec: WorkerSpec) -> str:
        return f"{spec.component}:{spec.name}"

    def _set_ready(self, st: WorkerState) -> None:
        """fix: set ready flag + signal pending waiters atomically.

        Call instead of ``st.ready = True`` so ``wait_for_ready`` observers
        are notified in the same logical step.
        """
        st.ready = True
        st.ready_event.set()

    def _clear_ready(self, st: WorkerState) -> None:
        """fix: clear ready flag + re-arm the event.

        ``wait_for_ready`` callers will block on ``ready_event`` until the
        next ``_set_ready`` re-announces readiness.
        """
        st.ready = False
        st.ready_event.clear()

    def _resolve_worker(
        self,
        worker_name: str,
        *,
        component: str | None = None,
    ) -> WorkerState | None:
        """fix: Look up WorkerState by name, optionally scoped to component."""
        wn = str(worker_name).strip()
        if component:
            key = f"{str(component).strip()}:{wn}"
            return self.states.get(key)
        match: WorkerState | None = None
        for st in self.states.values():
            if st.spec.name == wn:
                if match is not None:
                    raise RuntimeError(
                        f"ambiguous worker name {wn!r}; specify component"
                    )
                match = st
        return match

    @staticmethod
    def _container_name(spec: WorkerSpec, host: str) -> str:
        username = getpass.getuser().strip() or "user"
        safe_host = host.replace(".", "-").replace(":", "-")
        return f"mw-{spec.component}-{spec.name}-{username}-{safe_host}"

    @staticmethod
    def _container_labels(spec: WorkerSpec, host: str) -> dict[str, str]:
        return {
            "bio_gateway_managed": "true",
            "bio_gateway_component": str(spec.component),
            "bio_gateway_worker_name": str(spec.name),
            "bio_gateway_host": str(host),
        }

    @staticmethod
    def _safe_name_part(value: object) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value)
        ).strip("-")
        return safe or "x"

    @classmethod
    def _mps_container_name(cls, gpu_id: str, host: str) -> str:
        username = cls._safe_name_part(getpass.getuser().strip() or "user")
        safe_host = cls._safe_name_part(host)
        safe_gpu = cls._safe_name_part(gpu_id)
        return f"mw-mps-{safe_gpu}-{username}-{safe_host}"

    def _mps_root_dir(self, gpu_id: str) -> Path:
        username = self._safe_name_part(getpass.getuser().strip() or "user")
        safe_host = self._safe_name_part(self.host)
        safe_gpu = self._safe_name_part(gpu_id)
        return Path("/tmp") / f"proton-gw-mps-{username}-{safe_host}-{safe_gpu}"

    def _mps_dirs(self, gpu_id: str) -> tuple[Path, Path]:
        root = self._mps_root_dir(gpu_id)
        return root / "pipe", root / "log"

    def _mps_container_labels(self, gpu_id: str) -> dict[str, str]:
        return {
            "bio_gateway_managed": "true",
            "bio_gateway_role": "cuda_mps",
            "bio_gateway_gpu": str(gpu_id),
            "bio_gateway_host": str(self.host),
            "bio_gateway_mps_root": str(self._mps_root_dir(gpu_id)),
        }

    @staticmethod
    def _volume_bind(path: Path) -> dict[str, str]:
        return {"bind": str(path), "mode": "rw"}

    @staticmethod
    def _is_container_running(container: object) -> bool:
        status = getattr(container, "status", "")
        attrs = getattr(container, "attrs", {}) or {}
        return str(status).lower() == "running" or bool(
            attrs.get("State", {}).get("Running")
        )

    async def _mps_control(self, container: Any, command: str) -> list[str]:
        result = await asyncio.to_thread(
            _call_blocking_with_timeout,
            container.exec_run,
            [
                "sh",
                "-lc",
                f"printf '%s\\n' {shlex.quote(command)} | nvidia-cuda-mps-control",
            ],
            timeout_s=5.0,
        )
        if hasattr(result, "exit_code"):
            exit_code = int(result.exit_code)
            output = result.output
        else:
            exit_code, output = result
        if int(exit_code) != 0:
            raise RuntimeError(f"MPS control command failed ({exit_code}): {command}")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return [line.strip() for line in str(output).splitlines() if line.strip()]

    async def _ensure_mps_daemon(self, gpu_id: str, image: str) -> _MpsDaemonState:
        gpu_id = str(gpu_id)
        name = self._mps_container_name(gpu_id, self.host)
        pipe_dir, log_dir = self._mps_dirs(gpu_id)
        for directory in (pipe_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o777)

        old = None
        try:
            old = await asyncio.to_thread(self.dclient.containers.get, name)
            await asyncio.to_thread(old.reload)
        except docker.errors.NotFound:
            self._mps_daemons.pop(gpu_id, None)
        except Exception as exc:
            raise RuntimeError(f"failed to inspect CUDA MPS daemon {name}") from exc

        if old is not None and self._is_container_running(old):
            try:
                server_ids = await self._mps_control(old, "get_server_list")
                if not server_ids:
                    status = "NO_SERVER"
                else:
                    if len(server_ids) != 1 or not server_ids[0].isdigit():
                        raise RuntimeError(f"invalid MPS server list: {server_ids!r}")
                    server_id = server_ids[0]
                    statuses = await self._mps_control(
                        old, f"get_server_status {server_id}"
                    )
                    if len(statuses) != 1:
                        raise RuntimeError(f"invalid MPS server status: {statuses!r}")
                    status = statuses[0]
                    if status == "ACTIVE":
                        daemon = _MpsDaemonState(
                            gpu_id=gpu_id,
                            name=name,
                            pipe_dir=pipe_dir,
                            log_dir=log_dir,
                            container_id=getattr(old, "id", None),
                        )
                        self._mps_daemons[gpu_id] = daemon
                        return daemon
                    if (
                        status != status.upper()
                        or not status.replace("_", "").isalpha()
                    ):
                        raise RuntimeError(f"invalid MPS server status: {status!r}")
            except Exception as exc:
                _LOG.warning("[mps] probe failed gpu=%s name=%s: %s", gpu_id, name, exc)
                raise RuntimeError(
                    f"CUDA MPS health is unknown for gpu={gpu_id}"
                ) from exc

            if status != "FAULT" and status != "NO_SERVER":
                _LOG.warning(
                    "[mps] refusing replacement gpu=%s name=%s status=%s",
                    gpu_id,
                    name,
                    status,
                )
                raise RuntimeError(f"CUDA MPS daemon {name} is {status}")
            _LOG.warning(
                "[mps] replacing unhealthy daemon gpu=%s name=%s status=%s",
                gpu_id,
                name,
                status,
            )

        if old is not None:
            self._mps_daemons.pop(gpu_id, None)
            try:
                await asyncio.to_thread(old.remove, force=True)
            except Exception as exc:
                raise RuntimeError(f"failed to remove CUDA MPS daemon {name}") from exc

        for entry in pipe_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)

        env = {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
        volumes = {
            str(pipe_dir): self._volume_bind(pipe_dir),
            str(log_dir): self._volume_bind(log_dir),
        }
        container = await asyncio.to_thread(
            self.dclient.containers.run,
            image=image,
            name=name,
            detach=True,
            remove=False,
            environment=env,
            volumes=volumes,
            device_requests=[docker_device_request([gpu_id])],
            devices=docker_explicit_nvidia_devices([gpu_id]),
            command=[
                "sh",
                "-lc",
                'nvidia-cuda-mps-control -d && echo "start_server -uid $(id -u)" '
                "| nvidia-cuda-mps-control && tail -f /dev/null",
            ],
            labels=self._mps_container_labels(gpu_id),
            network_mode="host",
            ipc_mode="host",
        )
        wait_s = float(getattr(self, "_mps_startup_wait_s", 2.0) or 0.0)
        if wait_s > 0:
            await asyncio.sleep(wait_s)

        try:
            await asyncio.to_thread(container.reload)
            server_ids = await self._mps_control(container, "get_server_list")
            if len(server_ids) != 1 or not server_ids[0].isdigit():
                raise RuntimeError(f"invalid MPS server list: {server_ids!r}")
            statuses = await self._mps_control(
                container, f"get_server_status {server_ids[0]}"
            )
            if statuses != ["ACTIVE"]:
                raise RuntimeError(f"new MPS daemon is not ACTIVE: {statuses!r}")
            if not self._is_container_running(container):
                raise RuntimeError("new MPS daemon container exited")
        except Exception as exc:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(container.remove, force=True)
            self._mps_daemons.pop(gpu_id, None)
            _LOG.warning(
                "[mps] replacement failed gpu=%s name=%s: %s", gpu_id, name, exc
            )
            raise RuntimeError(
                f"CUDA MPS daemon {name} failed health validation"
            ) from exc

        daemon = _MpsDaemonState(
            gpu_id=gpu_id,
            name=name,
            pipe_dir=pipe_dir,
            log_dir=log_dir,
            container_id=getattr(container, "id", None),
        )
        self._mps_daemons[gpu_id] = daemon
        _LOG.info("[mps] started %s for gpu=%s pipe=%s", name, gpu_id, pipe_dir)
        return daemon

    async def _stop_mps_daemons(self) -> None:
        daemons = list(self._mps_daemons.values())
        self._mps_daemons.clear()
        for daemon in daemons:
            try:
                container = await asyncio.to_thread(
                    self.dclient.containers.get, daemon.name
                )
            except Exception:
                container = None
            if container is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        container.exec_run,
                        "sh -lc 'echo quit | nvidia-cuda-mps-control || true'",
                    )
                try:
                    await asyncio.to_thread(container.remove, force=True)
                    _LOG.info("[mps] removed %s", daemon.name)
                except Exception as exc:
                    _LOG.warning("[mps] failed to remove %s: %s", daemon.name, exc)
            root = daemon.pipe_dir.parent
            if root.name.startswith("proton-gw-mps-"):
                shutil.rmtree(root, ignore_errors=True)

    def _ensure_state(self, spec: WorkerSpec) -> WorkerState:
        key = self._state_key(spec)
        st = self.states.get(key)
        if st is None:
            st = WorkerState(spec=spec)
            self.states[key] = st
        else:
            st.spec = spec
        if not st.assigned_gpus:
            st.assigned_gpus = list(spec.gpus or [])
        return st

    @staticmethod
    def _prepare_state_for_start(st: WorkerState, container_name: str) -> None:
        """Reset transient eviction flags before a deliberate worker start.

        Idle eviction marks a worker ``draining`` while it is being stopped.
        The same ``WorkerState`` object is reused when later demand starts the
        worker again.  If the flag survives into the next Ready state, the
        worker remains resident but is excluded from future idle-relief
        candidates, which can strand strict-FIFO baseline heads behind a tiny
        resource shortfall.
        """
        st.draining = False
        st.recovery_suppressed = False
        st.status = "Starting"
        st.lifecycle_state = "starting"
        st.last_error = ""
        st.container_name = container_name

    async def _stop_state(
        self,
        st: WorkerState,
        graceful: bool = True,
    ) -> None:
        if st.log_task:
            st.log_task.cancel()
            st.log_task = None
        old_addr = st.addr
        if old_addr and getattr(self, "_invalidate_channel_fn", None):
            with contextlib.suppress(Exception):
                self._invalidate_channel_fn(old_addr)
        self._clear_ready(st)

        async def _stop_container():
            try:
                container = await asyncio.to_thread(self._get_container, st)
                if container:
                    if graceful:
                        _LOG.info(
                            "[stop] Graceful stop for %s (timeout=%ds)",
                            st.container_name,
                            st.spec.grace_period_s,
                        )
                        await asyncio.to_thread(
                            container.stop, timeout=st.spec.grace_period_s
                        )
                    else:
                        await asyncio.to_thread(container.remove, force=True)
            except Exception as exc:
                st.last_error = str(exc)
                st.status = "StopError"
                _LOG.error("[stop] Failed to stop %s: %s", st.container_name, exc)

        try:
            await asyncio.shield(_stop_container())
        finally:
            st.container_id = None
            st.host_port = None
            st.host_pid = None
            st._prev_addr = st.addr
            st.addr = None
            self._clear_ready(st)
            st.caps = None
            self._invalidate_resident_baseline(st)
            st.queue_gw_inflight = 0
            self._reset_queue_fields(st)
            self._clear_dispatch_front(st)
            self._notify_dispatch_front_capacity(st)
            st.real_dispatch_count_in_generation = 0

            if st.memory_reserved_mb > 0:
                resource_tracker = getattr(self, "resource_tracker", None)
                if resource_tracker is not None:
                    await resource_tracker.release_weight(
                        st.assigned_gpus, st.memory_reserved_mb
                    )
                st.memory_reserved_mb = 0

            is_guard_kill = "VRAM violation" in (st.last_error or "")
            if st.status != "StopError":
                st.status = "Killed" if is_guard_kill else "Stopped"
                st.lifecycle_state = "killed" if is_guard_kill else "stopped"
            else:
                st.lifecycle_state = "error"

    def estimate_idle_freeable(self, gpu_id: str, *, ignore_grace: bool = False) -> int:
        """Estimate weight VRAM freeable by evicting idle workers on *gpu_id*.

        Same criteria as evict_idle_workers but no side effects — just sums.
        """
        gpu_id = str(gpu_id)
        total = 0
        for st in self._idle_evictable_states(gpu_id=gpu_id, ignore_grace=ignore_grace):
            total += self.get_component_weight(st.spec.component)
        return total

    async def evict_idle_workers(
        self,
        gpu_id: str,
        needed_mb: int,
        *,
        suppress_recovery: bool = True,
        ignore_grace: bool = False,
    ) -> int:
        """Evicts ready workers on a specific GPU using the configured policy.

        Idle eviction is scheduler-intended pressure relief, so background
        recovery is suppressed by default. Demand paths such as dispatch and
        pre-init still restart the worker through ``ensure_worker_ready``.
        """
        gpu_id = str(gpu_id)
        idle_candidates = self._idle_evictable_states(
            gpu_id=gpu_id,
            ignore_grace=ignore_grace,
        )

        if not idle_candidates:
            return 0

        if self._eviction_handler:
            to_evict = self._eviction_handler(idle_candidates, gpu_id, needed_mb)
        else:
            to_evict = self.eviction_policy.select_for_eviction(
                idle_candidates, needed_mb
            )

        freed_mb = 0
        for st in to_evict:
            st.draining = True
            st.recovery_suppressed = bool(suppress_recovery)
            weight_mb = self.get_component_weight(st.spec.component)
            print(
                f"[evict] Policy({self.eviction_policy.__class__.__name__}) Stopping idle worker {st.spec.component}/{st.spec.name} on GPU {gpu_id} to free {weight_mb}MB",
                flush=True,
            )
            mem_to_free = weight_mb
            if self._on_worker_killed and st.assigned_gpus:
                try:
                    self._on_worker_killed(st.spec.component, list(st.assigned_gpus))
                except Exception:
                    _LOG.warning(
                        "[evict] on_worker_killed callback failed for %s",
                        st.spec.name,
                        exc_info=True,
                    )
            await self._stop_state(st)
            freed_mb += mem_to_free

        return freed_mb

    def _idle_evictable_states(
        self,
        gpu_id: str | None = None,
        *,
        ignore_grace: bool = False,
    ) -> list[WorkerState]:
        now = time.time()
        target_gpu = str(gpu_id) if gpu_id is not None else None
        return [
            st
            for st in self.states.values()
            if self._is_idle_evictable_state(
                st,
                now=now,
                target_gpu=target_gpu,
                ignore_grace=ignore_grace,
            )
        ]

    def _is_idle_evictable_state(
        self,
        st: WorkerState,
        *,
        now: float,
        target_gpu: str | None = None,
        ignore_grace: bool = False,
    ) -> bool:
        if not getattr(st.spec, "preemptible", True):
            return False
        if getattr(st, "draining", False):
            return False
        if getattr(st, "lifecycle_state", "") in {
            "stopped",
            "killed",
            "error",
            "stopping",
        }:
            return False
        if not st.ready or not st.addr or not st.container_id:
            return False

        assigned = [str(g) for g in (st.assigned_gpus or [])]
        if target_gpu is not None and target_gpu not in assigned:
            return False

        worker_inflight = (
            st.queue_prepare_inflight
            + st.queue_execute_inflight
            + st.queue_finalize_inflight
        )
        inflight = max(worker_inflight, st.queue_gw_inflight) + st.dispatch_pending
        if inflight != 0:
            return False

        uptime = now - st.last_ready_at
        idle_time = now - st.last_used_at
        idle_grace = self.eviction_grace_period_s
        if self._grace_period_fn:
            try:
                hint_gpu = str(
                    (st.assigned_gpus or ([target_gpu] if target_gpu else [""]))[0]
                )
                idle_grace = self._grace_period_fn(
                    st.spec.component,
                    hint_gpu,
                )
            except Exception:
                pass

        return bool(
            ignore_grace
            or (uptime >= self.eviction_grace_period_s and idle_time >= idle_grace)
        )

    def snapshot_idle_evictable_resources(
        self,
        *,
        ignore_grace: bool = False,
    ) -> IdleEvictableResourceSnapshot:
        idle_ram_mib = 0.0
        idle_weight_by_gpu: dict[str, int] = {}
        idle_count_by_gpu: dict[str, int] = {}
        now = time.time()
        for st in self.states.values():
            if not self._is_idle_evictable_state(
                st,
                now=now,
                ignore_grace=ignore_grace,
            ):
                continue
            idle_ram_mib += float(self._estimated_idle_host_ram_mib(st) or 0)
            weight_mb = int(
                self.get_component_weight(st.spec.component)
                or st.memory_reserved_mb
                or st.actual_vram_mb
                or 0
            )
            for gpu in st.assigned_gpus or []:
                gpu_id = str(gpu)
                idle_weight_by_gpu[gpu_id] = (
                    idle_weight_by_gpu.get(gpu_id, 0) + weight_mb
                )
                idle_count_by_gpu[gpu_id] = idle_count_by_gpu.get(gpu_id, 0) + 1

        return IdleEvictableResourceSnapshot(
            idle_ram_mib=idle_ram_mib,
            idle_weight_by_gpu=idle_weight_by_gpu,
            idle_count_by_gpu=idle_count_by_gpu,
        )

    def _estimated_idle_host_ram_mib(self, st: WorkerState) -> int:
        live_rss = self._cached_idle_host_rss_mib(getattr(st, "host_pid", None))
        if live_rss is not None:
            live_rss_mb = int(max(0.0, live_rss))
            if live_rss_mb > 0:
                return live_rss_mb
        resident = getattr(st, "resident_memory_mib", None)
        if resident is not None:
            try:
                resident_mb = int(float(resident))
                if resident_mb > 0:
                    return resident_mb
            except (TypeError, ValueError):
                pass
        return max(0, int(self.get_component_weight(st.spec.component) or 0))

    def _cached_idle_host_rss_mib(self, pid: int | None) -> float | None:
        if pid is None:
            return None
        try:
            key = int(pid)
        except (TypeError, ValueError):
            return None
        now = time.time()
        cache = getattr(self, "_idle_host_rss_cache", None)
        if cache is None:
            cache = {}
            self._idle_host_rss_cache = cache
        ttl = float(getattr(self, "_idle_host_rss_cache_ttl_sec", 1.0) or 0.0)
        cached = cache.get(key)
        if cached is not None:
            cached_at, cached_value = cached
            if ttl > 0.0 and now - float(cached_at) <= ttl:
                return float(cached_value)
        live_rss = _read_process_rss_mib(key)
        if live_rss is None:
            return None
        live_rss_mb = int(max(0.0, live_rss))
        cache[key] = (now, live_rss_mb)
        return float(live_rss_mb)

    def estimate_idle_host_ram_freeable(self, *, ignore_grace: bool = False) -> int:
        return sum(
            self._estimated_idle_host_ram_mib(st)
            for st in self._idle_evictable_states(ignore_grace=ignore_grace)
        )

    def _cluster_dispatch_activity_count(self) -> int:
        total = 0
        for st in self.states.values():
            worker_inflight = (
                int(getattr(st, "queue_prepare_inflight", 0) or 0)
                + int(getattr(st, "queue_execute_inflight", 0) or 0)
                + int(getattr(st, "queue_finalize_inflight", 0) or 0)
            )
            total += max(worker_inflight, int(getattr(st, "queue_gw_inflight", 0) or 0))
            total += int(getattr(st, "dispatch_pending", 0) or 0)
        return total

    def _cluster_runtime_activity_count(self) -> int:
        total = 0
        for st in self.states.values():
            worker_inflight = (
                int(getattr(st, "queue_prepare_inflight", 0) or 0)
                + int(getattr(st, "queue_execute_inflight", 0) or 0)
                + int(getattr(st, "queue_finalize_inflight", 0) or 0)
            )
            total += max(worker_inflight, int(getattr(st, "queue_gw_inflight", 0) or 0))
        return total

    async def _clear_leaked_host_ram_if_idle(self, *, reason: str) -> int:
        tracker = getattr(self, "resource_tracker", None)
        if tracker is None:
            return 0
        reserved = int(getattr(tracker, "reserved_host_ram_mb", 0) or 0)
        if reserved <= 0:
            return 0
        now = time.time()
        runtime_activity = self._cluster_runtime_activity_count()
        dispatch_activity = self._cluster_dispatch_activity_count()
        if runtime_activity > 0:
            self._last_runtime_activity_ts = now
            return 0
        pending_admissions = 0
        pending_count = getattr(tracker, "pending_admission_count", None)
        if callable(pending_count):
            try:
                pending_admissions = int(await pending_count())
            except Exception:
                _LOG.warning(
                    "[admission-host-ram] pending admission count failed",
                    exc_info=True,
                )
                return 0
        idle_for = max(
            0.0, now - float(getattr(self, "_last_runtime_activity_ts", now))
        )
        if dispatch_activity > 0 or pending_admissions > 0:
            return 0
        if idle_for < float(getattr(self, "host_ram_leak_idle_grace_sec", 10.0)):
            return 0
        cleared = await tracker.clear_reserved_host_ram()
        if cleared > 0:
            _LOG.warning(
                "[admission-host-ram] cleared leaked host RAM reservation: "
                "%d MiB (reason=%s idle_for=%.1fs)",
                cleared,
                reason,
                idle_for,
            )
        return cleared

    async def evict_idle_workers_for_host_ram(
        self,
        needed_mib: int,
        *,
        suppress_recovery: bool = True,
        ignore_grace: bool = False,
    ) -> int:
        """Evict idle workers cluster-wide to relieve host RAM pressure.

        Mirrors VRAM idle eviction: do not let background recovery immediately
        undo an intentional scheduler-side resident-worker eviction.
        """
        remaining = max(0, int(needed_mib or 0))
        if remaining <= 0:
            return 0
        cleared_leak = await self._clear_leaked_host_ram_if_idle(
            reason="idle-host-ram-eviction",
        )
        if cleared_leak > 0:
            return cleared_leak
        idle_candidates = self._idle_evictable_states(ignore_grace=ignore_grace)
        if not idle_candidates:
            return 0

        if self._host_ram_eviction_handler:
            ordered = list(
                self._host_ram_eviction_handler(
                    idle_candidates,
                    remaining,
                    self._estimated_idle_host_ram_mib,
                )
                or []
            )
        else:
            ordered = list(
                self.eviction_policy.select_for_eviction(
                    idle_candidates,
                    remaining,
                    freeable_fn=self._estimated_idle_host_ram_mib,
                )
            )
        freed = 0
        for st in ordered:
            est = self._estimated_idle_host_ram_mib(st)
            if est <= 0:
                continue
            st.draining = True
            st.recovery_suppressed = bool(suppress_recovery)
            if self._on_worker_killed and st.assigned_gpus:
                try:
                    self._on_worker_killed(st.spec.component, list(st.assigned_gpus))
                except Exception:
                    _LOG.warning(
                        "[evict-host-ram] on_worker_killed callback failed for %s",
                        st.spec.name,
                        exc_info=True,
                    )
            await self._stop_state(st)
            freed += est
            print(
                f"[evict-host-ram] Stopping idle worker {st.spec.component}/{st.spec.name} "
                f"to free ~{est}MiB host RAM",
                flush=True,
            )
            if freed >= remaining:
                break
        return freed

    def drain_worker(self, addr_or_name: str) -> bool:
        """Mark a worker as draining — scheduler should stop dispatching new tasks.

        The worker continues running its in-flight tasks.  Once all tasks
        complete (``queue_gw_inflight == 0``), the caller can ``_stop_state``
        the worker safely.
        """
        for st in self.states.values():
            if (st.addr and st.addr == addr_or_name) or st.spec.name == addr_or_name:
                st.draining = True
                _LOG.info("[drain] Worker %s marked as draining", addr_or_name)
                return True
        return False

    def undrain_worker(self, addr_or_name: str) -> bool:
        """Remove the draining flag so the worker can accept new tasks again."""
        for st in self.states.values():
            if (st.addr and st.addr == addr_or_name) or st.spec.name == addr_or_name:
                st.draining = False
                return True
        return False

    async def selective_task_cancel(
        self,
        gpu_id: str,
        needed_mb: int,
        cancel_fn: Any | None = None,
    ) -> int:
        """Try to free VRAM by cancelling the lowest-priority task on busy
        workers sharing *gpu_id*, instead of killing the entire worker.

        Returns the estimated freed activation MB.  Falls back to 0 if no
        task could be cancelled (caller should then try full eviction).
        """
        if cancel_fn is None:
            return 0

        gpu_id = str(gpu_id)
        best_st: WorkerState | None = None
        best_activation = 0

        for st in self.states.values():
            if not st.ready or not st.addr:
                continue
            assigned = [str(g) for g in (st.assigned_gpus or [])]
            if gpu_id not in assigned:
                continue
            if st.queue_gw_inflight <= 0:
                continue
            if st.current_activation_mb > best_activation:
                best_activation = st.current_activation_mb
                best_st = st

        if best_st is None or best_activation <= 0:
            return 0

        try:
            ok = await cancel_fn(best_st.addr)
            if ok:
                _LOG.info(
                    "[selective-cancel] Cancelled task on %s/%s, expected to free ~%d MB",
                    best_st.spec.component,
                    best_st.spec.name,
                    best_activation,
                )
                return best_activation
        except Exception as exc:
            _LOG.warning("[selective-cancel] Failed on %s: %s", best_st.spec.name, exc)

        return 0

    def list_workers(
        self,
        *,
        component: str | None = None,
        include_not_ready: bool = True,
    ) -> list[WorkerState]:
        out: list[WorkerState] = []
        for st in self.states.values():
            if component and st.spec.component != component:
                continue
            if not include_not_ready and (not st.ready or not st.addr):
                continue
            out.append(st)
        out.sort(key=lambda item: (item.spec.component, item.spec.name))
        return out

    def get_ready_workers(
        self, *, component: str, replicas_only: bool | None = None
    ) -> list[WorkerState]:
        _ = replicas_only
        out: list[WorkerState] = []
        for st in self.states.values():
            if st.spec.component != component:
                continue
            if not st.ready or not st.addr:
                continue
            out.append(st)
        out.sort(key=lambda item: item.spec.name)
        return out

    def profile_parallelism(self, component: str) -> int:
        reserved = [
            st
            for st in self.states.values()
            if st.spec.component == component and st.profile_reserved
        ]
        if reserved:
            return max(1, len(reserved))
        all_workers = [
            st for st in self.states.values() if st.spec.component == component
        ]
        return max(1, len(all_workers))

    async def ensure_worker_ready(
        self,
        *,
        component: str,
        worker_name: str,
        campaign_id: str = "",
        on_weight_reserved: Callable[[], None] | None = None,
    ) -> tuple[WorkerState, bool]:
        """fix — propagates ``did_activate`` to the caller chain."""
        state = None
        for st in self.states.values():
            if st.spec.component == component and st.spec.name == worker_name:
                state = st
                break
        if state is None:
            raise RuntimeError(f"worker not found: {component}/{worker_name}")
        return await self._ensure_state_ready(
            state,
            campaign_id=campaign_id,
            on_weight_reserved=on_weight_reserved,
        )

    async def wait_for_ready(
        self,
        worker_name: str,
        *,
        component: str | None = None,
        timeout_s: float | None = None,
    ) -> WorkerState:
        """fix R2: block until worker is (re-)ready for dispatch.

        Distinct from ``ensure_worker_ready``: does **not** trigger a
        cold-start.  Intended as a post-activation re-check gate in
        Stage 4 against stats_loop reset sites (crash-detect / guard-kill
        / gRPC fail) that may fire between activation return and Stage 6
        dispatch.

        Behavior:
          - If ``st.ready and st.addr`` on entry → return immediately.
          - Else await ``st.ready_event`` (re-armed by ``_clear_ready``).
          - On wake, re-verify predicate; loop if transient.
          - Raises ``TimeoutError`` if ``timeout_s`` elapses.
          - Raises ``RuntimeError`` if worker not resolvable.
        """
        st = self._resolve_worker(worker_name, component=component)
        if st is None:
            raise RuntimeError(f"worker not found: {worker_name!r}")

        deadline = None if timeout_s is None else (time.time() + float(timeout_s))
        while True:
            if st.ready and st.addr:
                return st
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            if remaining == 0.0:
                raise TimeoutError(
                    f"wait_for_ready timeout for {worker_name!r} "
                    f"(component={component}, timeout_s={timeout_s})"
                )
            try:
                if remaining is None:
                    await st.ready_event.wait()
                else:
                    await asyncio.wait_for(st.ready_event.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"wait_for_ready timeout for {worker_name!r} "
                    f"(component={component}, timeout_s={timeout_s})"
                ) from exc

    async def ensure_component_pool_ready(
        self,
        *,
        component: str,
        gpu_ids: list[str] | None = None,
        campaign_id: str = "",
        on_weight_reserved: Callable[[], None] | None = None,
    ) -> list[WorkerState]:
        wanted = {
            str(item).strip() for item in list(gpu_ids or []) if str(item).strip()
        }
        targets: list[WorkerState] = []
        for st in self.states.values():
            if st.spec.component != component:
                continue
            assigned = [
                str(item).strip()
                for item in list(st.assigned_gpus or st.spec.gpus or [])
                if str(item).strip()
            ]
            if wanted and (not assigned or assigned[0] not in wanted):
                continue
            targets.append(st)

        if not targets:
            if wanted:
                wanted_txt = ", ".join(sorted(wanted))
                raise RuntimeError(
                    f"no workers configured for component '{component}' with gpu_ids [{wanted_txt}]"
                )
            raise RuntimeError(f"no workers configured for component '{component}'")

        if campaign_id or on_weight_reserved is not None:
            await asyncio.gather(
                *(
                    self._ensure_state_ready(
                        st,
                        campaign_id=campaign_id,
                        on_weight_reserved=on_weight_reserved,
                    )
                    for st in targets
                )
            )
        else:
            await asyncio.gather(*(self._ensure_state_ready(st) for st in targets))
        ready = [st for st in targets if st.ready and st.addr]
        if len(ready) != len(targets):
            raise RuntimeError(f"failed to prepare component pool for '{component}'")
        return ready

    def on_task_start(
        self,
        addr: str,
        activation_mb: int = 0,
        activation_is_guarded: bool = False,
        worker_name: str | None = None,
    ) -> None:
        """Called by Gateway when a task is dispatched to a worker.

        Increments ``gw_inflight`` (task is now active from the gateway's
        perspective). ``dispatch_pending`` is intentionally retained until a
        stats poll observes the request in the worker execute stage; clearing
        it at gRPC send time opens a hidden pre-execute backlog window.

        Worker recycle can invalidate the original ready-worker address
        between reservation acquire and task-start consume. Mirror the
        fallback discipline used by ``release_dispatch_pending`` /
        ``on_task_end`` so the accounting still lands on the current
        worker generation and does not strand idle workers behind a stale
        ``dispatch_pending`` count.
        """
        found = self._find_dispatch_token_worker(addr, worker_name=worker_name)
        if found is None:
            _LOG.warning(
                "[on_task_start] no worker match for addr=%s name=%s",
                addr,
                worker_name or "",
            )
            return
        self._clear_dispatch_front_handoff_protection(found)
        found.queue_gw_inflight += 1
        found.current_activation_mb += int(activation_mb)
        if activation_is_guarded:
            found.activation_guarded_count += 1
        if self._worker_generation_token(found):
            found.real_dispatch_count_in_generation += 1

    def bump_dispatch_pending(self, addr_or_name: str) -> bool:
        """Called when a task selects this worker but hasn't acquired sem/vram yet.
        Prevents eviction during the selection→dispatch window.
        Matches by addr first; falls back to worker name (for cold workers
        that have no addr yet)."""

        def _can_bump(st: WorkerState) -> bool:
            if int(getattr(st, "dispatch_pending", 0) or 0) <= 0:
                return True
            if self._stale_idle_dispatch_front(st):
                self._clear_dispatch_front(st)
                return True
            return False

        for st in self.states.values():
            if st.addr and st.addr == addr_or_name:
                if not _can_bump(st):
                    return False
                st.dispatch_pending += 1
                st.dispatch_pending_updated_at = time.time()
                return True
        for st in self.states.values():
            if st.spec.name == addr_or_name:
                if not _can_bump(st):
                    return False
                st.dispatch_pending += 1
                st.dispatch_pending_updated_at = time.time()
                return True
        return False

    def release_dispatch_pending(
        self, addr_or_name: str, worker_name: str | None = None
    ) -> None:
        """Called on early exit (VRAM fail, error) before on_task_start runs."""
        found = self._find_dispatch_token_worker(
            addr_or_name,
            worker_name=worker_name or addr_or_name,
        )
        if found is not None:
            before = found.dispatch_pending
            found.dispatch_pending = max(0, found.dispatch_pending - 1)
            found.dispatch_pending_updated_at = time.time()
            if found.dispatch_pending == 0:
                self._clear_dispatch_front_handoff_protection(found)
            if found.dispatch_pending < before:
                self._notify_dispatch_front_capacity(found)

    def adjust_worker_inflight_by_name(self, name: str, delta: int) -> None:
        for st in self.states.values():
            if st.spec.name == name:
                st.queue_gw_inflight = max(0, st.queue_gw_inflight + delta)
                break

    async def adjust_current_activation(self, addr: str, delta_mb: int) -> int:
        """Mid-flight upward upcap of an active worker's activation reservation.

        Used by the drift responder when a fresh GP-observed activation peak
        exceeds the admission-time reservation that was granted to the
        currently-running task.  The static reservation laid down by
        ``on_task_start`` was a cold-start point estimate; this method
        propagates a refreshed prediction into both worker state and the
        VRAM tracker so that subsequent ``acquire_activation`` calls evaluate
        feasibility against the latest evidence rather than the stale
        admission-time prediction.

        Monotone-up only: ``delta_mb`` must be > 0.  Lowering an active
        reservation is forbidden — a transient cold-start peak observation
        could otherwise yank reservation away from a worker whose current
        peak is still high.  The release is owned by ``on_task_end``.

        Lookup mirrors ``on_task_end``: addr first, ``_prev_addr`` fallback
        for guard-killed / crashed workers (only when no other worker
        currently owns this addr — prevents port-reuse collision).

        Returns: the new ``current_activation_mb`` value (post-upcap).

        Raises:
            ValueError: if ``delta_mb`` is not strictly positive.
            KeyError: if no worker can be located for ``addr``.
            RuntimeError: if the located worker has no assigned GPU
                          (cannot apply VRAM tracker delta).
        """
        if int(delta_mb) <= 0:
            raise ValueError(
                f"adjust_current_activation: delta_mb must be > 0, got {delta_mb}"
            )
        found = None
        for st in self.states.values():
            if st.addr == addr:
                found = st
                break
        if found is None:
            addr_in_use = any(s.addr == addr for s in self.states.values())
            if not addr_in_use:
                for st in self.states.values():
                    if getattr(st, "_prev_addr", None) == addr:
                        found = st
                        break
        if found is None:
            raise KeyError(f"adjust_current_activation: no worker for addr={addr}")
        if not found.assigned_gpus:
            raise RuntimeError(
                f"adjust_current_activation: worker {found.spec.name} (addr={addr}) "
                f"has no assigned GPU — cannot apply VRAM tracker delta"
            )

        old_total = int(found.current_activation_mb)
        new_total = old_total + int(delta_mb)
        found.current_activation_mb = new_total
        await self.resource_tracker.update_reservation(
            found.assigned_gpus,
            old_total,
            new_total,
        )
        await self.resource_tracker.notify_waiters()
        _LOG.info(
            "[activation-upcap] %s addr=%s gpus=%s: %d → %d MB (+%d)",
            found.spec.name,
            addr,
            found.assigned_gpus,
            old_total,
            new_total,
            int(delta_mb),
        )
        return new_total

    async def on_task_end(
        self,
        addr: str,
        activation_mb: int = 0,
        activation_is_guarded: bool = False,
        worker_name: str | None = None,
    ) -> None:
        """Called by Gateway when a task on a worker completes."""
        found = self._find_dispatch_token_worker(addr, worker_name=worker_name)
        if found is not None:
            before_dispatch_pending = found.dispatch_pending
            found.dispatch_pending = max(0, found.dispatch_pending - 1)
            found.dispatch_pending_updated_at = time.time()
            if found.dispatch_pending == 0:
                self._clear_dispatch_front_handoff_protection(found)
            found.queue_gw_inflight = max(0, found.queue_gw_inflight - 1)
            found.current_activation_mb = max(
                0, found.current_activation_mb - int(activation_mb)
            )
            if activation_is_guarded:
                found.activation_guarded_count = max(
                    0, found.activation_guarded_count - 1
                )
            if found.dispatch_pending < before_dispatch_pending:
                self._notify_dispatch_front_capacity(found)

        if found and found.queue_gw_inflight == 0 and found.queue_execute_inflight == 0:
            self._reset_queue_fields(found)
            await self.resource_tracker.notify_waiters()

        if found:
            _gpus = found.assigned_gpus or found.spec.gpus or []
            for gid in [str(g) for g in _gpus]:
                self.resource_tracker._notify_compute_waiters(gid)

    def _find_dispatch_token_worker(
        self,
        addr_or_name: str,
        *,
        worker_name: str | None = None,
    ) -> WorkerState | None:
        """Find the worker that owns a dispatch-front token.

        Gateway callbacks carry both the worker address used for the RPC and
        the planned worker name.  Prefer the name when present so a late
        completion from an old worker generation cannot debit a different
        worker that has since reused the same address.
        """
        addr = str(addr_or_name or "").strip()
        name = str(worker_name or "").strip()
        states = list(getattr(self, "states", {}).values())
        if name:
            for st in states:
                if st.spec.name == name and str(getattr(st, "addr", "") or "") == addr:
                    return st
            for st in states:
                if (
                    st.spec.name == name
                    and str(getattr(st, "_prev_addr", "") or "") == addr
                ):
                    return st
            for st in states:
                if st.spec.name == name:
                    return st
        for st in states:
            if str(getattr(st, "addr", "") or "") == addr:
                return st
        addr_in_use = any(str(getattr(st, "addr", "") or "") == addr for st in states)
        if not addr_in_use:
            for st in states:
                if str(getattr(st, "_prev_addr", "") or "") == addr:
                    return st
        return None

    def check_worker_status(self, addr: str) -> str | None:
        """Checks if a worker is still healthy. Returns error message if dead."""
        st = None
        for s in self.states.values():
            if s.addr == addr:
                st = s
                break
        if not st:
            return None

        container = self._get_container(st)
        if not container:
            return "worker container missing"

        if container.status in {"exited", "dead"}:
            state = container.attrs.get("State", {})
            exit_code = state.get("ExitCode", -1)
            oom_killed = state.get("OOMKilled", False)

            error_msg = f"worker container {st.container_name} {container.status} (exit code {exit_code})"
            if oom_killed:
                error_msg += " [OOM KILLED]"

            try:
                logs = container.logs(tail=20).decode("utf-8", errors="replace").strip()
                if logs:
                    error_msg += f"\nLast logs:\n{logs}"
            except Exception:
                pass

            return error_msg

        return None

    async def _ensure_state_ready(
        self,
        st: WorkerState,
        campaign_id: str = "",
        *,
        on_weight_reserved: Callable[[], None] | None = None,
    ) -> tuple[WorkerState, bool]:
        """Plan fix — returns ``(WorkerState, did_activate)``.

        ``did_activate=True`` iff this call performed real cold-start work
        (container restart / worker process initialization).  ``False`` on
        the idempotent early-return path when the worker was already
        ready + addressable.  The caller chain propagates the flag up to
        ``RealityValidator._activate_worker`` so ``was_cold_start``
        matches the supervisor's authoritative init-execution signal
        rather than proxy flags (``plan.needs_cold_start`` or
        ``worker.ready`` snapshots) that race under fix/f paths.
        """
        key = self._state_key(st.spec)
        lock = self._spawn_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if st.ready and st.addr:
                st.last_used_at = time.time()
                return st, False

            st.recovery_suppressed = False
            await self._stop_state(st)

            weight_mb = self.get_component_weight(st.spec.component)

            if weight_mb <= 0:
                await self.resource_tracker.ensure_initialized()
                any_gpu = (
                    self.resource_tracker.gpu_pool[0]
                    if self.resource_tracker.gpu_pool
                    else "0"
                )
                total_mb = self.resource_tracker.total_vram.get(
                    any_gpu, gpu_vram_fallback_mib()
                )

                weight_mb = self.weight_oom_guard.get_reservation_mb(
                    st.spec.component, total_mb
                )
                st.is_guarded = True
                _LOG.info(
                    "[oom-guard] Weight miss for %s, reserving %d MB via %s (Guarded Probing)",
                    st.spec.component,
                    weight_mb,
                    self.weight_oom_guard.__class__.__name__,
                )

            if weight_mb > 0:
                if not st.assigned_gpus:
                    st.assigned_gpus = list(st.spec.gpus or [])

                temp_assigned = st.assigned_gpus
                if not temp_assigned and st.spec.gpu_count > 0 and self.gpu_policy:
                    pending_usage = self._get_pending_gpu_usage()
                    temp_assigned = self.gpu_policy.allocate(
                        st.spec.gpu_count, pending_usage=pending_usage
                    )

                if temp_assigned:
                    await self.resource_tracker.acquire_weight(
                        temp_assigned,
                        weight_mb,
                        campaign_id=campaign_id,
                        evict_fn=self.evict_idle_workers,
                    )
                    st.assigned_gpus = temp_assigned
                    st.memory_reserved_mb = weight_mb
                    if on_weight_reserved is not None:
                        try:
                            on_weight_reserved()
                        except Exception:
                            _LOG.warning(
                                "pre-init soft-reservation handoff failed for %s",
                                st.spec.component,
                                exc_info=True,
                            )

            cold_start_begin = time.time()
            try:
                await self._start_one(st.spec)
                await self._ready_worker(st)
            except Exception:
                await self._stop_state(st)
                raise

            if not st.ready or not st.addr:
                await self._stop_state(st)
                raise RuntimeError(
                    f"worker failed to become ready: {st.spec.component}/{st.spec.name} ({st.last_error})"
                )
            st.last_used_at = time.time()
            st.last_ready_at = time.time()

            init_sec = time.time() - cold_start_begin
            if init_sec > 0 and self.init_tracker:
                recorded_init_sec = init_sec
                cap_raw = self.profiling_runtime.get("init_observation_cap_sec")
                if cap_raw is not None:
                    try:
                        cap_sec = float(cap_raw)
                    except (TypeError, ValueError):
                        cap_sec = 0.0
                    if cap_sec > 0:
                        recorded_init_sec = min(recorded_init_sec, cap_sec)
                co_located: list[str] = []
                for gpu_id in st.assigned_gpus or []:
                    for other_st in self.states.values():
                        if other_st is st:
                            continue
                        if not other_st.ready:
                            continue
                        other_gpus = other_st.assigned_gpus or other_st.spec.gpus or []
                        if gpu_id in [str(g) for g in other_gpus]:
                            co_located.append(other_st.spec.component)
                gpu_id_for_init = str(st.assigned_gpus[0]) if st.assigned_gpus else "0"
                self.init_tracker.record(
                    st.spec.component,
                    gpu_id_for_init,
                    recorded_init_sec,
                    co_located,
                )
                _LOG.info(
                    "[init] %s on GPU %s: %.1fs recorded=%.3fs (co_located=%s)",
                    st.spec.component,
                    gpu_id_for_init,
                    init_sec,
                    recorded_init_sec,
                    co_located,
                )
            return st, True

    async def start_profile_replicas(
        self, *, component: str, gpu_ids: list[str]
    ) -> list[WorkerState]:
        if self._isolated_component and self._isolated_component != component:
            raise RuntimeError(
                f"profiling isolation already active for component '{self._isolated_component}'"
            )
        if not gpu_ids:
            raise RuntimeError("profiling isolation requires at least one gpu id")

        wanted = {str(item).strip() for item in list(gpu_ids) if str(item).strip()}
        target_workers: list[WorkerState] = []
        for st in self.states.values():
            assigned = [
                str(item).strip()
                for item in list(st.assigned_gpus or st.spec.gpus or [])
                if str(item).strip()
            ]
            is_target_component = st.spec.component == component
            is_wanted_gpu = bool(assigned) and assigned[0] in wanted
            st.profile_reserved = bool(is_target_component and is_wanted_gpu)
            if st.profile_reserved:
                target_workers.append(st)

        if len(target_workers) != len(wanted):
            allocated = {
                str((st.assigned_gpus or st.spec.gpus or [""])[0]).strip()
                for st in target_workers
                if (st.assigned_gpus or st.spec.gpus or [])
            }
            missing = sorted(wanted - allocated)
            missing_txt = ", ".join(missing) if missing else "unknown"
            raise RuntimeError(
                f"profiling isolation missing component workers for gpu_ids [{missing_txt}]"
            )

        for st in self.states.values():
            if st.profile_reserved:
                continue
            if st.container_id:
                await self._stop_state(st)

        await asyncio.gather(*(self._ensure_state_ready(st) for st in target_workers))
        self._isolated_component = component
        return [st for st in target_workers if st.ready and st.addr]

    async def stop_profile_replicas(self) -> None:
        for st in self.states.values():
            st.profile_reserved = False
        self._isolated_component = None

    async def restore_base_workers(self) -> None:
        await self.stop_profile_replicas()

    @staticmethod
    def _invalidate_resident_baseline(st: WorkerState) -> None:
        st.resident_memory_mib = None
        st.resident_memory_source = ""
        st.resident_baseline_collected_at = None
        st.resident_baseline_lifecycle_token = ""
        st.resident_baseline_state = RESIDENT_BASELINE_STATE_INVALIDATED
        st.real_dispatch_count_in_generation = 0

    @staticmethod
    def _mark_resident_baseline_stale(st: WorkerState) -> None:
        if (
            str(st.resident_baseline_state or "").strip()
            == RESIDENT_BASELINE_STATE_READY
        ):
            st.resident_baseline_state = RESIDENT_BASELINE_STATE_STALE

    @staticmethod
    def _apply_resident_baseline(st: WorkerState, baseline: Any) -> None:
        prior_token = str(st.resident_baseline_lifecycle_token or "").strip()
        state = str(getattr(baseline, "resident_baseline_state", "") or "").strip()
        if not state:
            state = RESIDENT_BASELINE_STATE_MISSING
        resident_memory_source = str(
            getattr(baseline, "resident_memory_source", "") or ""
        ).strip()
        resident_memory_mib: float | None = None
        try:
            parsed_memory = float(getattr(baseline, "resident_memory_mib", 0.0) or 0.0)
        except Exception:
            parsed_memory = 0.0
        if resident_memory_source and state in {
            RESIDENT_BASELINE_STATE_READY,
            RESIDENT_BASELINE_STATE_STALE,
        }:
            resident_memory_mib = parsed_memory
        collected_at: float | None = None
        try:
            parsed_collected_at = float(
                getattr(baseline, "resident_baseline_collected_at", 0.0) or 0.0
            )
        except Exception:
            parsed_collected_at = 0.0
        if parsed_collected_at > 0:
            collected_at = parsed_collected_at
        st.resident_memory_mib = resident_memory_mib
        st.resident_memory_source = (
            resident_memory_source if resident_memory_mib is not None else ""
        )
        st.resident_baseline_collected_at = collected_at
        next_token = str(
            getattr(baseline, "resident_baseline_lifecycle_token", "") or ""
        ).strip()
        st.resident_baseline_lifecycle_token = next_token
        st.resident_baseline_state = state
        if not next_token or next_token != prior_token:
            st.real_dispatch_count_in_generation = 0

    @staticmethod
    def _is_hot(st: WorkerState) -> bool:
        token = str(st.resident_baseline_lifecycle_token or "").strip()
        return bool(token) and int(st.real_dispatch_count_in_generation or 0) > 0

    @classmethod
    def _worker_generation_token(cls, st: WorkerState) -> str:
        return str(st.resident_baseline_lifecycle_token or "").strip()

    @classmethod
    def _snapshot_worker_state(
        cls,
        st: WorkerState,
        *,
        include_live_rss: bool = True,
    ) -> dict[str, Any]:
        worker_inflight = (
            st.queue_prepare_inflight
            + st.queue_execute_inflight
            + st.queue_finalize_inflight
        )
        inflight = int(max(worker_inflight, st.queue_gw_inflight))
        resident_memory_mib = st.resident_memory_mib
        resident_memory_source = st.resident_memory_source
        if include_live_rss:
            live_rss = _read_process_rss_mib(getattr(st, "host_pid", None))
            if live_rss is not None and live_rss > 0:
                resident_memory_mib = float(live_rss)
                resident_memory_source = "host_pid_rss"
        return {
            "component": st.spec.component,
            "name": st.spec.name,
            "status": st.status,
            "state": st.lifecycle_state,
            "gpu_ids": list(st.assigned_gpus or st.spec.gpus or []),
            "reserved": "yes" if st.memory_reserved_mb > 0 else "no",
            "profile_reserved": bool(st.profile_reserved),
            "addr": st.addr or "",
            "ready": bool(st.ready),
            "image": st.spec.image,
            "error": st.last_error,
            "inflight": inflight,
            "in_queue": int(st.queue_in_queue),
            "prepared_queue": int(st.queue_prepared_queue),
            "output_queue": int(st.queue_output_queue),
            "prepare_inflight": int(st.queue_prepare_inflight),
            "execute_inflight": int(st.queue_execute_inflight),
            "finalize_inflight": int(st.queue_finalize_inflight),
            "gw_inflight": int(st.queue_gw_inflight),
            "dispatch_pending": int(st.dispatch_pending),
            "max_concurrency": int(st.max_concurrency),
            "priority": int(st.spec.priority),
            "actual_vram": int(st.actual_vram_mb),
            "gpu_util_percent": float(st.gpu_util_percent),
            "resident_memory_mib": resident_memory_mib,
            "resident_memory_source": resident_memory_source,
            "resident_baseline_collected_at": st.resident_baseline_collected_at,
            "resident_baseline_lifecycle_token": st.resident_baseline_lifecycle_token,
            "resident_baseline_state": st.resident_baseline_state,
            "worker_generation_token": cls._worker_generation_token(st),
            "real_dispatch_count_in_generation": int(
                st.real_dispatch_count_in_generation
            ),
            "is_hot": cls._is_hot(st),
        }

    def list_worker_snapshots(
        self, *, include_live_rss: bool = True
    ) -> list[dict[str, Any]]:
        return [
            self._snapshot_worker_state(st, include_live_rss=include_live_rss)
            for st in self.list_workers(include_not_ready=True)
        ]

    def list_worker_snapshots_light(self) -> list[dict[str, Any]]:
        """ fix #3 — narrow snapshot for ``PlacementContext.
        refresh_gpu_views``.  ``list_worker_snapshots`` produces a 30+
        field dict per worker (image, error, resident_baseline_*, queue
        stats, ...) but ``refresh_gpu_views`` only consumes six of them
        (``gpu_ids``, ``component``, ``status``, ``execute_inflight``,
        ``actual_vram``, ``gpu_util_percent``).  Building the heavy dict
        on every drift cascade — once per coalesce flush — accounted for
        ~12-15% of main-thread CPU during  9-pipeline saturation
        (py-spy 50-sample profile).  This light variant skips the
        unused fields and keeps the hot path proportional to *6 ×
        n_workers* attribute reads instead of *30+ × n_workers*.

        Used only by drift cascade ``_on_scenario_changed_impl``.  All
        other call sites (``_print_status``, ops console export, admin
        introspection) keep using ``list_worker_snapshots`` because they
        depend on the full field set.
        """
        return [
            {
                "gpu_ids": list(st.assigned_gpus or st.spec.gpus or []),
                "component": st.spec.component,
                "status": st.status,
                "execute_inflight": int(st.queue_execute_inflight),
                "actual_vram": float(st.actual_vram_mb),
                "gpu_util_percent": float(st.gpu_util_percent),
            }
            for st in self.list_workers(include_not_ready=True)
        ]

    def _print_status(self) -> None:
        headers = [
            "component",
            "name",
            "status",
            "gpu_ids",
            "reserved",
            "addr",
            "image",
            "error",
            "actual_vram",
        ]
        rows: list[list[str]] = []
        for snap in self.list_worker_snapshots():
            err = str(snap["error"]).replace("\t", " ")[:120]
            gpu_ids = ",".join([str(x) for x in snap["gpu_ids"]]) or "-"
            rows.append(
                [
                    str(snap["component"]),
                    str(snap["name"]),
                    str(snap["status"]),
                    gpu_ids,
                    str(snap["reserved"]),
                    str(snap["addr"] or "-"),
                    str(snap["image"]),
                    err,
                    str(snap["actual_vram"]) if int(snap["actual_vram"]) > 0 else "-",
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def _fmt(row: list[str]) -> str:
            return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

        print("", flush=True)
        print("Worker Status", flush=True)
        print(_fmt(headers), flush=True)
        print(_fmt(["-" * w for w in widths]), flush=True)
        for row in rows:
            print(_fmt(row), flush=True)

    def _render_queue_stats(self, snapshots: list[dict[str, object]]) -> str:
        headers = [
            "component",
            "name",
            "state",
            "prep_q",
            "prep_in",
            "exec_q",
            "exec_in",
            "fin_q",
            "fin_in",
            "gw_in",
            "pending",
            "resource_q",
        ]
        rows: list[list[str]] = []
        resource_tracker = getattr(self, "resource_tracker", None)
        for snap in snapshots:
            gpu_id = str((snap.get("gpu_ids") or ["-"])[0])
            resource_q = (
                resource_tracker.get_queue_length(gpu_id)
                if resource_tracker is not None and gpu_id != "-"
                else 0
            )

            rows.append(
                [
                    str(snap["component"]),
                    str(snap["name"]),
                    str(snap["state"]),
                    str(snap.get("in_queue", 0)),
                    str(snap.get("prepare_inflight", 0)),
                    str(snap.get("prepared_queue", 0)),
                    str(snap.get("execute_inflight", 0)),
                    str(snap.get("output_queue", 0)),
                    str(snap.get("finalize_inflight", 0)),
                    str(snap.get("gw_inflight", 0)),
                    str(snap.get("dispatch_pending", 0)),
                    str(resource_q),
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def _fmt(row: list[str]) -> str:
            return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

        lines = ["Worker Queues", _fmt(headers), _fmt(["-" * w for w in widths])]
        lines.extend(_fmt(row) for row in rows)
        return "\n".join(lines)

    def _print_queue_stats(self, snapshots: list[dict[str, object]]) -> None:
        print("", flush=True)
        print(self._render_queue_stats(snapshots), flush=True)

    @staticmethod
    def _reset_queue_fields(st: WorkerState) -> None:
        st.queue_in_queue = 0
        st.queue_prepared_queue = 0
        st.queue_output_queue = 0
        st.queue_prepare_inflight = 0
        st.queue_execute_inflight = 0
        st.queue_finalize_inflight = 0
        st._front_seen_execute_inflight = 0

    @staticmethod
    def _clear_dispatch_front_handoff_protection(st: WorkerState) -> None:
        st.dispatch_pending_handoff_protected = False
        st.dispatch_pending_handoff_started_at = 0.0

    @staticmethod
    def _clear_dispatch_front(st: WorkerState) -> None:
        st.dispatch_pending = 0
        st.dispatch_pending_updated_at = time.time()
        WorkerSupervisor._clear_dispatch_front_handoff_protection(st)
        st._front_seen_execute_inflight = int(
            getattr(st, "queue_execute_inflight", 0) or 0
        )

    @staticmethod
    def _worker_lifecycle_label(st: WorkerState) -> str:
        """Return the supervisor lifecycle label used for front-token safety.

        Older tests used an ad-hoc ``state`` attribute, but production
        ``WorkerState`` exposes ``lifecycle_state`` and ``status``. Prefer the
        real fields so cold-start tokens are not mistaken for stale idle tokens
        while the worker is still starting.
        """
        return (
            str(
                getattr(st, "lifecycle_state", "")
                or getattr(st, "state", "")
                or getattr(st, "status", "")
                or ""
            )
            .strip()
            .lower()
        )

    @classmethod
    def _worker_is_starting(cls, st: WorkerState) -> bool:
        label = cls._worker_lifecycle_label(st)
        return label in {"starting", "waitingready", "gettingcaps"}

    @staticmethod
    def _stale_idle_dispatch_front(
        st: WorkerState, *, now: float | None = None
    ) -> bool:
        if int(getattr(st, "dispatch_pending", 0) or 0) <= 0:
            return False
        ts = time.time() if now is None else float(now)
        if bool(getattr(st, "dispatch_pending_handoff_protected", False)):
            started_at = float(
                getattr(st, "dispatch_pending_handoff_started_at", 0.0) or 0.0
            )
            if started_at <= 0.0:
                return False
            if (
                ts - started_at
                < WorkerSupervisor.DISPATCH_FRONT_HANDOFF_ORPHAN_TIMEOUT_SEC
            ):
                return False
        if WorkerSupervisor._worker_is_starting(st):
            return False
        if (
            int(getattr(st, "queue_gw_inflight", 0) or 0) != 0
            or int(getattr(st, "queue_in_queue", 0) or 0) != 0
            or int(getattr(st, "queue_prepare_inflight", 0) or 0) != 0
            or int(getattr(st, "queue_prepared_queue", 0) or 0) != 0
            or int(getattr(st, "queue_execute_inflight", 0) or 0) != 0
            or int(getattr(st, "queue_finalize_inflight", 0) or 0) != 0
        ):
            return False
        updated_at = float(getattr(st, "dispatch_pending_updated_at", 0.0) or 0.0)
        if updated_at <= 0.0:
            return False
        return ts - updated_at >= 2.0

    def reconcile_dispatch_front_capacity(self, worker_name: str | None = None) -> int:
        """Clear stale idle front tokens and publish capacity immediately.

        Stats polling also performs this cleanup, but worker-front blocked
        tasks can otherwise wait for the next stats cadence even though the
        worker is already idle.  Dispatch-front capacity queries call this
        method before making admission decisions so stale local tokens cannot
        strand primary work behind an already-idle worker.
        """
        wanted = str(worker_name or "").strip()
        now = time.time()
        cleared = 0
        for st in list(getattr(self, "states", {}).values()):
            if wanted and st.spec.name != wanted:
                continue
            if not self._stale_idle_dispatch_front(st, now=now):
                continue
            pending = int(getattr(st, "dispatch_pending", 0) or 0)
            age = now - float(getattr(st, "dispatch_pending_updated_at", 0.0) or 0.0)
            _LOG.warning(
                "[dispatch-front] clearing stale front token worker=%s "
                "pending=%d gw_inflight=%d age=%.1fs",
                st.spec.name,
                pending,
                int(getattr(st, "queue_gw_inflight", 0) or 0),
                age,
            )
            self._clear_dispatch_front(st)
            cleared += pending
            self._notify_dispatch_front_capacity(st)
        return cleared

    @staticmethod
    def _dispatch_front_backlog_for_state(st: WorkerState) -> int:
        worker_not_executing = (
            int(getattr(st, "queue_in_queue", 0) or 0)
            + int(getattr(st, "queue_prepare_inflight", 0) or 0)
            + int(getattr(st, "queue_prepared_queue", 0) or 0)
        )
        return max(
            0,
            int(getattr(st, "dispatch_pending", 0) or 0),
            worker_not_executing,
        )

    def _notify_dispatch_front_capacity(self, st: WorkerState) -> None:
        hook = getattr(self, "dispatch_front_capacity_hook", None)
        if not callable(hook):
            return
        try:
            hook(st.spec.name)
        except Exception:
            _LOG.warning(
                "[dispatch-front] capacity hook failed for %s",
                st.spec.name,
                exc_info=True,
            )

    @classmethod
    def _apply_worker_stats(cls, st: WorkerState, stats: pb.WorkerStats) -> None:
        queues = stats.queues
        previous_execute = int(getattr(st, "queue_execute_inflight", 0) or 0)
        st.queue_in_queue = int(queues.in_queue)
        st.queue_prepared_queue = int(queues.prepared_queue)
        st.queue_output_queue = int(queues.output_queue)
        st.queue_prepare_inflight = int(queues.prepare_inflight)
        st.queue_execute_inflight = int(queues.execute_inflight)
        st.queue_finalize_inflight = int(queues.finalize_inflight)
        baseline = int(
            getattr(st, "_front_seen_execute_inflight", previous_execute) or 0
        )
        if st.queue_execute_inflight < baseline:
            baseline = st.queue_execute_inflight
        execute_delta = max(0, st.queue_execute_inflight - baseline)
        if execute_delta > 0 and st.dispatch_pending > 0:
            st.dispatch_pending = max(0, st.dispatch_pending - execute_delta)
            st.dispatch_pending_updated_at = time.time()
            if st.dispatch_pending == 0:
                cls._clear_dispatch_front_handoff_protection(st)
        st._front_seen_execute_inflight = st.queue_execute_inflight
        if cls._stale_idle_dispatch_front(st):
            age = time.time() - float(st.dispatch_pending_updated_at or 0.0)
            _LOG.warning(
                "[dispatch-front] clearing stale front token worker=%s "
                "pending=%d gw_inflight=%d age=%.1fs",
                st.spec.name,
                st.dispatch_pending,
                int(getattr(st, "queue_gw_inflight", 0) or 0),
                age,
            )
            cls._clear_dispatch_front(st)
        cls._apply_resident_baseline(st, getattr(stats, "resident_baseline", None))

    async def _update_worker_snapshot(self, st: WorkerState) -> dict[str, object]:
        if not st.addr or not st.ready:
            self._reset_queue_fields(st)
        else:
            container = await asyncio.to_thread(self._get_container, st)
            if container and container.status in {"exited", "dead"}:
                state_info = container.attrs.get("State", {})
                exit_code = state_info.get("ExitCode", -1)
                oom = state_info.get("OOMKilled", False)
                reason = (
                    f"container {st.container_name} unexpectedly "
                    f"{container.status} (exit={exit_code}{', OOM' if oom else ''})"
                )
                _LOG.error("[crash-detect] %s — marking stopped for recovery", reason)
                _crash_gpus = list(st.assigned_gpus or st.spec.gpus or [])
                self._clear_ready(st)
                st._prev_addr = st.addr
                st.addr = None
                st.host_pid = None
                st.assigned_gpus = []
                st.lifecycle_state = "stopped"
                st.last_error = reason
                st.queue_gw_inflight = 0
                self._invalidate_resident_baseline(st)
                self._reset_queue_fields(st)
                self._clear_dispatch_front(st)
                self._notify_dispatch_front_capacity(st)
                resource_tracker = getattr(self, "resource_tracker", None)
                if st.memory_reserved_mb > 0 and resource_tracker is not None:
                    await resource_tracker.release_weight(
                        [str(g) for g in _crash_gpus],
                        st.memory_reserved_mb,
                    )
                    st.memory_reserved_mb = 0
                if resource_tracker is not None:
                    await resource_tracker.notify_waiters()
                if self._on_worker_killed and _crash_gpus:
                    try:
                        self._on_worker_killed(
                            st.spec.component, [str(g) for g in _crash_gpus]
                        )
                    except Exception:
                        _LOG.warning(
                            "[crash-detect] on_worker_killed callback failed for %s",
                            st.spec.name,
                            exc_info=True,
                        )
            else:
                try:
                    stats = await get_stats(st.addr)
                    before_front_backlog = self._dispatch_front_backlog_for_state(st)
                    self._apply_worker_stats(st, stats)
                    after_front_backlog = self._dispatch_front_backlog_for_state(st)
                    if after_front_backlog < before_front_backlog:
                        self._notify_dispatch_front_capacity(st)
                except Exception as exc:
                    _LOG.debug(
                        "[stats] %s stats poll failed (transient): %s",
                        st.spec.name,
                        exc,
                    )
                    if not st.last_error:
                        st.last_error = f"stats failed: {type(exc).__name__}"
                    self._reset_queue_fields(st)
                    self._mark_resident_baseline_stale(st)

        return self._snapshot_worker_state(st)

    async def _stats_loop(self) -> None:
        """Background GPU monitoring, VRAM enforcement, and signal peak recording.

        Split into two cadences:
        - **Fast path** (every ``stats_interval_s``, default 0.1 s):
          nvidia-smi poll → actual_vram_mb update → activation peak recording
          → OOM enforcement → guard release.  No gRPC calls — uses only
          nvidia-smi data and gateway-local state (``gw_inflight``).
        - **Slow path** (every ``stats_report_interval_s``, default 30 s):
          gRPC worker snapshots (queue stats, crash detection) → UI rendering.
          This is purely for visualization and health monitoring.
        """
        live = None
        console = None
        can_live_render = self.stats_live and bool(
            getattr(sys.stdout, "isatty", lambda: False)()
        )
        if can_live_render:
            try:
                from rich.console import Console
                from rich.live import Live
                from rich.text import Text

                console = Console()
                live = Live(
                    Text("Worker Queues"), console=console, refresh_per_second=4
                )
                live.start()
            except Exception:
                live = None
                console = None

        last_grpc_time: float = 0.0

        while not self._stop.is_set():
            try:
                resource_tracker = getattr(self, "resource_tracker", None)
                if resource_tracker is not None:
                    await resource_tracker._ensure_smi_fresh()
                    procs = await self._get_smi_processes()
                else:
                    procs = {}

                for st in self.states.values():
                    if st.ready and st.addr:
                        _last_cc = getattr(st, "_last_container_check", 0.0)
                        _now_cc = time.time()
                        if (_now_cc - _last_cc) >= 5.0:
                            st._last_container_check = _now_cc
                            _ctr = await asyncio.to_thread(self._get_container, st)
                            if _ctr and _ctr.status in {"exited", "dead"}:
                                _si = _ctr.attrs.get("State", {})
                                _ec = _si.get("ExitCode", -1)
                                _oom = _si.get("OOMKilled", False)
                                _reason = (
                                    f"container {st.container_name} unexpectedly "
                                    f"{_ctr.status} (exit={_ec}{', OOM' if _oom else ''})"
                                )
                                _LOG.error(
                                    "[crash-detect-fast] %s — marking stopped", _reason
                                )
                                if st.addr and getattr(
                                    self, "_invalidate_channel_fn", None
                                ):
                                    with contextlib.suppress(Exception):
                                        self._invalidate_channel_fn(st.addr)
                                _crash_gpus_fast = list(
                                    st.assigned_gpus or st.spec.gpus or []
                                )
                                self._clear_ready(st)
                                st._prev_addr = (
                                    st.addr
                                )
                                st.addr = None
                                st.host_pid = None
                                st.assigned_gpus = []
                                st.lifecycle_state = "stopped"
                                st.last_error = _reason
                                st.queue_gw_inflight = 0
                                self._invalidate_resident_baseline(st)
                                self._reset_queue_fields(st)
                                self._clear_dispatch_front(st)
                                self._notify_dispatch_front_capacity(st)
                                resource_tracker = getattr(
                                    self, "resource_tracker", None
                                )
                                if (
                                    st.memory_reserved_mb > 0
                                    and resource_tracker is not None
                                ):
                                    await resource_tracker.release_weight(
                                        [str(g) for g in _crash_gpus_fast],
                                        st.memory_reserved_mb,
                                    )
                                    st.memory_reserved_mb = 0
                                if resource_tracker is not None:
                                    await resource_tracker.notify_waiters()
                                if self._on_worker_killed and _crash_gpus_fast:
                                    with contextlib.suppress(Exception):
                                        self._on_worker_killed(
                                            st.spec.component, _crash_gpus_fast
                                        )
                                continue

                    if st.host_pid and st.host_pid in procs:
                        st.actual_vram_mb = procs[st.host_pid]
                    else:
                        st.actual_vram_mb = 0

                    if (
                        self._activation_peak_recorder is not None
                        and st.addr
                        and st.queue_gw_inflight >= 1
                        and st.actual_vram_mb > 0
                    ):
                        baseline_weight = st.memory_reserved_mb
                        if baseline_weight <= 0:
                            baseline_weight = self.get_component_weight(
                                st.spec.component
                            )
                        effective_weight = (
                            min(baseline_weight, st.actual_vram_mb)
                            if baseline_weight > 0
                            else 0
                        )
                        if (
                            baseline_weight > 0
                            and st.actual_vram_mb < baseline_weight
                            and st.queue_gw_inflight >= 1
                        ):
                            if not hasattr(st, "_min_actual_during_exec"):
                                st._min_actual_during_exec = st.actual_vram_mb
                            else:
                                st._min_actual_during_exec = min(
                                    st._min_actual_during_exec, st.actual_vram_mb
                                )
                            effective_weight = st._min_actual_during_exec

                        activation_mb = float(
                            max(0, st.actual_vram_mb - effective_weight)
                        )
                        self._activation_peak_recorder(
                            st.addr,
                            activation_mb,
                            exec_inflight=st.queue_gw_inflight,
                        )

                    if st.assigned_gpus and resource_tracker is not None:
                        gid = st.assigned_gpus[0]
                        st.gpu_util_percent = float(
                            resource_tracker._smi_gpu_util.get(gid, 0)
                        )

                    if st.status == "Ready":
                        budget = st.memory_reserved_mb + st.current_activation_mb

                        gid = st.assigned_gpus[0] if st.assigned_gpus else "0"
                        total = (
                            self.resource_tracker.total_vram.get(
                                gid, gpu_vram_fallback_mib()
                            )
                            if self.resource_tracker
                            else gpu_vram_fallback_mib()
                        )
                        saturation_limit = int(total * 0.95)

                        if st.is_guarded or st.activation_guarded_count > 0:
                            kill_limit = max(budget, saturation_limit)
                        else:
                            kill_limit = saturation_limit

                        if st.actual_vram_mb > kill_limit:
                            st.violation_count += 1
                            _LOG.warning(
                                "[guard-STRICT] %s VRAM violation (%d/%d): used %d MB > limit %d MB",
                                st.spec.name,
                                st.violation_count,
                                st.spec.vram_violation_threshold,
                                st.actual_vram_mb,
                                kill_limit,
                            )
                        else:
                            st.violation_count = 0

                        current_known_weight = self.get_component_weight(
                            st.spec.component
                        )

                        if st.is_guarded and current_known_weight > 0:
                            old_reserve = st.memory_reserved_mb
                            st.memory_reserved_mb = current_known_weight
                            st.is_guarded = False
                            _LOG.info(
                                "[guard-RELEASE] %s weight is now known (%d MB). Releasing guard and updating reservation (was %d MB).",
                                st.spec.name,
                                current_known_weight,
                                old_reserve,
                            )
                            if self.resource_tracker and st.assigned_gpus:
                                await self.resource_tracker.update_reservation(
                                    st.assigned_gpus, old_reserve, current_known_weight
                                )
                                await self.resource_tracker.notify_waiters()

                        if st.violation_count >= st.spec.vram_violation_threshold:
                            _LOG.critical(
                                "[guard] KILLING worker %s: continuous VRAM violation. "
                                "Actual=%d MB, Allowed=%d MB. Protecting GPU %s.",
                                st.spec.name,
                                st.actual_vram_mb,
                                kill_limit,
                                st.assigned_gpus,
                            )
                            st.last_error = f"Killed for VRAM violation: {st.actual_vram_mb}MB used (limit {kill_limit}MB)"
                            comp_key = st.spec.component.strip().lower()
                            old_weight = self._component_weight_cache.pop(comp_key, 0)
                            if old_weight > 0:
                                _LOG.warning(
                                    "[guard-invalidate] %s weight cache invalidated (%d MB). "
                                    "Will re-measure on next startup.",
                                    st.spec.name,
                                    old_weight,
                                )
                            if self._on_worker_killed and st.assigned_gpus:
                                try:
                                    self._on_worker_killed(
                                        st.spec.component, list(st.assigned_gpus)
                                    )
                                except Exception:
                                    _LOG.warning(
                                        "[guard] on_worker_killed callback failed for %s",
                                        st.spec.name,
                                        exc_info=True,
                                    )
                            if st.addr and getattr(
                                self, "_invalidate_channel_fn", None
                            ):
                                with contextlib.suppress(Exception):
                                    self._invalidate_channel_fn(st.addr)
                            self._clear_ready(st)
                            st._prev_addr = st.addr
                            st.addr = None

                            async def _guarded_stop(st=st):
                                key = self._state_key(st.spec)
                                lock = self._spawn_locks.setdefault(key, asyncio.Lock())
                                async with lock:
                                    await self._stop_state(st, graceful=False)

                            asyncio.create_task(_guarded_stop())
                    else:
                        st.violation_count = 0

                now = time.time()
                if (now - last_grpc_time) >= self.stats_report_interval_s:
                    last_grpc_time = now

                    async def _safe_update(st):
                        try:
                            return await asyncio.wait_for(
                                self._update_worker_snapshot(st), timeout=3.0
                            )
                        except Exception:
                            return None

                    snapshots_raw = await asyncio.gather(
                        *(_safe_update(st) for st in self.states.values())
                    )
                    snapshots = [s for s in snapshots_raw if s is not None]

                    await self._clear_leaked_host_ram_if_idle(
                        reason="stats-loop-idle-reconcile",
                    )

                    if snapshots:
                        if live:
                            live.update(self._render_queue_stats(snapshots))
                        else:
                            self._print_queue_stats(snapshots)

            except Exception as e:
                _LOG.warning(f"Stats loop error: {e}")

            await asyncio.sleep(self.stats_interval_s)
        if live:
            live.stop()

    async def _get_smi_processes(self) -> dict[int, int]:
        """Returns map of PID -> used_gpu_memory (MB).

        -class anti-pattern fix: previously a sync ``def`` called
        directly from the ``_stats_loop`` coroutine at 10 Hz
        (``stats_interval_s=0.1``) and from ``_ready_worker`` on
        first-ready.  Each ``subprocess.check_output(nvidia-smi)`` blocked
        the main asyncio loop ~50-200 ms — at 10 Hz that translates to
        500-2000 ms/s of blocked event loop, structurally identical to
        the  root cause that was fixed for ``_do_smi_poll`` but
        never propagated here.  Now wraps via ``asyncio.to_thread`` per
        the Step 7 B pattern (see ``_do_smi_poll`` for prior art).
        """
        if _live_smi_poll_disabled():
            return {}

        def _blocking() -> dict[int, int]:
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ]
                out = subprocess.check_output(cmd, encoding="utf-8")
                res: dict[int, int] = {}
                for line in out.strip().split("\n"):
                    if not line or "," not in line:
                        continue
                    p, m = line.split(",")
                    res[int(p.strip())] = int(m.strip())
                return res
            except Exception:
                return {}

        return await asyncio.to_thread(_blocking)

    def _get_container(self, st: WorkerState):
        if not st.container_name:
            return None
        try:
            container = self.dclient.containers.get(st.container_name)
            container.reload()
            return container
        except Exception:
            return None

    def _check_container_running(self, st: WorkerState) -> None:
        container = self._get_container(st)
        if not container:
            return
        if container.status in {"exited", "dead"}:
            try:
                tail = container.logs(tail=50).decode("utf-8", errors="replace")
            except Exception:
                tail = ""
            status = container.attrs.get("State", {}).get("Status", container.status)
            raise RuntimeError(
                f"container {st.container_name} {status}: {tail.strip()}"
            )

    async def start_all(self) -> None:
        await self._cleanup_orphan_containers()

        for st in self.states.values():
            self._clear_ready(st)
            st.status = "Cold"
            st.lifecycle_state = "cold"
            st.addr = None
            st.host_port = None
            st.container_id = None
            st.container_name = self._container_name(st.spec, self.host)
            self._clear_dispatch_front(st)
            st.assigned_gpus = list(st.spec.gpus or [])
            st.host_pid = None
            st.actual_vram_mb = 0
            st.real_dispatch_count_in_generation = 0
            self._reset_queue_fields(st)
        self._print_status()
        print("", flush=True)
        print(
            "Initialized lazy worker inventory (containers are spawned on demand).",
            flush=True,
        )
        asyncio.create_task(self._recovery_loop())

    async def _cleanup_orphan_containers(self) -> None:
        """Remove stale containers from previous gateway instances on startup.

        The supervisor always starts with a lazy, cold inventory.  Therefore
        any already-running managed container for this gateway host is stale
        even when its name matches a worker in the current config.  Leaving a
        same-name container alive would make its RSS look like external host
        memory until that worker is explicitly cold-started again.
        """
        timeout_s = 10.0
        try:
            containers = _call_blocking_with_timeout(
                self.dclient.containers.list,
                timeout_s=timeout_s,
                all=True,
                filters={"label": "bio_gateway_managed=true"},
            )
            expected = {
                self._container_name(st.spec, self.host) for st in self.states.values()
            }
            stale = []
            for c in containers:
                labels = getattr(c, "labels", None)
                if labels is None:
                    labels = getattr(c, "attrs", {}).get("Config", {}).get("Labels", {})
                labels = labels or {}
                container_host = str(labels.get("bio_gateway_host", "") or "")
                if container_host == self.host:
                    stale.append(c)
                    continue
                if not container_host and c.name in expected:
                    stale.append(c)

            if stale:
                _LOG.warning(
                    "[startup] Removing %d stale worker containers for host %s: %s",
                    len(stale),
                    self.host,
                    [c.name for c in stale],
                )
                for c in stale:
                    labels = getattr(c, "labels", None)
                    if labels is None:
                        labels = (
                            getattr(c, "attrs", {}).get("Config", {}).get("Labels", {})
                        )
                    labels = labels or {}
                    mps_root = str(labels.get("bio_gateway_mps_root", "") or "")
                    try:
                        _call_blocking_with_timeout(
                            c.remove,
                            timeout_s=timeout_s,
                            force=True,
                        )
                        _LOG.info("[startup] Removed stale worker container %s", c.name)
                        if mps_root:
                            root = Path(mps_root)
                            if root.name.startswith("proton-gw-mps-"):
                                shutil.rmtree(root, ignore_errors=True)
                    except docker.errors.NotFound:
                        _LOG.info(
                            "[startup] Stale worker container already gone: %s",
                            c.name,
                        )
                    except _BlockingCallTimeout:
                        _LOG.warning(
                            "[startup] Timed out removing stale worker %s",
                            c.name,
                        )
                    except Exception as exc:
                        _LOG.warning(
                            "[startup] Failed to remove stale worker %s: %s",
                            c.name,
                            exc,
                        )
        except _BlockingCallTimeout:
            _LOG.warning(
                "[startup] Stale worker container cleanup timed out after %.1fs",
                timeout_s,
            )
        except Exception as exc:
            _LOG.warning("[startup] Stale worker container cleanup failed: %s", exc)

    async def _recovery_loop(self) -> None:
        """Background task that periodically attempts to restart evicted workers if VRAM is available."""
        while not self._stop.is_set():
            async with self.resource_tracker._cv:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self.resource_tracker._cv.wait(), timeout=10.0
                    )

            await self._attempt_worker_recovery()

    async def _attempt_worker_recovery(self) -> None:
        """Finds stopped/errored workers and restarts them if enough VRAM is available."""
        stopped_states = [
            st
            for st in self.states.values()
            if st.lifecycle_state in {"stopped", "error"}
            and not getattr(st, "recovery_suppressed", False)
        ]
        if not stopped_states:
            return

        if self._recovery_handler:
            to_recover = self._recovery_handler(stopped_states, self.resource_tracker)
        else:
            _LOG.debug(
                "[recovery] planner recovery handler unavailable; skipping "
                "%d stopped/error workers",
                len(stopped_states),
            )
            return

        for st in to_recover:
            if self._stop.is_set():
                break

            lock = self._spawn_locks.get(self._state_key(st.spec))
            if lock and lock.locked():
                continue

            label = f"{st.spec.component}/{st.spec.name}"
            print(f"[recovery] Restarting evicted worker {label}", flush=True)
            _task = asyncio.create_task(self._ensure_state_ready(st))
            _task.add_done_callback(_log_background_task_failure)

    async def _ready_worker(self, st: WorkerState) -> None:
        if not st.addr:
            st.status = "NoAddr"
            st.lifecycle_state = "error"
            st.last_error = "missing address"
            return
        try:
            st.status = "WaitingAlive"
            st.lifecycle_state = "starting"
            deadline = time.time() + self.readiness_timeout_s
            while time.time() < deadline:
                self._check_container_running(st)
                try:
                    await wait_alive(st.addr, timeout_s=10.0)
                    break
                except Exception:
                    await asyncio.sleep(0.2)
            else:
                raise TimeoutError(f"worker not alive: {st.addr}")

            st.status = "WaitingReady"
            st.lifecycle_state = "starting"
            deadline = time.time() + self.readiness_timeout_s
            while time.time() < deadline:
                self._check_container_running(st)
                try:
                    await wait_ready(st.addr, timeout_s=30.0)
                    break
                except Exception:
                    await asyncio.sleep(0.2)
            else:
                raise TimeoutError(f"worker not ready: {st.addr}")

            st.status = "GettingCaps"
            st.caps = await get_caps(st.addr)
            try:
                stats = await get_stats(st.addr)
                self._apply_worker_stats(st, stats)
            except Exception:
                self._reset_queue_fields(st)

            await self.resource_tracker.mark_active(
                st.assigned_gpus, st.memory_reserved_mb
            )
            st.last_error = ""
            label = f"{st.spec.component}/{st.spec.name}"

            caps_name = st.caps.model_name if st.caps else "unknown"
            caps_ver = st.caps.model_version if st.caps else "unknown"
            print(
                f"[ready] {label} at {st.addr} model={caps_name} ver={caps_ver}",
                flush=True,
            )

            self._set_ready(st)
            st.status = "Ready"
            st.lifecycle_state = "ready"
            ready_at = time.time()
            st.last_ready_at = ready_at
            if int(getattr(st, "dispatch_pending", 0) or 0) > 0:
                st.dispatch_pending_updated_at = ready_at
                st.dispatch_pending_handoff_protected = True
                st.dispatch_pending_handoff_started_at = ready_at
                st._front_seen_execute_inflight = int(
                    getattr(st, "queue_execute_inflight", 0) or 0,
                )
            self._notify_dispatch_front_capacity(st)

            if st.is_guarded:
                procs = await self._get_smi_processes()
                measured = procs.get(st.host_pid, 0) if st.host_pid else 0
                old_reserve = st.memory_reserved_mb
                st.memory_reserved_mb = measured
                st.is_guarded = False
                _LOG.info(
                    "[guard-MEASURE] %s measured weight at ready: %d MB. Releasing guard (was %d MB).",
                    st.spec.name,
                    measured,
                    old_reserve,
                )
                self._update_weight_cache_direct(st.spec.component, max(measured, 0))
                if measured > 0:
                    await self.update_component_weight(st.spec.component, measured)
                if self.resource_tracker and st.assigned_gpus:
                    await self.resource_tracker.update_reservation(
                        st.assigned_gpus, old_reserve, measured
                    )
                    await self.resource_tracker.notify_waiters()
        except Exception as exc:
            self._clear_ready(st)
            st.status = "Error"
            st.lifecycle_state = "error"
            st.last_error = str(exc)
            self._clear_dispatch_front(st)
            self._notify_dispatch_front_capacity(st)
            label = f"{st.spec.component}/{st.spec.name}"
            print(f"[error] {label} {exc}", flush=True)

    async def _start_one(self, spec: WorkerSpec) -> None:
        name = spec.name
        container_name = self._container_name(spec, self.host)
        st = self._ensure_state(spec)

        with contextlib.suppress(Exception):
            old = await asyncio.to_thread(self.dclient.containers.get, container_name)
            await asyncio.to_thread(old.remove, force=True)

        host_port = spec.grpc_port if spec.grpc_port else pick_free_port(self.host)

        env = dict(spec.env)

        ports = {f"{host_port}/tcp": (self.host, host_port)}

        device_requests = []
        if st.assigned_gpus:
            device_requests = [docker_device_request(st.assigned_gpus)]
        elif spec.gpus:
            device_requests = [docker_device_request(spec.gpus)]
            st.assigned_gpus = spec.gpus
        elif spec.gpu_count > 0 and self.gpu_policy:
            try:
                pending_usage = self._get_pending_gpu_usage()
                allocated = self.gpu_policy.allocate(
                    spec.gpu_count, pending_usage=pending_usage
                )
                device_requests = [docker_device_request(allocated)]
                st.assigned_gpus = allocated

                label = f"{spec.component}/{name}"
                print(
                    f"[allocated] {label} gpus={allocated} (pending_map={pending_usage})",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[error] {spec.component}/{name} gpu allocation failed: {e}",
                    flush=True,
                )
                raise
        else:
            st.assigned_gpus = []

        self._prepare_state_for_start(st, container_name)

        volumes = dict(spec.volumes or {})

        await self.resource_tracker.ensure_initialized()

        configured = int(
            self.profiling_runtime.get("profile_max_inflight_per_worker", 0)
        )
        st.max_concurrency = configured

        process_backend = bool(device_requests and worker_process_backend_enabled(spec))
        mps_enabled = bool(device_requests and worker_cuda_mps_enabled(spec))
        if mps_enabled:
            mps = None
            for gpu_id in st.assigned_gpus:
                lock = self._spawn_locks.setdefault(
                    ("mps-daemon", str(gpu_id)), asyncio.Lock()
                )
                async with lock:
                    daemon = await self._ensure_mps_daemon(gpu_id, spec.image)
                if mps is None:
                    mps = daemon
            if mps is not None:
                env.update(
                    {
                        "CUDA_MPS_PIPE_DIRECTORY": str(mps.pipe_dir),
                        "CUDA_MPS_LOG_DIRECTORY": str(mps.log_dir),
                    }
                )
                volumes[str(mps.pipe_dir)] = self._volume_bind(mps.pipe_dir)
                volumes[str(mps.log_dir)] = self._volume_bind(mps.log_dir)

        command = build_worker_command(
            spec, host_port, max_concurrency=st.max_concurrency
        )

        run_kwargs = {
            "image": spec.image,
            "name": container_name,
            "detach": True,
            "remove": False,
            "environment": env,
            "ports": ports,
            "device_requests": device_requests,
            "command": command,
            "volumes": volumes,
            "working_dir": "/workspace",
            "shm_size": spec.shm_size,
            "labels": self._container_labels(spec, self.host),
            "oom_score_adj": spec.oom_score_adj,
        }
        if device_requests:
            run_kwargs["devices"] = docker_explicit_nvidia_devices(st.assigned_gpus)
        if process_backend:
            run_kwargs["cap_add"] = ["SYS_PTRACE"]

        container = await asyncio.to_thread(self.dclient.containers.run, **run_kwargs)
        st.container_id = container.id
        if not spec.grpc_port:
            release_port(host_port)

        await asyncio.to_thread(container.reload)
        st.host_pid = container.attrs.get("State", {}).get("Pid")

        _LOG.info(
            "[start] %s (id=%s, pid=%s, GPUs=%s)",
            spec.name,
            container.id[:12],
            st.host_pid,
            spec.gpus,
        )

        st.host_port = host_port
        st.addr = f"{self.host}:{host_port}"
        st._prev_addr = None
        self._clear_ready(st)
        st.status = "Started"
        st.lifecycle_state = "starting"

        label = f"{spec.component}/{name}"
        print(
            f"[start] {label} image={spec.image} addr={st.addr} gpus={st.assigned_gpus or spec.gpus} concurrency={st.max_concurrency}",
            flush=True,
        )
        if self.stream_logs_enabled:
            try:
                container = self.dclient.containers.get(container_name)
                prefix = f"{st.spec.component}/{st.spec.name}"
                st.log_task = asyncio.create_task(stream_logs(container, prefix=prefix))
            except Exception as exc:
                st.last_error = f"log stream failed: {exc}"
                st.status = "LogError"

    async def stop_all(self) -> None:
        if self._stats_task:
            self._stats_task.cancel()

        await asyncio.gather(
            *(self._stop_state(st, graceful=False) for st in self.states.values())
        )
        await self._stop_mps_daemons()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()

        def _sigint() -> None:
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _sigint)

        await self.start_all()
        if self.stats_interval_s > 0:
            self._stats_task = asyncio.create_task(self._stats_loop())

        recovery_task = asyncio.create_task(self._recovery_loop())

        await self._stop.wait()
        recovery_task.cancel()
        print("", flush=True)
        print("Stopping all workers.", flush=True)
        await self.stop_all()
        self._print_status()


def _parse_components(raw_components: list[str]) -> list[str]:
    parts: list[str] = []
    for raw in raw_components:
        for part in str(raw).split(","):
            cleaned = part.strip().lower()
            if cleaned:
                parts.append(cleaned)
    return parts


async def main_async(
    runtime_path: str,
    worker_path: str,
    components: list[str],
    host: str,
    stream_logs: bool,
    readiness_timeout_s: float,
    stats_interval_s: float,
) -> None:
    specs, gpu_allocation, profiling_runtime = load_specs(
        runtime_path, worker_path, components
    )
    if not specs:
        comps = ", ".join(components)
        raise SystemExit(
            f"No workers defined for component(s) [{comps}] in {worker_path} (via {runtime_path})"
        )
    cfg = SupervisorConfig(
        specs=specs,
        host=host,
        readiness_timeout_s=readiness_timeout_s,
        stream_logs=stream_logs,
        stats_interval_s=stats_interval_s,
        gpu_allocation=gpu_allocation,
        profiling_runtime=profiling_runtime,
    )
    sup = WorkerSupervisor(cfg)
    await sup.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        "--config",
        type=str,
        default="configs/runtime.proton.yaml",
        help="Path to runtime config (default: configs/runtime.proton.yaml)",
        dest="runtime_config",
    )
    parser.add_argument(
        "--worker-config",
        type=str,
        default="configs/workers.yaml",
        help="Path to worker config (default: configs/workers.yaml)",
        dest="worker_config",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--components",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Component name(s) to start (default: rfdiffusion). "
            "Supports multiple values or comma-separated lists."
        ),
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Start all components defined in the config",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for published ports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for readiness per worker",
    )
    parser.add_argument(
        "--no-stream-logs",
        action="store_true",
        help="Disable log streaming",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Seconds between queue stats snapshots (default: 5)",
    )
    args = parser.parse_args()

    if args.all:
        components = discover_components(args.worker_config)
    else:
        raw_components = (
            args.components if args.components is not None else ["rfdiffusion"]
        )
        components = _parse_components(raw_components)
    if not components:
        raise SystemExit(f"No components found in config: {args.config}")
    asyncio.run(
        main_async(
            args.runtime_config,
            args.worker_config,
            components,
            args.host,
            not args.no_stream_logs,
            args.readiness_timeout,
            args.stats_interval,
        )
    )


if __name__ == "__main__":
    main()
