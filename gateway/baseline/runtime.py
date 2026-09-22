"""Baseline gateway runtime — minimal alternative to
``gateway.http_server.GatewayHTTPService`` for FIFO / K8s / Slurm
schedulers.

Surface:
  * Reuses ``WorkerSupervisor`` for worker process lifecycle + gRPC
    channel pool + workers.yaml.
  * Exposes the same Nextflow-facing HTTP API (``POST /api/v1/job/
    submit``, ``GET /api/v1/job/{job_id}``, ``GET /health``) so the
    nf-gw plugin can submit identically to proton.
  * Owns one ``BaselineSchedulerProtocol`` instance and runs a periodic
    ``step(now)`` → dispatch loop.
  * No SignalService at scheduling time (baselines never read GP).
  * No SchedulingScenario / ConstraintTracker / RealityValidator.
  * Failure handling = simple retry (re-submit to the scheduler queue).

Architecturally the baseline path is roughly 200-300 LOC of glue,
versus proton's full 3-layer ~340 LOC of wiring + ~5000 LOC of
http_server.  The trade-off is intentional: baselines are scheduler-
algorithm-only ports.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import grpc
import yaml
from aiohttp import web

from modelworker import modelworker_pb2 as pb
from modelworker import modelworker_pb2_grpc as pb_grpc

from gateway.baseline.shared import (
    BaselineSchedulerProtocol,
    DispatchDecision,
    TaskSubmission,
    WorkersProfile,
)

_LOG = logging.getLogger("gateway.baseline.runtime")

_SCHEDULER_TICK_SEC = 0.5


def _read_mem_total_mib() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0



_STATE_SUBMITTED = "SUBMITTED"
_STATE_RUNNING = "RUNNING"
_STATE_SUCCEEDED = "SUCCEEDED"
_STATE_FAILED = "FAILED"

_WORKDIR_HASH_RE = re.compile(r"/([0-9a-f]{2})/([0-9a-f]{6,})$", re.IGNORECASE)


@dataclass
class _BaselineTaskRecord:
    """In-memory task record for the baseline gateway.  Mirrors the
    proton ``TaskRecord`` subset that Nextflow needs to poll status.
    """

    task_id: str
    nf_task_id: str
    component: str
    submission: TaskSubmission
    state: str = _STATE_SUBMITTED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    target_gpu_id: str = ""
    target_worker_name: str = ""
    target_worker_addr: str = ""
    message: str = ""
    result_payload: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    workdir: str = ""
    timeout_s: int = 300
    env: Dict[str, str] = field(default_factory=dict)
    config_fingerprint: str = ""
    mode: str = "nextflow_task"
    argv: List[str] = field(default_factory=list)
    tool_cwd: str = ""



class BaselineGatewayRuntime:
    """Minimal gateway runtime for FIFO / K8s / Slurm baseline schedulers.

    Lifecycle:
      __init__   — load runtime config, instantiate scheduler + supervisor.
      run()      — start HTTP server + scheduler tick loop.

    HTTP API (subset of proton's GatewayHTTPService):
      POST /api/v1/job/submit
      GET  /api/v1/job/{task_id}
      GET  /health
      GET  /api/v1/profile/export   (read-only signals snapshot — proton-only;
                                     baseline returns empty stub)
    """

    _GRPC_CHANNEL_OPTIONS = [
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 2**31 - 1),
        ("grpc.keepalive_permit_without_calls", 0),
    ]

    def __init__(
        self,
        scheduler: BaselineSchedulerProtocol,
        supervisor: Any,
        gpu_capacity_mb: Dict[str, int],
        host: str = "127.0.0.1",
        port: int = 8098,
        worker_name_format: str = "{component}-gpu{gpu_id}",
        baseline_idle_eviction: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._scheduler = scheduler
        self._supervisor = supervisor
        self._gpu_capacity_mb = gpu_capacity_mb
        self._host = host
        self._port = port
        self._worker_name_format = worker_name_format
        self._tasks: Dict[str, _BaselineTaskRecord] = {}
        self._record_by_nf_workdir: Dict[Tuple[str, str], str] = {}
        self._tasks_lock = asyncio.Lock()
        self._dispatch_executors: Dict[str, asyncio.Task] = {}
        self._campaign_completions: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._channel_pool: Dict[str, grpc.aio.Channel] = {}
        self._batch_id_seq = itertools.count(1)
        cfg = dict(baseline_idle_eviction or {})
        self._baseline_idle_eviction_enabled = bool(cfg.get("enabled", False))
        self._baseline_idle_eviction_mode = str(
            cfg.get("mode", "pressure_only")
        ).strip().lower()
        self._gpu_alloc_export_path = self._resolve_gpu_alloc_export_path()
        self._gpu_alloc_write_lock = asyncio.Lock()

    def _get_channel(self, addr: str) -> grpc.aio.Channel:
        """Get or create a cached gRPC channel for the worker addr."""
        ch = self._channel_pool.get(addr)
        if ch is None:
            ch = grpc.aio.insecure_channel(addr, options=self._GRPC_CHANNEL_OPTIONS)
            self._channel_pool[addr] = ch
        return ch

    def _resolve_worker_addr(self, worker_name: str) -> Optional[str]:
        """Look up the gRPC addr for a worker by its supervisor-managed
        name.  Returns None if not ready.
        """
        try:
            states = getattr(self._supervisor, "states", None) or {}
            st = states.get(worker_name)
            if st is None:
                return None
            addr = getattr(st, "addr", "") or ""
            return addr if addr else None
        except Exception:
            return None

    def _resolve_gpu_alloc_export_path(self) -> Optional[Path]:
        raw = str(os.environ.get("PROTON_BASELINE_GPU_ALLOC_PATH", "") or "").strip()
        if raw:
            return Path(raw)
        gp_dump_path = str(os.environ.get("PROTON_GP_DUMP_PATH", "") or "").strip()
        if gp_dump_path:
            gp_path = Path(gp_dump_path)
            if gp_path.parent:
                return gp_path.parent / "gpu_alloc.jsonl"
        return None

    @staticmethod
    def _extract_task_hash(workdir: str) -> str:
        wd = str(workdir or "").strip().rstrip("/")
        if not wd:
            return ""
        match = _WORKDIR_HASH_RE.search(wd)
        if match:
            return f"{match.group(1).lower()}/{match.group(2)[:6].lower()}"
        path = Path(wd)
        try:
            prefix = path.parent.name.strip().lower()
            suffix = path.name.strip().lower()
        except Exception:
            return ""
        if re.fullmatch(r"[0-9a-f]{2}", prefix) and re.fullmatch(r"[0-9a-f]{6,}", suffix):
            return f"{prefix}/{suffix[:6]}"
        return ""

    async def _export_gpu_allocation_truth(
        self,
        rec: _BaselineTaskRecord,
        decision: DispatchDecision,
        *,
        event: str = "dispatch_start",
    ) -> None:
        path = self._gpu_alloc_export_path
        if path is None:
            return
        scheduler_name = str(getattr(self._scheduler, "name", "") or "").strip().lower()
        if scheduler_name not in {"k8s", "slurm"}:
            return
        task_hash = self._extract_task_hash(rec.workdir)
        row = {
            "ts": time.time(),
            "event": event,
            "scheduler": f"proton-{scheduler_name}",
            "gateway_task_id": rec.task_id,
            "nf_task_id": rec.nf_task_id,
            "component": rec.component,
            "gpu_id": str(decision.target_gpu_id or ""),
            "task_hash": task_hash,
            "workdir": str(rec.workdir or ""),
        }
        if scheduler_name == "k8s":
            row["pod_name"] = rec.nf_task_id or rec.task_id
            row["gpu_mem_idx"] = str(decision.target_gpu_id or "")
            row["env_idx"] = str(decision.target_gpu_id or "")
            row["env_visible"] = str(decision.target_gpu_id or "")
        else:
            row["jobid"] = rec.nf_task_id or rec.task_id
            row["gres"] = f"gpu:1(IDX:{decision.target_gpu_id})"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, separators=(",", ":")) + "\n"
            async with self._gpu_alloc_write_lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:
            _LOG.warning(
                "[baseline] failed to export gpu allocation truth to %s",
                path,
                exc_info=True,
            )


    async def health(self, request: web.Request) -> web.Response:
        _ = request
        return web.json_response({
            "status": "ok",
            "scheduler": self._scheduler.name,
            "task_count": len(self._tasks),
        })

    async def get_workers(self, request: web.Request) -> web.Response:
        """GET /api/v1/workers - baseline worker lifecycle snapshot.

        Baseline runtimes intentionally do not use PROTON's full
        GatewayHTTPService, but the ops console and live debugging still need
        the same worker queue view. Keep this bounded and read-only.
        """
        _ = request
        workers: List[Dict[str, Any]] = []
        try:
            list_workers = getattr(self._supervisor, "list_workers", None)
            states = (
                list_workers(include_not_ready=True)
                if callable(list_workers)
                else list(getattr(self._supervisor, "states", {}).values())
            )
            for st in states:
                spec = getattr(st, "spec", None)
                workers.append({
                    "component": str(getattr(spec, "component", "") or ""),
                    "name": str(getattr(spec, "name", "") or ""),
                    "state": str(getattr(st, "lifecycle_state", "") or getattr(st, "status", "") or ""),
                    "ready": bool(getattr(st, "ready", False)),
                    "addr": str(getattr(st, "addr", "") or ""),
                    "assigned_gpus": list(getattr(st, "assigned_gpus", []) or []),
                    "dispatch_pending": int(getattr(st, "dispatch_pending", 0) or 0),
                    "queue_gw_inflight": int(getattr(st, "queue_gw_inflight", 0) or 0),
                    "queue_prepare_inflight": int(getattr(st, "queue_prepare_inflight", 0) or 0),
                    "queue_execute_inflight": int(getattr(st, "queue_execute_inflight", 0) or 0),
                    "queue_finalize_inflight": int(getattr(st, "queue_finalize_inflight", 0) or 0),
                    "queue_in_queue": int(getattr(st, "queue_in_queue", 0) or 0),
                    "queue_prepared_queue": int(getattr(st, "queue_prepared_queue", 0) or 0),
                    "queue_finalize_queue": int(getattr(st, "queue_finalize_queue", 0) or 0),
                    "draining": bool(getattr(st, "draining", False)),
                    "container_id": str(getattr(st, "container_id", "") or ""),
                    "last_error": str(getattr(st, "last_error", "") or ""),
                })
        except Exception as exc:
            _LOG.warning("[baseline] worker snapshot failed: %s", exc, exc_info=True)
            return web.json_response({"error": "worker snapshot failed"}, status=500)
        return web.json_response({"workers": workers, "count": len(workers)})

    async def get_debug_tasks(self, request: web.Request) -> web.Response:
        """GET /api/v1/debug/tasks - inspect baseline in-memory state.

        This mirrors the minimal observability needed to distinguish true
        worker activity from scheduler-side pre-dispatch backlog.
        """
        _ = request
        async with self._tasks_lock:
            records = list(self._tasks.values())
        by_state = collections.Counter(rec.state for rec in records)
        by_component_state = collections.Counter(
            (rec.component, rec.state) for rec in records
        )
        executors = {
            "total": len(self._dispatch_executors),
            "pending": sum(1 for t in self._dispatch_executors.values() if not t.done()),
            "done": sum(1 for t in self._dispatch_executors.values() if t.done()),
            "cancelled": sum(1 for t in self._dispatch_executors.values() if t.cancelled()),
        }
        scheduler_debug = None
        snapshot = getattr(self._scheduler, "debug_snapshot", None)
        if callable(snapshot):
            try:
                scheduler_debug = snapshot()
            except Exception as exc:
                scheduler_debug = {"error": str(exc)}
        sample = [
            {
                "task_id": rec.task_id,
                "nf_task_id": rec.nf_task_id,
                "component": rec.component,
                "state": rec.state,
                "target_gpu_id": rec.target_gpu_id,
                "target_worker_name": rec.target_worker_name,
                "message": rec.message,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
            }
            for rec in records[-200:]
        ]
        return web.json_response({
            "task_count": len(records),
            "by_state": dict(by_state),
            "by_component_state": {
                f"{component}:{state}": count
                for (component, state), count in sorted(by_component_state.items())
            },
            "dispatch_executors": executors,
            "scheduler": scheduler_debug,
            "sample_recent": sample,
            "campaign_completions": dict(self._campaign_completions),
        })

    async def mark_campaign_complete(self, request: web.Request) -> web.Response:
        """POST /api/v1/campaign/complete - accept NF terminal signal.

        Baseline schedulers do not use campaign-primary handoff semantics, but
        the shared Nextflow runner sends the same terminal signal for all
        gateway-backed modes.  Returning 200 here keeps baseline benchmark logs
        clean and gives debug snapshots a compact record of NF-side completion.
        """
        try:
            body = await request.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        run_name = str(body.get("run_name", "") or "")
        status = str(body.get("status", "") or "")
        ok = bool(body.get("ok", status == "succeeded"))
        message = str(body.get("message", "") or "")
        if run_name:
            self._campaign_completions[run_name] = {
                "ok": ok,
                "status": status,
                "message": message,
                "updated_at": time.time(),
            }
        _LOG.info(
            "[baseline] campaign_complete run_name=%s ok=%s status=%s",
            run_name,
            ok,
            status,
        )
        return web.json_response({"ok": True})

    async def submit_job(self, request: web.Request) -> web.Response:
        """Accept a job submission from Nextflow.

        Schema (subset; baselines ignore proton-specific fields):
          {
            "task_id": str,           # nf-side id
            "component": str,
            "input_size": float,
            "workload_features": dict,
            ...
          }
        Returns: {"task_id": <gateway-side id>, "accepted": True}.
        """
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": f"invalid JSON: {e}"}, status=400,
            )

        if isinstance(body, dict) and "kind" in body and "payload" in body:
            payload = body.get("payload") or {}
            if not isinstance(payload, dict):
                return web.json_response(
                    {"error": "payload must be an object"}, status=400,
                )
        else:
            payload = body

        nf_task_id = str(payload.get("task_id") or payload.get("nf_task_id") or "")
        component = str(payload.get("component") or "").strip().lower()
        if not component:
            return web.json_response(
                {"error": "component required"}, status=400,
            )
        input_size = float(payload.get("input_size") or 0.0)
        gw_task_id = str(uuid.uuid4())
        now = time.time()
        workload_features = payload.get("workload_features") or {}
        if input_size <= 0.0 and isinstance(workload_features, dict):
            for key in ("scaffold_length", "num_residues", "receptor_atoms",
                        "input_size", "sequence_length"):
                if key in workload_features:
                    try:
                        v = float(workload_features[key])
                        if v > 0:
                            input_size = v
                            break
                    except (TypeError, ValueError):
                        pass
        env_dict = {
            str(k): str(v) for k, v in (payload.get("env") or {}).items()
            if isinstance(v, (str, int, float))
        }
        raw_account = (
            payload.get("account")
            or env_dict.get("SBATCH_ACCOUNT")
            or ""
        )
        account = str(raw_account or "").strip()
        try:
            fairshare = float(payload.get("fairshare") or 0.0)
        except (TypeError, ValueError):
            fairshare = 0.0
        if account and fairshare == 0.0:
            fairshare = {
                "c1": 800.0, "c2": 700.0, "c3": 600.0, "c4": 500.0,
                "c5": 400.0, "c6": 300.0, "c7": 200.0, "c8": 100.0,
            }.get(account, 0.0)
        sub = TaskSubmission(
            task_id=gw_task_id,
            component=component,
            input_size=input_size,
            submit_time=now,
            arrival_time=now,
            workload_features=workload_features,
            account=account,
            fairshare=fairshare,
        )
        argv_value = payload.get("argv")
        argv_list: List[str] = []
        if isinstance(argv_value, list):
            argv_list = [str(x) for x in argv_value]
        workdir = str(payload.get("workdir") or "")
        replay_key = (nf_task_id, str(Path(workdir).resolve())) if nf_task_id and workdir else None
        if replay_key is not None:
            async with self._tasks_lock:
                prev_id = self._record_by_nf_workdir.get(replay_key)
                prev_record = self._tasks.get(prev_id or "")
            if prev_record is not None:
                _LOG.info(
                    "[baseline] idempotent submit replay task=%s nf=%s component=%s",
                    prev_record.task_id[:8],
                    nf_task_id[:16],
                    component,
                )
                return web.json_response({
                    "job_id": prev_record.task_id,
                    "kind": "task",
                    "state": prev_record.state,
                    "task_id": prev_record.task_id,
                    "accepted": True,
                })
        record = _BaselineTaskRecord(
            task_id=gw_task_id,
            nf_task_id=nf_task_id,
            component=component,
            submission=sub,
            workdir=workdir,
            timeout_s=int(payload.get("timeout_s") or 300),
            env=env_dict,
            config_fingerprint=str(payload.get("config_fingerprint") or ""),
            mode=str(payload.get("mode") or "nextflow_task"),
            argv=argv_list,
            tool_cwd=str(payload.get("tool_cwd") or ""),
        )
        async with self._tasks_lock:
            self._tasks[gw_task_id] = record
            if replay_key is not None:
                self._record_by_nf_workdir[replay_key] = gw_task_id
        self._scheduler.submit(sub)
        _LOG.info(
            "[baseline] submit task=%s nf=%s component=%s scheduler=%s",
            gw_task_id[:8], nf_task_id[:16], component, self._scheduler.name,
        )
        return web.json_response({
            "job_id": gw_task_id,
            "kind": "task",
            "state": "PENDING",
            "task_id": gw_task_id,
            "accepted": True,
        })

    async def get_job(self, request: web.Request) -> web.Response:
        task_id = request.match_info["job_id"]
        async with self._tasks_lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return web.json_response(
                    {"error": "task not found"}, status=404,
                )
            ok = (rec.state == _STATE_SUCCEEDED)
            error = rec.message if rec.state == _STATE_FAILED else ""
            result_payload: Dict[str, Any] = dict(rec.result_payload or {})
            if rec.message and "message" not in result_payload:
                result_payload["message"] = rec.message
            return web.json_response({
                "job_id": rec.task_id,
                "task_id": rec.task_id,
                "nf_task_id": rec.nf_task_id,
                "component": rec.component,
                "state": rec.state,
                "ok": ok,
                "error": error,
                "result": result_payload,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "target_gpu_id": rec.target_gpu_id,
                "target_worker_name": rec.target_worker_name,
                "message": rec.message,
            })

    async def get_profile_export(self, request: web.Request) -> web.Response:
        """Stub for baseline (proton-only feature).  Returns empty map
        so tools/export_workers_profile.py can detect "no data" cleanly.
        """
        _ = request
        return web.json_response({
            "snapshot_at": time.time(),
            "components": {},
            "note": (
                "proton scheduler not active (baseline runtime); GP not "
                "accumulated. Run a proton bench to populate this endpoint."
            ),
        })


    async def _scheduler_tick(self) -> None:
        """Periodic step() → dispatch loop."""
        while self._running:
            try:
                now = time.time()
                decisions = self._scheduler.step(now)
            except Exception as e:
                _LOG.error("[baseline] scheduler.step raised: %s", e, exc_info=True)
                decisions = []
            if not decisions:
                try:
                    decisions = await self._maybe_evict_blocked_head_and_retry(now)
                except Exception as e:
                    _LOG.warning(
                        "[baseline] idle-eviction retry failed: %s",
                        e,
                        exc_info=True,
                    )
            for dec in decisions:
                if not dec.is_dispatchable:
                    continue
                await self._launch_dispatch(dec)
            await asyncio.sleep(_SCHEDULER_TICK_SEC)

    async def _maybe_evict_blocked_head_and_retry(
        self, now: float,
    ) -> List[DispatchDecision]:
        if not self._baseline_idle_eviction_enabled:
            return []
        if self._baseline_idle_eviction_mode != "pressure_only":
            return []
        describe = getattr(self._scheduler, "describe_head_blockage", None)
        if not callable(describe):
            return []
        blockage = describe(now)
        if not isinstance(blockage, dict):
            return []
        try:
            host_ram_shortfall = int(blockage.get("host_ram_shortfall_mb") or 0)
        except (TypeError, ValueError):
            host_ram_shortfall = 0
        if host_ram_shortfall > 0:
            estimate_host_freeable = getattr(
                self._supervisor, "estimate_idle_host_ram_freeable", None,
            )
            evict_host_idle = getattr(
                self._supervisor, "evict_idle_workers_for_host_ram", None,
            )
            if callable(estimate_host_freeable) and callable(evict_host_idle):
                idle_freeable = int(
                    estimate_host_freeable(ignore_grace=True) or 0
                )
                if idle_freeable >= host_ram_shortfall:
                    evicted_mb = int(
                        await evict_host_idle(
                            host_ram_shortfall,
                            suppress_recovery=True,
                            ignore_grace=True,
                        )
                        or 0
                    )
                    if evicted_mb > 0:
                        _LOG.info(
                            "[baseline] idle host-RAM eviction freed %d MiB for blocked head task=%s component=%s",
                            evicted_mb,
                            blockage.get("task_id", ""),
                            blockage.get("component", ""),
                        )
                        return self._scheduler.step(now)
        shortfalls = blockage.get("gpu_shortfalls_mb") or {}
        if not isinstance(shortfalls, dict) or not shortfalls:
            return []

        candidate_gpus: List[Tuple[int, str]] = []
        for gpu_id, shortfall in shortfalls.items():
            try:
                shortfall_mb = int(shortfall)
            except (TypeError, ValueError):
                continue
            if shortfall_mb <= 0:
                continue
            candidate_gpus.append((shortfall_mb, str(gpu_id)))
        candidate_gpus.sort(key=lambda item: (item[0], int(item[1])))
        if not candidate_gpus:
            return []

        estimate_freeable = getattr(self._supervisor, "estimate_idle_freeable", None)
        evict_idle_workers = getattr(self._supervisor, "evict_idle_workers", None)
        if not callable(estimate_freeable) or not callable(evict_idle_workers):
            return []

        for shortfall_mb, gpu_id in candidate_gpus:
            idle_freeable = int(
                estimate_freeable(gpu_id, ignore_grace=True) or 0
            )
            if idle_freeable < shortfall_mb:
                continue
            evicted_mb = int(
                await evict_idle_workers(
                    gpu_id,
                    shortfall_mb,
                    suppress_recovery=True,
                    ignore_grace=True,
                )
                or 0
            )
            if evicted_mb <= 0:
                continue
            _LOG.info(
                "[baseline] idle eviction freed %d MB on gpu=%s for blocked head task=%s component=%s",
                evicted_mb,
                gpu_id,
                blockage.get("task_id", ""),
                blockage.get("component", ""),
            )
            return self._scheduler.step(now)
        return []

    async def _launch_dispatch(self, decision: DispatchDecision) -> None:
        rec: Optional[_BaselineTaskRecord] = None
        async with self._tasks_lock:
            rec = self._tasks.get(decision.task_id)
            if rec is None:
                _LOG.warning(
                    "[baseline] dispatch decision for unknown task %s — drop",
                    decision.task_id[:8],
                )
                return
            if rec.state != _STATE_SUBMITTED:
                return

        bump_dispatch_pending = getattr(self._supervisor, "bump_dispatch_pending", None)
        if callable(bump_dispatch_pending):
            try:
                front_ok = bump_dispatch_pending(decision.target_worker_name)
            except Exception:
                _LOG.warning(
                    "[baseline] bump_dispatch_pending failed for %s",
                    decision.target_worker_name,
                    exc_info=True,
                )
                front_ok = False
            if front_ok is False:
                now = time.time()
                self._scheduler.on_dispatch_failure(
                    decision.task_id, "worker_front_saturated", now,
                )
                self._scheduler.submit(rec.submission)
                rec.retry_count += 1
                rec.message = "worker_front_saturated"
                rec.updated_at = now
                _LOG.debug(
                    "[baseline] deferred dispatch task=%s worker=%s reason=worker_front_saturated",
                    decision.task_id[:8],
                    decision.target_worker_name,
                )
                return

        async with self._tasks_lock:
            rec = self._tasks.get(decision.task_id)
            if rec is None or rec.state != _STATE_SUBMITTED:
                release_dispatch_pending = getattr(
                    self._supervisor, "release_dispatch_pending", None,
                )
                if callable(release_dispatch_pending):
                    try:
                        release_dispatch_pending(decision.target_worker_name)
                    except Exception:
                        _LOG.warning(
                            "[baseline] release_dispatch_pending failed for %s",
                            decision.target_worker_name,
                            exc_info=True,
                        )
                return
            rec.target_gpu_id = decision.target_gpu_id
            rec.target_worker_name = decision.target_worker_name
            rec.state = _STATE_RUNNING
            rec.updated_at = time.time()
        if hasattr(self._scheduler, "remember_dispatched"):
            try:
                self._scheduler.remember_dispatched(rec.submission)
            except Exception:
                pass
        task = asyncio.create_task(self._dispatch_one(rec, decision))
        self._dispatch_executors[rec.task_id] = task
        task.add_done_callback(lambda _task, task_id=rec.task_id: self._on_dispatch_task_done(task_id, _task))

    def _on_dispatch_task_done(self, task_id: str, task: asyncio.Task) -> None:
        self._dispatch_executors.pop(task_id, None)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception as err:
            _LOG.warning(
                "[baseline] dispatch executor inspection failed task=%s: %s",
                task_id[:8], err,
            )
            return
        if exc is not None:
            _LOG.error(
                "[baseline] dispatch executor crashed task=%s: %s",
                task_id[:8], exc, exc_info=exc,
            )

    async def _dispatch_one(
        self, rec: _BaselineTaskRecord, decision: DispatchDecision,
    ) -> None:
        """Cold-start worker (if needed) + gRPC InferBatch + lifecycle.

        Mirrors proton's dispatch path (``http_server._dispatch_selected_worker``)
        in the bare-minimum form: ensure worker ready → resolve gRPC addr
        → build BatchRequest → call InferBatch → parse BatchResponse.
        No SignalService / SchedulingScenario / RealityValidator hooks —
        baselines are scheduling-algorithm-only ports per plan .
        """
        dispatch_started = False
        dispatch_target = str(decision.target_worker_name or "").strip()
        try:
            ready = False
            ready_state: Any = None
            try:
                if hasattr(self._supervisor, "ensure_worker_ready"):
                    ready_result = await self._supervisor.ensure_worker_ready(
                        component=decision.component,
                        worker_name=decision.target_worker_name,
                    )
                    if isinstance(ready_result, tuple):
                        ready_state = ready_result[0]
                        ready = bool(getattr(ready_state, "ready", False))
                    else:
                        ready = bool(ready_result)
            except Exception as e:
                _LOG.warning(
                    "[baseline] ensure_worker_ready failed for %s: %s",
                    decision.target_worker_name, e,
                )
                ready = False
            if not ready:
                release_dispatch_pending = getattr(
                    self._supervisor, "release_dispatch_pending", None,
                )
                if callable(release_dispatch_pending):
                    try:
                        release_dispatch_pending(dispatch_target)
                    except Exception:
                        _LOG.warning(
                            "[baseline] release_dispatch_pending failed for %s",
                            dispatch_target,
                            exc_info=True,
                        )
                self._scheduler.on_dispatch_failure(
                    rec.task_id, "worker_not_ready", time.time(),
                )
                async with self._tasks_lock:
                    rec.state = _STATE_FAILED
                    rec.message = "worker_not_ready"
                    rec.updated_at = time.time()
                return

            addr = ""
            if ready_state is not None:
                addr = str(getattr(ready_state, "addr", "") or "")
            if not addr:
                addr = self._resolve_worker_addr(decision.target_worker_name) or ""
            if not addr:
                release_dispatch_pending = getattr(
                    self._supervisor, "release_dispatch_pending", None,
                )
                if callable(release_dispatch_pending):
                    try:
                        release_dispatch_pending(dispatch_target)
                    except Exception:
                        _LOG.warning(
                            "[baseline] release_dispatch_pending failed for %s",
                            dispatch_target,
                            exc_info=True,
                        )
                self._scheduler.on_dispatch_failure(
                    rec.task_id, "worker_addr_missing", time.time(),
                )
                async with self._tasks_lock:
                    rec.state = _STATE_FAILED
                    rec.message = (
                        f"worker {decision.target_worker_name} ready but "
                        f"no gRPC addr"
                    )
                    rec.updated_at = time.time()
                return
            async with self._tasks_lock:
                rec.target_worker_addr = addr

            on_task_start = getattr(self._supervisor, "on_task_start", None)
            if callable(on_task_start):
                try:
                    on_task_start(addr)
                    dispatch_started = True
                except Exception:
                    _LOG.warning(
                        "[baseline] on_task_start failed for %s",
                        addr,
                        exc_info=True,
                    )
                    raise
            self._scheduler.on_dispatch_success(
                rec.task_id, decision.target_gpu_id, time.time(),
            )
            await self._export_gpu_allocation_truth(
                rec, decision, event="dispatch_start",
            )

            try:
                batch_id = next(self._batch_id_seq)
                params = dict(rec.env or {})
                if rec.config_fingerprint:
                    params["config_fingerprint"] = rec.config_fingerprint
                if rec.workdir:
                    params["workdir"] = rec.workdir
                payload_dict = dict(rec.submission.workload_features or {})
                payload_dict["mode"] = rec.mode or "nextflow_task"
                if rec.workdir:
                    payload_dict["workdir"] = rec.workdir
                if rec.argv:
                    payload_dict["argv"] = list(rec.argv)
                if rec.tool_cwd:
                    payload_dict["tool_cwd"] = rec.tool_cwd
                if rec.env:
                    payload_dict.setdefault("env", dict(rec.env))
                payload_bytes = json.dumps(payload_dict).encode("utf-8")
                req = pb.BatchRequest(
                    batch_id=batch_id,
                    bucket_id=rec.config_fingerprint or rec.component,
                    params=params,
                    requests=[pb.RequestItem(
                        request_id=rec.task_id,
                        payload_json=payload_bytes,
                        params=params,
                    )],
                )

                channel = self._get_channel(addr)
                stub = pb_grpc.ModelWorkerStub(channel)
                try:
                    resp: pb.BatchResponse = await stub.InferBatch(
                        req, timeout=float(rec.timeout_s),
                    )
                except grpc.aio.AioRpcError as exc:
                    _LOG.warning(
                        "[baseline] InferBatch gRPC error for task=%s "
                        "addr=%s code=%s: %s",
                        rec.task_id[:8], addr, exc.code(), exc.details(),
                    )
                    self._scheduler.on_dispatch_failure(
                        rec.task_id,
                        f"grpc_{exc.code().name.lower()}",
                        time.time(),
                    )
                    async with self._tasks_lock:
                        rec.state = _STATE_FAILED
                        rec.message = f"gRPC {exc.code().name}: {exc.details()}"
                        rec.updated_at = time.time()
                    return

                ok = bool(getattr(resp, "ok", False))
                err = str(getattr(resp, "error_message", "") or "")
                response_items = list(getattr(resp, "responses", []) or [])
                item_payload: Optional[Dict[str, Any]] = None
                item_ok = ok
                item_err = err
                if response_items:
                    first = response_items[0]
                    item_ok = bool(getattr(first, "ok", ok))
                    item_err = str(getattr(first, "error_message", err) or err)
                    pj = getattr(first, "payload_json", b"") or b""
                    if pj:
                        try:
                            item_payload = json.loads(pj.decode("utf-8"))
                        except Exception:
                            item_payload = {"raw_payload_bytes": len(pj)}

                async with self._tasks_lock:
                    rec.state = (
                        _STATE_SUCCEEDED if (ok and item_ok) else _STATE_FAILED
                    )
                    rec.message = item_err if not (ok and item_ok) else ""
                    rec.result_payload = item_payload
                    rec.updated_at = time.time()
            finally:
                await self._export_gpu_allocation_truth(
                    rec, decision, event="dispatch_end",
                )
                self._scheduler.on_task_complete(rec.task_id, time.time())
        except Exception as e:
            _LOG.error(
                "[baseline] dispatch error for task %s: %s",
                rec.task_id[:8], e, exc_info=True,
            )
            if not dispatch_started:
                release_dispatch_pending = getattr(
                    self._supervisor, "release_dispatch_pending", None,
                )
                if callable(release_dispatch_pending):
                    try:
                        release_dispatch_pending(dispatch_target)
                    except Exception:
                        _LOG.warning(
                            "[baseline] release_dispatch_pending failed for %s",
                            dispatch_target,
                            exc_info=True,
                        )
            self._scheduler.on_dispatch_failure(
                rec.task_id, f"exception: {type(e).__name__}", time.time(),
            )
            async with self._tasks_lock:
                rec.state = _STATE_FAILED
                rec.message = f"{type(e).__name__}: {e}"
                rec.updated_at = time.time()
        finally:
            if dispatch_started:
                on_task_end = getattr(self._supervisor, "on_task_end", None)
                if callable(on_task_end):
                    try:
                        await on_task_end(
                            str(getattr(rec, "target_worker_addr", "") or ""),
                        )
                    except Exception:
                        _LOG.warning(
                            "[baseline] on_task_end failed for %s",
                            getattr(rec, "target_worker_addr", ""),
                            exc_info=True,
                        )


    async def run(self) -> None:
        """Start HTTP server + scheduler tick loop."""
        self._running = True

        app = web.Application()
        app.router.add_post("/api/v1/job/submit", self.submit_job)
        app.router.add_get("/api/v1/job/{job_id}", self.get_job)
        app.router.add_get("/api/v1/profile/export", self.get_profile_export)
        app.router.add_get("/api/v1/workers", self.get_workers)
        app.router.add_get("/api/v1/debug/tasks", self.get_debug_tasks)
        app.router.add_post("/api/v1/campaign/complete", self.mark_campaign_complete)
        app.router.add_get("/health", self.health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        _LOG.info(
            "[baseline] gateway listening on http://%s:%d (scheduler=%s)",
            self._host, self._port, self._scheduler.name,
        )

        tick_task = asyncio.create_task(self._scheduler_tick())
        try:
            await tick_task
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await runner.cleanup()



def select_scheduler(
    runtime_cfg: Dict[str, Any],
    workers_profile: WorkersProfile,
    gpu_capacity_mb: Dict[str, int],
    worker_name_format: str = "{component}-gpu{gpu_id}",
) -> BaselineSchedulerProtocol:
    """Instantiate the scheduler chosen by ``scheduler_implementation``.

    Recognised values: ``fifo``, ``k8s``, ``slurm``.  ``proton`` is
    handled by the existing ``_run_all`` path in ``__main__.py`` and
    must NOT reach this factory.
    """
    impl = str(runtime_cfg.get("scheduler_implementation", "")).strip().lower()
    if impl == "fifo":
        from gateway.baseline.fifo import FifoBaselineScheduler
        return FifoBaselineScheduler(
            profile=workers_profile,
            gpu_capacity_mb=gpu_capacity_mb,
            worker_name_format=worker_name_format,
        )
    if impl == "k8s":
        from gateway.baseline.k8s import K8sBaselineScheduler
        return K8sBaselineScheduler(
            profile=workers_profile,
            gpu_capacity_mb=gpu_capacity_mb,
            worker_name_format=worker_name_format,
            total_host_ram_mb=_read_mem_total_mib(),
            host_ram_min_available_mb=int(
                runtime_cfg.get("admission_host_ram_min_available_mib", 0) or 0
            ),
        )
    if impl == "slurm":
        from gateway.baseline.slurm import SlurmBaselineScheduler
        return SlurmBaselineScheduler(
            profile=workers_profile,
            gpu_capacity_mb=gpu_capacity_mb,
            worker_name_format=worker_name_format,
            age_factor=float(runtime_cfg.get("slurm_age_factor", 1.0)),
            easy_backfill=bool(runtime_cfg.get("slurm_easy_backfill", True)),
            whole_gpu_exclusive=bool(
                runtime_cfg.get("slurm_whole_gpu_exclusive", False)
            ),
            total_host_ram_mb=_read_mem_total_mib(),
            host_ram_min_available_mb=int(
                runtime_cfg.get("admission_host_ram_min_available_mib", 0) or 0
            ),
        )
    raise ValueError(
        f"Unknown scheduler_implementation={impl!r} — expected one of "
        f"{{fifo, k8s, slurm}} (proton handled separately)"
    )
