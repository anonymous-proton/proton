"""Model worker gRPC server."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import traceback
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import grpc

from . import modelworker_pb2 as pb
from . import modelworker_pb2_grpc as pb_grpc
from .memory_observer import BootstrapMemorySummary
from .model_adapter import ModelAdapter
from .nextflow_contract import ContractError, validate_task_payload
from .runtime_telemetry import (
    DispatchAttribution,
    ResidentBaselineSnapshot,
    WorkerTelemetryCollector,
)
from .worker_engine import FatalActorRestartSignal, WorkerEngine


_REPO_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTIES = _REPO_ROOT / "third_parties"
_MODEL_ADAPTERS = _REPO_ROOT / "model_adapters"
_LOG = logging.getLogger(__name__)
_SUFFIX = "_worker"
_FATAL_ACTOR_EXIT_CODE = 70
_FATAL_EXIT_FALLBACK_DELAY_SEC = 1.0


def _add_third_party_paths(package: str) -> bool:
    if not _THIRD_PARTIES.exists():
        return False
    added = False
    for entry in _THIRD_PARTIES.iterdir():
        candidate = entry
        if candidate.exists() and candidate.is_dir() and candidate.name == package:
            entry_str = str(entry)
            if entry_str not in sys.path:
                sys.path.insert(0, entry_str)
            added = True
    return added


def _resolve_adapter(path: str) -> ModelAdapter:
    module_path, _, class_name = path.rpartition(":")
    if not module_path or not class_name:
        raise ValueError("adapter must be in module:ClassName form")

    top_pkg = module_path.split(".")[0]
    adapter_pkg = top_pkg + _SUFFIX
    adapter_module_path = adapter_pkg + "." + module_path.split(".", 1)[1]

    if _MODEL_ADAPTERS.exists():
        pkg_dir = _MODEL_ADAPTERS / adapter_pkg
        if pkg_dir.exists() and pkg_dir.is_dir():
            adapter_root_str = str(_MODEL_ADAPTERS)
            if adapter_root_str not in sys.path:
                sys.path.insert(0, adapter_root_str)

    if not _add_third_party_paths(top_pkg):
        raise FileNotFoundError(
            f"Could not find model source for '{top_pkg}' in '{_THIRD_PARTIES}'. "
            "Please ensure the directory name in 'third_parties' matches the "
            "adapter package name (case-sensitive)."
        )

    try:
        module = importlib.import_module(adapter_module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to load adapter '{adapter_module_path}'. Checked third_parties "
            f"for '{adapter_pkg}'. Original error: {e}"
        ) from e

    adapter_cls: type[ModelAdapter] = getattr(module, class_name)
    return adapter_cls()


def _is_overridden(adapter: ModelAdapter, method: str) -> bool:
    fn = getattr(adapter, method)
    fn_obj = getattr(fn, "__func__", fn)
    base_fn = getattr(ModelAdapter, method, None)
    if base_fn is None:
        return True
    base_obj = getattr(base_fn, "__func__", base_fn)
    return fn_obj is not base_obj


def _resolve_addr(cli_addr: str) -> str:
    if cli_addr:
        return cli_addr
    return "0.0.0.0:50051"


def _resolve_adapter_from_env(cli_adapter: str) -> ModelAdapter:
    if not cli_adapter:
        raise SystemExit("Adapter is required. Use --adapter module:ClassName.")
    return _resolve_adapter(cli_adapter)


def _configure_logging(level: str) -> None:
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        resolved = logging.INFO
    logging.basicConfig(level=resolved, force=True)


def _build_response_payload(response: dict[str, Any]) -> bytes:
    payload: dict[str, Any] = {}
    memory_telemetry = response.get("_memory_telemetry")
    if isinstance(memory_telemetry, Mapping):
        for key in (
            "active_vram_mib",
            "peak_vram_mib",
            "vram_memory_basis",
            "vram_memory_measurement",
            "vram_memory_qc_keep",
            "vram_memory_attribution",
            "host_active_memory_mib",
            "host_peak_memory_mib",
            "host_resident_memory_mib",
            "host_resident_memory_source",
            "host_resident_baseline_collected_at",
            "host_resident_baseline_lifecycle_token",
            "host_resident_baseline_state",
            "host_memory_basis",
            "host_memory_measurement",
            "host_memory_qc_keep",
            "host_memory_attribution",
            "host_dispatch_memory_window",
            "host_bootstrap_memory_summary",
            "host_peak_fidelity",
            "logical_concurrent_count",
            "synthetic_supervisor_activation_mib",
            "memory_basis",
            "peak_fidelity",
            "active_memory_mib",
            "peak_memory_mib",
            "active_memory_scope",
            "active_memory_measurement",
            "total_upper_bound_mib",
            "resident_memory_mib",
            "resident_memory_source",
            "resident_baseline_collected_at",
            "resident_baseline_lifecycle_token",
            "resident_baseline_state",
            "memory_qc_keep",
            "concurrent_execute_overlap",
            "actor_id",
            "actor_generation",
            "actor_pid",
            "actor_resident_owner",
            "actor_resident_vram_mib",
            "actor_resident_host_mib",
            "shared_resident_vram_mib",
            "shared_resident_host_mib",
            "vram_source_complete",
            "host_source_complete",
            "device_pre_vram_mib",
            "device_peak_vram_mib",
            "device_post_vram_mib",
            "device_unattributed_pre_vram_mib",
            "device_unattributed_peak_vram_mib",
            "device_vram_measurement",
        ):
            if key in memory_telemetry:
                payload[key] = memory_telemetry.get(key)
    dynamic_batch_telemetry = response.get("_dynamic_batch_telemetry")
    if isinstance(dynamic_batch_telemetry, Mapping):
        for key in (
            "dynamic_batch_argument_applied_k",
            "dynamic_batch_consumed_k",
            "dynamic_batch_logical_n",
            "dynamic_batch_group_count",
        ):
            if key in dynamic_batch_telemetry:
                payload[key] = dynamic_batch_telemetry.get(key)
    batch_timing = response.get("_batch_timing_us")
    if isinstance(batch_timing, dict):
        try:
            total_us = int(batch_timing.get("total_us", 0))
        except Exception:
            total_us = 0
        payload["telemetry_wall_clock_sec"] = (
            total_us / 1_000_000.0 if total_us > 0 else None
        )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")


def _encode_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")


def _observation_json_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        return _encode_json_bytes(dict(value))
    if hasattr(value, "as_dict"):
        payload = value.as_dict()
        if isinstance(payload, Mapping):
            return _encode_json_bytes(dict(payload))
    return b""


def _build_baseline_stats(
    snapshot: ResidentBaselineSnapshot,
) -> pb.ResidentBaselineSnapshot:
    payload = snapshot.as_payload()
    resident_memory_mib = payload.get("resident_memory_mib")
    collected_at = payload.get("resident_baseline_collected_at")
    try:
        resident_memory_value = (
            float(resident_memory_mib) if resident_memory_mib is not None else 0.0
        )
        collected_at_value = float(collected_at) if collected_at is not None else 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid resident baseline numeric value") from exc
    return pb.ResidentBaselineSnapshot(
        resident_memory_mib=resident_memory_value,
        resident_memory_source=str(payload.get("resident_memory_source") or ""),
        resident_baseline_collected_at=collected_at_value,
        resident_baseline_lifecycle_token=str(
            payload.get("resident_baseline_lifecycle_token") or ""
        ),
        resident_baseline_state=str(payload.get("resident_baseline_state") or ""),
    )


class ModelWorkerService(pb_grpc.ModelWorkerServicer):
    def __init__(
        self,
        engine: WorkerEngine,
        adapter: ModelAdapter,
        worker_id: str,
        *,
        initial_ready: bool,
        engine_started: bool,
        fatal_exit: Callable[[int], None] | None = None,
    ):
        self.engine = engine
        self.adapter = adapter
        self.worker_id = worker_id
        self._ready = initial_ready
        self._engine_started = engine_started
        self._bootstrap_error = ""
        self._bootstrap_task: asyncio.Task[None] | None = None
        self._state_lock = asyncio.Lock()
        self._fatal_exit = fatal_exit or os._exit
        self._fatal_exit_scheduled = False

    def _arm_fatal_restart_after_response(
        self,
        context: grpc.aio.ServicerContext,
        signal: FatalActorRestartSignal,
    ) -> None:
        if self._fatal_exit_scheduled:
            return
        self._fatal_exit_scheduled = True
        self._ready = False
        _LOG.critical(
            "[execute-actor] fatal_cuda class=%s first_actor=%s "
            "first_generation=%s second_actor=%s second_generation=%s "
            "batch=%s decision=container_fallback_response_then_exit code=%s",
            signal.error_class,
            signal.first_actor_id,
            signal.first_generation,
            signal.second_actor_id,
            signal.second_generation,
            signal.second_batch_id,
            _FATAL_ACTOR_EXIT_CODE,
        )

        def _exit_after_response(_context: grpc.aio.ServicerContext) -> None:
            self._fatal_exit(_FATAL_ACTOR_EXIT_CODE)

        try:
            context.add_done_callback(cast(Any, _exit_after_response))
        except Exception:
            _LOG.exception(
                "failed to register response completion callback; "
                "using delayed fatal exit"
            )
            asyncio.get_running_loop().call_later(
                _FATAL_EXIT_FALLBACK_DELAY_SEC,
                self._fatal_exit,
                _FATAL_ACTOR_EXIT_CODE,
            )

    def set_bootstrap_task(self, task: asyncio.Task[None]) -> None:
        self._bootstrap_task = task

    async def bootstrap(self) -> None:
        try:
            await self.engine.start()
        except Exception as exc:
            async with self._state_lock:
                self._ready = False
                self._engine_started = False
                self._bootstrap_error = str(exc)
            _LOG.error("worker bootstrap failed: %s", exc)
            _LOG.error("Traceback:\n%s", traceback.format_exc())
            return

        async with self._state_lock:
            self._ready = True
            self._engine_started = True
            self._bootstrap_error = ""

    async def GetCapabilities(
        self, request: pb.Empty, context: grpc.aio.ServicerContext
    ) -> pb.Capabilities:
        return pb.Capabilities(
            worker_id=self.worker_id,
            model_name=self.adapter.model_name(),
            model_version=self.adapter.model_version(),
            supported_buckets=self.adapter.supported_buckets(),
            max_batch_size=self.adapter.max_batch_size(),
            max_inflight_batches=self.adapter.max_inflight_batches(),
            supports_prepare_batch=_is_overridden(self.adapter, "prepare_batch"),
            supports_finalize_batch=_is_overridden(self.adapter, "finalize_batch"),
            supports_cancel_batch=self.engine.supports_cancel_batch,
        )

    async def GetStats(
        self, request: pb.Empty, context: grpc.aio.ServicerContext
    ) -> pb.WorkerStats:
        stats = self.engine.stats()
        collector = self.engine.telemetry_collector
        snapshot = (
            collector.get_resident_baseline_snapshot()
            if collector is not None
            else ResidentBaselineSnapshot()
        )
        bootstrap_summary = (
            collector.current_bootstrap_summary()
            if collector is not None and hasattr(collector, "current_bootstrap_summary")
            else BootstrapMemorySummary()
        )
        return pb.WorkerStats(
            worker_id=self.worker_id,
            queues=pb.QueueStats(
                in_queue=stats["in_queue"],
                prepared_queue=stats["prepared_queue"],
                output_queue=stats["output_queue"],
                prepare_inflight=stats["prepare_inflight"],
                execute_inflight=stats["execute_inflight"],
                finalize_inflight=stats["finalize_inflight"],
                prepare_concurrency=stats["prepare_concurrency"],
                execute_concurrency=stats["execute_concurrency"],
                finalize_concurrency=stats["finalize_concurrency"],
                item_concurrency=stats["item_concurrency"],
            ),
            resident_baseline=_build_baseline_stats(snapshot),
            bootstrap_memory_summary_json=_observation_json_bytes(bootstrap_summary),
        )

    async def CancelBatch(
        self, request: pb.CancelBatchRequest, context: grpc.aio.ServicerContext
    ) -> pb.CancelBatchResponse:
        if not self.engine.supports_cancel_batch:
            return pb.CancelBatchResponse(
                ok=False,
                message="adapter does not support cancellation",
            )
        ok = await self.engine.cancel_batch(request.batch_id)
        return pb.CancelBatchResponse(
            ok=ok,
            message="cancelled" if ok else "batch not found",
        )

    async def Health(
        self, request: pb.HealthRequest, context: grpc.aio.ServicerContext
    ) -> pb.HealthResponse:
        if request.readiness:
            if self._bootstrap_error:
                return pb.HealthResponse(
                    ok=False, message=f"bootstrap failed: {self._bootstrap_error}"
                )
            return pb.HealthResponse(
                ok=self._ready, message="ready" if self._ready else "not ready"
            )
        return pb.HealthResponse(ok=True, message="alive")

    async def InferBatch(
        self, request: pb.BatchRequest, context: grpc.aio.ServicerContext
    ) -> pb.BatchResponse:
        if not self._ready:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            message = "worker not ready"
            if self._bootstrap_error:
                message = f"worker bootstrap failed: {self._bootstrap_error}"
            return pb.BatchResponse(
                batch_id=request.batch_id,
                ok=False,
                error_message=message,
            )

        attribution = DispatchAttribution()
        if self.engine.telemetry_collector is not None:
            attribution = (
                await self.engine.telemetry_collector.note_real_dispatch_attempt()
            )
        pb_attribution = pb.DispatchAttribution(
            worker_generation_token=str(attribution.worker_generation_token or ""),
            run_ordinal_in_generation=attribution.run_ordinal_in_generation or 0,
            is_first_real_run=bool(attribution.is_first_real_run),
        )

        batch_id = request.batch_id
        bucket_id = request.bucket_id if request.bucket_id else None
        batch_params = dict(request.params)

        req_ids: list[str] = []
        req_dicts: list[dict[str, Any]] = []
        per_req_params: list[dict[str, str]] = []
        response_slots: list[pb.ResponseItem] = []
        valid_indices: list[int] = []

        for idx, req in enumerate(request.requests):
            try:
                payload = (
                    json.loads(req.payload_json.decode("utf-8"))
                    if req.payload_json
                    else {}
                )
            except Exception as exc:
                response_slots.append(
                    pb.ResponseItem(
                        request_id=req.request_id,
                        ok=False,
                        payload_json=b"",
                        error_message=f"invalid json: {exc}",
                    )
                )
                continue
            try:
                payload = validate_task_payload(payload)
            except ContractError as exc:
                response_slots.append(
                    pb.ResponseItem(
                        request_id=req.request_id,
                        ok=False,
                        payload_json=b"",
                        error_message=str(exc),
                    )
                )
                continue
            except Exception as exc:
                response_slots.append(
                    pb.ResponseItem(
                        request_id=req.request_id,
                        ok=False,
                        payload_json=b"",
                        error_message=f"contract validation failed: {exc}",
                    )
                )
                continue
            req_ids.append(req.request_id)
            per_req_params.append(dict(req.params))
            req_dicts.append(payload)
            response_slots.append(pb.ResponseItem())
            valid_indices.append(idx)

        if not valid_indices:
            return pb.BatchResponse(
                batch_id=request.batch_id,
                responses=response_slots,
                timing=pb.BatchTiming(),
                attribution=pb_attribution,
                ok=True,
                error_message="",
            )

        try:
            responses = await self.engine.submit_batch(
                batch_id=batch_id,
                request_ids=req_ids,
                requests=req_dicts,
                bucket_id=bucket_id,
                params=batch_params,
                per_request_params=per_req_params,
                dispatch_attribution=attribution,
            )
        except asyncio.CancelledError as exc:
            await self.engine.cancel_batch(batch_id)
            fatal_restart = self.engine.claim_fatal_restart_signal(batch_id)
            if fatal_restart is not None:
                self._arm_fatal_restart_after_response(context, fatal_restart)
            context_cancelled = getattr(context, "cancelled", None)
            client_cancelled = False
            if callable(context_cancelled):
                client_cancelled = context_cancelled()
            if client_cancelled:
                raise exc
            return pb.BatchResponse(
                batch_id=request.batch_id,
                attribution=pb_attribution,
                ok=False,
                error_message="batch cancelled",
            )
        except Exception as exc:
            return pb.BatchResponse(
                batch_id=request.batch_id,
                attribution=pb_attribution,
                ok=False,
                error_message=str(exc),
            )

        timing_us = None
        bootstrap_memory_summary_json = b""
        dispatch_memory_window_json = b""

        for i, response in enumerate(responses):
            request_id = response.get("request_id", "")
            ok = bool(response.get("ok", False))
            error = response.get("error_message", "")
            memory_observation = response.get("_memory_observation")
            if not isinstance(memory_observation, Mapping):
                memory_observation = {}

            if "_batch_timing_us" in response:
                timing_us = response["_batch_timing_us"]
            if not bootstrap_memory_summary_json:
                bootstrap_memory_summary_json = _observation_json_bytes(
                    memory_observation.get("bootstrap_memory_summary")
                )
            if not dispatch_memory_window_json:
                dispatch_memory_window_json = _observation_json_bytes(
                    memory_observation.get("dispatch_memory_window")
                )

            if i < len(valid_indices):
                overall_idx = valid_indices[i]
                response_slots[overall_idx] = pb.ResponseItem(
                    request_id=request_id,
                    ok=ok,
                    payload_json=_build_response_payload(response),
                    error_message=error if not ok else "",
                )

        bt = pb.BatchTiming()
        if timing_us:
            try:
                bt = pb.BatchTiming(
                    queue_delay_us=int(timing_us.get("queue_delay_us", 0)),
                    prepare_us=int(timing_us.get("prepare_us", 0)),
                    execute_us=int(timing_us.get("execute_us", 0)),
                    finalize_us=int(timing_us.get("finalize_us", 0)),
                    total_us=int(timing_us.get("total_us", 0)),
                )
            except (TypeError, ValueError):
                _LOG.warning("invalid batch timing payload: %r", timing_us)

        response = pb.BatchResponse(
            batch_id=request.batch_id,
            responses=response_slots,
            timing=bt,
            attribution=pb_attribution,
            bootstrap_memory_summary_json=bootstrap_memory_summary_json,
            dispatch_memory_window_json=dispatch_memory_window_json,
            ok=True,
            error_message="",
        )
        fatal_restart = self.engine.claim_fatal_restart_signal(batch_id)
        if fatal_restart is not None:
            response.ok = False
            response.error_message = fatal_restart.detail
            self._arm_fatal_restart_after_response(context, fatal_restart)
        return response


async def serve(
    addr: str,
    adapter: ModelAdapter,
    *,
    prepare_concurrency: int,
    execute_concurrency: int,
    finalize_concurrency: int,
    item_concurrency: int,
    execute_processes: int,
) -> None:
    worker_id = os.environ.get("WORKER_ID", str(uuid.uuid4()))

    engine = WorkerEngine(
        adapter=adapter,
        prepare_concurrency=prepare_concurrency,
        execute_concurrency=execute_concurrency,
        finalize_concurrency=finalize_concurrency,
        item_concurrency=item_concurrency,
        execute_processes=execute_processes,
        telemetry_collector=WorkerTelemetryCollector(lifecycle_id_prefix=worker_id),
    )

    server = grpc.aio.server(
        options=[
            ("grpc.keepalive_time_ms", 2**31 - 1),
            ("grpc.http2.min_recv_ping_interval_without_data_ms", 5000),
            ("grpc.keepalive_permit_without_calls", 0),
        ]
    )
    svc = ModelWorkerService(
        engine=engine,
        adapter=adapter,
        worker_id=worker_id,
        initial_ready=False,
        engine_started=False,
    )
    pb_grpc.add_ModelWorkerServicer_to_server(svc, server)

    server.add_insecure_port(addr)
    await server.start()
    bootstrap_task = asyncio.create_task(svc.bootstrap())
    svc.set_bootstrap_task(bootstrap_task)
    try:
        await server.wait_for_termination()
    finally:
        if not bootstrap_task.done():
            bootstrap_task.cancel()
        await asyncio.gather(bootstrap_task, return_exceptions=True)
        if svc._engine_started:
            await engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--addr",
        type=str,
        default="",
        help="gRPC bind address. Defaults to 0.0.0.0:50051.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter in module:ClassName form.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, WARNING).",
    )
    parser.add_argument(
        "--prepare-concurrency",
        type=int,
        default=2,
        help="Number of prepare workers (default: 2).",
    )
    parser.add_argument(
        "--execute-concurrency",
        type=int,
        default=1,
        help="Number of execute workers (default: 1).",
    )
    parser.add_argument(
        "--finalize-concurrency",
        type=int,
        default=2,
        help="Number of finalize workers (default: 2).",
    )
    parser.add_argument(
        "--item-concurrency",
        type=int,
        default=8,
        help="Per-item concurrency for prepare/finalize (default: 8).",
    )
    parser.add_argument(
        "--execute-processes",
        type=int,
        default=0,
        help=(
            "Persistent execute actors: 0 or legacy -1 = grow up to "
            "execute-concurrency (default), >0 = additionally cap actors to N."
        ),
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    addr = _resolve_addr(args.addr)
    adapter = _resolve_adapter_from_env(args.adapter)
    asyncio.run(
        serve(
            addr,
            adapter,
            prepare_concurrency=args.prepare_concurrency,
            execute_concurrency=args.execute_concurrency,
            finalize_concurrency=args.finalize_concurrency,
            item_concurrency=args.item_concurrency,
            execute_processes=args.execute_processes,
        )
    )


if __name__ == "__main__":
    main()
