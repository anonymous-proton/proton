"""Async worker engine with a three-stage pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import logging
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concurrency_utils import classify_sticky_cuda_error, maybe_gc
from .memory_observer import MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE
from .model_adapter import ModelAdapter
from .runtime_telemetry import (
    DispatchAttribution,
    MemoryTelemetry,
    WorkerTelemetryCollector,
)

_LOG = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _us(dt_s: float) -> int:
    try:
        return int(dt_s * 1_000_000)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid duration: {dt_s!r}") from exc


@contextmanager
def _worker_execute_marker() -> Generator[None, None, None]:
    """Best-effort NVTX range around execute stage (adapter-agnostic)."""
    pop = None
    try:
        import nvtx

        nvtx.push_range("worker_execute")
        pop = nvtx.pop_range
    except Exception:
        try:
            import torch

            torch.cuda.nvtx.range_push("worker_execute")
            pop = torch.cuda.nvtx.range_pop
        except Exception:
            pop = None
    try:
        yield
    finally:
        if pop is not None:
            with suppress(Exception):
                pop()


@dataclass
class BatchEnvelope:
    batch_id: int
    requests: list[dict[str, Any]]
    request_ids: list[str]
    bucket_id: str | None
    params: dict[str, str]
    per_request_params: list[dict[str, str]]
    fut: asyncio.Future
    timing: dict[str, float] = field(default_factory=dict)
    memory_telemetry: MemoryTelemetry | None = None
    dispatch_attribution: DispatchAttribution | None = None
    telemetry_state: Any | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    execute_started: bool = False
    execute_finished: bool = False


@dataclass
class PreparedEnvelope:
    env: BatchEnvelope
    prepared: list[Any | None]
    errors: list[str | None]
    dynamic_batch_contexts: list[dict[str, Any] | None]


@dataclass
class OutputEnvelope:
    env: BatchEnvelope
    outputs: list[Any | None]
    errors: list[str | None]
    dynamic_batch_telemetry: list[dict[str, Any]]


_Q_IN_MAXSIZE: int = 1

_UNLIMITED_EXECUTE_CAP: int = 256


_PROCESS_JOIN_TIMEOUT_SEC = 2.0
_PROCESS_REAP_TIMEOUT_SEC = 4.0
_PROCESS_NATURAL_EXIT_TIMEOUT_SEC = 30.0
_PROCESS_REAP_POLL_SEC = 0.1
_PROCESS_PROBE_TIMEOUT_SEC = 120.0
_PROCESS_START_TIMEOUT_SEC = 30.0
_PR_SET_CHILD_SUBREAPER = 36


def _enable_child_subreaper() -> bool:
    """Keep killed actor descendants reapable by the worker parent on Linux."""
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return True


def _spawn_process_context() -> Any:
    """Use torch's CUDA reducers when present, otherwise stdlib spawn."""
    try:
        import torch.multiprocessing as process_mp
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "torch.multiprocessing"}:
            raise
        import multiprocessing as process_mp

    return process_mp.get_context("spawn")


@dataclass
class ExecuteActorHandle:
    actor_id: int
    generation: int
    process: Any
    conn: Any
    started_at: float
    batch_id: int | None = None
    ready_at: float | None = None
    process_group_id: int | None = None
    pid: int | None = None
    exitcode: int | None = None
    assignments: int = 0
    cancel_requested: bool = False
    reaped: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class FatalActorRestartSignal:
    error_class: str
    first_actor_id: int
    first_generation: int
    second_actor_id: int
    second_generation: int
    second_batch_id: int
    detail: str


_DYNAMIC_BATCH_ERROR = "dynamic_batch_contract_error:"


def _contract_error(message: str) -> RuntimeError:
    return RuntimeError(f"{_DYNAMIC_BATCH_ERROR} {message}")


def _rewrite_dynamic_batch_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    descriptor = request.get("_dynamic_batch")
    if not isinstance(descriptor, dict):
        return request, None
    overrides = request.get("execution_overrides")
    if not isinstance(overrides, dict):
        raise _contract_error("execution_overrides must contain selected k")
    try:
        k = int(overrides.get("batch_size"))
        logical_n = int(descriptor.get("logical_n"))
    except (TypeError, ValueError) as exc:
        raise _contract_error("selected k and logical N must be integers") from exc
    if logical_n <= 0 or k <= 0 or k > logical_n:
        raise _contract_error(f"invalid selected k/N: {k}/{logical_n}")
    arg = str(descriptor.get("batch_size_arg") or "").strip()
    if not arg or any(char.isspace() for char in arg) or "=" in arg:
        raise _contract_error("invalid batch_size_arg descriptor")

    copied = dict(request)
    argv = list(request.get("argv") or [])
    matches: list[tuple[int, str]] = []
    if arg.startswith("-"):
        for index, token in enumerate(argv):
            if token == arg:
                matches.append((index, "pair"))
            elif token.startswith(arg + "="):
                matches.append((index, "equal"))
    else:
        for index, token in enumerate(argv):
            if token.startswith(arg + "="):
                matches.append((index, "equal"))
    if len(matches) > 1:
        raise _contract_error(f"duplicate batch argument: {arg}")
    if not matches:
        if arg.startswith("-"):
            argv.extend((arg, str(k)))
        else:
            argv.append(f"{arg}={k}")
    else:
        index, form = matches[0]
        if form == "equal":
            argv[index] = f"{arg}={k}"
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise _contract_error(f"batch argument has no value: {arg}")
            argv[index + 1] = str(k)
    copied["argv"] = argv
    copied["execution_overrides"] = dict(overrides)
    return copied, {
        "batch_size_arg": arg,
        "execution_batch_size": k,
        "logical_batch_size": logical_n,
    }


def _fasta_records(path: Path) -> tuple[list[bytes], list[str]]:
    data = path.read_bytes()
    starts = [
        index
        for index, line in enumerate(data.splitlines(keepends=True))
        if line.startswith(b">")
    ]
    lines = data.splitlines(keepends=True)
    if not starts or any(line.strip() for line in lines[: starts[0]]):
        raise _contract_error("FASTA must contain ordered header records")
    records: list[bytes] = []
    names: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        header = lines[start][1:].strip().decode("utf-8", errors="strict")
        name = header.split()[0] if header else ""
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or any(not (char.isalnum() or char in "._-") for char in name)
        ):
            raise _contract_error("FASTA query names must be safe basenames")
        records.append(b"".join(lines[start:end]))
        names.append(name)
    if len(set(names)) != len(names):
        raise _contract_error("FASTA query names must be unique")
    return records, names


def _mmseqs2_stage_root(
    prepared: Any,
    batch_id: int,
    item_index: int,
) -> Path | None:
    if (
        not isinstance(prepared, dict)
        or not prepared.get("workdir")
        or not prepared.get("output_dir")
    ):
        return None
    workdir = Path(str(prepared["workdir"])).resolve()
    target = Path(str(prepared["output_dir"]))
    target = (target if target.is_absolute() else workdir / target).resolve()
    if not target.is_relative_to(workdir):
        return None
    return target.parent / f".{target.name}.mmseqs2-attempt-{batch_id}-{item_index}"


def _mmseqs2_stage_roots(
    adapter: ModelAdapter,
    prepared: list[Any],
    contexts: list[dict[str, Any] | None],
    batch_id: int,
) -> list[Path]:
    if len(prepared) != len(contexts):
        raise _contract_error("prepared/context length mismatch")
    if str(adapter.model_name() or "").strip().lower() != "mmseqs2":
        return []
    roots: list[Path] = []
    for index, item in enumerate(prepared):
        root = _mmseqs2_stage_root(item, batch_id, index)
        if root is not None:
            roots.append(root)
    return roots


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        _LOG.debug("failed to remove staging tree %s", path, exc_info=True)


def _cleanup_dynamic_stage_roots(roots: list[Path]) -> None:
    for root in roots:
        _remove_tree(root)


def _execute_mmseqs2_parent(
    adapter: ModelAdapter,
    prepared: dict[str, Any],
    context: dict[str, Any] | None,
    *,
    batch_id: int,
    item_index: int,
    bucket_id: str | None,
    params: dict[str, str],
    execute_ctx: Any,
    cancelled: threading.Event | None,
) -> tuple[Any, dict[str, Any]]:
    try:
        if context is None:
            k = logical_n = 0
        else:
            k = int(context["execution_batch_size"])
            logical_n = int(context["logical_batch_size"])
            prepared_k = int(prepared.get("batch_size"))
            if prepared_k != k:
                raise _contract_error(
                    f"MMseqs2 prepared k mismatch: {prepared_k} != {k}"
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error("MMseqs2 prepared batch context is incomplete") from exc
    if prepared.get("dump_dir"):
        raise _contract_error("MMseqs2 dump_dir is not atomic-output compatible")
    raw_workdir = str(prepared.get("workdir") or "")
    raw_fasta = str(prepared.get("fasta_path") or "")
    raw_target = str(prepared.get("output_dir") or "")
    if not raw_workdir or not raw_fasta or not raw_target:
        raise _contract_error("MMseqs2 workdir/FASTA/output path is missing")
    workdir = Path(raw_workdir).resolve()

    def _resolve_in_workdir(raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else workdir / path).resolve()
        if not resolved.is_relative_to(workdir):
            raise _contract_error("MMseqs2 paths must remain under workdir")
        return resolved

    fasta_path = _resolve_in_workdir(raw_fasta)
    target = _resolve_in_workdir(raw_target)
    if not fasta_path.is_file():
        raise _contract_error("MMseqs2 FASTA path is missing")
    records, names = _fasta_records(fasta_path)
    if context is None:
        k = logical_n = len(records)
    elif len(records) != logical_n:
        raise _contract_error(
            f"MMseqs2 actual N mismatch: {len(records)} != {logical_n}"
        )
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise _contract_error("MMseqs2 output destination must be absent or empty")

    root = _mmseqs2_stage_root(prepared, batch_id, item_index)
    if root is None:
        raise _contract_error("MMseqs2 staging path is unavailable")
    _remove_tree(root)
    children = root / "children"
    final = root / "final"
    children.mkdir(parents=True)
    final.mkdir()
    ordered_paths: list[Path] = []
    group_count = 0
    try:
        for group_count, start in enumerate(range(0, logical_n, k), start=1):
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("batch cancelled")
            group_records = records[start : start + k]
            group_names = names[start : start + k]
            group_dir = children / f"{group_count - 1:06d}"
            group_out = group_dir / "out"
            group_dir.mkdir()
            child_fasta = group_dir / "batch.fasta"
            child_fasta.write_bytes(b"".join(group_records))
            child = dict(prepared)
            child["fasta_path"] = str(child_fasta)
            child["output_dir"] = str(group_out)
            child_outputs = adapter.execute_batch(
                [child], bucket_id, params, execute_ctx
            )
            if not isinstance(child_outputs, list) or len(child_outputs) != 1:
                raise _contract_error("MMseqs2 child output length mismatch")
            child_output = child_outputs[0]
            if isinstance(child_output, dict) and child_output.get("error"):
                raise RuntimeError(
                    f"MMSeqs2 execution failed: {child_output.get('error')}"
                )
            if not isinstance(child_output, dict):
                raise _contract_error("MMseqs2 child output must be a mapping")
            raw_paths = child_output.get("a3m_files")
            if not isinstance(raw_paths, list) or len(raw_paths) != len(group_names):
                raise _contract_error("MMseqs2 child A3M cardinality mismatch")
            by_name: dict[str, Path] = {}
            child_output_root = group_out.resolve()
            for raw_path in raw_paths:
                path = Path(str(raw_path)).resolve()
                if (
                    not path.is_relative_to(child_output_root)
                    or not path.is_file()
                    or path.suffix != ".a3m"
                    or path.stat().st_size <= 0
                ):
                    raise _contract_error("MMseqs2 child A3M content is missing")
                if path.name in by_name:
                    raise _contract_error("MMseqs2 child A3M names are not unique")
                by_name[path.name] = path
            expected = [f"{name}.a3m" for name in group_names]
            if set(by_name) != set(expected):
                raise _contract_error(
                    "MMseqs2 child A3M names do not match FASTA order"
                )
            for filename in expected:
                destination = (final / filename).resolve()
                if not destination.is_relative_to(final.resolve()):
                    raise _contract_error("MMseqs2 merged A3M path escaped staging")
                if destination.exists():
                    raise _contract_error("MMseqs2 merged A3M names are not unique")
                shutil.copyfile(by_name[filename], destination)
                ordered_paths.append(destination)
        if len(ordered_paths) != logical_n:
            raise _contract_error("MMseqs2 merged A3M cardinality mismatch")
        os.replace(final, target)
        published = [str(target / path.name) for path in ordered_paths]
        telemetry = (
            {
                "dynamic_batch_argument_applied_k": k,
                "dynamic_batch_consumed_k": k,
                "dynamic_batch_logical_n": logical_n,
                "dynamic_batch_group_count": group_count,
            }
            if context is not None
            else {}
        )
        return {
            "a3m_files": published,
            "output_dir": str(target),
        }, telemetry
    finally:
        _remove_tree(root)


def _execute_adapter_batch(
    adapter: ModelAdapter,
    prepared: list[Any],
    contexts: list[dict[str, Any] | None],
    bucket_id: str | None,
    params: dict[str, str],
    execute_ctx: Any,
    *,
    batch_id: int,
    cancelled: threading.Event | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if len(prepared) != len(contexts):
        raise _contract_error("prepared/context length mismatch")
    is_mmseqs2 = str(adapter.model_name() or "").strip().lower() == "mmseqs2"
    if is_mmseqs2:
        outputs: list[Any] = []
        telemetry: list[dict[str, Any]] = []
        for index, item in enumerate(prepared):
            context = contexts[index]
            if not isinstance(item, dict):
                raise _contract_error("MMseqs2 prepared item must be a mapping")
            output, item_telemetry = _execute_mmseqs2_parent(
                adapter,
                item,
                context,
                batch_id=batch_id,
                item_index=index,
                bucket_id=bucket_id,
                params=params,
                execute_ctx=execute_ctx,
                cancelled=cancelled,
            )
            outputs.append(output)
            telemetry.append(item_telemetry)
        return outputs, telemetry

    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("batch cancelled")
    accepts_cancelled = (
        "cancelled" in inspect.signature(adapter.execute_batch).parameters
    )
    if accepts_cancelled:
        outputs = adapter.execute_batch(
            prepared,
            bucket_id,
            params,
            execute_ctx,
            cancelled=cancelled,
        )
    else:
        outputs = adapter.execute_batch(prepared, bucket_id, params, execute_ctx)
    telemetry = [
        {
            "dynamic_batch_argument_applied_k": context["execution_batch_size"],
            "dynamic_batch_logical_n": context["logical_batch_size"],
        }
        if context is not None
        else {}
        for context in contexts
    ]
    return outputs, telemetry


def _run_execute_actor(
    adapter: ModelAdapter,
    exported_execute_ctx: Any,
    actor_id: int,
    generation: int,
    conn: Any,
) -> None:
    """Import parent-owned state once and serially execute tagged batches."""
    try:
        with suppress(Exception):
            os.setsid()
        execute_ctx = adapter.import_execute_ctx_in_child(exported_execute_ctx)
        conn.send(("ready", actor_id, generation, os.getpid(), os.getpgrp()))
        while True:
            try:
                command = conn.recv()
            except EOFError:
                return
            if command == ("stop", generation):
                return
            if (
                not isinstance(command, tuple)
                or len(command) != 7
                or command[0] != "run"
                or command[1] != generation
            ):
                raise RuntimeError(f"invalid actor command: {command!r}")
            _, _, batch_id, prepared, contexts, bucket_id, params = command
            try:
                outputs, telemetry = _execute_adapter_batch(
                    adapter,
                    prepared,
                    contexts,
                    bucket_id,
                    params,
                    execute_ctx,
                    batch_id=batch_id,
                )
                conn.send(("result", generation, batch_id, "ok", outputs, telemetry))
                del outputs, telemetry
                maybe_gc()
            except BaseException as exc:
                with suppress(Exception):
                    conn.send(
                        (
                            "result",
                            generation,
                            batch_id,
                            "error",
                            f"{type(exc).__name__}: {exc}",
                            traceback.format_exc(),
                            classify_sticky_cuda_error(exc),
                        )
                    )
                return
    except BaseException as exc:
        with suppress(Exception):
            conn.send(
                (
                    "actor_error",
                    generation,
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
            )
    finally:
        with suppress(Exception):
            conn.close()


class WorkerEngine:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        in_queue_size: int = 128,
        prepared_queue_size: int = 128,
        output_queue_size: int = 128,
        prepare_concurrency: int = 2,
        execute_concurrency: int = 1,
        finalize_concurrency: int = 2,
        item_concurrency: int = 8,
        execute_processes: int = 0,
        telemetry_collector: WorkerTelemetryCollector | None = None,
    ):
        self.adapter = adapter
        self.telemetry_collector = telemetry_collector or WorkerTelemetryCollector()

        if execute_concurrency <= 0:
            execute_concurrency = _UNLIMITED_EXECUTE_CAP

        adapter_cap = self.adapter.max_inflight_batches()
        requested_execute_processes = execute_processes
        self._one_shot_execute = requested_execute_processes < 0
        self._effective_execute_concurrency = (
            min(execute_concurrency, adapter_cap)
            if adapter_cap > 0
            else execute_concurrency
        )
        if requested_execute_processes <= 0:
            self.execute_processes = self._effective_execute_concurrency
        else:
            self.execute_processes = requested_execute_processes
            self._effective_execute_concurrency = min(
                self._effective_execute_concurrency,
                self.execute_processes,
            )

        self.q_in: asyncio.Queue[BatchEnvelope] = asyncio.Queue(maxsize=_Q_IN_MAXSIZE)
        effective_prepared_size = min(
            prepared_queue_size,
            max(1, self._effective_execute_concurrency),
        )
        self.q_prepared: asyncio.Queue[PreparedEnvelope] = asyncio.Queue(
            maxsize=effective_prepared_size,
        )
        self.q_outputs: asyncio.Queue[OutputEnvelope] = asyncio.Queue(
            maxsize=output_queue_size
        )

        self.prepare_concurrency = prepare_concurrency
        self.execute_concurrency = execute_concurrency
        self.finalize_concurrency = finalize_concurrency
        self.item_concurrency = item_concurrency

        self.prepare_ctx = None
        self.execute_ctx = None
        self.finalize_ctx = None
        self._execute_process_context: Any = None
        self._exported_execute_ctx: Any = None
        self._execute_actors: dict[int, ExecuteActorHandle] = {}
        self._idle_execute_actors: asyncio.Queue[ExecuteActorHandle] = asyncio.Queue()
        self._active_processes: dict[int, ExecuteActorHandle] = {}
        self._actor_create_lock = asyncio.Lock()
        self._next_actor_id = 1
        self._next_actor_generation = 1
        self._sticky_cuda_failure_class: str | None = None
        self._sticky_cuda_failure_actor_id: int | None = None
        self._sticky_cuda_failure_generation: int | None = None
        self._fatal_restart_signal: FatalActorRestartSignal | None = None
        self._fatal_restart_claimed = False
        self._actor_growth_disabled = False
        self._actor_growth_disabled_reason = ""
        self._actor_spawn_timeout_sec = _PROCESS_START_TIMEOUT_SEC
        self._actor_spawn_timeouts = 0
        self._actor_spawn_tasks: set[asyncio.Task[ExecuteActorHandle]] = set()

        self._tasks: list[asyncio.Task] = []
        self._ready = asyncio.Event()
        self._stopped = False
        self._prepare_inflight = 0
        self._execute_inflight = 0
        self._finalize_inflight = 0

        self._active_batches: dict[int, BatchEnvelope] = {}
        self._dynamic_stage_roots_by_batch: dict[int, list[Path]] = {}

    async def start(self) -> None:
        self._stopped = False
        self._ready.clear()
        _enable_child_subreaper()
        self._tasks = []
        if self.telemetry_collector is not None:
            self.telemetry_collector.begin_lifecycle(
                model_version=self.adapter.model_version()
            )
            await self.telemetry_collector.startup_probe()
            await self.telemetry_collector.record_bootstrap_sample(
                "before_init_prepare"
            )
        self.prepare_ctx = await self._run_blocking(self.adapter.init_prepare)
        if self.telemetry_collector is not None:
            await self.telemetry_collector.record_bootstrap_sample("after_init_prepare")
        self.execute_ctx = await self._run_blocking(self.adapter.init_execute)
        if self.telemetry_collector is not None:
            await self.telemetry_collector.record_bootstrap_sample("after_init_execute")
        self.finalize_ctx = await self._run_blocking(self.adapter.init_finalize)
        self._execute_process_context = _spawn_process_context()
        self._exported_execute_ctx = self.adapter.export_execute_ctx_for_processes(
            self.execute_ctx
        )
        await self._probe_execute_process_backend()
        if self.telemetry_collector is not None:
            await self.telemetry_collector.refresh_resident_baseline()
            await self.telemetry_collector.record_bootstrap_sample(
                "after_init_finalize"
            )
            await self.telemetry_collector.record_bootstrap_sample("ready_quiescent")
            await self.telemetry_collector.finalize_bootstrap_summary()

        eec = self._effective_execute_concurrency
        execute_backend = f"persistent spawn actors (cap={self.execute_processes})"
        _LOG.info(
            "WorkerEngine starting: prepare=%d, execute=%d "
            "(effective=%d, adapter_max=%d), finalize=%d (execute stage: %s)",
            self.prepare_concurrency,
            self.execute_concurrency,
            eec,
            self.adapter.max_inflight_batches(),
            self.finalize_concurrency,
            execute_backend,
        )

        for _ in range(self.prepare_concurrency):
            self._tasks.append(asyncio.create_task(self._prepare_loop()))

        for _ in range(eec):
            self._tasks.append(asyncio.create_task(self._execute_loop()))

        for _ in range(self.finalize_concurrency):
            self._tasks.append(asyncio.create_task(self._finalize_loop()))

        self._ready.set()

    @property
    def supports_cancel_batch(self) -> bool:
        """Persistent actors make every execute batch hard-cancellable."""
        return True

    async def stop(self) -> None:
        self._stopped = True
        self._ready.clear()
        for env in self._active_batches.values():
            env.cancelled.set()
            if not env.fut.done():
                env.fut.cancel()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        actor_handles = list(self._execute_actors.values())
        actor_results = await asyncio.gather(
            *(self._stop_execute_actor(handle) for handle in actor_handles),
            return_exceptions=True,
        )
        unreaped = [handle for handle in actor_handles if not handle.reaped]
        actor_shutdown_error = None
        if unreaped:
            details = ", ".join(
                f"actor={handle.actor_id}/pid={handle.pid}" for handle in unreaped
            )
            actor_shutdown_error = RuntimeError(
                f"execute actor shutdown left unreaped processes: {details}"
            )
        else:
            self._active_processes.clear()
            self._execute_actors.clear()
            while not self._idle_execute_actors.empty():
                with suppress(asyncio.QueueEmpty):
                    self._idle_execute_actors.get_nowait()
            self._execute_process_context = None
            self._exported_execute_ctx = None
        for result in actor_results:
            if isinstance(result, BaseException):
                _LOG.error("execute actor shutdown error: %r", result)
        if self.telemetry_collector is not None:
            await self.telemetry_collector.shutdown()
        if actor_shutdown_error is not None:
            raise actor_shutdown_error

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def stats(self) -> dict[str, int]:
        return {
            "in_queue": self.q_in.qsize(),
            "prepared_queue": self.q_prepared.qsize(),
            "output_queue": self.q_outputs.qsize(),
            "prepare_inflight": self._prepare_inflight,
            "execute_inflight": self._execute_inflight,
            "finalize_inflight": self._finalize_inflight,
            "prepare_concurrency": self.prepare_concurrency,
            "execute_concurrency": self.execute_concurrency,
            "finalize_concurrency": self.finalize_concurrency,
            "item_concurrency": self.item_concurrency,
            "effective_execute_concurrency": self._effective_execute_concurrency,
            "execute_actor_count": len(self._execute_actors),
            "actor_growth_disabled": 1 if self._actor_growth_disabled else 0,
            "actor_spawn_timeouts": self._actor_spawn_timeouts,
        }

    def _is_quiescent(self) -> bool:
        return (
            self.q_in.qsize() == 0
            and self.q_prepared.qsize() == 0
            and self.q_outputs.qsize() == 0
            and self._prepare_inflight == 0
            and self._execute_inflight == 0
            and self._finalize_inflight == 0
        )

    def _schedule_resident_baseline_refresh_if_quiescent(self) -> None:
        if self.telemetry_collector is None:
            return
        if self._is_quiescent():
            self.telemetry_collector.schedule_resident_baseline_refresh()

    async def submit_batch(
        self,
        batch_id: int,
        request_ids: list[str],
        requests: list[dict[str, Any]],
        bucket_id: str | None,
        params: dict[str, str],
        per_request_params: list[dict[str, str]],
        dispatch_attribution: DispatchAttribution | None = None,
    ) -> list[dict[str, Any]]:
        if self._stopped:
            raise RuntimeError("engine is stopped, cannot accept new batches")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        env = BatchEnvelope(
            batch_id=batch_id,
            requests=requests,
            request_ids=request_ids,
            bucket_id=bucket_id if bucket_id else None,
            params=params,
            per_request_params=per_request_params,
            fut=fut,
            timing={"t_submit": _now()},
            dispatch_attribution=dispatch_attribution,
        )
        self._active_batches[batch_id] = env
        await self.q_in.put(env)
        try:
            return await fut
        except asyncio.CancelledError as exc:
            env.cancelled.set()
            if self.execute_processes:
                await asyncio.shield(self._terminate_batch_process(batch_id))
            raise exc
        finally:
            if batch_id not in self._active_processes:
                self._active_batches.pop(batch_id, None)

    async def cancel_batch(self, batch_id: int) -> bool:
        """Cancel one queued or running batch and reap its assigned actor."""
        env = self._active_batches.get(batch_id)
        if env is None:
            return False
        if (
            self.execute_processes
            and env.execute_finished
            and batch_id not in self._active_processes
        ):
            return False
        env.cancelled.set()
        if not await self._terminate_batch_process(batch_id):
            return False
        if not env.fut.done():
            env.fut.cancel()
        return True

    def drain(self) -> None:
        """Enter drain mode: reject new batches but let in-flight work finish.

        The gateway calls this before eviction to gracefully stop accepting
        new tasks while allowing running tasks to complete.
        """
        self._stopped = True
        self._ready.clear()

    def is_draining(self) -> bool:
        return self._stopped and bool(self._active_batches)

    async def _prepare_loop(self) -> None:
        while not self._stopped:
            env = await self.q_in.get()
            env.timing["t_prepare_deq"] = _now()
            self._prepare_inflight += 1
            try:
                prepared, errors, contexts = await self._prepare(env.requests)
                env.timing["t_prepare_done"] = _now()
                await self.q_prepared.put(
                    PreparedEnvelope(
                        env=env,
                        prepared=prepared,
                        errors=errors,
                        dynamic_batch_contexts=contexts,
                    )
                )
            except Exception as exc:
                if not env.fut.done():
                    env.fut.set_exception(exc)
            finally:
                self._prepare_inflight -= 1
                self.q_in.task_done()
                self._schedule_resident_baseline_refresh_if_quiescent()

    async def _execute_loop(self) -> None:
        while not self._stopped:
            item = await self.q_prepared.get()
            env = item.env
            env.timing["t_execute_deq"] = _now()
            self._execute_inflight += 1
            try:
                if env.cancelled.is_set():
                    if not env.fut.done():
                        env.fut.set_exception(RuntimeError("batch cancelled"))
                    continue
                env.execute_started = True
                with _worker_execute_marker():
                    outputs, errors, dynamic_telemetry = await self._execute(
                        batch_id=env.batch_id,
                        prepared=item.prepared,
                        errors=item.errors,
                        dynamic_batch_contexts=item.dynamic_batch_contexts,
                        bucket_id=env.bucket_id,
                        batch_params=env.params,
                        per_request_params=env.per_request_params,
                        cancelled=env.cancelled,
                    )
                if env.cancelled.is_set():
                    if not env.fut.done():
                        env.fut.set_exception(RuntimeError("batch cancelled"))
                    continue
                env.timing["t_execute_done"] = _now()
                env.execute_finished = True
                await self.q_outputs.put(
                    OutputEnvelope(
                        env=env,
                        outputs=outputs,
                        errors=errors,
                        dynamic_batch_telemetry=dynamic_telemetry,
                    )
                )
            except Exception as exc:
                if not env.fut.done():
                    env.fut.set_exception(exc)
            finally:
                self._execute_inflight -= 1
                self.q_prepared.task_done()
                maybe_gc()
                self._schedule_resident_baseline_refresh_if_quiescent()

    async def _finalize_loop(self) -> None:
        while not self._stopped:
            item = await self.q_outputs.get()
            env = item.env
            env.timing["t_finalize_deq"] = _now()
            self._finalize_inflight += 1
            responses: list[dict[str, Any]] | None = None
            finalize_error: Exception | None = None
            try:
                responses = await self._finalize(
                    request_ids=env.request_ids,
                    outputs=item.outputs,
                    errors=item.errors,
                )
                if len(responses) != len(item.dynamic_batch_telemetry):
                    raise _contract_error("response/telemetry length mismatch")
                for index, response in enumerate(responses):
                    telemetry = item.dynamic_batch_telemetry[index]
                    if telemetry:
                        response["_dynamic_batch_telemetry"] = dict(telemetry)
                env.timing["t_finalize_done"] = _now()
            except Exception as exc:
                finalize_error = exc
            finally:
                if responses is not None:
                    batch_timing = self._compute_batch_timing(env.timing)
                    for response in responses:
                        response["_batch_timing_us"] = batch_timing
                        synthetic_memory = response.get("_synthetic_memory_telemetry")
                        if not isinstance(synthetic_memory, dict):
                            result_payload = response.get("result")
                            if isinstance(result_payload, dict):
                                synthetic_memory = result_payload.get(
                                    "_synthetic_memory_telemetry"
                                )
                        if isinstance(synthetic_memory, dict):
                            response["_memory_telemetry"] = dict(synthetic_memory)
                            memory_observation = {}
                            dispatch_window = synthetic_memory.get(
                                "dispatch_memory_window"
                            )
                            bootstrap_summary = synthetic_memory.get(
                                "bootstrap_memory_summary"
                            )
                            if isinstance(dispatch_window, dict):
                                memory_observation["dispatch_memory_window"] = (
                                    dispatch_window
                                )
                            if isinstance(bootstrap_summary, dict):
                                memory_observation["bootstrap_memory_summary"] = (
                                    bootstrap_summary
                                )
                            response["_memory_observation"] = memory_observation
                        elif env.memory_telemetry is not None:
                            response["_memory_telemetry"] = (
                                env.memory_telemetry.as_payload()
                            )
                            response["_memory_observation"] = {
                                "bootstrap_memory_summary": (
                                    env.memory_telemetry.bootstrap_memory_summary
                                ),
                                "dispatch_memory_window": (
                                    env.memory_telemetry.dispatch_memory_window
                                ),
                            }
                    if not env.fut.done():
                        env.fut.set_result(responses)
                elif finalize_error is not None and not env.fut.done():
                    env.fut.set_exception(finalize_error)
                self._finalize_inflight -= 1
                self.q_outputs.task_done()
                self._schedule_resident_baseline_refresh_if_quiescent()

    def _compute_batch_timing(self, t: dict[str, float]) -> dict[str, int]:
        queue_delay = _us(t.get("t_prepare_deq", t["t_submit"]) - t["t_submit"])
        prepare_us = _us(
            t.get("t_prepare_done", t.get("t_prepare_deq", t["t_submit"]))
            - t.get("t_prepare_deq", t["t_submit"])
        )
        execute_us = _us(
            t.get("t_execute_done", t.get("t_execute_deq", t["t_submit"]))
            - t.get("t_execute_deq", t["t_submit"])
        )
        finalize_us = _us(
            t.get("t_finalize_done", t.get("t_finalize_deq", t["t_submit"]))
            - t.get("t_finalize_deq", t["t_submit"])
        )
        total_us = _us(t.get("t_finalize_done", t["t_submit"]) - t["t_submit"])
        return {
            "queue_delay_us": queue_delay,
            "prepare_us": prepare_us,
            "execute_us": execute_us,
            "finalize_us": finalize_us,
            "total_us": total_us,
        }

    async def _prepare(
        self, requests: list[dict[str, Any]]
    ) -> tuple[
        list[Any | None],
        list[str | None],
        list[dict[str, Any] | None],
    ]:
        rewritten: list[dict[str, Any]] = []
        contexts: list[dict[str, Any] | None] = []
        rewrite_errors: list[str | None] = []
        for request in requests:
            try:
                copied, context = _rewrite_dynamic_batch_request(request)
                rewritten.append(copied)
                contexts.append(context)
                rewrite_errors.append(None)
            except Exception as exc:
                rewritten.append(request)
                contexts.append(None)
                rewrite_errors.append(str(exc))

        if not any(rewrite_errors):
            prepared = await self._run_blocking(
                self.adapter.prepare_batch, rewritten, self.prepare_ctx
            )
            if prepared is not None:
                if len(prepared) != len(requests):
                    raise RuntimeError("prepare_batch length mismatch")
                return list(prepared), [None] * len(requests), contexts

        sem = asyncio.Semaphore(self.item_concurrency)

        async def one(
            i: int, req: dict[str, Any]
        ) -> tuple[int, Any | None, str | None]:
            if rewrite_errors[i] is not None:
                return i, None, rewrite_errors[i]
            async with sem:
                try:
                    prepared_item = await self._run_blocking(
                        self.adapter.prepare_one, req, self.prepare_ctx
                    )
                    return i, prepared_item, None
                except Exception as exc:
                    return i, None, str(exc)

        triples = await asyncio.gather(
            *[one(i, req) for i, req in enumerate(rewritten)]
        )
        out_p: list[Any | None] = [None] * len(requests)
        out_e: list[str | None] = [None] * len(requests)
        for i, prepared_item, err in triples:
            out_p[i] = prepared_item
            out_e[i] = err
        return out_p, out_e, contexts

    async def _execute(
        self,
        batch_id: int,
        prepared: list[Any | None],
        errors: list[str | None],
        dynamic_batch_contexts: list[dict[str, Any] | None],
        bucket_id: str | None,
        batch_params: dict[str, str],
        per_request_params: list[dict[str, str]],
        cancelled: threading.Event | None = None,
    ) -> tuple[
        list[Any | None],
        list[str | None],
        list[dict[str, Any]],
    ]:
        idx = [
            i
            for i, prepared_item in enumerate(prepared)
            if prepared_item is not None and errors[i] is None
        ]
        empty_telemetry: list[dict[str, Any]] = [{} for _ in prepared]
        if not idx:
            return [None] * len(prepared), list(errors), empty_telemetry

        subset_prepared = [prepared[i] for i in idx]
        subset_contexts = [dynamic_batch_contexts[i] for i in idx]
        merged_params = batch_params

        try:
            outputs_subset, telemetry_subset = await self._run_execute_process(
                batch_id,
                subset_prepared,
                subset_contexts,
                bucket_id,
                merged_params,
            )
        except Exception as exc:
            new_errors = list(errors)
            message = str(exc)
            marker_index = message.find(_DYNAMIC_BATCH_ERROR)
            rendered = (
                message[marker_index:]
                if marker_index >= 0
                else f"execute_batch failed: {exc}"
            )
            for i in idx:
                new_errors[i] = rendered
            return [None] * len(prepared), new_errors, empty_telemetry

        if len(outputs_subset) != len(idx) or len(telemetry_subset) != len(idx):
            new_errors = list(errors)
            for i in idx:
                new_errors[i] = "execute_batch output length mismatch"
            return [None] * len(prepared), new_errors, empty_telemetry

        outputs: list[Any | None] = [None] * len(prepared)
        telemetry = list(empty_telemetry)
        for j, i in enumerate(idx):
            outputs[i] = outputs_subset[j]
            telemetry[i] = dict(telemetry_subset[j])

        return outputs, list(errors), telemetry

    async def _finalize(
        self,
        request_ids: list[str],
        outputs: list[Any | None],
        errors: list[str | None],
    ) -> list[dict[str, Any]]:
        idx = [
            i for i, out in enumerate(outputs) if out is not None and errors[i] is None
        ]
        responses: list[dict[str, Any]] = [
            {
                "request_id": request_ids[i],
                "ok": False,
                "result": None,
                "error_message": errors[i] or "unknown error",
            }
            for i in range(len(outputs))
        ]

        if idx:
            subset_outputs = [outputs[i] for i in idx]
            batch_resp = await self._run_blocking(
                self.adapter.finalize_batch, subset_outputs, self.finalize_ctx
            )

            if batch_resp is not None:
                if len(batch_resp) != len(idx):
                    for i in idx:
                        responses[i] = {
                            "request_id": request_ids[i],
                            "ok": False,
                            "result": None,
                            "error_message": "finalize_batch length mismatch",
                        }
                    return responses

                for j, i in enumerate(idx):
                    responses[i] = {
                        "request_id": request_ids[i],
                        "ok": True,
                        "result": batch_resp[j],
                        "error_message": "",
                    }
                return responses

            sem = asyncio.Semaphore(self.item_concurrency)

            async def one(i: int, out: Any) -> tuple[int, bool, Any, str]:
                async with sem:
                    try:
                        result = await self._run_blocking(
                            self.adapter.finalize_one, out, self.finalize_ctx
                        )
                        synthetic_error = ""
                        if isinstance(result, dict):
                            synthetic_error = str(
                                result.get("_synthetic_item_error")
                                or result.get("_synthetic_error_message")
                                or ""
                            ).strip()
                        if synthetic_error:
                            return i, False, result, synthetic_error
                        return i, True, result, ""
                    except Exception as exc:
                        return i, False, None, str(exc)

            triples = await asyncio.gather(*[one(i, outputs[i]) for i in idx])
            for i, ok, result, err in triples:
                responses[i] = {
                    "request_id": request_ids[i],
                    "ok": ok,
                    "result": result,
                    "error_message": err,
                }

        return responses

    async def _run_blocking(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args))

    def _disable_actor_growth(self, reason: str) -> None:
        if self._actor_growth_disabled:
            return
        self._actor_growth_disabled = True
        self._actor_growth_disabled_reason = str(reason)
        _LOG.error(
            "[execute-actor] dynamic growth disabled; existing actors remain "
            "available: %s",
            reason,
        )

    def _arm_actor_pool_restart(self, batch_id: int) -> None:
        if self._fatal_restart_signal is not None:
            return
        detail = self._actor_growth_disabled_reason or "execute actor pool unavailable"
        self._fatal_restart_signal = FatalActorRestartSignal(
            error_class="actor_pool_unavailable",
            first_actor_id=0,
            first_generation=0,
            second_actor_id=0,
            second_generation=0,
            second_batch_id=batch_id,
            detail=detail,
        )

    def _on_late_actor_spawn(
        self,
        task: asyncio.Task[ExecuteActorHandle],
    ) -> None:
        self._actor_spawn_tasks.discard(task)
        try:
            handle = task.result()
        except BaseException as exc:
            _LOG.warning(
                "[execute-actor] abandoned spawn finished with %s: %s",
                type(exc).__name__,
                exc,
            )
            return
        if not self._actor_growth_disabled and not self._stopped:
            return
        try:
            asyncio.create_task(self._recover_late_actor_spawn(handle))
        except RuntimeError:
            self._kill_process_handle(handle)

    async def _recover_late_actor_spawn(
        self,
        handle: ExecuteActorHandle,
    ) -> None:
        if self._stopped or self._fatal_restart_signal is not None:
            await self._terminate_process_handle(handle)
            return
        try:
            await self._read_process_ready(handle)
        except BaseException as exc:
            _LOG.warning(
                "[execute-actor] abandoned spawn failed readiness actor=%s "
                "generation=%s pid=%s error=%s: %s",
                handle.actor_id,
                handle.generation,
                handle.pid,
                type(exc).__name__,
                exc,
            )
            await self._terminate_process_handle(handle)
            return
        if self._stopped or self._fatal_restart_signal is not None:
            await self._terminate_process_handle(handle)
            return
        self._actor_growth_disabled = False
        self._actor_growth_disabled_reason = ""
        _LOG.warning(
            "[execute-actor] retained ready actor from late spawn actor=%s "
            "generation=%s pid=%s decision=resume_growth",
            handle.actor_id,
            handle.generation,
            handle.pid,
        )
        self._release_execute_actor(handle)

    def _spawn_execute_actor_in_thread(self) -> ExecuteActorHandle:
        """Give spawn serialization the event-loop context some adapters need."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return self._spawn_execute_actor()
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    async def _spawn_execute_actor_nonblocking(
        self,
        batch_id: int,
    ) -> ExecuteActorHandle | None:
        """Spawn without blocking gRPC; timeout degrades to existing actors."""
        spawn_task = asyncio.create_task(
            asyncio.to_thread(self._spawn_execute_actor_in_thread),
            name=f"execute-actor-spawn:{batch_id}",
        )
        self._actor_spawn_tasks.add(spawn_task)
        spawn_task.add_done_callback(self._on_late_actor_spawn)
        try:
            return await asyncio.wait_for(
                asyncio.shield(spawn_task),
                timeout=self._actor_spawn_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            self._actor_spawn_timeouts += 1
            self._disable_actor_growth(
                f"actor spawn exceeded {self._actor_spawn_timeout_sec:.1f}s "
                f"for batch {batch_id} ({type(exc).__name__})"
            )
            return None

    def _spawn_execute_actor(self) -> ExecuteActorHandle:
        if self._execute_process_context is None:
            raise RuntimeError("execute actor backend is not initialized")
        actor_id = self._next_actor_id
        generation = self._next_actor_generation
        self._next_actor_id += 1
        self._next_actor_generation += 1
        parent_conn, child_conn = self._execute_process_context.Pipe(duplex=True)
        process = self._execute_process_context.Process(
            target=_run_execute_actor,
            args=(
                self.adapter,
                self._exported_execute_ctx,
                actor_id,
                generation,
                child_conn,
            ),
            name=f"modelworker-actor-{actor_id}-g{generation}",
            daemon=False,
        )
        handle = ExecuteActorHandle(
            actor_id=actor_id,
            generation=generation,
            process=process,
            conn=parent_conn,
            started_at=_now(),
        )
        try:
            process.start()
            handle.pid = process.pid
            handle.process_group_id = process.pid
        except BaseException:
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()
        self._execute_actors[actor_id] = handle
        _LOG.info(
            "[execute-actor] spawned actor=%s generation=%s pid=%s",
            actor_id,
            generation,
            handle.pid,
        )
        return handle

    @staticmethod
    async def _recv_process_message(conn: Any, *, timeout: float | None = None) -> Any:
        recv = asyncio.to_thread(conn.recv)
        return await asyncio.wait_for(recv, timeout=timeout) if timeout else await recv

    @staticmethod
    def _kill_process_handle(handle: ExecuteActorHandle) -> None:
        process = handle.process
        killed_group = False
        if handle.process_group_id and handle.process_group_id == handle.pid:
            try:
                os.killpg(handle.process_group_id, signal.SIGKILL)
                killed_group = True
            except (ProcessLookupError, PermissionError):
                _LOG.debug("execute actor process group already exited", exc_info=True)
        if not killed_group and process.is_alive():
            with suppress(ProcessLookupError):
                process.kill()

    @staticmethod
    def _process_group_alive(handle: ExecuteActorHandle) -> bool:
        if not handle.process_group_id or handle.process_group_id != handle.pid:
            return False
        try:
            os.killpg(handle.process_group_id, 0)
        except (ProcessLookupError, PermissionError) as exc:
            return isinstance(exc, PermissionError)
        return True

    @staticmethod
    def _reap_process_group_children(handle: ExecuteActorHandle) -> int:
        """Reap orphaned native descendants adopted from a killed actor."""
        if not handle.process_group_id or handle.process_group_id != handle.pid:
            return 0
        reaped = 0
        while True:
            try:
                pid, status = os.waitpid(-handle.process_group_id, os.WNOHANG)
            except InterruptedError:
                continue
            except ChildProcessError:
                break
            if pid <= 0:
                break
            reaped += 1
            _LOG.info(
                "[execute-actor] reaped descendant actor=%s generation=%s "
                "pid=%s status=%s",
                handle.actor_id,
                handle.generation,
                pid,
                status,
            )
        return reaped

    async def _reap_process_handle(
        self,
        handle: ExecuteActorHandle,
        *,
        terminate: bool,
    ) -> bool:
        if terminate:
            handle.cancel_requested = True
        async with handle.lock:
            if handle.reaped:
                return True
            natural_deadline = time.monotonic() + _PROCESS_NATURAL_EXIT_TIMEOUT_SEC
            cancel_deadline = (
                time.monotonic() + _PROCESS_REAP_TIMEOUT_SEC
                if handle.cancel_requested
                else None
            )
            while handle.process.is_alive():
                if handle.cancel_requested:
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + _PROCESS_REAP_TIMEOUT_SEC
                    self._kill_process_handle(handle)
                    deadline = cancel_deadline
                else:
                    deadline = natural_deadline
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_process_handle(handle)
                    await asyncio.to_thread(
                        handle.process.join,
                        _PROCESS_JOIN_TIMEOUT_SEC,
                    )
                    break
                await asyncio.to_thread(
                    handle.process.join,
                    min(_PROCESS_REAP_POLL_SEC, remaining),
                )
            if handle.process.is_alive():
                return False
            group_deadline = cancel_deadline or (
                time.monotonic() + _PROCESS_REAP_TIMEOUT_SEC
            )
            while self._process_group_alive(handle):
                self._kill_process_handle(handle)
                self._reap_process_group_children(handle)
                if time.monotonic() >= group_deadline:
                    return False
                await asyncio.sleep(0.01)
            self._reap_process_group_children(handle)
            handle.exitcode = handle.process.exitcode
            handle.reaped = True
            with suppress(Exception):
                handle.conn.close()
            if handle.batch_id is not None:
                current = self._active_processes.get(handle.batch_id)
                if current is handle:
                    self._active_processes.pop(handle.batch_id, None)
            self._execute_actors.pop(handle.actor_id, None)
            if self.telemetry_collector is not None:
                self.telemetry_collector.forget_actor(
                    handle.actor_id, handle.generation
                )
            _LOG.info(
                "[execute-actor] reaped actor=%s generation=%s batch=%s "
                "pid=%s exit=%s cancelled=%s",
                handle.actor_id,
                handle.generation,
                handle.batch_id,
                handle.pid,
                handle.exitcode,
                handle.cancel_requested,
            )
            with suppress(Exception):
                handle.process.close()
            return True

    async def _terminate_process_handle(self, handle: ExecuteActorHandle) -> bool:
        handle.cancel_requested = True
        _LOG.info(
            "[execute-actor] cancelling actor=%s generation=%s batch=%s pid=%s pgid=%s",
            handle.actor_id,
            handle.generation,
            handle.batch_id,
            handle.pid,
            handle.process_group_id,
        )
        return await self._reap_process_handle(handle, terminate=True)

    async def _stop_execute_actor(self, handle: ExecuteActorHandle) -> bool:
        if handle.reaped:
            return True
        if handle.batch_id is not None:
            return await self._terminate_process_handle(handle)
        try:
            await asyncio.to_thread(
                handle.conn.send,
                ("stop", handle.generation),
            )
        except (BrokenPipeError, EOFError, OSError):
            return await self._terminate_process_handle(handle)
        stopped = await self._reap_process_handle(handle, terminate=False)
        return stopped or await self._terminate_process_handle(handle)

    async def _terminate_batch_process(self, batch_id: int) -> bool:
        handle = self._active_processes.get(batch_id)
        if handle is None:
            return True
        terminated = await self._terminate_process_handle(handle)
        _cleanup_dynamic_stage_roots(
            self._dynamic_stage_roots_by_batch.pop(batch_id, [])
        )
        return terminated

    async def _read_process_ready(self, handle: ExecuteActorHandle) -> None:
        message = await self._recv_process_message(
            handle.conn,
            timeout=_PROCESS_PROBE_TIMEOUT_SEC,
        )
        if (
            not isinstance(message, tuple)
            or len(message) != 5
            or message[:3] != ("ready", handle.actor_id, handle.generation)
        ):
            raise RuntimeError(
                f"execute actor {handle.actor_id} did not report ready: {message!r}"
            )
        pid = message[3]
        pgid = message[4]
        if pid is None or pgid is None:
            raise RuntimeError(
                f"execute actor {handle.actor_id} reported invalid process IDs"
            )
        try:
            handle.pid = int(pid)
            handle.process_group_id = int(pgid)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"execute actor {handle.actor_id} reported invalid process IDs"
            ) from exc
        handle.ready_at = _now()
        _LOG.info(
            "[execute-actor] ready actor=%s generation=%s pid=%s pgid=%s "
            "spawn_sec=%.6f",
            handle.actor_id,
            handle.generation,
            handle.pid,
            handle.process_group_id,
            handle.ready_at - handle.started_at,
        )

    def _assign_execute_actor(
        self,
        handle: ExecuteActorHandle,
        batch_id: int,
    ) -> None:
        if handle.reaped or handle.batch_id is not None:
            raise RuntimeError(f"execute actor {handle.actor_id} is not idle")
        if batch_id in self._active_processes:
            raise RuntimeError(f"execute actor already active for batch {batch_id}")
        handle.batch_id = batch_id
        handle.cancel_requested = False
        handle.assignments += 1
        self._active_processes[batch_id] = handle
        _LOG.info(
            "[execute-actor] assigned actor=%s generation=%s batch=%s "
            "pid=%s assignment=%s",
            handle.actor_id,
            handle.generation,
            batch_id,
            handle.pid,
            handle.assignments,
        )

    def _record_execute_actor_success(
        self,
        handle: ExecuteActorHandle,
        batch_id: int,
    ) -> None:
        if (
            self._fatal_restart_signal is not None
            or self._sticky_cuda_failure_class is None
        ):
            return
        _LOG.info(
            "[execute-actor] fatal_cuda_reset actor=%s generation=%s batch=%s "
            "previous_class=%s previous_generation=%s decision=reset_after_success",
            handle.actor_id,
            handle.generation,
            batch_id,
            self._sticky_cuda_failure_class,
            self._sticky_cuda_failure_generation,
        )
        self._sticky_cuda_failure_class = None
        self._sticky_cuda_failure_actor_id = None
        self._sticky_cuda_failure_generation = None

    def _record_sticky_cuda_failure(
        self,
        handle: ExecuteActorHandle,
        batch_id: int,
        error_class: str,
        detail: str,
    ) -> None:
        if error_class == "mps_rpc_failure" and self._fatal_restart_signal is None:
            self._fatal_restart_signal = FatalActorRestartSignal(
                error_class=error_class,
                first_actor_id=handle.actor_id,
                first_generation=handle.generation,
                second_actor_id=handle.actor_id,
                second_generation=handle.generation,
                second_batch_id=batch_id,
                detail=detail,
            )
            _LOG.error(
                "[execute-actor] fatal_cuda actor=%s generation=%s batch=%s "
                "class=%s decision=container_fallback",
                handle.actor_id,
                handle.generation,
                batch_id,
                error_class,
            )
            return

        first_generation = self._sticky_cuda_failure_generation
        if (
            self._fatal_restart_signal is None
            and self._sticky_cuda_failure_class == error_class
            and first_generation is not None
            and first_generation != handle.generation
        ):
            first_actor_id = self._sticky_cuda_failure_actor_id
            if first_actor_id is None:
                raise RuntimeError("sticky CUDA failure actor state is inconsistent")
            self._fatal_restart_signal = FatalActorRestartSignal(
                error_class=error_class,
                first_actor_id=first_actor_id,
                first_generation=first_generation,
                second_actor_id=handle.actor_id,
                second_generation=handle.generation,
                second_batch_id=batch_id,
                detail=detail,
            )
            _LOG.error(
                "[execute-actor] fatal_cuda actor=%s generation=%s batch=%s "
                "class=%s first_actor=%s first_generation=%s "
                "decision=container_fallback",
                handle.actor_id,
                handle.generation,
                batch_id,
                error_class,
                first_actor_id,
                first_generation,
            )
            return

        if self._fatal_restart_signal is None:
            self._sticky_cuda_failure_class = error_class
            self._sticky_cuda_failure_actor_id = handle.actor_id
            self._sticky_cuda_failure_generation = handle.generation
        _LOG.error(
            "[execute-actor] fatal_cuda actor=%s generation=%s batch=%s "
            "class=%s decision=actor_recycle",
            handle.actor_id,
            handle.generation,
            batch_id,
            error_class,
        )

    def claim_fatal_restart_signal(
        self,
        batch_id: int,
    ) -> FatalActorRestartSignal | None:
        signal = self._fatal_restart_signal
        if (
            signal is None
            or signal.second_batch_id != batch_id
            or self._fatal_restart_claimed
        ):
            return None
        self._fatal_restart_claimed = True
        return signal

    def _release_execute_actor(self, handle: ExecuteActorHandle) -> None:
        batch_id = handle.batch_id
        if batch_id is not None and self._active_processes.get(batch_id) is handle:
            self._active_processes.pop(batch_id, None)
        handle.batch_id = None
        if (
            not handle.reaped
            and not handle.cancel_requested
            and handle.process.is_alive()
            and self._execute_actors.get(handle.actor_id) is handle
        ):
            self._idle_execute_actors.put_nowait(handle)
            _LOG.info(
                "[execute-actor] idle actor=%s generation=%s pid=%s assignments=%s",
                handle.actor_id,
                handle.generation,
                handle.pid,
                handle.assignments,
            )

    async def _take_idle_execute_actor(
        self,
        *,
        wait: bool,
        cancelled: threading.Event,
    ) -> ExecuteActorHandle | None:
        while True:
            if cancelled.is_set():
                raise RuntimeError("batch cancelled")
            try:
                if wait:
                    handle = await asyncio.wait_for(
                        self._idle_execute_actors.get(),
                        timeout=_PROCESS_REAP_POLL_SEC,
                    )
                else:
                    handle = self._idle_execute_actors.get_nowait()
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                return None
            usable = (
                not handle.reaped
                and handle.batch_id is None
                and handle.process.is_alive()
                and self._execute_actors.get(handle.actor_id) is handle
            )
            if cancelled.is_set():
                if usable:
                    self._idle_execute_actors.put_nowait(handle)
                elif not handle.reaped:
                    await self._reap_process_handle(handle, terminate=False)
                raise RuntimeError("batch cancelled")
            if usable:
                return handle
            if not handle.reaped:
                await self._reap_process_handle(handle, terminate=False)
            if wait:
                return None

    async def _acquire_execute_actor(
        self,
        batch_id: int,
        cancelled: threading.Event,
    ) -> ExecuteActorHandle:
        while not cancelled.is_set():
            handle = await self._take_idle_execute_actor(
                wait=False,
                cancelled=cancelled,
            )
            if handle is not None:
                self._assign_execute_actor(handle, batch_id)
                return handle
            if self._actor_growth_disabled and not self._execute_actors:
                self._arm_actor_pool_restart(batch_id)
                raise RuntimeError(self._actor_growth_disabled_reason)
            async with self._actor_create_lock:
                handle = await self._take_idle_execute_actor(
                    wait=False,
                    cancelled=cancelled,
                )
                if handle is not None:
                    self._assign_execute_actor(handle, batch_id)
                    return handle
                if (
                    not self._actor_growth_disabled
                    and len(self._execute_actors) < self.execute_processes
                ):
                    try:
                        handle = await self._spawn_execute_actor_nonblocking(batch_id)
                    except Exception as exc:
                        if self._execute_actors:
                            self._disable_actor_growth(
                                "actor process.start failed for batch "
                                f"{batch_id}: {exc}"
                            )
                            raise RuntimeError(
                                self._actor_growth_disabled_reason
                            ) from exc
                        self._disable_actor_growth(
                            "last actor process.start failed for batch "
                            f"{batch_id}: {exc}"
                        )
                        self._arm_actor_pool_restart(batch_id)
                        raise RuntimeError(self._actor_growth_disabled_reason) from exc
                    if handle is None:
                        raise RuntimeError(self._actor_growth_disabled_reason)
                    self._assign_execute_actor(handle, batch_id)
                    try:
                        await self._read_process_ready(handle)
                    except asyncio.CancelledError as exc:
                        await asyncio.shield(self._terminate_process_handle(handle))
                        raise exc
                    except Exception as exc:
                        await asyncio.shield(self._terminate_process_handle(handle))
                        if cancelled.is_set():
                            raise RuntimeError("batch cancelled") from exc
                        if self._execute_actors:
                            self._disable_actor_growth(
                                f"actor spawn failed for batch {batch_id}: {exc}"
                            )
                            raise RuntimeError(
                                self._actor_growth_disabled_reason
                            ) from exc
                        self._disable_actor_growth(
                            f"last actor spawn failed for batch {batch_id}: {exc}"
                        )
                        self._arm_actor_pool_restart(batch_id)
                        raise RuntimeError(self._actor_growth_disabled_reason) from exc
                    if cancelled.is_set():
                        await self._terminate_process_handle(handle)
                        raise RuntimeError("batch cancelled")
                    return handle
            handle = await self._take_idle_execute_actor(
                wait=True,
                cancelled=cancelled,
            )
            if handle is not None:
                self._assign_execute_actor(handle, batch_id)
                return handle
        raise RuntimeError("batch cancelled")

    async def _probe_execute_process_backend(self) -> None:
        handle = self._spawn_execute_actor()
        try:
            await self._read_process_ready(handle)
            self._release_execute_actor(handle)
        except BaseException:
            if not handle.reaped:
                await self._terminate_process_handle(handle)
            raise

    async def _run_execute_process(
        self,
        batch_id: int,
        prepared: list[Any],
        dynamic_batch_contexts: list[dict[str, Any] | None],
        bucket_id: str | None,
        params: dict[str, str],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        env = self._active_batches.get(batch_id)
        if env is None:
            raise RuntimeError(f"active batch {batch_id} not found")
        handle = await self._acquire_execute_actor(batch_id, env.cancelled)
        roots = _mmseqs2_stage_roots(
            self.adapter,
            prepared,
            dynamic_batch_contexts,
            batch_id,
        )
        self._dynamic_stage_roots_by_batch[batch_id] = roots
        return_to_idle = False
        try:
            if handle.pid is None:
                raise RuntimeError(f"execute actor {handle.actor_id} has no ready PID")
            if self.telemetry_collector is not None:
                env.telemetry_state = await self.telemetry_collector.begin_dispatch(
                    attribution=env.dispatch_attribution,
                    effective_dispatch_batch_size=len(env.requests),
                    dispatch_group_size=len(env.requests),
                    root_pid=handle.pid,
                    memory_attribution=MEMORY_ATTRIBUTION_REQUEST_PROCESS_TREE,
                    actor_id=handle.actor_id,
                    actor_generation=handle.generation,
                    actor_resident_owner="actor_process_tree",
                )
            await asyncio.to_thread(
                handle.conn.send,
                (
                    "run",
                    handle.generation,
                    batch_id,
                    prepared,
                    dynamic_batch_contexts,
                    bucket_id,
                    params,
                ),
            )
            message = await self._recv_process_message(handle.conn)
            if handle.cancel_requested or env.cancelled.is_set():
                raise RuntimeError("batch cancelled")
            if (
                not isinstance(message, tuple)
                or len(message) < 5
                or message[:3] != ("result", handle.generation, batch_id)
            ):
                raise RuntimeError(
                    f"invalid execute actor result for batch {batch_id}: {message!r}"
                )
            if message[3] == "ok" and len(message) == 6:
                env.execute_finished = True
                outputs = message[4]
                telemetry = message[5]
                self._record_execute_actor_success(handle, batch_id)
                return_to_idle = not self._one_shot_execute
                return outputs, telemetry
            if message[3] == "error" and len(message) >= 6:
                detail = str(message[4])
                trace = str(message[5])
                sticky_cuda_class = (
                    str(message[6])
                    if len(message) >= 7 and message[6] is not None
                    else None
                )
                if trace:
                    _LOG.error(
                        "execute actor=%s generation=%s batch=%s failed:\n%s",
                        handle.actor_id,
                        handle.generation,
                        batch_id,
                        trace,
                    )
                await self._reap_process_handle(handle, terminate=False)
                if sticky_cuda_class is not None:
                    self._record_sticky_cuda_failure(
                        handle,
                        batch_id,
                        sticky_cuda_class,
                        detail,
                    )
                raise RuntimeError(detail)
            raise RuntimeError(
                f"invalid execute actor result for batch {batch_id}: {message!r}"
            )
        except asyncio.CancelledError as exc:
            await asyncio.shield(self._terminate_process_handle(handle))
            raise exc
        except (EOFError, BrokenPipeError, OSError) as exc:
            await self._reap_process_handle(handle, terminate=False)
            raise RuntimeError(
                f"execute actor {handle.actor_id} exited before returning batch "
                f"{batch_id} (exit={handle.exitcode})"
            ) from exc
        finally:
            try:
                if (
                    self.telemetry_collector is not None
                    and env.telemetry_state is not None
                ):
                    env.memory_telemetry = await self.telemetry_collector.end_dispatch(
                        env.telemetry_state
                    )
            finally:
                env.telemetry_state = None
                if return_to_idle and not handle.reaped:
                    self._release_execute_actor(handle)
                elif not handle.reaped:
                    await asyncio.shield(self._terminate_process_handle(handle))
                _cleanup_dynamic_stage_roots(
                    self._dynamic_stage_roots_by_batch.pop(batch_id, [])
                )
