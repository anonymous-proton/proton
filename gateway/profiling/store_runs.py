"""Run-table helpers for profiling index store."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..gateway_identity import (
    UNKNOWN_GATEWAY_VALUE,
    normalize_gateway_git_commit,
    normalize_gateway_instance_id,
)
from .constants import matches_model_input_filter
from .index_models import RunQuery, RunRecord
from .run_identity import canonicalize_axes
from .runtime_regime import DEFAULT_ACTIVE_MEMORY_SCOPE, build_query_context, normalize_query_context

QC_STATUS_SOFT_DROP = "soft_drop"
LEGACY_CELL_SCHEMA_ID = "legacy_v0"
DEFAULT_CELL_SCHEMA_VERSION = 1
CELL_SCHEMA_ALGO_VERSION = "v1"

ALLOWED_SORT_COLUMNS = {
    "run_key",
    "run_source",
    "source_event_id",
    "cell_schema_id",
    "cell_key",
    "run_id",
    "campaign_id",
    "run_name",
    "submitter",
    "campaign_metadata_version",
    "config_fingerprint",
    "input_fingerprint",
    "component",
    "level",
    "state",
    "sample_id",
    "input_batch_size",
    "output_sample_count",
    "runtime_sec",
    "mean_gpu_util_percent",
    "peak_memory_mib",
    "active_memory_mib",
    "active_memory_scope",
    "active_memory_measurement",
    "peak_vram_mib",
    "active_vram_mib",
    "vram_memory_measurement",
    "vram_memory_qc_keep",
    "vram_memory_attribution",
    "host_peak_memory_mib",
    "host_active_memory_mib",
    "host_resident_memory_mib",
    "host_memory_measurement",
    "host_memory_qc_keep",
    "host_memory_attribution",
    "resident_memory_mib",
    "resident_memory_source",
    "resident_baseline_collected_at",
    "resident_baseline_lifecycle_token",
    "resident_baseline_state",
    "worker_generation_token",
    "run_ordinal_in_generation",
    "is_first_real_run",
    "memory_qc_keep",
    "concurrent_execute_overlap",
    "telemetry_wall_clock_sec",
    "telemetry_collected_at",
    "predicted_runtime_sec",
    "predicted_p90_sec",
    "created_at",
    "started_at",
    "finished_at",
    "updated_at",
    "gateway_instance_id",
    "gateway_bind_addr",
    "gateway_git_commit",
    "gateway_started_at",
}


def to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_qc_status(value: Any) -> str:
    return str(value or "").strip().lower()


def qc_keep_from_status(qc_status: str) -> int:
    return 0 if str(qc_status or "").strip().lower() == QC_STATUS_SOFT_DROP else 1


def artifact_exists(path: str) -> bool:
    try:
        file_path = Path(path).expanduser().resolve()
        return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
    except OSError:
        return False


def manifest_exists(manifest: Sequence[Mapping[str, Any]]) -> bool:
    required = [entry for entry in manifest if bool(entry.get("required", True))]
    if not required:
        return False
    for entry in required:
        path = str(entry.get("path") or "").strip()
        if not path or not artifact_exists(path):
            return False
    return True


def build_runtime_query_context(
    *,
    scheduler_decision: Optional[Mapping[str, Any]],
    component: str,
    level: str,
    config_fingerprint: str,
    input_fingerprint: str,
    axes: Optional[Mapping[str, Any]],
    campaign_id: str,
    input_batch_size: Optional[int],
    active_memory_scope: str,
    dispatch_worker_name: str,
    dispatch_gpu_ids: Optional[Sequence[str]],
    concurrent_execute_overlap: bool,
) -> Dict[str, Any]:
    decision_payload = dict(scheduler_decision or {}) if isinstance(scheduler_decision, Mapping) else {}
    signal_payload = (
        dict(decision_payload.get("signal") or {})
        if isinstance(decision_payload.get("signal"), Mapping)
        else {}
    )
    normalized_query_context = normalize_query_context(
        signal_payload.get("query_context") if signal_payload else decision_payload.get("query_context")
    )
    if normalized_query_context:
        return normalized_query_context
    signal_memory = (
        dict(signal_payload.get("memory") or {})
        if isinstance(signal_payload.get("memory"), Mapping)
        else {}
    )
    workload_features = canonicalize_axes(dict(axes or {}))
    worker_context = {
        "worker_name": str(dispatch_worker_name or "").strip(),
        "gpu_ids": [str(item).strip() for item in list(dispatch_gpu_ids or []) if str(item).strip()],
        "active_request_count": 2 if bool(concurrent_execute_overlap) else 1,
        "queue_depth": 0,
        "dispatch_group_size": 1,
    }
    return build_query_context(
        component=str(component or "").strip().lower(),
        level=str(level or "").strip().lower(),
        config_fingerprint=str(config_fingerprint or "").strip(),
        input_fingerprint=str(input_fingerprint or "").strip(),
        workload_features=workload_features,
        requested_batch_size=to_int(input_batch_size) or to_int(workload_features.get("input_batch_size")) or 1,
        worker_context=worker_context,
        campaign_id=str(campaign_id or "").strip(),
        memory_basis=(
            str(signal_memory.get("memory_basis") or "").strip()
            or str(active_memory_scope or "").strip()
            or DEFAULT_ACTIVE_MEMORY_SCOPE
        ),
        peak_fidelity=str(signal_memory.get("peak_fidelity") or "").strip(),
        active_memory_scope=str(active_memory_scope or "").strip() or DEFAULT_ACTIVE_MEMORY_SCOPE,
    )


def inject_runtime_query_context(
    scheduler_decision: Optional[Mapping[str, Any]],
    runtime_query_context: Mapping[str, Any],
) -> Dict[str, Any]:
    decision_payload = dict(scheduler_decision or {}) if isinstance(scheduler_decision, Mapping) else {}
    if int(to_int(decision_payload.get("version")) or 0) >= 5:
        signal_payload = dict(decision_payload.get("signal") or {})
        signal_payload["query_context"] = dict(runtime_query_context)
        decision_payload["signal"] = signal_payload
    else:
        decision_payload["query_context"] = dict(runtime_query_context)
    return decision_payload


def load_axes_conn(conn: sqlite3.Connection, run_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    if not run_keys:
        return {}
    placeholders = ",".join("?" for _ in run_keys)
    rows = conn.execute(
        f"""
        SELECT run_key, axis_key, axis_type, axis_value_text, axis_value_num
        FROM run_axes
        WHERE run_key IN ({placeholders})
        ORDER BY axis_key
        """,
        list(run_keys),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        run_key = str(row["run_key"])
        out.setdefault(run_key, {})
        if str(row["axis_type"]) == "number":
            out[run_key][str(row["axis_key"])] = to_float(row["axis_value_num"])
        else:
            out[run_key][str(row["axis_key"])] = str(row["axis_value_text"] or "")
    return out


def load_links_conn(conn: sqlite3.Connection, run_keys: Sequence[str]) -> Dict[str, List[Tuple[str, str]]]:
    if not run_keys:
        return {}
    placeholders = ",".join("?" for _ in run_keys)
    rows = conn.execute(
        f"""
        SELECT run_key, job_id, kind
        FROM job_links
        WHERE run_key IN ({placeholders})
        ORDER BY created_at DESC
        """,
        list(run_keys),
    ).fetchall()
    out: Dict[str, List[Tuple[str, str]]] = {}
    for row in rows:
        key = str(row["run_key"])
        out.setdefault(key, [])
        out[key].append((str(row["job_id"]), str(row["kind"])))
    return out


def row_to_record(
    row: Any,
    *,
    axes: Mapping[str, Any],
    links: Sequence[Tuple[str, str]],
) -> RunRecord:
    dispatch_gpu_ids = json_loads(row["dispatch_gpu_ids_json"], [])
    if not isinstance(dispatch_gpu_ids, list):
        dispatch_gpu_ids = []
    worker_timing = json_loads(row["worker_timing_us_json"], None)
    if worker_timing is not None and not isinstance(worker_timing, Mapping):
        worker_timing = None
    manifest = json_loads(row["artifact_manifest_json"], [])
    if not isinstance(manifest, list):
        manifest = []
    scheduler_decision = json_loads(row["scheduler_decision_json"], {})
    if not isinstance(scheduler_decision, Mapping):
        scheduler_decision = {}
    bootstrap_memory_summary = json_loads(row["bootstrap_memory_summary_json"], {})
    if not isinstance(bootstrap_memory_summary, Mapping):
        bootstrap_memory_summary = {}
    dispatch_memory_window = json_loads(row["dispatch_memory_window_json"], {})
    if not isinstance(dispatch_memory_window, Mapping):
        dispatch_memory_window = {}

    qc_keep_value = to_int(row["qc_keep"])
    return RunRecord(
        run_key=str(row["run_key"]),
        run_source=str(row["run_source"] or "task"),
        source_event_id=str(row["source_event_id"] or ""),
        cell_schema_id=str(row["cell_schema_id"] or LEGACY_CELL_SCHEMA_ID),
        cell_key=str(row["cell_key"] or ""),
        run_id=str(row["run_id"]),
        campaign_id=str(row["campaign_id"] or ""),
        run_name=str(row["run_name"] or ""),
        submitter=str(row["submitter"] or ""),
        campaign_metadata_version=to_int(row["campaign_metadata_version"]) or 0,
        config_fingerprint=str(row["config_fingerprint"] or ""),
        input_fingerprint=str(row["input_fingerprint"] or ""),
        component=str(row["component"]),
        level=str(row["level"]),
        state=str(row["state"]),
        sample_id=str(row["sample_id"] or ""),
        input_batch_size=to_int(row["input_batch_size"]),
        output_sample_count=to_int(row["output_sample_count"]),
        repeat_idx=to_int(row["repeat_idx"]),
        dispatch_worker_name=str(row["dispatch_worker_name"] or ""),
        dispatch_worker_addr=str(row["dispatch_worker_addr"] or ""),
        dispatch_gpu_ids=[str(item) for item in dispatch_gpu_ids],
        gateway_instance_id=str(row["gateway_instance_id"] or ""),
        gateway_bind_addr=str(row["gateway_bind_addr"] or ""),
        gateway_git_commit=str(row["gateway_git_commit"] or ""),
        gateway_started_at=to_float(row["gateway_started_at"]),
        worker_timing_us=dict(worker_timing) if isinstance(worker_timing, Mapping) else None,
        artifact_manifest=[dict(item) for item in manifest if isinstance(item, Mapping)],
        runtime_sec=to_float(row["runtime_sec"]),
        mean_gpu_util_percent=to_float(row["mean_gpu_util_percent"]),
        peak_memory_mib=to_float(row["peak_memory_mib"]),
        active_memory_mib=to_float(row["active_memory_mib"]),
        active_memory_scope=str(row["active_memory_scope"] or ""),
        active_memory_measurement=str(row["active_memory_measurement"] or ""),
        peak_vram_mib=to_float(row["peak_vram_mib"]),
        active_vram_mib=to_float(row["active_vram_mib"]),
        vram_memory_measurement=str(row["vram_memory_measurement"] or ""),
        vram_memory_qc_keep=bool(to_int(row["vram_memory_qc_keep"]) or 0),
        vram_memory_attribution=str(row["vram_memory_attribution"] or ""),
        host_peak_memory_mib=to_float(row["host_peak_memory_mib"]),
        host_active_memory_mib=to_float(row["host_active_memory_mib"]),
        host_resident_memory_mib=to_float(row["host_resident_memory_mib"]),
        host_memory_measurement=str(row["host_memory_measurement"] or ""),
        host_memory_qc_keep=bool(to_int(row["host_memory_qc_keep"]) or 0),
        host_memory_attribution=str(row["host_memory_attribution"] or ""),
        resident_memory_mib=to_float(row["resident_memory_mib"]),
        resident_memory_source=str(row["resident_memory_source"] or ""),
        resident_baseline_collected_at=to_float(row["resident_baseline_collected_at"]),
        resident_baseline_lifecycle_token=str(row["resident_baseline_lifecycle_token"] or ""),
        resident_baseline_state=str(row["resident_baseline_state"] or ""),
        worker_generation_token=str(row["worker_generation_token"] or ""),
        run_ordinal_in_generation=to_int(row["run_ordinal_in_generation"]),
        is_first_real_run=bool(to_int(row["is_first_real_run"]) or 0),
        memory_qc_keep=bool(to_int(row["memory_qc_keep"]) or 0),
        concurrent_execute_overlap=bool(to_int(row["concurrent_execute_overlap"]) or 0),
        telemetry_wall_clock_sec=to_float(row["telemetry_wall_clock_sec"]),
        telemetry_collected_at=to_float(row["telemetry_collected_at"]),
        bootstrap_memory_summary=dict(bootstrap_memory_summary),
        dispatch_memory_window=dict(dispatch_memory_window),
        scheduler_decision=dict(scheduler_decision),
        predicted_runtime_sec=to_float(row["predicted_runtime_sec"]),
        predicted_p90_sec=to_float(row["predicted_p90_sec"]),
        error=str(row["error"] or ""),
        qc_status=str(row["qc_status"] or ""),
        qc_reason=str(row["qc_reason"] or ""),
        qc_group_key=str(row["qc_group_key"] or ""),
        qc_attempt_count=to_int(row["qc_attempt_count"]),
        qc_keep=bool(qc_keep_value if qc_keep_value is not None else 1),
        created_at=to_float(row["created_at"]),
        started_at=to_float(row["started_at"]),
        finished_at=to_float(row["finished_at"]),
        updated_at=to_float(row["updated_at"]),
        axes=dict(axes),
        job_links=list(links),
    )


def upsert_run_conn(
    conn: sqlite3.Connection,
    *,
    run_key: str,
    run_source: str = "task",
    source_event_id: str = "",
    cell_schema_id: str = LEGACY_CELL_SCHEMA_ID,
    cell_key: str = "",
    run_id: str,
    campaign_id: str = "",
    run_name: str = "",
    submitter: str = "",
    campaign_metadata_version: int = 0,
    config_fingerprint: str = "",
    input_fingerprint: str = "",
    component: str,
    level: str,
    state: str,
    sample_id: str = "",
    input_batch_size: Optional[int] = None,
    output_sample_count: Optional[int] = None,
    repeat_idx: Optional[int] = None,
    dispatch_worker_name: str = "",
    dispatch_worker_addr: str = "",
    dispatch_gpu_ids: Optional[Sequence[str]] = None,
    gateway_instance_id: str = "",
    gateway_bind_addr: str = "",
    gateway_git_commit: str = "",
    gateway_started_at: Optional[float] = None,
    worker_timing_us: Optional[Mapping[str, Any]] = None,
    artifact_manifest: Optional[Sequence[Mapping[str, Any]]] = None,
    runtime_sec: Optional[float] = None,
    mean_gpu_util_percent: Optional[float] = None,
    peak_memory_mib: Optional[float] = None,
    active_memory_mib: Optional[float] = None,
    active_memory_scope: str = "",
    active_memory_measurement: str = "",
    peak_vram_mib: Optional[float] = None,
    active_vram_mib: Optional[float] = None,
    vram_memory_measurement: str = "",
    vram_memory_qc_keep: bool = False,
    vram_memory_attribution: str = "",
    host_peak_memory_mib: Optional[float] = None,
    host_active_memory_mib: Optional[float] = None,
    host_resident_memory_mib: Optional[float] = None,
    host_memory_measurement: str = "",
    host_memory_qc_keep: bool = False,
    host_memory_attribution: str = "",
    resident_memory_mib: Optional[float] = None,
    resident_memory_source: str = "",
    resident_baseline_collected_at: Optional[float] = None,
    resident_baseline_lifecycle_token: str = "",
    resident_baseline_state: str = "",
    worker_generation_token: str = "",
    run_ordinal_in_generation: Optional[int] = None,
    is_first_real_run: bool = False,
    memory_qc_keep: bool = False,
    concurrent_execute_overlap: bool = False,
    telemetry_wall_clock_sec: Optional[float] = None,
    telemetry_collected_at: Optional[float] = None,
    bootstrap_memory_summary: Optional[Mapping[str, Any]] = None,
    dispatch_memory_window: Optional[Mapping[str, Any]] = None,
    scheduler_decision: Optional[Mapping[str, Any]] = None,
    predicted_runtime_sec: Optional[float] = None,
    predicted_p90_sec: Optional[float] = None,
    error: str = "",
    qc_status: str = "",
    qc_keep: Optional[bool] = None,
    qc_reason: str = "",
    qc_group_key: str = "",
    qc_attempt_count: Optional[int] = None,
    created_at: Optional[float] = None,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    axes: Optional[Mapping[str, Any]] = None,
) -> None:
    normalized_run_key = str(run_key).strip()
    if not normalized_run_key:
        raise ValueError("run_key is required")
    timestamp = time.time()
    created = float(created_at or timestamp)
    started = to_float(started_at)
    finished = to_float(finished_at)
    updated = finished if finished is not None else timestamp

    dispatch_gpu_payload = [str(item).strip() for item in list(dispatch_gpu_ids or []) if str(item).strip()]
    worker_timing_payload = dict(worker_timing_us or {}) if isinstance(worker_timing_us, Mapping) else None
    manifest_payload = [dict(item) for item in list(artifact_manifest or []) if isinstance(item, Mapping)]
    axis_payload = canonicalize_axes(dict(axes or {}))
    normalized_qc_status = normalize_qc_status(qc_status)
    normalized_qc_keep = qc_keep_from_status(normalized_qc_status)
    if qc_keep is not None:
        normalized_qc_keep = 1 if bool(qc_keep) else 0
    normalized_cell_schema_id = str(cell_schema_id or "").strip() or LEGACY_CELL_SCHEMA_ID
    normalized_run_source = str(run_source or "").strip().lower() or "task"
    normalized_source_event_id = str(source_event_id or "").strip()
    normalized_gateway_instance_id = str(gateway_instance_id or "").strip()
    normalized_gateway_bind_addr = str(gateway_bind_addr or "").strip()
    normalized_gateway_git_commit = str(gateway_git_commit or "").strip().lower()

    scheduler_decision_payload = dict(scheduler_decision or {})
    runtime_query_context = build_runtime_query_context(
        scheduler_decision=scheduler_decision_payload,
        component=str(component).strip().lower(),
        level=str(level).strip().lower(),
        config_fingerprint=str(config_fingerprint or "").strip(),
        input_fingerprint=str(input_fingerprint or "").strip(),
        axes=axis_payload,
        campaign_id=str(campaign_id or "").strip(),
        input_batch_size=to_int(input_batch_size),
        active_memory_scope=str(active_memory_scope or "").strip(),
        dispatch_worker_name=str(dispatch_worker_name or "").strip(),
        dispatch_gpu_ids=dispatch_gpu_payload,
        concurrent_execute_overlap=bool(concurrent_execute_overlap),
    )
    scheduler_decision_payload = inject_runtime_query_context(
        scheduler_decision_payload,
        runtime_query_context,
    )

    conn.execute(
        """
        INSERT INTO profile_runs (
            run_key, run_source, source_event_id, cell_schema_id, cell_key, run_id,
            campaign_id, run_name, submitter,
            campaign_metadata_version,
            config_fingerprint, input_fingerprint,
            component, level, state, sample_id,
            input_batch_size, output_sample_count, repeat_idx,
            dispatch_worker_name, dispatch_worker_addr, dispatch_gpu_ids_json,
            gateway_instance_id, gateway_bind_addr, gateway_git_commit, gateway_started_at,
            worker_timing_us_json, artifact_manifest_json, runtime_sec,
            mean_gpu_util_percent, peak_memory_mib,
            active_memory_mib, active_memory_scope, active_memory_measurement,
            peak_vram_mib, active_vram_mib, vram_memory_measurement, vram_memory_qc_keep,
            vram_memory_attribution,
            host_peak_memory_mib, host_active_memory_mib, host_resident_memory_mib,
            host_memory_measurement, host_memory_qc_keep, host_memory_attribution,
            resident_memory_mib, resident_memory_source,
            resident_baseline_collected_at, resident_baseline_lifecycle_token, resident_baseline_state,
            worker_generation_token, run_ordinal_in_generation, is_first_real_run,
            memory_qc_keep, concurrent_execute_overlap,
            telemetry_wall_clock_sec, telemetry_collected_at,
            bootstrap_memory_summary_json, dispatch_memory_window_json,
            scheduler_decision_json, predicted_runtime_sec, predicted_p90_sec,
            error,
            qc_status, qc_keep, qc_reason, qc_group_key, qc_attempt_count,
            created_at, started_at, finished_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_key) DO UPDATE SET
            run_source = excluded.run_source,
            source_event_id = excluded.source_event_id,
            cell_schema_id = excluded.cell_schema_id,
            cell_key = excluded.cell_key,
            run_id = excluded.run_id,
            campaign_id = excluded.campaign_id,
            run_name = excluded.run_name,
            submitter = excluded.submitter,
            campaign_metadata_version = excluded.campaign_metadata_version,
            config_fingerprint = excluded.config_fingerprint,
            input_fingerprint = excluded.input_fingerprint,
            component = excluded.component,
            level = excluded.level,
            state = excluded.state,
            sample_id = excluded.sample_id,
            input_batch_size = excluded.input_batch_size,
            output_sample_count = excluded.output_sample_count,
            repeat_idx = excluded.repeat_idx,
            dispatch_worker_name = excluded.dispatch_worker_name,
            dispatch_worker_addr = excluded.dispatch_worker_addr,
            dispatch_gpu_ids_json = excluded.dispatch_gpu_ids_json,
            gateway_instance_id = excluded.gateway_instance_id,
            gateway_bind_addr = excluded.gateway_bind_addr,
            gateway_git_commit = excluded.gateway_git_commit,
            gateway_started_at = excluded.gateway_started_at,
            worker_timing_us_json = excluded.worker_timing_us_json,
            artifact_manifest_json = excluded.artifact_manifest_json,
            runtime_sec = excluded.runtime_sec,
            mean_gpu_util_percent = excluded.mean_gpu_util_percent,
            peak_memory_mib = excluded.peak_memory_mib,
            active_memory_mib = excluded.active_memory_mib,
            active_memory_scope = excluded.active_memory_scope,
            active_memory_measurement = excluded.active_memory_measurement,
            peak_vram_mib = excluded.peak_vram_mib,
            active_vram_mib = excluded.active_vram_mib,
            vram_memory_measurement = excluded.vram_memory_measurement,
            vram_memory_qc_keep = excluded.vram_memory_qc_keep,
            vram_memory_attribution = excluded.vram_memory_attribution,
            host_peak_memory_mib = excluded.host_peak_memory_mib,
            host_active_memory_mib = excluded.host_active_memory_mib,
            host_resident_memory_mib = excluded.host_resident_memory_mib,
            host_memory_measurement = excluded.host_memory_measurement,
            host_memory_qc_keep = excluded.host_memory_qc_keep,
            host_memory_attribution = excluded.host_memory_attribution,
            resident_memory_mib = excluded.resident_memory_mib,
            resident_memory_source = excluded.resident_memory_source,
            resident_baseline_collected_at = excluded.resident_baseline_collected_at,
            resident_baseline_lifecycle_token = excluded.resident_baseline_lifecycle_token,
            resident_baseline_state = excluded.resident_baseline_state,
            worker_generation_token = excluded.worker_generation_token,
            run_ordinal_in_generation = excluded.run_ordinal_in_generation,
            is_first_real_run = excluded.is_first_real_run,
            memory_qc_keep = excluded.memory_qc_keep,
            concurrent_execute_overlap = excluded.concurrent_execute_overlap,
            telemetry_wall_clock_sec = excluded.telemetry_wall_clock_sec,
            telemetry_collected_at = excluded.telemetry_collected_at,
            bootstrap_memory_summary_json = excluded.bootstrap_memory_summary_json,
            dispatch_memory_window_json = excluded.dispatch_memory_window_json,
            scheduler_decision_json = excluded.scheduler_decision_json,
            predicted_runtime_sec = excluded.predicted_runtime_sec,
            predicted_p90_sec = excluded.predicted_p90_sec,
            error = excluded.error,
            qc_status = excluded.qc_status,
            qc_keep = excluded.qc_keep,
            qc_reason = excluded.qc_reason,
            qc_group_key = excluded.qc_group_key,
            qc_attempt_count = excluded.qc_attempt_count,
            started_at = excluded.started_at,
            finished_at = excluded.finished_at,
            updated_at = excluded.updated_at,
            created_at = CASE
                WHEN profile_runs.created_at IS NULL OR profile_runs.created_at <= 0
                    THEN excluded.created_at
                ELSE profile_runs.created_at
            END
        """,
        (
            normalized_run_key,
            normalized_run_source,
            normalized_source_event_id,
            normalized_cell_schema_id,
            str(cell_key or "").strip(),
            str(run_id),
            str(campaign_id or "").strip(),
            str(run_name or "").strip(),
            str(submitter or "").strip(),
            max(0, to_int(campaign_metadata_version) or 0),
            str(config_fingerprint or "").strip(),
            str(input_fingerprint or "").strip(),
            str(component).strip().lower(),
            str(level).strip().lower(),
            str(state).strip().upper(),
            str(sample_id or "").strip(),
            to_int(input_batch_size),
            to_int(output_sample_count),
            to_int(repeat_idx),
            str(dispatch_worker_name or "").strip(),
            str(dispatch_worker_addr or "").strip(),
            json_dumps(dispatch_gpu_payload),
            normalized_gateway_instance_id,
            normalized_gateway_bind_addr,
            normalized_gateway_git_commit,
            to_float(gateway_started_at),
            json_dumps(worker_timing_payload) if worker_timing_payload is not None else "",
            json_dumps(manifest_payload),
            to_float(runtime_sec),
            to_float(mean_gpu_util_percent),
            to_float(peak_memory_mib),
            to_float(active_memory_mib),
            str(active_memory_scope or "").strip(),
            str(active_memory_measurement or "").strip(),
            to_float(peak_vram_mib),
            to_float(active_vram_mib),
            str(vram_memory_measurement or "").strip(),
            1 if bool(vram_memory_qc_keep) else 0,
            str(vram_memory_attribution or "").strip(),
            to_float(host_peak_memory_mib),
            to_float(host_active_memory_mib),
            to_float(host_resident_memory_mib),
            str(host_memory_measurement or "").strip(),
            1 if bool(host_memory_qc_keep) else 0,
            str(host_memory_attribution or "").strip(),
            to_float(resident_memory_mib),
            str(resident_memory_source or "").strip(),
            to_float(resident_baseline_collected_at),
            str(resident_baseline_lifecycle_token or "").strip(),
            str(resident_baseline_state or "").strip(),
            str(worker_generation_token or "").strip(),
            to_int(run_ordinal_in_generation),
            1 if bool(is_first_real_run) else 0,
            1 if bool(memory_qc_keep) else 0,
            1 if bool(concurrent_execute_overlap) else 0,
            to_float(telemetry_wall_clock_sec),
            to_float(telemetry_collected_at),
            json_dumps(dict(bootstrap_memory_summary or {})),
            json_dumps(dict(dispatch_memory_window or {})),
            json_dumps(scheduler_decision_payload),
            to_float(predicted_runtime_sec),
            to_float(predicted_p90_sec),
            str(error or ""),
            normalized_qc_status,
            int(normalized_qc_keep),
            str(qc_reason or ""),
            str(qc_group_key or ""),
            to_int(qc_attempt_count) or 1,
            created,
            started,
            finished,
            updated,
        ),
    )
    conn.execute(
        """
        INSERT INTO run_cell_memberships (
            run_key, cell_schema_id, cell_key, source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_key, cell_schema_id) DO UPDATE SET
            cell_key = excluded.cell_key,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        (
            normalized_run_key,
            normalized_cell_schema_id,
            str(cell_key or "").strip(),
            "upsert_run",
            created,
            updated,
        ),
    )
    conn.execute("DELETE FROM run_axes WHERE run_key = ?", (normalized_run_key,))
    for axis_key, axis_value in axis_payload.items():
        if isinstance(axis_value, (int, float)) and not isinstance(axis_value, bool):
            axis_type = "number"
            axis_value_num: Optional[float] = float(axis_value)
            axis_value_text: Optional[str] = None
        else:
            axis_type = "text"
            axis_value_num = None
            axis_value_text = str(axis_value)
        conn.execute(
            """
            INSERT OR REPLACE INTO run_axes (
                run_key, axis_key, axis_type, axis_value_text, axis_value_num
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                normalized_run_key,
                str(axis_key),
                axis_type,
                axis_value_text,
                axis_value_num,
            ),
        )


def link_job_conn(conn: sqlite3.Connection, *, job_id: str, kind: str, run_key: str) -> None:
    normalized_job_id = str(job_id).strip()
    normalized_kind = str(kind).strip().lower()
    normalized_run_key = str(run_key).strip()
    if not normalized_job_id or not normalized_run_key:
        return
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO job_links (job_id, kind, run_key, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (normalized_job_id, normalized_kind, normalized_run_key, time.time()),
        )
    except sqlite3.IntegrityError:
        return


def delete_run_conn(conn: sqlite3.Connection, run_key: str) -> None:
    normalized_run_key = str(run_key).strip()
    if not normalized_run_key:
        return
    conn.execute("DELETE FROM profile_runs WHERE run_key = ?", (normalized_run_key,))


def append_qc_event_conn(
    conn: sqlite3.Connection,
    *,
    run_key: str,
    action: str,
    reason: str = "",
    group_key: str = "",
    job_id: str = "",
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    normalized_run_key = str(run_key).strip()
    if not normalized_run_key:
        return
    conn.execute(
        """
        INSERT INTO run_qc_events (
            run_key, action, reason, group_key, job_id, created_at, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized_run_key,
            str(action or "").strip(),
            str(reason or ""),
            str(group_key or ""),
            str(job_id or ""),
            time.time(),
            json_dumps(dict(payload or {})),
        ),
    )


def get_run_conn(conn: sqlite3.Connection, run_key: str) -> Optional[RunRecord]:
    normalized_run_key = str(run_key).strip()
    if not normalized_run_key:
        return None
    row = conn.execute(
        "SELECT * FROM profile_runs WHERE run_key = ?",
        (normalized_run_key,),
    ).fetchone()
    if row is None:
        return None
    axes = load_axes_conn(conn, [normalized_run_key]).get(normalized_run_key, {})
    links = load_links_conn(conn, [normalized_run_key]).get(normalized_run_key, [])
    return row_to_record(row, axes=axes, links=links)


def lookup_succeeded_conn(conn: sqlite3.Connection, run_key: str) -> Optional[RunRecord]:
    """Return only MODEL_INPUT_FILTER-eligible succeeded rows with valid manifests."""
    record = get_run_conn(conn, run_key)
    if record is None:
        return None
    if not matches_model_input_filter(
        state=record.state,
        qc_keep=record.qc_keep,
        run_source=record.run_source,
    ):
        return None
    if not manifest_exists(record.artifact_manifest):
        return None
    return record


def query_runs_conn(conn: sqlite3.Connection, query: RunQuery) -> Tuple[List[RunRecord], int]:
    sort_col = str(query.sort or "finished_at").strip()
    if sort_col not in ALLOWED_SORT_COLUMNS:
        raise ValueError(f"unsupported sort column '{sort_col}'")
    direction = "DESC" if bool(query.descending) else "ASC"

    base_joins: List[str] = []
    where_clauses: List[str] = []
    where_params: List[Any] = []
    join_params: List[Any] = []

    if query.component:
        where_clauses.append("r.component = ?")
        where_params.append(str(query.component).strip().lower())
    if query.run_source:
        where_clauses.append("r.run_source = ?")
        where_params.append(str(query.run_source).strip().lower())
    if query.level:
        where_clauses.append("r.level = ?")
        where_params.append(str(query.level).strip().lower())
    if query.campaign_id:
        where_clauses.append("r.campaign_id = ?")
        where_params.append(str(query.campaign_id).strip())
    if query.cell_schema_id:
        where_clauses.append("r.cell_schema_id = ?")
        where_params.append(str(query.cell_schema_id).strip())
    if query.state:
        where_clauses.append("r.state = ?")
        where_params.append(str(query.state).strip().upper())
    if query.sample_id:
        where_clauses.append("r.sample_id = ?")
        where_params.append(str(query.sample_id).strip())
    if query.input_batch_size is not None:
        where_clauses.append("r.input_batch_size = ?")
        where_params.append(int(query.input_batch_size))
    if query.output_sample_count is not None:
        where_clauses.append("r.output_sample_count = ?")
        where_params.append(int(query.output_sample_count))
    if query.worker_name:
        where_clauses.append("r.dispatch_worker_name = ?")
        where_params.append(str(query.worker_name).strip())
    if query.gpu_id:
        where_clauses.append("r.dispatch_gpu_ids_json LIKE ?")
        where_params.append(f"%\"{str(query.gpu_id).strip()}\"%")
    gateway_instance_expr = (
        "CASE WHEN TRIM(COALESCE(r.gateway_instance_id, '')) = '' THEN ? "
        "ELSE TRIM(COALESCE(r.gateway_instance_id, '')) END"
    )
    gateway_commit_expr = (
        "CASE WHEN LOWER(TRIM(COALESCE(r.gateway_git_commit, ''))) = '' THEN ? "
        "ELSE LOWER(TRIM(COALESCE(r.gateway_git_commit, ''))) END"
    )
    include_gateway_instance_ids = list(
        dict.fromkeys(
            normalize_gateway_instance_id(item)
            for item in list(query.include_gateway_instance_ids or [])
            if str(item or "").strip()
        )
    )
    exclude_gateway_instance_ids = list(
        dict.fromkeys(
            normalize_gateway_instance_id(item)
            for item in list(query.exclude_gateway_instance_ids or [])
            if str(item or "").strip()
        )
    )
    include_gateway_git_commits = list(
        dict.fromkeys(
            normalize_gateway_git_commit(item)
            for item in list(query.include_gateway_git_commits or [])
            if str(item or "").strip()
        )
    )
    exclude_gateway_git_commits = list(
        dict.fromkeys(
            normalize_gateway_git_commit(item)
            for item in list(query.exclude_gateway_git_commits or [])
            if str(item or "").strip()
        )
    )
    if include_gateway_instance_ids:
        placeholders = ",".join("?" for _ in include_gateway_instance_ids)
        where_clauses.append(f"{gateway_instance_expr} IN ({placeholders})")
        where_params.extend([UNKNOWN_GATEWAY_VALUE, *include_gateway_instance_ids])
    if exclude_gateway_instance_ids:
        placeholders = ",".join("?" for _ in exclude_gateway_instance_ids)
        where_clauses.append(f"{gateway_instance_expr} NOT IN ({placeholders})")
        where_params.extend([UNKNOWN_GATEWAY_VALUE, *exclude_gateway_instance_ids])
    if include_gateway_git_commits:
        placeholders = ",".join("?" for _ in include_gateway_git_commits)
        where_clauses.append(f"{gateway_commit_expr} IN ({placeholders})")
        where_params.extend([UNKNOWN_GATEWAY_VALUE, *include_gateway_git_commits])
    if exclude_gateway_git_commits:
        placeholders = ",".join("?" for _ in exclude_gateway_git_commits)
        where_clauses.append(f"{gateway_commit_expr} NOT IN ({placeholders})")
        where_params.extend([UNKNOWN_GATEWAY_VALUE, *exclude_gateway_git_commits])
    if query.from_ts is not None:
        where_clauses.append("COALESCE(r.finished_at, r.updated_at, r.created_at) >= ?")
        where_params.append(float(query.from_ts))
    if query.to_ts is not None:
        where_clauses.append("COALESCE(r.finished_at, r.updated_at, r.created_at) <= ?")
        where_params.append(float(query.to_ts))

    for idx, item in enumerate(list(query.axis_filters or [])):
        alias = f"ax{idx}"
        key = str(item.key).strip()
        value = str(item.value).strip()
        if not key:
            continue
        numeric_value = to_float(value)
        if numeric_value is None:
            base_joins.append(
                f"JOIN run_axes {alias} ON {alias}.run_key = r.run_key "
                f"AND {alias}.axis_key = ? AND {alias}.axis_value_text = ?"
            )
            join_params.extend([key, value])
        else:
            base_joins.append(
                f"JOIN run_axes {alias} ON {alias}.run_key = r.run_key "
                f"AND {alias}.axis_key = ? AND ({alias}.axis_value_num = ? OR {alias}.axis_value_text = ?)"
            )
            join_params.extend([key, numeric_value, value])

    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    join_sql = " " + " ".join(base_joins) if base_joins else ""
    base_sql = f"FROM profile_runs r{join_sql}{where_sql}"
    params = [*join_params, *where_params]
    limit = max(1, min(int(query.limit), 1000))
    offset = max(0, int(query.offset))

    total = int(
        conn.execute(
            f"SELECT COUNT(DISTINCT r.run_key) {base_sql}",
            params,
        ).fetchone()[0]
    )
    rows = conn.execute(
        f"""
        SELECT r.*
        {base_sql}
        GROUP BY r.run_key
        ORDER BY r.{sort_col} {direction}
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()
    run_keys = [str(row["run_key"]) for row in rows]
    axes_map = load_axes_conn(conn, run_keys)
    links_map = load_links_conn(conn, run_keys)
    records = [
        row_to_record(
            row,
            axes=axes_map.get(str(row["run_key"]), {}),
            links=links_map.get(str(row["run_key"]), []),
        )
        for row in rows
    ]
    return records, total
