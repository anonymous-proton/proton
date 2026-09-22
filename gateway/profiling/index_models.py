"""Types for profiling run index queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..gateway_identity import gateway_payload_from_mapping
from .constants import matches_model_input_filter


def include_run_for_modeling(record: "RunRecord") -> bool:
    return matches_model_input_filter(
        state=getattr(record, "state", ""),
        qc_keep=getattr(record, "qc_keep", False),
        run_source=getattr(record, "run_source", ""),
    )


@dataclass(frozen=True)
class AxisFilter:
    key: str
    value: str


@dataclass(frozen=True)
class RunQuery:
    run_source: Optional[str] = None
    component: Optional[str] = None
    level: Optional[str] = None
    state: Optional[str] = None
    campaign_id: Optional[str] = None
    sample_id: Optional[str] = None
    cell_schema_id: Optional[str] = None
    input_batch_size: Optional[int] = None
    output_sample_count: Optional[int] = None
    from_ts: Optional[float] = None
    to_ts: Optional[float] = None
    worker_name: Optional[str] = None
    gpu_id: Optional[str] = None
    include_gateway_instance_ids: List[str] = field(default_factory=list)
    exclude_gateway_instance_ids: List[str] = field(default_factory=list)
    include_gateway_git_commits: List[str] = field(default_factory=list)
    exclude_gateway_git_commits: List[str] = field(default_factory=list)
    axis_filters: List[AxisFilter] = field(default_factory=list)
    limit: int = 200
    offset: int = 0
    sort: str = "finished_at"
    descending: bool = True


@dataclass
class RunRecord:
    run_key: str
    run_source: str
    source_event_id: str
    cell_schema_id: str
    cell_key: str
    run_id: str
    campaign_id: str
    run_name: str
    submitter: str
    campaign_metadata_version: int
    config_fingerprint: str
    input_fingerprint: str
    component: str
    level: str
    state: str
    sample_id: str
    input_batch_size: Optional[int]
    output_sample_count: Optional[int]
    repeat_idx: Optional[int]
    dispatch_worker_name: str
    dispatch_worker_addr: str
    dispatch_gpu_ids: List[str]
    gateway_instance_id: str
    gateway_bind_addr: str
    gateway_git_commit: str
    gateway_started_at: Optional[float]
    worker_timing_us: Optional[Dict[str, int]]
    artifact_manifest: List[Dict[str, Any]]
    runtime_sec: Optional[float]
    mean_gpu_util_percent: Optional[float]
    peak_memory_mib: Optional[float]
    active_memory_mib: Optional[float]
    active_memory_scope: str
    active_memory_measurement: str
    peak_vram_mib: Optional[float]
    active_vram_mib: Optional[float]
    vram_memory_measurement: str
    vram_memory_qc_keep: bool
    vram_memory_attribution: str
    host_peak_memory_mib: Optional[float]
    host_active_memory_mib: Optional[float]
    host_resident_memory_mib: Optional[float]
    host_memory_measurement: str
    host_memory_qc_keep: bool
    host_memory_attribution: str
    resident_memory_mib: Optional[float]
    resident_memory_source: str
    resident_baseline_collected_at: Optional[float]
    resident_baseline_lifecycle_token: str
    resident_baseline_state: str
    worker_generation_token: str
    run_ordinal_in_generation: Optional[int]
    is_first_real_run: bool
    memory_qc_keep: bool
    concurrent_execute_overlap: bool
    telemetry_wall_clock_sec: Optional[float]
    telemetry_collected_at: Optional[float]
    bootstrap_memory_summary: Dict[str, Any]
    dispatch_memory_window: Dict[str, Any]
    scheduler_decision: Dict[str, Any]
    predicted_runtime_sec: Optional[float]
    predicted_p90_sec: Optional[float]
    error: str
    qc_status: str = ""
    qc_keep: bool = True
    qc_reason: str = ""
    qc_group_key: str = ""
    qc_attempt_count: Optional[int] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    updated_at: Optional[float] = None
    axes: Dict[str, Any] = field(default_factory=dict)
    job_links: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_key": self.run_key,
            "run_source": self.run_source,
            "source_event_id": self.source_event_id,
            "cell_schema_id": self.cell_schema_id,
            "cell_key": self.cell_key,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "run_name": self.run_name,
            "submitter": self.submitter,
            "campaign_metadata_version": int(self.campaign_metadata_version),
            "config_fingerprint": self.config_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "component": self.component,
            "level": self.level,
            "state": self.state,
            "sample_id": self.sample_id,
            "input_batch_size": self.input_batch_size,
            "output_sample_count": self.output_sample_count,
            "repeat_idx": self.repeat_idx,
            "dispatch": {
                "worker_name": self.dispatch_worker_name,
                "worker_addr": self.dispatch_worker_addr,
                "gpu_ids": list(self.dispatch_gpu_ids),
            },
            "gateway": gateway_payload_from_mapping(
                {
                    "instance_id": self.gateway_instance_id,
                    "bind_addr": self.gateway_bind_addr,
                    "git_commit": self.gateway_git_commit,
                    "started_at": self.gateway_started_at,
                }
            ),
            "worker_timing_us": self.worker_timing_us,
            "artifact_manifest": list(self.artifact_manifest),
            "runtime_sec": self.runtime_sec,
            "mean_gpu_util_percent": self.mean_gpu_util_percent,
            "peak_memory_mib": self.peak_memory_mib,
            "active_memory_mib": self.active_memory_mib,
            "active_memory_scope": self.active_memory_scope,
            "active_memory_measurement": self.active_memory_measurement,
            "peak_vram_mib": self.peak_vram_mib,
            "active_vram_mib": self.active_vram_mib,
            "vram_memory_measurement": self.vram_memory_measurement,
            "vram_memory_qc_keep": bool(self.vram_memory_qc_keep),
            "vram_memory_attribution": self.vram_memory_attribution,
            "host_peak_memory_mib": self.host_peak_memory_mib,
            "host_active_memory_mib": self.host_active_memory_mib,
            "host_resident_memory_mib": self.host_resident_memory_mib,
            "host_memory_measurement": self.host_memory_measurement,
            "host_memory_qc_keep": bool(self.host_memory_qc_keep),
            "host_memory_attribution": self.host_memory_attribution,
            "resident_memory_mib": self.resident_memory_mib,
            "resident_memory_source": self.resident_memory_source,
            "resident_baseline_collected_at": self.resident_baseline_collected_at,
            "resident_baseline_lifecycle_token": self.resident_baseline_lifecycle_token,
            "resident_baseline_state": self.resident_baseline_state,
            "worker_generation_token": self.worker_generation_token,
            "run_ordinal_in_generation": self.run_ordinal_in_generation,
            "is_first_real_run": bool(self.is_first_real_run),
            "memory_qc_keep": bool(self.memory_qc_keep),
            "concurrent_execute_overlap": bool(self.concurrent_execute_overlap),
            "telemetry_wall_clock_sec": self.telemetry_wall_clock_sec,
            "telemetry_collected_at": self.telemetry_collected_at,
            "bootstrap_memory_summary": dict(self.bootstrap_memory_summary),
            "dispatch_memory_window": dict(self.dispatch_memory_window),
            "scheduler_decision": dict(self.scheduler_decision),
            "predicted_runtime_sec": self.predicted_runtime_sec,
            "predicted_p90_sec": self.predicted_p90_sec,
            "error": self.error,
            "qc_status": self.qc_status,
            "qc_keep": bool(self.qc_keep),
            "qc_reason": self.qc_reason,
            "qc_group_key": self.qc_group_key,
            "qc_attempt_count": self.qc_attempt_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "axes": dict(self.axes),
            "job_links": [
                {"job_id": job_id, "kind": kind}
                for job_id, kind in self.job_links
            ],
        }


@dataclass
class ComponentModelVersionRecord:
    component_model_version_id: str
    component: str
    parent_loop_model_version_id: str
    job_id: str
    loop_idx: int
    trained_at: Optional[float]
    status: str
    is_active: bool
    error: str
    cell_schema_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    decision_gate: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    priority: Dict[str, Any] = field(default_factory=dict)
    delta: Dict[str, Any] = field(default_factory=dict)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component_model_version_id": self.component_model_version_id,
            "component": self.component,
            "parent_loop_model_version_id": self.parent_loop_model_version_id,
            "job_id": self.job_id,
            "loop_idx": self.loop_idx,
            "trained_at": self.trained_at,
            "status": self.status,
            "is_active": self.is_active,
            "error": self.error,
            "cell_schema_ids": list(self.cell_schema_ids),
            "metrics": dict(self.metrics),
            "decision_gate": dict(self.decision_gate),
            "readiness": dict(self.readiness),
            "priority": dict(self.priority),
            "delta": dict(self.delta),
            "segments": [dict(item) for item in list(self.segments or []) if isinstance(item, dict)],
            "artifacts": dict(self.artifacts),
        }
