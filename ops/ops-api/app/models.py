from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceStatus(BaseModel):
    required: bool
    status: str = Field(default="unknown")
    stale: bool = False
    error_reason: Optional[str] = None
    last_attempt_at: Optional[str] = None
    last_ok_at: Optional[str] = None


class MetaPayload(BaseModel):
    generated_at: str
    data_age_seconds: float
    stale: bool
    stale_reason: List[str] = Field(default_factory=list)
    poll_interval_seconds: float
    ttl_seconds: float
    sources: Dict[str, SourceStatus]


class OverviewResponse(BaseModel):
    meta: MetaPayload
    fleet: Dict[str, Any]


class RecentRunsResponse(BaseModel):
    generated_at: str
    filters: Dict[str, Any]
    summary: Dict[str, Any]
    total: int
    limit: int
    offset: int
    runs: List[Dict[str, Any]]


class RunDetailResponse(BaseModel):
    generated_at: str
    run_key: str
    overview: Dict[str, Any]
    takeaway: Dict[str, Any]
    request_trace_context: Dict[str, Any]
    selected_worker_context: Dict[str, Any]
    estimator_explanations: Dict[str, Any]
    actual_telemetry: Dict[str, Any]
    raw: Dict[str, Any]
