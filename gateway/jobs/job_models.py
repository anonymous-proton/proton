"""Common models for single Job API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class JobKind(str, Enum):
    TASK = "task"


class JobState(str, Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass
class JobRecord:
    job_id: str
    kind: JobKind
    inner_id: str
    state: JobState = JobState.SUBMITTED
    ok: bool = False
    error: str = ""
    dispatch: Dict[str, Any] = field(default_factory=lambda: {
        "worker_addr": "",
        "worker_name": "",
        "gpu_ids": [],
    })
    result: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def as_response(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind.value,
            "state": self.state.value,
            "ok": self.ok,
            "error": self.error,
            "dispatch": dict(self.dispatch),
            "result": dict(self.result),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
