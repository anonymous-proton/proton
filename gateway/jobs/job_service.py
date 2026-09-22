"""Single submit/status/cancel facade over task backend."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .job_models import JobKind, JobRecord, JobState

SubmitFn = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]
StatusFn = Callable[[str], Awaitable[Mapping[str, Any] | None]]
CancelFn = Callable[[str], Awaitable[bool]]


class JobService:
    def __init__(
        self,
        *,
        submit_task: SubmitFn,
        get_task: StatusFn,
        cancel_task: CancelFn,
    ) -> None:
        self._submit_task = submit_task
        self._get_task = get_task
        self._cancel_task = cancel_task
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def parse_kind(raw: Any) -> JobKind:
        kind = str(raw or "").strip().lower()
        try:
            return JobKind(kind)
        except ValueError as exc:
            raise ValueError("kind must be one of [task]") from exc

    async def submit(self, *, kind: JobKind, payload: Mapping[str, Any]) -> JobRecord:
        if kind != JobKind.TASK:
            raise ValueError("kind must be one of [task]")

        submit_resp = dict(await self._submit_task(payload))
        inner_id = str(submit_resp.get("task_id") or "").strip()
        if not inner_id:
            raise RuntimeError("backend did not return identifier for kind=task")

        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, kind=JobKind.TASK, inner_id=inner_id)
        record.state = self._to_state(submit_resp.get("state"))
        record.touch()

        async with self._lock:
            self._jobs[job_id] = record
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            return None

        await self._refresh_record(record)
        return record

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            return False

        await self._refresh_record(record)
        if record.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            return True

        ok = await self._cancel_task(record.inner_id)
        if ok:
            await self._refresh_record(record)
            if record.state not in {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            }:
                record.state = JobState.CANCELLED
                record.ok = False
                if not record.error:
                    record.error = "cancelled"
                record.touch()
        return ok

    @staticmethod
    def _to_state(raw: Any) -> JobState:
        value = str(raw or "UNSPECIFIED").strip().upper()
        try:
            return JobState(value)
        except ValueError:
            return JobState.UNSPECIFIED

    @staticmethod
    def _normalize_dispatch(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            return {"worker_addr": "", "worker_name": "", "gpu_ids": []}
        return {
            "worker_addr": str(raw.get("worker_addr") or "").strip(),
            "worker_name": str(raw.get("worker_name") or "").strip(),
            "gpu_ids": [
                str(item).strip()
                for item in list(raw.get("gpu_ids") or [])
                if str(item).strip()
            ],
        }

    def _apply_status(self, record: JobRecord, payload: Mapping[str, Any]) -> None:
        state = self._to_state(payload.get("state"))
        record.state = state
        record.ok = bool(payload.get("ok", False))
        record.error = "" if record.ok else str(payload.get("message") or "")
        record.dispatch = self._normalize_dispatch(payload.get("dispatch"))
        record.result = {
            "component": str(payload.get("component") or ""),
            "exit_code": int(payload.get("exit_code", 1)),
            "message": str(payload.get("message") or ""),
            "worker_timing_us": payload.get("worker_timing_us"),
        }

        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        if isinstance(created_at, (int, float)) and created_at > 0:
            record.created_at = float(created_at)
        if isinstance(updated_at, (int, float)) and updated_at > 0:
            record.updated_at = float(updated_at)
        else:
            record.touch()

    async def _refresh_record(self, record: JobRecord) -> None:
        raw = await self._get_task(record.inner_id)
        if raw is None:
            record.state = JobState.FAILED
            record.ok = False
            record.error = "underlying job not found"
            record.touch()
            return
        self._apply_status(record, dict(raw))
