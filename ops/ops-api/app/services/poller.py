from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

from app.adapters.gateway_workers import normalize_workers
from app.config import REQUIRED_SOURCES, Settings
from app.state import SnapshotStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SourceMemory:
    required: bool
    last_ok_at: Optional[str] = None


class OpsPoller:
    def __init__(self, settings: Settings, store: SnapshotStore) -> None:
        self.settings = settings
        self.store = store
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

        self._source_memory: Dict[str, SourceMemory] = {
            name: SourceMemory(required=True) for name in REQUIRED_SOURCES
        }

        self._last_good_health: Dict[str, Any] = {"status": "unknown"}
        self._last_good_workers: list[dict[str, Any]] = []
        self._last_good_signals: Dict[str, Any] = {}

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="ops-api-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            started = asyncio.get_running_loop().time()
            try:
                await self.run_once()
            except Exception:
                pass

            elapsed = asyncio.get_running_loop().time() - started
            wait_s = max(0.1, self.settings.poll_interval_seconds - elapsed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> None:
        now = _utc_now_iso()

        health_task = asyncio.create_task(self._fetch_gateway_json("/health"))
        workers_task = asyncio.create_task(self._fetch_gateway_json("/api/v1/workers"))
        signals_task = asyncio.create_task(self._fetch_gateway_json("/api/v1/signals"))

        health_payload, health_error = await health_task
        workers_payload, workers_error = await workers_task
        signals_payload, signals_error = await signals_task

        source_status: Dict[str, Dict[str, Any]] = {}

        if health_error is None and isinstance(health_payload, Mapping):
            self._last_good_health = dict(health_payload)
            source_status["gateway_health"] = self._ok_source("gateway_health", now)
        else:
            source_status["gateway_health"] = self._error_source("gateway_health", now, health_error)

        if workers_error is None and isinstance(workers_payload, Mapping):
            self._last_good_workers = normalize_workers(workers_payload)
            workers_data = list(self._last_good_workers)
            source_status["gateway_workers"] = self._ok_source("gateway_workers", now)
        else:
            workers_data = list(self._last_good_workers)
            source_status["gateway_workers"] = self._error_source("gateway_workers", now, workers_error)

        if signals_error is None and isinstance(signals_payload, Mapping):
            self._last_good_signals = dict(signals_payload)
            source_status["gateway_signals"] = self._ok_source("gateway_signals", now)
        else:
            source_status["gateway_signals"] = self._error_source("gateway_signals", now, signals_error)

        snapshot = {
            "generated_at": now,
            "last_poll_at": now,
            "health": self._last_good_health,
            "workers": workers_data,
            "signals": dict(self._last_good_signals),
            "source_status": source_status,
        }
        self.store.replace(snapshot)

    async def _fetch_gateway_json(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = f"{self.settings.gateway_base_url}{path}"
        timeout = self.settings.gateway_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.get(url)
            except Exception as exc:
                return None, f"request_failed:{type(exc).__name__}:{exc}"

        try:
            payload = resp.json()
        except Exception as exc:
            return None, f"invalid_json:{type(exc).__name__}:{exc}"

        if not isinstance(payload, Mapping):
            return None, "non_object_json"

        if int(resp.status_code) != 200:
            detail = str(payload.get("error") or payload.get("detail") or "http_error")
            return None, f"http_{resp.status_code}:{detail}"

        return dict(payload), None

    def _ok_source(self, name: str, now_iso: str) -> Dict[str, Any]:
        memory = self._source_memory.get(name)
        if memory is not None:
            memory.last_ok_at = now_iso
        return {
            "required": bool(memory.required) if memory is not None else False,
            "status": "ok",
            "error_reason": None,
            "last_attempt_at": now_iso,
            "last_ok_at": memory.last_ok_at if memory is not None else now_iso,
        }

    def _error_source(self, name: str, now_iso: str, reason: Optional[str]) -> Dict[str, Any]:
        memory = self._source_memory.get(name)
        return {
            "required": bool(memory.required) if memory is not None else False,
            "status": "error",
            "error_reason": str(reason or "unknown_error"),
            "last_attempt_at": now_iso,
            "last_ok_at": memory.last_ok_at if memory is not None else None,
        }
