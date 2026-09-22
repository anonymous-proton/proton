from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_source_status(required: bool) -> Dict[str, Any]:
    return {
        "required": bool(required),
        "status": "unknown",
        "stale": True,
        "error_reason": None,
        "last_attempt_at": None,
        "last_ok_at": None,
    }


def default_snapshot() -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "generated_at": now,
        "health": {"status": "unknown"},
        "workers": [],
        "source_status": {},
        "last_poll_at": None,
    }


class SnapshotStore:
    """Thread-safe snapshot store with atomic replacement semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: Dict[str, Any] = default_snapshot()

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def replace(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = copy.deepcopy(snapshot)
