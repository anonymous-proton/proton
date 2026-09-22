from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


KNOWN_STATES = {
    "cold",
    "starting",
    "ready",
    "error",
    "stopped",
}



def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)



def _normalize_state(raw: Any, ready: bool) -> str:
    value = str(raw or "").strip().lower()
    if value in KNOWN_STATES:
        return value
    return "ready" if ready else "cold"



def _normalize_gpus(raw: Any) -> List[str]:
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return []
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(text)
    return out



def normalize_workers(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_workers = payload.get("workers")
    if not isinstance(raw_workers, list):
        return []

    out: List[Dict[str, Any]] = []
    for raw in raw_workers:
        if not isinstance(raw, Mapping):
            continue
        component = str(raw.get("component") or "").strip()
        worker_name = str(raw.get("name") or raw.get("worker_name") or "").strip()
        if not component or not worker_name:
            continue

        in_queue = _to_int(raw.get("in_queue", 0))
        prepared_queue = _to_int(raw.get("prepared_queue", 0))
        output_queue = _to_int(raw.get("output_queue", 0))
        prepare_inflight = _to_int(raw.get("prepare_inflight", 0))
        execute_inflight = _to_int(raw.get("execute_inflight", 0))
        finalize_inflight = _to_int(raw.get("finalize_inflight", 0))
        computed_inflight = prepare_inflight + execute_inflight + finalize_inflight
        ready = bool(raw.get("ready", False))

        co_located = raw.get("co_located_components")
        if isinstance(co_located, list):
            co_located_components = [str(c) for c in co_located if str(c).strip()]
        else:
            co_located_components = []

        out.append(
            {
                "component": component,
                "worker_name": worker_name,
                "state": _normalize_state(raw.get("state"), ready),
                "ready": ready,
                "profile_reserved": bool(raw.get("profile_reserved", False)),
                "addr": str(raw.get("addr") or "").strip(),
                "gpu_ids": _normalize_gpus(raw.get("gpu_ids") or []),
                "in_queue": in_queue,
                "prepared_queue": prepared_queue,
                "output_queue": output_queue,
                "prepare_inflight": prepare_inflight,
                "execute_inflight": execute_inflight,
                "finalize_inflight": finalize_inflight,
                "inflight": _to_int(
                    raw.get("inflight", computed_inflight)
                ),
                "co_located_components": co_located_components,
                "signal_active_task_count": _to_int(raw.get("signal_active_task_count", 0)),
                "workload_class": str(raw.get("workload_class") or "unknown"),
                "actual_vram": _to_int(raw.get("actual_vram", 0)),
                "max_concurrency": _to_int(raw.get("max_concurrency", 1)),
            }
        )

    out.sort(key=lambda row: (row["component"], row["worker_name"]))
    return out
