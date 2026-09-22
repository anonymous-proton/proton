#!/usr/bin/env python3
"""Collect gateway/worker/profiling observability events into NDJSON."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "monitoring.yaml"

TIMING_KEYS = ("queue_delay_us", "prepare_us", "execute_us", "finalize_us", "total_us")
JOB_STATES = ("SUBMITTED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    packed = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _parse_number(value: Any) -> Any:
    text = str(value).strip()
    if not text:
        return ""
    low = text.lower()
    if low in {"nan", "n/a", "na"}:
        return None
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_config_path(raw_path: str, *, config_path: Path) -> Path:
    expanded = Path(str(raw_path)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    _ = config_path
    return (REPO_ROOT / expanded).resolve()


def infer_component_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except Exception:
        return ""
    return str(rel.parts[0]).strip() if rel.parts else ""


def infer_run_id_from_name(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return ""


def parse_worker_timing_payload(payload: Any) -> Optional[Dict[str, int]]:
    if not isinstance(payload, Mapping):
        return None
    out: Dict[str, int] = {}
    for key in TIMING_KEYS:
        if key not in payload:
            return None
        try:
            out[key] = int(payload.get(key, 0))
        except Exception:
            return None
    return out


def load_worker_timing_file(path: Path) -> Optional[Dict[str, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return parse_worker_timing_payload(payload)


def parse_telemetry_row(row: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, raw in row.items():
        field = str(key).strip()
        if not field:
            continue
        if field == "timestamp":
            out["sample_ts"] = str(raw).strip()
            continue
        out[field] = _parse_number(raw)
    return out


def read_new_telemetry_rows(path: Path, offset: int) -> Tuple[List[Dict[str, str]], int]:
    if not path.exists() or not path.is_file():
        return [], 0

    try:
        file_size = int(path.stat().st_size)
    except OSError:
        return [], 0
    if file_size <= 0:
        return [], 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return [], 0
        try:
            header = next(csv.reader([header_line]))
        except StopIteration:
            return [], 0
        if not header:
            return [], int(handle.tell())

        data_start = int(handle.tell())
        start = int(offset)
        if start < data_start or start > file_size:
            start = data_start
        handle.seek(start)

        rows: List[Dict[str, str]] = []
        reader = csv.reader(handle)
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                continue
            rows.append({header[idx]: values[idx] for idx in range(len(header))})
        return rows, int(handle.tell())


class StateStore:
    """SQLite-backed persistence for offsets and dedup keys."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=30.0)
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, offset INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (event_key TEXT PRIMARY KEY, seen_at TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots (snapshot_key TEXT PRIMARY KEY, digest TEXT NOT NULL)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get_offset(self, path: str) -> int:
        cur = self._conn.execute("SELECT offset FROM offsets WHERE path = ?", (str(path),))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def set_offset(self, path: str, offset: int) -> None:
        self._conn.execute(
            "INSERT INTO offsets(path, offset) VALUES(?, ?) "
            "ON CONFLICT(path) DO UPDATE SET offset=excluded.offset",
            (str(path), int(offset)),
        )
        self._conn.commit()

    def mark_seen_if_new(self, event_key: str) -> bool:
        try:
            self._conn.execute(
                "INSERT INTO seen(event_key, seen_at) VALUES(?, ?)",
                (str(event_key), _utc_now_iso()),
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def snapshot_changed(self, snapshot_key: str, digest: str) -> bool:
        cur = self._conn.execute(
            "SELECT digest FROM snapshots WHERE snapshot_key = ?",
            (str(snapshot_key),),
        )
        row = cur.fetchone()
        if row and str(row[0]) == str(digest):
            return False
        self._conn.execute(
            "INSERT INTO snapshots(snapshot_key, digest) VALUES(?, ?) "
            "ON CONFLICT(snapshot_key) DO UPDATE SET digest=excluded.digest",
            (str(snapshot_key), str(digest)),
        )
        self._conn.commit()
        return True


def load_monitoring_config(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"failed to read config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("monitoring config must be a YAML object")

    gateway_node = raw.get("gateway")
    if not isinstance(gateway_node, Mapping):
        raise ValueError("config.gateway is required")
    endpoint = str(gateway_node.get("endpoint") or "").strip().rstrip("/")
    if not endpoint:
        raise ValueError("config.gateway.endpoint is required")

    paths_node = raw.get("paths")
    if not isinstance(paths_node, Mapping):
        raise ValueError("config.paths is required")

    required_paths = (
        "profile_raw_root",
        "events_file",
        "state_db",
    )
    resolved_paths: Dict[str, Path] = {}
    for key in required_paths:
        value = str(paths_node.get(key) or "").strip()
        if not value:
            raise ValueError(f"config.paths.{key} is required")
        resolved_paths[key] = _resolve_config_path(value, config_path=path)

    collector_node = raw.get("collector")
    if not isinstance(collector_node, Mapping):
        raise ValueError("config.collector is required")
    poll_node = collector_node.get("poll_intervals")
    if not isinstance(poll_node, Mapping):
        raise ValueError("config.collector.poll_intervals is required")

    def _interval(name: str, default: float) -> float:
        if name not in poll_node:
            raise ValueError(f"config.collector.poll_intervals.{name} is required")
        value = poll_node.get(name, default)
        try:
            parsed = float(value)
        except Exception as exc:
            raise ValueError(f"config.collector.poll_intervals.{name} must be numeric") from exc
        if parsed <= 0:
            raise ValueError(f"config.collector.poll_intervals.{name} must be > 0")
        return parsed

    heartbeat_raw = collector_node.get("heartbeat_s", 30)
    try:
        heartbeat_s = float(heartbeat_raw)
    except Exception as exc:
        raise ValueError("config.collector.heartbeat_s must be numeric") from exc
    if heartbeat_s <= 0:
        raise ValueError("config.collector.heartbeat_s must be > 0")

    return {
        "gateway_endpoint": endpoint,
        "profile_raw_root": resolved_paths["profile_raw_root"],
        "events_file": resolved_paths["events_file"],
        "state_db": resolved_paths["state_db"],
        "interval_gateway_s": _interval("gateway_s", 5.0),
        "interval_profile_s": _interval("profile_s", 10.0),
        "loop_sleep_s": _interval("loop_sleep_s", 1.0),
        "heartbeat_s": heartbeat_s,
    }


class ObservabilityCollector:
    """Collects gateway state and profiling artifacts into structured events."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.gateway_endpoint = str(config["gateway_endpoint"]).rstrip("/")
        self.profile_raw_root = Path(config["profile_raw_root"]).resolve()
        self.events_file = Path(config["events_file"]).resolve()
        self.state_db = Path(config["state_db"]).resolve()

        self.interval_gateway_s = float(config["interval_gateway_s"])
        self.interval_profile_s = float(config["interval_profile_s"])
        self.loop_sleep_s = float(config["loop_sleep_s"])
        self.heartbeat_s = float(config["heartbeat_s"])

        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = StateStore(self.state_db)
        self._last_emit_at: Dict[str, float] = {}

    def close(self) -> None:
        self._state.close()

    def run_once(self) -> None:
        self.poll_gateway()
        self.scan_profile_raw()

    def run_forever(self) -> None:
        next_gateway = 0.0
        next_profile = 0.0
        while True:
            now = time.monotonic()
            if now >= next_gateway:
                self.poll_gateway()
                next_gateway = now + self.interval_gateway_s
            if now >= next_profile:
                self.scan_profile_raw()
                next_profile = now + self.interval_profile_s
            time.sleep(self.loop_sleep_s)

    def poll_gateway(self) -> None:
        workers_payload = self._request_json(f"{self.gateway_endpoint}/api/v1/workers") or {"workers": []}
        workers = workers_payload.get("workers")
        if isinstance(workers, list):
            for worker in sorted(
                [item for item in workers if isinstance(item, Mapping)],
                key=lambda item: (str(item.get("component") or ""), str(item.get("name") or "")),
            ):
                component = str(worker.get("component") or "").strip()
                name = str(worker.get("name") or "").strip()
                in_queue = _to_int(worker.get("in_queue", 0))
                prepared_queue = _to_int(worker.get("prepared_queue", 0))
                output_queue = _to_int(worker.get("output_queue", 0))
                prepare_inflight = _to_int(worker.get("prepare_inflight", 0))
                execute_inflight = _to_int(worker.get("execute_inflight", 0))
                finalize_inflight = _to_int(worker.get("finalize_inflight", 0))
                total_queue_depth = in_queue + prepared_queue + output_queue
                total_inflight = prepare_inflight + execute_inflight + finalize_inflight
                payload = {
                    "component": component,
                    "worker_name": name,
                    "state": str(worker.get("state") or ""),
                    "ready": bool(worker.get("ready", False)),
                    "ready_int": 1 if bool(worker.get("ready", False)) else 0,
                    "profile_reserved": bool(worker.get("profile_reserved", False)),
                    "addr": str(worker.get("addr") or ""),
                    "gpu_ids": [str(item) for item in list(worker.get("gpu_ids") or []) if str(item)],
                    "inflight": _to_int(worker.get("inflight", total_inflight)),
                    "in_queue": in_queue,
                    "prepared_queue": prepared_queue,
                    "output_queue": output_queue,
                    "prepare_inflight": prepare_inflight,
                    "execute_inflight": execute_inflight,
                    "finalize_inflight": finalize_inflight,
                    "total_queue_depth": total_queue_depth,
                    "total_inflight": total_inflight,
                }
                snap_key = f"worker_snapshot:{component}:{name}"
                changed = self._state.snapshot_changed(snap_key, _canonical_digest(payload))
                now = time.monotonic()
                if changed or self._heartbeat_due(snap_key, now=now):
                    self._emit("worker_snapshot", payload)
                    self._mark_emitted(snap_key, now=now)

        task_runs_payload = (
            self._request_json(f"{self.gateway_endpoint}/api/v1/profile/runs?run_source=task&limit=1000&sort=updated_at:desc")
            or {}
        )
        task_runs = task_runs_payload.get("runs") if isinstance(task_runs_payload.get("runs"), list) else []

        task_summary: Dict[str, int] = {state: 0 for state in JOB_STATES}
        for row in task_runs:
            if not isinstance(row, Mapping):
                continue
            state = str(row.get("state") or "").strip().upper()
            if state == "ANALYZING":
                state = "RUNNING"
            if state in task_summary:
                task_summary[state] = int(task_summary.get(state, 0)) + 1

        flat: Dict[str, Any] = {
            "count": int(task_runs_payload.get("total", len(task_runs)) or 0),
            "count_task_runs": int(task_runs_payload.get("total", len(task_runs)) or 0),
        }
        for state in JOB_STATES:
            flat[f"state_{state.lower()}"] = int(task_summary.get(state, 0))
            flat[f"task_{state.lower()}"] = int(task_summary.get(state, 0))

        failed_jobs: List[Dict[str, Any]] = []
        for item in task_runs:
            if not isinstance(item, Mapping):
                continue
            state = str(item.get("state") or "").strip().upper()
            if state not in {"FAILED", "CANCELLED"}:
                continue
            failed_jobs.append(
                {
                    "job_id": str(item.get("run_key") or ""),
                    "kind": "task_run",
                    "component": str(item.get("component") or ""),
                    "error": str(item.get("error") or ""),
                    "updated_at": item.get("updated_at"),
                }
            )
        failed_jobs.sort(key=lambda row: float(row.get("updated_at") or 0.0), reverse=True)
        if failed_jobs:
            flat["failed_jobs_json"] = json.dumps(failed_jobs[:10], ensure_ascii=True, separators=(",", ":"))
        else:
            flat["failed_jobs_json"] = "[]"

        changed = self._state.snapshot_changed("job_summary", _canonical_digest(flat))
        now = time.monotonic()
        if changed or self._heartbeat_due("job_summary", now=now):
            self._emit("job_summary", flat)
            self._mark_emitted("job_summary", now=now)

    def scan_profile_raw(self) -> None:
        if not self.profile_raw_root.exists():
            return

        for timing_path in sorted(self.profile_raw_root.rglob("*_worker_timing.json")):
            if not timing_path.is_file():
                continue
            try:
                stat = timing_path.stat()
            except OSError:
                continue
            dedup_key = f"worker_timing:{timing_path}:{stat.st_mtime_ns}:{stat.st_size}"
            if not self._state.mark_seen_if_new(dedup_key):
                continue

            timing = load_worker_timing_file(timing_path)
            if not timing:
                continue

            component = infer_component_from_path(timing_path, self.profile_raw_root)
            run_id = infer_run_id_from_name(timing_path, "_worker_timing.json")
            payload: Dict[str, Any] = {
                "component": component,
                "run_id": run_id,
                "path": str(timing_path),
            }
            payload.update(timing)
            self._emit("worker_timing", payload)

        for telemetry_path in sorted(self.profile_raw_root.rglob("*_telemetry.csv")):
            if not telemetry_path.is_file():
                continue
            path_key = str(telemetry_path.resolve())
            old_offset = self._state.get_offset(path_key)
            rows, new_offset = read_new_telemetry_rows(telemetry_path, old_offset)
            if new_offset != old_offset:
                self._state.set_offset(path_key, new_offset)
            if not rows:
                continue

            component = infer_component_from_path(telemetry_path, self.profile_raw_root)
            run_id = infer_run_id_from_name(telemetry_path, "_telemetry.csv")
            for row in rows:
                payload = parse_telemetry_row(row)
                payload["component"] = component
                payload["run_id"] = run_id
                payload["path"] = str(telemetry_path)
                self._emit("telemetry_sample", payload)

    def _emit(self, event_name: str, payload: Mapping[str, Any]) -> None:
        event = {"ts": _utc_now_iso(), "event": str(event_name)}
        event.update(dict(payload))
        with self.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")

    def _heartbeat_due(self, snapshot_key: str, *, now: Optional[float] = None) -> bool:
        tick = time.monotonic() if now is None else float(now)
        last = self._last_emit_at.get(str(snapshot_key))
        if last is None:
            return True
        return (tick - last) >= self.heartbeat_s

    def _mark_emitted(self, snapshot_key: str, *, now: Optional[float] = None) -> None:
        tick = time.monotonic() if now is None else float(now)
        self._last_emit_at[str(snapshot_key)] = tick

    @staticmethod
    def _request_json(url: str, *, timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8") if resp else ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return dict(payload) if isinstance(payload, Mapping) else {}


def apply_runtime_overrides(
    config: Mapping[str, Any],
    *,
    gateway_endpoint: str = "",
    heartbeat_s: Optional[float] = None,
) -> Dict[str, Any]:
    merged = dict(config)
    if str(gateway_endpoint or "").strip():
        merged["gateway_endpoint"] = str(gateway_endpoint).strip().rstrip("/")
    if heartbeat_s is not None:
        parsed = float(heartbeat_s)
        if parsed <= 0:
            raise ValueError("heartbeat_s override must be > 0")
        merged["heartbeat_s"] = parsed
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to monitoring YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one collection pass and exit.",
    )
    parser.add_argument(
        "--gateway-endpoint",
        default="",
        help="Override gateway endpoint (e.g. http://127.0.0.7:8098)",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=None,
        help="Snapshot heartbeat interval in seconds (default: config value, 30)",
    )
    args = parser.parse_args()

    config = apply_runtime_overrides(
        load_monitoring_config(Path(args.config).expanduser().resolve()),
        gateway_endpoint=args.gateway_endpoint,
        heartbeat_s=args.heartbeat_s,
    )
    collector = ObservabilityCollector(config)
    try:
        if args.once:
            collector.run_once()
        else:
            collector.run_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        collector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
