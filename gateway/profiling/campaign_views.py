"""Campaign-oriented read models for gateway operations endpoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
import sqlite3
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..gateway_identity import (
    UNKNOWN_GATEWAY_VALUE,
    gateway_payload_from_mapping,
    normalize_gateway_git_commit,
    normalize_gateway_instance_id,
)
from .index_store import RunIndexStore

UNKNOWN_CAMPAIGN_ID = "__unknown__"
_ACTIVE_STATES = {"SUBMITTED", "RUNNING"}
_FAILED_TERMINAL_STATES = {"FAILED", "CANCELLED"}


@dataclass(frozen=True)
class _CampaignCatalog:
    all_active_rows: List[Dict[str, Any]]
    registry_rows: Dict[str, Dict[str, Any]]
    active_by_campaign: Dict[str, List[Dict[str, Any]]]
    filtered_registry_rows: Dict[str, Dict[str, Any]]
    aggregate_rows: Dict[str, Dict[str, Any]]
    component_rows: Dict[Tuple[str, str], Dict[str, Any]]
    campaign_source_rows: Dict[str, List[Dict[str, Any]]]
    unassigned_summary: Dict[str, Any]
    available_filter_rows: List[Dict[str, Any]]


def _utc_iso(ts: float | None) -> Optional[str]:
    if ts is None:
        return None
    try:
        value = float(ts)
    except Exception:
        return None
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")


def _to_float(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    if value != value:
        return None
    return value


def _to_int(raw: Any) -> int:
    try:
        return int(raw)
    except Exception:
        return 0


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _normalize_campaign_id(raw: Any) -> str:
    token = str(raw or "").strip()
    if not token or token == UNKNOWN_CAMPAIGN_ID:
        return ""
    return token


def _normalize_status(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token in {"waiting", "active", "completed", "failed", "unknown"}:
        return token
    return "unknown"


def _status_rank(status: str) -> int:
    if status == "active":
        return 0
    if status == "waiting":
        return 1
    if status == "failed":
        return 2
    if status == "completed":
        return 3
    return 4


def _max_ts(*values: Optional[float]) -> Optional[float]:
    out: Optional[float] = None
    for value in values:
        if value is None:
            continue
        out = value if out is None else max(out, value)
    return out


def _min_ts(*values: Optional[float]) -> Optional[float]:
    out: Optional[float] = None
    for value in values:
        if value is None:
            continue
        out = value if out is None else min(out, value)
    return out


def derive_campaign_status(active_states: list[str], terminal_states: list[str]) -> str:
    normalized_active = {str(item or "").strip().upper() for item in list(active_states or [])}
    normalized_terminal = {
        str(item or "").strip().upper()
        for item in list(terminal_states or [])
        if str(item or "").strip()
    }
    if normalized_active & _ACTIVE_STATES:
        return "active"
    if normalized_terminal & _FAILED_TERMINAL_STATES:
        return "failed"
    if normalized_terminal and normalized_terminal.issubset({"SUCCEEDED"}):
        return "completed"
    return "unknown"


def _derive_queue_status(
    *,
    submitted: int,
    running: int,
    succeeded: int,
    failed: int,
    cancelled: int,
    observed: int,
    has_registry: bool,
    registry_status: str,
) -> str:
    if submitted > 0 or running > 0:
        return "active"
    if failed > 0 or cancelled > 0:
        return "failed"
    if observed > 0 and succeeded > 0 and (failed + cancelled) == 0:
        return "completed"
    if has_registry:
        if registry_status in {"waiting", "active", "completed", "failed"}:
            return registry_status
        return "waiting"
    return "unknown"


def _connect(run_index: RunIndexStore) -> sqlite3.Connection:
    cur_name = threading.current_thread().name
    if (
        not cur_name.startswith("sqlite-")
        and not os.environ.get("GATEWAY_SQLITE_THREAD_BYPASS")
    ):
        stack_summary = "".join(traceback.format_stack()[-5:-1])
        raise RuntimeError(
            f"campaign_views._connect from non-sqlite thread {cur_name!r}.  "
            f"All SQLite access must route through "
            f"GatewayHTTPService._run_index_call (sqlite-* dedicated "
            f"executor, max_workers=1).  This guard prevents -class "
            f"fcntl deadlock.  Set env GATEWAY_SQLITE_THREAD_BYPASS=1 "
            f"to bypass for unit tests / scripts.\n"
            f"Caller stack:\n{stack_summary}"
        )
    conn = sqlite3.connect(str(run_index.db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _normalized_gateway_filters(
    *,
    include_gateway_instance_ids: Sequence[str],
    exclude_gateway_instance_ids: Sequence[str],
    include_gateway_git_commits: Sequence[str],
    exclude_gateway_git_commits: Sequence[str],
) -> Dict[str, List[str]]:
    return {
        "include_gateway_instance_ids": list(
            dict.fromkeys(
                normalize_gateway_instance_id(item)
                for item in list(include_gateway_instance_ids or [])
                if str(item or "").strip()
            )
        ),
        "exclude_gateway_instance_ids": list(
            dict.fromkeys(
                normalize_gateway_instance_id(item)
                for item in list(exclude_gateway_instance_ids or [])
                if str(item or "").strip()
            )
        ),
        "include_gateway_git_commits": list(
            dict.fromkeys(
                normalize_gateway_git_commit(item)
                for item in list(include_gateway_git_commits or [])
                if str(item or "").strip()
            )
        ),
        "exclude_gateway_git_commits": list(
            dict.fromkeys(
                normalize_gateway_git_commit(item)
                for item in list(exclude_gateway_git_commits or [])
                if str(item or "").strip()
            )
        ),
    }


def _gateway_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    keys = set(row.keys()) if hasattr(row, "keys") else set(row)
    gateway = row["gateway"] if "gateway" in keys else None
    if isinstance(gateway, Mapping):
        return gateway_payload_from_mapping(gateway)
    return gateway_payload_from_mapping(
        {
            "instance_id": row["gateway_instance_id"] if "gateway_instance_id" in keys else None,
            "bind_addr": row["gateway_bind_addr"] if "gateway_bind_addr" in keys else None,
            "git_commit": row["gateway_git_commit"] if "gateway_git_commit" in keys else None,
            "started_at": row["gateway_started_at"] if "gateway_started_at" in keys else None,
        }
    )


def _gateway_matches_filters(gateway: Mapping[str, Any], filters: Mapping[str, Sequence[str]]) -> bool:
    instance_id = normalize_gateway_instance_id(gateway.get("instance_id"))
    git_commit = normalize_gateway_git_commit(gateway.get("git_commit"))
    include_instances = set(filters.get("include_gateway_instance_ids") or [])
    exclude_instances = set(filters.get("exclude_gateway_instance_ids") or [])
    include_commits = set(filters.get("include_gateway_git_commits") or [])
    exclude_commits = set(filters.get("exclude_gateway_git_commits") or [])
    if include_instances and instance_id not in include_instances:
        return False
    if exclude_instances and instance_id in exclude_instances:
        return False
    if include_commits and git_commit not in include_commits:
        return False
    if exclude_commits and git_commit in exclude_commits:
        return False
    return True


def _source_entry(*, gateway: Mapping[str, Any], count: int, last_seen_at: Optional[float] = None) -> Dict[str, Any]:
    payload = gateway_payload_from_mapping(gateway)
    return {
        "instance_id": payload["instance_id"],
        "bind_addr": payload["bind_addr"],
        "git_commit": payload["git_commit"],
        "label": payload["label"],
        "count": int(count),
        "last_seen_at": _utc_iso(last_seen_at),
    }


def _sort_source_entries(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [dict(item) for item in list(entries or [])],
        key=lambda item: (
            -(int(item.get("count") or 0)),
            str(item.get("label") or ""),
            str(item.get("instance_id") or ""),
        ),
    )


def _summarize_sources(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in list(rows or []):
        gateway = _gateway_from_row(row)
        key = (
            str(gateway.get("instance_id") or ""),
            str(gateway.get("bind_addr") or ""),
            str(gateway.get("git_commit") or ""),
        )
        bucket = buckets.setdefault(
            key,
            {
                "gateway": gateway,
                "count": 0,
                "last_seen_at": None,
            },
        )
        bucket["count"] = int(bucket["count"]) + int(row.get("count") or 1)
        bucket["last_seen_at"] = _max_ts(bucket.get("last_seen_at"), _to_float(row.get("last_seen_at")))
    return _sort_source_entries(
        [
            _source_entry(
                gateway=bucket["gateway"],
                count=int(bucket["count"]),
                last_seen_at=_to_float(bucket["last_seen_at"]),
            )
            for bucket in buckets.values()
        ]
    )


def _available_gateway_filters(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    instance_buckets: Dict[str, Dict[str, Any]] = {}
    commit_buckets: Dict[str, Dict[str, Any]] = {}
    for row in list(rows or []):
        gateway = _gateway_from_row(row)
        count = int(row.get("count") or 1)
        first_seen_at = _to_float(row.get("first_seen_at"))
        last_seen_at = _to_float(row.get("last_seen_at"))
        if first_seen_at is None:
            first_seen_at = last_seen_at
        if last_seen_at is None:
            last_seen_at = first_seen_at
        instance_id = str(gateway.get("instance_id") or UNKNOWN_GATEWAY_VALUE)
        git_commit = str(gateway.get("git_commit") or UNKNOWN_GATEWAY_VALUE)
        instance_bucket = instance_buckets.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "label": str(gateway.get("label") or UNKNOWN_GATEWAY_VALUE),
                "bind_addr": str(gateway.get("bind_addr") or UNKNOWN_GATEWAY_VALUE),
                "git_commit": git_commit,
                "count": 0,
                "first_seen_at": None,
                "last_seen_at": None,
            },
        )
        instance_bucket["count"] = int(instance_bucket["count"]) + count
        instance_bucket["first_seen_at"] = _min_ts(_to_float(instance_bucket.get("first_seen_at")), first_seen_at)
        instance_bucket["last_seen_at"] = _max_ts(_to_float(instance_bucket.get("last_seen_at")), last_seen_at)
        commit_bucket = commit_buckets.setdefault(
            git_commit,
            {
                "git_commit": git_commit,
                "label": git_commit[:7] if git_commit != UNKNOWN_GATEWAY_VALUE else UNKNOWN_GATEWAY_VALUE,
                "count": 0,
                "first_seen_at": None,
                "last_seen_at": None,
            },
        )
        commit_bucket["count"] = int(commit_bucket["count"]) + count
        commit_bucket["first_seen_at"] = _min_ts(_to_float(commit_bucket.get("first_seen_at")), first_seen_at)
        commit_bucket["last_seen_at"] = _max_ts(_to_float(commit_bucket.get("last_seen_at")), last_seen_at)

    instance_rows = []
    for item in instance_buckets.values():
        payload = dict(item)
        payload["first_seen_at"] = _utc_iso(_to_float(item.get("first_seen_at")))
        payload["last_seen_at"] = _utc_iso(_to_float(item.get("last_seen_at")))
        instance_rows.append(payload)

    commit_rows = []
    for item in commit_buckets.values():
        payload = dict(item)
        payload["first_seen_at"] = _utc_iso(_to_float(item.get("first_seen_at")))
        payload["last_seen_at"] = _utc_iso(_to_float(item.get("last_seen_at")))
        commit_rows.append(payload)

    return {
        "instances": sorted(
            instance_rows,
            key=lambda item: (-(int(item.get("count") or 0)), str(item.get("label") or "")),
        ),
        "commits": sorted(
            commit_rows,
            key=lambda item: (-(int(item.get("count") or 0)), str(item.get("label") or "")),
        ),
    }


def _active_task_rows(active_tasks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(active_tasks or []):
        state = str(item.get("state") or "").strip().upper()
        if state not in _ACTIVE_STATES:
            continue
        is_dispatched = bool(item.get("is_dispatched", state == "RUNNING"))
        rows.append(
            {
                "campaign_id": _normalize_campaign_id(item.get("campaign_id")),
                "run_name": str(item.get("run_name") or "").strip(),
                "submitter": str(item.get("submitter") or "").strip(),
                "component": str(item.get("component") or "").strip().lower(),
                "state": state,
                "created_at": _to_float(item.get("created_at")),
                "updated_at": _to_float(item.get("updated_at")),
                "task_id": str(item.get("task_id") or "").strip(),
                "run_id": str(item.get("run_id") or "").strip(),
                "is_dispatched": is_dispatched,
                "gateway": _gateway_from_row(item),
            }
        )
    return rows


def _append_gateway_sql_filters(
    where_clauses: List[str],
    params: List[Any],
    *,
    alias: str,
    filters: Mapping[str, Sequence[str]],
) -> None:
    include_instances = list(filters.get("include_gateway_instance_ids") or [])
    exclude_instances = list(filters.get("exclude_gateway_instance_ids") or [])
    include_commits = list(filters.get("include_gateway_git_commits") or [])
    exclude_commits = list(filters.get("exclude_gateway_git_commits") or [])
    instance_expr = (
        f"CASE WHEN TRIM(COALESCE({alias}.gateway_instance_id, '')) = '' THEN ? "
        f"ELSE TRIM(COALESCE({alias}.gateway_instance_id, '')) END"
    )
    commit_expr = (
        f"CASE WHEN LOWER(TRIM(COALESCE({alias}.gateway_git_commit, ''))) = '' THEN ? "
        f"ELSE LOWER(TRIM(COALESCE({alias}.gateway_git_commit, ''))) END"
    )
    if include_instances:
        placeholders = ",".join("?" for _ in include_instances)
        where_clauses.append(f"{instance_expr} IN ({placeholders})")
        params.extend([UNKNOWN_GATEWAY_VALUE, *include_instances])
    if exclude_instances:
        placeholders = ",".join("?" for _ in exclude_instances)
        where_clauses.append(f"{instance_expr} NOT IN ({placeholders})")
        params.extend([UNKNOWN_GATEWAY_VALUE, *exclude_instances])
    if include_commits:
        placeholders = ",".join("?" for _ in include_commits)
        where_clauses.append(f"{commit_expr} IN ({placeholders})")
        params.extend([UNKNOWN_GATEWAY_VALUE, *include_commits])
    if exclude_commits:
        placeholders = ",".join("?" for _ in exclude_commits)
        where_clauses.append(f"{commit_expr} NOT IN ({placeholders})")
        params.extend([UNKNOWN_GATEWAY_VALUE, *exclude_commits])


def _fetch_registry_rows(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT campaign_id, run_name, submitter,
               gateway_instance_id, gateway_bind_addr, gateway_git_commit, gateway_started_at,
               status, created_at, started_at, last_seen_at, ended_at, updated_at
        FROM campaign_registry
        WHERE TRIM(COALESCE(campaign_id, '')) != ''
          AND campaign_id != ?
        """,
        (UNKNOWN_CAMPAIGN_ID,),
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        campaign_id = _normalize_campaign_id(row["campaign_id"])
        if not campaign_id:
            continue
        out[campaign_id] = {
            "campaign_id": campaign_id,
            "run_name": str(row["run_name"] or "").strip(),
            "submitter": str(row["submitter"] or "").strip(),
            "status": _normalize_status(row["status"]),
            "created_at": _to_float(row["created_at"]),
            "started_at": _to_float(row["started_at"]),
            "last_seen_at": _to_float(row["last_seen_at"]),
            "ended_at": _to_float(row["ended_at"]),
            "updated_at": _to_float(row["updated_at"]),
            "gateway": _gateway_from_row(row),
        }
    return out


def _fetch_aggregate_rows(conn: sqlite3.Connection, *, filters: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, Any]]:
    where_clauses = [
        "r.run_source = 'task'",
        "COALESCE(r.campaign_metadata_version, 0) >= 1",
        "TRIM(COALESCE(r.campaign_id, '')) != ''",
        "r.campaign_id != ?",
    ]
    params: List[Any] = [UNKNOWN_CAMPAIGN_ID]
    _append_gateway_sql_filters(where_clauses, params, alias="r", filters=filters)
    rows = conn.execute(
        f"""
        SELECT
            r.campaign_id AS campaign_id,
            MAX(TRIM(COALESCE(r.run_name, ''))) AS run_name,
            MAX(TRIM(COALESCE(r.submitter, ''))) AS submitter,
            MIN(r.created_at) AS started_at,
            MAX(COALESCE(r.updated_at, r.finished_at, r.created_at)) AS last_seen_at,
            MAX(r.finished_at) AS ended_at,
            COUNT(*) AS observed_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' THEN 1 ELSE 0 END), 0) AS succeeded_count,
            COALESCE(SUM(CASE WHEN r.state = 'FAILED' THEN 1 ELSE 0 END), 0) AS failed_count,
            COALESCE(SUM(CASE WHEN r.state = 'CANCELLED' THEN 1 ELSE 0 END), 0) AS cancelled_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' AND r.runtime_sec IS NOT NULL THEN 1 ELSE 0 END), 0) AS runtime_present_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' AND r.peak_memory_mib IS NOT NULL THEN 1 ELSE 0 END), 0) AS peak_memory_present_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' AND r.mean_gpu_util_percent IS NOT NULL THEN 1 ELSE 0 END), 0) AS mean_gpu_util_present_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' AND r.telemetry_wall_clock_sec IS NOT NULL THEN 1 ELSE 0 END), 0) AS wallclock_present_count
        FROM profile_runs r
        WHERE {" AND ".join(where_clauses)}
        GROUP BY r.campaign_id
        """,
        params,
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        campaign_id = _normalize_campaign_id(row["campaign_id"])
        if not campaign_id:
            continue
        succeeded_count = _to_int(row["succeeded_count"])
        failed_count = _to_int(row["failed_count"])
        cancelled_count = _to_int(row["cancelled_count"])
        observed_count = _to_int(row["observed_count"])
        out[campaign_id] = {
            "campaign_id": campaign_id,
            "run_name": str(row["run_name"] or "").strip(),
            "submitter": str(row["submitter"] or "").strip(),
            "started_at": _to_float(row["started_at"]),
            "last_seen_at": _to_float(row["last_seen_at"]),
            "ended_at": _to_float(row["ended_at"]),
            "task_counts": {
                "submitted": 0,
                "running": 0,
                "succeeded": succeeded_count,
                "failed": failed_count,
                "cancelled": cancelled_count,
                "observed": observed_count,
                "total": observed_count,
            },
            "telemetry_readiness": {
                "succeeded_rows": succeeded_count,
                "runtime_present_rate": _ratio(_to_int(row["runtime_present_count"]), succeeded_count),
                "peak_memory_present_rate": _ratio(_to_int(row["peak_memory_present_count"]), succeeded_count),
                "mean_gpu_util_present_rate": _ratio(_to_int(row["mean_gpu_util_present_count"]), succeeded_count),
                "wallclock_present_rate": _ratio(_to_int(row["wallclock_present_count"]), succeeded_count),
            },
        }
    return out


def _fetch_component_rows(
    conn: sqlite3.Connection,
    *,
    filters: Mapping[str, Sequence[str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    where_clauses = [
        "r.run_source = 'task'",
        "COALESCE(r.campaign_metadata_version, 0) >= 1",
        "TRIM(COALESCE(r.campaign_id, '')) != ''",
        "r.campaign_id != ?",
    ]
    params: List[Any] = [UNKNOWN_CAMPAIGN_ID]
    _append_gateway_sql_filters(where_clauses, params, alias="r", filters=filters)
    rows = conn.execute(
        f"""
        SELECT
            r.campaign_id AS campaign_id,
            LOWER(TRIM(COALESCE(r.component, ''))) AS component,
            COUNT(*) AS observed_count,
            COALESCE(SUM(CASE WHEN r.state = 'SUCCEEDED' THEN 1 ELSE 0 END), 0) AS succeeded_count,
            COALESCE(SUM(CASE WHEN r.state IN ('FAILED', 'CANCELLED') THEN 1 ELSE 0 END), 0) AS failed_count,
            MAX(COALESCE(r.updated_at, r.finished_at, r.created_at)) AS last_seen_at
        FROM profile_runs r
        WHERE {" AND ".join(where_clauses)}
        GROUP BY r.campaign_id, component
        """,
        params,
    ).fetchall()
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        campaign_id = _normalize_campaign_id(row["campaign_id"])
        component = str(row["component"] or "").strip()
        if not campaign_id or not component:
            continue
        out[(campaign_id, component)] = {
            "component": component,
            "observed": _to_int(row["observed_count"]),
            "active": 0,
            "succeeded": _to_int(row["succeeded_count"]),
            "failed": _to_int(row["failed_count"]),
            "last_seen_at": _utc_iso(_to_float(row["last_seen_at"])),
        }
    return out


def _fetch_campaign_source_rows(
    conn: sqlite3.Connection,
    *,
    filters: Mapping[str, Sequence[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    where_clauses = [
        "r.run_source = 'task'",
        "COALESCE(r.campaign_metadata_version, 0) >= 1",
        "TRIM(COALESCE(r.campaign_id, '')) != ''",
        "r.campaign_id != ?",
    ]
    params: List[Any] = [UNKNOWN_CAMPAIGN_ID]
    _append_gateway_sql_filters(where_clauses, params, alias="r", filters=filters)
    rows = conn.execute(
        f"""
        SELECT
            r.campaign_id AS campaign_id,
            CASE WHEN TRIM(COALESCE(r.gateway_instance_id, '')) = '' THEN ? ELSE TRIM(COALESCE(r.gateway_instance_id, '')) END AS gateway_instance_id,
            CASE WHEN TRIM(COALESCE(r.gateway_bind_addr, '')) = '' THEN ? ELSE TRIM(COALESCE(r.gateway_bind_addr, '')) END AS gateway_bind_addr,
            CASE WHEN LOWER(TRIM(COALESCE(r.gateway_git_commit, ''))) = '' THEN ? ELSE LOWER(TRIM(COALESCE(r.gateway_git_commit, ''))) END AS gateway_git_commit,
            MAX(r.gateway_started_at) AS gateway_started_at,
            MAX(COALESCE(r.updated_at, r.finished_at, r.created_at)) AS last_seen_at,
            COUNT(*) AS source_count
        FROM profile_runs r
        WHERE {" AND ".join(where_clauses)}
        GROUP BY r.campaign_id, gateway_instance_id, gateway_bind_addr, gateway_git_commit
        """,
        [UNKNOWN_GATEWAY_VALUE, UNKNOWN_GATEWAY_VALUE, UNKNOWN_GATEWAY_VALUE, *params],
    ).fetchall()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        campaign_id = _normalize_campaign_id(row["campaign_id"])
        if not campaign_id:
            continue
        out.setdefault(campaign_id, []).append(
            {
                "gateway": _gateway_from_row(row),
                "count": _to_int(row["source_count"]),
                "last_seen_at": _to_float(row["last_seen_at"]),
            }
        )
    return out


def _fetch_unassigned_summary(
    conn: sqlite3.Connection,
    *,
    filters: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    where_clauses = [
        "run_source = 'task'",
        "("
        "TRIM(COALESCE(campaign_id, '')) = '' "
        "OR campaign_id = ? "
        "OR COALESCE(campaign_metadata_version, 0) < 1"
        ")",
    ]
    params: List[Any] = [UNKNOWN_CAMPAIGN_ID]
    _append_gateway_sql_filters(where_clauses, params, alias="profile_runs", filters=filters)
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            MAX(COALESCE(updated_at, finished_at, created_at)) AS last_seen_at
        FROM profile_runs
        WHERE {" AND ".join(where_clauses)}
        """,
        params,
    ).fetchone()
    reason_rows = conn.execute(
        f"""
        SELECT reason, COUNT(*) AS count
        FROM (
            SELECT
                CASE
                    WHEN TRIM(COALESCE(campaign_id, '')) = '' THEN 'missing_campaign_id'
                    WHEN campaign_id = ? THEN 'synthetic_unknown_id'
                    WHEN COALESCE(campaign_metadata_version, 0) < 1 THEN 'legacy_metadata_version'
                    ELSE 'other'
                END AS reason
            FROM profile_runs
            WHERE {" AND ".join(where_clauses)}
        ) t
        GROUP BY reason
        ORDER BY reason ASC
        """,
        [UNKNOWN_CAMPAIGN_ID, *params],
    ).fetchall()
    return {
        "rows": _to_int(row["row_count"]) if row else 0,
        "last_seen_at": _utc_iso(_to_float(row["last_seen_at"]) if row else None),
        "reasons": [
            {"reason": str(item["reason"] or ""), "count": _to_int(item["count"])}
            for item in reason_rows
            if _to_int(item["count"]) > 0
        ],
    }


def _fetch_available_gateway_filter_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            CASE WHEN TRIM(COALESCE(gateway_instance_id, '')) = '' THEN ? ELSE TRIM(COALESCE(gateway_instance_id, '')) END AS gateway_instance_id,
            CASE WHEN TRIM(COALESCE(gateway_bind_addr, '')) = '' THEN ? ELSE TRIM(COALESCE(gateway_bind_addr, '')) END AS gateway_bind_addr,
            CASE WHEN LOWER(TRIM(COALESCE(gateway_git_commit, ''))) = '' THEN ? ELSE LOWER(TRIM(COALESCE(gateway_git_commit, ''))) END AS gateway_git_commit,
            MAX(gateway_started_at) AS gateway_started_at,
            MIN(COALESCE(created_at, updated_at, finished_at)) AS first_seen_at,
            MAX(COALESCE(updated_at, finished_at, created_at)) AS last_seen_at,
            COUNT(*) AS source_count
        FROM profile_runs
        WHERE run_source = 'task'
        GROUP BY gateway_instance_id, gateway_bind_addr, gateway_git_commit
        """,
        (UNKNOWN_GATEWAY_VALUE, UNKNOWN_GATEWAY_VALUE, UNKNOWN_GATEWAY_VALUE),
    ).fetchall()
    return [
        {
            "gateway": _gateway_from_row(row),
            "count": _to_int(row["source_count"]),
            "first_seen_at": _to_float(row["first_seen_at"]),
            "last_seen_at": _to_float(row["last_seen_at"]),
        }
        for row in rows
    ]


def _build_campaign_row(
    *,
    campaign_id: str,
    registry_row: Optional[Mapping[str, Any]],
    aggregate_row: Optional[Mapping[str, Any]],
    components: List[Dict[str, Any]],
    submitted_count: int,
    running_count: int,
    source_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    agg_counts = dict((aggregate_row or {}).get("task_counts") or {})
    observed_count = _to_int(agg_counts.get("observed"))
    succeeded_count = _to_int(agg_counts.get("succeeded"))
    failed_count = _to_int(agg_counts.get("failed"))
    cancelled_count = _to_int(agg_counts.get("cancelled"))
    submitted_count = max(0, int(submitted_count))
    running_count = max(0, int(running_count))
    total_count = observed_count + submitted_count + running_count

    registry_status = _normalize_status((registry_row or {}).get("status"))
    status = _derive_queue_status(
        submitted=submitted_count,
        running=running_count,
        succeeded=succeeded_count,
        failed=failed_count,
        cancelled=cancelled_count,
        observed=observed_count,
        has_registry=registry_row is not None,
        registry_status=registry_status,
    )

    started_at = _min_ts(
        _to_float((registry_row or {}).get("started_at")),
        _to_float((aggregate_row or {}).get("started_at")),
    )
    if started_at is None:
        started_at = _to_float((registry_row or {}).get("created_at"))
    last_seen_at = _max_ts(
        _to_float((registry_row or {}).get("last_seen_at")),
        _to_float((registry_row or {}).get("updated_at")),
        _to_float((aggregate_row or {}).get("last_seen_at")),
    )
    ended_at = _max_ts(
        _to_float((registry_row or {}).get("ended_at")),
        _to_float((aggregate_row or {}).get("ended_at")),
    )
    if status in {"active", "waiting"}:
        ended_at = None

    telemetry_readiness = dict((aggregate_row or {}).get("telemetry_readiness") or {})
    if not telemetry_readiness:
        telemetry_readiness = {
            "succeeded_rows": 0,
            "runtime_present_rate": 0.0,
            "peak_memory_present_rate": 0.0,
            "mean_gpu_util_present_rate": 0.0,
            "wallclock_present_rate": 0.0,
        }

    effective_source_summary = list(source_summary)
    if not effective_source_summary and registry_row is not None:
        effective_source_summary = [
            _source_entry(
                gateway=_gateway_from_row(registry_row),
                count=1,
                last_seen_at=_to_float(registry_row.get("last_seen_at")),
            )
        ]

    return {
        "campaign_id": campaign_id,
        "run_name": str((registry_row or {}).get("run_name") or (aggregate_row or {}).get("run_name") or ""),
        "submitter": str((registry_row or {}).get("submitter") or (aggregate_row or {}).get("submitter") or ""),
        "status": status,
        "started_at": _utc_iso(started_at),
        "last_seen_at": _utc_iso(last_seen_at),
        "ended_at": _utc_iso(ended_at),
        "task_counts": {
            "submitted": submitted_count,
            "running": running_count,
            "succeeded": succeeded_count,
            "failed": failed_count,
            "cancelled": cancelled_count,
            "observed": observed_count,
            "total": total_count,
        },
        "component_summary": components[:8],
        "telemetry_readiness": telemetry_readiness,
        "gateway_summary": effective_source_summary[:6],
    }


def _load_campaign_catalog(
    *,
    run_index: RunIndexStore,
    active_tasks: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Sequence[str]],
    include_available_filter_rows: bool,
) -> _CampaignCatalog:
    all_active_rows = _active_task_rows(active_tasks)
    active_rows = [row for row in all_active_rows if _gateway_matches_filters(row.get("gateway") or {}, filters)]
    active_by_campaign: Dict[str, List[Dict[str, Any]]] = {}
    unassigned_active_count = 0
    unassigned_active_last_seen: Optional[float] = None
    for row in active_rows:
        campaign_id = _normalize_campaign_id(row.get("campaign_id"))
        if not campaign_id:
            unassigned_active_count += 1
            unassigned_active_last_seen = _max_ts(unassigned_active_last_seen, _to_float(row.get("updated_at")))
            continue
        active_by_campaign.setdefault(campaign_id, []).append(dict(row))

    with _connect(run_index) as conn:
        registry_rows = _fetch_registry_rows(conn)
        aggregate_rows = _fetch_aggregate_rows(conn, filters=filters)
        component_rows = _fetch_component_rows(conn, filters=filters)
        campaign_source_rows = _fetch_campaign_source_rows(conn, filters=filters)
        unassigned_summary = _fetch_unassigned_summary(conn, filters=filters)
        available_filter_rows = _fetch_available_gateway_filter_rows(conn) if include_available_filter_rows else []

    filtered_registry_rows = {
        campaign_id: row
        for campaign_id, row in registry_rows.items()
        if _gateway_matches_filters(row.get("gateway") or {}, filters)
    }

    if unassigned_active_count > 0:
        unassigned_summary = dict(unassigned_summary)
        unassigned_summary["rows"] = int(unassigned_summary.get("rows") or 0) + unassigned_active_count
        current_last_seen = (
            datetime.fromisoformat(str(unassigned_summary.get("last_seen_at")).replace("Z", "+00:00")).timestamp()
            if unassigned_summary.get("last_seen_at")
            else None
        )
        unassigned_summary["last_seen_at"] = _utc_iso(_max_ts(_to_float(current_last_seen), unassigned_active_last_seen))
        reasons = list(unassigned_summary.get("reasons") or [])
        reasons.append({"reason": "active_missing_campaign_id", "count": unassigned_active_count})
        unassigned_summary["reasons"] = reasons

    return _CampaignCatalog(
        all_active_rows=all_active_rows,
        registry_rows=registry_rows,
        active_by_campaign=active_by_campaign,
        filtered_registry_rows=filtered_registry_rows,
        aggregate_rows=aggregate_rows,
        component_rows=component_rows,
        campaign_source_rows=campaign_source_rows,
        unassigned_summary=dict(unassigned_summary),
        available_filter_rows=available_filter_rows,
    )


def _campaign_candidate_ids(catalog: _CampaignCatalog) -> List[str]:
    return sorted(set(catalog.filtered_registry_rows.keys()) | set(catalog.aggregate_rows.keys()) | set(catalog.active_by_campaign.keys()))


def _build_campaign_payload(catalog: _CampaignCatalog, *, campaign_id: str) -> Optional[Dict[str, Any]]:
    normalized_campaign_id = _normalize_campaign_id(campaign_id)
    if not normalized_campaign_id:
        return None

    registry_row = catalog.filtered_registry_rows.get(normalized_campaign_id)
    aggregate_row = catalog.aggregate_rows.get(normalized_campaign_id)
    active_for_campaign = list(catalog.active_by_campaign.get(normalized_campaign_id) or [])
    if registry_row is None and aggregate_row is None and not active_for_campaign:
        return None

    submitted_count = sum(
        1
        for row in active_for_campaign
        if str(row.get("state") or "") == "SUBMITTED"
        or (
            str(row.get("state") or "") == "RUNNING"
            and not bool(row.get("is_dispatched", True))
        )
    )
    running_count = sum(
        1
        for row in active_for_campaign
        if str(row.get("state") or "") == "RUNNING"
        and bool(row.get("is_dispatched", True))
    )

    component_map: Dict[str, Dict[str, Any]] = {}
    for (cmp_id, component), payload in catalog.component_rows.items():
        if cmp_id != normalized_campaign_id:
            continue
        component_map[component] = dict(payload)
    for row in active_for_campaign:
        component = str(row.get("component") or "").strip().lower()
        if not component:
            continue
        bucket = component_map.setdefault(
            component,
            {
                "component": component,
                "observed": 0,
                "pending": 0,
                "active": 0,
                "succeeded": 0,
                "failed": 0,
                "last_seen_at": None,
            },
        )
        if str(row.get("state") or "") == "RUNNING" and bool(
            row.get("is_dispatched", True)
        ):
            bucket["active"] = _to_int(bucket.get("active")) + 1
        else:
            bucket["pending"] = _to_int(bucket.get("pending")) + 1
        bucket["last_seen_at"] = _utc_iso(
            _max_ts(
                _to_float(row.get("updated_at")),
                (
                    datetime.fromisoformat(str(bucket.get("last_seen_at")).replace("Z", "+00:00")).timestamp()
                    if bucket.get("last_seen_at")
                    else None
                ),
            )
        )

    source_rows = list(catalog.campaign_source_rows.get(normalized_campaign_id) or [])
    source_rows.extend(
        {
            "gateway": row.get("gateway") or {},
            "count": 1,
            "last_seen_at": _to_float(row.get("updated_at")),
        }
        for row in active_for_campaign
    )
    source_summary = _summarize_sources(source_rows)
    components = sorted(
        list(component_map.values()),
        key=lambda item: (-(_to_int(item.get("observed")) + _to_int(item.get("active"))), str(item.get("component") or "")),
    )
    return _build_campaign_row(
        campaign_id=normalized_campaign_id,
        registry_row=registry_row,
        aggregate_row=aggregate_row,
        components=components,
        submitted_count=submitted_count,
        running_count=running_count,
        source_summary=source_summary,
    )


def list_campaigns(
    *,
    run_index: RunIndexStore,
    active_tasks: Sequence[Mapping[str, Any]],
    limit: int = 200,
    offset: int = 0,
    include_unassigned: bool = False,
    include_gateway_instance_ids: Sequence[str] = (),
    exclude_gateway_instance_ids: Sequence[str] = (),
    include_gateway_git_commits: Sequence[str] = (),
    exclude_gateway_git_commits: Sequence[str] = (),
) -> Dict[str, Any]:
    capped_limit = max(1, min(int(limit), 1000))
    capped_offset = max(0, int(offset))
    gateway_filters = _normalized_gateway_filters(
        include_gateway_instance_ids=include_gateway_instance_ids,
        exclude_gateway_instance_ids=exclude_gateway_instance_ids,
        include_gateway_git_commits=include_gateway_git_commits,
        exclude_gateway_git_commits=exclude_gateway_git_commits,
    )
    catalog = _load_campaign_catalog(
        run_index=run_index,
        active_tasks=active_tasks,
        filters=gateway_filters,
        include_available_filter_rows=True,
    )

    candidate_ids = _campaign_candidate_ids(catalog)
    campaigns: List[Dict[str, Any]] = []
    summary = {"waiting": 0, "active": 0, "completed": 0, "failed": 0, "unknown": 0}

    for campaign_id in candidate_ids:
        payload = _build_campaign_payload(catalog, campaign_id=campaign_id)
        if payload is None:
            continue
        normalized_status = _normalize_status(payload.get("status"))
        summary[normalized_status] = int(summary.get(normalized_status, 0)) + 1
        campaigns.append(payload)

    campaigns.sort(
        key=lambda item: (
            _status_rank(str(item.get("status") or "unknown")),
            -(
                datetime.fromisoformat(str(item["last_seen_at"]).replace("Z", "+00:00")).timestamp()
                if item.get("last_seen_at")
                else 0.0
            ),
            str(item.get("campaign_id") or ""),
        )
    )

    if include_unassigned and int(catalog.unassigned_summary.get("rows") or 0) > 0:
        campaigns.append(
            {
                "campaign_id": UNKNOWN_CAMPAIGN_ID,
                "run_name": "",
                "submitter": "",
                "status": "unknown",
                "started_at": None,
                "last_seen_at": catalog.unassigned_summary.get("last_seen_at"),
                "ended_at": None,
                "task_counts": {
                    "submitted": 0,
                    "running": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "observed": int(catalog.unassigned_summary.get("rows") or 0),
                    "total": int(catalog.unassigned_summary.get("rows") or 0),
                },
                "component_summary": [],
                "telemetry_readiness": {
                    "succeeded_rows": 0,
                    "runtime_present_rate": 0.0,
                    "peak_memory_present_rate": 0.0,
                    "mean_gpu_util_present_rate": 0.0,
                    "wallclock_present_rate": 0.0,
                },
                "gateway_summary": [],
            }
        )

    facet_rows = list(catalog.available_filter_rows)
    facet_rows.extend(
        {
            "gateway": row.get("gateway") or {},
            "count": 1,
            "first_seen_at": _to_float(row.get("created_at")),
            "last_seen_at": _max_ts(
                _to_float(row.get("last_seen_at")),
                _to_float(row.get("updated_at")),
                _to_float(row.get("created_at")),
            ),
        }
        for row in catalog.registry_rows.values()
    )
    facet_rows.extend(
        {
            "gateway": row.get("gateway") or {},
            "count": 1,
            "first_seen_at": _to_float(row.get("created_at")),
            "last_seen_at": _max_ts(_to_float(row.get("updated_at")), _to_float(row.get("created_at"))),
        }
        for row in catalog.all_active_rows
    )

    total = len(campaigns)
    return {
        "summary": summary,
        "count": total,
        "limit": capped_limit,
        "offset": capped_offset,
        "campaigns": campaigns[capped_offset : capped_offset + capped_limit],
        "unassigned": {
            "rows": int(catalog.unassigned_summary.get("rows") or 0),
            "last_seen_at": catalog.unassigned_summary.get("last_seen_at"),
            "reasons": list(catalog.unassigned_summary.get("reasons") or []),
        },
        "available_gateway_filters": _available_gateway_filters(facet_rows),
    }


def get_campaign_detail(
    *,
    run_index: RunIndexStore,
    active_tasks: Sequence[Mapping[str, Any]],
    campaign_id: str,
    include_gateway_instance_ids: Sequence[str] = (),
    exclude_gateway_instance_ids: Sequence[str] = (),
    include_gateway_git_commits: Sequence[str] = (),
    exclude_gateway_git_commits: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    normalized_campaign_id = _normalize_campaign_id(campaign_id)
    if not normalized_campaign_id:
        return None

    gateway_filters = _normalized_gateway_filters(
        include_gateway_instance_ids=include_gateway_instance_ids,
        exclude_gateway_instance_ids=exclude_gateway_instance_ids,
        include_gateway_git_commits=include_gateway_git_commits,
        exclude_gateway_git_commits=exclude_gateway_git_commits,
    )
    catalog = _load_campaign_catalog(
        run_index=run_index,
        active_tasks=active_tasks,
        filters=gateway_filters,
        include_available_filter_rows=False,
    )
    campaign = _build_campaign_payload(catalog, campaign_id=normalized_campaign_id)
    if campaign is None:
        return None
    campaign = dict(campaign)

    where_clauses = [
        "r.run_source = 'task'",
        "r.campaign_id = ?",
        "COALESCE(r.campaign_metadata_version, 0) >= 1",
    ]
    params: List[Any] = [normalized_campaign_id]
    _append_gateway_sql_filters(where_clauses, params, alias="r", filters=gateway_filters)
    with _connect(run_index) as conn:
        rows = conn.execute(
            f"""
            SELECT
                r.run_key,
                r.run_id,
                r.component,
                r.state,
                r.created_at,
                r.updated_at,
                r.finished_at,
                r.gateway_instance_id,
                r.gateway_bind_addr,
                r.gateway_git_commit,
                r.gateway_started_at
            FROM profile_runs r
            WHERE {" AND ".join(where_clauses)}
            ORDER BY COALESCE(r.updated_at, r.finished_at, r.created_at) DESC
            LIMIT 500
            """,
            params,
        ).fetchall()

    task_rows = [
        {
            "run_key": str(row["run_key"] or ""),
            "run_id": str(row["run_id"] or ""),
            "component": str(row["component"] or ""),
            "state": str(row["state"] or ""),
            "created_at": _utc_iso(_to_float(row["created_at"])),
            "updated_at": _utc_iso(_to_float(row["updated_at"])),
            "finished_at": _utc_iso(_to_float(row["finished_at"])),
            "source": "observed",
            "gateway": _gateway_from_row(row),
        }
        for row in rows
    ]

    for row in _active_task_rows(active_tasks):
        if str(row.get("campaign_id") or "") != normalized_campaign_id:
            continue
        if not _gateway_matches_filters(row.get("gateway") or {}, gateway_filters):
            continue
        task_rows.append(
            {
                "run_key": "",
                "run_id": str(row.get("run_id") or row.get("task_id") or ""),
                "component": str(row.get("component") or ""),
                "state": str(row.get("state") or ""),
                "created_at": _utc_iso(_to_float(row.get("created_at"))),
                "updated_at": _utc_iso(_to_float(row.get("updated_at"))),
                "finished_at": None,
                "source": "active",
                "gateway": _gateway_from_row(row),
            }
        )

    campaign["overview"] = {
        "campaign_id": str(campaign.get("campaign_id") or ""),
        "run_name": str(campaign.get("run_name") or ""),
        "submitter": str(campaign.get("submitter") or ""),
        "status": str(campaign.get("status") or "unknown"),
        "started_at": campaign.get("started_at"),
        "last_seen_at": campaign.get("last_seen_at"),
        "ended_at": campaign.get("ended_at"),
        "task_counts": dict(campaign.get("task_counts") or {}),
        "gateway_summary": list(campaign.get("gateway_summary") or []),
    }
    campaign["components"] = list(campaign.get("component_summary") or [])
    campaign["tasks"] = task_rows
    return campaign


__all__ = [
    "UNKNOWN_CAMPAIGN_ID",
    "derive_campaign_status",
    "get_campaign_detail",
    "list_campaigns",
]
