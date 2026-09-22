"""Campaign registry helpers for gateway profiling index store."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional

from ..gateway_identity import gateway_payload_from_mapping
from .store_runs import to_float

UNKNOWN_CAMPAIGN_ID = "__unknown__"
_VALID_STATUSES = {"waiting", "active", "completed", "failed", "unknown"}


def _normalize_campaign_id(value: Any) -> str:
    token = str(value or "").strip()
    if not token or token == UNKNOWN_CAMPAIGN_ID:
        return ""
    return token


def _normalize_status(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token not in _VALID_STATUSES:
        return "unknown"
    return token


def _coalesce_max(current: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if current is None:
        return candidate
    if candidate is None:
        return current
    return max(current, candidate)


def campaign_registry_row_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "campaign_id": str(row["campaign_id"] or "").strip(),
        "run_name": str(row["run_name"] or "").strip(),
        "submitter": str(row["submitter"] or "").strip(),
        "gateway": gateway_payload_from_mapping(
            {
                "instance_id": row["gateway_instance_id"],
                "bind_addr": row["gateway_bind_addr"],
                "git_commit": row["gateway_git_commit"],
                "started_at": row["gateway_started_at"],
            }
        ),
        "status": _normalize_status(row["status"]),
        "created_at": to_float(row["created_at"]),
        "started_at": to_float(row["started_at"]),
        "last_seen_at": to_float(row["last_seen_at"]),
        "ended_at": to_float(row["ended_at"]),
        "updated_at": to_float(row["updated_at"]),
    }


def upsert_campaign_registry_conn(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    run_name: str = "",
    submitter: str = "",
    gateway_instance_id: str = "",
    gateway_bind_addr: str = "",
    gateway_git_commit: str = "",
    gateway_started_at: Optional[float] = None,
    status: str = "waiting",
    created_at: Optional[float] = None,
    started_at: Optional[float] = None,
    last_seen_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    event_ts: Optional[float] = None,
) -> None:
    normalized_campaign_id = _normalize_campaign_id(campaign_id)
    if not normalized_campaign_id:
        return

    now_ts = to_float(event_ts) or time.time()
    normalized_status = _normalize_status(status)

    created = to_float(created_at)
    started = to_float(started_at)
    last_seen = to_float(last_seen_at)
    ended = to_float(ended_at)

    if created is None:
        created = now_ts
    if last_seen is None:
        last_seen = now_ts
    if normalized_status == "active" and started is None:
        started = now_ts
    if normalized_status in {"completed", "failed"} and ended is None:
        ended = now_ts

    existing = conn.execute(
        """
        SELECT campaign_id, run_name, submitter,
               gateway_instance_id, gateway_bind_addr, gateway_git_commit, gateway_started_at,
               status, created_at, started_at, last_seen_at, ended_at, updated_at
        FROM campaign_registry
        WHERE campaign_id = ?
        """,
        (normalized_campaign_id,),
    ).fetchone()

    if existing is not None:
        existing_row = campaign_registry_row_to_dict(existing)
        current_status = str(existing_row.get("status") or "unknown")
        merged_status = normalized_status
        if normalized_status == "waiting" and current_status in {"active", "completed", "failed"}:
            merged_status = current_status
        if normalized_status == "completed" and current_status == "failed":
            merged_status = current_status
        if normalized_status == "active" and current_status == "failed":
            merged_status = current_status

        run_name_value = str(existing_row.get("run_name") or "")
        submitter_value = str(existing_row.get("submitter") or "")
        if not run_name_value:
            run_name_value = str(run_name or "").strip()
        if not submitter_value:
            submitter_value = str(submitter or "").strip()
        existing_gateway = dict(existing_row.get("gateway") or {})
        gateway_instance_value = str(gateway_instance_id or "").strip() or str(existing_gateway.get("instance_id") or "")
        gateway_bind_value = str(gateway_bind_addr or "").strip() or str(existing_gateway.get("bind_addr") or "")
        gateway_commit_value = str(gateway_git_commit or "").strip().lower() or str(existing_gateway.get("git_commit") or "")
        gateway_started_value = to_float(gateway_started_at)
        if gateway_started_value is None:
            gateway_started_value = to_float(existing_gateway.get("started_at"))

        conn.execute(
            """
            UPDATE campaign_registry
            SET
                run_name = ?,
                submitter = ?,
                gateway_instance_id = ?,
                gateway_bind_addr = ?,
                gateway_git_commit = ?,
                gateway_started_at = COALESCE(gateway_started_at, ?),
                status = ?,
                created_at = COALESCE(created_at, ?),
                started_at = COALESCE(started_at, ?),
                last_seen_at = ?,
                ended_at = CASE
                    WHEN ? IS NOT NULL THEN ?
                    ELSE ended_at
                END,
                updated_at = ?
            WHERE campaign_id = ?
            """,
            (
                run_name_value,
                submitter_value,
                gateway_instance_value,
                gateway_bind_value,
                gateway_commit_value,
                gateway_started_value,
                merged_status,
                created,
                started,
                _coalesce_max(to_float(existing_row.get("last_seen_at")), last_seen),
                ended,
                ended,
                now_ts,
                normalized_campaign_id,
            ),
        )
        return

    conn.execute(
        """
        INSERT INTO campaign_registry (
            campaign_id,
            run_name,
            submitter,
            gateway_instance_id,
            gateway_bind_addr,
            gateway_git_commit,
            gateway_started_at,
            status,
            created_at,
            started_at,
            last_seen_at,
            ended_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            normalized_campaign_id,
            str(run_name or "").strip(),
            str(submitter or "").strip(),
            str(gateway_instance_id or "").strip(),
            str(gateway_bind_addr or "").strip(),
            str(gateway_git_commit or "").strip().lower(),
            to_float(gateway_started_at),
            normalized_status,
            created,
            started,
            last_seen,
            ended,
            now_ts,
        ),
    )


def mark_campaign_task_state_conn(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    run_name: str = "",
    submitter: str = "",
    gateway_instance_id: str = "",
    gateway_bind_addr: str = "",
    gateway_git_commit: str = "",
    gateway_started_at: Optional[float] = None,
    task_state: str,
    event_ts: Optional[float] = None,
    created_at: Optional[float] = None,
) -> None:
    normalized_state = str(task_state or "").strip().upper()
    now_ts = to_float(event_ts) or time.time()

    status = "unknown"
    started_at = None
    ended_at = None
    if normalized_state == "SUBMITTED":
        status = "waiting"
    elif normalized_state == "RUNNING":
        status = "active"
        started_at = now_ts
    elif normalized_state == "SUCCEEDED":
        status = "completed"
        ended_at = now_ts
    elif normalized_state in {"FAILED", "CANCELLED"}:
        status = "failed"
        ended_at = now_ts

    upsert_campaign_registry_conn(
        conn,
        campaign_id=campaign_id,
        run_name=run_name,
        submitter=submitter,
        gateway_instance_id=gateway_instance_id,
        gateway_bind_addr=gateway_bind_addr,
        gateway_git_commit=gateway_git_commit,
        gateway_started_at=gateway_started_at,
        status=status,
        created_at=created_at,
        started_at=started_at,
        last_seen_at=now_ts,
        ended_at=ended_at,
        event_ts=now_ts,
    )


def get_campaign_registry_conn(conn: sqlite3.Connection, campaign_id: str) -> Optional[Dict[str, Any]]:
    normalized_campaign_id = _normalize_campaign_id(campaign_id)
    if not normalized_campaign_id:
        return None
    row = conn.execute(
        """
        SELECT campaign_id, run_name, submitter,
               gateway_instance_id, gateway_bind_addr, gateway_git_commit, gateway_started_at,
               status,
               created_at, started_at, last_seen_at, ended_at, updated_at
        FROM campaign_registry
        WHERE campaign_id = ?
        """,
        (normalized_campaign_id,),
    ).fetchone()
    if row is None:
        return None
    return campaign_registry_row_to_dict(row)


def list_campaign_registry_conn(
    conn: sqlite3.Connection,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    params: list[Any] = []
    sql = """
        SELECT campaign_id, run_name, submitter,
               gateway_instance_id, gateway_bind_addr, gateway_git_commit, gateway_started_at,
               status,
               created_at, started_at, last_seen_at, ended_at, updated_at
        FROM campaign_registry
        ORDER BY COALESCE(last_seen_at, created_at, updated_at) DESC, campaign_id ASC
    """
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([int(max(1, limit)), int(max(0, offset))])

    rows = conn.execute(sql, params).fetchall()
    return [campaign_registry_row_to_dict(row) for row in rows]
