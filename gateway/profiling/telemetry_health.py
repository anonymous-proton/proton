"""Task telemetry ingestion health report builder."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from .index_store import RunIndexStore


def _utc_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        value = float(ts)
    except Exception:
        return None
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")


def _ratio(num: float | int, den: float | int) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if parsed != parsed:
        return 0.0
    return parsed


def _normalize_components(components: Sequence[str] | None) -> Tuple[str, ...]:
    if components is None:
        return ()
    out: List[str] = []
    for item in list(components or []):
        token = str(item or "").strip().lower()
        if token:
            out.append(token)
    return tuple(dict.fromkeys(out))


def _component_clause(*, alias: str, components: Sequence[str]) -> tuple[str, list[Any]]:
    if not components:
        return "", []
    placeholders = ",".join("?" for _ in components)
    return f" AND {alias}.component IN ({placeholders})", list(components)


def _campaign_clause(*, alias: str, campaign_id: str | None) -> tuple[str, list[Any]]:
    token = str(campaign_id or "").strip()
    if not token:
        return "", []
    if token == "__unknown__":
        return f" AND TRIM(COALESCE({alias}.campaign_id, '')) = ''", []
    return f" AND {alias}.campaign_id = ?", [token]


def _connect(db_path: str) -> sqlite3.Connection:
    cur_name = threading.current_thread().name
    if (
        not cur_name.startswith("sqlite-")
        and not os.environ.get("GATEWAY_SQLITE_THREAD_BYPASS")
    ):
        stack_summary = "".join(traceback.format_stack()[-5:-1])
        raise RuntimeError(
            f"telemetry_health._connect from non-sqlite thread {cur_name!r}.  "
            f"All SQLite access must route through "
            f"GatewayHTTPService._run_index_call (sqlite-* dedicated "
            f"executor, max_workers=1).  This guard prevents -class "
            f"fcntl deadlock.  Set env GATEWAY_SQLITE_THREAD_BYPASS=1 "
            f"to bypass for unit tests / scripts.\n"
            f"Caller stack:\n{stack_summary}"
        )
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _task_cte(component_clause: str, campaign_clause: str) -> str:
    return f"""
    WITH task_rows AS (
        SELECT
            r.run_key,
            r.component,
            r.state,
            r.qc_keep,
            r.config_fingerprint,
            r.input_fingerprint,
            r.runtime_sec,
            r.peak_memory_mib,
            r.mean_gpu_util_percent,
            r.telemetry_wall_clock_sec,
            COALESCE(r.finished_at, r.updated_at, r.created_at) AS event_ts
        FROM profile_runs r
        WHERE r.run_source = 'task'
          {component_clause.lstrip()}{campaign_clause}
    ),
    task_24h AS (
        SELECT * FROM task_rows WHERE event_ts >= ?
    )
    """


def build_task_telemetry_health_report(
    *,
    run_index: RunIndexStore,
    components: Sequence[str] | None = None,
    campaign_id: str | None = None,
    window_hours: int = 24,
    include_all_time: bool = True,
    now_ts: float | None = None,
) -> Dict[str, Any]:
    normalized_components = _normalize_components(components)
    normalized_window_hours = max(1, int(window_hours))
    now = float(now_ts if now_ts is not None else time.time())
    window_start = now - float(normalized_window_hours * 3600)
    component_sql, component_params = _component_clause(alias="r", components=normalized_components)
    campaign_sql, campaign_params = _campaign_clause(alias="r", campaign_id=campaign_id)
    cte = _task_cte(component_sql, campaign_sql)
    base_params = [*component_params, *campaign_params, window_start]

    with _connect(str(run_index.db_path)) as conn:
        ingestion_row = conn.execute(
            cte
            + """
            SELECT
                (SELECT COUNT(*) FROM task_24h) AS rows_24h,
                (SELECT COALESCE(SUM(CASE WHEN state = 'SUCCEEDED' THEN 1 ELSE 0 END), 0) FROM task_24h) AS rows_succeeded_24h,
                (SELECT COALESCE(SUM(CASE WHEN state = 'SUCCEEDED' AND qc_keep = 1 THEN 1 ELSE 0 END), 0) FROM task_24h) AS rows_qc_keep_24h,
                (SELECT COALESCE(SUM(
                    CASE
                        WHEN state = 'SUCCEEDED'
                         AND qc_keep = 1
                         AND TRIM(COALESCE(config_fingerprint, '')) != ''
                         AND TRIM(COALESCE(input_fingerprint, '')) != ''
                        THEN 1 ELSE 0
                    END
                ), 0) FROM task_24h) AS rows_model_input_24h,
                (SELECT MAX(event_ts) FROM task_rows) AS last_task_at,
                (SELECT COUNT(*) FROM task_rows) AS rows_all_time
            """,
            base_params,
        ).fetchone()

        presence_row = conn.execute(
            cte
            + """
            SELECT
                COUNT(*) AS denom,
                COALESCE(SUM(CASE WHEN t.runtime_sec IS NOT NULL THEN 1 ELSE 0 END), 0) AS runtime_sec_non_null,
                COALESCE(SUM(CASE WHEN t.peak_memory_mib IS NOT NULL THEN 1 ELSE 0 END), 0) AS peak_memory_mib_non_null,
                COALESCE(SUM(CASE WHEN t.mean_gpu_util_percent IS NOT NULL THEN 1 ELSE 0 END), 0) AS mean_gpu_util_percent_non_null,
                COALESCE(SUM(CASE WHEN t.telemetry_wall_clock_sec IS NOT NULL THEN 1 ELSE 0 END), 0) AS telemetry_wall_clock_sec_non_null,
                COALESCE(SUM(CASE WHEN TRIM(COALESCE(t.config_fingerprint, '')) != '' THEN 1 ELSE 0 END), 0) AS config_fingerprint_non_null,
                COALESCE(SUM(CASE WHEN TRIM(COALESCE(t.input_fingerprint, '')) != '' THEN 1 ELSE 0 END), 0) AS input_fingerprint_non_null
            FROM task_24h t
            WHERE t.state = 'SUCCEEDED'
            """,
            base_params,
        ).fetchone()

        distribution_row = conn.execute(
            cte
            + """
            SELECT
                COUNT(DISTINCT CASE
                    WHEN t.state = 'SUCCEEDED'
                     AND t.qc_keep = 1
                     AND TRIM(COALESCE(t.config_fingerprint, '')) != ''
                    THEN t.config_fingerprint
                    ELSE NULL
                END) AS unique_configs_24h,
                COUNT(DISTINCT CASE
                    WHEN t.state = 'SUCCEEDED'
                     AND t.qc_keep = 1
                     AND TRIM(COALESCE(t.input_fingerprint, '')) != ''
                    THEN t.input_fingerprint
                    ELSE NULL
                END) AS unique_inputs_24h,
                COALESCE(SUM(CASE WHEN t.state = 'SUCCEEDED' THEN 1 ELSE 0 END), 0) AS rows_succeeded_24h
            FROM task_24h t
            """,
            base_params,
        ).fetchone()

        component_rows = conn.execute(
            cte
            + """
            SELECT
                t.component AS component,
                COUNT(*) AS rows_24h,
                COALESCE(SUM(
                    CASE
                        WHEN t.state = 'SUCCEEDED'
                         AND t.qc_keep = 1
                         AND TRIM(COALESCE(t.config_fingerprint, '')) != ''
                         AND TRIM(COALESCE(t.input_fingerprint, '')) != ''
                        THEN 1 ELSE 0
                    END
                ), 0) AS rows_model_input_24h,
                MAX(t.event_ts) AS last_seen_at,
                COALESCE(SUM(CASE WHEN t.state = 'SUCCEEDED' THEN 1 ELSE 0 END), 0) AS rows_succeeded_24h,
                COALESCE(SUM(CASE WHEN t.state = 'SUCCEEDED' AND t.peak_memory_mib IS NOT NULL THEN 1 ELSE 0 END), 0) AS peak_memory_mib_non_null,
                COALESCE(SUM(CASE WHEN t.state = 'SUCCEEDED' AND t.mean_gpu_util_percent IS NOT NULL THEN 1 ELSE 0 END), 0) AS mean_gpu_util_percent_non_null,
                COALESCE(SUM(CASE WHEN t.state = 'SUCCEEDED' AND t.telemetry_wall_clock_sec IS NOT NULL THEN 1 ELSE 0 END), 0) AS telemetry_wall_clock_sec_non_null,
                COUNT(DISTINCT CASE
                    WHEN t.state = 'SUCCEEDED'
                     AND t.qc_keep = 1
                     AND TRIM(COALESCE(t.config_fingerprint, '')) != ''
                    THEN t.config_fingerprint
                    ELSE NULL
                END) AS unique_configs_24h,
                COUNT(DISTINCT CASE
                    WHEN t.state = 'SUCCEEDED'
                     AND t.qc_keep = 1
                     AND TRIM(COALESCE(t.input_fingerprint, '')) != ''
                    THEN t.input_fingerprint
                    ELSE NULL
                END) AS unique_inputs_24h
            FROM task_24h t
            GROUP BY t.component
            ORDER BY t.component ASC
            """,
            base_params,
        ).fetchall()
    field_denom = _as_int(presence_row["denom"]) if presence_row else 0
    rows_all_time_value = _as_int(ingestion_row["rows_all_time"]) if ingestion_row else 0

    components_payload: List[Dict[str, Any]] = []
    for row in component_rows:
        component_succeeded = _as_int(row["rows_succeeded_24h"])
        components_payload.append(
            {
                "component": str(row["component"] or ""),
                "rows_24h": _as_int(row["rows_24h"]),
                "rows_model_input_24h": _as_int(row["rows_model_input_24h"]),
                "last_seen_at": _utc_iso(_as_float(row["last_seen_at"]) if row["last_seen_at"] is not None else None),
                "field_presence": {
                    "peak_memory_mib": _ratio(_as_int(row["peak_memory_mib_non_null"]), component_succeeded),
                    "mean_gpu_util_percent": _ratio(_as_int(row["mean_gpu_util_percent_non_null"]), component_succeeded),
                    "telemetry_wall_clock_sec": _ratio(_as_int(row["telemetry_wall_clock_sec_non_null"]), component_succeeded),
                },
                "unique_configs_24h": _as_int(row["unique_configs_24h"]),
                "unique_inputs_24h": _as_int(row["unique_inputs_24h"]),
            }
        )

    task_ingestion = {
        "rows_24h": _as_int(ingestion_row["rows_24h"]) if ingestion_row else 0,
        "rows_succeeded_24h": _as_int(ingestion_row["rows_succeeded_24h"]) if ingestion_row else 0,
        "rows_qc_keep_24h": _as_int(ingestion_row["rows_qc_keep_24h"]) if ingestion_row else 0,
        "rows_model_input_24h": _as_int(ingestion_row["rows_model_input_24h"]) if ingestion_row else 0,
        "rows_all_time": rows_all_time_value if include_all_time else 0,
        "last_task_at": _utc_iso(_as_float(ingestion_row["last_task_at"]) if ingestion_row and ingestion_row["last_task_at"] is not None else None),
    }

    field_presence = {
        "runtime_sec": _ratio(_as_int(presence_row["runtime_sec_non_null"]) if presence_row else 0, field_denom),
        "peak_memory_mib": _ratio(_as_int(presence_row["peak_memory_mib_non_null"]) if presence_row else 0, field_denom),
        "mean_gpu_util_percent": _ratio(_as_int(presence_row["mean_gpu_util_percent_non_null"]) if presence_row else 0, field_denom),
        "telemetry_wall_clock_sec": _ratio(_as_int(presence_row["telemetry_wall_clock_sec_non_null"]) if presence_row else 0, field_denom),
        "config_fingerprint": _ratio(_as_int(presence_row["config_fingerprint_non_null"]) if presence_row else 0, field_denom),
        "input_fingerprint": _ratio(_as_int(presence_row["input_fingerprint_non_null"]) if presence_row else 0, field_denom),
    }

    distribution = {
        "unique_configs_24h": _as_int(distribution_row["unique_configs_24h"]) if distribution_row else 0,
        "unique_inputs_24h": _as_int(distribution_row["unique_inputs_24h"]) if distribution_row else 0,
    }

    return {
        "meta": {
            "generated_at": _utc_iso(now),
            "window_hours": normalized_window_hours,
            "components": list(normalized_components),
            "campaign_id": str(campaign_id or "").strip() or None,
            "include_all_time": bool(include_all_time),
        },
        "task_ingestion": task_ingestion,
        "field_presence": field_presence,
        "distribution": distribution,
        "components": components_payload,
    }


__all__ = ["build_task_telemetry_health_report"]
