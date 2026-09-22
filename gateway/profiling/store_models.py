"""Model/cell-stats row and CRUD helpers for profiling index store."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .store_runs import json_dumps, json_loads, to_float, to_int


def cell_stats_row_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "cell_key": str(row["cell_key"]),
        "component": str(row["component"]),
        "level": str(row["level"] or ""),
        "n_valid": int(row["n_valid"] or 0),
        "mean_log_exec": to_float(row["mean_log_exec"]),
        "std_log_exec": to_float(row["std_log_exec"]),
        "ci95_halfwidth": to_float(row["ci95_halfwidth"]),
        "rel_halfwidth": to_float(row["rel_halfwidth"]),
        "bimodal_flag": bool(int(row["bimodal_flag"] or 0)),
        "contam_rate": to_float(row["contam_rate"]),
        "last_updated": to_float(row["last_updated"]),
        "stats": json_loads(row["stats_json"], {}),
    }


def model_version_row_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "model_version": str(row["model_version"]),
        "trained_at": to_float(row["trained_at"]),
        "data_cut_ts": to_float(row["data_cut_ts"]),
        "metrics": json_loads(row["metrics_json"], {}),
        "decision_gate": json_loads(row["decision_gate_json"], {}),
        "cell_schema_ids": json_loads(row["cell_schema_ids_json"], []),
        "is_active": bool(int(row["is_active"] or 0)),
    }


def component_model_version_row_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "component_model_version_id": str(row["component_model_version_id"]),
        "component": str(row["component"] or ""),
        "parent_loop_model_version_id": str(row["parent_loop_model_version_id"] or ""),
        "job_id": str(row["job_id"] or ""),
        "loop_idx": int(row["loop_idx"] or 0),
        "trained_at": to_float(row["trained_at"]),
        "status": str(row["status"] or ""),
        "is_active": bool(int(row["is_active"] or 0)),
        "error": str(row["error"] or ""),
        "cell_schema_ids": json_loads(row["cell_schema_ids_json"], []),
        "metrics": json_loads(row["metrics_json"], {}),
        "decision_gate": json_loads(row["decision_gate_json"], {}),
        "readiness": json_loads(row["readiness_json"], {}),
        "priority": json_loads(row["priority_json"], {}),
        "delta": json_loads(row["delta_json"], {}),
        "segments": json_loads(row["segments_json"], []),
        "artifacts": json_loads(row["artifacts_json"], {}),
    }


def upsert_cell_stats_conn(
    conn: sqlite3.Connection,
    *,
    cell_key: str,
    component: str,
    level: str,
    n_valid: int,
    mean_log_exec: Optional[float],
    std_log_exec: Optional[float],
    ci95_halfwidth: Optional[float],
    rel_halfwidth: Optional[float],
    bimodal_flag: bool,
    contam_rate: Optional[float],
    stats: Optional[Mapping[str, Any]] = None,
) -> None:
    normalized_cell_key = str(cell_key).strip()
    if not normalized_cell_key:
        return
    conn.execute(
        """
        INSERT INTO cell_stats (
            cell_key, component, level, n_valid, mean_log_exec, std_log_exec,
            ci95_halfwidth, rel_halfwidth, bimodal_flag, contam_rate, last_updated, stats_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cell_key) DO UPDATE SET
            component = excluded.component,
            level = excluded.level,
            n_valid = excluded.n_valid,
            mean_log_exec = excluded.mean_log_exec,
            std_log_exec = excluded.std_log_exec,
            ci95_halfwidth = excluded.ci95_halfwidth,
            rel_halfwidth = excluded.rel_halfwidth,
            bimodal_flag = excluded.bimodal_flag,
            contam_rate = excluded.contam_rate,
            last_updated = excluded.last_updated,
            stats_json = excluded.stats_json
        """,
        (
            normalized_cell_key,
            str(component or "").strip().lower(),
            str(level or "").strip().lower(),
            int(max(0, n_valid)),
            to_float(mean_log_exec),
            to_float(std_log_exec),
            to_float(ci95_halfwidth),
            to_float(rel_halfwidth),
            1 if bool(bimodal_flag) else 0,
            to_float(contam_rate),
            time.time(),
            json_dumps(dict(stats or {})),
        ),
    )


def list_cell_stats_conn(
    conn: sqlite3.Connection,
    *,
    component: Optional[str] = None,
    level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    where: List[str] = []
    params: List[Any] = []
    if component:
        where.append("component = ?")
        params.append(str(component).strip().lower())
    if level:
        where.append("level = ?")
        params.append(str(level).strip().lower())
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT cell_key, component, level, n_valid, mean_log_exec, std_log_exec,
               ci95_halfwidth, rel_halfwidth, bimodal_flag, contam_rate, last_updated, stats_json
        FROM cell_stats
        {where_sql}
        ORDER BY last_updated DESC
        """,
        params,
    ).fetchall()
    return [cell_stats_row_to_dict(row) for row in rows]


def summarize_component_segments(segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(item) for item in list(segments or []) if isinstance(item, Mapping)]
    if not rows:
        return {
            "status": "CAUTION",
            "ready_ratio": None,
            "blocked_ratio": None,
            "contam_cell_ratio": None,
            "estimated_remaining_runs": 0,
            "eta_gpu_sec_to_ready": 0.0,
            "priority_rank": None,
            "priority_need": None,
            "priority_value": None,
            "priority_cost": None,
            "priority_risk": None,
            "priority_reasons": [],
            "delta_ready_ratio": None,
            "delta_eta_gpu_sec_to_ready": None,
            "delta_runtime_ape_p90": None,
            "delta_blocked_ratio": None,
            "delta_contam_cell_ratio": None,
            "delta_state": "N/A",
            "delta_reason": "",
        }

    def _avg(key: str) -> Optional[float]:
        values = [to_float(row.get(key)) for row in rows]
        keep = [float(item) for item in values if item is not None]
        if not keep:
            return None
        return float(sum(keep) / max(1, len(keep)))

    ready_ratio = _avg("ready_ratio")
    blocked_ratio = _avg("blocked_ratio")
    contam_cell_ratio = _avg("contam_cell_ratio")
    est_remaining = int(sum(int(to_int(row.get("estimated_remaining_runs")) or 0) for row in rows))
    eta = float(sum(float(to_float(row.get("eta_gpu_sec_to_ready")) or 0.0) for row in rows))

    priority_rows = [row for row in rows if to_int(row.get("priority_rank")) is not None]
    priority_rows.sort(key=lambda item: int(to_int(item.get("priority_rank")) or 0))
    first_priority = priority_rows[0] if priority_rows else {}

    status_values = {
        str(row.get("status") or "").strip().upper()
        for row in rows
    }
    if "BLOCKED" in status_values:
        status = "BLOCKED"
    elif status_values and status_values.issubset({"READY"}):
        status = "READY"
    else:
        status = "CAUTION"

    delta_states = [str(row.get("delta_state") or "").strip().upper() for row in rows]
    delta_state = "N/A"
    if "SCHEMA_DRIFT" in delta_states:
        delta_state = "SCHEMA_DRIFT"
    elif "REGRESSED" in delta_states:
        delta_state = "REGRESSED"
    elif "IMPROVED" in delta_states:
        delta_state = "IMPROVED"
    elif "MIXED" in delta_states:
        delta_state = "MIXED"

    delta_reason_values = sorted(
        {
            str(row.get("delta_reason") or "").strip()
            for row in rows
            if str(row.get("delta_reason") or "").strip()
        }
    )

    return {
        "status": status,
        "ready_ratio": ready_ratio,
        "blocked_ratio": blocked_ratio,
        "contam_cell_ratio": contam_cell_ratio,
        "estimated_remaining_runs": est_remaining,
        "eta_gpu_sec_to_ready": eta,
        "priority_rank": to_int(first_priority.get("priority_rank")),
        "priority_need": to_float(first_priority.get("priority_need")),
        "priority_value": to_float(first_priority.get("priority_value")),
        "priority_cost": to_float(first_priority.get("priority_cost")),
        "priority_risk": to_float(first_priority.get("priority_risk")),
        "priority_reasons": list(first_priority.get("priority_reasons") or []),
        "delta_ready_ratio": _avg("delta_ready_ratio"),
        "delta_eta_gpu_sec_to_ready": _avg("delta_eta_gpu_sec_to_ready"),
        "delta_runtime_ape_p90": _avg("delta_runtime_ape_p90"),
        "delta_blocked_ratio": _avg("delta_blocked_ratio"),
        "delta_contam_cell_ratio": _avg("delta_contam_cell_ratio"),
        "delta_state": delta_state,
        "delta_reason": "; ".join(delta_reason_values),
    }


def upsert_model_version_conn(
    conn: sqlite3.Connection,
    *,
    model_version: str,
    trained_at: Optional[float] = None,
    data_cut_ts: Optional[float] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    decision_gate: Optional[Mapping[str, Any]] = None,
    cell_schema_ids: Optional[Sequence[str]] = None,
    is_active: bool = False,
) -> None:
    key = str(model_version or "").strip()
    if not key:
        return
    trained = float(trained_at or time.time())
    conn.execute(
        """
        INSERT INTO model_versions (
            model_version, trained_at, data_cut_ts, metrics_json, decision_gate_json, cell_schema_ids_json, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_version) DO UPDATE SET
            trained_at = excluded.trained_at,
            data_cut_ts = excluded.data_cut_ts,
            metrics_json = excluded.metrics_json,
            decision_gate_json = excluded.decision_gate_json,
            cell_schema_ids_json = excluded.cell_schema_ids_json,
            is_active = excluded.is_active
        """,
        (
            key,
            trained,
            to_float(data_cut_ts),
            json_dumps(dict(metrics or {})),
            json_dumps(dict(decision_gate or {})),
            json_dumps(
                sorted(
                    {
                        str(item).strip()
                        for item in list(cell_schema_ids or [])
                        if str(item).strip()
                    }
                )
            ),
            1 if bool(is_active) else 0,
        ),
    )


def list_model_versions_conn(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT model_version, trained_at, data_cut_ts, metrics_json, decision_gate_json, cell_schema_ids_json, is_active
        FROM model_versions
        ORDER BY trained_at DESC
        """
    ).fetchall()
    return [model_version_row_to_dict(row) for row in rows]


def upsert_component_model_version_conn(
    conn: sqlite3.Connection,
    *,
    component_model_version_id: str,
    component: str,
    parent_loop_model_version_id: str,
    job_id: str,
    loop_idx: int,
    cell_schema_ids: Optional[Sequence[str]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    decision_gate: Optional[Mapping[str, Any]] = None,
    readiness: Optional[Mapping[str, Any]] = None,
    priority: Optional[Mapping[str, Any]] = None,
    delta: Optional[Mapping[str, Any]] = None,
    segments: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    status: str = "",
    is_active: bool = False,
    trained_at: Optional[float] = None,
    error: str = "",
) -> None:
    key = str(component_model_version_id or "").strip()
    if not key:
        return
    conn.execute(
        """
        INSERT INTO component_model_versions (
            component_model_version_id,
            component,
            parent_loop_model_version_id,
            job_id,
            loop_idx,
            cell_schema_ids_json,
            metrics_json,
            decision_gate_json,
            readiness_json,
            priority_json,
            delta_json,
            segments_json,
            artifacts_json,
            status,
            is_active,
            trained_at,
            error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(component_model_version_id) DO UPDATE SET
            component = excluded.component,
            parent_loop_model_version_id = excluded.parent_loop_model_version_id,
            job_id = excluded.job_id,
            loop_idx = excluded.loop_idx,
            cell_schema_ids_json = excluded.cell_schema_ids_json,
            metrics_json = excluded.metrics_json,
            decision_gate_json = excluded.decision_gate_json,
            readiness_json = excluded.readiness_json,
            priority_json = excluded.priority_json,
            delta_json = excluded.delta_json,
            segments_json = excluded.segments_json,
            artifacts_json = excluded.artifacts_json,
            status = excluded.status,
            is_active = excluded.is_active,
            trained_at = excluded.trained_at,
            error = excluded.error
        """,
        (
            key,
            str(component or "").strip().lower(),
            str(parent_loop_model_version_id or "").strip(),
            str(job_id or "").strip(),
            int(loop_idx or 0),
            json_dumps(
                sorted(
                    {
                        str(item).strip()
                        for item in list(cell_schema_ids or [])
                        if str(item).strip()
                    }
                )
            ),
            json_dumps(dict(metrics or {})),
            json_dumps(dict(decision_gate or {})),
            json_dumps(dict(readiness or {})),
            json_dumps(dict(priority or {})),
            json_dumps(dict(delta or {})),
            json_dumps([dict(item) for item in list(segments or []) if isinstance(item, Mapping)]),
            json_dumps(dict(artifacts or {})),
            str(status or "").strip().upper(),
            1 if bool(is_active) else 0,
            float(trained_at or time.time()),
            str(error or ""),
        ),
    )


def get_component_model_version_conn(conn: sqlite3.Connection, component_model_version_id: str) -> Optional[Dict[str, Any]]:
    key = str(component_model_version_id or "").strip()
    if not key:
        return None
    row = conn.execute(
        """
        SELECT component_model_version_id, component, parent_loop_model_version_id,
               job_id, loop_idx, cell_schema_ids_json, metrics_json, decision_gate_json,
               readiness_json, priority_json, delta_json, segments_json, artifacts_json,
               status, is_active, trained_at, error
        FROM component_model_versions
        WHERE component_model_version_id = ?
        """,
        (key,),
    ).fetchone()
    if row is None:
        return None
    return component_model_version_row_to_dict(row)


def list_component_model_versions_conn(
    conn: sqlite3.Connection,
    *,
    component: Optional[str] = None,
    job_id: Optional[str] = None,
    loop_idx: Optional[int] = None,
    is_active: Optional[bool] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    where: List[str] = []
    params: List[Any] = []
    if component:
        where.append("component = ?")
        params.append(str(component).strip().lower())
    if job_id:
        where.append("job_id = ?")
        params.append(str(job_id).strip())
    if loop_idx is not None:
        where.append("loop_idx = ?")
        params.append(int(loop_idx))
    if is_active is not None:
        where.append("is_active = ?")
        params.append(1 if bool(is_active) else 0)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    bounded_limit = max(1, min(int(limit), 5000))
    rows = conn.execute(
        f"""
        SELECT component_model_version_id, component, parent_loop_model_version_id,
               job_id, loop_idx, cell_schema_ids_json, metrics_json, decision_gate_json,
               readiness_json, priority_json, delta_json, segments_json, artifacts_json,
               status, is_active, trained_at, error
        FROM component_model_versions
        {where_sql}
        ORDER BY trained_at DESC
        LIMIT ?
        """,
        [*params, bounded_limit],
    ).fetchall()
    return [component_model_version_row_to_dict(row) for row in rows]


def list_latest_component_models_conn(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cmv.component_model_version_id, cmv.component, cmv.parent_loop_model_version_id,
               cmv.job_id, cmv.loop_idx, cmv.cell_schema_ids_json, cmv.metrics_json,
               cmv.decision_gate_json, cmv.readiness_json, cmv.priority_json, cmv.delta_json,
               cmv.segments_json, cmv.artifacts_json, cmv.status, cmv.is_active,
               cmv.trained_at, cmv.error
        FROM component_model_versions cmv
        INNER JOIN (
            SELECT component, MAX(trained_at) AS max_trained_at
            FROM component_model_versions
            GROUP BY component
        ) latest
        ON latest.component = cmv.component
        AND latest.max_trained_at = cmv.trained_at
        ORDER BY cmv.component ASC
        """
    ).fetchall()
    return [component_model_version_row_to_dict(row) for row in rows]


def list_active_component_models_conn(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT component_model_version_id, component, parent_loop_model_version_id,
               job_id, loop_idx, cell_schema_ids_json, metrics_json,
               decision_gate_json, readiness_json, priority_json, delta_json,
               segments_json, artifacts_json, status, is_active,
               trained_at, error
        FROM component_model_versions
        WHERE is_active = 1
        ORDER BY component ASC, trained_at DESC
        """
    ).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        parsed = component_model_version_row_to_dict(row)
        component = str(parsed.get("component") or "").strip().lower()
        if not component:
            continue
        out.setdefault(component, parsed)
    return [out[key] for key in sorted(out.keys())]


def get_active_component_model_conn(conn: sqlite3.Connection, *, component: str) -> Optional[Dict[str, Any]]:
    normalized_component = str(component or "").strip().lower()
    if not normalized_component:
        return None
    row = conn.execute(
        """
        SELECT component_model_version_id, component, parent_loop_model_version_id,
               job_id, loop_idx, cell_schema_ids_json, metrics_json,
               decision_gate_json, readiness_json, priority_json, delta_json,
               segments_json, artifacts_json, status, is_active,
               trained_at, error
        FROM component_model_versions
        WHERE component = ? AND is_active = 1
        ORDER BY trained_at DESC
        LIMIT 1
        """,
        (normalized_component,),
    ).fetchone()
    if row is None:
        return None
    return component_model_version_row_to_dict(row)


def set_component_model_active_conn(conn: sqlite3.Connection, *, component: str, component_model_version_id: str) -> None:
    normalized_component = str(component or "").strip().lower()
    normalized_id = str(component_model_version_id or "").strip()
    if not normalized_component or not normalized_id:
        return
    conn.execute(
        "UPDATE component_model_versions SET is_active = 0 WHERE component = ?",
        (normalized_component,),
    )
    conn.execute(
        """
        UPDATE component_model_versions
        SET is_active = 1
        WHERE component = ? AND component_model_version_id = ?
        """,
        (normalized_component, normalized_id),
    )

