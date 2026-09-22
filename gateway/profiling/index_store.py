"""SQLite-backed profiling run index for querying and deduplication."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def boot_phase_bypass():
    """ fix — context manager for legitimate boot-time
    MainThread SQLite access (e.g. ``bootstrap_from_history``).

    Sets ``GATEWAY_SQLITE_THREAD_BYPASS`` for the duration of the
    block, so the ``_connect`` thread guard does not raise.
    Production runtime paths (post-boot) are NOT supposed to use this
    — they must route through the dedicated ``sqlite-*`` executor.
    """
    saved = os.environ.get("GATEWAY_SQLITE_THREAD_BYPASS")
    os.environ["GATEWAY_SQLITE_THREAD_BYPASS"] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("GATEWAY_SQLITE_THREAD_BYPASS", None)
        else:
            os.environ["GATEWAY_SQLITE_THREAD_BYPASS"] = saved
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract_models import load_contracts
from .campaign_registry import (
    get_campaign_registry_conn,
    list_campaign_registry_conn,
    mark_campaign_task_state_conn,
    upsert_campaign_registry_conn,
)
from .index_models import AxisFilter, RunQuery, RunRecord
from .run_identity import build_cell_key, build_cell_schema_id, canonicalize_axes
from .store_models import (
    get_active_component_model_conn,
    get_component_model_version_conn,
    list_cell_stats_conn,
    list_active_component_models_conn,
    list_component_model_versions_conn,
    list_latest_component_models_conn,
    list_model_versions_conn,
    set_component_model_active_conn,
    summarize_component_segments,
    upsert_cell_stats_conn,
    upsert_component_model_version_conn,
    upsert_model_version_conn,
)
from .store_runs import (
    CELL_SCHEMA_ALGO_VERSION as _CELL_SCHEMA_ALGO_VERSION,
    DEFAULT_CELL_SCHEMA_VERSION as _DEFAULT_CELL_SCHEMA_VERSION,
    LEGACY_CELL_SCHEMA_ID as _LEGACY_CELL_SCHEMA_ID,
    build_runtime_query_context as _build_runtime_query_context,
    inject_runtime_query_context as _inject_runtime_query_context,
    append_qc_event_conn as _append_qc_event_conn,
    delete_run_conn as _delete_run_conn,
    get_run_conn as _get_run_conn,
    link_job_conn as _link_job_conn,
    load_axes_conn as _load_axes_conn,
    load_links_conn as _load_links_conn,
    lookup_succeeded_conn as _lookup_succeeded_conn,
    json_dumps as _json_dumps,
    json_loads as _json_loads,
    query_runs_conn as _query_runs_conn,
    row_to_record as _row_to_record,
    to_float as _to_float,
    to_int as _to_int,
    upsert_run_conn as _upsert_run_conn,
)
from .store_schema import initialize_schema


class _ClosingConnection(sqlite3.Connection):
    """SQLite connection whose context-manager exit also closes the handle."""

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, tb))
        finally:
            self.close()


class RunIndexStore:
    """Persist/query profiling run metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrapping = True
        try:
            self._init_schema()
        finally:
            self._bootstrapping = False

    def _connect(self) -> sqlite3.Connection:
        cur_name = threading.current_thread().name
        if (
            not cur_name.startswith("sqlite-")
            and not os.environ.get("GATEWAY_SQLITE_THREAD_BYPASS")
            and not getattr(self, "_bootstrapping", False)
        ):
            stack_summary = "".join(traceback.format_stack()[-5:-1])
            raise RuntimeError(
                f"SQLite connect from non-sqlite thread {cur_name!r}.  "
                f"All SQLite access must route through "
                f"GatewayHTTPService._run_index_call (sqlite-* "
                f"dedicated executor, max_workers=1).  This guard "
                f"prevents -class fcntl deadlock + -class "
                f"MainThread block.  Set env "
                f"GATEWAY_SQLITE_THREAD_BYPASS=1 to bypass for unit "
                f"tests / scripts.\n"
                f"Caller stack:\n{stack_summary}"
            )
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            factory=_ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_core_schema(conn)
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            initialize_schema(
                conn,
                legacy_cell_schema_id=_LEGACY_CELL_SCHEMA_ID,
                ensure_column=self._ensure_column,
                run_startup_backfill=self._run_startup_backfill,
            )

    def _ensure_core_schema(self, conn: sqlite3.Connection) -> None:
        """Self-heal an empty/recreated DB file before serving queries.

        The benchmark wrappers intentionally wipe ``.index/profile_runs.sqlite3``
        between reps, and operational cleanup can race with a live gateway if a
        stray script or operator action removes the file mid-run.  In that case
        SQLite happily creates a fresh empty file on the next connect, and every
        query crashes with ``OperationalError: no such table``.  Scheduler
        lifecycle must not depend on profiling DB continuity, so we cheaply
        verify the two core tables on every connect and recreate the schema if
        the file was replaced underneath us.
        """
        required = {"profile_runs", "campaign_registry"}
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('profile_runs', 'campaign_registry')
            """,
        ).fetchall()
        existing = {str(row["name"]) for row in rows}
        if required.issubset(existing):
            return
        initialize_schema(
            conn,
            legacy_cell_schema_id=_LEGACY_CELL_SCHEMA_ID,
            ensure_column=self._ensure_column,
            run_startup_backfill=self._run_startup_backfill,
        )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, declaration: str) -> None:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing = {str(row["name"]) for row in columns}
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {declaration}")

    def _run_startup_backfill(self, conn: sqlite3.Connection) -> None:
        contracts = self._load_contract_backfill_contracts()
        if not contracts:
            self._backfill_runtime_query_context(conn)
            self._backfill_component_model_versions(conn)
            return
        signature = self._backfill_signature(contracts)
        migration_key = f"current_memberships:{signature}"
        existing = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE key = ?",
            (migration_key,),
        ).fetchone()
        if existing is None:
            self._backfill_current_memberships(conn, contracts)
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_migrations (key, applied_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    migration_key,
                    time.time(),
                    _json_dumps(
                        {
                            "algo_version": _CELL_SCHEMA_ALGO_VERSION,
                            "contracts": contracts,
                        }
                    ),
                ),
            )
        self._backfill_runtime_query_context(conn)
        self._backfill_component_model_versions(conn)

    def _backfill_runtime_query_context(self, conn: sqlite3.Connection) -> None:
        migration_key = "runtime_query_context:canonical_regime:v1"
        existing = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE key = ?",
            (migration_key,),
        ).fetchone()
        if existing is not None:
            return
        run_rows = conn.execute(
            """
            SELECT
                run_key,
                component,
                level,
                campaign_id,
                config_fingerprint,
                input_fingerprint,
                input_batch_size,
                active_memory_scope,
                dispatch_worker_name,
                dispatch_gpu_ids_json,
                concurrent_execute_overlap,
                scheduler_decision_json
            FROM profile_runs
            """
        ).fetchall()
        if run_rows:
            run_keys = [str(row["run_key"] or "").strip() for row in run_rows if str(row["run_key"] or "").strip()]
            axes_map = self._load_axes(conn, run_keys)
            for row in run_rows:
                run_key = str(row["run_key"] or "").strip()
                if not run_key:
                    continue
                scheduler_decision = _json_loads(row["scheduler_decision_json"], {})
                if not isinstance(scheduler_decision, Mapping):
                    scheduler_decision = {}
                runtime_query_context = _build_runtime_query_context(
                    scheduler_decision=scheduler_decision,
                    component=str(row["component"] or "").strip().lower(),
                    level=str(row["level"] or "").strip().lower(),
                    config_fingerprint=str(row["config_fingerprint"] or "").strip(),
                    input_fingerprint=str(row["input_fingerprint"] or "").strip(),
                    axes=axes_map.get(run_key, {}),
                    campaign_id=str(row["campaign_id"] or "").strip(),
                    input_batch_size=_to_int(row["input_batch_size"]),
                    active_memory_scope=str(row["active_memory_scope"] or "").strip(),
                    dispatch_worker_name=str(row["dispatch_worker_name"] or "").strip(),
                    dispatch_gpu_ids=_json_loads(row["dispatch_gpu_ids_json"], []),
                    concurrent_execute_overlap=bool(_to_int(row["concurrent_execute_overlap"]) or 0),
                )
                scheduler_decision = _inject_runtime_query_context(
                    scheduler_decision,
                    runtime_query_context,
                )
                conn.execute(
                    """
                    UPDATE profile_runs
                    SET scheduler_decision_json = ?
                    WHERE run_key = ?
                    """,
                    (_json_dumps(scheduler_decision), run_key),
                )
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_migrations (key, applied_at, payload_json)
            VALUES (?, ?, ?)
            """,
            (migration_key, time.time(), _json_dumps({"version": 1, "mode": "canonical_runtime_query_context"})),
        )

    @staticmethod
    def _load_contract_backfill_contracts() -> Dict[str, Dict[str, Any]]:
        contracts_dir = (Path(__file__).resolve().parent / "contracts").resolve()
        contracts = load_contracts(contracts_dir)
        out: Dict[str, Dict[str, Any]] = {}
        for component, contract in contracts.items():
            subset = [str(item).strip() for item in list(contract.cell_axis_subset or []) if str(item).strip()]
            out[str(component).strip().lower()] = {
                "subset": sorted(dict.fromkeys(subset)),
                "version": int(contract.cell_schema_version or _DEFAULT_CELL_SCHEMA_VERSION),
            }
        return out

    @staticmethod
    def _backfill_signature(contracts: Mapping[str, Mapping[str, Any]]) -> str:
        payload = _json_dumps(dict(contracts))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _backfill_current_memberships(
        self,
        conn: sqlite3.Connection,
        contracts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        run_rows = conn.execute(
            """
            SELECT run_key, component, level, sample_id, input_batch_size, output_sample_count
            FROM profile_runs
            """
        ).fetchall()
        if not run_rows:
            return
        run_keys = [str(row["run_key"]) for row in run_rows if str(row["run_key"]).strip()]
        axes_map = self._load_axes(conn, run_keys)
        now_ts = time.time()

        for row in run_rows:
            run_key = str(row["run_key"] or "").strip()
            if not run_key:
                continue
            component = str(row["component"] or "").strip().lower()
            contract = contracts.get(component)
            if not contract:
                continue
            level = str(row["level"] or "").strip().lower() or "level_a"
            subset = [str(item).strip() for item in list(contract.get("subset") or []) if str(item).strip()]
            schema_version = int(contract.get("version") or _DEFAULT_CELL_SCHEMA_VERSION)
            schema_id = build_cell_schema_id(
                component=component,
                level=level,
                cell_axis_subset=subset,
                cell_schema_version=schema_version,
                algo_version=_CELL_SCHEMA_ALGO_VERSION,
            )
            axes = dict(axes_map.get(run_key, {}))
            sample_id = str(row["sample_id"] or "").strip()
            input_batch_size = _to_int(row["input_batch_size"])
            output_sample_count = _to_int(row["output_sample_count"])
            if sample_id:
                axes.setdefault("sample_id", sample_id)
            if input_batch_size is not None:
                axes.setdefault("input_batch_size", int(input_batch_size))
            if output_sample_count is not None:
                axes.setdefault("output_sample_count", int(output_sample_count))
            axes.setdefault("component", component)
            cell_key = build_cell_key(
                component=component,
                level=level,
                axes=axes,
                subset=subset if subset else None,
                schema_id=schema_id,
            )
            conn.execute(
                """
                INSERT INTO run_cell_memberships (
                    run_key, cell_schema_id, cell_key, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key, cell_schema_id) DO UPDATE SET
                    cell_key = excluded.cell_key,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    run_key,
                    schema_id,
                    str(cell_key or ""),
                    "contract_backfill",
                    now_ts,
                    now_ts,
                ),
            )
            conn.execute(
                """
                UPDATE profile_runs
                SET cell_schema_id = ?, cell_key = ?
                WHERE run_key = ?
                """,
                (schema_id, str(cell_key or ""), run_key),
            )

    def _backfill_component_model_versions(self, conn: sqlite3.Connection) -> None:
        migration_key = "component_model_versions:legacy_backfill:v2"
        existing = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE key = ?",
            (migration_key,),
        ).fetchone()
        if existing is not None:
            return
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_migrations (key, applied_at, payload_json)
            VALUES (?, ?, ?)
            """,
            (migration_key, time.time(), _json_dumps({"version": 2, "mode": "no_control_backfill"})),
        )

    def upsert_run(
        self,
        *,
        run_key: str,
        run_source: str = "task",
        source_event_id: str = "",
        cell_schema_id: str = _LEGACY_CELL_SCHEMA_ID,
        cell_key: str = "",
        run_id: str,
        campaign_id: str = "",
        run_name: str = "",
        submitter: str = "",
        campaign_metadata_version: int = 0,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        component: str,
        level: str,
        state: str,
        sample_id: str = "",
        input_batch_size: Optional[int] = None,
        output_sample_count: Optional[int] = None,
        repeat_idx: Optional[int] = None,
        dispatch_worker_name: str = "",
        dispatch_worker_addr: str = "",
        dispatch_gpu_ids: Optional[Sequence[str]] = None,
        gateway_instance_id: str = "",
        gateway_bind_addr: str = "",
        gateway_git_commit: str = "",
        gateway_started_at: Optional[float] = None,
        worker_timing_us: Optional[Mapping[str, Any]] = None,
        artifact_manifest: Optional[Sequence[Mapping[str, Any]]] = None,
        runtime_sec: Optional[float] = None,
        mean_gpu_util_percent: Optional[float] = None,
        peak_memory_mib: Optional[float] = None,
        active_memory_mib: Optional[float] = None,
        active_memory_scope: str = "",
        active_memory_measurement: str = "",
        peak_vram_mib: Optional[float] = None,
        active_vram_mib: Optional[float] = None,
        vram_memory_measurement: str = "",
        vram_memory_qc_keep: bool = False,
        vram_memory_attribution: str = "",
        host_peak_memory_mib: Optional[float] = None,
        host_active_memory_mib: Optional[float] = None,
        host_resident_memory_mib: Optional[float] = None,
        host_memory_measurement: str = "",
        host_memory_qc_keep: bool = False,
        host_memory_attribution: str = "",
        resident_memory_mib: Optional[float] = None,
        resident_memory_source: str = "",
        resident_baseline_collected_at: Optional[float] = None,
        resident_baseline_lifecycle_token: str = "",
        resident_baseline_state: str = "",
        worker_generation_token: str = "",
        run_ordinal_in_generation: Optional[int] = None,
        is_first_real_run: bool = False,
        memory_qc_keep: bool = False,
        concurrent_execute_overlap: bool = False,
        telemetry_wall_clock_sec: Optional[float] = None,
        telemetry_collected_at: Optional[float] = None,
        bootstrap_memory_summary: Optional[Mapping[str, Any]] = None,
        dispatch_memory_window: Optional[Mapping[str, Any]] = None,
        scheduler_decision: Optional[Mapping[str, Any]] = None,
        predicted_runtime_sec: Optional[float] = None,
        predicted_p90_sec: Optional[float] = None,
        error: str = "",
        qc_status: str = "",
        qc_keep: Optional[bool] = None,
        qc_reason: str = "",
        qc_group_key: str = "",
        qc_attempt_count: Optional[int] = None,
        created_at: Optional[float] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        axes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            _upsert_run_conn(
                conn,
                run_key=run_key,
                run_source=run_source,
                source_event_id=source_event_id,
                cell_schema_id=cell_schema_id,
                cell_key=cell_key,
                run_id=run_id,
                campaign_id=campaign_id,
                run_name=run_name,
                submitter=submitter,
                campaign_metadata_version=campaign_metadata_version,
                config_fingerprint=config_fingerprint,
                input_fingerprint=input_fingerprint,
                component=component,
                level=level,
                state=state,
                sample_id=sample_id,
                input_batch_size=input_batch_size,
                output_sample_count=output_sample_count,
                repeat_idx=repeat_idx,
                dispatch_worker_name=dispatch_worker_name,
                dispatch_worker_addr=dispatch_worker_addr,
                dispatch_gpu_ids=dispatch_gpu_ids,
                gateway_instance_id=gateway_instance_id,
                gateway_bind_addr=gateway_bind_addr,
                gateway_git_commit=gateway_git_commit,
                gateway_started_at=gateway_started_at,
                worker_timing_us=worker_timing_us,
                artifact_manifest=artifact_manifest,
                runtime_sec=runtime_sec,
                mean_gpu_util_percent=mean_gpu_util_percent,
                peak_memory_mib=peak_memory_mib,
                active_memory_mib=active_memory_mib,
                active_memory_scope=active_memory_scope,
                active_memory_measurement=active_memory_measurement,
                peak_vram_mib=peak_vram_mib,
                active_vram_mib=active_vram_mib,
                vram_memory_measurement=vram_memory_measurement,
                vram_memory_qc_keep=vram_memory_qc_keep,
                vram_memory_attribution=vram_memory_attribution,
                host_peak_memory_mib=host_peak_memory_mib,
                host_active_memory_mib=host_active_memory_mib,
                host_resident_memory_mib=host_resident_memory_mib,
                host_memory_measurement=host_memory_measurement,
                host_memory_qc_keep=host_memory_qc_keep,
                host_memory_attribution=host_memory_attribution,
                resident_memory_mib=resident_memory_mib,
                resident_memory_source=resident_memory_source,
                resident_baseline_collected_at=resident_baseline_collected_at,
                resident_baseline_lifecycle_token=resident_baseline_lifecycle_token,
                resident_baseline_state=resident_baseline_state,
                worker_generation_token=worker_generation_token,
                run_ordinal_in_generation=run_ordinal_in_generation,
                is_first_real_run=is_first_real_run,
                memory_qc_keep=memory_qc_keep,
                concurrent_execute_overlap=concurrent_execute_overlap,
                telemetry_wall_clock_sec=telemetry_wall_clock_sec,
                telemetry_collected_at=telemetry_collected_at,
                bootstrap_memory_summary=bootstrap_memory_summary,
                dispatch_memory_window=dispatch_memory_window,
                scheduler_decision=scheduler_decision,
                predicted_runtime_sec=predicted_runtime_sec,
                predicted_p90_sec=predicted_p90_sec,
                error=error,
                qc_status=qc_status,
                qc_keep=qc_keep,
                qc_reason=qc_reason,
                qc_group_key=qc_group_key,
                qc_attempt_count=qc_attempt_count,
                created_at=created_at,
                started_at=started_at,
                finished_at=finished_at,
                axes=axes,
            )

    def upsert_campaign_registry(
        self,
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
        with self._connect() as conn:
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
                last_seen_at=last_seen_at,
                ended_at=ended_at,
                event_ts=event_ts,
            )

    def mark_campaign_task_state(
        self,
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
        with self._connect() as conn:
            mark_campaign_task_state_conn(
                conn,
                campaign_id=campaign_id,
                run_name=run_name,
                submitter=submitter,
                gateway_instance_id=gateway_instance_id,
                gateway_bind_addr=gateway_bind_addr,
                gateway_git_commit=gateway_git_commit,
                gateway_started_at=gateway_started_at,
                task_state=task_state,
                event_ts=event_ts,
                created_at=created_at,
            )

    def get_campaign_registry(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            return get_campaign_registry_conn(conn, campaign_id)

    def list_campaign_registry(self, *, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_campaign_registry_conn(conn, limit=limit, offset=offset)

    def link_job(self, *, job_id: str, kind: str, run_key: str) -> None:
        with self._connect() as conn:
            _link_job_conn(conn, job_id=job_id, kind=kind, run_key=run_key)

    def delete_run(self, run_key: str) -> None:
        with self._connect() as conn:
            _delete_run_conn(conn, run_key)

    def append_qc_event(
        self,
        *,
        run_key: str,
        action: str,
        reason: str = "",
        group_key: str = "",
        job_id: str = "",
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            _append_qc_event_conn(
                conn,
                run_key=run_key,
                action=action,
                reason=reason,
                group_key=group_key,
                job_id=job_id,
                payload=payload,
            )

    def get_run(self, run_key: str) -> Optional[RunRecord]:
        with self._connect() as conn:
            return _get_run_conn(conn, run_key)

    def lookup_succeeded(self, run_key: str) -> Optional[RunRecord]:
        """Lookup MODEL_INPUT_FILTER-eligible succeeded row with valid artifacts."""
        with self._connect() as conn:
            return _lookup_succeeded_conn(conn, run_key)

    def query_runs(self, query: RunQuery) -> Tuple[List[RunRecord], int]:
        with self._connect() as conn:
            return _query_runs_conn(conn, query)

    def query_observations(
        self,
        *,
        run_sources: Sequence[str],
        component: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 1000,
    ) -> List[RunRecord]:
        records: List[RunRecord] = []
        for run_source in [str(item).strip().lower() for item in list(run_sources or []) if str(item).strip()]:
            rows, _ = self.query_runs(
                RunQuery(
                    run_source=run_source,
                    component=component,
                    level=level,
                    state="SUCCEEDED",
                    sort="finished_at",
                    descending=True,
                    limit=limit,
                    offset=0,
                )
            )
            records.extend(rows)
        records.sort(
            key=lambda row: float(row.finished_at or row.updated_at or row.created_at or 0.0),
            reverse=True,
        )
        return records[: max(1, int(limit))]

    def upsert_cell_stats(
        self,
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
        with self._connect() as conn:
            upsert_cell_stats_conn(
                conn,
                cell_key=cell_key,
                component=component,
                level=level,
                n_valid=n_valid,
                mean_log_exec=mean_log_exec,
                std_log_exec=std_log_exec,
                ci95_halfwidth=ci95_halfwidth,
                rel_halfwidth=rel_halfwidth,
                bimodal_flag=bimodal_flag,
                contam_rate=contam_rate,
                stats=stats,
            )

    def list_cell_stats(
        self,
        *,
        component: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_cell_stats_conn(conn, component=component, level=level)

    def upsert_model_version(
        self,
        *,
        model_version: str,
        trained_at: Optional[float] = None,
        data_cut_ts: Optional[float] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        decision_gate: Optional[Mapping[str, Any]] = None,
        cell_schema_ids: Optional[Sequence[str]] = None,
        is_active: bool = False,
    ) -> None:
        with self._connect() as conn:
            upsert_model_version_conn(
                conn,
                model_version=model_version,
                trained_at=trained_at,
                data_cut_ts=data_cut_ts,
                metrics=metrics,
                decision_gate=decision_gate,
                cell_schema_ids=cell_schema_ids,
                is_active=is_active,
            )

    def list_model_versions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_model_versions_conn(conn)

    def upsert_component_model_version(
        self,
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
        with self._connect() as conn:
            upsert_component_model_version_conn(
                conn,
                component_model_version_id=component_model_version_id,
                component=component,
                parent_loop_model_version_id=parent_loop_model_version_id,
                job_id=job_id,
                loop_idx=loop_idx,
                cell_schema_ids=cell_schema_ids,
                metrics=metrics,
                decision_gate=decision_gate,
                readiness=readiness,
                priority=priority,
                delta=delta,
                segments=segments,
                artifacts=artifacts,
                status=status,
                is_active=is_active,
                trained_at=trained_at,
                error=error,
            )

    def get_component_model_version(self, component_model_version_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            return get_component_model_version_conn(conn, component_model_version_id)

    def list_component_model_versions(
        self,
        *,
        component: Optional[str] = None,
        job_id: Optional[str] = None,
        loop_idx: Optional[int] = None,
        is_active: Optional[bool] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_component_model_versions_conn(
                conn,
                component=component,
                job_id=job_id,
                loop_idx=loop_idx,
                is_active=is_active,
                limit=limit,
            )

    def list_latest_component_models(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_latest_component_models_conn(conn)

    def list_active_component_models(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            return list_active_component_models_conn(conn)

    def get_active_component_model(self, *, component: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            return get_active_component_model_conn(conn, component=component)

    def set_component_model_active(self, *, component: str, component_model_version_id: str) -> None:
        with self._connect() as conn:
            set_component_model_active_conn(
                conn,
                component=component,
                component_model_version_id=component_model_version_id,
            )

    def query_all_succeeded(self) -> List[RunRecord]:
        """Return ALL succeeded task runs, ordered oldest-first for replay.

        Unlike ``query_observations`` (capped at 1000), this method has no
        limit — it loads every succeeded task run so signal channels can be
        fully reconstructed on gateway startup.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*
                FROM profile_runs r
                WHERE r.run_source = 'task'
                  AND r.state = 'SUCCEEDED'
                ORDER BY r.finished_at ASC
                """
            ).fetchall()
            if not rows:
                return []
            run_keys = [str(row["run_key"]) for row in rows]
            axes_map = _load_axes_conn(conn, run_keys)
            return [
                _row_to_record(
                    row,
                    axes=axes_map.get(str(row["run_key"]), {}),
                    links=[],
                )
                for row in rows
            ]

    def query_all_signal_replay_rows(self) -> List[RunRecord]:
        """Return task rows eligible for SignalService bootstrap replay.

        Latency and workload-classifier channels still consume only
        ``SUCCEEDED`` rows.  Memory channels also need terminal failed/cancelled
        rows because live execution records retry/eviction/OOM partial memory
        observations before closing or requeueing an attempt.  Replaying only
        successful tasks would silently forget those risk signals after a
        gateway restart.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*
                FROM profile_runs r
                WHERE r.run_source = 'task'
                  AND r.state IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
                ORDER BY r.finished_at ASC
                """
            ).fetchall()
            if not rows:
                return []
            run_keys = [str(row["run_key"]) for row in rows]
            axes_map = _load_axes_conn(conn, run_keys)
            return [
                _row_to_record(
                    row,
                    axes=axes_map.get(str(row["run_key"]), {}),
                    links=[],
                )
                for row in rows
            ]

    def query_latest_resident_baselines(self) -> List[RunRecord]:
        """Return the latest resident-weight baseline row for each component.

        ``WorkerSupervisor.bootstrap_weights`` needs component coverage, not the
        latest N task rows.  Fan-out tails can contain hundreds of Vina/Protenix
        rows after early RFdiffusion/ProteinMPNN measurements, so a global
        ``LIMIT`` can silently drop valid resident baselines for components
        that only appear early in the run.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*
                FROM profile_runs r
                WHERE r.run_key IN (
                    SELECT p.run_key
                    FROM profile_runs p
                    WHERE p.run_source = 'task'
                      AND p.state = 'SUCCEEDED'
                      AND p.component = r.component
                      AND p.resident_memory_mib IS NOT NULL
                      AND p.resident_memory_mib > 0
                    ORDER BY p.finished_at DESC, p.updated_at DESC, p.run_key DESC
                    LIMIT 1
                )
                ORDER BY r.component ASC
                """
            ).fetchall()
            if not rows:
                return []
            run_keys = [str(row["run_key"]) for row in rows]
            axes_map = _load_axes_conn(conn, run_keys)
            return [
                _row_to_record(
                    row,
                    axes=axes_map.get(str(row["run_key"]), {}),
                    links=[],
                )
                for row in rows
            ]

    def save_signal_snapshot(self, key: str, data: Dict[str, Any]) -> None:
        """Persist a signal snapshot (e.g., interference matrix) to DB."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_snapshots (key, snapshot_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    snapshot_json = excluded.snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (key, _json_dumps(data), time.time()),
            )

    def load_signal_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a signal snapshot from DB. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot_json, updated_at FROM signal_snapshots WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return _json_loads(row["snapshot_json"], {})

    def query_all_signal_replay_timeline_entries(self) -> List[Dict[str, Any]]:
        """Return exported GPU timeline entries for signal bootstrap replay.

        The live signal-convergence renderer stores ``signals_post`` GPU
        timeline entries in a small replay facade so historical bootstrap can
        reconstruct co-location-aware latency observations.  Production
        bootstrap uses the same hook when a seeded profile DB contains this
        optional snapshot.  Normal benchmark DBs do not carry it, so this is a
        no-op unless an experiment explicitly imports the timeline context.
        """
        snapshot = self.load_signal_snapshot("signal_replay_timeline_entries_v1")
        if not isinstance(snapshot, Mapping):
            return []
        entries = snapshot.get("entries")
        if not isinstance(entries, list):
            return []
        return [dict(entry) for entry in entries if isinstance(entry, Mapping)]

    def summarize_component_segments(self, segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return summarize_component_segments(segments)

    def _load_axes(self, conn: sqlite3.Connection, run_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        return _load_axes_conn(conn, run_keys)

    def _load_links(self, conn: sqlite3.Connection, run_keys: Sequence[str]) -> Dict[str, List[Tuple[str, str]]]:
        return _load_links_conn(conn, run_keys)

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
        *,
        axes: Mapping[str, Any],
        links: Sequence[Tuple[str, str]],
    ) -> RunRecord:
        return _row_to_record(row, axes=axes, links=links)
