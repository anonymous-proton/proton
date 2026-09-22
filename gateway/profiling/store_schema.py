"""Schema/bootstrap helpers for profiling index store."""

from __future__ import annotations

import sqlite3
from typing import Callable


def _apply_hardcut_cleanup(conn: sqlite3.Connection) -> None:
    migration_key = "hardcut_remove_control_surfaces_v2"
    existing = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE key = ?",
        (migration_key,),
    ).fetchone()
    if existing is not None:
        return

    conn.execute("DROP INDEX IF EXISTS idx_control_steps_job_updated;")
    conn.execute("DROP INDEX IF EXISTS idx_control_jobs_state_updated;")
    conn.execute("DROP INDEX IF EXISTS idx_component_policy_state_policy_updated;")
    conn.execute("DROP TABLE IF EXISTS control_steps;")
    conn.execute("DROP TABLE IF EXISTS control_jobs;")
    conn.execute("DROP TABLE IF EXISTS component_policy_state;")
    conn.execute("DROP TABLE IF EXISTS component_policy_state__legacy_hardcut;")

    conn.execute(
        """
        INSERT OR REPLACE INTO schema_migrations (key, applied_at, payload_json)
        VALUES (?, strftime('%s','now'), '{"mode":"hardcut_cleanup_v2"}')
        """,
        (migration_key,),
    )


def initialize_schema(
    conn: sqlite3.Connection,
    *,
    legacy_cell_schema_id: str,
    ensure_column: Callable[[sqlite3.Connection, str, str, str], None],
    run_startup_backfill: Callable[[sqlite3.Connection], None],
) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS profile_runs (
            run_key TEXT PRIMARY KEY,
            cell_schema_id TEXT NOT NULL DEFAULT 'legacy_v0',
            cell_key TEXT NOT NULL DEFAULT '',
            run_id TEXT NOT NULL,
            run_source TEXT NOT NULL DEFAULT 'task',
            source_event_id TEXT NOT NULL DEFAULT '',
            config_fingerprint TEXT NOT NULL DEFAULT '',
            input_fingerprint TEXT NOT NULL DEFAULT '',
            campaign_id TEXT NOT NULL DEFAULT '',
            run_name TEXT NOT NULL DEFAULT '',
            submitter TEXT NOT NULL DEFAULT '',
            campaign_metadata_version INTEGER NOT NULL DEFAULT 0,
            component TEXT NOT NULL,
            level TEXT NOT NULL,
            state TEXT NOT NULL,
            sample_id TEXT,
            input_batch_size INTEGER,
            output_sample_count INTEGER,
            repeat_idx INTEGER,
            dispatch_worker_name TEXT,
            dispatch_worker_addr TEXT,
            dispatch_gpu_ids_json TEXT NOT NULL DEFAULT '[]',
            worker_timing_us_json TEXT,
            artifact_manifest_json TEXT NOT NULL DEFAULT '[]',
            runtime_sec REAL,
            mean_gpu_util_percent REAL,
            peak_memory_mib REAL,
            active_memory_mib REAL,
            active_memory_scope TEXT NOT NULL DEFAULT '',
            active_memory_measurement TEXT NOT NULL DEFAULT '',
            peak_vram_mib REAL,
            active_vram_mib REAL,
            vram_memory_measurement TEXT NOT NULL DEFAULT '',
            vram_memory_qc_keep INTEGER NOT NULL DEFAULT 0,
            vram_memory_attribution TEXT NOT NULL DEFAULT '',
            host_peak_memory_mib REAL,
            host_active_memory_mib REAL,
            host_resident_memory_mib REAL,
            host_memory_measurement TEXT NOT NULL DEFAULT '',
            host_memory_qc_keep INTEGER NOT NULL DEFAULT 0,
            host_memory_attribution TEXT NOT NULL DEFAULT '',
            resident_memory_mib REAL,
            resident_memory_source TEXT NOT NULL DEFAULT '',
            resident_baseline_collected_at REAL,
            resident_baseline_lifecycle_token TEXT NOT NULL DEFAULT '',
            resident_baseline_state TEXT NOT NULL DEFAULT '',
            worker_generation_token TEXT NOT NULL DEFAULT '',
            run_ordinal_in_generation INTEGER,
            is_first_real_run INTEGER NOT NULL DEFAULT 0,
            memory_qc_keep INTEGER NOT NULL DEFAULT 0,
            concurrent_execute_overlap INTEGER NOT NULL DEFAULT 0,
            telemetry_wall_clock_sec REAL,
            telemetry_collected_at REAL,
            bootstrap_memory_summary_json TEXT NOT NULL DEFAULT '{}',
            dispatch_memory_window_json TEXT NOT NULL DEFAULT '{}',
            scheduler_decision_json TEXT NOT NULL DEFAULT '{}',
            predicted_runtime_sec REAL,
            predicted_p90_sec REAL,
            error TEXT NOT NULL DEFAULT '',
            qc_keep INTEGER NOT NULL DEFAULT 1,
            created_at REAL,
            started_at REAL,
            finished_at REAL,
            updated_at REAL
        );

        CREATE TABLE IF NOT EXISTS campaign_registry (
            campaign_id TEXT PRIMARY KEY,
            run_name TEXT NOT NULL DEFAULT '',
            submitter TEXT NOT NULL DEFAULT '',
            gateway_instance_id TEXT NOT NULL DEFAULT '',
            gateway_bind_addr TEXT NOT NULL DEFAULT '',
            gateway_git_commit TEXT NOT NULL DEFAULT '',
            gateway_started_at REAL,
            status TEXT NOT NULL DEFAULT 'waiting',
            created_at REAL,
            started_at REAL,
            last_seen_at REAL,
            ended_at REAL,
            updated_at REAL
        );

        CREATE TABLE IF NOT EXISTS run_axes (
            run_key TEXT NOT NULL,
            axis_key TEXT NOT NULL,
            axis_type TEXT NOT NULL,
            axis_value_text TEXT,
            axis_value_num REAL,
            PRIMARY KEY (run_key, axis_key),
            FOREIGN KEY (run_key) REFERENCES profile_runs(run_key) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS job_links (
            job_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            run_key TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
            PRIMARY KEY (job_id, run_key),
            FOREIGN KEY (run_key) REFERENCES profile_runs(run_key) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS run_qc_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            group_key TEXT NOT NULL DEFAULT '',
            job_id TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS run_cell_memberships (
            run_key TEXT NOT NULL,
            cell_schema_id TEXT NOT NULL,
            cell_key TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (run_key, cell_schema_id),
            FOREIGN KEY (run_key) REFERENCES profile_runs(run_key) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS cell_stats (
            cell_key TEXT PRIMARY KEY,
            component TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT '',
            n_valid INTEGER NOT NULL DEFAULT 0,
            mean_log_exec REAL,
            std_log_exec REAL,
            ci95_halfwidth REAL,
            rel_halfwidth REAL,
            bimodal_flag INTEGER NOT NULL DEFAULT 0,
            contam_rate REAL,
            last_updated REAL NOT NULL,
            stats_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            model_version TEXT PRIMARY KEY,
            trained_at REAL NOT NULL,
            data_cut_ts REAL,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            decision_gate_json TEXT NOT NULL DEFAULT '{}',
            cell_schema_ids_json TEXT NOT NULL DEFAULT '[]',
            is_active INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS component_model_versions (
            component_model_version_id TEXT PRIMARY KEY,
            component TEXT NOT NULL,
            parent_loop_model_version_id TEXT NOT NULL,
            job_id TEXT NOT NULL DEFAULT '',
            loop_idx INTEGER NOT NULL DEFAULT 0,
            cell_schema_ids_json TEXT NOT NULL DEFAULT '[]',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            decision_gate_json TEXT NOT NULL DEFAULT '{}',
            readiness_json TEXT NOT NULL DEFAULT '{}',
            priority_json TEXT NOT NULL DEFAULT '{}',
            delta_json TEXT NOT NULL DEFAULT '{}',
            segments_json TEXT NOT NULL DEFAULT '[]',
            artifacts_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 0,
            trained_at REAL NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            UNIQUE (job_id, loop_idx, component)
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            key TEXT PRIMARY KEY,
            applied_at REAL NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS signal_snapshots (
            key TEXT PRIMARY KEY,
            snapshot_json TEXT NOT NULL DEFAULT '{}',
            updated_at REAL NOT NULL
        );
        """
    )

    ensure_column(conn, "profile_runs", "component", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "level", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "state", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "sample_id", "TEXT")
    ensure_column(conn, "profile_runs", "run_source", "TEXT NOT NULL DEFAULT 'task'")
    ensure_column(conn, "profile_runs", "source_event_id", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "config_fingerprint", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "input_fingerprint", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "campaign_id", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "run_name", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "submitter", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "campaign_metadata_version", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "input_batch_size", "INTEGER")
    ensure_column(conn, "profile_runs", "output_sample_count", "INTEGER")
    ensure_column(conn, "profile_runs", "repeat_idx", "INTEGER")
    ensure_column(conn, "profile_runs", "dispatch_worker_name", "TEXT")
    ensure_column(conn, "profile_runs", "dispatch_worker_addr", "TEXT")
    ensure_column(conn, "profile_runs", "dispatch_gpu_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    ensure_column(conn, "profile_runs", "worker_timing_us_json", "TEXT")
    ensure_column(conn, "profile_runs", "artifact_manifest_json", "TEXT NOT NULL DEFAULT '[]'")
    ensure_column(conn, "profile_runs", "runtime_sec", "REAL")
    ensure_column(conn, "profile_runs", "error", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "created_at", "REAL")
    ensure_column(conn, "profile_runs", "started_at", "REAL")
    ensure_column(conn, "profile_runs", "updated_at", "REAL")
    ensure_column(conn, "profile_runs", "finished_at", "REAL")
    ensure_column(conn, "profile_runs", "mean_gpu_util_percent", "REAL")
    ensure_column(conn, "profile_runs", "peak_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "active_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "active_memory_scope", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "active_memory_measurement", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "peak_vram_mib", "REAL")
    ensure_column(conn, "profile_runs", "active_vram_mib", "REAL")
    ensure_column(conn, "profile_runs", "vram_memory_measurement", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "vram_memory_qc_keep", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "vram_memory_attribution", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "host_peak_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "host_active_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "host_resident_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "host_memory_measurement", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "host_memory_qc_keep", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "host_memory_attribution", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "resident_memory_mib", "REAL")
    ensure_column(conn, "profile_runs", "resident_memory_source", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "resident_baseline_collected_at", "REAL")
    ensure_column(conn, "profile_runs", "resident_baseline_lifecycle_token", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "resident_baseline_state", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "worker_generation_token", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "run_ordinal_in_generation", "INTEGER")
    ensure_column(conn, "profile_runs", "is_first_real_run", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "memory_qc_keep", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "concurrent_execute_overlap", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "profile_runs", "telemetry_wall_clock_sec", "REAL")
    ensure_column(conn, "profile_runs", "telemetry_collected_at", "REAL")
    ensure_column(conn, "profile_runs", "bootstrap_memory_summary_json", "TEXT NOT NULL DEFAULT '{}'")
    ensure_column(conn, "profile_runs", "dispatch_memory_window_json", "TEXT NOT NULL DEFAULT '{}'")
    ensure_column(conn, "profile_runs", "scheduler_decision_json", "TEXT NOT NULL DEFAULT '{}'")
    ensure_column(conn, "profile_runs", "predicted_runtime_sec", "REAL")
    ensure_column(conn, "profile_runs", "predicted_p90_sec", "REAL")

    ensure_column(conn, "profile_runs", "qc_status", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "qc_reason", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "qc_group_key", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "qc_attempt_count", "INTEGER NOT NULL DEFAULT 1")
    ensure_column(conn, "profile_runs", "qc_keep", "INTEGER NOT NULL DEFAULT 1")
    ensure_column(conn, "profile_runs", "cell_key", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "cell_schema_id", f"TEXT NOT NULL DEFAULT '{legacy_cell_schema_id}'")
    ensure_column(conn, "profile_runs", "gateway_instance_id", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "gateway_bind_addr", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "gateway_git_commit", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "profile_runs", "gateway_started_at", "REAL")
    ensure_column(conn, "campaign_registry", "gateway_instance_id", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "campaign_registry", "gateway_bind_addr", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "campaign_registry", "gateway_git_commit", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "campaign_registry", "gateway_started_at", "REAL")
    ensure_column(conn, "cell_stats", "level", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "model_versions", "cell_schema_ids_json", "TEXT NOT NULL DEFAULT '[]'")

    conn.execute(
        """
        UPDATE profile_runs
        SET qc_status = 'soft_drop'
        WHERE lower(trim(qc_status)) IN ('drop', 'deleted_outlier')
        """
    )
    conn.execute(
        """
        UPDATE profile_runs
        SET qc_status = lower(trim(qc_status))
        WHERE qc_status IS NOT NULL
        """
    )
    conn.execute(
        """
        UPDATE profile_runs
        SET qc_keep = CASE
            WHEN lower(trim(qc_status)) = 'soft_drop' THEN 0
            ELSE 1
        END
        """
    )
    conn.execute(
        f"""
        UPDATE profile_runs
        SET cell_schema_id = '{legacy_cell_schema_id}'
        WHERE cell_schema_id IS NULL OR trim(cell_schema_id) = ''
        """
    )
    conn.execute(
        """
        UPDATE profile_runs
        SET run_source = CASE
            WHEN run_source IS NULL OR trim(run_source) = '' THEN 'task'
            ELSE lower(trim(run_source))
        END
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO run_cell_memberships (
            run_key, cell_schema_id, cell_key, source, created_at, updated_at
        )
        SELECT
            run_key,
            ?,
            COALESCE(cell_key, ''),
            'legacy',
            COALESCE(created_at, strftime('%s','now')),
            COALESCE(updated_at, strftime('%s','now'))
        FROM profile_runs
        """,
        (legacy_cell_schema_id,),
    )
    _apply_hardcut_cleanup(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profile_runs_qc_group ON profile_runs(qc_group_key);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profile_runs_qc_status ON profile_runs(qc_status);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_component_level_state_finished "
        "ON profile_runs(component, level, state, finished_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_source_component_state_finished "
        "ON profile_runs(run_source, component, state, finished_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_campaign_state_finished "
        "ON profile_runs(campaign_id, state, finished_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_campaign_component_state_finished "
        "ON profile_runs(campaign_id, component, state, finished_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_campaign_updated "
        "ON profile_runs(campaign_id, updated_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_campaign_meta_queue "
        "ON profile_runs(campaign_metadata_version, campaign_id, state, updated_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_source_component_cfg_input "
        "ON profile_runs(run_source, component, config_fingerprint, input_fingerprint);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_campaign_registry_status_last_seen "
        "ON campaign_registry(status, last_seen_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_campaign_registry_updated_at "
        "ON campaign_registry(updated_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_campaign_registry_gateway_instance_last_seen "
        "ON campaign_registry(gateway_instance_id, last_seen_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_campaign_registry_gateway_commit_last_seen "
        "ON campaign_registry(gateway_git_commit, last_seen_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_qc_source_finished "
        "ON profile_runs(qc_keep, run_source, finished_at);"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_runs_source_event_id_unique "
        "ON profile_runs(run_source, source_event_id) WHERE source_event_id != '';"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profile_runs_cell_key ON profile_runs(cell_key);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profile_runs_cell_schema ON profile_runs(cell_schema_id);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_sample_inp_out "
        "ON profile_runs(sample_id, input_batch_size, output_sample_count);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_profile_runs_worker_name ON profile_runs(dispatch_worker_name);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_gateway_instance_campaign_state_finished "
        "ON profile_runs(gateway_instance_id, campaign_id, state, finished_at);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_runs_gateway_commit_campaign_state_finished "
        "ON profile_runs(gateway_git_commit, campaign_id, state, finished_at);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_axes_key_text ON run_axes(axis_key, axis_value_text);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_axes_key_num ON run_axes(axis_key, axis_value_num);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_links_kind ON job_links(kind, created_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_qc_events_run_key_created ON run_qc_events(run_key, created_at);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_run_cell_memberships_schema_cell "
        "ON run_cell_memberships(cell_schema_id, cell_key);"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cell_stats_component_level ON cell_stats(component, level, last_updated);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_component_model_versions_component_trained "
        "ON component_model_versions(component, trained_at DESC);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_component_model_versions_component_active "
        "ON component_model_versions(component, is_active);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_component_model_versions_parent "
        "ON component_model_versions(parent_loop_model_version_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_component_model_versions_job_loop "
        "ON component_model_versions(job_id, loop_idx);"
    )

    run_startup_backfill(conn)
