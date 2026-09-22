"""In-process telemetry-first component-scoped evaluator for task runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from modeling.collect_features import collect_features_dataframe

from .constants import MODEL_INPUT_RUN_SOURCES
from .history_model import evaluate_component_history


class ModelEvalRunner:
    def __init__(self, *, repo_root: Path) -> None:
        _ = repo_root

    def run(
        self,
        *,
        index_db: Path,
        level: str,
        components: Iterable[str],
        cell_schema_ids: Iterable[str],
        outdir: Path,
        timeout_s: int = 3600,
    ) -> Dict[str, Any]:
        _ = timeout_s
        out_root = Path(outdir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        features_dir = out_root / "features"
        modeling_root = out_root / "modeling"
        features_csv = features_dir / "features.csv"

        component_args = [str(item).strip().lower() for item in list(components or []) if str(item).strip()]
        schema_args = [str(item).strip() for item in list(cell_schema_ids or []) if str(item).strip()]

        features_dir.mkdir(parents=True, exist_ok=True)
        feature_df = collect_features_dataframe(
            index_db=Path(index_db).resolve(),
            level=str(level),
            run_sources=list(MODEL_INPUT_RUN_SOURCES),
            components=component_args,
            cell_schema_ids=schema_args if schema_args else None,
            qc_keep_only=True,
        )
        feature_df.to_csv(features_csv, index=False)
        component_results: Dict[str, Dict[str, Any]] = {}
        for component in component_args:
            comp_df = feature_df[feature_df["component"].astype(str).str.strip().str.lower() == component].copy()
            comp_outdir = modeling_root / component
            comp_gate_path = comp_outdir / "decision_gate.json"
            comp_summary_path = comp_outdir / "summary.json"
            if comp_df.empty:
                component_results[component] = {
                    "ok": False,
                    "error": "no rows for component in collected features",
                    "modeling_dir": str(comp_outdir),
                    "summary_json": "",
                    "decision_gate_json": "",
                    "summary": {},
                    "decision_gate": {},
                }
                continue
            last_error = ""
            for attempt in range(2):
                try:
                    comp_outdir.mkdir(parents=True, exist_ok=True)
                    metrics = evaluate_component_history(comp_df)
                    n_obs_recent = int(metrics.get("n_obs_recent") or 0)
                    unique_inputs = int(metrics.get("unique_inputs") or 0)
                    unique_configs = int(metrics.get("unique_configs") or 0)
                    runtime_ape_p50 = metrics.get("runtime_ape_p50")
                    if runtime_ape_p50 is not None:
                        runtime_ape_p50 = float(runtime_ape_p50)
                    runtime_ape_p90 = metrics.get("runtime_ape_p90")
                    if runtime_ape_p90 is not None:
                        runtime_ape_p90 = float(runtime_ape_p90)
                    runtime_bias_sec_median = metrics.get("runtime_bias_sec_median")
                    if runtime_bias_sec_median is not None:
                        runtime_bias_sec_median = float(runtime_bias_sec_median)
                    runtime_log_mae = metrics.get("runtime_log_mae")
                    if runtime_log_mae is not None:
                        runtime_log_mae = float(runtime_log_mae)
                    mem_fn_rate = metrics.get("mem_fn_rate")
                    if mem_fn_rate is not None:
                        mem_fn_rate = float(mem_fn_rate)
                    gap_fit_fp_rate = metrics.get("gap_fit_fp_rate")
                    if gap_fit_fp_rate is not None:
                        gap_fit_fp_rate = float(gap_fit_fp_rate)
                    residual_p90 = metrics.get("residual_p90")
                    if residual_p90 is not None:
                        residual_p90 = float(residual_p90)
                    residual_p95 = metrics.get("residual_p95")
                    if residual_p95 is not None:
                        residual_p95 = float(residual_p95)
                    calibration_error = metrics.get("calibration_error")
                    if calibration_error is not None:
                        calibration_error = float(calibration_error)
                    summary_payload = {
                        "run_meta": {
                            "rows_total": int(len(comp_df)),
                            "component": component,
                            "level": str(level),
                        },
                        "history_metrics": dict(metrics),
                        "target_data_points": {
                            "runtime_sec": {
                                "validation_datapoints_total": int(len(comp_df)),
                                "training_datapoints_per_fold_avg": None,
                                "k_folds_min": None,
                                "k_folds_max": None,
                                "k_folds_mode": None,
                                "segments": int(comp_df.get("dispatch_gpu_id", []).nunique() or 0)
                                if "dispatch_gpu_id" in comp_df.columns
                                else None,
                            }
                        },
                    }
                    gate_payload = {
                        "promote": False,
                        "readiness_summary": {
                            "n_obs_recent": n_obs_recent,
                            "unique_inputs": unique_inputs,
                            "unique_configs": unique_configs,
                        },
                        "decision_quality": {
                            "runtime_ape_p50": runtime_ape_p50,
                            "runtime_ape_p90": runtime_ape_p90,
                            "runtime_bias_sec_median": runtime_bias_sec_median,
                            "runtime_log_mae": runtime_log_mae,
                            "mem_fn_rate": mem_fn_rate,
                            "gap_fit_fp_rate": gap_fit_fp_rate,
                            "residual_p90": residual_p90,
                            "residual_p95": residual_p95,
                            "calibration_error": calibration_error,
                            "n_runtime_checks": int(metrics.get("n_runtime_checks") or 0),
                            "n_mem_checks": int(metrics.get("n_mem_checks") or 0),
                            "n_gap_fit_checks": int(metrics.get("n_gap_fit_checks") or 0),
                            "n_residual_checks": int(metrics.get("n_residual_checks") or 0),
                            "n_calibration_checks": int(metrics.get("n_calibration_checks") or 0),
                            "quality_score": None,
                            "confidence": None,
                        },
                    }
                    comp_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
                    comp_gate_path.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
                    component_results[component] = {
                        "ok": True,
                        "error": "",
                        "attempt_count": attempt + 1,
                        "modeling_dir": str(comp_outdir),
                        "summary_json": str(comp_summary_path),
                        "decision_gate_json": str(comp_gate_path),
                        "summary": summary_payload,
                        "decision_gate": gate_payload,
                    }
                    break
                except Exception as exc:
                    last_error = str(exc)
            else:
                component_results[component] = {
                    "ok": False,
                    "error": last_error,
                    "attempt_count": 2,
                    "modeling_dir": str(comp_outdir),
                    "summary_json": "",
                    "decision_gate_json": "",
                    "summary": {},
                    "decision_gate": {},
                }
        return {
            "features_csv": str(features_csv),
            "modeling_dir": str(modeling_root),
            "component_results": component_results,
        }
