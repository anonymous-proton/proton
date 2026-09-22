"""Lightweight telemetry-first component history evaluator."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import pandas as pd

_DEFAULT_MEM_SAFE_LIMIT_MIB = 0.9 * 24576.0


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _series_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return float(data[0])
    rank = max(0.0, min(1.0, float(p))) * (len(data) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(data[lo])
    frac = rank - lo
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(item) for item in values) / max(1, len(values)))


def _to_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    return None


def _first_non_null(row: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _runtime_prediction(row: Mapping[str, Any]) -> float | None:
    return _safe_float(
        _first_non_null(
            row,
            [
                "predicted_runtime_p50_sec",
                "predicted_runtime_p50",
                "predicted_runtime_sec",
                "predicted_runtime",
            ],
        )
    )


def _runtime_prediction_p90(row: Mapping[str, Any]) -> float | None:
    return _safe_float(
        _first_non_null(
            row,
            [
                "predicted_runtime_p90_sec",
                "predicted_runtime_p90",
                "predicted_p90_sec",
                "predicted_runtime_sec",
            ],
        )
    )


def _predicted_peak_mem_p90(row: Mapping[str, Any]) -> float | None:
    return _safe_float(
        _first_non_null(
            row,
            [
                "predicted_peak_mem_p90_mib",
                "predicted_peak_mem_p90",
                "predicted_peak_mem_p90_mib_guarded",
            ],
        )
    )


def _mem_safe_limit_mib(row: Mapping[str, Any]) -> float:
    value = _safe_float(
        _first_non_null(
            row,
            [
                "constraint_mem_safe_limit_mib",
                "mem_safe_limit_mib",
            ],
        )
    )
    if value is None or value <= 0:
        return float(_DEFAULT_MEM_SAFE_LIMIT_MIB)
    return float(value)


def evaluate_component_history(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute telemetry-first component metrics from task observations."""
    if frame.empty:
        return {
            "n_obs_recent": 0,
            "unique_inputs": 0,
            "unique_configs": 0,
            "median_runtime_sec": None,
            "pi_width_rel": None,
            "runtime_ape_p50": None,
            "runtime_ape_p90": None,
            "runtime_bias_sec_median": None,
            "runtime_log_mae": None,
            "mem_fn_rate": None,
            "gap_fit_fp_rate": None,
            "residual_p90": None,
            "residual_p95": None,
            "calibration_error": None,
            "n_runtime_checks": 0,
            "n_mem_checks": 0,
            "n_gap_fit_checks": 0,
            "n_residual_checks": 0,
            "n_calibration_checks": 0,
        }

    runtime = pd.to_numeric(frame.get("runtime_sec"), errors="coerce")
    valid = frame[runtime.notna()].copy()
    valid["runtime_sec"] = runtime[runtime.notna()].astype(float)
    if valid.empty:
        return {
            "n_obs_recent": 0,
            "unique_inputs": 0,
            "unique_configs": 0,
            "median_runtime_sec": None,
            "pi_width_rel": None,
            "runtime_ape_p50": None,
            "runtime_ape_p90": None,
            "runtime_bias_sec_median": None,
            "runtime_log_mae": None,
            "mem_fn_rate": None,
            "gap_fit_fp_rate": None,
            "residual_p90": None,
            "residual_p95": None,
            "calibration_error": None,
            "n_runtime_checks": 0,
            "n_mem_checks": 0,
            "n_gap_fit_checks": 0,
            "n_residual_checks": 0,
            "n_calibration_checks": 0,
        }

    input_series = valid.get("input_fingerprint")
    if input_series is None:
        input_series = valid.get("sample_id", pd.Series([""] * len(valid)))
    input_keys = input_series.fillna("").astype(str).str.strip()
    cfg_keys = valid.get("config_fingerprint", pd.Series([""] * len(valid))).fillna("").astype(str).str.strip()
    n_obs_recent = int(len(valid))
    unique_inputs = int(input_keys[input_keys != ""].nunique())
    unique_configs = int(cfg_keys[cfg_keys != ""].nunique())
    median_runtime = _series_median(valid["runtime_sec"])

    log_runtime = valid["runtime_sec"].apply(lambda x: math.log(max(float(x), 1e-12)))
    std_log = float(log_runtime.std(ddof=0)) if len(log_runtime) >= 2 else 0.0
    pi_width_rel = None
    if median_runtime is not None and median_runtime > 0:
        mu = float(log_runtime.mean())
        low = math.exp(mu - 1.645 * max(std_log, 1e-9))
        high = math.exp(mu + 1.645 * max(std_log, 1e-9))
        pi_width_rel = float((high - low) / max(1e-12, math.exp(mu)))

    runtime_apes: list[float] = []
    runtime_bias_sec: list[float] = []
    residuals: list[float] = []
    mem_checks = 0
    mem_false_negatives = 0
    gap_fit_checks = 0
    gap_fit_false_positives = 0

    for row in valid.to_dict(orient="records"):
        if not isinstance(row, dict):
            continue
        actual_runtime = _safe_float(row.get("runtime_sec"))
        pred_runtime = _runtime_prediction(row)
        if (
            actual_runtime is not None
            and pred_runtime is not None
            and actual_runtime > 0
            and pred_runtime > 0
        ):
            ape = abs(float(pred_runtime) - float(actual_runtime)) / max(1e-9, float(actual_runtime))
            runtime_apes.append(float(ape))
            runtime_bias_sec.append(float(pred_runtime) - float(actual_runtime))
            residual = math.log(max(float(actual_runtime), 1e-12)) - math.log(max(float(pred_runtime), 1e-12))
            residuals.append(float(residual))

        actual_peak_mem = _safe_float(row.get("peak_memory_mib"))
        pred_peak_mem = _predicted_peak_mem_p90(row)
        mem_limit = _mem_safe_limit_mib(row)
        if actual_peak_mem is not None and pred_peak_mem is not None and mem_limit > 0:
            mem_checks += 1
            pred_safe = float(pred_peak_mem) <= float(mem_limit)
            actual_unsafe = float(actual_peak_mem) > float(mem_limit)
            if pred_safe and actual_unsafe:
                mem_false_negatives += 1

        backfill_gap_sec = _safe_float(
            _first_non_null(row, ["constraint_backfill_gap_sec", "backfill_gap_sec"])
        )
        if (
            backfill_gap_sec is not None
            and backfill_gap_sec > 0
            and actual_runtime is not None
            and actual_runtime > 0
        ):
            pred_runtime_p90 = _runtime_prediction_p90(row)
            if pred_runtime_p90 is None or pred_runtime_p90 <= 0:
                continue
            fit_pred = _to_optional_bool(
                _first_non_null(row, ["constraint_backfill_fit_pred", "backfill_fit_pred"])
            )
            if fit_pred is None:
                fit_pred = bool(float(pred_runtime_p90) <= float(backfill_gap_sec))
            gap_fit_checks += 1
            actual_gap_miss = float(actual_runtime) > float(backfill_gap_sec)
            if bool(fit_pred) and actual_gap_miss:
                gap_fit_false_positives += 1

    residual_abs = [abs(float(item)) for item in residuals]
    median_residual = _percentile(residuals, 0.50)

    return {
        "n_obs_recent": n_obs_recent,
        "unique_inputs": unique_inputs,
        "unique_configs": unique_configs,
        "median_runtime_sec": median_runtime,
        "pi_width_rel": pi_width_rel,
        "runtime_ape_p50": _percentile(runtime_apes, 0.50),
        "runtime_ape_p90": _percentile(runtime_apes, 0.90),
        "runtime_bias_sec_median": _percentile(runtime_bias_sec, 0.50),
        "runtime_log_mae": _mean(residual_abs),
        "mem_fn_rate": (
            float(mem_false_negatives / max(1, mem_checks))
            if mem_checks > 0
            else None
        ),
        "gap_fit_fp_rate": (
            float(gap_fit_false_positives / max(1, gap_fit_checks))
            if gap_fit_checks > 0
            else None
        ),
        "residual_p90": _percentile(residual_abs, 0.90),
        "residual_p95": _percentile(residual_abs, 0.95),
        "calibration_error": abs(float(median_residual)) if median_residual is not None else None,
        "n_runtime_checks": int(len(runtime_apes)),
        "n_mem_checks": int(mem_checks),
        "n_gap_fit_checks": int(gap_fit_checks),
        "n_residual_checks": int(len(residual_abs)),
        "n_calibration_checks": int(len(residual_abs)),
    }
