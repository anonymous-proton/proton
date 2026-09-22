"""Confidence scoring helpers for telemetry-first task runs."""

from __future__ import annotations

from typing import Any, Dict


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_lower_is_better(value: float | None, bad: float) -> float | None:
    if value is None:
        return None
    return _clamp01(1.0 - (float(value) / max(1e-9, float(bad))))


def _quality_score(
    *,
    runtime_ape_p90: float | None,
    mem_fn_rate: float | None,
    gap_fit_fp_rate: float | None,
    residual_p95: float | None,
    n_runtime_checks: int,
    n_mem_checks: int,
    n_gap_fit_checks: int,
    n_residual_checks: int,
    runtime_ape_bad: float,
    mem_fn_bad: float,
    gap_fit_fp_bad: float,
    residual_bad: float,
) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    runtime_score = _score_lower_is_better(runtime_ape_p90, runtime_ape_bad)
    mem_score = _score_lower_is_better(mem_fn_rate, mem_fn_bad)
    gap_score = _score_lower_is_better(gap_fit_fp_rate, gap_fit_fp_bad)
    residual_score = _score_lower_is_better(residual_p95, residual_bad)

    if runtime_score is not None and int(n_runtime_checks) > 0:
        weighted_sum += 0.35 * float(runtime_score)
        weight_total += 0.35
    if mem_score is not None and int(n_mem_checks) > 0:
        weighted_sum += 0.35 * float(mem_score)
        weight_total += 0.35
    if gap_score is not None and int(n_gap_fit_checks) > 0:
        weighted_sum += 0.20 * float(gap_score)
        weight_total += 0.20
    if residual_score is not None and int(n_residual_checks) > 0:
        weighted_sum += 0.10 * float(residual_score)
        weight_total += 0.10
    if weight_total <= 0:
        return 0.0
    return float(_clamp01(weighted_sum / weight_total))


def compute_confidence(
    *,
    n_obs_recent: int,
    unique_configs: int,
    unique_inputs: int,
    runtime_ape_p90: float | None,
    mem_fn_rate: float | None,
    gap_fit_fp_rate: float | None,
    residual_p95: float | None,
    contam_rate: float | None,
    staleness_days: float | None,
    bimodal_flag_rate: float | None,
    n_runtime_checks: int = 0,
    n_mem_checks: int = 0,
    n_gap_fit_checks: int = 0,
    n_residual_checks: int = 0,
    n_target: int = 200,
    config_target: int = 20,
    input_target: int = 20,
    runtime_ape_bad: float = 0.50,
    mem_fn_bad: float = 0.05,
    gap_fit_fp_bad: float = 0.10,
    residual_bad: float = 1.00,
    contam_bad: float = 0.20,
    stale_bad: float = 14.0,
) -> Dict[str, float]:
    coverage = (
        0.5 * _clamp01(float(n_obs_recent) / max(1.0, float(n_target)))
        + 0.25 * _clamp01(float(unique_configs) / max(1.0, float(config_target)))
        + 0.25 * _clamp01(float(unique_inputs) / max(1.0, float(input_target)))
    )
    quality = _quality_score(
        runtime_ape_p90=runtime_ape_p90,
        mem_fn_rate=mem_fn_rate,
        gap_fit_fp_rate=gap_fit_fp_rate,
        residual_p95=residual_p95,
        n_runtime_checks=n_runtime_checks,
        n_mem_checks=n_mem_checks,
        n_gap_fit_checks=n_gap_fit_checks,
        n_residual_checks=n_residual_checks,
        runtime_ape_bad=runtime_ape_bad,
        mem_fn_bad=mem_fn_bad,
        gap_fit_fp_bad=gap_fit_fp_bad,
        residual_bad=residual_bad,
    )
    contam_score = _clamp01(1.0 - (float(contam_rate) / max(1e-9, float(contam_bad)))) if contam_rate is not None else 1.0
    stale_score = _clamp01(1.0 - (float(staleness_days) / max(1e-9, float(stale_bad)))) if staleness_days is not None else 1.0
    bimodal_score = _clamp01(1.0 - float(bimodal_flag_rate or 0.0))
    health = contam_score * stale_score * bimodal_score
    confidence = (0.45 * coverage) + (0.35 * quality) + (0.20 * health)
    return {
        "coverage_score": float(coverage),
        "quality_score": float(quality),
        "health_score": float(health),
        "confidence": float(_clamp01(confidence)),
    }
