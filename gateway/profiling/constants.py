"""Shared telemetry-first filtering and model-view policy constants."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

MODEL_INPUT_STATE_SUCCEEDED = "SUCCEEDED"
MODEL_INPUT_RUN_SOURCES: Tuple[str, ...] = ("task",)
MODEL_INPUT_FILTER_SQL = "state='SUCCEEDED' AND qc_keep=1 AND run_source IN ('task')"
_MODEL_INPUT_RUN_SOURCE_SET = set(MODEL_INPUT_RUN_SOURCES)
PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_NAME = "pin_active_until_gate_pass"
PIN_ACTIVE_UNTIL_GATE_PASS_POLICY_DESCRIPTION = (
    "Latest may remain BLOCKED while active stays pinned; "
    "effective selects active until promotion gate passes."
)


def normalize_model_input_run_sources(run_sources: Sequence[str] | None = None) -> Tuple[str, ...]:
    if run_sources is None:
        return MODEL_INPUT_RUN_SOURCES
    normalized = [
        str(item).strip().lower()
        for item in list(run_sources or [])
        if str(item).strip()
    ]
    if not normalized:
        return MODEL_INPUT_RUN_SOURCES
    invalid = sorted({item for item in normalized if item not in _MODEL_INPUT_RUN_SOURCE_SET})
    if invalid:
        allowed = ", ".join(MODEL_INPUT_RUN_SOURCES)
        bad = ", ".join(invalid)
        raise ValueError(
            f"run_sources must be a subset of [{allowed}] for MODEL_INPUT_FILTER; got [{bad}]"
        )
    return tuple(dict.fromkeys(normalized))


def matches_succeeded_kept(*, state: Any, qc_keep: Any) -> bool:
    return str(state or "").strip().upper() == MODEL_INPUT_STATE_SUCCEEDED and bool(qc_keep)


def matches_model_input_filter(*, state: Any, qc_keep: Any, run_source: Any) -> bool:
    return (
        matches_succeeded_kept(state=state, qc_keep=qc_keep)
        and str(run_source or "").strip().lower() in _MODEL_INPUT_RUN_SOURCE_SET
    )
