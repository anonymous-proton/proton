"""GPObservation — structured GP-update channel (Plan Blocker 4).

Plan .

ConstraintViolation covers *failure-side* information.  Successful task
completions — together with their measured VRAM / duration — also feed
GP posteriors, so the signal service converges on real-workload
statistics rather than only learning from failures.  This module
provides the common dataclass and a thin emission helper that
Dispatcher / on_task_complete / on_worker_killed call uniformly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

_LOG = logging.getLogger(__name__)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid observation value: {value!r}") from exc


@dataclass
class GPObservation:
    """Observation record for GP posterior update.

    ``terminal_state`` ∈ {"completed", "failed", "cancelled"}.  Partial
    observations (failed/cancelled runs with meaningful telemetry) are
    still useful for GP training, so emission is unconditional on
    success — the SignalService filters on ``terminal_state`` if
    needed.
    """

    component: str
    gpu_id: str
    input_size: int = 0
    predicted_vram_mb: float = 0.0
    actual_vram_mb: float = 0.0
    predicted_duration_sec: float = 0.0
    actual_duration_sec: float = 0.0
    concurrent_components: list[str] = field(default_factory=list)
    solo_fraction: float = 1.0
    succeeded: bool = True
    terminal_state: str = "completed"
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dispatch_result(
        cls,
        *,
        component: str,
        gpu_id: str,
        actual_duration_sec: float,
        actual_vram_mb: float = 0.0,
        predicted_vram_mb: float = 0.0,
        predicted_duration_sec: float = 0.0,
        concurrent_components: list[str] | None = None,
        solo_fraction: float = 1.0,
        succeeded: bool = True,
    ) -> GPObservation:
        return cls(
            component=component,
            gpu_id=str(gpu_id),
            actual_vram_mb=_as_float(actual_vram_mb),
            predicted_vram_mb=_as_float(predicted_vram_mb),
            actual_duration_sec=_as_float(actual_duration_sec),
            predicted_duration_sec=_as_float(predicted_duration_sec),
            concurrent_components=list(concurrent_components or []),
            solo_fraction=_as_float(solo_fraction),
            succeeded=bool(succeeded),
            terminal_state="completed" if succeeded else "failed",
        )


def emit_observation(signal_service: Any, observation: GPObservation) -> None:
    """Route through the one supported ingest API and surface failures."""
    record = getattr(signal_service, "record_observation", None)
    if not callable(record):
        _LOG.error("[gp-observation] SignalService.record_observation is unavailable")
        raise AttributeError("SignalService.record_observation is required")
    try:
        record(observation)
    except Exception:
        _LOG.error("[gp-observation] ingest failed", exc_info=True)
        raise
