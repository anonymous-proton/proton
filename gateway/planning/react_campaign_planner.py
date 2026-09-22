"""Campaign HEFT ordering restricted to a fixed current-state start.

proton-react is proton-heft with speculative future scheduling removed:

* the placement strategy evaluates a fixed NOW start only (no sliding
  future-fit search, no insertion into future gaps between predicted
  entries);
* the campaign scheduler's speculative pending-task projection
  (``_project_task`` / ``_reproject_pending``) is disabled, so unaccepted
  tasks never create or reserve future timeline state.

Every other behaviour — campaign FIFO, task-specific HEFT b-rank, backfill
policy, PLS, primary-deadline protection, resource/concurrency/front
checks, interference and reciprocal correction, dynamic batching, GPU
tie-breaking, eviction and cancellation, cold-worker lifecycle, runtime
admission, dispatch ownership, wake/retry cadence, validation and
validation memoisation — is the shared proton-heft implementation.
"""

from __future__ import annotations

from typing import Any

from .global_planner import (
    DeterministicEFTPlacement,
    GlobalPlanner,
    NoPreInit,
    PointEstimateBackfill,
)
from .heft_campaign_planner import HeftCampaignPlanner


class ReactiveCurrentPlacement(DeterministicEFTPlacement):
    """μ-only EFT placement that never searches a later start.

    The shared ``evaluate_gpu`` honours ``requires_fixed_now_start`` by
    fitting the candidate at the solve's current wall clock only; a busy
    or resource-blocked GPU yields no candidate instead of a future one.
    """

    requires_fixed_now_start = True


class ReactCampaignPlanner(HeftCampaignPlanner):
    """proton-heft semantics minus speculative future scheduling."""

    current_state_only = True

    def __init__(self, *args: Any, delegate: Any | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("placement", ReactiveCurrentPlacement())
        if kwargs.get("backfill_latency_basis", "mean") != "mean":
            raise ValueError("proton_react requires mean estimators")
        if not kwargs.get("reciprocal_interference_correction", True):
            raise ValueError("proton_react requires reciprocal correction")
        super().__init__(*args, delegate=delegate, **kwargs)
        if not isinstance(self._delegate, GlobalPlanner) or not (
            isinstance(self._delegate.placement, ReactiveCurrentPlacement)
            and isinstance(self._delegate.backfill, PointEstimateBackfill)
            and isinstance(self._delegate.preinit, NoPreInit)
        ):
            raise ValueError(
                "proton_react requires reactive_current/point backfill/NoPreInit"
            )
        self.campaign_scheduler._speculative_task_projection_disabled = True
