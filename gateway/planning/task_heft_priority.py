"""Task-specific HEFT priority using the shared predictor and existing DAG rank.

Only the current node uses actual task inputs. Unknown successor instances use
DeterministicHEFTPriority's existing campaign/component expectation. No placement,
interference, residency or phase decision belongs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .global_planner import GlobalPlanner, HEFTPriority
from .predictive_support import _safe_float


@dataclass(frozen=True)
class TaskHEFTRank:
    own_cost: float
    downstream_cost: float

    @property
    def total(self) -> float:
        return self.own_cost + self.downstream_cost


def task_own_cost(task: Mapping[str, Any], planner: GlobalPlanner) -> float:
    """Current intrinsic latency, at HEFT's existing pooled GPU scope.

    Input size/config are already normalized by the common gateway path. An
    input fingerprint is identity, not an additional scalar-GP feature. Do not
    substitute the campaign's average input or cache by component here.
    """
    campaign_id = str(task.get("campaign_id", "") or "")
    component = str(task.get("component", "") or "").strip().lower()
    config = str(task.get("config_fingerprint", "") or "").strip()
    if not config:
        config = planner._campaign_component_fingerprint(campaign_id, component) or ""
    scheduler = planner.campaign_scheduler
    latency = scheduler._predict_latency(
        component,
        _safe_float(task.get("input_size")),
        "",
        config_fingerprint=config,
    )
    if latency is None or latency <= 0.0:
        latency = scheduler._cold_inference_latency_sec()
    return latency


def task_heft_rank(
    task: Mapping[str, Any],
    planner: GlobalPlanner,
    downstream_priority: HEFTPriority,
) -> TaskHEFTRank:
    """Own task prediction plus the existing maximum successor b-rank.

    The caller invalidates the downstream estimator once per solve. The current
    DAG has component edges, not exact task-instance successor associations.
    Missing topology means zero downstream contribution, not zero own cost.
    """
    campaign_id = str(task.get("campaign_id", "") or "")
    component = str(task.get("component", "") or "").strip().lower()
    queue = planner.campaign_scheduler._campaign_queues.get(campaign_id)
    dag = getattr(queue, "dag_context", None)
    successors = getattr(dag, "downstream_map", {}).get(component, ())
    downstream = max(
        (
            downstream_priority.compute_rank(successor, campaign_id, planner)
            for successor in successors
        ),
        default=0.0,
    )
    return TaskHEFTRank(task_own_cost(task, planner), downstream)
