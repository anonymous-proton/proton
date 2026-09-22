"""Stage-unaware campaign HEFT over the current predictive PROTON machinery."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Generator
from typing import Any

from .constraint_tracker import PlanningExhausted
from .contracts import DispatchPlan
from .global_planner import (
    CurrentSolvePlacements,
    DeterministicEFTPlacement,
    DeterministicHEFTPriority,
    GlobalPlanner,
    NoPreInit,
    PointEstimateBackfill,
    PrimaryDeadlineTracker,
)
from .predictive_support import (
    PredictiveTaskSupport,
    primary_incumbent_slowdown_ratio,
)
from .task_heft_priority import task_heft_rank

_LOG = logging.getLogger(__name__)


class HeftCampaignPlanner(PredictiveTaskSupport):
    """Campaign priority, then b-rank, with on-demand predictive placement.

    The delegate owns resource prediction, reciprocal correction, GPU search,
    dispatch ownership and runtime feedback. No phase facade is instantiated.
    """

    _retry_log_tag = "heft-alt-gpu-retry"

    def __init__(
        self,
        *args: Any,
        delegate: Any | None = None,
        reciprocal_interference_correction: bool = True,
        backfill_latency_basis: str = "mean",
        backfill_primary_tail_slo_factor: float | None = 1.0,
        dynamic_batch_cold_policy: str = "largest_safe",
        dynamic_batch_warm_policy: str = "throughput_optimal",
        primary_tail_slo_observer: Any = None,
        primary_tail_slo_observability_path: str = "",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("priority", DeterministicHEFTPriority())
        kwargs.setdefault("placement", DeterministicEFTPlacement())
        kwargs.setdefault("backfill", PointEstimateBackfill())
        kwargs.setdefault("preinit", NoPreInit())
        self._delegate = delegate or GlobalPlanner(*args, **kwargs)
        if isinstance(self._delegate, GlobalPlanner) and not isinstance(
            self._delegate.preinit, NoPreInit
        ):
            raise ValueError("proton_heft requires NoPreInit")
        basis = str(backfill_latency_basis).strip().lower()
        if basis not in {"mean", "ucb"}:
            raise ValueError(f"unsupported backfill_latency_basis: {basis}")
        delegate_basis = getattr(self._delegate, "interference_estimator_basis", None)
        if delegate_basis in {"mean", "ucb"} and basis != delegate_basis:
            raise ValueError("backfill_latency_basis must match placement_strategy")
        self.backfill_latency_basis = basis
        self._delegate.interference_estimator_basis = basis
        self.reciprocal_interference_correction = bool(
            reciprocal_interference_correction
        )
        self._delegate.reciprocal_interference_correction = (
            self.reciprocal_interference_correction
        )
        try:
            factor = (
                None
                if backfill_primary_tail_slo_factor is None
                else float(backfill_primary_tail_slo_factor)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "backfill_primary_tail_slo_factor must be >= 1.0"
            ) from error
        self.backfill_primary_tail_slo_factor = factor
        if factor is not None:
            if not math.isfinite(factor) or factor < 1.0:
                raise ValueError("backfill_primary_tail_slo_factor must be >= 1.0")
            if not self.reciprocal_interference_correction:
                raise ValueError(
                    "backfill_primary_tail_slo_factor requires reciprocal_interference_correction"
                )
        if dynamic_batch_cold_policy not in {
            "constant_memory_linear_latency",
            "fixed_n",
            "largest_safe",
        }:
            raise ValueError(
                f"unsupported dynamic batch cold policy: {dynamic_batch_cold_policy}"
            )
        if dynamic_batch_warm_policy not in {
            "fixed_n",
            "largest_safe",
            "throughput_optimal",
        }:
            raise ValueError(
                f"unsupported dynamic batch warm policy: {dynamic_batch_warm_policy}"
            )
        self.dynamic_batch_cold_policy = dynamic_batch_cold_policy
        self.dynamic_batch_warm_policy = dynamic_batch_warm_policy
        self._dynamic_batch_profiles: dict = {}
        self._dynamic_batch_blocked: set = set()
        self._dynamic_batch_metadata: dict = {}
        self._dynamic_batch_fallbacks: dict = {}
        self._alternate_gpu_retry_stats = {"attempts": 0, "successes": 0, "failures": 0}
        self._alternate_gpu_retry_rescued_task_ids: set[str] = set()
        self._primary_tail_slo_rejections: set[str] = set()
        self._primary_tail_slo_observer = (
            primary_tail_slo_observer if callable(primary_tail_slo_observer) else None
        )
        if primary_tail_slo_observability_path:

            def write_record(record: dict[str, Any]) -> None:
                try:
                    with open(
                        primary_tail_slo_observability_path, "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                except OSError:
                    _LOG.warning(
                        "Unable to write primary-tail SLO trace", exc_info=True
                    )

            self._primary_tail_slo_observer = write_record
        self.campaign_scheduler._pre_init_disabled = True
        self.campaign_scheduler._downstream_projection_disabled = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def campaign_scheduler(self) -> Any:
        return self._delegate.campaign_scheduler

    @property
    def timelines(self) -> Any:
        return self._delegate.timelines

    def _ordered_tasks(
        self, tasks: list[dict[str, Any]], primary_id: str | None
    ) -> list[dict[str, Any]]:
        campaign_order = self._ordered_campaign_ids(tasks)
        ranks = {campaign: rank for rank, campaign in enumerate(campaign_order)}
        self._delegate.priority.invalidate()
        ranked = []
        for original in tasks:
            task = dict(original)
            task["component"] = str(task.get("component", "") or "").strip().lower()
            task["campaign_id"] = str(task.get("campaign_id", "") or "")
            task["task_id"] = str(task.get("task_id", "") or "")
            task["campaign_rank"] = ranks.get(task["campaign_id"], len(campaign_order))
            rank = task_heft_rank(task, self._delegate, self._delegate.priority)
            task["b_rank"] = rank.total
            task["own_cost"] = rank.own_cost
            task["downstream_cost"] = rank.downstream_cost
            ranked.append(task)
        return sorted(
            ranked,
            key=lambda task: (
                task["campaign_id"] != primary_id,
                task["campaign_rank"],
                -task["b_rank"],
                task["task_id"],
            ),
        )

    def _plan_ready_task(
        self,
        task: dict[str, Any],
        primary_id: str | None,
        primary_budget: PrimaryDeadlineTracker,
        current_solve: CurrentSolvePlacements,
    ) -> DispatchPlan:
        task_id, campaign_id, component = (
            task["task_id"],
            task["campaign_id"],
            task["component"],
        )
        if not task_id or not campaign_id or not component:
            raise PlanningExhausted("HEFT planner task invariant missing")
        is_backfill = campaign_id != primary_id
        evaluations: list[dict[str, Any]] = []
        pls_records: dict[str, dict[str, Any]] = {}

        def candidate_filter(
            gpu_id: str, eft: float, projection: Any, _option: Any
        ) -> bool:
            reason = ""
            factor = self.backfill_primary_tail_slo_factor
            if factor is not None:
                ratio = (
                    primary_incumbent_slowdown_ratio(
                        self.timelines, primary_id or "", projection
                    )
                    if projection is not None
                    else math.inf
                )
                headroom = factor - ratio
                allowed = headroom >= 0.0
                record = {
                    "task_id": task_id,
                    "candidate_gpu": str(gpu_id),
                    "epoch": primary_id,
                    "factor": factor,
                    "projected": ratio,
                    "headroom": headroom,
                    "allowed": allowed,
                    "primary_deadline": primary_budget.current(),
                }
                pls_records[str(gpu_id)] = record
                if not allowed:
                    reason = "primary_tail_slo"
                    self._primary_tail_slo_rejections.add(task_id)
                if self._primary_tail_slo_observer is not None:
                    self._primary_tail_slo_observer(record)
            if not reason and not self._candidate_worker_front_has_capacity(
                component, gpu_id
            ):
                reason = "worker_front"
            if not reason and self._force_single_self_concurrency_blocks_gpu(
                component, gpu_id, task_id
            ):
                reason = "force_single"
            if not reason and not eft <= primary_budget.current():
                reason = "primary_deadline"
            evaluations.append(
                {
                    "gpu_id": str(gpu_id),
                    "eft": eft,
                    "allowed": not reason,
                    "reason": reason or "admitted",
                }
            )
            return not reason

        try:
            plan = self._delegate._place_single(
                task_id,
                campaign_id,
                component,
                self._as_float(task.get("input_size"), default=0.0),
                is_backfill,
                primary_budget,
                self._gpu_ids(),
                current_solve,
                config_fingerprint=str(task.get("config_fingerprint", "") or ""),
                input_fingerprint=str(task.get("input_fingerprint", "") or ""),
                candidate_filter=candidate_filter if is_backfill else None,
            )
        except PlanningExhausted:
            self._record_alternate_gpu_retry(task, evaluations, selected_gpu=None)
            raise
        if is_backfill and not self._worker_front_has_capacity(
            str(plan.target_worker_name or "")
        ):
            self.timelines.remove_predicted_entries_for_task(task_id)
            raise PlanningExhausted("HEFT worker front saturated")
        self._attach_dynamic_batch_metadata(plan)
        metadata = {
            "campaign_id": campaign_id,
            "primary_campaign_id": primary_id,
            "campaign_rank": task["campaign_rank"],
            "b_rank": task["b_rank"],
            "own_cost": task["own_cost"],
            "downstream_cost": task["downstream_cost"],
            "input_fingerprint": str(task.get("input_fingerprint", "") or ""),
            "config_fingerprint": str(task.get("config_fingerprint", "") or ""),
            "is_backfill": is_backfill,
            "ordering": "campaign_heft",
            "runtime_prediction_basis": self.backfill_latency_basis,
            "primary_deadline": primary_budget.current(),
            "backfill_admission": "primary_deadline" if is_backfill else "primary",
        }
        retry = self._record_alternate_gpu_retry(
            task, evaluations, selected_gpu=str(plan.target_gpu_id)
        )
        if retry is not None:
            metadata["alternate_gpu_retry"] = retry
        if str(plan.target_gpu_id) in pls_records:
            metadata["primary_tail_slo"] = pls_records[str(plan.target_gpu_id)]
            plan.worker_metadata["primary_tail_slo"] = dict(
                pls_records[str(plan.target_gpu_id)]
            )
        plan.worker_metadata["heft_scheduler"] = metadata
        _LOG.info(
            "[heft-backfill] task=%s campaign=%s decision=%s gpu=%s",
            task_id,
            campaign_id,
            "accept" if is_backfill else "primary",
            plan.target_gpu_id,
        )
        return plan

    def _solve_steps(
        self, tasks: list[dict[str, Any]], *, commit_predictions: bool
    ) -> Generator[
        tuple[str, DispatchPlan | None, PlanningExhausted | None], None, None
    ]:
        task_ids = [str(task.get("task_id", "") or "") for task in tasks]
        completed = False
        try:
            self._delegate._candidate_eft_details.clear()
            self._prepare_legacy_dynamic_batch_profiles(tasks)
            self._delegate._self_concurrency_best_n_cache.clear()
            self._delegate._reciprocal_query_results.clear()
            self.timelines.gc_orphaned_predicted_entries()
            primary_id = self._primary_campaign_id(tasks)
            self._delegate._cached_primary_id = primary_id
            self._delegate._cached_primary_id_set = True
            budget = self._delegate._build_primary_deadline_tracker(
                self._campaign_queues().get(primary_id or ""), self._gpu_ids()
            )
            self._primary_tail_slo_rejections.clear()
            current_solve = CurrentSolvePlacements.for_task_ids(task_ids)
            for task in self._ordered_tasks(tasks, primary_id):
                current_solve.begin_task(task["task_id"])
                try:
                    plan = self._plan_ready_task(
                        task, primary_id, budget, current_solve
                    )
                except PlanningExhausted as error:
                    yield task["task_id"], None, error
                else:
                    yield task["task_id"], plan, None
            completed = True
        finally:
            if not completed or not commit_predictions:
                self._delegate._detach_batch_solve_predictions(task_ids)
            self._delegate._cached_primary_id = None
            self._delegate._cached_primary_id_set = False
            self._delegate._self_concurrency_best_n_cache.clear()
            self._delegate._candidate_eft_details.clear()
            self._delegate._reciprocal_query_results.clear()
            self._clear_legacy_dynamic_batch_profiles()

    async def solve_admissible(
        self, pending_tasks: list[dict[str, Any]], *, commit_predictions: bool = True
    ) -> tuple[list[DispatchPlan], dict[str, PlanningExhausted]]:
        plans: list[DispatchPlan] = []
        skipped: dict[str, PlanningExhausted] = {}
        steps = self._solve_steps(pending_tasks, commit_predictions=commit_predictions)
        try:
            while True:
                await asyncio.sleep(0)
                try:
                    task_id, plan, error = next(steps)
                except StopIteration:
                    return plans, skipped
                if error is not None:
                    skipped[task_id] = error
                elif plan is not None:
                    plans.append(plan)
        finally:
            steps.close()

    def solve(self, pending_tasks: list[dict[str, Any]]) -> list[DispatchPlan]:
        plans = []
        steps = self._solve_steps(pending_tasks, commit_predictions=True)
        try:
            for _task_id, plan, error in steps:
                if error is not None:
                    raise error
                if plan is not None:
                    plans.append(plan)
            return plans
        finally:
            steps.close()

    def plan(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float = 0.0,
        is_backfill: bool = False,
        config_fingerprint: str = "",
        input_fingerprint: str = "",
        logical_batch_size: int = 0,
        execution_overrides: dict[str, Any] | None = None,
    ) -> DispatchPlan:
        del is_backfill
        return self.solve(
            [
                {
                    "task_id": task_id,
                    "campaign_id": campaign_id,
                    "component": component,
                    "input_size": input_size,
                    "config_fingerprint": config_fingerprint,
                    "input_fingerprint": input_fingerprint,
                    "logical_batch_size": logical_batch_size,
                    "execution_overrides": dict(execution_overrides or {}),
                }
            ]
        )[0]

    def commit_dispatch_plan_prediction(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate.commit_dispatch_plan_prediction(*args, **kwargs)
