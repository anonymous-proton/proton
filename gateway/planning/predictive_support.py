"""Stage-independent predictive helpers shared by planner facades.

Facades initialize their own state; this mixin does not own scheduling policy.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from ..extraction.component_features import get_dynamic_batching
from .contracts import DispatchPlan
from .global_planner import HEFTPriority

_LOG = logging.getLogger("gateway.planning.phase_campaign_planner")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DynamicBatchCandidate:
    k: int
    logical_n: int
    latency_mean: float
    latency_sigma: float
    latency_upper: float
    vram_mean: float
    vram_upper: float
    ram_mean: float
    ram_upper: float
    supported: bool
    fits: bool
    source: str
    upper_pressure: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_batch_size": self.k,
            "logical_batch_size": self.logical_n,
            "latency_mean": self.latency_mean,
            "latency_sigma": self.latency_sigma,
            "latency_upper": self.latency_upper,
            "vram_mean": self.vram_mean,
            "vram_upper": self.vram_upper,
            "ram_mean": self.ram_mean,
            "ram_upper": self.ram_upper,
            "supported": self.supported,
            "fits": self.fits,
            "source": self.source,
            "upper_pressure": self.upper_pressure,
        }


def _fixed_n(
    candidates: tuple[DynamicBatchCandidate, ...],
    logical_n: int,
) -> DynamicBatchCandidate | None:
    return next(
        (item for item in candidates if item.k == logical_n and item.supported),
        None,
    )


def _largest_safe(
    candidates: tuple[DynamicBatchCandidate, ...],
    _logical_n: int,
) -> DynamicBatchCandidate | None:
    feasible = [item for item in candidates if item.supported and item.fits]
    return max(feasible, key=lambda item: item.k, default=None)


def _throughput_optimal(
    candidates: tuple[DynamicBatchCandidate, ...],
    logical_n: int,
) -> DynamicBatchCandidate | None:
    feasible = [
        item
        for item in candidates
        if item.supported and item.fits and item.latency_mean > 0
    ]
    return max(
        feasible,
        key=lambda item: (
            logical_n / item.latency_mean,
            item.k,
        ),
        default=None,
    )


_DYNAMIC_BATCH_SELECTORS = {
    "fixed_n": _fixed_n,
    "largest_safe": _largest_safe,
    "throughput_optimal": _throughput_optimal,
}


def primary_incumbent_slowdown_ratio(
    timelines: Any,
    primary_campaign_id: str,
    projection: Any,
) -> float:
    """Return the legacy task-level PLS ratio, excluding tail time and queueing."""
    slowdowns = []
    projected_latencies = dict(getattr(projection, "incumbent_projected_latencies", ()))
    for incumbent_id, projected_latency in projected_latencies.items():
        entry = timelines.find_entry(str(incumbent_id))
        if (
            entry is None
            or str(getattr(entry, "campaign_id", "")) != primary_campaign_id
        ):
            continue
        solo = _safe_float(getattr(entry, "reciprocal_base_duration_sec", 0.0))
        if solo > 0.0:
            slowdowns.append(max(0.0, _safe_float(projected_latency)) / solo)
    return max(slowdowns, default=1.0)


class PredictiveTaskSupport:
    """Mechanical shared helpers; state remains initialized by each facade."""

    _retry_log_tag = "phase-alt-gpu-retry"

    _delegate: Any
    campaign_scheduler: Any
    timelines: Any
    dynamic_batch_cold_policy: str
    dynamic_batch_warm_policy: str
    backfill_latency_basis: str
    _dynamic_batch_profiles: dict[tuple[str, str], DynamicBatchCandidate]
    _dynamic_batch_blocked: set[tuple[str, str]]
    _dynamic_batch_metadata: dict[tuple[str, str], dict[str, Any]]
    _dynamic_batch_fallbacks: dict[str, dict[str, Any]]
    _alternate_gpu_retry_stats: dict[str, int]
    _alternate_gpu_retry_rescued_task_ids: set[str]

    @staticmethod
    def _get_dynamic_batching(component: str) -> dict[str, Any]:
        return get_dynamic_batching(component)

    @staticmethod
    def _batch_gp_mature(gp: Any, observations_per_dim: int) -> bool:
        return bool(
            gp is not None
            and _safe_int(getattr(gp, "input_dim", 0)) == 2
            and _safe_int(getattr(gp, "n", 0)) >= max(1, observations_per_dim) * 2
            and gp.has_full_rank_support()
        )

    def _batch_gp(
        self,
        registry: Any,
        component: str,
        config_fingerprint: str,
        metric: str,
        gpu_id: str | None,
        observations_per_dim: int,
    ) -> Any | None:
        gp = registry.get_batch_gp(
            component,
            config_fingerprint,
            metric,
            gpu_id=gpu_id,
        )
        if self._batch_gp_mature(gp, observations_per_dim):
            return gp
        if gpu_id is not None:
            pooled = registry.get_batch_gp(
                component,
                config_fingerprint,
                metric,
                gpu_id=None,
            )
            if self._batch_gp_mature(pooled, observations_per_dim):
                return pooled
        return None

    def _cold_batch_candidates(
        self,
        *,
        component: str,
        input_size: float,
        config_fingerprint: str,
        gpu_id: str,
        logical_n: int,
        available_vram: float,
        available_ram: float,
    ) -> tuple[DynamicBatchCandidate, ...]:
        cs = self.campaign_scheduler
        latency_mean = _safe_float(
            cs._predict_latency(
                component,
                input_size,
                gpu_id,
                config_fingerprint=config_fingerprint,
            )
        )
        latency_sigma = _safe_float(
            cs._predict_latency_sigma(
                component,
                input_size,
                gpu_id,
                config_fingerprint=config_fingerprint,
            )
        )
        vram_mean = _safe_float(
            cs._predict_vram(
                component,
                input_size,
                gpu_id,
                use_upper=False,
                config_fingerprint=config_fingerprint,
            )
        )
        vram_upper = _safe_float(
            cs._predict_vram(
                component,
                input_size,
                gpu_id,
                use_upper=True,
                config_fingerprint=config_fingerprint,
            )
        )
        ram_mean = _safe_float(
            cs._predict_ram(
                component,
                input_size,
                gpu_id,
                use_upper=False,
                config_fingerprint=config_fingerprint,
            )
        )
        ram_upper = _safe_float(
            cs._predict_ram(
                component,
                input_size,
                gpu_id,
                use_upper=True,
                config_fingerprint=config_fingerprint,
            )
        )
        if (
            not all(
                math.isfinite(value)
                for value in (
                    latency_mean,
                    latency_sigma,
                    vram_mean,
                    vram_upper,
                    ram_mean,
                    ram_upper,
                )
            )
            or latency_mean <= 0
            or latency_sigma < 0
            or vram_mean <= 0
            or vram_upper <= 0
            or ram_mean <= 0
            or ram_upper <= 0
        ):
            return ()
        alpha_z = self._latency_uncertainty_z(
            component,
            input_size,
            config_fingerprint,
        )
        candidates: list[DynamicBatchCandidate] = []
        constant_memory = (
            self.dynamic_batch_cold_policy == "constant_memory_linear_latency"
        )
        for k in range(1, logical_n + 1):
            if constant_memory:
                group_sizes = tuple(
                    min(k, logical_n - start) for start in range(0, logical_n, k)
                )
                duration_scale = sum(group_sizes) / logical_n
                resource_scale = 1.0
                source = "constant_memory_linear_latency"
            else:
                duration_scale = math.ceil(logical_n / k)
                resource_scale = k / logical_n
                source = "scalar_scaled"
            scaled_vram_upper = vram_upper * resource_scale
            scaled_ram_upper = ram_upper * resource_scale
            candidates.append(
                DynamicBatchCandidate(
                    k=k,
                    logical_n=logical_n,
                    latency_mean=latency_mean * duration_scale,
                    latency_sigma=latency_sigma * duration_scale,
                    latency_upper=(latency_mean + alpha_z * latency_sigma)
                    * duration_scale,
                    vram_mean=vram_mean * resource_scale,
                    vram_upper=scaled_vram_upper,
                    ram_mean=ram_mean * resource_scale,
                    ram_upper=scaled_ram_upper,
                    supported=True,
                    fits=(
                        scaled_vram_upper <= available_vram
                        and scaled_ram_upper <= available_ram
                    ),
                    source=source,
                    upper_pressure=max(
                        scaled_vram_upper / max(available_vram, 1e-9),
                        scaled_ram_upper / max(available_ram, 1e-9),
                    ),
                )
            )
        return tuple(candidates)

    def _warm_batch_candidates(
        self,
        *,
        component: str,
        input_size: float,
        config_fingerprint: str,
        gpu_id: str,
        logical_n: int,
        available_vram: float,
        available_ram: float,
    ) -> tuple[DynamicBatchCandidate, ...]:
        cs = self.campaign_scheduler
        service = getattr(cs, "_signal_service", None)
        registry = getattr(service, "resource_profiles", None)
        if registry is None or input_size <= 0:
            return ()
        supervisor = getattr(cs, "_supervisor", None)
        observations_per_dim = max(
            1,
            _safe_int(getattr(supervisor, "gp_maturity_min_observations", 10), 10),
        )
        latency_gp = self._batch_gp(
            registry,
            component,
            config_fingerprint,
            "latency",
            gpu_id,
            observations_per_dim,
        )
        vram_gp = self._batch_gp(
            registry,
            component,
            config_fingerprint,
            "vram",
            None,
            observations_per_dim,
        )
        ram_gp = self._batch_gp(
            registry,
            component,
            config_fingerprint,
            "ram",
            None,
            observations_per_dim,
        )
        if latency_gp is None or vram_gp is None or ram_gp is None:
            return ()
        gps = (latency_gp, vram_gp, ram_gp)
        points = [(input_size, k) for k in range(1, logical_n + 1)]
        latency_means, latency_variances = latency_gp.predict_grid(points)
        vram_means, vram_variances = vram_gp.predict_grid(points)
        ram_means, ram_variances = ram_gp.predict_grid(points)
        latency_z = self._latency_uncertainty_z(
            component,
            input_size,
            config_fingerprint,
        )
        resource_z = _safe_float(getattr(cs, "resource_upper_z", 1.96), 1.96)
        candidates: list[DynamicBatchCandidate] = []
        for index, (_x, k_value) in enumerate(points):
            latency_mean = latency_means[index]
            latency_variance = latency_variances[index]
            vram_mean = vram_means[index]
            vram_variance = vram_variances[index]
            ram_mean = ram_means[index]
            ram_variance = ram_variances[index]
            finite = all(
                math.isfinite(value)
                for value in (
                    latency_mean,
                    latency_variance,
                    vram_mean,
                    vram_variance,
                    ram_mean,
                    ram_variance,
                )
            )
            nonnegative = all(
                value >= 0
                for value in (
                    latency_mean,
                    latency_variance,
                    vram_mean,
                    vram_variance,
                    ram_mean,
                    ram_variance,
                )
            )
            supported = (
                finite
                and nonnegative
                and all(gp.is_sufficient((input_size, k_value), 0.2) for gp in gps)
            )
            latency_sigma = math.sqrt(max(0.0, latency_variance))
            vram_sigma = math.sqrt(max(0.0, vram_variance))
            ram_sigma = math.sqrt(max(0.0, ram_variance))
            latency_upper = max(
                0.0,
                latency_mean + latency_z * latency_sigma,
            )
            vram_upper = max(0.0, vram_mean + resource_z * vram_sigma)
            ram_upper = max(0.0, ram_mean + resource_z * ram_sigma)
            supported = supported and all(
                math.isfinite(value) for value in (latency_upper, vram_upper, ram_upper)
            )
            candidates.append(
                DynamicBatchCandidate(
                    k=_safe_int(k_value),
                    logical_n=logical_n,
                    latency_mean=max(0.0, latency_mean),
                    latency_sigma=max(0.0, latency_sigma),
                    latency_upper=latency_upper,
                    vram_mean=max(0.0, vram_mean),
                    vram_upper=vram_upper,
                    ram_mean=max(0.0, ram_mean),
                    ram_upper=ram_upper,
                    supported=supported,
                    fits=(
                        supported
                        and vram_upper <= available_vram
                        and ram_upper <= available_ram
                    ),
                    source="gp_2d",
                    upper_pressure=max(
                        vram_upper / max(available_vram, 1e-9),
                        ram_upper / max(available_ram, 1e-9),
                    ),
                )
            )
        return tuple(candidates) if any(item.supported for item in candidates) else ()

    def _prepare_legacy_dynamic_batch_profiles(
        self,
        pending_tasks: list[dict[str, Any]],
    ) -> None:
        """Select one pre-placement batch profile per task/GPU.

        This preserves the pre-persistent-actor decision boundary: telemetry
        can update scalar predictions, but it cannot expand placement into a
        task/GPU/batch-option joint search.
        """
        self._dynamic_batch_profiles.clear()
        self._dynamic_batch_blocked.clear()
        self._dynamic_batch_metadata.clear()
        self._dynamic_batch_fallbacks.clear()
        now = time.time()
        availability = self.timelines
        for task in pending_tasks:
            task_id = str(task.get("task_id", "") or "")
            component = str(task.get("component", "") or "").strip().lower()
            config = self._get_dynamic_batching(component)
            if not task_id or not config or not config.get("enabled", False):
                continue
            logical_n = _safe_int(task.get("logical_batch_size"))
            input_size = _safe_float(task.get("input_size"))
            batch_size_arg = str(config["batch_size_arg"])
            if logical_n <= 1 or input_size <= 0:
                self._dynamic_batch_fallbacks[task_id] = {
                    "enabled": True,
                    "batch_size_arg": batch_size_arg,
                    "logical_batch_size": logical_n,
                    "selected_batch_size": 0,
                    "phase": "exact",
                    "policy": "fixed_n",
                    "fallback_reason": "exact_n_missing_dynamic_axes",
                }
                continue
            overrides = task.get("execution_overrides")
            pinned_k = _safe_int(
                overrides.get("batch_size") if isinstance(overrides, dict) else 0
            )
            if pinned_k and not 1 <= pinned_k <= logical_n:
                raise ValueError(
                    f"dynamic batch override must be in [1, {logical_n}], got {pinned_k}"
                )
            config_fingerprint = str(task.get("config_fingerprint", "") or "")
            for gpu_id in self._gpu_ids():
                available_vram = availability.available_vram_at(
                    gpu_id,
                    now,
                    exclude_task_id=task_id,
                )
                available_ram = availability.available_host_ram_at(
                    now,
                    exclude_task_id=task_id,
                )
                cold = self._cold_batch_candidates(
                    component=component,
                    input_size=input_size,
                    config_fingerprint=config_fingerprint,
                    gpu_id=gpu_id,
                    logical_n=logical_n,
                    available_vram=available_vram,
                    available_ram=available_ram,
                )
                if not cold:
                    self._dynamic_batch_fallbacks[task_id] = {
                        "enabled": True,
                        "batch_size_arg": batch_size_arg,
                        "logical_batch_size": logical_n,
                        "selected_batch_size": 0,
                        "phase": "exact",
                        "policy": "fixed_n",
                        "fallback_reason": "missing_scalar_evidence",
                    }
                    continue
                warm = self._warm_batch_candidates(
                    component=component,
                    input_size=input_size,
                    config_fingerprint=config_fingerprint,
                    gpu_id=gpu_id,
                    logical_n=logical_n,
                    available_vram=available_vram,
                    available_ram=available_ram,
                )
                phase = "warm" if warm else "cold"
                candidates = warm or cold
                policy = (
                    "pinned"
                    if pinned_k
                    else (
                        self.dynamic_batch_warm_policy
                        if warm
                        else self.dynamic_batch_cold_policy
                    )
                )
                if pinned_k:
                    selected = next(
                        (
                            item
                            for item in candidates
                            if item.k == pinned_k and item.supported
                        ),
                        None,
                    )
                    if selected is None and warm:
                        phase = "cold"
                        candidates = cold
                        selected = next(
                            (item for item in cold if item.k == pinned_k),
                            None,
                        )
                else:
                    selector = _DYNAMIC_BATCH_SELECTORS.get(policy, _largest_safe)
                    selected = selector(candidates, logical_n)
                key = (task_id, gpu_id)
                if selected is None:
                    self._dynamic_batch_blocked.add(key)
                    continue
                if not any(selected is item for item in candidates):
                    raise ValueError(
                        "dynamic batch selector returned a foreign candidate"
                    )
                self._dynamic_batch_profiles[key] = selected
                self._dynamic_batch_metadata[key] = {
                    "enabled": True,
                    "batch_size_arg": batch_size_arg,
                    "logical_batch_size": logical_n,
                    "selected_batch_size": selected.k,
                    "phase": phase,
                    "policy": policy,
                    "fallback_reason": "",
                    "profile_source": selected.source,
                }
        setter = getattr(self._delegate, "set_dynamic_batch_profiles", None)
        if callable(setter):
            setter(self._dynamic_batch_profiles, self._dynamic_batch_blocked)

    def _clear_legacy_dynamic_batch_profiles(self) -> None:
        setter = getattr(self._delegate, "set_dynamic_batch_profiles", None)
        if callable(setter):
            setter()
        self._dynamic_batch_profiles.clear()
        self._dynamic_batch_blocked.clear()
        self._dynamic_batch_metadata.clear()
        self._dynamic_batch_fallbacks.clear()

    def _attach_dynamic_batch_metadata(self, plan: DispatchPlan) -> dict[str, Any]:
        """Attach execution context while preserving the delegate's batch profile."""
        batch_meta = self._dynamic_batch_metadata.get(
            (str(plan.task_id), str(plan.target_gpu_id)),
            self._dynamic_batch_fallbacks.get(str(plan.task_id), {}),
        )
        if batch_meta:
            plan.worker_metadata["dynamic_batch"] = dict(batch_meta)
        return batch_meta

    def _latency_uncertainty_z(
        self,
        component: str,
        input_size: float,
        config: str = "",
    ) -> float:
        if self.backfill_latency_basis == "mean":
            return 0.0
        alpha = HEFTPriority._get_alpha(
            self.campaign_scheduler,
            component,
            input_size,
            config,
        )
        placement_z = getattr(
            getattr(self._delegate, "placement", None),
            "_uncertainty_z",
            None,
        )
        if callable(placement_z):
            return max(0.0, _safe_float(placement_z(alpha)))
        return max(0.0, NormalDist().inv_cdf(alpha))

    def _record_alternate_gpu_retry(
        self,
        task: dict[str, Any],
        evaluations: list[dict[str, Any]],
        *,
        selected_gpu: str | None,
    ) -> dict[str, Any] | None:
        per_gpu: dict[str, dict[str, Any]] = {}
        for evaluation in evaluations:
            gpu_id = str(evaluation["gpu_id"])
            row = per_gpu.setdefault(
                gpu_id,
                {"best_eft": math.inf, "allowed_eft": math.inf, "reasons": set()},
            )
            eft = _safe_float(evaluation["eft"], math.inf)
            row["best_eft"] = min(row["best_eft"], eft)
            if evaluation["allowed"]:
                row["allowed_eft"] = min(row["allowed_eft"], eft)
            else:
                row["reasons"].add(str(evaluation["reason"]))
        ranked = sorted(
            per_gpu.items(), key=lambda item: (item[1]["best_eft"], item[0])
        )
        if selected_gpu is None:
            if not ranked or any(row["allowed_eft"] < math.inf for _, row in ranked):
                return None
            outcome = "failure"
            selected_rank = 0
            selected_eft = math.inf
            prior_failed = ranked
        else:
            selected_rank = next(
                (
                    rank
                    for rank, (gpu_id, _row) in enumerate(ranked, start=1)
                    if gpu_id == selected_gpu
                ),
                0,
            )
            selected_eft = per_gpu.get(selected_gpu, {}).get("allowed_eft", math.inf)
            prior_failed = [
                (gpu_id, row)
                for gpu_id, row in ranked[: max(0, selected_rank - 1)]
                if row["allowed_eft"] == math.inf
            ]
            if not prior_failed:
                return None
            outcome = "success"
        best_rejected_gpu, best_rejected = prior_failed[0]
        best_rejected_eft = _safe_float(best_rejected["best_eft"], math.inf)
        record = {
            "task_id": str(task.get("task_id", "") or ""),
            "campaign_id": str(task.get("campaign_id", "") or ""),
            "component": str(task.get("component", "") or ""),
            "outcome": outcome,
            "candidate_gpu_count": len(ranked),
            "rejected_gpu_count": sum(
                row["allowed_eft"] == math.inf for _gpu_id, row in ranked
            ),
            "selected_gpu": selected_gpu or "",
            "selected_rank": selected_rank,
            "selected_eft": selected_eft if math.isfinite(selected_eft) else None,
            "best_rejected_gpu": best_rejected_gpu,
            "best_rejected_eft": best_rejected_eft,
            "delta_eft": (
                selected_eft - best_rejected_eft
                if math.isfinite(selected_eft)
                else None
            ),
            "rejection_reasons": sorted(
                {reason for _gpu_id, row in prior_failed for reason in row["reasons"]}
            ),
        }
        self._alternate_gpu_retry_stats["attempts"] += 1
        self._alternate_gpu_retry_stats[
            "successes" if outcome == "success" else "failures"
        ] += 1
        if outcome == "success":
            self._alternate_gpu_retry_rescued_task_ids.add(record["task_id"])
        _LOG.info("[%s] %s", self._retry_log_tag, json.dumps(record, sort_keys=True))
        return record

    @property
    def alternate_gpu_retry_stats(self) -> dict[str, int]:
        return {
            **self._alternate_gpu_retry_stats,
            "rescued_tasks": len(self._alternate_gpu_retry_rescued_task_ids),
        }

    def _candidate_worker_front_has_capacity(
        self,
        component: str,
        gpu_id: str,
    ) -> bool:
        resolve = getattr(self._delegate, "_resolve_worker", None)
        if not callable(resolve):
            return True
        resolved: Any = resolve(component, gpu_id)
        worker_name = resolved[0] if isinstance(resolved, tuple) and resolved else ""
        return self._worker_front_has_capacity(str(worker_name or ""))

    def _worker_front_has_capacity(self, worker_name: str) -> bool:
        if not worker_name:
            return True
        scheduler = getattr(self.campaign_scheduler, "_supervisor_wake_hook", None)
        has_capacity = getattr(scheduler, "_worker_front_has_capacity", None)
        provider = getattr(scheduler, "_worker_front_has_capacity_provider", None)
        if callable(has_capacity) and provider is not None:
            return bool(has_capacity(worker_name))
        supervisor = getattr(self.campaign_scheduler, "_supervisor", None)
        states = getattr(supervisor, "states", {}) if supervisor else {}
        state = states.get(worker_name) if isinstance(states, dict) else None
        return True if state is None else not self._worker_busy(state)

    def _worker_busy(self, state: Any) -> bool:
        attrs = (
            "dispatch_pending",
            "queue_in_queue",
            "queue_prepare_inflight",
            "queue_prepared_queue",
            "queue_execute_inflight",
            "queue_finalize_inflight",
        )
        return any(_safe_int(getattr(state, attr, 0)) > 0 for attr in attrs)

    def _force_single_self_concurrency_enabled(self) -> bool:
        supervisor = getattr(self.campaign_scheduler, "_supervisor", None)
        profiling_runtime = getattr(supervisor, "profiling_runtime", {}) or {}
        return bool(
            getattr(supervisor, "force_single_self_concurrency_gate", False)
        ) or bool(
            isinstance(profiling_runtime, dict)
            and profiling_runtime.get("force_single_self_concurrency_gate")
        )

    def _force_single_self_concurrency_blocks_gpu(
        self,
        component: str,
        gpu_id: str,
        task_id: str = "",
    ) -> bool:
        if not self._force_single_self_concurrency_enabled():
            return False
        timeline = self.timelines.get(str(gpu_id))
        if timeline is None:
            return False
        getter = getattr(timeline, "active_entries_for_component", None)
        if callable(getter):
            raw_entries = getter(component)
        else:
            raw_entries = (
                getattr(timeline, "active_entries", None)
                or getattr(timeline, "entries", [])
                or []
            )
        entries = raw_entries if isinstance(raw_entries, list) else []
        now = time.time()
        for entry in entries:
            if str(getattr(entry, "component", "") or "") != component:
                continue
            if bool(getattr(entry, "is_init", False)):
                continue
            if bool(getattr(entry, "is_evict_masked", False)):
                continue
            if (
                task_id
                and str(getattr(entry, "task_id", "") or "") == task_id
                and bool(getattr(entry, "is_predicted", False))
            ):
                continue
            if (
                bool(getattr(entry, "is_predicted", False))
                and not bool(getattr(entry, "is_dispatching", False))
                and _safe_float(
                    getattr(entry, "predicted_end_time", math.inf),
                    math.inf,
                )
                < now
            ):
                continue
            return True
        return False

    def _primary_campaign_id(self, pending_tasks: list[dict[str, Any]]) -> str | None:
        primary = self._delegate.campaign.select_primary(self._campaign_queues())
        if primary is not None:
            return str(primary.campaign_id)
        ordered = self._ordered_campaign_ids(pending_tasks)
        return ordered[0] if ordered else None

    def _ordered_campaign_ids(self, pending_tasks: list[dict[str, Any]]) -> list[str]:
        ordered_fn = getattr(self.campaign_scheduler, "_ordered_campaigns", None)
        if callable(ordered_fn):
            raw = ordered_fn()
            if isinstance(raw, list) and raw:
                return [
                    str(c.campaign_id) for c in raw if getattr(c, "campaign_id", "")
                ]
        queues = self._campaign_queues()
        ids = sorted(
            queues,
            key=lambda cid: self._as_float(
                getattr(queues[cid], "arrival_time", math.inf)
            ),
        )
        for task in pending_tasks:
            campaign_id = str(task.get("campaign_id", "") or "")
            if campaign_id and campaign_id not in ids:
                ids.append(campaign_id)
        return ids

    def _campaign_rank(
        self, campaign_id: str, pending_tasks: list[dict[str, Any]]
    ) -> int:
        ordered = self._ordered_campaign_ids(pending_tasks)
        return ordered.index(campaign_id) if campaign_id in ordered else len(ordered)

    def _campaign_queues(self) -> dict[str, Any]:
        raw = getattr(self.campaign_scheduler, "_campaign_queues", {}) or {}
        return raw if isinstance(raw, dict) else {}

    def _gpu_ids(self) -> list[str]:
        raw = getattr(self.timelines, "gpu_ids", []) or []
        return [str(g) for g in raw] if isinstance(raw, (list, tuple, set)) else []

    def _as_float(self, value: Any, *, default: float = 0.0) -> float:
        return _safe_float(value, default)
