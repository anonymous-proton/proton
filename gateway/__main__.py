"""Entrypoint for running the gateway supervisor and HTTP server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import web

from . import supervisor as sup
from .extraction.component_features import auto_register_all as _auto_register_features
from .gateway_identity import build_gateway_identity

from .gpu_capacity import gpu_vram_fallback_mib
from .http_server import GATEWAY_SERVICE_APP_KEY, create_app
from .planning.pipeline_dag import PipelineDAG
from .planning.scheduler import PlacementDecision, SchedulingContext, WorkerSelection

from .planning import predictive_probe as _predictive_probe

_predictive_probe.maybe_install()

from .profiling.runtime_regime import normalize_worker_context
from .signals.contracts import IntrinsicSignalSummary, PlannerIntent
from .signals.resource_profile import warm_gp_process_pool


class RegistryError(Exception):
    """Raised when no suitable worker is found."""


class WorkerRegistry:
    """Supervisor-backed worker selection and profiling isolation manager."""

    def __init__(
        self,
        supervisor: sup.WorkerSupervisor,
        profiling_runtime: dict[str, Any],
        scheduler: Any | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.profiling_runtime = dict(profiling_runtime or {})
        self._rr_cursor: dict[str, int] = {}
        self._scheduler = scheduler
        self._lock = asyncio.Lock()
        self._profile_session_id: str = ""
        self._profile_component: str = ""
        self._profile_refcount: int = 0
        self._profile_gpu_pool: list[str] = []

    @staticmethod
    def _normalize_preferred_gpu_ids(
        preferred_gpu_ids: list[str] | None,
    ) -> list[str]:
        normalized = [
            str(item).strip()
            for item in list(preferred_gpu_ids or [])
            if str(item).strip()
        ]
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _sort_gpu_ids(gpu_ids: list[str]) -> list[str]:
        def _key(item: str) -> tuple[int, int | str]:
            stripped = str(item).strip()
            if stripped.isdigit():
                return (0, int(stripped))
            return (1, stripped)

        return sorted(
            [str(item).strip() for item in gpu_ids if str(item).strip()], key=_key
        )

    @staticmethod
    def _worker_gpu_ids(st: sup.WorkerState) -> list[str]:
        raw = list(st.assigned_gpus or st.spec.gpus or [])
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def _resident_baseline_snapshot(st: sup.WorkerState) -> dict[str, Any]:
        return {
            "resident_memory_mib": st.resident_memory_mib,
            "resident_memory_source": str(st.resident_memory_source or "").strip(),
            "resident_baseline_collected_at": st.resident_baseline_collected_at,
            "resident_baseline_lifecycle_token": str(
                st.resident_baseline_lifecycle_token or ""
            ).strip(),
            "resident_baseline_state": str(st.resident_baseline_state or "").strip(),
        }

    @classmethod
    def _residency_state(cls, st: sup.WorkerState) -> str:
        return "hot" if sup.WorkerSupervisor._is_hot(st) else "cold"

    def _list_ready_candidates(
        self,
        *,
        component: str,
        preferred_gpu_ids: list[str] | None = None,
    ) -> list[sup.WorkerState]:
        workers = self.supervisor.get_ready_workers(
            component=component, replicas_only=None
        )
        if self._profile_session_id and component == self._profile_component:
            workers = [st for st in workers if st.profile_reserved]
        normalized_preferred_gpu_ids = self._normalize_preferred_gpu_ids(
            preferred_gpu_ids
        )
        if normalized_preferred_gpu_ids:
            allowed = set(normalized_preferred_gpu_ids)
            workers = [
                st for st in workers if allowed.intersection(self._worker_gpu_ids(st))
            ]
        workers = [st for st in workers if st.addr]
        workers.sort(key=lambda st: str(st.addr))
        return workers

    def _list_all_candidates(
        self,
        *,
        component: str,
        preferred_gpu_ids: list[str] | None = None,
    ) -> list[sup.WorkerState]:
        all_workers = self.supervisor.list_workers(
            component=component, include_not_ready=True
        )

        active_count = sum(
            1 for st in all_workers if st.lifecycle_state not in {"cold", "stopped"}
        )

        workers = []
        for st in all_workers:
            if (
                st.spec.max_instances is not None
                and active_count >= st.spec.max_instances
            ):
                if st.lifecycle_state in {"cold", "stopped"}:
                    continue
            workers.append(st)
        if self._profile_session_id and component == self._profile_component:
            workers = [st for st in workers if st.profile_reserved]
        normalized_preferred_gpu_ids = self._normalize_preferred_gpu_ids(
            preferred_gpu_ids
        )
        if normalized_preferred_gpu_ids:
            allowed = set(normalized_preferred_gpu_ids)
            workers = [
                st for st in workers if allowed.intersection(self._worker_gpu_ids(st))
            ]
        workers.sort(key=lambda st: st.spec.name)
        return workers

    def _pick_round_robin(
        self, *, key: str, candidates: list[sup.WorkerState]
    ) -> sup.WorkerState:
        if not candidates:
            raise RegistryError("no worker candidates")
        cursor = self._rr_cursor.get(key, 0)
        selected = candidates[cursor % len(candidates)]
        self._rr_cursor[key] = cursor + 1
        return selected

    def _select_candidate(
        self,
        *,
        key: str,
        candidates: list[sup.WorkerState],
        context: SchedulingContext,
    ) -> sup.WorkerState:
        """Select a worker using the configured scheduler, falling back to round-robin."""
        excluded_worker_names = {
            str(item).strip()
            for item in list(context.hints.get("exclude_worker_names") or [])
            if str(item).strip()
        }
        excluded_worker_addrs = {
            str(item).strip()
            for item in list(context.hints.get("exclude_worker_addrs") or [])
            if str(item).strip()
        }
        candidates = [
            st
            for st in candidates
            if str(st.spec.name or "").strip() not in excluded_worker_names
            and str(st.addr or "").strip() not in excluded_worker_addrs
        ]
        if self._scheduler is not None:
            if candidates:
                resource_tracker = getattr(self.supervisor, "resource_tracker", None)
                if resource_tracker is not None:
                    context.hints["vram_query_fn"] = (
                        resource_tracker.get_available_memory
                    )

            filtered = self._scheduler.filter(
                candidates,
                context,
                gpu_ids_fn=self._worker_gpu_ids,
            )
            if not filtered:
                raise RegistryError("no worker candidates passed the filter phase")

            selected = self._scheduler.select(
                filtered,
                context,
                gpu_ids_fn=self._worker_gpu_ids,
            )
            return selected
        return self._pick_round_robin(key=key, candidates=candidates)

    @staticmethod
    def _to_optional_non_negative_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed != parsed:
            return None
        if parsed < 0:
            return None
        return float(parsed)

    @staticmethod
    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if parsed != parsed:
            parsed = float(default)
        return max(0.0, min(1.0, float(parsed)))

    @staticmethod
    def _to_optional_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except Exception:
            return None
        if parsed < 1:
            return None
        return int(parsed)

    def _normalize_schedule_hint(
        self, schedule_hint: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        raw = dict(schedule_hint or {}) if isinstance(schedule_hint, Mapping) else {}
        result = dict(raw)
        result["component"] = str(raw.get("component") or "").strip().lower()
        result["reasons"] = [
            str(item).strip()
            for item in list(raw.get("reasons") or [])
            if str(item).strip()
        ]
        result["exclude_worker_names"] = [
            str(item).strip()
            for item in list(raw.get("exclude_worker_names") or [])
            if str(item).strip()
        ]
        result["exclude_worker_addrs"] = [
            str(item).strip()
            for item in list(raw.get("exclude_worker_addrs") or [])
            if str(item).strip()
        ]
        return result

    @staticmethod
    def _post_dispatch_active_request_count(st: sup.WorkerState) -> int:
        current = max(
            int(getattr(st, "queue_execute_inflight", 0) or 0),
            int(getattr(st, "queue_gw_inflight", 0) or 0),
            int(getattr(st, "dispatch_pending", 0) or 0),
        )
        return max(1, current + 1)

    @staticmethod
    def _post_dispatch_queue_depth(st: sup.WorkerState) -> int:
        worker_inflight = (
            int(getattr(st, "queue_prepare_inflight", 0) or 0)
            + int(getattr(st, "queue_execute_inflight", 0) or 0)
            + int(getattr(st, "queue_finalize_inflight", 0) or 0)
        )
        current = max(worker_inflight, int(getattr(st, "queue_gw_inflight", 0) or 0))
        current += int(getattr(st, "queue_in_queue", 0) or 0)
        current += int(getattr(st, "dispatch_pending", 0) or 0)
        return max(0, current + 1)

    def _co_location_signature(self, st: sup.WorkerState) -> str:
        active = WorkerRegistry._post_dispatch_active_request_count(st)
        if active <= 1:
            gpu_set = set(str(g) for g in (st.assigned_gpus or []))
            if gpu_set:
                co_components: list[str] = []
                for other in self.supervisor.states.values():
                    if other is st or not other.ready:
                        continue
                    other_gpus = set(str(g) for g in (other.assigned_gpus or []))
                    if gpu_set & other_gpus:
                        other_inflight = max(
                            int(getattr(other, "queue_execute_inflight", 0) or 0),
                            int(getattr(other, "queue_gw_inflight", 0) or 0),
                        )
                        if other_inflight > 0:
                            co_components.append(
                                f"{other.spec.component}:{other_inflight}"
                            )
                if co_components:
                    parts = sorted(co_components)
                    return f"shared_execute:{','.join(parts)}"
            return "solo_execute"
        component = str(st.spec.component or "").strip()
        return f"shared_execute:{component}:{active}"

    @staticmethod
    def _hardware_software_signature(st: sup.WorkerState) -> str:
        parts: list[str] = []
        worker_name = str(st.spec.name or "").strip()
        adapter = str(st.spec.adapter or "").strip()
        image = str(st.spec.image or "").strip()
        model_version = ""
        if st.caps is not None:
            model_version = str(getattr(st.caps, "model_version", "") or "").strip()
        gpu_ids = [
            str(item).strip()
            for item in list(st.assigned_gpus or st.spec.gpus or [])
            if str(item).strip()
        ]
        if worker_name:
            parts.append(f"worker:{worker_name}")
        if adapter:
            parts.append(f"adapter:{adapter}")
        if image:
            parts.append(f"image:{image}")
        if model_version:
            parts.append(f"model:{model_version}")
        if gpu_ids:
            parts.append(f"gpus:{len(gpu_ids)}")
        return "|".join(parts) or "default"

    def _estimator_worker_context(self, st: sup.WorkerState) -> dict[str, Any]:
        return normalize_worker_context(
            {
                "worker_addr": str(st.addr or "").strip(),
                "worker_name": str(st.spec.name or "").strip(),
                "gpu_ids": [
                    str(item).strip()
                    for item in list(st.assigned_gpus or st.spec.gpus or [])
                    if str(item).strip()
                ],
                "adapter": str(st.spec.adapter or "").strip(),
                "image": str(st.spec.image or "").strip(),
                "model_version": str(
                    getattr(st.caps, "model_version", "") or ""
                ).strip()
                if st.caps is not None
                else "",
                "active_request_count": self._post_dispatch_active_request_count(st),
                "queue_depth": self._post_dispatch_queue_depth(st),
                "dispatch_group_size": 1,
                "co_location_signature": self._co_location_signature(st),
                "hardware_software": self._hardware_software_signature(st),
                "residency_state": self._residency_state(st),
            }
        )

    def select_worker(
        self,
        component: str,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        schedule_hint: Mapping[str, Any] | None = None,
        include_not_ready: bool = False,
        preferred_worker_name: str | None = None,
        campaign_id: str | None = None,
        intrinsic_signal: IntrinsicSignalSummary | None = None,
        planner_intent: PlannerIntent | None = None,
    ) -> WorkerSelection:
        normalized_component = str(component).strip().lower()

        if include_not_ready:
            candidates = self._list_all_candidates(
                component=normalized_component,
                preferred_gpu_ids=preferred_gpu_ids,
            )
        else:
            candidates = self._list_ready_candidates(
                component=normalized_component,
                preferred_gpu_ids=preferred_gpu_ids,
            )

        if preferred_worker_addr:
            preferred_addr = str(preferred_worker_addr).strip()
            candidates = [st for st in candidates if st.addr == preferred_addr]
        elif preferred_worker_name:
            p_name = str(preferred_worker_name).strip()
            candidates = [st for st in candidates if st.spec.name == p_name]

        if not candidates:
            raise RegistryError(
                f"no ready workers for component: {normalized_component}"
            )

        key = f"{normalized_component}:{campaign_id or 'default'}"
        normalized_hint = self._normalize_schedule_hint(schedule_hint)
        ctx = SchedulingContext(
            component=normalized_component,
            preferred_gpu_ids=preferred_gpu_ids,
            preferred_worker_addr=preferred_worker_addr,
            hints=dict(normalized_hint),
            campaign_id=campaign_id,
            intrinsic_signal=intrinsic_signal,
            planner_intent=planner_intent,
        )
        selected = self._select_candidate(key=key, candidates=candidates, context=ctx)
        reasons: list[str] = [
            str(item)
            for item in list(normalized_hint.get("reasons") or [])
            if str(item).strip()
        ]
        reasons.append("routing_invariant_keep_baseline")

        selected.last_used_at = time.time()
        return WorkerSelection(
            addr=str(selected.addr or ""),
            ready=bool(selected.ready),
            gpu_ids=tuple(
                str(item).strip()
                for item in list(selected.assigned_gpus or [])
                if str(item).strip()
            ),
            worker_name=str(selected.spec.name or ""),
            resident_baseline_snapshot=self._resident_baseline_snapshot(selected),
            max_concurrency=int(selected.max_concurrency),
            priority=int(selected.spec.priority),
            estimator_worker_context=self._estimator_worker_context(selected),
            placement=PlacementDecision(
                version=1,
                chosen_worker_name=str(selected.spec.name or ""),
                chosen_worker_addr=str(selected.addr or ""),
                chosen_gpu_ids=tuple(
                    str(item).strip()
                    for item in list(selected.assigned_gpus or [])
                    if str(item).strip()
                ),
                reasons=tuple(
                    dict.fromkeys(
                        str(reason).strip() for reason in reasons if str(reason).strip()
                    )
                ),
                intrinsic_signal=intrinsic_signal.as_dict()
                if intrinsic_signal is not None
                else {},
                planner_intent=(planner_intent or PlannerIntent()).as_dict(),
            ),
            worker_addr=str(selected.addr or ""),
            worker_pid=selected.host_pid,
            current_activation_mib=float(selected.current_activation_mb),
        )

    def resolve_worker_capabilities(
        self,
        component: str,
        preferred_worker_name: str | None = None,
    ) -> dict[str, object]:
        normalized_component = str(component).strip().lower()
        candidates = self._list_all_candidates(component=normalized_component)
        if not candidates:
            raise RegistryError(
                f"no workers configured for component '{normalized_component}'"
            )
        selected = candidates[0]
        preferred = str(preferred_worker_name or "").strip()
        if preferred:
            for st in candidates:
                if (
                    str(st.spec.name or "").strip() == preferred
                    or str(st.addr or "").strip() == preferred
                ):
                    selected = st
                    break
        caps = selected.caps
        supports_cancel = any(
            bool(getattr(getattr(st, "caps", None), "supports_cancel_batch", True))
            for st in candidates
        )
        return {
            "addr": selected.addr or "",
            "gpu_ids": list(selected.assigned_gpus or selected.spec.gpus or []),
            "worker_name": selected.spec.name,
            "max_concurrency": int(selected.max_concurrency),
            "supports_cancel_batch": supports_cancel
            if not preferred
            else bool(getattr(caps, "supports_cancel_batch", True))
            if caps is not None
            else True,
        }

    async def ensure_worker_for_dispatch(
        self,
        component: str,
        preferred_worker_addr: str | None = None,
        preferred_gpu_ids: list[str] | None = None,
        preferred_worker_name: str | None = None,
        campaign_id: str | None = None,
    ) -> bool:
        """Plan fix — returns ``did_activate`` (True iff
        supervisor actually performed cold-start work for any candidate
        during this call).  Early-return path (all candidates already
        ready) returns False.  Caller (``_activate_selected_worker``)
        propagates the flag up to ``RealityValidator._activate_worker``
        so ``was_cold_start`` reflects the supervisor's authoritative
        init execution, not proxy signals.
        """
        normalized_component = str(component).strip().lower()

        candidates = self._list_all_candidates(
            component=normalized_component,
            preferred_gpu_ids=preferred_gpu_ids,
        )

        try:
            ready_candidates = self._list_ready_candidates(
                component=normalized_component,
                preferred_gpu_ids=preferred_gpu_ids,
            )
            if len(ready_candidates) >= len(candidates):
                return False
        except RegistryError:
            pass
        if preferred_worker_name:
            candidates = [
                st for st in candidates if st.spec.name == preferred_worker_name
            ]
        elif preferred_worker_addr:
            preferred_addr = str(preferred_worker_addr).strip()
            candidates = [st for st in candidates if st.addr == preferred_addr]

        if not candidates:
            raise RegistryError(
                f"no workers configured for component '{normalized_component}'"
            )

        key = f"{normalized_component}:{campaign_id or 'default'}:cold"
        if preferred_worker_name and candidates:
            selected = candidates[0]
        else:
            selected = self._pick_round_robin(key=key, candidates=candidates)
        _, did_activate = await self.supervisor.ensure_worker_ready(
            component=normalized_component,
            worker_name=selected.spec.name,
            campaign_id=str(campaign_id or "").strip(),
        )
        return did_activate

    def is_profile_isolation_active(self) -> bool:
        return bool(self._profile_session_id)

    def profile_parallelism(self, component: str) -> int:
        if (
            self._profile_session_id
            and str(component).strip().lower() == self._profile_component
        ):
            return self.supervisor.profile_parallelism(self._profile_component)
        pool = [
            str(item).strip()
            for item in list(self.profiling_runtime.get("gpu_pool") or [])
            if str(item).strip()
        ]
        return max(1, len(pool))

    async def acquire_profile_isolation(
        self,
        component: str,
        preferred_gpu_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_component = str(component).strip().lower()
        async with self._lock:
            requested_gpus = self._normalize_preferred_gpu_ids(preferred_gpu_ids)
            if self._profile_session_id:
                if normalized_component != self._profile_component:
                    raise RegistryError(
                        f"profiling isolation already active for component '{self._profile_component}'"
                    )
                if requested_gpus:
                    active_pool = set(self._profile_gpu_pool)
                    requested_pool = set(requested_gpus)
                    if not requested_pool.issubset(active_pool):
                        merged_pool = self._sort_gpu_ids(
                            list(active_pool.union(requested_pool))
                        )
                        replicas = await self.supervisor.start_profile_replicas(
                            component=normalized_component,
                            gpu_ids=merged_pool,
                        )
                        if not replicas:
                            raise RegistryError(
                                f"profiling isolation failed: no ready replicas for component '{normalized_component}'"
                            )
                        self._profile_gpu_pool = merged_pool
                self._profile_refcount += 1
                return {
                    "session_id": self._profile_session_id,
                    "parallelism": self.profile_parallelism(normalized_component),
                }

            if requested_gpus:
                pool = requested_gpus
            else:
                pool = [
                    str(item).strip()
                    for item in list(self.profiling_runtime.get("gpu_pool") or [])
                    if str(item).strip()
                ]
            if not pool:
                pool = [
                    str(item).strip()
                    for item in list(self.profiling_runtime.get("gpu_pool") or [])
                    if str(item).strip()
                ]
            if not pool:
                raise RegistryError(
                    "profiling isolation requires profiling_runtime.gpu_pool"
                )

            replicas = await self.supervisor.start_profile_replicas(
                component=normalized_component, gpu_ids=pool
            )
            if not replicas:
                raise RegistryError(
                    f"profiling isolation failed: no ready replicas for component '{normalized_component}'"
                )

            self._profile_component = normalized_component
            self._profile_session_id = str(uuid.uuid4())
            self._profile_refcount = 1
            self._profile_gpu_pool = self._sort_gpu_ids(list(pool))
            self._rr_cursor.clear()
            if hasattr(self._scheduler, "reset"):
                self._scheduler.reset()
            return {
                "session_id": self._profile_session_id,
                "parallelism": len(replicas),
            }

    async def release_profile_isolation(self, session_id: str) -> None:
        normalized_session = str(session_id).strip()
        if not normalized_session:
            return
        async with self._lock:
            if normalized_session != self._profile_session_id:
                return
            self._profile_refcount -= 1
            if self._profile_refcount > 0:
                return
            await self.supervisor.restore_base_workers()
            self._profile_component = ""
            self._profile_session_id = ""
            self._profile_refcount = 0
            self._profile_gpu_pool = []
            self._rr_cursor.clear()
            if hasattr(self._scheduler, "reset"):
                self._scheduler.reset()

    def list_workers(self) -> list[dict[str, Any]]:
        return self.supervisor.list_worker_snapshots()


def _parse_components(raw_components: list[str]) -> list[str]:
    return sup._parse_components(raw_components)


def _resolve_components(args: argparse.Namespace) -> list[str]:
    if args.all or args.components is None:
        return sup.discover_components(args.worker_config)
    return _parse_components(args.components)


async def _run_supervisor_only(args: argparse.Namespace) -> None:
    components = _resolve_components(args)
    if not components:
        raise SystemExit(f"No components found in config: {args.worker_config}")
    await sup.main_async(
        args.runtime_config,
        args.worker_config,
        components,
        args.host,
        not args.no_stream_logs,
        args.readiness_timeout,
        args.stats_interval,
    )


async def _run_all(args: argparse.Namespace) -> None:
    components = _resolve_components(args)
    if not components:
        raise SystemExit(f"No components found in config: {args.worker_config}")

    specs, gpu_allocation, profiling_runtime = sup.load_specs(
        args.runtime_config, args.worker_config, components
    )
    if not specs:
        comps = ", ".join(components)
        raise SystemExit(
            f"No workers defined for component(s) [{comps}] in {args.worker_config} (via {args.runtime_config})"
        )

    _auto_register_features(specs=specs)

    pipeline_dag = PipelineDAG()

    cfg = sup.SupervisorConfig(
        specs=specs,
        host=args.host,
        readiness_timeout_s=args.readiness_timeout,
        stream_logs=not (args.no_stream_logs or args.no_log),
        stats_interval_s=args.stats_interval,
        gpu_allocation=gpu_allocation,
        profiling_runtime=profiling_runtime,
        eviction_policy=str(profiling_runtime.get("eviction_policy", "lru"))
        .strip()
        .lower(),
        recovery_policy=str(profiling_runtime.get("recovery_policy", "mru"))
        .strip()
        .lower(),
        weight_oom_policy=str(profiling_runtime.get("weight_oom_policy", "optimistic"))
        .strip()
        .lower(),
        cold_start_activation_ratio=float(
            profiling_runtime.get("cold_start_activation_ratio", 1.0)
        ),
        max_primary_slowdown=float(profiling_runtime.get("max_primary_slowdown", 0.5)),
        slowdown_cold_start_default=float(
            profiling_runtime.get("slowdown_cold_start_default", 1.0)
        ),
        pairwise_slowdown_cold_start=float(
            profiling_runtime.get("pairwise_slowdown_cold_start", 1.3)
        ),
        self_slowdown_cold_start_default=float(
            profiling_runtime.get("self_slowdown_cold_start_default", 2.0)
        ),
        self_concurrency_max_primary_cold_start=int(
            profiling_runtime.get("self_concurrency_max_primary_cold_start", 1)
        ),
        self_concurrency_max_backfill_cold_start=int(
            profiling_runtime.get("self_concurrency_max_backfill_cold_start", 2)
        ),
        gp_maturity_observations_per_dim=int(
            profiling_runtime.get("gp_maturity_observations_per_dim", 10)
        ),
        disable_self_interference_gate=bool(
            profiling_runtime.get("disable_self_interference_gate", False)
        ),
        disable_pbbc_gate=bool(profiling_runtime.get("disable_pbbc_gate", False)),
        disable_primary_slo_protection=bool(
            profiling_runtime.get("disable_primary_slo_protection", False)
        ),
        disable_slowdown_cost=bool(
            profiling_runtime.get("disable_slowdown_cost", False)
        ),
        admission_host_ram_min_available_mib=int(
            profiling_runtime.get("admission_host_ram_min_available_mib", 2048)
        ),
        host_ram_leak_idle_grace_sec=float(
            profiling_runtime.get("host_ram_leak_idle_grace_sec", 10.0)
        ),
    )
    supervisor = sup.WorkerSupervisor(cfg)

    scheduler_policy_name = str(
        profiling_runtime.get("scheduler_policy", "campaign_fifo")
    ).strip()
    print(
        f"Scheduler policy (CampaignScheduler variant): {scheduler_policy_name}",
        flush=True,
    )

    registry = WorkerRegistry(supervisor, profiling_runtime, scheduler=None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)

    import atexit
    import queue as _queue
    from logging.handlers import QueueHandler, QueueListener

    class _FastQueueHandler(QueueHandler):
        def prepare(self, record):
            return record

    _root = logging.getLogger()
    _existing = list(_root.handlers)
    _log_queue: _queue.Queue = _queue.Queue(
        -1
    )
    _qh = _FastQueueHandler(_log_queue)
    _qh.setLevel(level)
    for h in _existing:
        _root.removeHandler(h)
    _root.addHandler(_qh)
    _qlistener = QueueListener(_log_queue, *_existing, respect_handler_level=True)
    _qlistener.start()

    def _stop_qlistener() -> None:
        try:
            _qlistener.stop()
        except Exception:
            pass

    atexit.register(_stop_qlistener)

    log = logging.getLogger(__name__)

    try:
        warm_gp_process_pool()
        log.info("[gp-process-pool] warmed with spawn-safe workers")
    except Exception:
        log.warning(
            "[gp-process-pool] warm-up failed; continuing with lazy startup",
            exc_info=True,
        )

    work_root = Path(args.work_root).expanduser().resolve() if args.work_root else None
    gateway_identity = build_gateway_identity(
        host=args.http_host,
        port=args.http_port,
        repo_root=Path(__file__).resolve().parents[1],
    )

    repo_root = Path(__file__).resolve().parents[1]
    index_db_path = Path(
        str(
            profiling_runtime.get("index_db_path")
            or (repo_root / ".index" / "profile_runs.sqlite3")
        )
    )
    from .profiling.index_store import RunIndexStore, boot_phase_bypass

    run_index = RunIndexStore(index_db_path)

    def _env_truthy(name: str) -> bool:
        return str(os.environ.get(name, "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    control_plane_overhead_enabled = bool(
        profiling_runtime.get("control_plane_overhead_recorder_enabled", False)
    ) or _env_truthy("PROTON_CONTROL_PLANE_OVERHEAD")

    with boot_phase_bypass():
        supervisor.bootstrap_weights(run_index)

    http_app = create_app(
        worker_selector=registry.select_worker,
        work_root=work_root,
        max_inflight_per_worker=args.gateway_max_inflight,
        ensure_worker_ready=registry.ensure_worker_for_dispatch,
        worker_capabilities_resolver=registry.resolve_worker_capabilities,
        worker_inventory_provider=registry.list_workers,
        profiling_index_db_path=index_db_path,
        on_task_start=supervisor.on_task_start,
        on_task_end=supervisor.on_task_end,
        resource_tracker=supervisor.resource_tracker,
        check_worker_status=supervisor.check_worker_status,
        evict_fn=supervisor.evict_idle_workers,
        adjust_worker_inflight_by_name=supervisor.adjust_worker_inflight_by_name,
        bump_dispatch_pending=supervisor.bump_dispatch_pending,
        release_dispatch_pending=supervisor.release_dispatch_pending,
        gateway_identity=gateway_identity.as_dict(),
        pipeline_dag=pipeline_dag,
        max_task_retries=int(profiling_runtime.get("max_task_retries", 5)),
        scheduler_policy=scheduler_policy_name,
        adapter_retry_backoff=profiling_runtime.get("adapter_retry_backoff"),
        dispatch_backlog_per_worker=int(
            profiling_runtime.get("profile_dispatch_backlog_per_worker", 1),
        ),
        dispatch_start_grace_sec=float(
            profiling_runtime.get("profile_dispatch_start_grace_sec", 1.0),
        ),
        control_plane_overhead_enabled=control_plane_overhead_enabled,
        control_plane_overhead_top_k=int(
            profiling_runtime.get("control_plane_overhead_top_k", 32),
        ),
    )
    if control_plane_overhead_enabled:
        log.info("[control-plane-overhead] recorder enabled")
    gateway_service = http_app[GATEWAY_SERVICE_APP_KEY]
    supervisor.set_activation_peak_recorder(
        gateway_service._signal_service.record_activation_peak
    )

    gateway_service.supervisor = supervisor
    with boot_phase_bypass():
        try:
            _runtime_state = run_index.load_signal_snapshot("runtime_state_v1")
        except Exception:
            _runtime_state = None
    if isinstance(_runtime_state, dict):
        _signal_state = _runtime_state.get("signal_service") or {}
        if isinstance(_signal_state, Mapping):
            if _signal_state.get("latency_active"):
                raise RuntimeError(
                    "runtime_state_v1 contains active transient signal state; "
                    "refusing completed-run bootstrap"
                )
            _activation_peaks = _signal_state.get("activation_peaks")
            if isinstance(_activation_peaks, Mapping) and _activation_peaks:
                log.warning(
                    "[runtime-state] Dropping %d stale activation peak(s) "
                    "from completed-run bootstrap snapshot",
                    len(_activation_peaks),
                )
                _signal_state["activation_peaks"] = {}
        try:
            _sup_state = _runtime_state.get("supervisor") or {}
            if _sup_state and hasattr(supervisor, "import_runtime_state"):
                _counts = supervisor.import_runtime_state(_sup_state)
                log.info("[runtime-state] Restored supervisor state: %s", _counts)
        except Exception:
            log.warning(
                "[runtime-state] Failed to restore supervisor state", exc_info=True
            )
        try:
            _init_state = _runtime_state.get("init_profile") or {}
            if _init_state and hasattr(supervisor.init_tracker, "import_state"):
                supervisor.init_tracker.import_state(_init_state)
                log.info("[runtime-state] Restored init profile state")
        except Exception:
            log.warning(
                "[runtime-state] Failed to restore init profile state", exc_info=True
            )
        try:
            _planner_state = _runtime_state.get("planner") or {}
            _import_planner_state = getattr(
                gateway_service._planner, "import_runtime_state", None
            )
            if _planner_state and callable(_import_planner_state):
                _count = _import_planner_state(_planner_state)
                log.info("[runtime-state] Restored planner state: %d epoch(s)", _count)
        except Exception:
            log.warning(
                "[runtime-state] Failed to restore planner state", exc_info=True
            )
    gateway_service._runtime_config = (
        dict(profiling_runtime) if profiling_runtime else {}
    )

    def _on_worker_killed_handler(comp, gpus):
        dead_gpus = {str(g) for g in gpus}
        dead_addrs: set[str] = set()
        for st in supervisor.states.values():
            if st.spec.component != comp:
                continue
            if any(str(g) in dead_gpus for g in (st.spec.gpus or [])):
                for ref in (getattr(st, "_prev_addr", None), getattr(st, "addr", None)):
                    if ref:
                        dead_addrs.add(str(ref).strip())
        affected_records = []
        for record in gateway_service._tasks.values():
            if getattr(record, "component", "") != comp:
                continue
            addr = str(getattr(record, "dispatch_worker_addr", "") or "").strip()
            if addr and addr in dead_addrs:
                affected_records.append(record)
                record._pending_vram_releases = []

        gateway_service._planner.campaign_scheduler.on_worker_killed(comp, gpus)

        try:
            tracker = getattr(
                gateway_service._planner,
                "_constraint_tracker",
                None,
            )
            if tracker is not None:
                cold_set = getattr(tracker, "_worker_cold_on_next_plan", None)
                if cold_set is not None:
                    for st in supervisor.states.values():
                        if st.spec.component != comp:
                            continue
                        if not any(str(g) in dead_gpus for g in (st.spec.gpus or [])):
                            continue
                        wname = getattr(st, "worker_name", None) or getattr(
                            st.spec,
                            "worker_name",
                            None,
                        )
                        if wname:
                            cold_set.add(str(wname))
        except Exception:
            log.warning(
                "[] _worker_cold_on_next_plan mark failed for comp=%s gpus=%s",
                comp,
                gpus,
                exc_info=True,
            )

        for record in affected_records:
            if not getattr(record, "_cancel_reason", ""):
                record._cancel_reason = "worker_killed_reenqueue"
        for st in supervisor.states.values():
            if st.spec.component == comp and any(
                str(g) in [str(x) for x in gpus] for g in (st.spec.gpus or [])
            ):
                _dead_addr = getattr(st, "_prev_addr", None)
                if _dead_addr:
                    gateway_service._signal_service.cleanup_worker(_dead_addr)

    supervisor._on_worker_killed = _on_worker_killed_handler
    supervisor._invalidate_channel_fn = gateway_service.invalidate_channel

    async def _pre_init_cb(component: str, gpu_id: str, campaign_id: str) -> bool:
        try:
            wanted_gpu = str(gpu_id).strip()
            campaign_scheduler = gateway_service._planner.campaign_scheduler

            def _handoff_weight_reservation() -> None:
                campaign_scheduler._handoff_pre_init_weight_reservation(
                    component,
                    wanted_gpu,
                    campaign_id,
                )

            target_worker = ""
            for st in supervisor.states.values():
                if st.spec.component != component:
                    continue
                assigned = [
                    str(item).strip()
                    for item in list(st.assigned_gpus or st.spec.gpus or [])
                    if str(item).strip()
                ]
                if assigned and assigned[0] == wanted_gpu:
                    target_worker = st.spec.name
                    break

            if target_worker:
                _, did_activate = await supervisor.ensure_worker_ready(
                    component=component,
                    worker_name=target_worker,
                    campaign_id=campaign_id,
                    on_weight_reserved=_handoff_weight_reservation,
                )
                return bool(did_activate)

            await supervisor.ensure_component_pool_ready(
                component=component,
                gpu_ids=[gpu_id],
                campaign_id=campaign_id,
                on_weight_reserved=_handoff_weight_reservation,
            )
            return True
        except Exception as exc:
            log.debug("[pre-init] failed for %s on GPU %s: %s", component, gpu_id, exc)
            raise

    gateway_service._planner.campaign_scheduler._pre_init_fn = _pre_init_cb
    supervisor._grace_period_fn = (
        gateway_service._planner.campaign_scheduler._get_eviction_grace_period
    )
    gateway_service._planner.campaign_scheduler._cancel_fn = (
        gateway_service._cancel_task_impl
    )
    gateway_service._planner.campaign_scheduler.register_cancel_callback(
        gateway_service._cancel_task_impl
    )
    supervisor._eviction_handler = gateway_service._planner.select_for_eviction
    supervisor._host_ram_eviction_handler = (
        gateway_service._planner.select_for_host_ram_eviction
    )
    supervisor._recovery_handler = gateway_service._planner.select_for_recovery
    from gateway.profiling.index_store import boot_phase_bypass

    gateway_svc = http_app[GATEWAY_SERVICE_APP_KEY]
    with boot_phase_bypass():
        gateway_svc._signal_service.bootstrap_from_history()

    http_runner = web.AppRunner(
        http_app,
        access_log=None if args.no_log else logging.getLogger("aiohttp.access"),
    )

    loop = asyncio.get_running_loop()

    def _sigint() -> None:
        supervisor.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sigint)
        except NotImplementedError:
            pass

    await supervisor.start_all()
    if args.stats_interval > 0:
        supervisor._stats_task = asyncio.create_task(supervisor._stats_loop())

    async def _snapshot_loop() -> None:
        interval = 300
        while True:
            await asyncio.sleep(interval)
            try:
                await gateway_svc._run_index_call(
                    gateway_svc._signal_service.save_interference_snapshot
                )
            except Exception:
                log.debug("Periodic interference snapshot failed", exc_info=True)

    _snapshot_task = asyncio.create_task(_snapshot_loop())

    await http_runner.setup()
    http_site = web.TCPSite(http_runner, args.http_host, args.http_port, backlog=65535)
    try:
        await http_site.start()
    except OSError as e:
        if e.errno == 98:
            log.error(
                "Port %d on host %s is already in use. "
                "If running multiple gateways, specify a unique port with --http-port.",
                args.http_port,
                args.http_host,
            )
            await http_runner.cleanup()
            await supervisor.stop_all()
        raise e
    log.info(
        "Gateway HTTP server listening on http://%s:%d", args.http_host, args.http_port
    )

    await supervisor.stop_event().wait()

    _snapshot_task.cancel()
    try:
        await gateway_svc.save_runtime_state_snapshot()
    except Exception:
        log.warning("Failed to save runtime_state_v1 at shutdown", exc_info=True)
    try:
        await gateway_svc.save_control_plane_overhead_snapshot()
    except Exception:
        log.warning(
            "Failed to save control_plane_overhead_v1 at shutdown",
            exc_info=True,
        )
    try:
        _signal_service = gateway_svc._signal_service
        _snapshot = _signal_service._interference_registry.export_snapshot()
        await gateway_svc._run_index_call(
            gateway_svc._run_index.save_signal_snapshot,
            _signal_service._SNAPSHOT_KEY,
            _snapshot,
        )
    except Exception:
        log.warning("Failed to save interference snapshot at shutdown", exc_info=True)

    _gp_dump_path = os.environ.get("PROTON_GP_DUMP_PATH", "")
    if _gp_dump_path:
        try:
            _intf_reg = getattr(
                gateway_svc._signal_service,
                "_interference_registry",
                None,
            )
            if _intf_reg is not None and hasattr(_intf_reg, "dump_to_jsonl"):
                n_records = _intf_reg.dump_to_jsonl(_gp_dump_path)
                log.info(
                    "InterferenceRegistry dumped %d records to %s",
                    n_records,
                    _gp_dump_path,
                )
        except Exception:
            log.warning(
                "InterferenceRegistry dump_to_jsonl failed",
                exc_info=True,
            )

    _event_handler = getattr(gateway_svc, "_event_handler", None)
    if _event_handler is not None:
        try:
            await _event_handler.stop()
        except Exception:
            log.warning("EventHandler.stop failed", exc_info=True)
    _supervisor = getattr(gateway_svc, "_scheduling_supervisor", None)
    if _supervisor is not None:
        try:
            await _supervisor.stop()
        except Exception:
            log.warning("SchedulingSupervisor.stop failed", exc_info=True)
    _observer = getattr(gateway_svc, "_gpu_health_observer", None)
    if _observer is not None:
        try:
            await _observer.stop()
        except Exception:
            log.warning("GpuHealthObserver.stop failed", exc_info=True)
    _gplanner = getattr(gateway_svc._planner, "_global_planner", None)
    _tracker = (
        getattr(_gplanner, "_constraint_tracker", None)
        if _gplanner is not None
        else None
    )
    if _tracker is not None:
        try:
            await _tracker.stop_sweeper()
        except Exception:
            log.warning("ConstraintTracker.stop_sweeper failed", exc_info=True)

    await http_runner.cleanup()
    await supervisor.stop_all()
    supervisor._print_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "supervisor"],
        default="all",
        help="Run supervisor + server (all), or supervisor only.",
    )
    parser.add_argument(
        "--runtime-config",
        "--config",
        type=str,
        default="configs/runtime.proton.yaml",
        help="Path to runtime config (default: configs/runtime.proton.yaml)",
        dest="runtime_config",
    )
    parser.add_argument(
        "--worker-config",
        type=str,
        default="configs/workers.yaml",
        help="Path to worker config (default: configs/workers.yaml)",
        dest="worker_config",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--components",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Component name(s) to start (default: all components in config). "
            "Supports multiple values or comma-separated lists."
        ),
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Start all components defined in the config",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for published ports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for readiness per worker",
    )
    parser.add_argument(
        "--no-stream-logs",
        action="store_true",
        help="Disable log streaming",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Suppress aiohttp access logs and worker logs (kept critical events like eviction)",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Seconds between queue stats snapshots (default: 5)",
    )
    parser.add_argument(
        "--work-root",
        type=str,
        default="",
        help="Required work root prefix for task workdirs",
    )
    parser.add_argument(
        "--gateway-max-inflight",
        type=int,
        default=8,
        help="Max in-flight RPCs per worker (default: 8 to enable pipelining)",
    )
    parser.add_argument(
        "--http-host",
        type=str,
        default=None,
        help="HTTP server bind host (default: same as --host)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8098,
        help="HTTP server bind port (default: 8098)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.http_host is None:
        args.http_host = args.host

    if args.mode == "supervisor":
        asyncio.run(_run_supervisor_only(args))
        return

    impl = _detect_scheduler_implementation(args.runtime_config)
    if impl in ("fifo", "k8s", "slurm"):
        asyncio.run(_run_all_baseline(args, impl))
        return
    asyncio.run(_run_all(args))


def _detect_scheduler_implementation(runtime_config_path: str) -> str:
    """Read ``scheduler_implementation`` from runtime config; default
    ``proton`` if absent (existing behaviour)."""
    import yaml as _yaml

    try:
        with open(runtime_config_path) as f:
            raw = _yaml.safe_load(f) or {}
    except FileNotFoundError:
        return "proton"
    pr = raw.get("profiling_runtime") or {}
    return str(pr.get("scheduler_implementation", "proton")).strip().lower() or "proton"


async def _run_all_baseline(args: argparse.Namespace, impl: str) -> None:
    """Baseline gateway runtime — minimal alternative to ``_run_all``
    for FIFO / K8s / Slurm schedulers.  Reuses ``WorkerSupervisor``
    but replaces ``GatewayHTTPService`` + ``GlobalPlanner`` +
    ``RealityValidator`` + ``EventHandler`` with the much simpler
    ``BaselineGatewayRuntime`` from ``gateway.baseline.runtime``.
    """
    from pathlib import Path

    from gateway.baseline.runtime import BaselineGatewayRuntime, select_scheduler
    from gateway.baseline.shared import WorkersProfile

    components = _resolve_components(args)
    if not components:
        raise SystemExit(f"No components found in config: {args.worker_config}")

    specs, gpu_allocation, profiling_runtime = sup.load_specs(
        args.runtime_config,
        args.worker_config,
        components,
    )
    if not specs:
        comps = ", ".join(components)
        raise SystemExit(
            f"No workers defined for component(s) [{comps}] in {args.worker_config} "
            f"(via {args.runtime_config})"
        )

    _auto_register_features(specs=specs)

    profile_path = os.environ.get("WORKERS_PROFILE_PATH") or profiling_runtime.get(
        "workers_profile_path",
        "configs/workers.profile.yaml",
    )
    workers_profile = WorkersProfile.load(Path(profile_path))

    cfg = sup.SupervisorConfig(
        specs=specs,
        host=args.host,
        readiness_timeout_s=args.readiness_timeout,
        stream_logs=not (args.no_stream_logs or args.no_log),
        stats_interval_s=args.stats_interval,
        gpu_allocation=gpu_allocation,
        profiling_runtime=profiling_runtime,
        eviction_policy=str(profiling_runtime.get("eviction_policy", "lru"))
        .strip()
        .lower(),
        recovery_policy=str(profiling_runtime.get("recovery_policy", "mru"))
        .strip()
        .lower(),
        eviction_grace_period_s=float(
            profiling_runtime.get("eviction_grace_period_s", 5.0)
        ),
        admission_host_ram_min_available_mib=int(
            profiling_runtime.get("admission_host_ram_min_available_mib", 2048)
        ),
        host_ram_leak_idle_grace_sec=float(
            profiling_runtime.get("host_ram_leak_idle_grace_sec", 10.0)
        ),
    )
    supervisor = sup.WorkerSupervisor(cfg)

    gpu_pool = list(gpu_allocation.get("pool") or [0, 1, 2, 3])
    gpu_capacity_mb: dict[str, int] = {
        str(g): gpu_vram_fallback_mib()
        for g in gpu_pool
    }
    try:
        await supervisor.ensure_initialized()
        for g in gpu_pool:
            total = supervisor.total_vram.get(str(g))
            if total:
                gpu_capacity_mb[str(g)] = int(total)
    except Exception:
        pass

    scheduler = select_scheduler(
        runtime_cfg=profiling_runtime,
        workers_profile=workers_profile,
        gpu_capacity_mb=gpu_capacity_mb,
        worker_name_format=str(
            profiling_runtime.get("worker_name_format", "{component}-gpu{gpu_id}"),
        ),
    )

    def _on_worker_killed_handler(component: str, gpu_ids: list[str]) -> None:
        handler = getattr(scheduler, "on_worker_killed", None)
        if callable(handler):
            handler(component, gpu_ids)

    supervisor._on_worker_killed = _on_worker_killed_handler
    print(
        f"[baseline] runtime initialised — scheduler={impl} components={components} "
        f"gpus={gpu_pool} workers_profile={profile_path}",
        flush=True,
    )

    runtime = BaselineGatewayRuntime(
        scheduler=scheduler,
        supervisor=supervisor,
        gpu_capacity_mb=gpu_capacity_mb,
        host=args.http_host,
        port=args.http_port,
        worker_name_format=str(
            profiling_runtime.get("worker_name_format", "{component}-gpu{gpu_id}"),
        ),
        baseline_idle_eviction=dict(
            profiling_runtime.get("baseline_idle_eviction") or {}
        ),
    )
    await asyncio.gather(
        supervisor.run(),
        runtime.run(),
    )


if __name__ == "__main__":
    main()
