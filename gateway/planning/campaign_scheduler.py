"""Campaign-level scheduling orchestrator.

Sits above ``PlannerService`` and manages:

1. Campaign priority queue (ordered by arrival time).
2. GPU timeline model for lookahead planning.
3. Primary vs backfill classification.
4. Backfill admission with GP-predicted deadlines.
5. Eviction trigger when primary needs resources.
6. Pre-init for downstream pipeline stages.
7. Fan-out pre-assignment.
8. Drift response (timeline invalidation + backfill reassessment).

See <docs> for full design.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

from ..gpu_capacity import gpu_vram_fallback_mib
from .pipeline_dag import DAGContext
from .scenario import GpuTimeline, SchedulingScenario, TimelineEntry

_LOG = logging.getLogger(__name__)


def _read_mem_total_mib() -> int:
    """Read /proc/meminfo MemTotal in MiB; return 0 on failure."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        _LOG.debug("failed to read MemTotal from /proc/meminfo", exc_info=True)
    return 0


def _read_mem_available_mib() -> int:
    """Read /proc/meminfo MemAvailable in MiB; fail open on error."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        _LOG.debug("failed to read MemAvailable from /proc/meminfo", exc_info=True)
    return 1 << 30


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise




@dataclass
class CampaignQueue:
    """Per-campaign state tracked by the scheduler.

    Pending/active liveness is tracked with explicit counters because hot-path
    scans over gateway task records were too expensive under bench fan-out.
    Terminal breakdown (succeeded / failed / cancelled) still reads the
    authoritative per-campaign TaskRecord view for ops visibility.
    """

    _fallback_warned: ClassVar[bool] = False

    campaign_id: str
    arrival_time: float
    dag_context: DAGContext | None = None

    fan_out_assignments: dict[str, list[str]] = field(default_factory=dict)
    fan_out_counters: dict[str, int] = field(default_factory=dict)

    _observed_input_sizes: dict[str, list[float]] = field(default_factory=dict)

    _gateway_tasks: Any = field(default=None, repr=False)

    _tasks_by_campaign_ref: Any = field(default=None, repr=False)

    _STATE_SUBMITTED: int = 1
    _STATE_RUNNING: int = 2
    _STATE_SUCCEEDED: int = 3
    _STATE_FAILED: int = 4
    _STATE_CANCELLED: int = 5

    _pending_count: int = 0
    _active_count: int = 0
    _completed_count: int = 0
    _pending_ids: set[str] = field(default_factory=set)
    _active_ids: set[str] = field(default_factory=set)
    _workflow_completion_expected: bool = False
    _workflow_completion_received: bool = False
    _workflow_completion_ok: bool = False
    _workflow_completion_message: str = ""

    _dispatched_components: set[str] = field(default_factory=set)

    _remaining_est_cache: float | None = None

    _is_empty_slow_cache: tuple[tuple[int, int], bool] | None = None

    def _campaign_records(self) -> list:
        """Return all TaskRecords belonging to this campaign.

        Plan fix — O(1) dispatch through the shared
        ``_tasks_by_campaign_ref`` index (owned by CampaignScheduler).
        Prior behaviour (full ``_gateway_tasks`` scan) cost O(M=cluster-
        wide) per call; under bench concurrency=9 this was the dominant
        MainThread hot frame driving ``_on_scenario_changed`` to ~1 s
        per invocation (hot-path warnings 322× in a single bench run).

        Fallback path (scan) is retained exclusively for unit-test mocks
        that instantiate ``CampaignQueue`` without running through
        ``CampaignScheduler._get_or_create`` (so the index ref never
        gets bound).  If activated from a production path the first
        invocation emits one WARNING with ``stack_info`` — subsequent
        calls are class-level suppressed to avoid log spam.  A live
        WARNING here is a P1 signal for index wiring regressions.
        """
        if not self._gateway_tasks:
            return []
        idx = self._tasks_by_campaign_ref
        if idx is not None:
            return list(idx.get(self.campaign_id, {}).values())
        cls = type(self)
        if not getattr(cls, "_fallback_warned", False):
            _LOG.warning(
                "[patch-v4-04] _campaign_records: _tasks_by_campaign_ref "
                "missing (campaign_id=%s).  Falling back to O(M) scan.  "
                "If observed in production, this indicates an index "
                "binding bug (CampaignScheduler ↔ CampaignQueue wiring).  "
                "Stack trace follows.",
                self.campaign_id,
                stack_info=True,
            )
            cls._fallback_warned = True
        return [
            r
            for r in self._gateway_tasks.values()
            if getattr(r, "campaign_id", "") == self.campaign_id
        ]

    @property
    def pending_tasks(self) -> int:
        return self._pending_count

    @pending_tasks.setter
    def pending_tasks(self, _: int) -> None:
        pass

    @property
    def active_tasks(self) -> int:
        return self._active_count

    @active_tasks.setter
    def active_tasks(self, _: int) -> None:
        pass

    @property
    def completed_tasks(self) -> int:
        if not self._gateway_tasks:
            return self._completed_count
        return sum(
            1 for r in self._campaign_records() if self._record_counts_as_terminal(r)
        )

    @completed_tasks.setter
    def completed_tasks(self, _: int) -> None:
        pass

    @property
    def succeeded_tasks(self) -> int:
        if self._gateway_tasks is None:
            return 0
        return sum(
            1 for r in self._campaign_records() if r.state == self._STATE_SUCCEEDED
        )

    @property
    def failed_tasks(self) -> int:
        if self._gateway_tasks is None:
            return 0
        return sum(1 for r in self._campaign_records() if r.state == self._STATE_FAILED)

    @property
    def cancelled_tasks(self) -> int:
        if self._gateway_tasks is None:
            return 0
        return sum(
            1 for r in self._campaign_records() if r.state == self._STATE_CANCELLED
        )

    def _record_is_retry_superseded(self, record: Any) -> bool:
        """Return True for internal NF retry cleanup records.

        Superseded records retire an older gateway attempt for the same
        Nextflow logical task.  They are not real DAG progress, so they
        must not satisfy expected fan-out cardinality or make a campaign
        look complete while NF still has retry work outstanding.
        """
        try:
            if int(getattr(record, "state", 0) or 0) != self._STATE_CANCELLED:
                return False
            msg = str(getattr(record, "message", "") or "").lower()
            return "superseded" in msg or "nf_retry_supersede" in msg
        except Exception:
            return False

    def _record_counts_as_terminal(self, record: Any) -> bool:
        try:
            state = int(getattr(record, "state", 0) or 0)
        except Exception:
            return False
        if state in (self._STATE_SUCCEEDED, self._STATE_FAILED):
            return True
        if state == self._STATE_CANCELLED:
            return not self._record_is_retry_superseded(record)
        return False

    @property
    def active_task_ids(self) -> set[str]:
        if not self._gateway_tasks:
            return self._active_ids
        return {
            r.task_id
            for r in self._campaign_records()
            if self._record_counts_as_active(r)
        }

    def _record_counts_as_active(self, record: Any) -> bool:
        """True only after the dispatch lifecycle reached real execution.

        ``RUNNING`` + selected worker metadata is too early: retries,
        admission bounces, and SKIP_THIS_CYCLE exits can leave those
        fields populated even though compute never started.  The
        dispatch-time ``_dispatched_gpu_id`` marker is set immediately
        before gRPC inference begins and cleared on failed attempts /
        retry re-plans, making it the tighter live-activity signal.
        """
        try:
            task_id = str(getattr(record, "task_id", "") or "").strip()
            if not task_id:
                return False
            state = int(getattr(record, "state", 0) or 0)
            if state != self._STATE_RUNNING:
                return False
            return bool(str(getattr(record, "_dispatched_gpu_id", "") or "").strip())
        except Exception:
            return False

    def _record_counts_as_pending(self, record: Any) -> bool:
        """True while Nextflow is waiting but compute has not started.

        ``_run_task`` marks a TaskRecord RUNNING before it enters the
        planner/retry loop.  Until ``_dispatched_gpu_id`` is stamped, that
        task is not active compute, but it is still live pending work from
        Nextflow's perspective.  Counting it as pending prevents campaign
        views from briefly becoming ``pending=0, active=0`` during
        plan/retry attempts.
        """
        try:
            state = int(getattr(record, "state", 0) or 0)
            if state == self._STATE_SUBMITTED:
                return True
            return bool(
                state == self._STATE_RUNNING
                and not self._record_counts_as_active(record)
            )
        except Exception:
            return False

    def reconcile_live_counters(self) -> bool:
        """Refresh pending/active counters from authoritative TaskRecords.

        The hot path still uses explicit counters, but terminal callback
        loss or retry/supersede corner cases can leave ``_pending_count`` /
        ``_active_count`` stale.  Rebuilding the live-id sets from the
        per-campaign TaskRecord index gives us a bounded-cost safety net
        without returning to cluster-wide scans.
        """
        if not self._gateway_tasks:
            return False
        pending_ids: set[str] = set()
        active_ids: set[str] = set()
        for record in self._campaign_records():
            task_id = str(getattr(record, "task_id", "") or "").strip()
            if not task_id:
                continue
            if self._record_counts_as_pending(record):
                pending_ids.add(task_id)
            elif self._record_counts_as_active(record):
                active_ids.add(task_id)
        changed = (
            pending_ids != self._pending_ids
            or active_ids != self._active_ids
            or len(pending_ids) != self._pending_count
            or len(active_ids) != self._active_count
        )
        if changed:
            self._pending_ids = pending_ids
            self._active_ids = active_ids
            self._pending_count = len(pending_ids)
            self._active_count = len(active_ids)
            self._is_empty_slow_cache = None
        return changed

    def _tally_by_component(
        self,
    ) -> dict[str, tuple[int, int, str | None]]:
        """Plan fix (F-xx) + fix (F-xx) — single
        ``_campaign_records()`` scan that tallies ``(seen, done,
        first_fingerprint)`` for **every** component at once.

        F-xx replaced per-component ``_count_terminal_for`` /
        ``_component_complete`` calls (12-18 × O(M) per refresh).
        F-xx additionally subsumes ``_campaign_component_fingerprint``
        (Tier A lookup in ``refresh_remaining_est``) which, prior to
        this patch, scanned ``_campaign_records()`` **once per
        component** again — bench13 py-spy samples 2/4/6 pinned this
        path as the new dominant hot path after F-xx/F-xx eliminated the
        preceding scans.

        The ``first_fingerprint`` preserves the pre-patch
        ``_campaign_component_fingerprint`` semantics bit-exactly:
        first non-empty ``config_fingerprint`` encountered in dict
        insertion order (= task submission order, Python 3.7+
        guarantee) wins.  Empty / whitespace-only fingerprints are
        treated as absent (``str(...).strip() or None`` normalisation).

        Components without any dispatched task are absent from the
        mapping (caller uses ``tally.get(comp, (0, 0, None))`` default).
        """
        tally: dict[str, tuple[int, int, str | None]] = {}
        if not self._gateway_tasks:
            return tally
        for r in self._campaign_records():
            comp = getattr(r, "component", "")
            if not comp:
                continue
            seen, done, fp = tally.get(comp, (0, 0, None))
            seen += 1
            if self._record_counts_as_terminal(r):
                done += 1
            if fp is None:
                candidate = getattr(r, "config_fingerprint", None)
                if candidate:
                    normalised = str(candidate).strip() or None
                    if normalised:
                        fp = normalised
            tally[comp] = (seen, done, fp)
        return tally

    def _count_terminal_for(self, component: str) -> int:
        """Plan fix — count this campaign's TaskRecords for
        *component* whose state is terminal (SUCCEEDED / FAILED /
        CANCELLED).  Returns 0 when ``_gateway_tasks`` is not connected
        (unit-test shim path).  Prefer ``_tally_by_component`` on hot
        paths (see ``refresh_remaining_est``); this single-component
        helper remains for legacy callers.
        """
        if not self._gateway_tasks:
            return 0
        done = 0
        for r in self._campaign_records():
            if getattr(
                r, "component", ""
            ) == component and self._record_counts_as_terminal(r):
                done += 1
        return done

    def _component_complete(self, component: str) -> bool:
        """Plan fix — True iff every dispatched task of
        *component* in this campaign has reached a terminal state, AND
        at least one such task exists.  "seen==0" (not yet submitted)
        returns False so ``refresh_remaining_est`` keeps the stage in
        the sum for the future-wave case.  "seen>0, done<seen"
        (in-flight) also returns False.
        """
        if not self._gateway_tasks:
            return False
        seen = 0
        done = 0
        for r in self._campaign_records():
            if getattr(r, "component", "") == component:
                seen += 1
                if self._record_counts_as_terminal(r):
                    done += 1
        return seen > 0 and done == seen

    def _expected_total_for_component(self, component: str) -> int | None:
        """Expected task cardinality for *component* in this campaign.

        ``is_dag_complete`` must prove that Nextflow has no future tasks
        left to submit for this campaign.  A component merely being seen
        once is not enough in fan-out DAGs: between stages, ``pending``
        and ``active`` can legitimately drop to zero while Nextflow is
        still preparing more downstream tasks.  Mirror
        ``PipelineDAG.cumulative_fan_in_to`` using the campaign-local
        ``DAGContext`` so completion requires the expected number of
        terminal instances per component.
        """
        dag = self.dag_context
        if dag is None:
            return None
        upstream_map = getattr(dag, "upstream_map", {}) or {}
        fan_out_map = getattr(dag, "fan_out_map", {}) or {}

        def _join_type(comp: str) -> str:
            try:
                jt = getattr(dag, "join_type", None)
                if callable(jt):
                    return str(jt(comp) or "independent")
            except Exception:
                _LOG.debug("campaign join_type lookup failed", exc_info=True)
            try:
                return str(
                    (getattr(dag, "join_type_map", {}) or {}).get(comp, "independent")
                )
            except Exception:
                return "independent"

        def _walk(comp: str, visited: set[str]) -> int | None:
            key = str(comp or "").strip().lower()
            if key in visited:
                return 1
            visited.add(key)
            upstream = list(upstream_map.get(key, []) or [])
            if not upstream:
                return 1
            if _join_type(key) == "barrier":
                return 1
            parent = str(upstream[0] or "").strip().lower()
            parent_total = _walk(parent, visited)
            if parent_total is None:
                return None
            parent_fan = fan_out_map.get(parent)
            if parent_fan is None:
                return None
            try:
                parent_fan_i = int(round(float(parent_fan)))
            except Exception:
                return None
            if parent_fan_i < 1:
                return None
            return parent_total * parent_fan_i

        return _walk(component, set())

    def record_input_size(self, component: str, input_size: float) -> None:
        """Record an observed input_size for critical-path weight accuracy."""
        if input_size > 0:
            self._observed_input_sizes.setdefault(component, []).append(input_size)

    def avg_input_size(self, component: str) -> float:
        """Average observed input_size for a component (0.0 if none)."""
        sizes = self._observed_input_sizes.get(component, [])
        return sum(sizes) / len(sizes) if sizes else 0.0

    def effective_fan_out(self, component: str) -> float | None:
        """Best fan-out estimate: DAG hint or None.

        Plan fix retired the legacy "observed median"
        fallback — fan-out source-of-truth moved to
        ``PipelineDAG._campaign_fan_out`` (populated by
        ``register_fan_out`` from ``ext.output_sample_count`` on every
        task submit) and surfaces here through
        ``cq.dag_context.fan_out_map``.  A separate per-campaign
        ``observed_fan_outs`` accumulator was kept for one cycle as a
        secondary fallback but its writer
        ``record_fan_out_observation`` was never wired to any caller,
        so the fallback never fired.  Cleanup removes both the field
        and the fallback branch — the DAG-hint path covers every
        scheduler query.
        """
        if self.dag_context:
            hint = self.dag_context.fan_out_map.get(component)
            if hint is not None:
                return _safe_float(hint)
        return None

    @property
    def is_empty(self) -> bool:
        if (self._pending_count + self._active_count) > 0:
            return False
        if (
            self._workflow_completion_expected
            and not self._workflow_completion_received
        ):
            return False
        if self._gateway_tasks:
            cache_key = (id(self._gateway_tasks), len(self._gateway_tasks))
            cached = self._is_empty_slow_cache
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            result = self.pending_tasks <= 0 and self.active_tasks <= 0
            self._is_empty_slow_cache = (cache_key, result)
            return result
        return self.pending_tasks <= 0 and self.active_tasks <= 0

    @property
    def is_dag_complete(self) -> bool:
        """Best-effort ops "done" signal based on gateway liveness.

        Do not use ``dag_context`` as an authoritative completion proof.
        The gateway's component DAG is global/static and Nextflow's topology
        payload is only partial precedence information; branch choices and
        optional stages are campaign-local runtime facts.  Scheduler primary
        hand-off therefore follows live counters, while this property simply
        reports whether all gateway-visible work for the campaign has reached
        successful terminal states.
        """
        if self.pending_tasks > 0 or self.active_tasks > 0:
            return False
        if self._workflow_completion_expected:
            return self._workflow_completion_received and self._workflow_completion_ok
        if not self._gateway_tasks:
            return self._completed_count > 0
        if self.failed_tasks > 0 or self.cancelled_tasks > 0:
            return False
        return self.succeeded_tasks > 0

    def refresh_remaining_est(
        self,
        signal_service: Any = None,
        now: float | None = None,
        *,
        scheduler: Any = None,
    ) -> float | None:
        """Plan — recompute ``_remaining_est_cache``.

        ``remaining_est = Σ_{stage s ∈ DAG, not complete} μ_lat(s) × waves(s)``.

        Returns the cached value (also stored on the instance).  Returns
        ``None`` — which triggers the FIFO fallback in ``SmithRuleCampaign``
        — only when DAG context is absent or no component has any GP
        observation across the entire GPU pool.

        **Whole-cluster GP maturity semantics**: this routine queries the
        ``ResourceProfileRegistry.predict_latency(component, config="",
        gpu_id=None)`` accessor which already performs the per-GPU →
        cross-GPU-pool fallback (resource_profile.py
        ``ConfigProfile.predict_latency``).  A single completed task on
        any GPU seeds the pool, so WSJF priorities become usable as soon
        as any observation exists cluster-wide — not per-GPU.  Previously
        this routine called a non-existent ``signal_service.query_mu_variance``
        (→ None) and then a wrong-signature ``predict_cost(rp, comp, "",
        0)`` fallback (→ exception), producing ``_remaining_est_cache =
        None`` for every campaign and forcing ``SmithRuleCampaign`` into
        its FIFO fallback ranking for the entire cold-start window.
        """
        dag = self.dag_context
        if dag is None or signal_service is None:
            self._remaining_est_cache = None
            return None
        try:
            components = list(
                getattr(dag, "component_order", None) or getattr(dag, "components", [])
            )
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "components = []",
                exc_info=True,
            )
            components = []
        if not components:
            self._remaining_est_cache = None
            return None

        pipeline_dag = getattr(scheduler, "_pipeline_dag", None) if scheduler else None
        if pipeline_dag is not None:
            try:
                eager_complete = bool(
                    getattr(
                        pipeline_dag,
                        "is_eager_component_mapping_complete",
                        False,
                    )
                )
            except Exception:
                _LOG.warning(
                    "[remaining-est] swallowed an exception reading "
                    "is_eager_component_mapping_complete — treating as False",
                    exc_info=True,
                )
                eager_complete = False
            if not eager_complete:
                self._remaining_est_cache = None
                return None
        rp = getattr(signal_service, "resource_profiles", None)
        if rp is None:
            self._remaining_est_cache = None
            return None
        pipeline_dag = getattr(scheduler, "_pipeline_dag", None) if scheduler else None
        fan_out_resolver: Callable[[str], float | None] | None = None
        if scheduler is not None and hasattr(
            scheduler, "effective_fan_out_with_global"
        ):

            def _resolve_fan_out(parent: str) -> float | None:
                return scheduler.effective_fan_out_with_global(
                    self.campaign_id,
                    parent,
                )

            fan_out_resolver = _resolve_fan_out
        parallelism = 1
        if scheduler is not None and hasattr(scheduler, "_system_parallelism"):
            try:
                parallelism = max(1, int(scheduler._system_parallelism()))
            except Exception:
                parallelism = 1

        tally = self._tally_by_component()

        remaining = 0.0
        for comp in components:
            seen, done, campaign_fp = tally.get(comp, (0, 0, None))
            if seen > 0 and done == seen:
                continue

            expected_total: int | None = None
            if pipeline_dag is not None and fan_out_resolver is not None:
                try:
                    expected_total = pipeline_dag.cumulative_fan_in_to(
                        comp,
                        fan_out_resolver,
                    )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s cumulative_fan_in_to raised; falling back to local fan_out",
                        __name__,
                        exc_info=True,
                    )
                    expected_total = None

            in_flight = max(0, seen - done)
            if expected_total is not None and expected_total > 0:
                remaining_inst = max(expected_total - done, in_flight)
            else:
                fan_out = self.effective_fan_out(comp) or 1.0
                remaining_inst = max(fan_out - done, in_flight)
            if remaining_inst <= 0:
                continue

            profile = rp._profiles.get(comp)
            if profile is None:
                self._remaining_est_cache = None
                return None

            input_size = self.avg_input_size(comp)
            if input_size <= 0 and scheduler is not None:
                try:
                    input_size = float(
                        scheduler._cluster_avg_input_size(comp) or 0.0,
                    )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed cluster_avg_input_size; body=%s",
                        __name__,
                        "input_size=0.0",
                        exc_info=True,
                    )
                    input_size = 0.0
            if input_size <= 0:
                self._remaining_est_cache = None
                return None

            mu: float | None = None
            if campaign_fp:
                try:
                    cfg = profile._config_baselines.get(campaign_fp)
                    if cfg is not None:
                        result = cfg.predict_latency(input_size, gpu_id=None)
                        if result is not None and result > 0:
                            mu = float(result)
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed Tier-A predict_latency; body=%s",
                        __name__,
                        "mu = None",
                        exc_info=True,
                    )
            if mu is None:
                try:
                    for cfg in profile._config_baselines.values():
                        result = cfg.predict_latency(input_size, gpu_id=None)
                        if result is not None and result > 0:
                            mu = float(result)
                            break
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s swallowed Tier-B predict_latency; body=%s",
                        __name__,
                        "mu = None",
                        exc_info=True,
                    )
            if mu is None or mu <= 0:
                self._remaining_est_cache = None
                return None
            stage_waves = math.ceil(remaining_inst / parallelism)
            remaining += mu * stage_waves
        self._remaining_est_cache = remaining
        return self._remaining_est_cache

    def _campaign_component_fingerprint(self, component: str) -> str | None:
        """Return the ``config_fingerprint`` this campaign has actually
        dispatched for *component* (Tier A lookup for remaining_est).

        Scans the gateway task records for the first task of this
        campaign with a non-empty ``config_fingerprint`` matching
        *component*.  Returns None when the campaign has not yet
        dispatched that component — forces Tier B / C fallback.

        Plan fix (F-xx) — Prefer ``_tally_by_component`` on
        hot paths (``refresh_remaining_est`` reads ``campaign_fp`` from
        the tally tuple directly).  This single-component helper remains
        for legacy / test-path callers; bit-exact first-match semantics
        identical to the single-pass tally.
        """
        if not self._gateway_tasks:
            return None
        for r in self._campaign_records():
            if getattr(r, "component", "") != component:
                continue
            fp = getattr(r, "config_fingerprint", None)
            if fp:
                return str(fp).strip() or None
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "arrival_time": round(self.arrival_time, 3),
            "pending_tasks": self.pending_tasks,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "succeeded_tasks": self.succeeded_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "remaining_est_sec": self._remaining_est_cache,
            "is_dag_complete": self.is_dag_complete,
            "workflow_completion_expected": self._workflow_completion_expected,
            "workflow_completion_received": self._workflow_completion_received,
            "workflow_completion_ok": self._workflow_completion_ok,
            "workflow_completion_message": self._workflow_completion_message,
            "fan_out_assignments": dict(self.fan_out_assignments),
        }




@dataclass
class CampaignPlacement:
    """Result of campaign-level scheduling decision."""

    is_backfill: bool = False
    preferred_gpu_ids: list[str] | None = None
    scheduling_hints: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_backfill": self.is_backfill,
            "preferred_gpu_ids": self.preferred_gpu_ids,
            "scheduling_hints": dict(self.scheduling_hints),
        }




class CampaignScheduler:
    """Campaign-level orchestrator sitting above PlannerService.

    Thread-safety: all mutation methods (on_task_dispatched, on_task_complete,
    _on_scenario_changed) are synchronous — they run to completion within a
    single event loop tick, preventing interleaving.  No lock is needed.
    """

    _MAX_SUBSET_SIZE = 3
    _speculative_task_projection_disabled: bool = False

    def __init__(
        self,
        signal_service: Any = None,
        init_tracker: Any = None,
        pipeline_dag: Any = None,
        gateway_tasks: Any = None,
        default_init_sec: float = 10.0,
    ) -> None:
        self._signal_service = signal_service
        self._init_tracker = init_tracker
        self._pipeline_dag = pipeline_dag
        self._gateway_tasks = gateway_tasks
        self._default_init_sec = default_init_sec
        self._campaign_queues: dict[str, CampaignQueue] = {}
        self._timelines = SchedulingScenario()

        self._resource_tracker: Any = None

        self._evict_fn: Callable | None = None
        self._pre_init_fn: Callable | None = None

        self._pending_pre_inits: dict[
            tuple[str, str, str],
            tuple[float, str, str],
        ] = {}
        self._fired_pre_inits: set[tuple[str, str]] = set()
        self._pending_cancel_tasks: set[asyncio.Task[Any]] = set()

        self._pre_init_disabled: bool = False
        self._downstream_projection_disabled: bool = False

        self._last_admission_cost: float = 0.0
        self._last_admission_benefit: float = 0.0
        self._last_admission_result: bool = True


        self._tasks_by_campaign: dict[str, dict[str, Any]] = {}
        self._pending_workflow_completions_by_run_name: dict[str, tuple[bool, str]] = {}
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._main_thread_id: int | None = None
        self.drift_applied_epoch: int = 0
        self._pending_profile_drift_batch: dict[str, dict[str, Any]] = {}
        self._profile_drift_publish_handle: asyncio.Handle | None = None
        self._profile_drift_coalesce_sec: float = 0.50
        self._last_live_counter_reconcile_monotonic: float = 0.0
        self._live_counter_reconcile_interval_sec: float = 2.0
        self._resource_upper_percentile: float = 0.95
        self._resource_upper_z: float = self._z_from_percentile(
            self._resource_upper_percentile
        )
        self._host_meminfo_cache_ttl_sec: float = 1.0
        self._host_meminfo_cache_at: float = 0.0
        self._host_meminfo_cache: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _z_from_percentile(percentile: Any) -> float:
        try:
            p = float(percentile)
        except (TypeError, ValueError):
            p = 0.95
        if not math.isfinite(p):
            p = 0.95
        if 1.0 < p <= 100.0:
            p /= 100.0
        if not 0.5 <= p < 1.0:
            p = 0.95
        from .global_planner import _norm_ppf

        return _safe_float(_norm_ppf(p))

    def set_resource_upper_percentile(self, percentile: Any) -> None:
        """Configure the one-sided GP percentile for VRAM/RAM reservations."""
        try:
            p = float(percentile)
        except (TypeError, ValueError):
            p = 0.95
        if not math.isfinite(p):
            p = 0.95
        if 1.0 < p <= 100.0:
            p /= 100.0
        if not 0.5 <= p < 1.0:
            _LOG.warning(
                "[campaign-scheduler] invalid resource_upper_percentile=%r; "
                "falling back to 0.95",
                percentile,
            )
            p = 0.95
        self._resource_upper_percentile = p
        self._resource_upper_z = self._z_from_percentile(p)
        if hasattr(self, "_timelines"):
            self._timelines.set_vram_reservation_model(
                getattr(self._timelines, "_vram_reservation_model_name", "full_wall"),
                signal_service=self._signal_service,
                resource_z=self._resource_upper_z,
            )

    def set_vram_reservation_model(self, name: str | None) -> None:
        """Select the shared pluggable VRAM reservation model."""
        self._timelines.set_vram_reservation_model(
            name,
            signal_service=self._signal_service,
            resource_z=self._resource_upper_z,
        )

    @property
    def resource_upper_percentile(self) -> float:
        return _safe_float(getattr(self, "_resource_upper_percentile", 0.95))

    @property
    def resource_upper_z(self) -> float:
        return _safe_float(getattr(self, "_resource_upper_z", 1.645))

    def _capture_main_loop(self, *, replace: bool = False) -> None:
        current = getattr(self, "_main_loop", None)
        if not replace and current is not None and not current.is_closed():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._main_loop = loop
        self._main_thread_id = threading.get_ident()


    @classmethod
    def create(cls, policy_name: str, **kwargs: Any) -> CampaignScheduler:
        """Create a CampaignScheduler instance.

        3-layer architecture: scheduling policies are now config-driven
        via GlobalPlanner's pluggable strategies (runtime.*.yaml
        global_planner section).  The subclass hierarchy and
        CAMPAIGN_SCHEDULER_REGISTRY have been removed.  All policy
        names map to the base CampaignScheduler.
        """
        if policy_name and policy_name != "campaign_fifo":
            _LOG.info(
                "[campaign-scheduler] policy_name=%s → base CampaignScheduler "
                "(policies now config-driven via GlobalPlanner)",
                policy_name,
            )
        return cls(**kwargs)


    def _get_or_create(self, campaign_id: str) -> CampaignQueue:
        cid = str(campaign_id or "").strip()
        if not cid:
            cid = f"__anon_{time.time():.6f}"
        cq = self._campaign_queues.get(cid)
        if cq is None:
            cq = CampaignQueue(
                campaign_id=cid,
                arrival_time=time.time(),
                _gateway_tasks=self._gateway_tasks,
                _tasks_by_campaign_ref=getattr(
                    self,
                    "_tasks_by_campaign",
                    None,
                ),
            )
            self._campaign_queues[cid] = cq
            _LOG.info(
                "[campaign-scheduler] new campaign %s (total=%d)",
                cid,
                len(self._campaign_queues),
            )
        return cq

    def note_task_submitted(self, record: Any) -> None:
        """Plan fix — register a submitted TaskRecord in
        the per-campaign index.  Called by ``GatewayHTTPService.
        _submit_task_impl`` right after ``self._tasks[task_id] =
        record`` so that ``CampaignQueue._campaign_records()`` can do
        O(1) dispatch instead of scanning ``_gateway_tasks`` (~288
        records under concurrency=9 bench).

        Idempotent — same ``task_id`` re-registration overwrites the
        entry, matching ``_gateway_tasks[task_id] = record`` semantics.
        Missing ``campaign_id`` / ``task_id`` is a no-op (anonymous
        campaigns are created lazily in ``_get_or_create`` with a
        ``__anon_`` prefix and their records participate in the index
        only if the submitter provides a valid id).
        """
        self._capture_main_loop(replace=True)
        cid = str(getattr(record, "campaign_id", "") or "").strip()
        tid = str(getattr(record, "task_id", "") or "").strip()
        if not cid or not tid:
            return
        self._tasks_by_campaign.setdefault(cid, {})[tid] = record
        cq = self._get_or_create(cid)
        if bool(getattr(record, "workflow_completion_expected", False)):
            cq._workflow_completion_expected = True
        rn = str(getattr(record, "run_name", "") or "").strip()
        if rn and rn in self._pending_workflow_completions_by_run_name:
            ok, message = self._pending_workflow_completions_by_run_name.pop(rn)
            self._apply_external_completion(cq, ok=ok, message=message)

    def _apply_external_completion(
        self,
        cq: CampaignQueue,
        *,
        ok: bool,
        message: str = "",
    ) -> None:
        cq._workflow_completion_expected = True
        cq._workflow_completion_received = True
        cq._workflow_completion_ok = bool(ok)
        cq._workflow_completion_message = str(message or "")
        cq._is_empty_slow_cache = None

    def mark_external_completion(
        self,
        *,
        campaign_id: str = "",
        run_name: str = "",
        ok: bool = False,
        message: str = "",
    ) -> list[str]:
        """Mark campaigns terminal from an external workflow-level signal.

        This is intentionally separate from task completion.  Nextflow can
        have inter-stage gaps with no gateway-visible tasks; when the launcher
        or plugin provides a workflow terminal signal, primary hand-off waits
        for that signal instead of trusting static DAG inference.
        """
        cid = str(campaign_id or "").strip()
        rn = str(run_name or "").strip()
        targets: set[str] = set()
        if cid and cid in self._campaign_queues:
            targets.add(cid)
        if rn:
            for candidate_id, records in self._tasks_by_campaign.items():
                for record in records.values():
                    if str(getattr(record, "run_name", "") or "").strip() == rn:
                        targets.add(candidate_id)
                        break
            if not targets and self._gateway_tasks:
                for record in self._gateway_tasks.values():
                    if str(getattr(record, "run_name", "") or "").strip() == rn:
                        candidate_id = str(
                            getattr(record, "campaign_id", "") or ""
                        ).strip()
                        if candidate_id:
                            targets.add(candidate_id)
        if not targets and cid:
            targets.add(cid)
        matched: list[str] = []
        for target_id in sorted(targets):
            cq = self._get_or_create(target_id)
            self._apply_external_completion(cq, ok=ok, message=message)
            matched.append(target_id)
        if rn and not matched:
            self._pending_workflow_completions_by_run_name[rn] = (
                bool(ok),
                str(message or ""),
            )
            _LOG.warning(
                "[campaign-scheduler] deferred workflow completion for "
                "unmatched run_name=%s; it will be applied on first task submit",
                rn,
            )
        self._on_scenario_changed([])
        self._notify_supervisor_wake("workflow_completion")
        return matched

    def _cluster_avg_input_size(self, component: str) -> float:
        """Plan input_size Tier-2 — mean observed ``input_size`` for
        *component* pooled across **every** campaign queue.

        Used by ``CampaignQueue.refresh_remaining_est`` when this
        campaign has not yet dispatched the component itself (i.e. its
        own ``avg_input_size`` returns 0.0).  Returns 0.0 when no
        campaign anywhere has observed the component — callers treat
        that as Tier-3 "skip GP query → FIFO fallback".
        """
        totals: list[float] = []
        for cq in self._campaign_queues.values():
            totals.extend(cq._observed_input_sizes.get(component, []))
        if not totals:
            return 0.0
        return sum(totals) / len(totals)

    def set_campaign_strategy(self, strategy: Any) -> None:
        """Plan Integration — bind the Planner-injected
        ``CampaignStrategy`` so ``_primary_campaign`` routes through it.
         fix: only ``FIFOCampaign`` remains; the WSPT-style
        ``SmithRuleCampaign`` / ``WSJFCampaign`` were removed because
        their score (``age / remaining_est``) had no formal optimality
        bound (Plan fix honesty label) yet drove ~12-15%
        of main-thread CPU through ``refresh_remaining_est`` per drift
        cascade.  ``_primary_campaign`` still routes through whatever
        strategy is bound (test injection support), but in production
        only FIFO is ever selected.
        """
        self._campaign_strategy = strategy

    def _primary_campaign(self) -> CampaignQueue | None:
        """Return the highest-priority active campaign.

        Delegates to the injected ``CampaignStrategy.select_primary`` when
        one has been bound (see ``set_campaign_strategy``); otherwise
        falls back to earliest-arrival FIFO for callers that construct
        ``CampaignScheduler`` directly (test paths, legacy harnesses).
        """
        self._reconcile_campaign_live_counters()
        strategy = getattr(self, "_campaign_strategy", None)
        if strategy is not None:
            try:
                cq = strategy.select_primary(self._campaign_queues)
                if cq is not None:
                    return cq
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] campaign strategy raised; "
                    "falling back to FIFO primary selection",
                    exc_info=True,
                )
        active = [cq for cq in self._campaign_queues.values() if not cq.is_empty]
        if not active:
            return None
        return min(active, key=lambda cq: cq.arrival_time)

    def _is_primary(self, campaign_id: str) -> bool:
        planner = getattr(self, "_planner_ref", None)
        if planner is not None and getattr(planner, "_cached_primary_id_set", False):
            return planner._cached_primary_id == campaign_id
        primary = self._primary_campaign()
        return primary is not None and primary.campaign_id == campaign_id

    def _ordered_campaigns(self) -> list[CampaignQueue]:
        """All active campaigns sorted by arrival time."""
        return sorted(
            [cq for cq in self._campaign_queues.values() if not cq.is_empty],
            key=lambda cq: cq.arrival_time,
        )

    def is_needed_by_primary(self, component: str) -> bool:
        """Check if primary campaign has pending/active work for *component*.

        Used by recovery to prioritize workers that primary campaign needs.
        Checks:
          1. Active timeline entries (actual or predicted) for this component
             belonging to the primary campaign.
          2. Pending/active TaskRecords for this component in primary campaign.
        """
        primary = self._primary_campaign()
        if not primary:
            return True
        cid = primary.campaign_id
        for gpu_id in self._timelines.gpu_ids:
            tl = self._timelines.get(gpu_id)
            if not tl:
                continue
            for e in tl.entries_for_component(component):
                if e.campaign_id == cid and not e.is_completed:
                    return True
        if primary._gateway_tasks:
            for r in primary._gateway_tasks.values():
                if (
                    getattr(r, "campaign_id", "") == cid
                    and getattr(r, "component", "") == component
                    and getattr(r, "state", 0)
                    in (primary._STATE_SUBMITTED, primary._STATE_RUNNING)
                ):
                    return True
        return False


    def sync_gpu_pool(
        self,
        vram_totals: dict[str, int],
        reserved_weights: dict[str, int] | None = None,
        idle_weights: dict[str, int] | None = None,
        idle_ram: float = 0.0,
    ) -> None:
        """Sync GPU pool VRAM capacities and idle resources from ResourceAdmissionTracker."""
        self._timelines.sync_vram_totals(vram_totals)
        if reserved_weights is not None:
            self._timelines.sync_reserved_weights(reserved_weights)
        if idle_weights is not None:
            self._timelines.sync_idle_weights(idle_weights)
        self._timelines.sync_idle_ram(idle_ram)
        self._sync_host_ram_capacity()

    def _read_host_meminfo_cached(
        self, force_refresh: bool = False
    ) -> tuple[float, float]:
        """Read host /proc/meminfo fields with a short monotonic TTL cache."""
        now = time.monotonic()
        ttl = _safe_float(getattr(self, "_host_meminfo_cache_ttl_sec", 0.0) or 0.0)
        last_refresh = _safe_float(getattr(self, "_host_meminfo_cache_at", 0.0) or 0.0)
        if (
            not force_refresh
            and ttl > 0.0
            and self._host_meminfo_cache != (0.0, 0.0)
            and (now - last_refresh) < ttl
        ):
            return self._host_meminfo_cache

        snapshot = (_read_mem_total_mib(), _read_mem_available_mib())
        self._host_meminfo_cache = snapshot
        self._host_meminfo_cache_at = now
        return snapshot

    def _sync_host_ram_capacity(self) -> None:
        """Sync host-wide RAM capacity for planner-side RAM reservations."""
        min_available = 0.0
        sup = getattr(self, "_supervisor", None)
        if sup is not None:
            min_available = _safe_float(
                getattr(sup, "admission_host_ram_min_available_mib", 0) or 0
            )
        elif self._resource_tracker is not None:
            min_available = _safe_float(
                getattr(self._resource_tracker, "_host_ram_min_available_mib", 0) or 0
            )
        total_ram_mb, mem_available_mb = self._read_host_meminfo_cached()
        self._timelines.sync_host_ram(
            total_ram_mb=total_ram_mb,
            min_available_mb=min_available,
            mem_available_mb=mem_available_mb,
            active_reserved_mb=self._timelines.reserved_host_ram_at(),
        )

    def _reconcile_campaign_live_counters(self, *, force: bool = False) -> int:
        """Safety-net refresh for per-campaign pending/active counters."""
        if not hasattr(self, "_gateway_tasks") or self._gateway_tasks is None:
            return 0
        now = time.monotonic()
        last = _safe_float(
            getattr(self, "_last_live_counter_reconcile_monotonic", 0.0) or 0.0
        )
        interval = _safe_float(
            getattr(self, "_live_counter_reconcile_interval_sec", 2.0) or 0.0
        )
        if not force and interval > 0.0 and (now - last) < interval:
            return 0
        self._last_live_counter_reconcile_monotonic = now
        changed = 0
        for cq in self._campaign_queues.values():
            try:
                if cq.reconcile_live_counters():
                    changed += 1
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] live-counter reconciliation failed "
                    "for campaign=%s",
                    cq.campaign_id,
                    exc_info=True,
                )
        if changed:
            _LOG.info(
                "[campaign-scheduler] reconciled live counters for %d campaign(s)",
                changed,
            )
        return changed

    def _system_parallelism(self) -> int:
        """Plan fix — nominal system parallelism (GPU count).

        Used by ``CampaignQueue.refresh_remaining_est`` to convert a
        stage's remaining instance count into a wall-clock wave count
        via ``ceil(remaining_inst / parallelism)``.  Cross-campaign
        contention is ignored here; the resulting absolute ETA is
        optimistic but every campaign shares the same divisor so WSJF
        relative ordering stays invariant.  Falls back to 1 when the
        GPU pool hasn't been initialised yet (very early cold-start).
        """
        count = len(list(self._timelines.gpu_ids or []))
        return max(1, count)

    def _ensure_gpu_pool(self) -> None:
        """Ensure GPU timelines have total_vram populated from ResourceAdmissionTracker."""
        if self._timelines.gpu_ids and all(
            (self._timelines.get(g) or GpuTimeline("", 0)).total_vram_mb > 0
            for g in self._timelines.gpu_ids
        ):
            self._sync_host_ram_capacity()
            return
        if self._resource_tracker:
            try:
                totals = dict(self._resource_tracker.total_vram or {})
                if not totals:
                    pool = getattr(self._resource_tracker, "gpu_pool", [])
                    if pool:
                        totals = {str(gid): gpu_vram_fallback_mib() for gid in pool}
                if totals:
                    self._timelines.sync_vram_totals(totals)
                    self._sync_host_ram_capacity()
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )

    def _reconcile_actual_timeline_entries(
        self,
        *,
        grace_period_sec: float = 30.0,
    ) -> int:
        """Safety-net reconcile for stale actual timeline entries.

        Predicted entries already have Layer-1/2/3 GC. Actual entries rely on
        task terminal callbacks; if one is missed, planned occupancy and ops
        timelines can drift indefinitely. We use authoritative gateway task
        state to reconcile those entries back to completed/killed.
        """
        if self._gateway_tasks is None:
            return 0
        try:
            gateway_task_items = list(self._gateway_tasks.items())
            task_states = {
                str(task_id): int(getattr(record, "state", 0) or 0)
                for task_id, record in gateway_task_items
            }
            task_updated_at = {
                str(task_id): float(getattr(record, "updated_at", 0.0) or 0.0)
                for task_id, record in gateway_task_items
            }
            task_completion_pending = {
                str(task_id)
                for task_id, record in gateway_task_items
                if bool(getattr(record, "_terminal_callback_pending", False))
            }
            return self._timelines.reconcile_actual_entries(
                task_states=task_states,
                task_updated_at=task_updated_at,
                task_completion_pending=task_completion_pending,
                state_succeeded=CampaignQueue._STATE_SUCCEEDED,
                state_failed=CampaignQueue._STATE_FAILED,
                state_cancelled=CampaignQueue._STATE_CANCELLED,
                state_running=CampaignQueue._STATE_RUNNING,
                state_submitted=CampaignQueue._STATE_SUBMITTED,
                grace_period_sec=grace_period_sec,
            )
        except Exception:
            _LOG.warning(
                "[campaign-scheduler] actual-entry reconciliation failed",
                exc_info=True,
            )
            return 0

    @property
    def timelines(self) -> SchedulingScenario:
        return self._timelines


    def on_task_submit(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float = 0.0,
        gpu_ids: list[str] | None = None,
    ) -> CampaignPlacement:
        """Called before ``generate_plan`` to enrich the task with campaign context.

        Returns a ``CampaignPlacement`` that the caller uses to set
        ``preferred_gpu_ids`` and ``scheduling_hints`` on the
        ``PlannerTaskRequest``.
        """
        self._capture_main_loop()
        self._ensure_gpu_pool()
        self._reconcile_actual_timeline_entries()
        cq = self._get_or_create(campaign_id)
        cq.record_input_size(component, input_size)
        if self._pipeline_dag and self._pipeline_dag.is_topology_registered:
            cq.dag_context = self._pipeline_dag.get_dag_context(campaign_id)
        newly_pending = task_id not in cq._pending_ids and task_id not in cq._active_ids
        if newly_pending:
            cq._pending_count += 1
            cq._pending_ids.add(task_id)
        is_primary = self._is_primary(campaign_id)
        is_backfill = not is_primary and len(self._ordered_campaigns()) > 1

        placement = CampaignPlacement(
            is_backfill=is_backfill,
            scheduling_hints={
                "is_backfill": is_backfill,
                "_campaign_arrival": cq.arrival_time,
            },
        )

        assignment = cq.fan_out_assignments.get(component)
        if assignment:
            idx = cq.fan_out_counters.get(component, 0)
            cq.fan_out_counters[component] = idx + 1
            placement.preferred_gpu_ids = [assignment[idx % len(assignment)]]

        existing_projection = self._timelines.find_entry(task_id)
        has_pending_projection = (
            existing_projection is not None
            and existing_projection.is_predicted
            and not existing_projection.is_completed
            and not existing_projection.is_dispatching
        )
        if newly_pending or not has_pending_projection:
            projected_gpu = self._project_task(
                task_id,
                campaign_id,
                component,
                input_size,
                is_backfill,
            )
        else:
            projected_gpu = (
                existing_projection.gpu_id if existing_projection is not None else None
            )
        if projected_gpu and not placement.preferred_gpu_ids and not is_backfill:
            placement.preferred_gpu_ids = [projected_gpu]


        downstream_key = f"{campaign_id}:{component}"
        if not hasattr(self, "_downstream_projected"):
            self._downstream_projected: set[str] = set()
        if downstream_key not in self._downstream_projected:
            self._downstream_projected.add(downstream_key)
            self._project_downstream(
                campaign_id, component, input_size, is_backfill, projected_gpu
            )

        return placement

    def on_task_dispatched(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
        input_size: float = 0.0,
        config_fingerprint: str = "",
        is_backfill: bool = False,
        was_cold_start: bool = False,
        actual_init_sec: float = 0.0,
        worker_name: str = "",
        active_cancel_safe: bool = False,
    ) -> None:
        """Called after a task is successfully dispatched to a worker.

        Adds a timeline entry so lookahead can see this task.
        ``was_cold_start``: True if the worker went through cold start.
        ``actual_init_sec``: measured init time (0 if not cold start).
        """
        self._capture_main_loop()
        _LOG.info(
            "[campaign-scheduler] on_task_dispatched task=%s component=%s gpu=%s",
            task_id[:12],
            component,
            gpu_id,
        )
        cq = self._campaign_queues.get(campaign_id or "")
        if cq is not None and component:
            cq._dispatched_components.add(component)
        if cq:
            newly_active = task_id not in cq._active_ids
            if newly_active:
                cq._active_count += 1
                cq._active_ids.add(task_id)
            if task_id in cq._pending_ids:
                cq._pending_ids.discard(task_id)
                cq._pending_count = max(0, cq._pending_count - 1)
            elif newly_active and cq._pending_count > 0:
                cq._pending_count = max(0, cq._pending_count - 1)

        config_fp = str(config_fingerprint or "").strip()
        record = self._gateway_tasks.get(task_id) if self._gateway_tasks else None
        if input_size <= 0 or not config_fp:
            try:
                if record is not None:
                    if not config_fp:
                        config_fp = str(
                            getattr(record, "config_fingerprint", "") or ""
                        ).strip()
                    if input_size <= 0:
                        from ..signals.resource_profile import ResourceProfileRegistry

                        input_size = ResourceProfileRegistry.extract_input_size(
                            dict(getattr(record, "workload_features", {}) or {}),
                            component=component,
                        )
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] dispatch prediction context lookup failed "
                    "task=%s component=%s",
                    task_id[:12],
                    component,
                    exc_info=True,
                )
        if input_size <= 0:
            try:
                input_size = cq.avg_input_size(component) if cq is not None else 0.0
                if input_size <= 0:
                    input_size = self._cluster_avg_input_size(component)
            except Exception:
                input_size = 0.0

        now = time.time()
        batch_context = getattr(record, "_dynamic_batch_context", None)
        execution_profile = (
            batch_context.get("execution_profile")
            if isinstance(batch_context, Mapping)
            else None
        )
        if isinstance(execution_profile, Mapping):
            predicted_latency = _safe_float(execution_profile.get("latency_mean"))
            predicted_vram = _safe_float(execution_profile.get("vram_mean"))
            predicted_ram = _safe_float(execution_profile.get("ram_mean"))
        else:
            predicted_latency = (
                self._predict_latency(
                    component,
                    input_size,
                    gpu_id,
                    config_fingerprint=config_fp,
                )
                if gpu_id
                else None
            ) or 0.0
            predicted_vram = (
                self._predict_vram(
                    component,
                    input_size,
                    gpu_id,
                    config_fingerprint=config_fp,
                )
                if gpu_id
                else 0.0
            )
            predicted_ram = (
                self._predict_ram(
                    component,
                    input_size,
                    gpu_id,
                    config_fingerprint=config_fp,
                )
                if gpu_id
                else 0.0
            )

        cid_prefix = (campaign_id or "")[:8]
        removed = self._timelines.remove_all_entries(
            f"__lookahead_{cid_prefix}_{component}"
        )
        removed += self._timelines.remove_all_entries(
            f"__reproject_{cid_prefix}_{component}"
        )

        if not gpu_id:
            removed += self._timelines.remove_all_entries(task_id)
            return

        tl = self._timelines.get(gpu_id)
        if tl:
            entries = (
                tl.planned_occupancy_entries()
                if hasattr(tl, "planned_occupancy_entries")
                else tl.active_entries
            )
            active_count = sum(
                1 for e in entries if e.component == component and not e.is_init
            )
            max_concurrent = self._get_worker_concurrency(component, gpu_id)
            if max_concurrent > 0 and active_count >= max_concurrent:
                _LOG.info(
                    "[campaign-scheduler] GPU %s has %d/%d active %s — "
                    "entry added anyway (concurrency enforced at compute slot) for %s",
                    gpu_id,
                    active_count,
                    max_concurrent,
                    component,
                    task_id[:12],
                )

        try:
            _actual_was_primary = bool(self._is_primary(campaign_id or ""))
        except Exception:
            _LOG.warning(
                "[silent-except] %s swallowed an exception; body=%s",
                __name__,
                "_actual_was_primary = False",
                exc_info=True,
            )
            _actual_was_primary = False

        if _actual_was_primary and gpu_id:
            planner = getattr(self, "_planner_ref", None)
            tracker = getattr(planner, "_constraint_tracker", None) if planner else None
            if tracker is not None:
                config_fp = ""
                try:
                    record = (
                        self._gateway_tasks.get(task_id)
                        if self._gateway_tasks
                        else None
                    )
                    config_fp = str(
                        getattr(record, "config_fingerprint", "") or ""
                    ).strip()
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s config_fingerprint lookup failed "
                        "task_id=%s",
                        __name__,
                        task_id[:12],
                        exc_info=True,
                    )
                    config_fp = ""
                try:
                    tracker.mark_primary_dispatched(
                        component,
                        gpu_id,
                        config_fingerprint=config_fp,
                    )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s mark_primary_dispatched failed "
                        "component=%s gpu=%s",
                        __name__,
                        component,
                        gpu_id,
                        exc_info=True,
                    )
        logical_batch_size = (
            _safe_int(batch_context.get("logical_batch_size") or 0)
            if isinstance(batch_context, Mapping)
            else 0
        )
        execution_batch_size = (
            _safe_int(batch_context.get("execution_batch_size") or 0)
            if isinstance(batch_context, Mapping)
            else 0
        )
        planned_entry = self._timelines.find_entry(task_id)
        prediction_matches = bool(
            planned_entry is not None
            and planned_entry.is_predicted
            and not planned_entry.is_completed
            and str(planned_entry.gpu_id) == str(gpu_id)
            and str(planned_entry.component) == str(component)
            and str(planned_entry.config_fingerprint or "") == config_fp
        )
        entry = (
            self._timelines.promote_predicted_to_active(task_id)
            if prediction_matches
            else None
        )
        if entry is None:
            removed += self._timelines.remove_all_entries(task_id)
            self._timelines.force_full_wall_for_task(task_id)
            entry = TimelineEntry(
                task_id=task_id,
                component=component,
                campaign_id=campaign_id or "",
                gpu_id=gpu_id,
                start_time=now,
                predicted_end_time=now + predicted_latency,
                predicted_vram_mb=predicted_vram,
                predicted_ram_mb=predicted_ram,
                is_backfill=is_backfill,
                input_size=_safe_float(input_size or 0.0),
                config_fingerprint=config_fp,
                was_primary_at_dispatch=_actual_was_primary,
            )
            self._timelines.add_entry(entry)
        entry.campaign_id = campaign_id or ""
        entry.is_backfill = bool(is_backfill)
        entry.input_size = _safe_float(input_size or 0.0)
        entry.logical_batch_size = logical_batch_size
        entry.execution_batch_size = execution_batch_size
        entry.was_primary_at_dispatch = _actual_was_primary
        entry.worker_name = str(worker_name or "").strip()
        entry.active_cancel_safe = bool(active_cancel_safe)
        _LOG.info(
            "[campaign-scheduler] dispatched %s → GPU %s (latency=%.0fs, vram=%.0fMB)",
            task_id[:12],
            gpu_id,
            entry.predicted_duration,
            entry.predicted_vram_mb,
        )

        if gpu_id:
            tl = self._timelines.get_or_create(gpu_id)
            if tl.total_vram_mb <= 0 and self._resource_tracker:
                try:
                    if self._resource_tracker.total_vram:
                        self._timelines.sync_vram_totals(
                            dict(self._resource_tracker.total_vram)
                        )
                except Exception:
                    _LOG.warning(
                        "[silent-except] %s:%d (%s)",
                        __name__,
                        0,
                        "swallowed_pass",
                        exc_info=True,
                    )

        if was_cold_start and gpu_id:
            init_sec = (
                actual_init_sec
                if actual_init_sec > 0
                else self._get_init_latency(component, gpu_id)
            )
            if init_sec > 0.5:
                init_entry = TimelineEntry(
                    task_id=f"{task_id}__init_phase",
                    component=component,
                    campaign_id=campaign_id or "",
                    gpu_id=gpu_id,
                    start_time=now - init_sec,
                    predicted_end_time=now,
                    predicted_vram_mb=predicted_vram,
                    predicted_ram_mb=predicted_ram,
                    is_backfill=is_backfill,
                    is_init=True,
                    is_predicted=False,
                    was_primary_at_dispatch=_actual_was_primary,
                )
                init_entry.completed_at = now
                self._timelines.add_entry(init_entry)

        self.check_pending_pre_inits()

        self._schedule_downstream_pre_init(
            task_id,
            campaign_id or "",
            component,
            gpu_id,
            predicted_end_time=entry.predicted_end_time,
        )

        self._reproject_downstream_on_dispatch(
            campaign_id or "", component, input_size, gpu_id, is_backfill
        )

    def on_worker_killed(self, component: str, gpu_ids: list[str]) -> int:
        """Called when guard kills a worker mid-task.

        Marks inflight timeline entries as killed (not completed) and
        adjusts campaign queue stats: active--, but NOT completed++.
        Also fires Layer-1 GC for any predicted entries owned by killed
        workers on the affected GPUs (Plan scenario wiring).

        Plan phantom-inflight fix — also releases the
        ``placement_context.gpu_views.inflight_components`` list entries
        and the corresponding reserved VRAM that ``on_task_dispatched_event``
        had appended at dispatch time.  Without this release, every
        guard-kill cycle leaks one ``component`` entry per killed task
        into ``gpu_views.inflight_components``, and the running
        accumulation produces phantom counts (e.g. ``inflight=[rfd, rfd,
        rfd]`` for a GPU whose worker is actually dead — observed during
         bench).  The dispatch-time push and this kill-time release
        are the symmetric pair to ``_pending_vram_releases`` for VRAM
        accounting; the previous code only paired them via the natural
        completion path (``on_task_complete_event`` in ``_run_task``
        finally), leaving guard-kill as the unpaired branch.
        """
        cleared_pre_init = 0
        for gid in gpu_ids:
            pre_init_key = (str(component), str(gid))
            if pre_init_key in self._fired_pre_inits:
                self._fired_pre_inits.discard(pre_init_key)
                cleared_pre_init += 1
        if cleared_pre_init:
            _LOG.info(
                "[pre-init] cleared %d fired marker(s) for stopped worker "
                "%s on GPUs %s",
                cleared_pre_init,
                component,
                gpu_ids,
            )

        killed = self._timelines.kill_entries_for_worker(component, gpu_ids)
        for entry in killed:
            cq = self._campaign_queues.get(entry.campaign_id)
            if cq and entry.task_id in cq._active_ids:
                cq._active_count = max(0, cq._active_count - 1)
                cq._active_ids.discard(entry.task_id)
            try:
                planner = getattr(self, "_planner_ref", None)
                release_fn = getattr(planner, "on_task_complete_event", None)
                if release_fn is not None:
                    release_fn(
                        str(entry.gpu_id),
                        str(entry.component),
                        float(getattr(entry, "predicted_vram_mb", 0.0) or 0.0),
                    )
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] phantom-inflight release failed "
                    "for killed entry task_id=%s gpu=%s component=%s",
                    entry.task_id,
                    entry.gpu_id,
                    entry.component,
                    exc_info=True,
                )
        if killed:
            _LOG.warning(
                "[campaign-scheduler] worker %s killed on GPUs %s — "
                "marked %d inflight entries as killed",
                component,
                gpu_ids,
                len(killed),
            )
        try:
            for gid in gpu_ids:
                removed = self._timelines.remove_predicted_entries_for_gpu(str(gid))
                if removed:
                    _LOG.info(
                        "[campaign-scheduler] Layer-1 GC swept %d predicted entries on GPU %s "
                        "(worker=%s killed)",
                        removed,
                        gid,
                        component,
                    )
        except AttributeError:
            _LOG.debug("scenario build lacks Layer-1 predicted-entry GC")
        self._on_scenario_changed(gpu_ids)
        self._notify_supervisor_wake("eviction_complete")
        return len(killed)

    def on_task_complete(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
        *,
        from_eviction: bool = False,
    ) -> None:
        """Called when a task finishes (success or failure).

        ``from_eviction=True`` (fix) is set by ``_handle_cancel_
        reenqueue`` when a cooperative cancel re-enqueue path fires on
        an in-flight task.  In that case the task is **not actually
        completed** — a fresh handle is being submitted to retry — so
        ``_completed_count`` must not increment (otherwise a single task
        cancelled N times before succeeding inflates the counter to N+1).
        Active-count decrement and timeline cleanup still happen because
        the original dispatch lifecycle did end; only the "finished a
        task" accounting is suppressed.

        The eventual fresh-handle completion (which fires this method
        again with ``from_eviction=False``) does the single legitimate
        ``_completed_count += 1`` for the task.
        """
        self._capture_main_loop()
        cq = self._campaign_queues.get(campaign_id or "")
        if cq:
            existing = self._timelines.find_entry(task_id)
            already_killed = existing is not None and existing.is_killed
            never_dispatched = existing is not None and existing.is_predicted

            if never_dispatched:
                if task_id in cq._pending_ids:
                    cq._pending_ids.discard(task_id)
                    cq._pending_count = max(0, cq._pending_count - 1)
                elif cq._pending_count > 0:
                    cq._pending_count = max(0, cq._pending_count - 1)
            elif already_killed:
                cq._pending_ids.discard(task_id)
                if not from_eviction:
                    cq._completed_count += 1
            else:
                cq._pending_ids.discard(task_id)
                if task_id in cq._active_ids:
                    cq._active_count = max(0, cq._active_count - 1)
                if not from_eviction:
                    cq._completed_count += 1
                cq._active_ids.discard(task_id)

        if from_eviction:
            self._timelines.mark_entry_killed(task_id)
        else:
            self._timelines.complete_entry(task_id)

        existing_after = self._timelines.find_entry(task_id)
        if existing_after is not None and existing_after.is_predicted:
            self._timelines.remove_all_entries(task_id)

        if gpu_id:
            try:
                planner_ref = getattr(self, "_planner_ref", None)
                global_planner = getattr(planner_ref, "_global_planner", None)
                tracker = getattr(global_planner, "_constraint_tracker", None)
                clear = getattr(tracker, "clear_saturation", None)
                if callable(clear):
                    clear(gpu_id)
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] clear compute saturation failed "
                    "gpu=%s task=%s",
                    gpu_id,
                    task_id[:12],
                    exc_info=True,
                )

        stale_pre_inits = [
            key
            for key in self._pending_pre_inits
            if key[2] == task_id
        ]
        for key in stale_pre_inits:
            self._pending_pre_inits.pop(key, None)
            comp = key[1]
            self._fired_pre_inits = {k for k in self._fired_pre_inits if k[0] != comp}

        self._on_scenario_changed([gpu_id] if gpu_id else [])

        self._notify_supervisor_wake("task_completion")

        self._check_pre_init(component, campaign_id, gpu_id)

        if cq and cq.is_empty:
            cid_prefix = (campaign_id or "")[:8]
            stale_removed = 0
            for tl in self._timelines._timelines.values():
                before = len(tl._entries)
                tl._entries = [
                    e
                    for e in tl._entries
                    if not (
                        e.is_predicted
                        and not e.is_completed
                        and (
                            e.task_id.startswith(f"__lookahead_{cid_prefix}_")
                            or e.task_id.startswith(f"__reproject_{cid_prefix}_")
                            or e.campaign_id == campaign_id
                        )
                    )
                ]
                if before != len(tl._entries):
                    tl._rebuild_component_index()
                    tl._bump_state_version()
                stale_removed += before - len(tl._entries)
            _LOG.info(
                "[campaign-scheduler] campaign %s completed (%d tasks, cleaned %d stale predicted)",
                campaign_id,
                cq.completed_tasks,
                stale_removed,
            )


    def register_cancel_callback(self, callback: Any) -> None:
        """Register a ``cancel_backfill_fn(task_id, *, reason)``
        awaitable — invoked by ``mark_for_eviction`` from the
        GlobalPlanner's MCPSE (plan) to cancel an in-flight
        backfill task on its worker.
        """
        self._cancel_backfill_fn = callback

    def mark_for_eviction(
        self,
        task_id: str,
        *,
        reason: str = "mcpse_eviction",
    ) -> asyncio.Task[Any] | None:
        """Plan /   — cancel an in-flight backfill.

        Fires the registered ``cancel_backfill_fn`` when available.
        Returns the scheduled cancel task, or ``None`` when cancel could not be
        scheduled.  Callers use the return value as the authority for whether
        an eviction marker may be committed to the timeline.
        Also clears any wake waiters on the host GPU so the Planner
        can advance immediately.

        Plan fix — `reason` parameter added so
        ``PrimaryWeightedCancellation`` can pass ``"primary_protection"``
        (distinct from MCPSE eviction's ``"mcpse_eviction"``).  The
        ``_handle_cancel_reenqueue`` path uses the reason for logging /
        re-enqueue policy.
        """
        cancel_fn = getattr(self, "_cancel_backfill_fn", None)
        if cancel_fn is None:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.debug(
                "[%s] mark_for_eviction: no running event loop — skipping",
                reason,
            )
            return None
        try:
            coro = cancel_fn(task_id, reason=reason)
            if asyncio.iscoroutine(coro):
                task = loop.create_task(coro)
                pending = getattr(self, "_pending_cancel_tasks", None)
                if pending is None:
                    self._pending_cancel_tasks = set()
                    pending = self._pending_cancel_tasks
                pending.add(task)

                def _consume_cancel_result(done: asyncio.Task[Any]) -> None:
                    self._pending_cancel_tasks.discard(done)
                    if done.cancelled():
                        _LOG.warning(
                            "[%s] mark_for_eviction cancel task cancelled task=%s",
                            reason,
                            task_id,
                        )
                        return
                    try:
                        result = done.result()
                    except Exception as exc:
                        _LOG.warning(
                            "[%s] mark_for_eviction cancel task failed task=%s: %s",
                            reason,
                            task_id,
                            exc,
                            exc_info=True,
                        )
                        return
                    if isinstance(result, bool) and not result:
                        _LOG.warning(
                            "[%s] mark_for_eviction cancel returned False task=%s",
                            reason,
                            task_id,
                        )

                task.add_done_callback(_consume_cancel_result)
                return task
            if isinstance(coro, bool) and not coro:
                _LOG.warning(
                    "[%s] mark_for_eviction cancel returned False task=%s",
                    reason,
                    task_id,
                )
                return None
            _LOG.warning(
                "[%s] mark_for_eviction callback did not return coroutine task=%s",
                reason,
                task_id,
            )
            return None
        except Exception as exc:
            _LOG.warning(
                "[%s] mark_for_eviction failed for task=%s: %s",
                reason,
                task_id,
                exc,
            )
            return None

    def attach_supervisor_wake(self, supervisor: Any) -> None:
        """Register a ``SchedulingSupervisor`` so event handlers can fire
        wake triggers (``task_completion``, ``eviction_complete``,
        ``profile_drift``).  See Plan wake trigger
        table for the full list."""
        self._supervisor_wake_hook = supervisor

    def _notify_supervisor_wake(self, trigger: str) -> None:
        sup = getattr(self, "_supervisor_wake_hook", None)
        if sup is None:
            return
        notify = getattr(sup, "notify_wake", None)
        if callable(notify):
            try:
                notify(trigger)
            except Exception:
                _LOG.warning(
                    "[campaign-scheduler] notify_wake(%s) failed",
                    trigger,
                    exc_info=True,
                )

    def _constraint_tracker(self) -> Any | None:
        planner = getattr(self, "_planner_ref", None)
        global_planner = getattr(planner, "_global_planner", None)
        return getattr(global_planner, "_constraint_tracker", None)

    def _worker_marker_names_for_component_gpu(
        self,
        component: str,
        gpu_id: str,
    ) -> set[str]:
        """Return canonical and supervisor-key names for a component/GPU pair."""
        sup = getattr(self, "_supervisor", None)
        states = getattr(sup, "states", None) if sup is not None else None
        if not states:
            return set()

        target_component = str(component or "")
        target_gpu = str(gpu_id)
        names: set[str] = set()
        for key, st in list(states.items()):
            spec = getattr(st, "spec", None)
            st_component = str(
                getattr(spec, "component", getattr(st, "component", "")) or ""
            )
            if st_component != target_component:
                continue
            gpus = list(
                getattr(st, "assigned_gpus", None) or getattr(spec, "gpus", None) or []
            )
            if target_gpu not in {str(item) for item in gpus}:
                continue
            names.add(str(key))
            spec_name = str(getattr(spec, "name", "") or "")
            if spec_name:
                names.add(spec_name)
        return names

    def _clear_ready_markers_for_component_gpu(
        self,
        component: str,
        gpu_id: str,
    ) -> None:
        """Clear transient readiness blockers after a concrete ready signal."""
        tracker = self._constraint_tracker()
        if tracker is None:
            return
        clear_activation = getattr(tracker, "clear_activation_exclusion", None)
        if callable(clear_activation):
            clear_activation(component, str(gpu_id))
        for name in self._worker_marker_names_for_component_gpu(component, str(gpu_id)):
            reset_failures = getattr(tracker, "reset_transient_grpc_failures", None)
            if callable(reset_failures):
                reset_failures(name)
            clear_cold = getattr(tracker, "clear_worker_cold_marker", None)
            if callable(clear_cold):
                clear_cold(name)
            clear_unreachable = getattr(tracker, "clear_worker_unreachable", None)
            if callable(clear_unreachable):
                clear_unreachable(name)

    def _notify_pre_init_state_change(self, gpu_id: str, trigger: str) -> None:
        gpu_ids = [str(gpu_id)] if gpu_id is not None else []
        try:
            self._on_scenario_changed(gpu_ids)
        except Exception:
            _LOG.warning(
                "[pre-init] scenario refresh failed after %s",
                trigger,
                exc_info=True,
            )
        self._notify_supervisor_wake(trigger)

    def _on_scenario_changed(self, affected_gpu_ids: list[str]) -> None:
        _osc_start = time.time()
        try:
            return self._on_scenario_changed_impl(affected_gpu_ids)
        finally:
            _osc_elapsed_ms = (time.time() - _osc_start) * 1000.0
            if _osc_elapsed_ms > 20.0:
                _LOG.warning(
                    "[hot-path] _on_scenario_changed gpus=%s elapsed=%.1fms",
                    list(affected_gpu_ids or []),
                    _osc_elapsed_ms,
                )

    def _on_scenario_changed_impl(self, affected_gpu_ids: list[str]) -> None:
        """Common handler after any timeline change event.

        Called by on_task_complete, on_worker_killed, on_profile_drift,
        and task cancel. Performs:
        1. Refresh PlacementContext GPU views (prevents available=0 deadlock
           when no new generate_plan calls reach refresh_gpu_views)
        2. Notify compute slot waiters (unblocks primary, triggers backfill re-schedule)
        3. Check pending pre-inits
        4. Prune stale predicted entries
        """
        sup = getattr(self, "_supervisor", None)
        resource_tracker = getattr(sup, "resource_tracker", None) if sup else None
        planner = getattr(self, "_planner_ref", None)
        if planner and sup:
            try:
                snapshots = sup.list_worker_snapshots_light()
                planner._placement_ctx.refresh_gpu_views(
                    snapshots, resource_tracker=resource_tracker
                )
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )
        if resource_tracker:
            for gid in affected_gpu_ids:
                resource_tracker._notify_compute_waiters(gid)
        self.check_pending_pre_inits()
        self._timelines.prune_stale_predicted()


    def _get_eviction_grace_period(self, component: str, gpu_id: str) -> float:
        """Return the eviction grace period (seconds) for *component* on *gpu_id*.

        Base implementation returns the static config value.  DG subclass
        overrides this with InitProfile GP to return component-specific
        init latency — the physical minimum time before a re-dispatched
        worker can be ready.
        """
        sup = getattr(self, "_supervisor", None)
        if sup:
            return _safe_float(getattr(sup, "eviction_grace_period_s", 5.0))
        return 5.0


    def _estimate_candidate_wait(self, component: str, current_gpu_id: str) -> float:
        """Estimate how long the candidate would wait if not admitted here.

        Scans all other GPUs and returns the time until the earliest one
        becomes free (no active non-predicted entries).  All values from
        timeline data — no arbitrary constants.
        """
        now = time.time()
        best_wait = math.inf

        for gpu_id in self._timelines.gpu_ids:
            if gpu_id == current_gpu_id:
                continue
            tl = self._timelines.get(gpu_id)
            if not tl:
                return 0.0
            active = (
                tl.planned_occupancy_entries(now)
                if hasattr(tl, "planned_occupancy_entries")
                else list(tl.active_entries)
            )
            if not active:
                return 0.0
            latest_end = max(e.predicted_end_time for e in active)
            wait = max(0.0, latest_end - now)
            best_wait = min(best_wait, wait)

        return best_wait if best_wait != math.inf else 0.0

    def _higher_priority_entries(
        self, candidate_campaign_id: str, cross_campaign_entries: list
    ) -> list:
        """From cross-campaign entries, return those from higher-priority campaigns.

        Priority = FIFO order (earlier arrival_time = higher priority).
        """
        candidate_arrival = math.inf
        cq = self._campaign_queues.get(candidate_campaign_id)
        if cq:
            candidate_arrival = cq.arrival_time

        return [
            e
            for e in cross_campaign_entries
            if self._campaign_arrival(e.campaign_id) < candidate_arrival
        ]

    def _campaign_arrival(self, campaign_id: str) -> float:
        """Return arrival_time for a campaign (inf if unknown)."""
        cq = self._campaign_queues.get(campaign_id)
        return cq.arrival_time if cq else math.inf

    def is_backfill_feasible(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        campaign_id: str,
    ) -> bool:
        """Check whether a backfill task can be placed on *gpu_id* without
        delaying the primary campaign.

        Returns ``False`` if:
        - GP prediction is unavailable (low confidence → no backfill).
        - VRAM doesn't fit.
        - Predicted completion would delay the primary campaign.
        - Adding this task would unacceptably slow running primary tasks.
        """
        primary = self._primary_campaign()
        if primary is None:
            return True

        tl = self._timelines.get(gpu_id)
        if tl is None:
            return True

        confidence = self._get_confidence(component, input_size, gpu_id)

        vram_pred = self._predict_vram(component, input_size, gpu_id)
        if vram_pred > 0:
            available = tl.available_vram_at()
            if available < vram_pred:
                return False

        latency_pred = self._predict_latency(component, input_size, gpu_id)
        if latency_pred is None:
            return False

        if confidence == "low":
            primary_entries = tl.active_entries_for_campaign(primary.campaign_id)
            if primary_entries:
                return False

        from .global_planner import _norm_ppf

        sigma = self._predict_latency_sigma(component, input_size, gpu_id) or 0.0
        if sigma > 0:
            alpha = self._alpha_from_confidence(confidence)
            latency_pred = latency_pred + sigma * _norm_ppf(alpha)

        primary_entries = tl.active_entries_for_campaign(primary.campaign_id)
        if not primary_entries:
            return True

        now = time.time()
        backfill_end = now + latency_pred
        for pe in primary_entries:
            if pe.predicted_end_time > now and backfill_end > pe.predicted_end_time:
                return False

        return True


    def get_eviction_candidates(
        self,
        gpu_id: str,
        shortfall_mb: int,
        primary_campaign_id: str,
    ) -> tuple[list[str], str]:
        """Return (task_ids_to_evict, eviction_phase) for the given GPU.

        Implements the three-phase eviction strategy from 
        Returns task_ids (not WorkerStates) — the caller maps these
        to actual worker stop operations.
        """
        tl = self._timelines.get(gpu_id)
        if tl is None:
            return [], "none"

        backfill_entries = []
        lower_priority_entries = []
        for e in tl.active_entries:
            if e.campaign_id == primary_campaign_id:
                continue
            if e.is_backfill:
                backfill_entries.append(e)
            else:
                lower_priority_entries.append(e)

        sorted_bf = sorted(backfill_entries, key=lambda e: self._eviction_cost(e))
        phase1_evict: list[TimelineEntry] = []
        phase1_freed = 0.0
        for e in sorted_bf:
            phase1_evict.append(e)
            phase1_freed += e.predicted_vram_mb
            if phase1_freed >= shortfall_mb:
                return [e.task_id for e in phase1_evict], "phase1_backfill"

        sorted_lp = sorted(lower_priority_entries, key=lambda e: self._eviction_cost(e))

        for e in sorted_lp:
            if e.predicted_vram_mb >= shortfall_mb:
                single_cost = self._eviction_cost(e)
                all_bf_cost = sum(self._eviction_cost(b) for b in sorted_bf)
                if single_cost < all_bf_cost and sorted_bf:
                    return [e.task_id], "phase2_single_nonbackfill"

        remaining = shortfall_mb - phase1_freed
        phase2_evict = list(phase1_evict)
        for e in sorted_lp:
            phase2_evict.append(e)
            remaining -= e.predicted_vram_mb
            if remaining <= 0:
                return [e.task_id for e in phase2_evict], "phase2_mixed"

        return [e.task_id for e in phase2_evict], "phase3_insufficient"

    def _eviction_cost(self, entry: TimelineEntry) -> float:
        """Lower cost = cheaper to evict."""
        init_latency = self._get_init_latency(entry.component, entry.gpu_id)
        latency = entry.predicted_duration
        return entry.elapsed + init_latency + latency

    def _find_cheapest_backfill_to_evict(
        self,
        gpu_id: str = "",
    ) -> tuple[str, list[str]]:
        """Find the cheapest backfill task to cancel to unblock a GPU for primary.

        If *gpu_id* is given, only considers backfill on that GPU — evicting
        backfill on a different GPU cannot unblock the requesting GPU.
        If empty, scans all GPUs (legacy fallback).

        Returns (gpu_id, [task_id]) for the cheapest entry, or ("", []).
        """
        best_gpu = ""
        best_task_id = ""
        best_cost = math.inf

        scan_gpus = [gpu_id] if gpu_id else list(self._timelines.gpu_ids)
        for gid in scan_gpus:
            tl = self._timelines.get(gid)
            if not tl:
                continue
            for e in tl.active_entries:
                if not e.is_backfill or e.is_predicted:
                    continue
                cost = self._eviction_cost(e)
                if cost < best_cost:
                    best_cost = cost
                    best_gpu = gid
                    best_task_id = e.task_id

        if best_task_id:
            return best_gpu, [best_task_id]
        return "", []


    def _schedule_downstream_pre_init(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        gpu_id: str,
        predicted_end_time: float,
    ) -> None:
        """Schedule pre_init for downstream components DURING task execution.

        Called from on_task_dispatched. Computes:
          trigger_at = predicted_end - init - buffer

        If trigger_at <= now, fires immediately (task is almost done).
        Otherwise, stores in _pending_pre_inits for later firing.

        Plan _workspace/plan_proton_naive_baseline.md Sub-step 5 follow-up:
        when ``GlobalPlanner.preinit`` is ``NoPreInit`` (proton-naive
        baseline), short-circuit so the PreInitStrategy choice
        actually takes effect.  See ``_pre_init_disabled`` field doc.
        ``getattr`` fallback covers test fixtures that bypass
        ``__init__`` via ``CampaignScheduler.__new__`` and only set the
        attributes they touch.
        """
        if getattr(self, "_pre_init_disabled", False):
            return
        cq = self._campaign_queues.get(campaign_id)
        if cq is None or cq.dag_context is None:
            return

        downstream = cq.dag_context.downstream_map.get(component, [])
        if not downstream:
            return

        now = time.time()

        for next_comp in downstream:
            target_gpus = self._select_pre_init_gpus(
                next_comp,
                gpu_id,
                campaign_id,
            )
            for target_gpu in target_gpus:
                pre_init_key = (next_comp, target_gpu)
                if pre_init_key in self._fired_pre_inits:
                    continue

                init_mu = self._get_init_latency(next_comp, target_gpu)
                from .global_planner import _norm_ppf

                init_margin = 0.0
                if self._init_tracker:
                    try:
                        _, init_var = self._init_tracker.predict(next_comp, target_gpu)
                        conf = self._init_tracker.confidence(next_comp, target_gpu)
                        alpha_init = max(0.5, min(0.99, float(conf)))
                        if init_var and init_var > 0:
                            init_margin = _norm_ppf(alpha_init) * (init_var**0.5)
                    except (AttributeError, KeyError, TypeError, ValueError) as exc:
                        _LOG.debug(
                            "[pre-init] init_tracker predict failed: %s",
                            exc,
                        )
                effective_init = init_mu + init_margin
                trigger_at = predicted_end_time - effective_init

                if trigger_at <= now:
                    self._fire_pre_init(next_comp, target_gpu, campaign_id, component)
                else:
                    sched_key = (campaign_id, next_comp, f"{task_id}_{target_gpu}")
                    self._pending_pre_inits[sched_key] = (
                        trigger_at,
                        target_gpu,
                        component,
                    )

                _LOG.info(
                    "[pre-init-scheduled] %s on GPU %s in %.0fs "
                    "(source=%s finishes in %.0fs, init=%.0fs)",
                    next_comp,
                    target_gpu,
                    max(0, trigger_at - now),
                    component,
                    predicted_end_time - now,
                    effective_init,
                )

    def check_pending_pre_inits(self) -> int:
        """Fire any scheduled pre_inits whose trigger time has arrived.

        Called from generate_plan (runs on every task submit) and can also
        be called from a periodic background task.

        Plan _workspace/plan_proton_naive_baseline.md Sub-step 5 follow-up:
        short-circuit when ``_pre_init_disabled`` (NoPreInit wired) so
        the PreInitStrategy choice actually takes effect.  ``getattr``
        fallback covers test fixtures that bypass ``__init__``.

        Returns the number of pre_inits fired.
        """
        if getattr(self, "_pre_init_disabled", False):
            return 0
        if not self._pending_pre_inits:
            return 0

        now = time.time()
        fired = 0
        to_remove: list[tuple[str, str, str]] = []

        for key, (trigger_at, gpu_id, source_comp) in self._pending_pre_inits.items():
            campaign_id, next_comp, _task_id = key
            if now >= trigger_at:
                self._fire_pre_init(next_comp, gpu_id, campaign_id, source_comp)
                to_remove.append(key)
                fired += 1

        for key in to_remove:
            self._pending_pre_inits.pop(key, None)

        return fired

    def _handoff_pre_init_weight_reservation(
        self,
        component: str,
        gpu_id: str,
        campaign_id: str = "",
    ) -> int:
        """Drop soft init weight once supervisor owns the hard reservation."""
        timeline = self._timelines.get(str(gpu_id))
        if timeline is None:
            return 0
        updated = 0
        for entry in timeline._entries:
            if not entry.is_init or not entry.task_id.startswith("preinit_"):
                continue
            if entry.component != component or entry.is_completed:
                continue
            if campaign_id and entry.campaign_id != campaign_id:
                continue
            entry.predicted_vram_mb = 0.0
            entry.predicted_ram_mb = 0.0
            updated += 1
        if updated:
            timeline._bump_state_version()
        return updated

    def _fire_pre_init(
        self,
        component: str,
        gpu_id: str,
        campaign_id: str,
        source_comp: str,
        *,
        weight_vram_mb: float = 0.0,
        weight_ram_mb: float = 0.0,
    ) -> None:
        """Actually trigger a pre_init (call the pre_init callback).

        Plan _workspace/plan_proton_naive_baseline.md Sub-step 5 follow-up
        (post): single-point disable for proton-naive's
        ``NoPreInit`` so any caller path (predictive _schedule_downstream_
        pre_init / pending check_pending_pre_inits / reactive _check_
        pre_init / future paths) goes through this guard.  Defense-in-
        depth — the per-caller short-circuits in those three sites
        catch most calls before they reach here, but a future code path
        that calls ``_fire_pre_init`` directly without going through one
        of those gates will still respect the disable flag.
        """
        if getattr(self, "_pre_init_disabled", False):
            return
        pre_init_key = (component, str(gpu_id))
        if pre_init_key in self._fired_pre_inits:
            return

        self._fired_pre_inits.add(pre_init_key)

        if self._pre_init_fn:
            now = time.time()
            init_latency = self._get_init_latency(component, gpu_id)
            init_phase_task_id = (
                f"preinit_{component}_{gpu_id}_{_safe_int(now * 1000)}__init_phase"
            )
            init_entry_added = False

            def _add_init_entry() -> None:
                nonlocal init_entry_added
                if init_entry_added or init_latency <= 0.5:
                    return
                init_entry = TimelineEntry(
                    task_id=init_phase_task_id,
                    component=component,
                    campaign_id=campaign_id or "",
                    gpu_id=str(gpu_id),
                    start_time=now,
                    predicted_end_time=now + init_latency,
                    predicted_vram_mb=max(0.0, _safe_float(weight_vram_mb)),
                    predicted_ram_mb=max(0.0, _safe_float(weight_ram_mb)),
                    is_backfill=True,
                    is_init=True,
                    is_predicted=False,
                    completed_at=None,
                    was_primary_at_dispatch=False,
                )
                self._timelines.add_entry(init_entry)
                init_entry_added = True

            def _remove_init_entry() -> None:
                if init_entry_added:
                    self._timelines.remove_all_entries(init_phase_task_id)

            try:
                coro = self._pre_init_fn(component, gpu_id, campaign_id)
            except Exception as exc:
                _LOG.warning("[pre-init-fire-failed] %s: %s", component, exc)
                self._fired_pre_inits.discard(pre_init_key)
                return

            if asyncio.iscoroutine(coro):
                wrapped = None
                try:
                    _add_init_entry()

                    async def _wrap_complete_init() -> None:
                        did_activate = True
                        try:
                            result = await coro
                            if isinstance(result, bool):
                                did_activate = result
                        except Exception as exc:
                            _LOG.debug(
                                "[pre-init] failed for %s on GPU %s: %s",
                                component,
                                gpu_id,
                                exc,
                            )
                            self._fired_pre_inits.discard(pre_init_key)
                            _remove_init_entry()
                            self._notify_pre_init_state_change(gpu_id, "preinit_failed")
                            return

                        done_at = time.time()
                        if not did_activate:
                            self._clear_ready_markers_for_component_gpu(
                                component, gpu_id
                            )
                            _remove_init_entry()
                            self._notify_pre_init_state_change(
                                gpu_id, "preinit_warm_hit"
                            )
                            return

                        self._clear_ready_markers_for_component_gpu(component, gpu_id)
                        for tl in self._timelines._timelines.values():
                            for e in list(tl._entries):
                                if (
                                    e.task_id == init_phase_task_id
                                    and e.completed_at is None
                                ):
                                    e.completed_at = done_at
                                    e.predicted_end_time = done_at
                                    tl._retire_entry(e)
                                    self._notify_pre_init_state_change(
                                        gpu_id, "preinit_complete"
                                    )
                                    return
                        self._notify_pre_init_state_change(gpu_id, "preinit_complete")

                    wrapped = _wrap_complete_init()
                    target_loop = getattr(self, "_main_loop", None)
                    if target_loop is None or target_loop.is_closed():
                        self._capture_main_loop()
                        target_loop = getattr(self, "_main_loop", None)
                    if target_loop is None or target_loop.is_closed():
                        raise RuntimeError("gateway event loop is unavailable")
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None
                    if running_loop is target_loop:
                        target_loop.create_task(wrapped)
                    else:
                        asyncio.run_coroutine_threadsafe(wrapped, target_loop)
                except Exception as exc:
                    if wrapped is not None:
                        wrapped.close()
                    coro.close()
                    _remove_init_entry()
                    self._fired_pre_inits.discard(pre_init_key)
                    self._notify_pre_init_state_change(gpu_id, "preinit_failed")
                    _LOG.warning("[pre-init-fire-failed] %s: %s", component, exc)
                    return

            _LOG.info(
                "[pre-init-fired] %s on GPU %s (campaign=%s, triggered_by=%s)",
                component,
                gpu_id,
                campaign_id,
                source_comp,
            )

            proj_suffix = f"__{component}__pre_init"
            for tl in self._timelines._timelines.values():
                before = len(tl._entries)
                tl._entries = [
                    e
                    for e in tl._entries
                    if not (e.task_id.endswith(proj_suffix) and e.is_predicted)
                ]
                if before != len(tl._entries):
                    tl._rebuild_component_index()
                    tl._bump_state_version()


    def _check_pre_init(
        self,
        component: str,
        campaign_id: str,
        gpu_id: str,
    ) -> None:
        """After a task completes, check if downstream components need pre_init.

        Implements  from planner_scheduler_theory.md:
        - Checks DAG for downstream components
        - Selects best GPU for pre_init (pipeline locality, VRAM, interference)
        - If DAG unknown: no pre_init (graceful degradation)
        - Also plans fan-out pre-assignment if fan-out is known

        Plan _workspace/plan_proton_naive_baseline.md Sub-step 5 follow-up
        (post): NoPreInit short-circuit applies to this
        reactive path too.  Sub-step 8.5 (commit <rev>) only patched
        ``_schedule_downstream_pre_init`` + ``check_pending_pre_inits``;
        bench evidence (exp2c proton-naive smoke <stamp>, 97
        ``[pre-init-fired]`` events despite ``preinit=NoPreInit``)
        confirmed the reactive path bypassed the disable flag.
        """
        if getattr(self, "_pre_init_disabled", False):
            return
        cq = self._campaign_queues.get(campaign_id or "")
        if cq is None or cq.dag_context is None:
            return

        downstream = cq.dag_context.downstream_map.get(component, [])
        if not downstream:
            return

        for next_comp in downstream:
            target_gpus = self._select_pre_init_gpus(
                next_comp,
                gpu_id,
                campaign_id,
            )
            for target_gpu in target_gpus:
                if (next_comp, target_gpu) in self._fired_pre_inits:
                    continue
                self._fire_pre_init(next_comp, target_gpu, campaign_id, component)

            fan_out = cq.effective_fan_out(component)
            if fan_out is not None and fan_out > 1:
                self.plan_fan_out(campaign_id, component, next_comp, gpu_id)

    def _select_pre_init_gpu(
        self,
        next_comp: str,
        source_gpu: str,
        campaign_id: str,
    ) -> str | None:
        """Select single best GPU for pre-initing a downstream component.

        Locality-first: prefer the source GPU (where the current pipeline
        stage is running) because it will be freed when the current task
        completes. Placement scoring is unsuitable here — it reflects
        *current* GPU state, but pre-init targets *future* state.

        Fallback to best_gpu_for() only when source GPU can't fit the weight.
        Returns None when no GPU has enough projected VRAM; pre-init must not
        force a worker init that the planner's own memory view says is unsafe.
        """
        vram_needed = self._predict_vram(next_comp, 0.0, source_gpu)

        source_tl = self._timelines.get(source_gpu)
        if source_tl and source_tl.available_vram_at() >= vram_needed:
            return source_gpu

        best = self._timelines.best_gpu_for(
            vram_needed if vram_needed > 0 else 1.0,
            exclude={source_gpu}
            if source_tl and source_tl.available_vram_at() < vram_needed
            else None,
        )
        return best

    def _select_pre_init_gpus(
        self,
        next_comp: str,
        source_gpu: str,
        campaign_id: str,
    ) -> list[str]:
        """Select all GPUs with sufficient VRAM to pre-init a downstream
        component.  Source GPU first (pipeline locality), then remaining
        GPUs in arbitrary deterministic order.  Returns empty list if none
        fit.

        Multi-GPU pre-init covers downstream fan-out (RFD 10 → MPNN 40 →
        ESM 120) where every GPU will eventually need the worker.
        Supervisor's ``ensure_component_pool_ready`` is idempotent — a
        spawn request on an already-ready GPU is a cheap no-op.
        """
        vram_needed = self._predict_vram(next_comp, 0.0, source_gpu)
        candidates: list[str] = []
        source_tl = self._timelines.get(source_gpu)
        if source_tl and source_tl.available_vram_at() >= vram_needed:
            candidates.append(source_gpu)
        for gid in sorted(self._timelines._timelines.keys()):
            if gid == source_gpu:
                continue
            tl = self._timelines.get(gid)
            if tl and tl.available_vram_at() >= vram_needed:
                candidates.append(gid)
        return candidates



    def effective_fan_out_with_global(
        self,
        campaign_id: str,
        component: str,
    ) -> float | None:
        """Fan-out estimate via the DAG-hint resolution.

        Plan fix retired the global ``observed`` median
        fallback.  Fan-out source-of-truth is now
        ``PipelineDAG._campaign_fan_out[campaign_id][component]``
        (populated by ``register_fan_out`` from
        ``ext.output_sample_count``) which surfaces through
        ``cq.dag_context.fan_out_map`` and out via
        ``cq.effective_fan_out``.  The cross-campaign median fallback
        had no input feeder after fix (its writer
        ``record_fan_out_observation`` was unwired) so the branch
        never fired in production — kept this method as a thin
        passthrough to ``cq.effective_fan_out`` so the call site at
        ``CampaignQueue.refresh_remaining_est`` (line 540) continues
        to compile.
        """
        cq = self._campaign_queues.get(campaign_id)
        if cq is None:
            return None
        return cq.effective_fan_out(component)

    def plan_fan_out(
        self,
        campaign_id: str,
        current_component: str,
        next_component: str,
        current_gpu_id: str,
    ) -> list[str] | None:
        """Pre-assign GPUs for a fan-out batch. Returns assigned gpu_ids or None."""
        cq = self._campaign_queues.get(campaign_id or "")
        if cq is None:
            return None

        fan_out = cq.effective_fan_out(current_component)
        if fan_out is None or fan_out <= 1:
            return None

        vram_per_task = self._predict_vram(next_component, 0.0, current_gpu_id)
        if vram_per_task <= 0:
            return None

        total_needed = fan_out * vram_per_task
        assigned: list[str] = []
        assigned_capacity = 0.0
        exclude: set = set()

        while assigned_capacity < total_needed:
            gpu = self._timelines.best_gpu_for(vram_per_task, exclude=exclude)
            if gpu is None:
                break
            tl = self._timelines.get(gpu)
            if tl:
                capacity = tl.available_vram_at()
                tasks_on_gpu = max(1, _safe_int(capacity / vram_per_task))
                assigned_capacity += tasks_on_gpu * vram_per_task
            assigned.append(gpu)
            exclude.add(gpu)

        if assigned:
            cq.fan_out_assignments[next_component] = assigned
            cq.fan_out_counters[next_component] = 0
            _LOG.info(
                "[fan-out] pre-assigned %d GPUs for %s (fan_out=%.0f, campaign=%s)",
                len(assigned),
                next_component,
                fan_out,
                campaign_id,
            )
        return assigned if assigned else None


    def on_profile_drift(self, component: str, metric: str, **kwargs: Any) -> None:
        """Single-event drift entry (legacy / test path).

        Wraps the batched cascade so behaviour is identical to the pre-B1
        per-event path.  All real production callers should reach
        ``on_profile_drift_batch`` via SignalService coalescing.
        """
        ev = kwargs.get("event")
        if ev is None:
            ev = SimpleNamespace(
                component=component,
                metric=metric,
                observed=0.0,
                predicted=0.0,
                drift_ratio=0.0,
                n_baseline=0,
                config_fingerprint="",
                gpu_id=None,
                campaign_id="",
            )
        self.on_profile_drift_batch({component: {metric: ev}})

    def _drift_apply_per_component(
        self,
        component: str,
        metrics: dict[str, Any],
        primary: Any,
    ) -> tuple[int, int, int, int]:
        """Per-component drift work — invalidation + reprojection +
        backfill deadline recompute.  Cascade-invariant pieces
        (``_primary_campaign``, ``_on_scenario_changed``) are NOT
        called here; the caller (batch driver) runs them once per
        flush regardless of how many components were dirty.
        """
        if not hasattr(self, "_drift_log_dedup"):
            self._drift_log_dedup: dict[str, float] = {}
        primary_metric = next(iter(metrics)) if metrics else ""
        _dedup_key = f"{component}:{primary_metric}"
        _drift_now = time.time()
        _drift_is_burst = (
            _drift_now - self._drift_log_dedup.get(_dedup_key, 0.0)
        ) < 2.0
        self._drift_log_dedup[_dedup_key] = _drift_now

        if not hasattr(self, "_last_invalidation_at"):
            self._last_invalidation_at: dict[str, float] = {}
        _now_inv = time.time()
        _MIN_INVALIDATION_INTERVAL_SEC = 1.0
        if (
            _now_inv - self._last_invalidation_at.get(component, 0.0)
            < _MIN_INVALIDATION_INTERVAL_SEC
        ):
            (_LOG.debug if _drift_is_burst else _LOG.info)(
                "[campaign-scheduler-drift] component=%s metrics=%s "
                "invalidated=0 (cooldown, skipped) reprojected=0 refreshed=0 "
                "deadlines_recalced=0",
                component,
                ",".join(sorted(metrics.keys())),
            )
            return 0, 0, 0, 0
        self._last_invalidation_at[component] = _now_inv

        n_invalidated = self._timelines.invalidate_component(
            component,
            metrics=set(metrics.keys()),
        )

        if "vram" in metrics:
            for cq in self._campaign_queues.values():
                cq.fan_out_assignments.pop(component, None)

        to_cancel = [key for key in self._pending_pre_inits if key[1] == component]
        for key in to_cancel:
            self._pending_pre_inits.pop(key, None)
        n_reprojected = 0
        n_refreshed = 0
        n_recalced = 0

        (_LOG.debug if _drift_is_burst else _LOG.info)(
            "[campaign-scheduler-drift] component=%s metrics=%s "
            "invalidated=%d, reprojected=%d, refreshed=%d, "
            "deadlines_recalced=%d",
            component,
            ",".join(sorted(metrics.keys())),
            n_invalidated,
            n_reprojected,
            n_refreshed,
            n_recalced,
        )
        return n_invalidated, n_reprojected, n_refreshed, n_recalced

    def on_profile_drift_batch(
        self,
        by_component: dict[str, dict[str, Any]],
    ) -> None:
        """Plan v4.x B1 — single cascade per drift coalesce-flush.

        ``signals.SignalService._flush_pending_drifts`` groups buffered
        events by component and invokes this once per flush instead of
        per (component, metric) event.  Cascade-invariant work runs
        exactly once across the whole batch:

          • ``_primary_campaign()``     — 1× (was N× before B1)
          • ``_on_scenario_changed()``  — 1× (was N×; this is the
            biggest single saving — list_worker_snapshots over 24
            workers + refresh_gpu_views was the dominant hot path)

        Per-component work runs once per *unique component* in the
        batch (≤ 6 in the current pipeline) instead of once per
        event (worst-case 12 with two metrics × 6 components).
        """
        if not by_component:
            return
        self._capture_main_loop()
        loop = getattr(self, "_main_loop", None)
        main_tid = getattr(self, "_main_thread_id", None)
        if (
            loop is not None
            and not loop.is_closed()
            and main_tid is not None
            and threading.get_ident() != main_tid
        ):
            snapshot = {
                component: dict(metrics) for component, metrics in by_component.items()
            }
            loop.call_soon_threadsafe(
                self._enqueue_profile_drift_batch,
                snapshot,
            )
            return
        self._enqueue_profile_drift_batch(by_component)

    def _enqueue_profile_drift_batch(
        self,
        by_component: dict[str, dict[str, Any]],
    ) -> None:
        """Merge drift batches and publish once per short coalesce window.

        SignalService already groups raw drift events, but live expr-b/mock-b
        still produced many main-loop publishes back-to-back.  Publishing is
        cheap relative to old reprojection, yet it still invalidates timelines
        and wakes the supervisor.  This second-stage scheduler-side coalesce
        keeps dispatch from being interrupted by a burst of equivalent drift
        notifications while preserving the "fresh GP on next planning epoch"
        contract at sub-second granularity.
        """
        if not by_component:
            return
        pending = getattr(self, "_pending_profile_drift_batch", None)
        if pending is None:
            pending = {}
            self._pending_profile_drift_batch = pending
        for component, metrics in by_component.items():
            bucket = pending.setdefault(str(component), {})
            bucket.update(dict(metrics or {}))

        handle = getattr(self, "_profile_drift_publish_handle", None)
        if handle is not None and not handle.cancelled():
            return

        self._capture_main_loop()
        loop = getattr(self, "_main_loop", None)
        if loop is None or loop.is_closed():
            self._flush_profile_drift_batch()
            return
        self._profile_drift_publish_handle = loop.call_later(
            _safe_float(getattr(self, "_profile_drift_coalesce_sec", 0.25) or 0.25),
            self._flush_profile_drift_batch,
        )

    def _flush_profile_drift_batch(self) -> None:
        self._profile_drift_publish_handle = None
        pending = getattr(self, "_pending_profile_drift_batch", None)
        if not pending:
            return
        self._pending_profile_drift_batch = {}
        self._publish_profile_drift_batch(pending)

    def _publish_profile_drift_batch(
        self,
        by_component: dict[str, dict[str, Any]],
    ) -> None:
        """Apply an already-coalesced drift batch as a short main-loop publish."""
        if not by_component:
            return
        started = time.perf_counter()
        total_invalidated = 0
        for component, metrics in by_component.items():
            n_invalidated, _, _, _ = self._drift_apply_per_component(
                component,
                metrics,
                None,
            )
            total_invalidated += _safe_int(n_invalidated or 0)
        self.drift_applied_epoch += 1
        self._notify_supervisor_wake("profile_drift")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 20.0:
            _LOG.warning(
                "[hot-path] drift_publish components=%d invalidated=%d elapsed=%.1fms",
                len(by_component),
                total_invalidated,
                elapsed_ms,
            )

    _STALE_REFRESH_THROTTLE_N = 10

    def _refresh_stale_entries(self, component: str) -> int:
        """Re-predict and clear stale flag on running entries after drift.

        After ``invalidate_component`` marks running entries as
        ``prediction_stale=True``, this method re-queries the (now updated)
        GP for fresh latency predictions and clears the stale flag.
        Without this, stale entries would show orange hatching in the ops
        console indefinitely until the task completes.

         fix Part 3 — N  cascade  1  (throttle).
        cascade rate     _predict_latency Cholesky refit O(N²)
        × N entries    main loop  .  per-component
        counter  N=10 cascade  actual scan + refresh.

        Returns the number of entries refreshed.
        """
        if not hasattr(self, "_stale_refresh_counters"):
            self._stale_refresh_counters: dict[str, int] = {}
        self._stale_refresh_counters[component] = (
            self._stale_refresh_counters.get(component, 0) + 1
        )
        if (
            self._stale_refresh_counters[component] % self._STALE_REFRESH_THROTTLE_N
            != 0
        ):
            return 0

        n_refreshed = 0
        for gpu_id in self._timelines.gpu_ids:
            tl = self._timelines.get(gpu_id)
            if tl is None:
                continue
            timeline_mutated = False
            for entry in tl.active_entries:
                if entry.component != component or not entry.prediction_stale:
                    continue
                if entry.is_completed or entry.is_predicted:
                    continue
                if _safe_int(getattr(entry, "execution_batch_size", 0)) > 0:
                    entry.prediction_stale = False
                    n_refreshed += 1
                    continue
                entry_fp = str(getattr(entry, "config_fingerprint", "") or "")
                mu = self._predict_latency(
                    component,
                    entry.input_size,
                    gpu_id,
                    config_fingerprint=entry_fp,
                )
                if mu is not None and mu > 0:
                    entry.predicted_end_time = entry.start_time + mu
                    entry.prediction_stale = False
                    new_vram = self._predict_vram(
                        component,
                        entry.input_size,
                        gpu_id,
                        config_fingerprint=entry_fp,
                    )
                    if new_vram > 0:
                        entry.predicted_vram_mb = new_vram
                    new_ram = self._predict_ram(
                        component,
                        entry.input_size,
                        gpu_id,
                        config_fingerprint=entry_fp,
                    )
                    if new_ram > 0:
                        entry.predicted_ram_mb = new_ram
                    timeline_mutated = True
                    n_refreshed += 1
            if timeline_mutated:
                tl._bump_state_version()
        return n_refreshed

    def _recalculate_backfill_deadlines(
        self,
        component: str,
        primary: CampaignQueue | None = None,
    ) -> int:
        """Recalculate deadlines for running backfill tasks after drift ().

        When a drift event changes GP predictions for *component*, backfill
        tasks on GPUs where *component* is primary may have stale deadlines.
        Re-query GP with each backfill entry's original ``input_size`` and
        update ``predicted_end_time`` accordingly.

        Plan fix (F-xx) — accepts optional pre-computed
        *primary* from ``on_profile_drift`` to avoid a redundant
        ``_primary_campaign()`` → ``is_empty`` → ``_campaign_records()``
        scan chain (O(K × M)).  Legacy callers (None default) fall back to
        the original self-lookup for backward compatibility.

        Returns the number of entries recalculated.
        """
        if primary is None:
            primary = self._primary_campaign()
        if primary is None:
            return 0

        n_recalced = 0
        for gpu_id in self._timelines.gpu_ids:
            tl = self._timelines.get(gpu_id)
            if tl is None:
                continue

            has_primary = any(
                e.component == component
                and e.campaign_id == primary.campaign_id
                and not e.is_completed
                for e in tl.active_entries
            )
            if not has_primary:
                continue

            timeline_mutated = False
            for entry in tl.active_entries:
                if not entry.is_backfill or entry.is_completed or entry.is_predicted:
                    continue

                entry_fp = str(getattr(entry, "config_fingerprint", "") or "")
                confidence = self._get_confidence(
                    entry.component,
                    entry.input_size,
                    gpu_id,
                    config_fingerprint=entry_fp,
                )
                mu = self._predict_latency(
                    entry.component,
                    entry.input_size,
                    gpu_id,
                    config_fingerprint=entry_fp,
                )
                if mu is None:
                    continue

                from .global_planner import _norm_ppf

                if confidence in ("high", "medium"):
                    sigma = (
                        self._predict_latency_sigma(
                            entry.component,
                            entry.input_size,
                            gpu_id,
                            config_fingerprint=entry_fp,
                        )
                        or 0.0
                    )
                    alpha = self._alpha_from_confidence(confidence)
                    entry.predicted_end_time = (
                        entry.start_time + mu + sigma * _norm_ppf(alpha)
                    )
                    entry.deadline_uncertain = False
                    timeline_mutated = True
                    n_recalced += 1
                else:
                    entry.deadline_uncertain = True
                    n_recalced += 1
            if timeline_mutated:
                tl._bump_state_version()

        if n_recalced > 0:
            _LOG.debug(
                "[campaign-scheduler-drift] recalculated %d backfill deadlines "
                "after %s drift",
                n_recalced,
                component,
            )
        return n_recalced

    def _predict_lookahead_vram(self, component: str) -> float:
        """Speculative VRAM for downstream lookahead entries.

        The lookahead pre-places FUTURE tasks, so a cold (no-GP) component has
        no known VRAM yet.  Reserve 0 instead of the cold-budget fallback
        (<rev>) which would otherwise block the CURRENT phase's backfill on
        that GPU.  The actual placement keeps the cold budget; only the
        speculative lookahead uses 0.
        """
        if self._signal_service is None:
            return 0.0
        profiles = self._signal_service.resource_profiles
        if profiles._profiles.get(component) is None:
            return 0.0
        return self._predict_vram(component, 0.0, "")

    def _project_downstream(
        self,
        campaign_id: str,
        component: str,
        input_size: float,
        is_backfill: bool,
        parent_gpu: str | None = None,
    ) -> int:
        """Project downstream tasks from DAG onto the timeline.

        When task A (component) is submitted, if the DAG shows A→B→C,
        and B and C haven't been submitted yet, project predicted entries
        for B and C using GP predictions. This extends the predictive
        scenario further into the future.

        Only projects for components not yet seen in this campaign's
        active/pending tasks (avoids duplicate projections).
        """
        if getattr(self, "_downstream_projection_disabled", False):
            return 0
        cq = self._campaign_queues.get(campaign_id)
        if not cq or not cq.dag_context:
            return 0

        downstream = cq.dag_context.downstream_map.get(component, [])
        if not downstream:
            return 0

        count = 0
        predicted_latency = self._predict_latency(component, input_size, "") or 0.0
        parent_end_time = time.time() + predicted_latency

        for next_comp in downstream:
            proj_id = f"__lookahead_{campaign_id[:8]}_{next_comp}"
            existing = self._timelines.find_entry(proj_id)
            if existing:
                continue

            next_latency = self._predict_latency(next_comp, 0.0, "") or 0.0
            next_vram = self._predict_lookahead_vram(next_comp)
            next_ram = self._predict_ram(next_comp, 0.0, "")

            best_gpu = self._timelines.best_gpu_for(
                next_vram if next_vram > 0 else 1.0,
                prefer_gpu=parent_gpu,
            )
            if not best_gpu and self._timelines.gpu_ids:
                best_gpu = self._timelines.gpu_ids[0]
            if not best_gpu:
                continue

            init_sec = self._get_init_latency(next_comp, best_gpu)
            start_time = parent_end_time + init_sec

            try:
                _ds_was_primary = bool(self._is_primary(campaign_id))
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "_ds_was_primary = False",
                    exc_info=True,
                )
                _ds_was_primary = False
            entry = TimelineEntry(
                task_id=proj_id,
                component=next_comp,
                campaign_id=campaign_id,
                gpu_id=best_gpu,
                start_time=start_time,
                predicted_end_time=start_time + next_latency,
                predicted_vram_mb=next_vram,
                predicted_ram_mb=next_ram,
                is_backfill=is_backfill,
                is_predicted=True,
                was_primary_at_dispatch=_ds_was_primary,
            )
            self._timelines.add_entry(entry)
            count += 1

            second_level = cq.dag_context.downstream_map.get(next_comp, [])
            for comp2 in second_level:
                proj_id2 = f"__lookahead_{campaign_id[:8]}_{comp2}"
                if self._timelines.find_entry(proj_id2):
                    continue
                lat2 = self._predict_latency(comp2, 0.0, "") or 0.0
                vram2 = self._predict_lookahead_vram(comp2)
                ram2 = self._predict_ram(comp2, 0.0, "")
                gpu2 = (
                    self._timelines.best_gpu_for(
                        vram2 if vram2 > 0 else 1.0,
                        prefer_gpu=best_gpu,
                    )
                    or best_gpu
                )
                start2 = start_time + next_latency
                entry2 = TimelineEntry(
                    task_id=proj_id2,
                    component=comp2,
                    campaign_id=campaign_id,
                    gpu_id=gpu2,
                    start_time=start2,
                    predicted_end_time=start2 + lat2,
                    predicted_vram_mb=vram2,
                    predicted_ram_mb=ram2,
                    is_backfill=is_backfill,
                    is_predicted=True,
                    was_primary_at_dispatch=_ds_was_primary,
                )
                self._timelines.add_entry(entry2)
                count += 1

        if count > 0:
            _LOG.debug(
                "[campaign-scheduler] projected %d downstream entries for %s",
                count,
                component,
            )
        return count

    def _reproject_downstream_on_dispatch(
        self,
        campaign_id: str,
        component: str,
        input_size: float,
        gpu_id: str,
        is_backfill: bool,
    ) -> None:
        """Re-place downstream lookahead entries on the actual dispatch GPU.

        ``on_task_submit`` projects the downstream using ``_project_task``'s
        spread heuristic, which can differ from the min-EFT dispatch GPU.
        After dispatch the real parent GPU is known, so remove the stale
        lookahead entries and re-project with ``parent_gpu=<actual gpu_id>``.
        """
        if getattr(self, "_downstream_projection_disabled", False):
            return
        if not campaign_id:
            return
        cq = self._campaign_queues.get(campaign_id)
        if not cq or not cq.dag_context:
            return
        downstream = cq.dag_context.downstream_map.get(component, [])
        if not downstream:
            return
        cid_prefix = campaign_id[:8]
        first = self._timelines.find_entry(f"__lookahead_{cid_prefix}_{downstream[0]}")
        if first is not None and str(first.gpu_id) == str(gpu_id):
            return
        for next_comp in downstream:
            self._timelines.remove_all_entries(f"__lookahead_{cid_prefix}_{next_comp}")
            for comp2 in cq.dag_context.downstream_map.get(next_comp, []):
                self._timelines.remove_all_entries(f"__lookahead_{cid_prefix}_{comp2}")
        self._project_downstream(
            campaign_id, component, input_size, is_backfill, gpu_id
        )

    def _reproject_pending(
        self,
        component: str,
        primary: CampaignQueue | None = None,
    ) -> int:
        """**DEPRECATED in  fix** — no longer called from drift
        cascade (`_drift_apply_per_component`).  Phantom entries it created
        (``__reproject_<campaign>_<component>``) are no longer needed:
        fresh GP posterior is applied by the next ``solve()`` call (wake
        triggers task_completion / periodic_tick / activation_ttl_expired
        all run solve() with up-to-date GP).  Per-call cost was O(K × N)
        (campaign records linear scan via ``cq.pending_tasks`` × 9 campaigns)
        — combined with  fix making MCPSE projection functional
        (more drift events from cancel/re-enqueue), the cumulative cascade
        cost saturated the asyncio event loop.

        Function body kept for historical reference + edge cases where
        offline / test code may want forecast phantoms.  Production drift
        cascade no longer invokes it.

        Re-project predicted entries for *component* using fresh GP predictions.

        Called after drift invalidation removes/stales predicted entries.
        Uses a stable task_id per (campaign, component) so repeated drift
        events replace rather than accumulate entries.

        Plan fix (C) — compute the primary campaign **once**
        per invocation and reuse the result for the ``is_backfill``
        classification of every pending queue.  Prior code called
        ``self._is_primary(cq.campaign_id)`` in the loop body, which in
        turn invokes ``_primary_campaign`` → ``SmithRuleCampaign.select_
        primary`` → ``is_empty`` on every campaign (each a K × M scan
        of ``_gateway_tasks``), producing an O(K² × M) blow-up under
        the drift cascade.  This method is synchronous (no ``await``)
        and runs inside a single asyncio tick, so the memoised primary
        cannot go stale mid-loop.

        Plan fix (F-xx) — *primary* may now also be supplied
        by ``on_profile_drift`` so the shared lookup happens **once per
        cascade** rather than once per helper.  Legacy callers (None
        default) fall back to the original self-lookup.
        """
        if getattr(self, "_speculative_task_projection_disabled", False):
            return 0
        if primary is None:
            primary = self._primary_campaign()
        primary_id = primary.campaign_id if primary is not None else None
        count = 0
        for cq in self._campaign_queues.values():
            if cq.pending_tasks <= 0:
                continue
            is_backfill = cq.campaign_id != primary_id
            was_primary = False
            stable_id = f"__reproject_{cq.campaign_id[:8]}_{component}"
            self._timelines.remove_all_entries(stable_id)
            gpu = self._project_task(
                task_id=stable_id,
                campaign_id=cq.campaign_id,
                component=component,
                input_size=0.0,
                is_backfill=is_backfill,
                was_primary=was_primary,
            )
            if gpu:
                count += 1
        return count

    def _get_worker_concurrency(self, component: str, gpu_id: str) -> int:
        """Get max concurrent tasks for a component on a GPU.

        Queries the supervisor's worker states if available.
        Returns 0 for unlimited (VRAM-bounded), >0 for explicit cap.
        Callers should treat 0 as "no concurrency limit".
        """
        cache = getattr(self, "_max_concurrency_cache", None)
        if cache is None:
            cache = {}
            self._max_concurrency_cache = cache
        key = (component, gpu_id)
        cached = cache.get(key)
        if cached is not None:
            return cached
        sup = getattr(self, "_supervisor", None)
        if sup:
            try:
                for st in sup.states.values():
                    if st.spec.component == component and gpu_id in [
                        str(g) for g in (st.spec.gpus or [])
                    ]:
                        cache[key] = st.max_concurrency
                        return st.max_concurrency
            except Exception:
                _LOG.warning(
                    "[silent-except] %s:%d (%s)",
                    __name__,
                    0,
                    "swallowed_pass",
                    exc_info=True,
                )
        cache[key] = 0
        return 0


    def _project_task(
        self,
        task_id: str,
        campaign_id: str,
        component: str,
        input_size: float,
        is_backfill: bool,
        was_primary: bool | None = None,
    ) -> str | None:
        """Project a pending task onto the best GPU timeline as a future entry.

        Finds the GPU where this task can start earliest (based on predicted
        VRAM availability and current timeline entries), then adds a projected
        TimelineEntry. Returns the chosen gpu_id or None if projection is
        not possible (no GP data, no GPU pool).

        Projected entries are visually distinct and get replaced when the
        actual dispatch happens (on_task_dispatched removes the projected
        entry and adds the real one).
        """
        if getattr(self, "_speculative_task_projection_disabled", False):
            return None
        predicted_vram = self._predict_vram(component, input_size, "")
        predicted_ram = self._predict_ram(component, input_size, "")

        if not self._timelines.gpu_ids:
            return None

        confidence = self._get_confidence(component, input_size, "")
        if confidence == "high":
            horizon = 120.0
        elif confidence == "medium":
            horizon = 30.0
        else:
            horizon = 10.0

        best_gpu: str | None = None
        best_key: tuple = (math.inf, math.inf, math.inf)
        now = time.time()

        for gpu_id in self._timelines.gpu_ids:
            tl = self._timelines.get(gpu_id)
            if tl is None:
                continue


            gpu_vram = self._predict_vram(component, input_size, gpu_id)
            gpu_ram = self._predict_ram(component, input_size, gpu_id)
            effective_vram = gpu_vram if gpu_vram > 0 else predicted_vram
            effective_ram = gpu_ram if gpu_ram > 0 else predicted_ram

            fit_time = self._timelines.earliest_dual_fit_time(
                gpu_id,
                effective_vram if effective_vram > 0 else 1.0,
                effective_ram if effective_ram > 0 else 0.0,
                horizon_sec=horizon,
                required_duration_sec=(
                    self._predict_latency(component, input_size, gpu_id) or 0.0
                ),
                candidate_force_full_wall=True,
            )
            if fit_time is not None:
                can_fit_now = (fit_time - now) < 0.5
                key = (
                    0 if can_fit_now else 1,
                    tl.planned_occupancy_count(now),
                    fit_time,
                )
                if key < best_key:
                    best_key = key
                    best_gpu = gpu_id

        if best_gpu is None:
            best_gpu = self._timelines.best_gpu_for(0.0)

        if best_gpu is None:
            return None

        gpu_latency = self._predict_latency(component, input_size, best_gpu) or 0.0
        gpu_vram = self._predict_vram(component, input_size, best_gpu)
        gpu_ram = self._predict_ram(component, input_size, best_gpu)

        best_fit_time = best_key[2] if len(best_key) > 2 else time.time()
        now = time.time()
        start = max(now, best_fit_time) if best_fit_time < math.inf else now

        if was_primary is None:
            try:
                _was_primary = bool(self._is_primary(campaign_id))
            except Exception:
                _LOG.warning(
                    "[silent-except] %s swallowed an exception; body=%s",
                    __name__,
                    "_was_primary = False",
                    exc_info=True,
                )
                _was_primary = False
        else:
            _was_primary = bool(was_primary)
        self._timelines.add_predicted_entry(
            task_id=task_id,
            component=component,
            gpu_id=best_gpu,
            worker_name="",
            start_time=start,
            predicted_end_time=start + gpu_latency,
            predicted_vram_mb=gpu_vram,
            predicted_ram_mb=gpu_ram,
            campaign_id=campaign_id,
            is_backfill=is_backfill,
            input_size=input_size,
            was_primary_at_dispatch=_was_primary,
        )

        return best_gpu


    def update_dag_context(self, campaign_id: str, dag_context: DAGContext) -> None:
        """Update the DAG context for a campaign."""
        cq = self._get_or_create(campaign_id)
        cq.dag_context = dag_context


    _MIN_GPU_CONFIDENCE = 0.3

    @staticmethod
    def _config_prediction_candidates(
        profile: Any, config_fingerprint: str = ""
    ) -> list[Any]:
        baselines = getattr(profile, "_config_baselines", {}) or {}
        if not baselines:
            return []
        ordered_keys: list[str] = []
        fp = str(config_fingerprint or "").strip()
        if fp:
            ordered_keys.append(fp)
        ordered_keys.append("__default__")
        ordered_keys.extend(sorted(str(key) for key in baselines))
        seen: set[str] = set()
        candidates: list[Any] = []
        for key in ordered_keys:
            if key in seen:
                continue
            seen.add(key)
            cfg = baselines.get(key)
            if cfg is not None:
                candidates.append(cfg)
        return candidates

    def _predict_latency(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        config_fingerprint: str = "",
    ) -> float | None:
        """Get GP latency prediction with confidence-based fallback.

        Returns ``cold_inference_latency_sec`` (workers.yaml shared
        config, default 30s) when the signal service is absent or the
        component has no observations cluster-wide — a genuine cold-
        start.  Previously returned ``None`` and forced every caller to
        chain ``or 0.0`` fallback (13+ sites), which silently zero-d the
        EFT μ component and triggered the "RFD co-location on cold"
        pathology (cold path EFT_min always wins over warm path).  By
        returning the configured cold default here, every caller (b-rank,
        EFT, wave, evict, MCPSE) automatically picks up the same
        cold-start fallback.
        """
        cold_fallback = self._cold_inference_latency_sec()
        if self._signal_service is None:
            return cold_fallback
        profiles = self._signal_service.resource_profiles
        profile = profiles._profiles.get(component)
        if profile is None:
            return cold_fallback
        for cfg in self._config_prediction_candidates(profile, config_fingerprint):
            if gpu_id:
                bl = cfg.resolve(gpu_id)
                pool = cfg.resolve()
                if bl and bl is not pool:
                    conf = bl.latency_confidence(input_size)
                    if conf >= self._MIN_GPU_CONFIDENCE:
                        result = cfg.predict_latency(input_size, gpu_id=gpu_id)
                        if result is not None and result > 0:
                            return _safe_float(result)
            result = cfg.predict_latency(input_size, gpu_id="")
            if result is not None and result > 0:
                return _safe_float(result)
        return cold_fallback

    def _predict_latency_sigma(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        config_fingerprint: str = "",
    ) -> float | None:
        """Get GP latency std dev.  Returns None only for a genuine
        cold-start (no component profile, or the GP has zero
        observations cluster-wide).  Coding bugs raise — see
        ``_predict_latency`` docstring.
        """
        if self._signal_service is None:
            return None
        profiles = self._signal_service.resource_profiles
        profile = profiles._profiles.get(component)
        if profile is None:
            return None
        for cfg in self._config_prediction_candidates(profile, config_fingerprint):
            if gpu_id:
                bl = cfg.resolve(gpu_id)
                pool = cfg.resolve()
                if bl and bl is not pool:
                    conf = bl.latency_confidence(input_size)
                    if conf >= self._MIN_GPU_CONFIDENCE:
                        _, var = bl._cached_latency_predict(input_size)
                        if var is None:
                            return None
                        return var**0.5
            bl = cfg.resolve()
            if bl and bl.latency_sec.n > 0:
                _, var = bl._cached_latency_predict(input_size)
                if var is None:
                    return None
                return var**0.5
        return None

    def _predict_vram(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        *,
        use_upper: bool = False,
        config_fingerprint: str = "",
    ) -> float:
        """Get the GP mean VRAM prediction with confidence-based fallback.

        Normal reservations use the GP mean; ``use_upper`` remains available
        for explicit diagnostic callers.

        1. Per-GPU prediction if GP confidence(input_size) >= threshold.
        2. Cross-GPU pool fallback if per-GPU confidence is too low.
        3. Existing guarded cold-start reservation when GP evidence is absent.

        Coding bugs raise — see ``_predict_latency``.
        """
        if self._signal_service is None:
            return self._cold_vram_budget_mb(component, gpu_id)
        profiles = self._signal_service.resource_profiles
        profile = profiles._profiles.get(component)
        if profile is None:
            return self._cold_vram_budget_mb(component, gpu_id)
        resource_z = self.resource_upper_z
        for cfg in self._config_prediction_candidates(profile, config_fingerprint):
            if gpu_id:
                bl = cfg.resolve(gpu_id)
                pool = cfg.resolve()
                if bl and bl is not pool:
                    conf = bl.vram_confidence(input_size)
                    if conf >= self._MIN_GPU_CONFIDENCE:
                        result = (
                            cfg.predict_vram_upper(
                                input_size,
                                gpu_id=gpu_id,
                                z=resource_z,
                            )
                            if use_upper
                            else cfg.predict_vram(input_size, gpu_id=gpu_id)
                        )
                        if result is not None and result > 0:
                            return _safe_float(result)
            result = (
                cfg.predict_vram_upper(input_size, gpu_id="", z=resource_z)
                if use_upper
                else cfg.predict_vram(input_size, gpu_id="")
            )
            if result is not None and result > 0:
                return _safe_float(result)
        return self._cold_vram_budget_mb(component, gpu_id)

    def _cold_vram_budget_mb(self, component: str, gpu_id: str) -> float:
        """Match cold planner reservations to the supervisor's guarded budget."""
        supervisor = getattr(self, "_supervisor", None)
        if supervisor is None:
            return 0.0
        target_gpu = str(gpu_id)
        ratio = (
            _safe_float(getattr(supervisor, "cold_start_activation_ratio", 1.0)) or 1.0
        )
        for state in getattr(supervisor, "states", {}).values():
            spec = getattr(state, "spec", None)
            if str(getattr(spec, "component", "")) == component and target_gpu in {
                str(gpu) for gpu in getattr(state, "assigned_gpus", ())
            }:
                weight_mb = _safe_float(getattr(state, "memory_reserved_mb", 0))
                if weight_mb > 0:
                    return weight_mb * ratio
        guard = getattr(supervisor, "weight_oom_guard", None)
        if guard is None:
            return 0.0
        timeline = self._timelines.get_or_create(target_gpu)
        try:
            return _safe_float(
                guard.get_reservation_mb(component, int(timeline.total_vram_mb or 0))
            )
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _predict_ram(
        self,
        component: str,
        input_size: float,
        gpu_id: str = "",
        *,
        use_upper: bool = True,
        config_fingerprint: str = "",
    ) -> float:
        """Get GP CPU RAM budget, mirroring ``_predict_vram`` semantics."""
        if self._signal_service is None:
            return 0.0
        profiles = self._signal_service.resource_profiles
        profile = profiles._profiles.get(component)
        if profile is None:
            return 0.0
        resource_z = self.resource_upper_z
        for cfg in self._config_prediction_candidates(profile, config_fingerprint):
            if gpu_id:
                bl = cfg.resolve(gpu_id)
                pool = cfg.resolve()
                if bl and bl is not pool:
                    conf = bl.ram_confidence(input_size)
                    if conf >= self._MIN_GPU_CONFIDENCE:
                        result = (
                            cfg.predict_ram_upper(
                                input_size,
                                gpu_id=gpu_id,
                                z=resource_z,
                            )
                            if use_upper
                            else cfg.predict_ram(input_size, gpu_id=gpu_id)
                        )
                        if result is not None and result > 0:
                            return _safe_float(result)
            result = (
                cfg.predict_ram_upper(input_size, gpu_id="", z=resource_z)
                if use_upper
                else cfg.predict_ram(input_size, gpu_id="")
            )
            if result is not None and result > 0:
                return _safe_float(result)
        return 0.0

    @staticmethod
    def _alpha_from_confidence(confidence: str) -> float:
        """Map GP confidence label to continuous α (one-sided quantile level).

        Plan A Note on α — α is a continuous value, not a
        categorical bucket.  The 3-label bucket here is a transitional
        shim until the GP confidence is exposed as a continuous score;
        plan still specifies the α used per bucket in  α mapping:
        high ⇒ 0.95 (Φ⁻¹=1.645), medium ⇒ 0.80, low ⇒ 0.50 (μ-only).
        """
        if confidence == "high":
            return 0.95
        if confidence == "medium":
            return 0.80
        return 0.50

    def _get_confidence(
        self,
        component: str,
        input_size: float,
        gpu_id: str,
        config_fingerprint: str = "",
    ) -> str:
        """Get confidence label for a component's GP predictions.

        Returns ``"low"`` only for genuine cold-start (signal service
        absent, no component profile, or all GPs have insufficient
        observations).  Coding bugs raise.
        """
        if self._signal_service is None:
            return "low"
        profiles = self._signal_service.resource_profiles
        profile = profiles._profiles.get(component)
        if profile is None:
            return "low"
        for cfg in self._config_prediction_candidates(profile, config_fingerprint):
            bl = cfg.resolve(gpu_id)
            if not bl:
                bl = cfg.resolve()
            if bl:
                v_suff = bl.is_vram_sufficient(input_size, 0.10)
                l_suff = bl.is_latency_sufficient(input_size, 0.10)
                if v_suff and l_suff:
                    return "high"
                v_med = bl.is_vram_sufficient(input_size, 0.20)
                l_med = bl.is_latency_sufficient(input_size, 0.20)
                if v_med and l_med:
                    return "medium"
        return "low"

    def _get_init_latency(self, component: str, gpu_id: str) -> float:
        """Get initialization latency estimate.

        Returns the tracker's μ when available, otherwise the Category
        A ``default_init_sec`` fallback (Plan Bootstrapping).  No
        silent exception swallow — coding bugs raise; the only path
        that returns the fallback is "tracker has no observation yet".

        Supervisor-wired override: when ``self._supervisor.cold_init_
        latency_sec`` is available (workers.yaml shared.cold_init_
        latency_sec) it takes precedence over the ctor default.
        """
        sup = getattr(self, "_supervisor", None)
        cold_init = getattr(sup, "cold_init_latency_sec", None)
        fallback = (
            _safe_float(cold_init) if cold_init is not None else self._default_init_sec
        )
        if self._init_tracker is None:
            return fallback
        mu, _ = self._init_tracker.predict(component, gpu_id)
        if mu and mu > 0:
            return _safe_float(mu)
        return fallback

    def _cold_inference_latency_sec(self) -> float:
        """Single global cold-inference fallback (Category A, post ).

        Consumed by `evaluate_gpu` 's `_predict_latency or <fallback>`
        fallback site so genuine cold-start (zero observations on any GPU)
        gets a non-zero EFT contribution — prevents the "EFT_cold ≈
        init_cost (10s) + 0" pathology that drove RFD co-location on cold.

        Wired from ``workers.yaml shared.cold_inference_latency_sec`` via
        supervisor reference.  Default: 30s.
        """
        sup = getattr(self, "_supervisor", None)
        v = getattr(sup, "cold_inference_latency_sec", None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 30.0
        return 30.0


    def cleanup_campaign(self, campaign_id: str) -> None:
        """Remove a completed campaign from tracking."""
        self._campaign_queues.pop(campaign_id, None)


    def as_dict(
        self,
        *,
        include_retired: bool = True,
        retired_limit: int | None = 128,
    ) -> dict[str, Any]:
        self._reconcile_campaign_live_counters(force=True)
        self._reconcile_actual_timeline_entries()
        primary = self._primary_campaign()
        visible = {
            cid: cq
            for cid, cq in self._campaign_queues.items()
            if not cq.is_empty or cq.completed_tasks > 0
        }
        return {
            "primary_campaign": primary.campaign_id if primary else None,
            "campaign_count": len(visible),
            "campaigns": {
                cid: cq.as_dict()
                for cid, cq in sorted(visible.items(), key=lambda x: x[1].arrival_time)
            },
            "gpu_timelines": self._timelines.as_dict(
                include_retired=include_retired,
                retired_limit=retired_limit,
            ),
            "makespan_admission": {
                "last_cost": round(self._last_admission_cost, 2),
                "last_benefit": round(self._last_admission_benefit, 2),
                "last_admitted": self._last_admission_result,
            },
        }




@dataclass
class _PendingBatch:
    """Buffer for fan-out tasks awaiting joint placement."""

    campaign_id: str
    component: str
    tasks: list[tuple[str, float]]
    first_arrival: float
    buffer_ms: float = 1000.0

    @property
    def window_elapsed(self) -> bool:
        return (time.time() - self.first_arrival) >= (self.buffer_ms / 1000.0)
