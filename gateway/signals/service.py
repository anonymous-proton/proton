from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..profiling.predict_cost import predict_cost
from ..profiling.runtime_regime import normalize_worker_context
from .contracts import (
    MemorySignal,
    PlannerIntent,
    ProfileDriftEvent,
    ReciprocalInterferenceQuery,
    RuntimeSignal,
    SignalArtifacts,
    SignalBundle,
    SignalProvenance,
    SignalResult,
)
from .interference import (
    InterferencePrediction,
    InterferenceRegistry,
    WorkloadClassifier,
)
from .latency_tracker import (
    LatencyObservation,
    LatencyTaskIdentity,
    WorkerLatencyTracker,
)
from .resource_profile import ResourceProfileRegistry

_RESOURCE_MEASUREMENT_SEMANTICS = "request_process_tree_v1"


_F = TypeVar("_F", bound=Callable[..., Any])


def _locked(method: _F) -> _F:
    """Decorator: acquire ``self._lock`` for the call duration.

    Applied only to *mutation* methods on ``SignalService``.  The
    decorator wraps ``method`` in a ``with self._lock:`` block, so
    re-entry from the same thread (RLock) is fine, but the wrapped
    method body must not ``await`` anything (see invariant 2 above).
    """

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped




@dataclass
class GPPosteriorSnapshot:
    """Immutable GP posterior snapshot for a single Planner ``solve()`` call.

    Plan (v3 Blocker 1).  Prevents re-queries
    during solve() from observing mid-flight GP updates, which would
    make EFT comparisons inconsistent across the same planning pass
    (Omega-style optimistic concurrency).
    """

    taken_at: float
    predictions: dict[tuple[str, str, int], tuple[float, float]] = field(
        default_factory=dict
    )
    init_predictions: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    interference: dict[tuple[str, str], float] = field(default_factory=dict)

    def age_sec(self, now: float | None = None) -> float:
        ref = now if now is not None else time.time()
        return max(0.0, ref - self.taken_at)


_LOG = logging.getLogger(__name__)
_CANONICAL_RUNTIME_LEVEL = "level_a"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _to_float(value: Any) -> float | None:
    try:
        parsed = _as_float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return _as_float(parsed)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _positive_mapping_int(mapping: Mapping[str, Any], name: str) -> int | None:
    raw = mapping.get(name)
    if raw in (None, ""):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _batch_selected_sizes(
    context: Mapping[str, Any] | None,
) -> tuple[int, int] | None:
    if not isinstance(context, Mapping):
        return None
    raw_execution_k = context.get("execution_batch_size")
    raw_logical_n = context.get("logical_batch_size")
    if raw_execution_k is None or raw_logical_n is None:
        return None
    try:
        execution_k = int(raw_execution_k)
        logical_n = int(raw_logical_n)
    except (TypeError, ValueError):
        return None
    if execution_k <= 0 or logical_n < execution_k:
        return None
    return execution_k, logical_n


def _batch_observation_point(
    context: Mapping[str, Any] | None,
    input_size: float,
) -> tuple[float, int, int] | None:
    selected = _batch_selected_sizes(context)
    if (
        selected is None
        or input_size <= 0
        or not isinstance(context, Mapping)
        or not bool(context.get("companion_eligible", False))
    ):
        return None
    return input_size, selected[0], selected[1]


def _canonical_memory_semantics(
    *,
    memory_basis: Any = None,
    peak_fidelity: Any = None,
    legacy_scope: Any = None,
) -> tuple[str, str]:
    normalized_basis = str(memory_basis or "").strip().lower()
    if normalized_basis in {"", "execution_envelope", "active_total_execution_peak"}:
        normalized_basis = "total_execution_peak"
    elif normalized_basis not in {"increment_over_resident", "total_execution_peak"}:
        normalized_basis = normalized_basis or str(legacy_scope or "").strip().lower()
    normalized_fidelity = str(peak_fidelity or "").strip().lower()
    if not normalized_basis:
        legacy_token = str(legacy_scope or "").strip().lower()
        if legacy_token == "increment_over_resident":
            normalized_basis = "increment_over_resident"
        elif legacy_token:
            normalized_basis = (
                "total_execution_peak"
                if legacy_token
                in {
                    "total_execution_peak",
                    "execution_envelope",
                    "active_total_execution_peak",
                }
                else legacy_token
            )
        else:
            normalized_basis = "total_execution_peak"
    if not normalized_fidelity:
        if normalized_basis == "increment_over_resident":
            normalized_fidelity = "incremental_peak"
        elif normalized_basis == "total_execution_peak":
            normalized_fidelity = "full_peak"
        else:
            normalized_fidelity = "observed_peak"
    return (normalized_basis, normalized_fidelity)


class SignalService:
    """Central signal service for resource prediction and runtime tracking.

    Manages three signal channels:

    1. **VRAM activation peak** — tracks per-worker peak activation memory
       (fed by supervisor stats loop, consumed at task completion).
    2. **Latency tracking** — tracks per-task execution latency with
       co-location context (which models share the GPU).
    3. **Interference modeling** — learns pairwise slowdown factors between
       co-located components and predicts interference for scheduling.
    """

    def __init__(self, *, run_index: Any, history_limit: int = 400) -> None:
        self._run_index = run_index
        self._history_limit = max(1, _as_int(history_limit))

        self._peak_activation_mb: dict[str, float] = {}

        self._latency_trackers: dict[str, WorkerLatencyTracker] = {}

        self._gpu_active_tasks: dict[str, set[tuple[str, str, str]]] = {}

        self._observation_counters: dict[str, int] = {
            "ingested": 0,
            "ingest_failures": 0,
            "variable_composition_skips": 0,
            "multi_neighbor_skips": 0,
            "prior_corrected_skips": 0,
        }

        self._workload_classifier = WorkloadClassifier()
        self._interference_registry = InterferenceRegistry(self._workload_classifier)

        self._resource_profiles = ResourceProfileRegistry()

        self._interference_registry.set_signal_service_ref(self)

        self._drift_callbacks: list[Callable[[ProfileDriftEvent], None]] = []
        self._interference_registry.register_evidence_callback(
            self._on_interference_evidence
        )

        self._bootstrapping: bool = False

        self._drift_cooldown_sec: float = 10.0
        self._last_drift_at: dict[str, tuple] = {}
        self._last_drift_callback_at: dict[str, float] = {}

        self._drift_coalesce_sec: float = 1.0
        self._drift_max_components_per_flush: int = 6
        self._drift_pending: dict[str, ProfileDriftEvent] = {}
        self._drift_flush_handle: Any | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._drift_queue: queue.Queue[
            tuple[dict[str, dict[str, Any]], dict[str, Any]]
        ] = queue.Queue()
        self._drift_worker_started = False

        self._as_dict_cache: dict[str, Any] | None = None
        self._as_dict_cache_at: float = 0.0
        self._as_dict_cache_ttl_sec: float = 1.0

        self._lock = threading.RLock()

        from ..profiling.predict_cost import set_interference_correction

        set_interference_correction(self._correct_concurrent_runtime)

    def _correct_concurrent_runtime(
        self, component: str, runtime_sec: float
    ) -> float | None:
        """Decontaminate a concurrent runtime observation to solo-equivalent.

        Called by ``predict_cost._hot_execution_time_sec`` for rows with
        ``concurrent_execute_overlap=True``.  Uses the interference model's
        known slowdown for this component to reverse the inflation:

            solo_equivalent ≈ runtime_sec / (1 + predicted_slowdown)

        Returns None if no correction is possible (unknown component).
        """
        return None


    @_locked
    def record_activation_peak(
        self,
        addr: str,
        activation_mb: float,
        *,
        exec_inflight: int = 1,
    ) -> None:
        """Called from supervisor stats loop when exec_in>0 and weight is known.

        When ``exec_inflight == 1`` (solo): direct attribution — the sole task
        owns all activation.

        When ``exec_inflight > 1`` (concurrent): attributes per-task activation
        shares using baselines from the interference registry and co-location
        context from the latency tracker.

        Attribution strategies (in priority order):
        1. **Homogeneous** — all tasks are the same component → equal division.
        2. **Baseline-proportional** — all mixed-component tasks have known
           solo VRAM baselines → proportion by baseline ratio.

        Mixed-component partial peaks with missing VRAM baselines are skipped.
        Latency baselines are not VRAM proxies; using them here would silently
        turn a timing signal into a memory observation.
        """
        if exec_inflight <= 1:
            current = self._peak_activation_mb.get(addr, 0.0)
            if activation_mb > current:
                self._peak_activation_mb[addr] = activation_mb
            return

        tracker = self._latency_trackers.get(addr)
        if tracker is None or tracker.active_count == 0:
            return

        slots = list(tracker._slots.values())
        n = len(slots)
        if n == 0:
            return

        components = [s.component for s in slots]
        task_ids = [s.task_id for s in slots]

        if len(set(components)) == 1:
            per_task = activation_mb / n
            for tid in task_ids:
                key = f"{addr}:{tid}"
                current = self._peak_activation_mb.get(key, 0.0)
                if per_task > current:
                    self._peak_activation_mb[key] = per_task
            return

        baselines = []
        for comp in components:
            b = self._interference_registry.get_solo_vram_baseline(comp)
            baselines.append(b)

        known_sum = sum(b for b in baselines if b is not None and b > 0)
        n_unknown = sum(1 for b in baselines if b is None or b <= 0)

        if known_sum > 0 and n_unknown == 0:
            for index, tid in enumerate(task_ids):
                share = activation_mb * (_as_float(baselines[index]) / known_sum)
                key = f"{addr}:{tid}"
                current = self._peak_activation_mb.get(key, 0.0)
                if share > current:
                    self._peak_activation_mb[key] = share
            return

        _LOG.debug(
            "[activation-peak] skip mixed concurrent attribution without "
            "complete VRAM baselines addr=%s components=%s activation_mb=%.1f",
            addr,
            components,
            activation_mb,
        )

    @_locked
    def consume_activation_peak(self, addr: str, task_id: str = "") -> float | None:
        """Called at task completion: returns and clears the peak activation.

        For solo tasks, uses the worker-level key ``addr``.
        For concurrent tasks, uses the task-specific key ``addr:task_id``.
        Falls back to the worker-level key if no task-specific peak exists.
        """
        if task_id:
            task_key = f"{addr}:{task_id}"
            peak = self._peak_activation_mb.pop(task_key, None)
            if peak is not None:
                return peak
        return self._peak_activation_mb.pop(addr, None)


    def _get_latency_tracker(self, addr: str) -> WorkerLatencyTracker:
        tracker = self._latency_trackers.get(addr)
        if tracker is None:
            tracker = WorkerLatencyTracker()
            self._latency_trackers[addr] = tracker
        return tracker

    def _get_gpu_co_located_components(
        self,
        gpu_ids: list[str],
        exclude_addr: str = "",
    ) -> frozenset[str]:
        """Legacy component-set view; segment tracking uses task instances."""
        return frozenset(
            comp
            for gid in gpu_ids
            for _addr, _tid, comp in self._gpu_active_tasks.get(gid, set())
            if _addr != exclude_addr
        )

    def _active_slot(self, addr: str, task_id: str) -> Any:
        tracker = self._latency_trackers.get(addr)
        return getattr(tracker, "_slots", {}).get(task_id) if tracker else None

    def _refresh_gpu_segments(self, gpu_ids: list[str], now_mono: float) -> None:
        """Apply one arrival/departure boundary to every affected task instance."""
        active_keys = {
            (addr, task_id)
            for gid in gpu_ids
            for addr, task_id, _component in self._gpu_active_tasks.get(gid, set())
        }
        for addr, task_id in active_keys:
            slot = self._active_slot(addr, task_id)
            if slot is None:
                raise RuntimeError(
                    f"GPU latency membership has no slot: {addr}/{task_id}"
                )
            peer_keys = {
                (peer_addr, peer_task_id)
                for gid in slot.gpu_ids
                for peer_addr, peer_task_id, _component in self._gpu_active_tasks.get(
                    gid, set()
                )
                if (peer_addr, peer_task_id) != (addr, task_id)
            }
            peers: list[LatencyTaskIdentity] = []
            for peer_addr, peer_task_id in sorted(peer_keys):
                peer_slot = self._active_slot(peer_addr, peer_task_id)
                if peer_slot is None:
                    raise RuntimeError(
                        "GPU latency peer membership has no slot: "
                        f"{peer_addr}/{peer_task_id}"
                    )
                peers.append(peer_slot.identity())
            self._latency_trackers[addr].set_task_neighbors(
                task_id, peers, now_mono=now_mono
            )

    @_locked
    def latency_on_task_enter(
        self,
        addr: str,
        *,
        task_id: str,
        component: str,
        registry_key: tuple[str, str, str],
        workload_features: dict[str, Any],
        config_fingerprint: str,
        campaign_id: str = "",
        gpu_ids: list[str] | None = None,
        gpu_id: str | None = None,
        input_fingerprint: str = "",
        gpu_model: str = "",
        mps_mode: str = "",
        worker_backend: str = "",
        actor_model: str = "",
        adapter_version: str = "",
    ) -> None:
        """Called when a task begins execution on a worker.

        Records the start time and co-location context for latency tracking.
        Co-location is determined at **GPU level** — all components executing
        on the same GPU(s) across any worker are considered co-located.
        """
        effective_gpu_ids = [str(item) for item in (gpu_ids or [])]
        if any(
            active_task_id == task_id
            for entries in self._gpu_active_tasks.values()
            for _active_addr, active_task_id, _component in entries
        ):
            raise ValueError(f"duplicate active latency task_id: {task_id}")
        now_mono = time.monotonic()
        tracker = self._get_latency_tracker(addr)
        tracker.on_task_enter(
            task_id=task_id,
            component=component,
            registry_key=registry_key,
            workload_features=workload_features,
            config_fingerprint=config_fingerprint,
            campaign_id=campaign_id,
            gpu_ids=effective_gpu_ids,
            gpu_id=gpu_id,
            input_fingerprint=input_fingerprint,
            gpu_model=gpu_model,
            mps_mode=mps_mode,
            worker_backend=worker_backend,
            actor_model=actor_model,
            adapter_version=adapter_version,
            now_mono=now_mono,
        )
        slot = tracker._slots[task_id]
        entry = (addr, task_id, component)
        for gid in slot.gpu_ids:
            self._gpu_active_tasks.setdefault(gid, set()).add(entry)
        self._refresh_gpu_segments(slot.gpu_ids, now_mono)

    @_locked
    def latency_drop_task(self, addr: str, task_id: str) -> None:
        """Plan fix — discard a task's latency slot without
        emitting an observation.

        Called from the dispatcher's finally block when ``record._cancel_reason``
        is non-empty.  ``_cancel_reason`` is the single source of truth for
        non-natural task exit: cooperative cancel (D6 ``mcpse_eviction`` /
        PrimaryWeighted ``primary_protection`` / user explicit), guard-kill
        (``__main__._on_worker_killed_handler`` sets
        ``worker_killed_reenqueue``), and any future cancel reason all funnel
        through this attribute.  The task's measured
        ``duration_sec`` is **truncated** (mid-execution termination), so
        feeding it to ``record_observation`` would contaminate three downstream
        GP signals:

          1. ``InterferenceRegistry._solo_baselines`` (solo baseline GP) —
             affects ``is_self_mature.solo_exists`` and slowdown delta calc.
          2. ``InterferenceRegistry._self_slowdown_obs`` /
             ``_pairwise[(c1, c2)]`` (interference GP) — affects
             ``is_self_mature.self_observation_count`` and D2 EFT
             slowdown penalty.
          3. ``ResourceProfileRegistry._config_baselines.*.predict_latency``
             (latency GP) — affects D2 EFT μ_lat / σ_lat for placement.

        Pre-check moved to caller (``http_server._dispatch_selected_worker``
        finally) keeps the cancellation source-of-truth (``_cancel_reason``,
        ``killed_at``) where it already lives — no dependency injection
        into tracker / registry.

        Side effects:
          * Pop the slot from ``WorkerLatencyTracker._slots`` so subsequent
            ``on_task_enter`` for the same task_id (post-re-enqueue) starts
            fresh.
          * Clean up ``_gpu_active_tasks`` entries identical to the cleanup
            in ``latency_on_task_exit``.
          * No call to ``record_observation`` / ``record_latency`` /
            ``_emit_drift_event``.
        """
        tracker = self._latency_trackers.get(addr)
        slot = self._active_slot(addr, task_id)
        affected_gpu_ids: list[str] = []
        if slot is not None:
            affected_gpu_ids.extend(str(gpu_id) for gpu_id in slot.gpu_ids)
        now_mono = time.monotonic()
        for gid in list(self._gpu_active_tasks):
            self._gpu_active_tasks[gid] = {
                e
                for e in self._gpu_active_tasks[gid]
                if not (e[0] == addr and e[1] == task_id)
            }
            if not self._gpu_active_tasks[gid]:
                del self._gpu_active_tasks[gid]
        if tracker is None:
            return
        tracker.drop_task(task_id, now_mono=now_mono)
        self._refresh_gpu_segments(affected_gpu_ids, now_mono)

    @_locked
    def latency_on_task_exit(
        self,
        addr: str,
        task_id: str,
        batch_context: Mapping[str, Any] | None = None,
    ) -> LatencyObservation | None:
        """Called when a task completes on a worker.

        Returns the latency observation (duration + co-location context) and
        feeds it to the interference registry for learning.
        """
        tracker = self._latency_trackers.get(addr)
        if tracker is None:
            return None
        slot = self._active_slot(addr, task_id)
        if slot is None:
            obs = tracker.on_task_exit(task_id)
        else:
            affected_gpu_ids = list(slot.gpu_ids)
            now_mono = time.monotonic()
            for gid in list(self._gpu_active_tasks):
                self._gpu_active_tasks[gid] = {
                    e
                    for e in self._gpu_active_tasks[gid]
                    if not (e[0] == addr and e[1] == task_id)
                }
                if not self._gpu_active_tasks[gid]:
                    del self._gpu_active_tasks[gid]

            obs = tracker.on_task_exit(task_id, now_mono=now_mono)
            self._refresh_gpu_segments(affected_gpu_ids, now_mono)
        if obs is None:
            return None

        self._record_latency_observation(
            obs,
            record_interference=True,
            batch_context=batch_context,
        )
        return obs

    def _integrate_segment_runtime(
        self, obs: LatencyObservation
    ) -> tuple[float, float, tuple[str, ...], bool] | None:
        segments = obs.composition_segments
        if not segments:
            return None
        corrected = 0.0
        variance = 0.0
        provenances: list[str] = []
        all_exact = True
        gpu_id = obs.gpu_ids[0] if len(obs.gpu_ids) == 1 else ""
        for segment in segments:
            duration = segment.duration_sec
            if duration <= 0.0:
                continue
            multiplier = 1.0
            uncertainties: list[float] = []
            same_peers = [
                peer for peer in segment.neighbors if peer.component == obs.component
            ]
            if same_peers:
                result = self._interference_registry.query_reciprocal(
                    ReciprocalInterferenceQuery(
                        victim_component=obs.component,
                        victim_config_fingerprint=obs.config_fingerprint,
                        victim_input_fingerprint=obs.input_fingerprint,
                        gpu_id=gpu_id,
                        gpu_model=obs.gpu_model,
                        mps_mode=obs.mps_mode,
                        worker_backend=obs.worker_backend,
                        actor_model=obs.actor_model,
                        adapter_version=obs.adapter_version,
                        self_n=len(same_peers) + 1,
                    )
                )
                multiplier += result.delta
                uncertainties.append(result.uncertainty)
                provenances.append(result.provenance)
                all_exact = all_exact and result.provenance == "exact"
            for peer in segment.neighbors:
                if peer.component == obs.component:
                    continue
                result = self._interference_registry.query_reciprocal(
                    ReciprocalInterferenceQuery(
                        victim_component=obs.component,
                        victim_config_fingerprint=obs.config_fingerprint,
                        victim_input_fingerprint=obs.input_fingerprint,
                        interferer_component=peer.component,
                        interferer_config_fingerprint=peer.config_fingerprint,
                        interferer_input_fingerprint=peer.input_fingerprint,
                        gpu_id=gpu_id,
                        gpu_model=obs.gpu_model,
                        mps_mode=obs.mps_mode,
                        worker_backend=obs.worker_backend,
                        actor_model=obs.actor_model,
                        adapter_version=obs.adapter_version,
                    )
                )
                multiplier += result.delta
                uncertainties.append(result.uncertainty)
                provenances.append(result.provenance)
                all_exact = all_exact and result.provenance == "exact"
            corrected += duration / multiplier
            segment_uncertainty = math.sqrt(
                sum(value * value for value in uncertainties)
            )
            variance += (
                duration * segment_uncertainty / (multiplier * multiplier)
            ) ** 2
        return corrected, variance, tuple(provenances), all_exact

    def _record_latency_observation(
        self,
        obs: LatencyObservation,
        *,
        record_interference: bool,
        batch_context: Mapping[str, Any] | None = None,
    ) -> None:
        """Feed one latency observation through PROTON's correction path."""
        input_size = ResourceProfileRegistry.extract_input_size(
            obs.workload_features, component=obs.component
        )

        obs_gpu_id = obs.gpu_ids[0] if len(obs.gpu_ids) == 1 else None
        _LOG.info(
            "[resource-profile] %s input_size=%.1f gpu_id=%s workload_features=%s",
            obs.component,
            input_size,
            obs_gpu_id,
            obs.workload_features,
        )

        corrected_latency = obs.duration_sec
        R_latency: float | None = None
        segment_correction = self._integrate_segment_runtime(obs)
        if segment_correction is not None:
            corrected, variance, provenance, all_exact = segment_correction
            obs.correction_provenance = provenance
            obs.correction_applied = all_exact and bool(provenance)
            if all_exact:
                corrected_latency = corrected
                R_latency = variance or None
            else:
                R_latency = max(1.0, obs.duration_sec) ** 2
        elif obs.solo_fraction < 0.8:
            co_components = sorted(obs.co_located_components)
            total_slowdown = 0.0
            total_slowdown_var = 0.0
            all_known = bool(co_components)

            for neighbor in co_components:
                if neighbor == obs.component:
                    self_tuple = (
                        self._interference_registry.get_self_slowdown_if_mature(
                            obs.component,
                            n_concurrent=max(
                                2, _as_int(getattr(obs, "max_concurrent", 2) or 2)
                            ),
                            fp=obs.config_fingerprint or "",
                        )
                    )
                    if self_tuple is not None:
                        sd_mean, sd_std = self_tuple
                        total_slowdown += max(0.0, sd_mean)
                        total_slowdown_var += max(0.0, sd_std) ** 2
                    else:
                        all_known = False
                    continue

                sd = self._interference_registry.get_pairwise_slowdown(
                    obs.component,
                    neighbor,
                    fp=obs.config_fingerprint or "",
                )
                if sd is not None:
                    total_slowdown += sd
                    pair_key = self._interference_registry._pair_key(
                        obs.component,
                        neighbor,
                        obs.config_fingerprint or "",
                    )
                    record = self._interference_registry._pairwise.get(pair_key)
                    n_pair = len(record.latency_slowdowns) if record else 1
                    total_slowdown_var += (sd * sd) / max(n_pair, 1)
                else:
                    all_known = False

            if all_known and total_slowdown > 0:
                divisor = 1.0 + total_slowdown
                corrected_latency = obs.duration_sec / divisor
                sigma2_correction = (
                    obs.duration_sec / (divisor * divisor)
                ) ** 2 * total_slowdown_var
                profile = self._resource_profiles.get_or_create(obs.component)
                cfg = profile.get_or_create_config(
                    obs.config_fingerprint or "__default__"
                )
                cross_bl = cfg.get_or_create_gpu(cfg.CROSS_GPU)
                s2_solo = cross_bl.latency_sec.sigma2_solo
                R_latency = s2_solo / (divisor * divisor) + sigma2_correction
                _LOG.debug(
                    "[kalman-correction] %s Case 2: δ=%.3f corrected=%.3fs R=%.6f",
                    obs.component,
                    total_slowdown,
                    corrected_latency,
                    R_latency,
                )
            else:
                profile = self._resource_profiles.get_or_create(obs.component)
                cfg = profile.get_or_create_config(
                    obs.config_fingerprint or "__default__"
                )
                cross_bl = cfg.get_or_create_gpu(cfg.CROSS_GPU)
                s2_solo = cross_bl.latency_sec.sigma2_solo
                max_sd = self._interference_registry.max_observed_slowdown()
                mu_est = (
                    cross_bl.latency_sec.mean
                    if cross_bl.latency_sec.n > 0
                    else obs.duration_sec
                )
                sigma2_bias = ((max_sd if max_sd > 0 else 0.5) * mu_est) ** 2
                R_latency = s2_solo + sigma2_bias
                corrected_latency = obs.duration_sec
                _LOG.debug(
                    "[kalman-correction] %s Case 3: unknown pair, R=%.6f",
                    obs.component,
                    R_latency,
                )

        interference_latency_drift = (
            self._interference_registry.record_observation(obs)
            if record_interference
            else None
        )
        if interference_latency_drift is not None:
            self._emit_drift_event(
                component=obs.component,
                config_fingerprint="",
                gpu_id=None,
                metric="interference_latency",
                drift_info=interference_latency_drift,
                input_size=input_size,
                campaign_id=obs.campaign_id,
            )

        batch_point = _batch_observation_point(batch_context, input_size)
        execution_k = batch_point[1] if batch_point is not None else 0
        selected_sizes = _batch_selected_sizes(batch_context)
        latency_drift = None
        if selected_sizes is None or selected_sizes[0] == selected_sizes[1]:
            latency_drift = self._resource_profiles.record_latency(
                obs.component,
                obs.config_fingerprint,
                corrected_latency,
                input_size=input_size,
                campaign_id=obs.campaign_id,
                gpu_id=obs_gpu_id,
                R=R_latency,
            )
        if batch_point is not None:
            self._resource_profiles.record_batch_metric(
                obs.component,
                obs.config_fingerprint,
                "latency",
                corrected_latency,
                input_size=batch_point[0],
                execution_batch_size=execution_k,
                campaign_id=obs.campaign_id,
                gpu_id=obs_gpu_id,
                R=R_latency,
            )
        if latency_drift is not None:
            self._emit_drift_event(
                component=obs.component,
                config_fingerprint=obs.config_fingerprint,
                gpu_id=obs_gpu_id,
                metric="latency",
                drift_info=latency_drift,
                input_size=input_size,
                campaign_id=obs.campaign_id,
                primary_tail_reference_qualified=bool(
                    obs.was_solo_throughout or obs.correction_applied
                ),
            )

        _LOG.debug(
            "[latency] %s task=%s duration=%.2fs solo=%s co_located=%s max_concurrent=%d campaign=%s",
            obs.component,
            obs.task_id,
            obs.duration_sec,
            obs.was_solo_throughout,
            sorted(obs.co_located_components),
            obs.max_concurrent,
            obs.campaign_id,
        )

    @_locked
    def clear_latency_tracker(self, addr: str) -> None:
        """Clear latency tracker for a worker (e.g. on restart)."""
        tracker = self._latency_trackers.get(addr)
        if tracker is not None:
            tracker.clear()

    @_locked
    def cleanup_worker(self, addr: str) -> None:
        """Remove all state for a dead worker (guard-kill / crash).

        Cleans up latency tracker, GPU active tasks, and peak activation
        entries referencing this addr to prevent phantom co-location and
        stale data accumulation.
        """
        affected_gpu_ids = [
            gid
            for gid, entries in self._gpu_active_tasks.items()
            if any(entry[0] == addr for entry in entries)
        ]
        self._latency_trackers.pop(addr, None)
        stale_peaks = [
            k for k in self._peak_activation_mb if k == addr or k.startswith(f"{addr}:")
        ]
        for k in stale_peaks:
            del self._peak_activation_mb[k]
        for gid in list(self._gpu_active_tasks):
            self._gpu_active_tasks[gid] = {
                e for e in self._gpu_active_tasks[gid] if e[0] != addr
            }
            if not self._gpu_active_tasks[gid]:
                del self._gpu_active_tasks[gid]
        self._refresh_gpu_segments(affected_gpu_ids, time.monotonic())

    def get_worker_co_located_components(
        self,
        addr: str,
        *,
        gpu_ids: list[str] | None = None,
    ) -> frozenset[str]:
        """Return the set of components currently executing on a worker.

        ``gpu_ids`` scopes the view to the candidate's GPUs: a live task
        on a non-overlapping GPU of a multi-GPU worker is not a
        co-location neighbor.  Empty/None keeps the whole-worker view
        (legacy callers: ops inventory).
        """
        tracker = self._latency_trackers.get(addr)
        if tracker is None:
            return frozenset()
        return tracker.active_components_on_gpus(gpu_ids)

    def get_worker_self_concurrency(
        self,
        addr: str,
        component: str,
        *,
        gpu_ids: list[str] | None = None,
    ) -> int:
        """Projected self-interference degree N for a candidate on *addr*.

        Live same-component slots on the worker (GPU-scoped) plus the
        candidate itself.  Gateway-authoritative: the tracker is fed by
        ``latency_on_task_enter``/``latency_on_task_exit`` at InferBatch
        dispatch/completion, unlike heartbeat-counted supervisor queue
        state which can lag the actual execution start.
        """
        tracker = self._latency_trackers.get(addr)
        if tracker is None:
            return 1
        return tracker.active_component_concurrency(component, gpu_ids) + 1


    @_locked
    def update_workload_profile(
        self,
        component: str,
        *,
        gpu_util_percent: float | None = None,
        execute_us: float | None = None,
        active_memory_mib: float | None = None,
        is_solo: bool = True,
        concurrent_task_count: int = 1,
    ) -> str:
        """Update workload classification for a component from telemetry.

        Both solo and concurrent observations are accepted.  Concurrent GPU
        utilization is normalized by ``concurrent_task_count``.  Solo
        observations are weighted higher (α=0.30 vs α=0.15).
        Returns the updated workload class.
        """
        self._resource_profiles.update_compute_profile(
            component,
            gpu_util_percent=gpu_util_percent,
            execute_us=execute_us,
            active_memory_mib=active_memory_mib,
            is_solo=is_solo,
            concurrent_task_count=concurrent_task_count,
        )
        return self._workload_classifier.update_profile(
            component,
            gpu_util_percent=gpu_util_percent,
            execute_us=execute_us,
            active_memory_mib=active_memory_mib,
            is_solo=is_solo,
            concurrent_task_count=concurrent_task_count,
        )


    @_locked
    @_locked
    def record_observation(self, observation: Any) -> None:
        """Validate the terminal observation channel; detailed learning is latency-owned."""
        try:
            component = str(observation.component or "")
            gpu_id = str(observation.gpu_id or "")
            duration = _to_float(observation.actual_duration_sec)
            if not component or not gpu_id or duration is None or duration < 0.0:
                raise ValueError(
                    "component, gpu_id, and non-negative duration are required"
                )
        except (AttributeError, TypeError, ValueError) as exc:
            self._observation_counters["ingest_failures"] += 1
            _LOG.error("[gp-observation] invalid observation: %s", exc)
            raise ValueError("invalid GP observation") from exc
        self._observation_counters["ingested"] += 1

    def _on_interference_evidence(
        self, epoch: int, components: tuple[str, ...]
    ) -> None:
        for component in components:
            self._emit_drift_event(
                component=component,
                config_fingerprint="",
                gpu_id=None,
                metric="interference_evidence",
                drift_info=(epoch, max(0, epoch - 1), 1.0, 1),
                input_size=0.0,
                campaign_id="",
            )

    @_locked
    def register_drift_callback(
        self,
        callback: Callable[[ProfileDriftEvent], None],
    ) -> None:
        """Register a callback to be invoked whenever a significant baseline drift is detected.

        The callback fires **synchronously** from within the VRAM or latency
        recording path, so it must be lightweight (log, enqueue — never block or
        await).  Multiple callbacks may be registered; each is called in
        registration order.

        Typical caller: ``PlannerService`` registers ``self.on_profile_drift``
        so the Planner can invalidate stale scheduling scenarios when baselines
        shift. Repeated registration of the same callback is idempotent.
        """
        if callback not in self._drift_callbacks:
            self._drift_callbacks.append(callback)

    def _emit_drift_event(
        self,
        *,
        component: str,
        config_fingerprint: str,
        gpu_id: str | None,
        metric: str,
        drift_info: tuple,
        input_size: float,
        campaign_id: str,
        primary_tail_reference_qualified: bool = False,
    ) -> None:
        """Buffer a ProfileDriftEvent into the coalescing window.

        ``drift_info`` is the ``(observed, predicted, drift_ratio, n_baseline)``
        tuple returned by ``ConfigProfile._check_vram_drift`` /
        ``_check_latency_drift``.

        Plan fix — events are not fired synchronously here;
        they are stored in ``_drift_pending`` keyed by ``(component,
        metric)`` (latest event wins) and a 50 ms ``_drift_flush_handle``
        is armed on the asyncio loop.  ``GlobalPlanner.solve()`` calls
        ``flush_pending_drifts()`` on entry so re-plan always sees the
        latest cascade applied.
        """
        if self._bootstrapping or not self._drift_callbacks:
            return

        observed, predicted, drift_ratio, n_baseline = drift_info
        event = ProfileDriftEvent.make(
            component=component,
            config_fingerprint=config_fingerprint,
            gpu_id=gpu_id,
            metric=metric,
            observed=observed,
            predicted=predicted,
            drift_ratio=drift_ratio,
            n_baseline=n_baseline,
            input_size=input_size,
            campaign_id=campaign_id,
            primary_tail_reference_qualified=primary_tail_reference_qualified,
        )
        should_schedule = False
        key = f"{component}:{metric}"
        with self._lock:
            now = time.monotonic()
            if key not in self._drift_pending:
                last_callback = self._last_drift_callback_at.get(key, 0.0)
                if (
                    metric != "interference_evidence"
                    and now - last_callback < self._drift_cooldown_sec
                ):
                    last_log = self._last_drift_at.get(key)
                    if last_log is None:
                        self._last_drift_at[key] = (last_callback or now, 1)
                    else:
                        last_time, suppressed_count = last_log
                        self._last_drift_at[key] = (
                            last_time,
                            suppressed_count + 1,
                        )
                    return
                self._last_drift_callback_at[key] = now
            self._drift_pending[key] = event
            should_schedule = self._drift_flush_handle is None
        if should_schedule:
            self._schedule_drift_flush()


    def _schedule_drift_flush(self) -> None:
        """Arm the ``_drift_flush_handle`` if it is not already armed.

         Step 7 (A) — route through ``self._main_loop`` whenever
        possible.  The wake-paths that reach this function are:

          1. Main asyncio thread (HTTP handler, supervisor stats
             callback after loop start, ``_drift_apply_*`` follow-ups).
             ``asyncio.get_running_loop()`` returns the main loop.
          2. ``asyncio.to_thread`` worker (Step 3 ``latency_on_task_*``
             offloads).  No loop is running here — pre-Step 7 fell
             through to a synchronous ``_flush_pending_drifts()``,
             which executed the entire reprojection cascade *inside
             the worker thread*, holding the GIL for ~hundreds of ms
             ~ tens of seconds ( 60 s spike root cause).
          3. Supervisor stats thread fired before any asyncio loop
             starts (genuine sync test path).  Behaviour unchanged.

        Cases 1 and 2 now both schedule via ``self._main_loop``: case 1
        captures it on first invocation, case 2 reuses the captured
        reference and uses ``call_soon_threadsafe`` so the timer arms
        on the main loop.  Case 3 still falls back to synchronous
        flush when no loop has ever been observed.
        """
        if self._drift_flush_handle is not None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            self._main_loop = running
            self._drift_flush_handle = running.call_later(
                self._drift_coalesce_sec, self._on_drift_flush_timeout
            )
            return
        main_loop = self._main_loop
        if main_loop is not None and not main_loop.is_closed():
            main_loop.call_soon_threadsafe(self._arm_drift_flush_on_main)
            return
        self._flush_pending_drifts()

    def _arm_drift_flush_on_main(self) -> None:
        """Helper executed on the main asyncio thread (via
        ``call_soon_threadsafe``) — arms the coalesce timer there.

        Idempotent: if another arming raced ahead (e.g. main thread
        observed the same drift event independently) we leave the
        existing handle alone.
        """
        if self._drift_flush_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._flush_pending_drifts()
            return
        self._main_loop = loop
        self._drift_flush_handle = loop.call_later(
            self._drift_coalesce_sec, self._on_drift_flush_timeout
        )

    def _on_drift_flush_timeout(self) -> None:
        """Timer callback — clear the handle and flush everything."""
        self._drift_flush_handle = None
        self._flush_pending_drifts()

    def has_pending_drifts(self) -> bool:
        """Used by ``GlobalPlanner.solve()`` to decide whether a flush
        is required before re-plan."""
        return bool(self._drift_pending)

    def flush_pending_drifts(self) -> None:
        """Public — synchronously fire every buffered drift event.

        ``GlobalPlanner.solve()`` calls this on entry so the placement
        decision sees the same posterior the latest observation would
        have produced (D2 EFT internal-consistency invariant).

         Step 7 C — the public flush is split into a tiny
        lock-protected drain + a lock-free callback invocation.  This
        used to be one ``@_locked`` body; the cascade callback (which
        ends up calling ``predict_latency`` / ``predict_vram`` and
        therefore the GP Cholesky refit, O(N²)~O(N³)) ran inside the
        single SignalService RLock and serialised every concurrent
        ``record_observation`` / ``latency_on_task_*`` from
        c=9 worker threads.   py-spy showed 5+ asyncio worker
        threads simultaneously blocked at ``wrapped (signals/service.py:60)``
        — the RLock acquire — while the main thread held it inside
        the cascade.  After the split the cascade fires without
        holding the RLock; record / latency mutations from worker
        threads acquire and release the lock in microseconds even
        while a cascade is running on the main thread.
        """
        with self._lock:
            if self._drift_flush_handle is not None:
                with contextlib.suppress(Exception):
                    self._drift_flush_handle.cancel()
                self._drift_flush_handle = None
            by_component, pending = self._drain_pending_drifts_locked()
        if by_component:
            self._submit_drift_callbacks(by_component, pending)

    def _flush_pending_drifts(self) -> None:
        """Internal flush (timer-fired path).  Caller is the asyncio
        loop's ``_on_drift_flush_timeout`` which already runs without
        holding the SignalService lock — but ``_drift_pending`` is
        shared with worker-thread emits, so the *drain* must take
        the lock briefly.  The cascade itself runs lock-free.
        """
        with self._lock:
            by_component, pending = self._drain_pending_drifts_locked()
        if by_component:
            self._submit_drift_callbacks(by_component, pending)

    def _submit_drift_callbacks(
        self,
        by_component: dict[str, dict[str, Any]],
        pending: dict[str, Any],
    ) -> None:
        """Run drift cascade callbacks on a daemon single-worker queue."""
        self._ensure_drift_worker()
        self._drift_queue.put((by_component, pending))

    def _ensure_drift_worker(self) -> None:
        if self._drift_worker_started:
            return
        with self._lock:
            if self._drift_worker_started:
                return
            worker = threading.Thread(
                target=self._drift_worker_loop,
                name="proton-drift",
                daemon=True,
            )
            worker.start()
            self._drift_worker_started = True

    def _drift_worker_loop(self) -> None:
        while True:
            by_component, pending = self._drift_queue.get()
            try:
                self._invoke_drift_callbacks_unlocked(by_component, pending)
            except Exception:
                _LOG.exception("profile-drift-callback-failed")
            finally:
                self._drift_queue.task_done()

    def _drain_pending_drifts_locked(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Lock-held critical section — snapshot + clear ``_drift_pending``.

        Caller must hold ``self._lock``.  Returns
        ``(by_component, pending_for_log)``.  When the per-flush
        component cap is exceeded the deferred fraction stays in
        ``_drift_pending`` and a new timer is armed for the next
        coalesce window (Step 3 fix-A bounded cascade work).

        This was previously inlined in ``_flush_pending_drifts``;
        Step 7 C extracts it so cascade callbacks can run without
        the RLock held.
        """
        if not self._drift_pending:
            return {}, {}
        all_pending = self._drift_pending
        all_by_component: dict[str, dict[str, Any]] = {}
        for ev in all_pending.values():
            all_by_component.setdefault(ev.component, {})[ev.metric] = ev

        cap = max(1, _as_int(self._drift_max_components_per_flush))
        if len(all_by_component) > cap:
            kept = list(all_by_component.keys())[:cap]
            kept_set = set(kept)
            by_component = {c: all_by_component[c] for c in kept}
            deferred = {
                key: ev
                for key, ev in all_pending.items()
                if ev.component not in kept_set
            }
            self._drift_pending = deferred
            pending = {
                key: ev for key, ev in all_pending.items() if ev.component in kept_set
            }
            if deferred:
                self._schedule_drift_flush()
        else:
            pending = all_pending
            self._drift_pending = {}
            by_component = all_by_component
        return by_component, pending

    def _invoke_drift_callbacks_unlocked(
        self,
        by_component: dict[str, dict[str, Any]],
        pending: dict[str, Any],
    ) -> None:
        """Lock-free callback invocation — caller MUST NOT hold
        ``self._lock``.

        Runs the cascade callbacks (typically
        ``CampaignScheduler.on_profile_drift_batch`` →
        ``_drift_apply_per_component`` → ``_reproject_pending`` →
        ``predict_latency``/``predict_vram`` → GP Cholesky refit)
        without serialising concurrent ``record_observation`` /
        ``latency_on_task_*`` calls from worker threads.  Drift-log
        dedup (the ``_last_drift_at`` dict) is also touched here;
        it is only used for logging granularity, so a benign race
        on its updates is acceptable (no scheduling impact).
        """
        if not by_component:
            return
        import time as _time

        now = _time.monotonic()
        for cb in self._drift_callbacks:
            owner = getattr(cb, "__self__", None)
            batch_method = (
                getattr(owner, "on_profile_drift_batch", None) if owner else None
            )
            try:
                if batch_method is not None:
                    batch_method(by_component)
                else:
                    for events_by_metric in by_component.values():
                        for ev in events_by_metric.values():
                            cb(ev)
            except Exception:
                _LOG.exception(
                    "[drift] callback %r raised an exception",
                    cb,
                )

        for ev in pending.values():
            dedup_key = f"{ev.component}:{ev.metric}"
            last_info = self._last_drift_at.get(dedup_key)
            if last_info is not None:
                last_time, suppressed_count = last_info
                if now - last_time < self._drift_cooldown_sec:
                    self._last_drift_at[dedup_key] = (
                        last_time,
                        suppressed_count + 1,
                    )
                    continue
                if suppressed_count > 0:
                    _LOG.info(
                        "[drift] %s metric=%s — %d additional drift events "
                        "suppressed in %.1fs window",
                        ev.component,
                        ev.metric,
                        suppressed_count,
                        self._drift_cooldown_sec,
                    )
            self._last_drift_at[dedup_key] = (now, 0)
            _LOG.warning(
                "[drift] %s metric=%s observed=%.3f predicted=%.3f "
                "drift=%.1f%% n=%d config=%s gpu_id=%s",
                ev.component,
                ev.metric,
                ev.observed,
                ev.predicted,
                ev.drift_ratio * 100,
                ev.n_baseline,
                ev.config_fingerprint or "__default__",
                ev.gpu_id,
            )

    @_locked
    def record_solo_vram(
        self,
        component: str,
        vram_mib: float,
        *,
        config_fingerprint: str = "",
        input_size: float = 0.0,
        campaign_id: str = "",
        gpu_id: str | None = None,
        is_solo: bool = True,
        progress_ratio: float | None = None,
        batch_context: Mapping[str, Any] | None = None,
        memory_attribution: str = "request_process_tree",
    ) -> None:
        """Record VRAM observation for interference and resource profiles.

        Only ``request_process_tree`` measurements train scalar resource GPs.
        Peer concurrency remains latency/interference context; it does not
        reduce request-owned memory or create recursive observation noise.
        ``progress_ratio`` remains accepted for request compatibility, while
        lower-bound filtering stays at the Gateway observation boundary.

        ``gpu_id`` should be supplied when the caller knows which physical GPU
        the worker is running on.  Single-GPU tasks record to both the GPU-specific
        baseline and the cross-GPU pool; multi-GPU tasks (gpu_id=None) record to
        the cross-GPU pool only.
        """
        if str(memory_attribution or "").strip() != "request_process_tree":
            _LOG.info(
                "[resource-observation] telemetry-only VRAM component=%s "
                "attribution=%s",
                component,
                memory_attribution,
            )
            return
        interference_vram_drift = self._interference_registry.record_solo_vram(
            component,
            vram_mib,
            fp=config_fingerprint,
        )
        if interference_vram_drift is not None:
            self._emit_drift_event(
                component=component,
                config_fingerprint=config_fingerprint,
                gpu_id=None,
                metric="interference_vram",
                drift_info=interference_vram_drift,
                input_size=input_size,
                campaign_id=campaign_id,
            )

        R_vram: float | None = None
        batch_point = _batch_observation_point(batch_context, input_size)
        execution_k = batch_point[1] if batch_point is not None else 0
        selected_sizes = _batch_selected_sizes(batch_context)
        profile_vram_drift = None
        if selected_sizes is None or selected_sizes[0] == selected_sizes[1]:
            profile_vram_drift = self._resource_profiles.record_vram(
                component,
                config_fingerprint,
                vram_mib,
                input_size=input_size,
                campaign_id=campaign_id,
                R=R_vram,
            )
        if batch_point is not None:
            self._resource_profiles.record_batch_metric(
                component,
                config_fingerprint,
                "vram",
                vram_mib,
                input_size=batch_point[0],
                execution_batch_size=execution_k,
                campaign_id=campaign_id,
                R=R_vram,
            )
        if profile_vram_drift is not None:
            self._emit_drift_event(
                component=component,
                config_fingerprint=config_fingerprint,
                gpu_id=None,
                metric="vram",
                drift_info=profile_vram_drift,
                input_size=input_size,
                campaign_id=campaign_id,
            )

    @_locked
    def record_temporal_vram(
        self,
        component: str,
        low_vram: float,
        peak_start_ratio: float,
        peak_end_ratio: float,
        *,
        config_fingerprint: str = "",
        input_size: float | None = None,
        runtime_sec: float | None = None,
        memory_attribution: str = "request_process_tree",
    ) -> None:
        """Record request-owned low VRAM and input-conditioned peak timing."""
        if str(memory_attribution or "").strip() != "request_process_tree":
            return
        self._resource_profiles.record_temporal_vram(
            component,
            config_fingerprint,
            low_vram,
            peak_start_ratio,
            peak_end_ratio,
            input_size=input_size,
            runtime_sec=runtime_sec,
        )

    @_locked
    def record_solo_ram(
        self,
        component: str,
        ram_mib: float,
        *,
        config_fingerprint: str = "",
        input_size: float = 0.0,
        campaign_id: str = "",
        is_solo: bool = True,
        concurrent_task_count: int = 1,
        batch_context: Mapping[str, Any] | None = None,
        memory_attribution: str = "request_process_tree",
    ) -> None:
        """Record active CPU RAM observation for resource-profile GP.

        RAM mirrors the VRAM profile path but does not feed the GPU
        interference registry; it is a host-wide scheduling resource.
        """
        if str(memory_attribution or "").strip() != "request_process_tree":
            _LOG.info(
                "[resource-observation] telemetry-only RAM component=%s attribution=%s",
                component,
                memory_attribution,
            )
            return
        R_ram: float | None = None
        batch_point = _batch_observation_point(batch_context, input_size)
        execution_k = batch_point[1] if batch_point is not None else 0
        selected_sizes = _batch_selected_sizes(batch_context)
        profile_ram_drift = None
        if selected_sizes is None or selected_sizes[0] == selected_sizes[1]:
            profile_ram_drift = self._resource_profiles.record_ram(
                component,
                config_fingerprint,
                ram_mib,
                input_size=input_size,
                campaign_id=campaign_id,
                R=R_ram,
            )
        if batch_point is not None:
            self._resource_profiles.record_batch_metric(
                component,
                config_fingerprint,
                "ram",
                ram_mib,
                input_size=batch_point[0],
                execution_batch_size=execution_k,
                campaign_id=campaign_id,
                R=R_ram,
            )
        if profile_ram_drift is not None:
            self._emit_drift_event(
                component=component,
                config_fingerprint=config_fingerprint,
                gpu_id=None,
                metric="ram",
                drift_info=profile_ram_drift,
                input_size=input_size,
                campaign_id=campaign_id,
            )

    @_locked
    def evict_stale_interference_data(self) -> int:
        """Remove expired observations from the interference registry."""
        return self._interference_registry.evict_stale()

    def predict_interference(
        self,
        component: str,
        co_located_components: frozenset[str],
    ) -> InterferencePrediction:
        """Predict interference for a component given its co-located neighbors.

        Used by the planner to adjust runtime estimates for co-located dispatch.
        """
        return self._interference_registry.predict(component, co_located_components)

    @property
    def workload_classifier(self) -> WorkloadClassifier:
        return self._workload_classifier

    @property
    def interference_registry(self) -> InterferenceRegistry:
        return self._interference_registry

    @property
    def resource_profiles(self) -> ResourceProfileRegistry:
        return self._resource_profiles


    def snapshot_for_planning(
        self,
        components: list[str],
        gpus: list[str],
        input_sizes: list[int] | None = None,
    ) -> GPPosteriorSnapshot:
        """Freeze GP posterior for one ``solve()`` invocation.

        Plan : keeps per-iteration EFT comparisons
        consistent (Omega-style optimistic concurrency); the Planner calls
        this at the top of ``solve()`` and feeds the returned snapshot
        through strategy calls instead of re-querying the live service.
        """
        snap = GPPosteriorSnapshot(taken_at=time.time())
        sizes = input_sizes or [0]
        for component in components:
            for gpu_id in gpus:
                for isize in sizes:
                    target = self._live_runtime_target(
                        component=component,
                        config_fingerprint="",
                        workload_features={"input_size": isize},
                        worker_context={"gpu_ids": [str(gpu_id)]},
                    )
                    estimate = _mapping(target.get("estimate") if target else None)
                    mu = _to_float(estimate.get("center"))
                    upper = _to_float(estimate.get("upper"))
                    if mu is None:
                        continue
                    sigma = max(0.0, _as_float(upper or mu) - _as_float(mu))
                    snap.predictions[(component, str(gpu_id), _as_int(isize))] = (
                        _as_float(mu),
                        sigma,
                    )
        for a in components:
            for b in components:
                try:
                    sd = self._interference_registry.get_pairwise_slowdown(a, b)
                except Exception:
                    sd = None
                if sd is not None:
                    snap.interference[(a, b)] = _as_float(sd)
        return snap


    GP_MATURITY_MIN_OBSERVATIONS_PER_DIM: int = 10
    GP_MATURITY_CV_THRESHOLD: float = 0.5

    def is_mature(
        self,
        component: str,
        gpu_id: str | None = None,
    ) -> bool:
        """Plan — returns True when the GP has
        enough observations AND the coefficient of variation
        (``σ / μ``) falls below ``GP_MATURITY_CV_THRESHOLD``.

        Callers (e.g., campaign_scheduler's confidence-gated paths)
        use this to decide whether to trust GP-derived values or to
        fall back to conservative defaults.

        False-positive > false-negative: the threshold is conservative
        on purpose (plan  C note).
        """
        rp = getattr(self, "_resource_profiles", None)
        if rp is None:
            return False
        try:
            input_dim = self._gp_input_dimension(component)
        except Exception:
            input_dim = 1
        required = self.GP_MATURITY_MIN_OBSERVATIONS_PER_DIM * max(
            1, _as_int(input_dim)
        )

        obs_count = 0
        try:
            profile = rp._profiles.get(component)
            if profile is not None:
                for cfg in profile._config_baselines.values():
                    bl = cfg.resolve(gpu_id) if gpu_id else cfg.resolve()
                    if bl is None:
                        continue
                    latency_n = getattr(bl.latency_sec, "n", 0) if bl.latency_sec else 0
                    vram_n = getattr(bl.vram_mib, "n", 0) if bl.vram_mib else 0
                    obs_count = max(obs_count, min(latency_n, vram_n))
        except Exception:
            return False
        if obs_count < required:
            return False

        mu = 0.0
        sigma = 0.0
        try:
            profile = rp._profiles.get(component)
            if profile is not None:
                for cfg in profile._config_baselines.values():
                    bl = cfg.resolve(gpu_id) if gpu_id else cfg.resolve()
                    if bl is None:
                        continue
                    lat = bl.latency_sec
                    if lat is None or getattr(lat, "n", 0) <= 0:
                        continue
                    mu = _as_float(getattr(lat, "mean", 0.0) or 0.0)
                    var = _as_float(getattr(lat, "var", 0.0) or 0.0)
                    sigma = var**0.5 if var > 0 else 0.0
                    break
        except Exception:
            return False
        if mu <= 0.0:
            return False
        cv = sigma / mu
        return cv < self.GP_MATURITY_CV_THRESHOLD

    def _gp_input_dimension(self, component: str) -> int:
        """GP input dimension (plan  C — 10 × d samples needed).

        Default 1D (component-only); subclasses / config extensions may
        expose a (component, input_size, batch_size, ...) signature.
        """
        return 1

    def refresh_if_stale(self, *, max_age_sec: float = 10.0) -> bool:
        """Invalidate cached snapshots older than *max_age_sec*.

        Plan item 2 — owned by EventHandler.  This
        implementation is a pass-through hook so callers can consult the
        service without guessing at cache ages; the concrete cache
        (if any) lives inside the resource profile registry.  Returns
        True when any cache was invalidated.
        """
        invalidate = getattr(self._resource_profiles, "invalidate_stale", None)
        if callable(invalidate):
            try:
                return bool(invalidate(max_age_sec=max_age_sec))
            except Exception:
                _LOG.debug(
                    "[signal-service] resource_profiles.invalidate_stale failed",
                    exc_info=True,
                )
        return False


    @staticmethod
    def _live_support_level(n_rows: int) -> str:
        if n_rows >= 6:
            return "exact"
        if n_rows >= 3:
            return "nearby"
        return "coarse" if n_rows >= 1 else "none"

    def _live_runtime_target(
        self,
        *,
        component: str,
        config_fingerprint: str,
        workload_features: Mapping[str, Any] | None,
        worker_context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        input_size = ResourceProfileRegistry.extract_input_size(
            dict(workload_features or {}),
            component=component,
        )
        gpu_ids = list(worker_context.get("gpu_ids") or [])
        gpu_id = str(gpu_ids[0]).strip() if gpu_ids else ""
        profile = self._resource_profiles._profiles.get(component)
        if profile is None:
            return None
        config = profile._config_baselines.get(config_fingerprint or "__default__")
        if config is None:
            return None
        baseline = config.resolve(gpu_id or None)
        if baseline is None or baseline.latency_sec.n <= 0:
            return None
        mu, var = baseline._cached_latency_predict(input_size)
        if mu is None:
            return None
        sigma = (_as_float(var) ** 0.5) if var and var > 0 else 0.0
        upper = max(_as_float(mu), _as_float(mu) + 1.28155 * sigma)
        n_rows = _as_int(baseline.latency_sec.n)
        support = self._live_support_level(n_rows)
        return {
            "estimate": {
                "center": _as_float(mu),
                "upper": upper,
                "unit": "sec",
                "scope": "within_run_resource_profile",
            },
            "support": {
                "level": support,
                "fallback_level": support,
                "effective_support": n_rows,
                "n_history_rows": n_rows,
            },
            "source": "within_run_resource_profile",
        }

    def _merge_live_runtime_prediction(
        self,
        prediction: Mapping[str, Any],
        *,
        component: str,
        config_fingerprint: str,
        workload_features: Mapping[str, Any] | None,
        worker_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        compatibility = _mapping(prediction.get("compatibility"))
        if _to_float(compatibility.get("predicted_runtime_sec")) is not None:
            return prediction
        target = self._live_runtime_target(
            component=component,
            config_fingerprint=config_fingerprint,
            workload_features=workload_features,
            worker_context=worker_context,
        )
        if target is None:
            return prediction

        estimate = _mapping(target.get("estimate"))
        support = _mapping(target.get("support"))
        patched = dict(prediction)
        execution_envelope = _mapping(patched.get("execution_envelope"))
        execution_envelope["hot_execution_time_sec"] = target
        patched["execution_envelope"] = execution_envelope

        reasons = list(compatibility.get("reasons") or [])
        reasons.append("within_run_resource_profile")
        compatibility.update(
            {
                "predicted_runtime_sec": _to_float(estimate.get("center")),
                "predicted_p90_sec": _to_float(estimate.get("upper")),
                "support_level": str(support.get("level") or "coarse"),
                "fallback_level": str(support.get("fallback_level") or "coarse"),
                "reasons": list(dict.fromkeys(str(item) for item in reasons if item)),
            }
        )
        patched["compatibility"] = compatibility
        return patched

    def query(
        self,
        *,
        component: str,
        config_fingerprint: str,
        input_fingerprint: str,
        workload_features: Mapping[str, Any] | None = None,
        execution_overrides: Mapping[str, Any] | None = None,
        worker_context: Mapping[str, Any] | None = None,
        campaign_id: str = "",
        planner_intent: PlannerIntent | None = None,
        execution_profile: Mapping[str, Any] | None = None,
    ) -> SignalResult:
        normalized_worker_context = (
            normalize_worker_context(worker_context)
            if isinstance(worker_context, Mapping) and worker_context
            else {}
        )
        prediction = predict_cost(
            run_index=self._run_index,
            component=component,
            level=_CANONICAL_RUNTIME_LEVEL,
            config_fingerprint=config_fingerprint,
            input_fingerprint=input_fingerprint,
            workload_features=workload_features,
            execution_overrides=execution_overrides,
            worker_context=normalized_worker_context,
            campaign_id=campaign_id,
            history_limit=self._history_limit,
        )
        if isinstance(execution_profile, Mapping):
            profile = dict(execution_profile)
            prediction = dict(prediction)
            compatibility = _mapping(prediction.get("compatibility"))
            compatibility.update(
                {
                    "predicted_runtime_sec": _to_float(profile.get("latency_mean")),
                    "predicted_p90_sec": _to_float(profile.get("latency_upper")),
                    "predicted_peak_mem_p90_mib": _to_float(profile.get("vram_upper")),
                    "runtime_guard_margin_sec": max(
                        0.0,
                        _as_float(profile.get("latency_upper"))
                        - _as_float(profile.get("latency_mean")),
                    ),
                    "support_level": "planner_frozen",
                    "memory_support_level": "planner_frozen",
                    "memory_basis": "increment_over_resident",
                    "active_memory_scope": "increment_over_resident",
                }
            )
            compatibility.pop("total_upper_bound_mib", None)
            prediction["compatibility"] = compatibility
            execution_envelope = _mapping(prediction.get("execution_envelope"))
            active_memory = _mapping(execution_envelope.get("active_memory_mib"))
            active_estimate = _mapping(active_memory.get("estimate"))
            active_estimate["center"] = _to_float(profile.get("vram_mean"))
            active_memory["estimate"] = active_estimate
            execution_envelope["active_memory_mib"] = active_memory
            prediction["execution_envelope"] = execution_envelope
            query_context = _mapping(prediction.get("query_context"))
            query_context["planner_execution_profile"] = profile
            prediction["query_context"] = query_context
        else:
            prediction = self._merge_live_runtime_prediction(
                prediction,
                component=component,
                config_fingerprint=config_fingerprint,
                workload_features=workload_features,
                worker_context=normalized_worker_context,
            )

        co_located: frozenset[str] = frozenset()
        self_n: int | None = None
        worker_addr = str(normalized_worker_context.get("worker_addr") or "").strip()
        if worker_addr:
            candidate_gpu_ids = [
                str(item)
                for item in (normalized_worker_context.get("gpu_ids") or [])
                if str(item)
            ]
            co_located = self.get_worker_co_located_components(
                worker_addr, gpu_ids=candidate_gpu_ids
            )
            if component in co_located:
                self_n = self.get_worker_self_concurrency(
                    worker_addr, component, gpu_ids=candidate_gpu_ids
                )

        interference = self._interference_registry.predict(
            component, co_located, self_n=self_n
        )

        return self._map_prediction(
            prediction=prediction,
            worker_context=normalized_worker_context,
            planner_intent=planner_intent or PlannerIntent(),
            interference=interference,
        )

    @staticmethod
    def _map_prediction(
        *,
        prediction: Mapping[str, Any],
        worker_context: Mapping[str, Any],
        planner_intent: PlannerIntent,
        interference: InterferencePrediction | None = None,
    ) -> SignalResult:
        compatibility = _mapping(prediction.get("compatibility"))
        query_context = _mapping(prediction.get("query_context"))
        execution_envelope = _mapping(prediction.get("execution_envelope"))
        replica_baseline = _mapping(prediction.get("replica_baseline"))
        corrections = _mapping(prediction.get("corrections"))
        guards = _mapping(prediction.get("guards"))
        execution_targets = _mapping(execution_envelope.get("active_memory_mib"))
        memory_basis, peak_fidelity = _canonical_memory_semantics(
            memory_basis=compatibility.get("memory_basis")
            or _mapping(execution_envelope.get("query_context")).get("memory_basis")
            or _mapping(execution_targets.get("estimate")).get("scope"),
            peak_fidelity=compatibility.get("peak_fidelity"),
            legacy_scope=compatibility.get("active_memory_scope"),
        )
        active_upper_mib = _to_float(compatibility.get("predicted_peak_mem_p90_mib"))
        resident_upper_mib = _to_float(
            compatibility.get("predicted_resident_memory_p95_mib")
        )
        total_upper_bound_mib = _to_float(compatibility.get("total_upper_bound_mib"))
        if total_upper_bound_mib is None:
            if memory_basis == "increment_over_resident":
                if active_upper_mib is not None and resident_upper_mib is not None:
                    total_upper_bound_mib = _as_float(
                        active_upper_mib + resident_upper_mib
                    )
            else:
                total_upper_bound_mib = active_upper_mib

        solo_estimate_sec = _to_float(compatibility.get("predicted_runtime_sec"))
        solo_upper_sec = _to_float(compatibility.get("predicted_p90_sec"))
        adjusted_estimate = solo_estimate_sec
        adjusted_upper = solo_upper_sec
        adjusted_active_upper = active_upper_mib

        if interference is not None:
            if interference.predicted_slowdown > 0:
                factor = 1.0 + interference.predicted_slowdown
                adjusted_estimate = (
                    solo_estimate_sec * factor
                    if solo_estimate_sec is not None
                    else None
                )
                adjusted_upper = (
                    solo_upper_sec * factor if solo_upper_sec is not None else None
                )
            if (
                interference.predicted_vram_overhead > 0
                and active_upper_mib is not None
            ):
                adjusted_active_upper = active_upper_mib * (
                    1.0 + interference.predicted_vram_overhead
                )
                if total_upper_bound_mib is not None:
                    if memory_basis == "increment_over_resident":
                        if resident_upper_mib is not None:
                            total_upper_bound_mib = _as_float(
                                adjusted_active_upper + resident_upper_mib
                            )
                    else:
                        total_upper_bound_mib = adjusted_active_upper

        interference_artifacts: dict[str, Any] = {}
        if interference is not None:
            interference_artifacts = {
                "predicted_slowdown": interference.predicted_slowdown,
                "predicted_vram_overhead": interference.predicted_vram_overhead,
                "workload_class": interference.workload_class,
                "neighbor_components": sorted(interference.neighbor_components),
                "basis": interference.basis,
                "confidence": interference.confidence,
                "sample_count": interference.sample_count,
                "estimate_source": interference.estimate_source,
                "solo_estimate_sec": solo_estimate_sec,
                "solo_upper_sec": solo_upper_sec,
                "solo_active_upper_mib": active_upper_mib,
            }

        merged_guards = dict(guards)
        if interference_artifacts:
            merged_guards["interference"] = interference_artifacts

        bundle = SignalBundle(
            runtime=RuntimeSignal(
                estimate_sec=adjusted_estimate,
                upper_sec=adjusted_upper,
                guard_margin_sec=_to_float(
                    compatibility.get("runtime_guard_margin_sec")
                ),
            ),
            memory=MemorySignal(
                active_estimate_mib=_to_float(
                    _mapping(execution_targets.get("estimate")).get("center")
                ),
                active_upper_mib=adjusted_active_upper,
                resident_estimate_mib=_to_float(
                    compatibility.get("predicted_resident_memory_mib")
                ),
                resident_upper_mib=resident_upper_mib,
                total_upper_bound_mib=total_upper_bound_mib,
                memory_basis=memory_basis,
                peak_fidelity=peak_fidelity,
                guard_margin_mib=_to_float(
                    compatibility.get("memory_guard_margin_mib")
                ),
            ),
            provenance=SignalProvenance(
                runtime_support_level=str(compatibility.get("support_level") or "none"),
                runtime_fallback_level=str(
                    compatibility.get("fallback_level") or "none"
                ),
                memory_support_level=str(
                    compatibility.get("memory_support_level") or "none"
                ),
                memory_fallback_level=str(
                    compatibility.get("memory_fallback_level") or "none"
                ),
                resident_support_level=str(
                    compatibility.get("resident_support_level") or "none"
                ),
                resident_fallback_level=str(
                    compatibility.get("resident_fallback_level") or "none"
                ),
                reasons=tuple(
                    str(item)
                    for item in list(compatibility.get("reasons") or [])
                    if str(item)
                ),
            ),
            artifacts=SignalArtifacts(
                query_context=query_context,
                execution_envelope=execution_envelope,
                replica_baseline=replica_baseline,
                corrections=corrections,
                guards=merged_guards,
            ),
            planner_intent=planner_intent,
        )
        return SignalResult(
            bundle=bundle,
            worker_context_applied=bool(worker_context),
            worker_context=dict(worker_context),
        )


    @staticmethod
    def _record_completion_ts(rec: Any) -> float | None:
        for attr in ("finished_at", "updated_at", "created_at"):
            ts = _to_float(getattr(rec, attr, None))
            if ts is not None:
                return ts
        return None

    @classmethod
    def _record_runtime_interval(cls, rec: Any) -> tuple[float, float] | None:
        runtime = _to_float(getattr(rec, "runtime_sec", None))
        if runtime is None or runtime <= 0.0:
            return None
        end_ts = cls._record_completion_ts(rec)
        if end_ts is None:
            return None
        return (end_ts - runtime, end_ts)

    @staticmethod
    def _record_gpu_ids(rec: Any) -> list[str]:
        return [
            str(gid)
            for gid in list(getattr(rec, "dispatch_gpu_ids", []) or [])
            if str(gid)
        ]

    @staticmethod
    def _merged_interval_length(segments: list[tuple[float, float]]) -> float:
        if not segments:
            return 0.0
        merged = []
        for start, end in sorted(segments):
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return sum(end - start for start, end in merged)

    @staticmethod
    def _max_overlap_count(segments: list[tuple[float, float]]) -> int:
        events: list[tuple[float, int]] = []
        for start, end in segments:
            if end <= start:
                continue
            events.append((start, 1))
            events.append((end, -1))
        active = 0
        max_active = 0
        for _ts, delta in sorted(events, key=lambda item: (item[0], item[1])):
            active += delta
            max_active = max(max_active, active)
        return max_active

    def _build_historical_latency_observations(
        self,
        records: list,
    ) -> dict[str, LatencyObservation]:
        """Reconstruct live LatencyObservation context from terminal rows.

        Historical SQL rows do not store the full ``LatencyObservation`` object
        produced by the live tracker.  For bootstrap we infer each task's
        execution interval from ``finished_at - runtime_sec`` and rebuild GPU
        co-location context from overlapping intervals on the same GPU.
        """
        indexed: list[dict[str, Any]] = []
        for rec in records:
            state = str(getattr(rec, "state", "") or "").strip().upper()
            comp = str(getattr(rec, "component", "") or "").strip().lower()
            run_key = str(getattr(rec, "run_key", "") or "")
            interval = self._record_runtime_interval(rec)
            if state != "SUCCEEDED" or not comp or not run_key or interval is None:
                continue
            start_ts, end_ts = interval
            indexed.append(
                {
                    "rec": rec,
                    "run_key": run_key,
                    "component": comp,
                    "start": start_ts,
                    "end": end_ts,
                    "gpu_ids": self._record_gpu_ids(rec),
                    "gpu_set": set(self._record_gpu_ids(rec)),
                }
            )

        observations: dict[str, LatencyObservation] = {}
        for item in indexed:
            rec = item["rec"]
            start_ts = _as_float(item["start"])
            end_ts = _as_float(item["end"])
            duration = max(0.0, end_ts - start_ts)
            if duration <= 0.0:
                continue
            gpu_set = set(item["gpu_set"])
            co_components: set[str] = set()
            overlap_segments: list[tuple[float, float]] = []
            if not gpu_set:
                overlap_segments = []
            else:
                for other in indexed:
                    if other is item:
                        continue
                    if not other["gpu_set"] or not (gpu_set & other["gpu_set"]):
                        continue
                    overlap_start = max(start_ts, _as_float(other["start"]))
                    overlap_end = min(end_ts, _as_float(other["end"]))
                    if overlap_end <= overlap_start:
                        continue
                    co_components.add(str(other["component"]))
                    overlap_segments.append((overlap_start, overlap_end))

            overlap_sec = self._merged_interval_length(overlap_segments)
            solo_fraction = 1.0 - (overlap_sec / duration)
            solo_fraction = round(min(1.0, max(0.0, solo_fraction)), 4)
            max_concurrent = 1 + self._max_overlap_count(overlap_segments)
            axes = dict(getattr(rec, "axes", {}) or {})
            cfg_fp = str(getattr(rec, "config_fingerprint", "") or "").strip()
            observations[str(item["run_key"])] = LatencyObservation(
                task_id=str(item["run_key"]),
                component=str(item["component"]),
                registry_key=(
                    str(item["component"]),
                    cfg_fp,
                    str(getattr(rec, "input_fingerprint", "") or ""),
                ),
                workload_features=axes,
                config_fingerprint=cfg_fp,
                duration_sec=duration,
                was_solo_throughout=not co_components and solo_fraction >= 0.9999,
                was_ever_solo=solo_fraction > 0.0,
                co_located_components=frozenset(co_components),
                max_concurrent=max(1, max_concurrent),
                campaign_id=str(getattr(rec, "campaign_id", "") or ""),
                solo_fraction=solo_fraction,
                gpu_ids=list(item["gpu_ids"]),
            )
        return observations

    @staticmethod
    def _record_task_id_candidates(rec: Any) -> set[str]:
        candidates: set[str] = set()
        for attr in ("run_key", "run_id", "source_event_id"):
            raw = str(getattr(rec, attr, "") or "").strip()
            if not raw:
                continue
            candidates.add(raw)
            if raw.startswith("task:"):
                candidates.add(raw.split(":", 1)[1])
        return candidates

    def _build_historical_latency_observations_from_timeline(
        self,
        records: list,
        timeline_entries: list,
    ) -> dict[str, LatencyObservation]:
        """Reconstruct latency observations from exported scheduler timeline.

        ``signals_post.json`` carries the scheduler's actual GPU entries with
        wall-clock start/end times.  When available, those intervals are closer
        to the live ``latency_on_task_enter``/``latency_on_task_exit`` window
        than deriving start time from SQL ``finished_at - runtime_sec``.
        """
        record_by_task_id: dict[str, Any] = {}
        for rec in records:
            state = str(getattr(rec, "state", "") or "").strip().upper()
            comp = str(getattr(rec, "component", "") or "").strip().lower()
            run_key = str(getattr(rec, "run_key", "") or "").strip()
            if state != "SUCCEEDED" or not comp or not run_key:
                continue
            for candidate in self._record_task_id_candidates(rec):
                record_by_task_id.setdefault(candidate, rec)

        indexed: list[dict[str, Any]] = []
        for entry in timeline_entries:
            if not isinstance(entry, Mapping):
                continue
            if bool(entry.get("is_predicted")) or bool(entry.get("is_init")):
                continue
            if not bool(entry.get("completed", True)) or bool(entry.get("is_killed")):
                continue
            task_id = str(entry.get("task_id") or "").strip()
            rec = record_by_task_id.get(task_id)
            if rec is None:
                rec = record_by_task_id.get(f"task:{task_id}")
            if rec is None:
                continue
            comp = str(getattr(rec, "component", "") or "").strip().lower()
            start_ts = _to_float(entry.get("start_time"))
            end_ts = _to_float(entry.get("completed_at"))
            if end_ts is None:
                end_ts = _to_float(entry.get("end_time"))
            if end_ts is None:
                end_ts = _to_float(entry.get("predicted_end_time"))
            if start_ts is None or end_ts is None or end_ts <= start_ts:
                elapsed = _to_float(entry.get("elapsed_sec"))
                if start_ts is None or elapsed is None or elapsed <= 0.0:
                    continue
                end_ts = start_ts + elapsed
            gpu_id = str(entry.get("gpu_id") or "").strip()
            gpu_ids = [gpu_id] if gpu_id else self._record_gpu_ids(rec)
            worker_addr = str(entry.get("worker_name") or "").strip()
            if not worker_addr:
                worker_addr = f"{comp}-gpu{gpu_ids[0]}" if gpu_ids else comp
            indexed.append(
                {
                    "rec": rec,
                    "entry": entry,
                    "task_id": task_id,
                    "worker_addr": worker_addr,
                    "component": comp,
                    "start": _as_float(start_ts),
                    "end": _as_float(end_ts),
                    "gpu_ids": gpu_ids,
                    "gpu_set": set(gpu_ids),
                }
            )

        observations: dict[str, LatencyObservation] = {}
        trackers: dict[str, dict[str, dict[str, Any]]] = {}
        gpu_active: dict[str, set[tuple[str, str, str]]] = {}

        def _snapshot_solo_time(addr: str, now: float) -> None:
            for slot in trackers.get(addr, {}).values():
                if bool(slot["currently_solo"]):
                    slot["solo_time_acc"] += now - _as_float(slot["last_transition"])
                slot["last_transition"] = now

        events: list[tuple[float, int, str, dict[str, Any]]] = []
        for item in indexed:
            events.append((_as_float(item["start"]), 1, str(item["task_id"]), item))
            events.append((_as_float(item["end"]), 0, str(item["task_id"]), item))

        for ts, kind, _task_id, item in sorted(events):
            addr = str(item["worker_addr"])
            task_id = str(item["task_id"])
            comp = str(item["component"])
            gpu_ids = list(item["gpu_ids"])
            slots = trackers.setdefault(addr, {})

            if kind == 1:
                _snapshot_solo_time(addr, ts)
                gpu_co_located: set[str] = set()
                for gid in gpu_ids:
                    for _addr, _tid, active_comp in gpu_active.get(str(gid), set()):
                        gpu_co_located.add(active_comp)
                worker_co_located = {str(slot["component"]) for slot in slots.values()}
                co_located = worker_co_located | gpu_co_located
                n_others = len(slots) + len(gpu_co_located - worker_co_located)
                is_solo = len(co_located) == 0
                for gid in gpu_ids:
                    gpu_active.setdefault(str(gid), set()).add((addr, task_id, comp))
                slots[task_id] = {
                    "item": item,
                    "component": comp,
                    "start": ts,
                    "co_located_components": frozenset(co_located),
                    "max_concurrent": n_others + 1,
                    "was_ever_solo": is_solo,
                    "solo_time_acc": 0.0,
                    "last_transition": ts,
                    "currently_solo": is_solo,
                }
                new_total = len(slots)
                for slot in slots.values():
                    if _as_int(slot["max_concurrent"]) < new_total:
                        slot["max_concurrent"] = new_total
                    if new_total > 1:
                        slot["currently_solo"] = False
                continue

            for gid in list(gpu_active):
                gpu_active[gid] = {
                    entry
                    for entry in gpu_active[gid]
                    if not (entry[0] == addr and entry[1] == task_id)
                }
                if not gpu_active[gid]:
                    del gpu_active[gid]
            slot = slots.pop(task_id, None)
            if slot is None:
                continue
            if bool(slot["currently_solo"]):
                slot["solo_time_acc"] += ts - _as_float(slot["last_transition"])
            if len(slots) == 1:
                remaining = next(iter(slots.values()))
                remaining["was_ever_solo"] = True
                remaining["currently_solo"] = True
                remaining["last_transition"] = ts

            duration = ts - _as_float(slot["start"])
            if duration <= 0.0:
                continue
            rec = item["rec"]
            exit_co_located = {str(s["component"]) for s in slots.values()}
            co_components = set(slot["co_located_components"]) | exit_co_located
            solo_fraction = _as_float(slot["solo_time_acc"]) / duration
            solo_fraction = round(min(1.0, max(0.0, solo_fraction)), 4)
            max_concurrent = max(1, _as_int(slot["max_concurrent"]))
            axes = dict(getattr(rec, "axes", {}) or {})
            cfg_fp = str(getattr(rec, "config_fingerprint", "") or "").strip()
            run_key = str(getattr(rec, "run_key", "") or "")
            observations[run_key] = LatencyObservation(
                task_id=run_key,
                component=comp,
                registry_key=(
                    comp,
                    cfg_fp,
                    str(getattr(rec, "input_fingerprint", "") or ""),
                ),
                workload_features=axes,
                config_fingerprint=cfg_fp,
                duration_sec=duration,
                was_solo_throughout=bool(slot["was_ever_solo"]) and max_concurrent == 1,
                was_ever_solo=bool(slot["was_ever_solo"]),
                co_located_components=frozenset(co_components),
                max_concurrent=max_concurrent,
                campaign_id=str(getattr(rec, "campaign_id", "") or ""),
                solo_fraction=solo_fraction,
                gpu_ids=gpu_ids,
            )
        return observations

    @_locked
    def bootstrap_from_history(self) -> dict[str, int]:
        """Populate in-memory signal state from snapshot or historical replay.

        First attempts to load a persisted interference snapshot (solo
        baselines + pairwise data).  If a snapshot exists, interference
        replay from run history is skipped entirely.  Resource profiles
        (GP) and workload classifier are always replayed from history
        since they require the full observation sequence for GP fitting.

        Returns a summary dict with counts per channel.
        """
        runtime_counts = self._load_runtime_state_snapshot()
        if runtime_counts is not None:
            runtime_counts["runtime_state_loaded"] = True
            _LOG.info(
                "[signal-bootstrap] Loaded exact runtime_state_v1 snapshot: %s",
                runtime_counts,
            )
            return runtime_counts

        snapshot_loaded = self._load_interference_snapshot()

        query_signal_replay = getattr(
            self._run_index, "query_all_signal_replay_rows", None
        )
        raw_records = (
            query_signal_replay()
            if callable(query_signal_replay)
            else self._run_index.query_all_succeeded()
        )
        records = list(raw_records or [])
        if not records:
            if snapshot_loaded:
                _LOG.info(
                    "[signal-bootstrap] Interference loaded from snapshot; no run history to replay."
                )
            else:
                _LOG.info(
                    "[signal-bootstrap] No historical records found — cold start."
                )
            return {"total": 0, "snapshot_loaded": snapshot_loaded}

        query_context_rows = getattr(
            self._run_index,
            "query_all_signal_replay_context_rows",
            None,
        )
        context_records = (
            query_context_rows() if callable(query_context_rows) else records
        )
        if not context_records:
            context_records = records
        context_records_list = list(context_records or [])
        query_timeline_entries = getattr(
            self._run_index,
            "query_all_signal_replay_timeline_entries",
            None,
        )
        timeline_entries = (
            query_timeline_entries() if callable(query_timeline_entries) else None
        )
        timeline_entries_list = list(timeline_entries or [])
        historical_latency = (
            self._build_historical_latency_observations_from_timeline(
                context_records_list,
                timeline_entries_list,
            )
            if timeline_entries_list
            else {}
        )
        if not historical_latency:
            historical_latency = self._build_historical_latency_observations(
                context_records_list
            )

        self._bootstrapping = True

        counts: dict[str, int] = {
            "total": len(records),
            "latency": 0,
            "vram": 0,
            "ram": 0,
            "batch_latency": 0,
            "batch_vram": 0,
            "batch_ram": 0,
            "activation": 0,
            "gpu_util": 0,
            "snapshot_loaded": snapshot_loaded,
        }

        for rec in records:
            comp = str(rec.component or "").strip().lower()
            if not comp:
                continue
            cfg_fp = str(rec.config_fingerprint or "").strip()
            state = str(getattr(rec, "state", "") or "").strip().upper()
            state_succeeded = state == "SUCCEEDED"
            is_solo = not bool(rec.concurrent_execute_overlap)
            axes = dict(rec.axes or {})

            batch_context: dict[str, Any] | None = None
            batch_phase = str(axes.get("batch_phase") or "")
            batch_policy = str(axes.get("batch_policy") or "")
            if batch_phase or batch_policy:
                selected_k = _positive_mapping_int(axes, "dynamic_batch_selected_k")
                final_k = _positive_mapping_int(axes, "dynamic_batch_final_admission_k")
                argument_k = _positive_mapping_int(
                    axes, "dynamic_batch_argument_applied_k"
                )
                consumed_k = _positive_mapping_int(axes, "dynamic_batch_consumed_k")
                logical_n = _positive_mapping_int(axes, "dynamic_batch_logical_n")
                eligible = (
                    state_succeeded
                    and selected_k is not None
                    and final_k == selected_k
                    and argument_k == selected_k
                    and logical_n is not None
                    and selected_k <= logical_n
                    and (comp != "mmseqs2" or consumed_k == selected_k)
                )
                batch_context = {
                    "companion_eligible": eligible,
                    "execution_batch_size": selected_k,
                    "logical_batch_size": logical_n,
                }
            elif comp == "mmseqs2" and state_succeeded:
                legacy_n = _positive_mapping_int(axes, "input_batch_size")
                if legacy_n is not None:
                    batch_context = {
                        "companion_eligible": True,
                        "execution_batch_size": legacy_n,
                        "logical_batch_size": legacy_n,
                    }

            boot_gpu_ids = list(rec.dispatch_gpu_ids or [])

            runtime = rec.runtime_sec
            if state_succeeded and runtime is not None and runtime > 0:
                latency_obs = historical_latency.get(
                    str(getattr(rec, "run_key", "") or "")
                )
                if latency_obs is None:
                    latency_obs = LatencyObservation(
                        task_id=str(
                            getattr(rec, "run_key", "")
                            or getattr(rec, "run_id", "")
                            or ""
                        ),
                        component=comp,
                        registry_key=(
                            comp,
                            cfg_fp,
                            str(getattr(rec, "input_fingerprint", "") or ""),
                        ),
                        workload_features=axes,
                        config_fingerprint=cfg_fp,
                        duration_sec=_as_float(runtime),
                        was_solo_throughout=is_solo,
                        was_ever_solo=is_solo,
                        co_located_components=frozenset(),
                        max_concurrent=1 if is_solo else 2,
                        campaign_id=str(rec.campaign_id or ""),
                        solo_fraction=1.0 if is_solo else 0.0,
                        gpu_ids=list(boot_gpu_ids),
                    )
                self._record_latency_observation(
                    latency_obs,
                    record_interference=not snapshot_loaded,
                    batch_context=batch_context,
                )
                counts["latency"] += 1
                if (
                    _batch_observation_point(
                        batch_context,
                        ResourceProfileRegistry.extract_input_size(
                            axes, component=comp
                        ),
                    )
                    is not None
                ):
                    counts["batch_latency"] += 1

            active_vram = getattr(rec, "active_vram_mib", None)
            vram_qc_keep = bool(getattr(rec, "vram_memory_qc_keep", False))
            vram_attribution = str(getattr(rec, "vram_memory_attribution", "") or "")
            vram_is_request_owned = vram_attribution == "request_process_tree"
            if (
                active_vram is not None
                and active_vram > 0
                and vram_qc_keep
                and vram_is_request_owned
            ):
                input_size = ResourceProfileRegistry.extract_input_size(
                    axes,
                    component=comp,
                )
                vram_is_solo = state_succeeded and is_solo
                if vram_is_solo and not snapshot_loaded:
                    self._interference_registry.record_solo_vram(
                        comp,
                        active_vram,
                        fp=cfg_fp,
                    )

                R_vram_boot: float | None = None

                selected_sizes = _batch_selected_sizes(batch_context)
                if selected_sizes is None or selected_sizes[0] == selected_sizes[1]:
                    self._resource_profiles.record_vram(
                        comp,
                        cfg_fp,
                        active_vram,
                        input_size=input_size,
                        campaign_id=str(rec.campaign_id or ""),
                        R=R_vram_boot,
                    )
                    counts["vram"] += 1
                batch_point = _batch_observation_point(batch_context, input_size)
                if batch_point is not None:
                    self._resource_profiles.record_batch_metric(
                        comp,
                        cfg_fp,
                        "vram",
                        active_vram,
                        input_size=batch_point[0],
                        execution_batch_size=batch_point[1],
                        campaign_id=str(rec.campaign_id or ""),
                        R=R_vram_boot,
                    )
                    counts["batch_vram"] += 1

            active_mem = getattr(rec, "host_active_memory_mib", None)
            host_qc_keep = bool(getattr(rec, "host_memory_qc_keep", False))
            host_attribution = str(getattr(rec, "host_memory_attribution", "") or "")
            host_is_request_owned = host_attribution == "request_process_tree"
            if (
                active_mem is not None
                and active_mem > 0
                and host_qc_keep
                and host_is_request_owned
            ):
                input_size = ResourceProfileRegistry.extract_input_size(
                    axes,
                    component=comp,
                )
                R_ram_boot: float | None = None

                selected_sizes = _batch_selected_sizes(batch_context)
                if selected_sizes is None or selected_sizes[0] == selected_sizes[1]:
                    self._resource_profiles.record_ram(
                        comp,
                        cfg_fp,
                        active_mem,
                        input_size=input_size,
                        campaign_id=str(rec.campaign_id or ""),
                        R=R_ram_boot,
                    )
                    counts["ram"] += 1
                batch_point = _batch_observation_point(batch_context, input_size)
                if batch_point is not None:
                    self._resource_profiles.record_batch_metric(
                        comp,
                        cfg_fp,
                        "ram",
                        active_mem,
                        input_size=batch_point[0],
                        execution_batch_size=batch_point[1],
                        campaign_id=str(rec.campaign_id or ""),
                        R=R_ram_boot,
                    )
                    counts["batch_ram"] += 1

            gpu_util = rec.mean_gpu_util_percent
            if state_succeeded and gpu_util is not None and gpu_util > 0:
                latency_obs = historical_latency.get(
                    str(getattr(rec, "run_key", "") or "")
                )
                classifier_is_solo = (
                    latency_obs.solo_fraction >= 0.8
                    if latency_obs is not None
                    else is_solo
                )
                self._workload_classifier.update_profile(
                    comp,
                    gpu_util_percent=gpu_util,
                    is_solo=classifier_is_solo,
                )
                counts["gpu_util"] += 1

        self._bootstrapping = False

        snapshot_msg = " (interference from snapshot)" if snapshot_loaded else ""
        _LOG.info(
            "[signal-bootstrap] Loaded %d records: latency=%d, vram=%d, "
            "ram=%d, activation=%d, gpu_util=%d%s",
            counts["total"],
            counts["latency"],
            counts["vram"],
            counts["ram"],
            counts["activation"],
            counts["gpu_util"],
            snapshot_msg,
        )
        return counts


    _SNAPSHOT_KEY = "interference_v1"
    _RUNTIME_SNAPSHOT_KEY = "runtime_state_v1"

    def export_runtime_state(self) -> dict[str, Any]:
        latency_active = {
            str(addr): {
                "active_count": _as_int(getattr(tracker, "active_count", 0) or 0),
                "task_ids": sorted(str(tid) for tid in getattr(tracker, "_slots", {})),
            }
            for addr, tracker in sorted(self._latency_trackers.items())
            if _as_int(getattr(tracker, "active_count", 0) or 0) > 0
        }
        return {
            "schema": self._RUNTIME_SNAPSHOT_KEY,
            "exported_at_wall": time.time(),
            "resource_measurement_semantics": _RESOURCE_MEASUREMENT_SEMANTICS,
            "resource_profiles": self._resource_profiles.export_state(),
            "interference_registry": self._interference_registry.export_state(),
            "workload_classifier": self._workload_classifier.export_state(),
            "activation_peaks": {
                str(key): _as_float(value)
                for key, value in sorted(self._peak_activation_mb.items())
            },
            "latency_active": latency_active,
        }

    def import_runtime_state(
        self,
        data: Mapping[str, Any],
        *,
        completed_run: bool = True,
    ) -> dict[str, int]:
        if not isinstance(data, Mapping):
            raise ValueError("runtime_state_v1 payload must be a mapping")
        if isinstance(data.get("signal_service"), Mapping):
            data = data["signal_service"]
        activation_peaks = data.get("activation_peaks") or {}
        latency_active = data.get("latency_active") or {}
        if any(tracker.active_count for tracker in self._latency_trackers.values()) or (
            isinstance(latency_active, Mapping) and latency_active
        ):
            raise ValueError(
                "active latency trackers: runtime state import is unsafe while latency tasks are active"
            )
        if completed_run and isinstance(activation_peaks, Mapping) and activation_peaks:
            _LOG.warning(
                "[runtime-state] Dropping %d stale activation peak(s) "
                "from completed-run bootstrap snapshot",
                len(activation_peaks),
            )
            activation_peaks = {}
        resource_state = _mapping(data.get("resource_profiles"))
        resource_state_trusted = (
            str(data.get("resource_measurement_semantics") or "").strip()
            == _RESOURCE_MEASUREMENT_SEMANTICS
        )
        if not resource_state_trusted:
            resource_state = copy.deepcopy(resource_state)
            profiles = _mapping(resource_state.get("profiles"))
            for profile_state in profiles.values():
                configs = _mapping(_mapping(profile_state).get("config_baselines"))
                for config_state in configs.values():
                    baselines = _mapping(_mapping(config_state).get("gpu_baselines"))
                    for baseline_state in baselines.values():
                        if not isinstance(baseline_state, dict):
                            continue
                        baseline_state.pop("vram_gp", None)
                        baseline_state.pop("ram_gp", None)
                        batch_gps = baseline_state.get("batch_gps")
                        if isinstance(batch_gps, dict):
                            batch_gps.pop("vram", None)
                            batch_gps.pop("ram", None)
            _LOG.warning(
                "[runtime-state] Ignoring legacy scalar/batch RAM and VRAM "
                "state without %s provenance",
                _RESOURCE_MEASUREMENT_SEMANTICS,
            )
        self._resource_profiles.import_state(resource_state)
        self._workload_classifier.import_state(
            _mapping(data.get("workload_classifier"))
        )
        counts = self._interference_registry.import_state(
            _mapping(data.get("interference_registry"))
        )
        self._interference_registry.set_signal_service_ref(self)
        self._peak_activation_mb = {
            str(key): _as_float(value)
            for key, value in (
                activation_peaks.items()
                if isinstance(activation_peaks, Mapping)
                else []
            )
        }
        self._latency_trackers = {}
        self._gpu_active_tasks = {}
        self._as_dict_cache = None
        return {
            "resource_profiles": len(self._resource_profiles._profiles),
            "resource_state_trusted": 1 if resource_state_trusted else 0,
            "workload_profiles": len(self._workload_classifier._profiles),
            "activation_peaks": len(self._peak_activation_mb),
            **{f"interference_{k}": _as_int(v) for k, v in counts.items()},
        }

    def save_runtime_state_snapshot(
        self, extra: Mapping[str, Any] | None = None
    ) -> None:
        data = self.export_runtime_state()
        if extra:
            data["extra"] = dict(extra)
        try:
            self._run_index.save_signal_snapshot(self._RUNTIME_SNAPSHOT_KEY, data)
            _LOG.info("[runtime-state] Saved runtime_state_v1 snapshot")
        except Exception:
            _LOG.warning(
                "[runtime-state] Failed to save runtime_state_v1", exc_info=True
            )

    def _load_runtime_state_snapshot(self) -> dict[str, int] | None:
        try:
            data = self._run_index.load_signal_snapshot(self._RUNTIME_SNAPSHOT_KEY)
        except Exception:
            _LOG.warning(
                "[runtime-state] Failed to load runtime_state_v1", exc_info=True
            )
            return None
        if not data:
            return None
        try:
            return self.import_runtime_state(data, completed_run=True)
        except Exception as exc:
            _LOG.error(
                "[runtime-state] Failed to import runtime_state_v1", exc_info=True
            )
            raise RuntimeError("runtime_state_v1 import failed") from exc

    def save_interference_snapshot(self) -> None:
        """Persist interference registry state to DB.

        Saves solo baselines (latency + VRAM) and pairwise correction data
        so the next gateway restart can load them directly without replay.
        """
        data = self._interference_registry.export_snapshot()
        try:
            self._run_index.save_signal_snapshot(self._SNAPSHOT_KEY, data)
            total = (
                sum(len(v) for v in data.get("solo_baselines", {}).values())
                + sum(len(v) for v in data.get("solo_vram", {}).values())
                + sum(
                    sum(len(obs) for obs in entry.values())
                    for entry in data.get("pairwise", {}).values()
                )
            )
            _LOG.info(
                "[signal-snapshot] Saved interference snapshot: %d observations", total
            )
        except Exception:
            _LOG.warning(
                "[signal-snapshot] Failed to save interference snapshot", exc_info=True
            )

    def _load_interference_snapshot(self) -> bool:
        """Load interference snapshot from DB if available.

        Returns True if a snapshot was loaded, False otherwise.
        """
        try:
            data = self._run_index.load_signal_snapshot(self._SNAPSHOT_KEY)
        except Exception:
            _LOG.warning(
                "[signal-snapshot] Failed to load interference snapshot", exc_info=True
            )
            return False

        if not data:
            return False

        counts = self._interference_registry.import_snapshot(data)
        _LOG.info(
            "[signal-snapshot] Loaded interference snapshot: "
            "solo_baselines=%d, solo_vram=%d, pairwise=%d, self_slowdown=%d",
            counts["solo_baselines"],
            counts["solo_vram"],
            counts["pairwise"],
            counts.get("self_slowdown", 0),
        )
        return True


    def as_dict(self) -> dict[str, Any]:
        """Serialize full SignalService state for the ops endpoint.

        Returns a snapshot of all three signal channels:
        - activation_peaks: per-worker/task VRAM peak tracking
        - latency_trackers: per-worker active task slots
        - interference: registry state (baselines, pairwise, workload classes)
        - workload_profiles: per-component profiling metrics

        Step 3 fix-B — short TTL cache.  Recomputing the body every
        call costs O(N) iteration over latency-trackers + a full GP
        confidence matrix recompute (predict_grid → numpy.linalg.solve)
        per component when ``ResourceProfileRegistry.as_dict`` walks
        each ConfigBaseline's ``vram_confidence(None)`` /
        ``latency_confidence(None)`` paths.  py-spy  dump showed
        this as 20% of main-thread CPU under k=3 load.

        ops UI polls every 2 s and tolerates ``ttl_seconds=5`` of
        staleness, so a 1 s cache is well within the freshness budget
        and halves the recompute load.  ``record_*`` paths intentionally
        do NOT invalidate this cache — let it expire naturally so a
        single mutation does not fan out into a full GP solve right
        away.
        """
        now = time.monotonic()
        if (
            self._as_dict_cache is not None
            and now - self._as_dict_cache_at < self._as_dict_cache_ttl_sec
        ):
            return self._as_dict_cache

        peaks: dict[str, float] = {}
        for key, val in sorted(self._peak_activation_mb.items()):
            peaks[key] = round(val, 1)

        trackers: dict[str, Any] = {}
        for addr, tracker in sorted(self._latency_trackers.items()):
            slots_info = []
            for tid, slot in tracker._slots.items():
                elapsed = time.monotonic() - slot.start_mono
                slots_info.append(
                    {
                        "task_id": tid,
                        "component": slot.component,
                        "elapsed_sec": round(elapsed, 2),
                        "co_located_at_entry": sorted(slot.co_located_components),
                        "max_concurrent": slot.max_concurrent,
                        "currently_solo": slot._currently_solo,
                        "solo_time_acc_sec": round(slot._solo_time_acc, 2),
                    }
                )
            trackers[addr] = {
                "active_count": tracker.active_count,
                "active_components": sorted(tracker.active_components),
                "slots": slots_info,
            }

        interference = self._interference_registry.as_dict()

        profiles: dict[str, Any] = {}
        for comp, profile in sorted(self._workload_classifier._profiles.items()):
            profiles[comp] = {
                "workload_class": self._workload_classifier.classify(comp),
                "mean_gpu_util_percent": round(profile.mean_gpu_util_percent, 1)
                if profile.mean_gpu_util_percent is not None
                else None,
                "mean_execute_us": round(profile.mean_execute_us, 1)
                if profile.mean_execute_us is not None
                else None,
                "mean_active_memory_mib": round(profile.mean_active_memory_mib, 1)
                if profile.mean_active_memory_mib is not None
                else None,
                "sample_count": profile.sample_count,
            }

        resource_profiles = self._resource_profiles.as_dict()

        result = {
            "activation_peaks": peaks,
            "latency_trackers": trackers,
            "observation_counters": dict(self._observation_counters),
            "interference": interference,
            "workload_profiles": profiles,
            "resource_profiles": resource_profiles,
        }
        self._as_dict_cache = result
        self._as_dict_cache_at = now
        return result


__all__ = ["SignalService"]
