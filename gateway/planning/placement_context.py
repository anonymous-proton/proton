"""Scheduling Scenario: system-wide resource view for informed placement decisions.

The Planner consults a single ``PlacementContext`` before each scheduling
decision.  The scenario aggregates three layers of information:

1. **GPU Resource View** — per-GPU capacity snapshot (VRAM, utilization,
   inflight components, projected clear time).
2. **Component Envelope** — per-component resource summary aggregated from
   Signal (VRAM / latency statistics, workload class, interference slowdowns,
   CV-based confidence).
3. **Reliability** — scenario-level staleness indicators driven by drift events.

The ``PlacementContextManager`` is owned by ``PlannerService`` and refreshed at the
start of each ``generate_plan()`` call.  Drift events from ``SignalService``
mark individual component envelopes as stale; fresh signal observations reset
the flag.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..signals.contracts import ProfileDriftEvent

_LOG = logging.getLogger(__name__)



@dataclass
class GpuResourceView:
    """Live resource snapshot for a single GPU."""

    gpu_id: str
    total_vram_mib: float = 0.0
    reserved_vram_mib: float = 0.0
    available_vram_mib: float = 0.0
    utilization_pct: float = 0.0
    inflight_components: List[str] = field(default_factory=list)
    projected_clear_sec: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "total_vram_mib": round(self.total_vram_mib, 1),
            "reserved_vram_mib": round(self.reserved_vram_mib, 1),
            "available_vram_mib": round(self.available_vram_mib, 1),
            "utilization_pct": round(self.utilization_pct, 1),
            "inflight_components": list(self.inflight_components),
            "projected_clear_sec": round(self.projected_clear_sec, 2),
        }



@dataclass
class ComponentEnvelope:
    """Aggregated resource profile for a known component type.

    Built from ``ResourceProfileRegistry`` and ``InterferenceRegistry`` data.
    ``confidence`` is a computed property derived from the underlying
    ``GaussianEstimate.confidence()`` formula (Kalman covariance-derived):

        confidence = max(0, 1 - P_t / P_1)

    where P_t is the Kalman state covariance and P_1 is the covariance after
    the first observation.  See <docs> for derivation.
    """

    component: str
    vram_p50_mib: float = 0.0
    vram_p95_mib: float = 0.0
    latency_p50_sec: float = 0.0
    latency_p95_sec: float = 0.0
    workload_class: str = "unknown"
    interference_compute_slowdown: float = 0.0
    interference_memory_slowdown: float = 0.0
    vram_confidence: float = 0.0
    latency_confidence: float = 0.0
    vram_relative_error: float = float("inf")
    latency_relative_error: float = float("inf")
    n_observations: int = 0
    last_updated_at: float = 0.0
    drift_stale: bool = False
    drift_events_count: int = 0

    @property
    def confidence(self) -> float:
        """Overall confidence: min of VRAM and latency confidence.

        Returns the lower of the two when both are available, since the
        weakest dimension limits the reliability of the envelope.
        """
        if self.vram_confidence <= 0.0 and self.latency_confidence <= 0.0:
            return 0.0
        if self.vram_confidence <= 0.0:
            return self.latency_confidence
        if self.latency_confidence <= 0.0:
            return self.vram_confidence
        return min(self.vram_confidence, self.latency_confidence)

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence level based on relative estimation error.

        Uses CRLB-backed criterion (see <docs>
        .4):
        - "high": both VRAM and latency relative error ≤ 10%
        - "medium": both ≤ 20%
        - "low": either > 20% or drift_stale
        """
        if self.drift_stale:
            return "low"
        worst_re = max(self.vram_relative_error, self.latency_relative_error)
        if worst_re <= 0.10:
            return "high"
        if worst_re <= 0.20:
            return "medium"
        return "low"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "vram_p50_mib": round(self.vram_p50_mib, 1),
            "vram_p95_mib": round(self.vram_p95_mib, 1),
            "latency_p50_sec": round(self.latency_p50_sec, 3),
            "latency_p95_sec": round(self.latency_p95_sec, 3),
            "workload_class": self.workload_class,
            "interference_compute_slowdown": round(self.interference_compute_slowdown, 3),
            "interference_memory_slowdown": round(self.interference_memory_slowdown, 3),
            "vram_confidence": round(self.vram_confidence, 3),
            "latency_confidence": round(self.latency_confidence, 3),
            "vram_relative_error": round(self.vram_relative_error, 4) if self.vram_relative_error < float("inf") else None,
            "latency_relative_error": round(self.latency_relative_error, 4) if self.latency_relative_error < float("inf") else None,
            "confidence": round(self.confidence, 3),
            "confidence_label": self.confidence_label,
            "n_observations": self.n_observations,
            "drift_stale": self.drift_stale,
            "drift_events_count": self.drift_events_count,
        }



@dataclass
class PlacementContext:
    """System-wide resource snapshot consulted before each scheduling decision."""

    snapshot_at: float = 0.0

    gpu_views: Dict[str, GpuResourceView] = field(default_factory=dict)

    component_envelopes: Dict[str, ComponentEnvelope] = field(default_factory=dict)

    colocation_map: Dict[str, Set[str]] = field(default_factory=dict)

    n_drift_events_total: int = 0
    n_stale_envelopes: int = 0
    last_drift_at: Optional[float] = None

    def gpu_view_for(self, gpu_id: str) -> Optional[GpuResourceView]:
        return self.gpu_views.get(gpu_id)

    def envelope_for(self, component: str) -> Optional[ComponentEnvelope]:
        return self.component_envelopes.get(component)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_at": self.snapshot_at,
            "gpu_views": {gid: v.as_dict() for gid, v in sorted(self.gpu_views.items())},
            "component_envelopes": {
                c: e.as_dict() for c, e in sorted(self.component_envelopes.items())
            },
            "colocation_map": {
                gid: sorted(comps) for gid, comps in sorted(self.colocation_map.items())
            },
            "n_drift_events_total": self.n_drift_events_total,
            "n_stale_envelopes": self.n_stale_envelopes,
            "last_drift_at": self.last_drift_at,
        }



class PlacementContextManager:
    """Maintains the system-wide ``PlacementContext``.

    Owned by ``PlannerService``.  Thread-safe (drift callbacks may arrive from
    the Signal recording path which runs on the event loop, but the manager is
    also read/written from ``generate_plan``).
    """

    def __init__(self) -> None:
        self._scenario = PlacementContext()
        self._lock = threading.Lock()


    def get_scenario(self) -> PlacementContext:
        """Return the current scenario (caller should treat as read-only)."""
        return self._scenario


    def refresh_gpu_views(
        self,
        worker_snapshots: List[Dict[str, Any]],
        resource_tracker: Any = None,
    ) -> None:
        """Rebuild GPU views from live worker state.

        ``worker_snapshots`` is the list returned by
        ``WorkerSupervisor.list_worker_snapshots()``.
        ``resource_tracker`` is the ``ResourceAdmissionTracker`` instance for VRAM totals and
        available memory.

        Called at the start of every ``generate_plan()``.
        """
        gpu_views: Dict[str, GpuResourceView] = {}
        colocation_map: Dict[str, Set[str]] = {}
        gpu_inflight: Dict[str, List[str]] = {}
        gpu_reserved: Dict[str, float] = {}
        gpu_util: Dict[str, float] = {}
        gpu_max_clear: Dict[str, float] = {}

        for snap in worker_snapshots:
            gpu_ids = snap.get("gpu_ids", [])
            component = snap.get("component", "")
            status = snap.get("status", "")
            exec_inflight = int(snap.get("execute_inflight", 0))
            actual_vram = float(snap.get("actual_vram", 0))
            gpu_util_pct = float(snap.get("gpu_util_percent", 0))

            for gid in gpu_ids:
                gid = str(gid)

                gpu_util[gid] = max(gpu_util.get(gid, 0.0), gpu_util_pct)

                if status == "Ready" and exec_inflight > 0:
                    gpu_inflight.setdefault(gid, []).append(component)
                    colocation_map.setdefault(gid, set()).add(component)
                    gpu_reserved[gid] = gpu_reserved.get(gid, 0.0) + actual_vram

                    envelope = self._scenario.component_envelopes.get(component)
                    if envelope and envelope.latency_p50_sec > 0:
                        gpu_max_clear[gid] = max(
                            gpu_max_clear.get(gid, 0.0),
                            envelope.latency_p50_sec,
                        )

        gpu_pool: List[str] = []
        if resource_tracker is not None:
            gpu_pool = list(getattr(resource_tracker, "gpu_pool", []))

        all_gpu_ids = set(gpu_pool) | set(gpu_inflight.keys()) | set(gpu_util.keys())
        for gid in all_gpu_ids:
            total = float(resource_tracker.total_vram.get(gid, 0)) if resource_tracker else 0.0
            available = float(resource_tracker.get_available_memory(gid)) if resource_tracker else 0.0
            gpu_views[gid] = GpuResourceView(
                gpu_id=gid,
                total_vram_mib=total,
                reserved_vram_mib=gpu_reserved.get(gid, 0.0),
                available_vram_mib=available,
                utilization_pct=gpu_util.get(gid, 0.0),
                inflight_components=gpu_inflight.get(gid, []),
                projected_clear_sec=gpu_max_clear.get(gid, 0.0),
            )

        with self._lock:
            self._scenario.gpu_views = gpu_views
            self._scenario.colocation_map = colocation_map
            self._scenario.snapshot_at = time.time()


    def notify_gpu_change(
        self,
        gpu_id: str,
        delta_vram_mb: float,
        component: str,
        is_release: bool,
    ) -> None:
        """Lightweight GPU view update on task dispatch or completion.

        Adjusts reserved/available VRAM and inflight component list without
        a full supervisor snapshot.  The full ``refresh_gpu_views()`` still
        runs at the start of every ``generate_plan()`` call and corrects any
        accumulated drift from these incremental updates.
        """
        with self._lock:
            view = self._scenario.gpu_views.get(str(gpu_id))
            if view is None:
                return
            if is_release:
                view.reserved_vram_mib = max(0, view.reserved_vram_mib - delta_vram_mb)
                view.available_vram_mib = view.total_vram_mib - view.reserved_vram_mib
                if component in view.inflight_components:
                    view.inflight_components.remove(component)
            else:
                view.reserved_vram_mib += delta_vram_mb
                view.available_vram_mib = view.total_vram_mib - view.reserved_vram_mib
                view.inflight_components.append(component)


    def update_component_envelope(
        self,
        component: str,
        compute_signal: Any,
        signal_service: Any,
    ) -> None:
        """Update (or create) a component envelope from a fresh signal query.

        ``compute_signal`` is a ``WorkerComputeSignal`` built after
        ``signal_service.query()``.

        This resets ``drift_stale`` since we now have a fresh observation.
        """
        mem = compute_signal.memory_profile
        run = compute_signal.runtime_profile
        con = compute_signal.concurrency_profile

        profiles = signal_service.resource_profiles
        profile = profiles._profiles.get(component)

        vram_conf = 0.0
        latency_conf = 0.0
        vram_re = float("inf")
        latency_re = float("inf")
        n_obs = 0
        vram_p50 = mem.required_vram_mb
        vram_p95 = mem.expected_peak_vram_mb
        latency_p50 = run.expected_duration_sec
        latency_p95 = run.upper_bound_duration_sec

        if profile:
            vram_conf = profile.overall_vram_confidence()
            latency_conf = profile.overall_latency_confidence()
            n_obs = profile._sample_count
            for cfg in profile._config_baselines.values():
                bl = cfg.resolve()
                if bl:
                    v_gp = bl.vram_mib
                    l_gp = bl.latency_sec
                    if v_gp.n > 0 and v_gp.mean > 0:
                        vram_p50 = v_gp.mean
                        vram_p95 = v_gp.mean + 1.645 * v_gp.std if v_gp.std > 0 else v_gp.mean * 1.2
                    if l_gp.n > 0 and l_gp.mean > 0:
                        latency_p50 = l_gp.mean
                        latency_p95 = l_gp.mean + 1.645 * l_gp.std if l_gp.std > 0 else l_gp.mean * 1.5
                    v_re = v_gp.relative_error(0.0) if callable(getattr(v_gp, 'relative_error', None)) else (v_gp.relative_error if hasattr(v_gp, 'relative_error') else float("inf"))
                    l_re = l_gp.relative_error(0.0) if callable(getattr(l_gp, 'relative_error', None)) else (l_gp.relative_error if hasattr(l_gp, 'relative_error') else float("inf"))
                    if v_re < vram_re:
                        vram_re = v_re
                    if l_re < latency_re:
                        latency_re = l_re

        with self._lock:
            existing = self._scenario.component_envelopes.get(component)
            drift_events_count = existing.drift_events_count if existing else 0

            envelope = ComponentEnvelope(
                component=component,
                vram_p50_mib=vram_p50,
                vram_p95_mib=vram_p95,
                latency_p50_sec=latency_p50,
                latency_p95_sec=latency_p95,
                workload_class=con.workload_class,
                interference_compute_slowdown=con.interference_compute_bound_slowdown,
                interference_memory_slowdown=con.interference_memory_bound_slowdown,
                vram_confidence=vram_conf,
                latency_confidence=latency_conf,
                vram_relative_error=vram_re,
                latency_relative_error=latency_re,
                n_observations=n_obs,
                last_updated_at=time.time(),
                drift_stale=False,
                drift_events_count=drift_events_count,
            )
            self._scenario.component_envelopes[component] = envelope

            self._refresh_stale_envelopes_locked(signal_service)

            self._scenario.n_stale_envelopes = sum(
                1 for e in self._scenario.component_envelopes.values() if e.drift_stale
            )

    def _refresh_stale_envelopes_locked(self, signal_service: Any) -> int:
        """Clear stale flag on envelopes whose GP has been updated since drift.

        Must be called while ``self._lock`` is held.  For each stale envelope,
        re-query the GP and check if observations have arrived since the drift
        event.  If so, refresh the envelope statistics and clear the flag.

        Returns the number of envelopes refreshed.
        """
        if signal_service is None:
            return 0

        profiles = getattr(signal_service, 'resource_profiles', None)
        if profiles is None:
            return 0

        n_refreshed = 0
        for comp, env in list(self._scenario.component_envelopes.items()):
            if not env.drift_stale:
                continue

            profile = profiles._profiles.get(comp)
            if profile is None:
                continue

            current_count = profile._sample_count
            if current_count <= env.n_observations:
                continue

            vram_p50 = env.vram_p50_mib
            vram_p95 = env.vram_p95_mib
            latency_p50 = env.latency_p50_sec
            latency_p95 = env.latency_p95_sec
            vram_conf = 0.0
            latency_conf = 0.0

            try:
                vram_conf = profile.overall_vram_confidence()
                latency_conf = profile.overall_latency_confidence()
                for cfg in profile._config_baselines.values():
                    bl = cfg.resolve()
                    if bl:
                        v_gp = bl.vram_mib
                        l_gp = bl.latency_sec
                        if v_gp.n > 0 and v_gp.mean > 0:
                            vram_p50 = v_gp.mean
                            vram_p95 = v_gp.mean + 1.645 * v_gp.std if v_gp.std > 0 else v_gp.mean * 1.2
                        if l_gp.n > 0 and l_gp.mean > 0:
                            latency_p50 = l_gp.mean
                            latency_p95 = l_gp.mean + 1.645 * l_gp.std if l_gp.std > 0 else l_gp.mean * 1.5
            except Exception:
                _LOG.warning(
                    '[silent-except] %s swallowed an exception; body=%s',
                    __name__, "continue  # can't refresh — stay stale", exc_info=True,
                )
                continue

            env.vram_p50_mib = vram_p50
            env.vram_p95_mib = vram_p95
            env.latency_p50_sec = latency_p50
            env.latency_p95_sec = latency_p95
            env.vram_confidence = vram_conf
            env.latency_confidence = latency_conf
            env.n_observations = current_count
            env.last_updated_at = time.time()
            env.drift_stale = False
            n_refreshed += 1

        if n_refreshed > 0:
            _LOG.info(
                "[scenario] refreshed %d stale envelope(s)", n_refreshed,
            )
        return n_refreshed


    def on_drift(self, event: ProfileDriftEvent) -> int:
        """Mark the component envelope as stale due to a drift event.

        Returns the number of envelopes currently marked stale.
        """
        with self._lock:
            envelope = self._scenario.component_envelopes.get(event.component)
            if envelope is not None:
                envelope.drift_stale = True
                envelope.drift_events_count += 1

            self._scenario.n_drift_events_total += 1
            self._scenario.last_drift_at = time.time()
            self._scenario.n_stale_envelopes = sum(
                1 for e in self._scenario.component_envelopes.values() if e.drift_stale
            )

            _LOG.debug(
                "[scenario-drift] component=%s metric=%s stale_envelopes=%d/%d",
                event.component, event.metric,
                self._scenario.n_stale_envelopes,
                len(self._scenario.component_envelopes),
            )
            return self._scenario.n_stale_envelopes


    def summarize(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of the current scenario."""
        with self._lock:
            return self._scenario.as_dict()
