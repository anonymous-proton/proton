"""predictive_probe.py — env-gated, read-only overhead probe for the
predictive scheduling path (proton_phase).

Enabled ONLY when ``PROTON_PREDICTIVE_PROBE=1``.  Installs pure timing /
counting wrappers around existing scheduler methods.  Wrappers:

- never modify arguments, return values, or exceptions (they are re-raised
  unchanged);
- never take scheduler locks or touch scheduler state;
- aggregate into in-memory histograms and counters, flushed by a daemon
  thread every ``PROTON_PREDICTIVE_PROBE_FLUSH_SEC`` (default 5.0s) as one
  JSON line to ``PROTON_PREDICTIVE_PROBE_PATH`` (default
  ``/tmp/predictive_probe.jsonl``).

Latency is bucketed into power-of-two nanosecond buckets (``bit_length``),
so p50/p95 are bucket-lower-bound approximations; ``max_ns`` and
``total_ns`` are exact.  Result-derived counters (e.g. entries removed by
``invalidate_component``) are captured from return values when they are
ints.

Categories:
- ``maint.*``    prediction maintenance (create/update/query projected state)
- ``inval.*``    invalidation (remove/mark stale predictions)
- ``replan.*``   planner entry points (solve / placement) driven by wakes
- ``wake.*``     event counters (no timing)

This module is instrumentation ONLY — it must never change scheduling
behavior.  It is safe to delete along with its single call site in
``gateway/__main__.py`` after the measurement campaign.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from typing import Any

_ENABLED = os.environ.get("PROTON_PREDICTIVE_PROBE", "") == "1"
try:
    _FLUSH_SEC = float(os.environ.get("PROTON_PREDICTIVE_PROBE_FLUSH_SEC", "5") or 5)
except (TypeError, ValueError):
    _FLUSH_SEC = 5.0
_OUT_PATH = os.environ.get(
    "PROTON_PREDICTIVE_PROBE_PATH", "/tmp/predictive_probe.jsonl"
)

_LOCK = threading.Lock()
_PHASES: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"count": 0, "total_ns": 0, "max_ns": 0, "buckets": defaultdict(int)}
)
_COUNTERS: dict[str, int] = defaultdict(int)
_ATTRS: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_LAST_COMMIT_GPU: dict[str, str] = {}
_installed = False
_started_at = time.time()


def _bucket_index(ns: int) -> int:
    try:
        return max(1, int(ns).bit_length())
    except (TypeError, ValueError):
        return 1


def record_elapsed_ns(phase: str, elapsed_ns: int) -> None:
    with _LOCK:
        stats = _PHASES[phase]
        stats["count"] += 1
        stats["total_ns"] += elapsed_ns
        if elapsed_ns > stats["max_ns"]:
            stats["max_ns"] = elapsed_ns
        stats["buckets"][_bucket_index(elapsed_ns)] += 1


def count(counter: str, value: int = 1, **attrs: int) -> None:
    with _LOCK:
        _COUNTERS[counter] += value
        for key, val in attrs.items():
            try:
                _ATTRS[counter][key] += int(val)
            except (TypeError, ValueError):
                continue


def _snapshot() -> dict[str, Any]:
    with _LOCK:
        phases = {}
        for name, stats in _PHASES.items():
            buckets = dict(stats["buckets"])
            phases[name] = {
                "count": stats["count"],
                "total_ns": stats["total_ns"],
                "max_ns": stats["max_ns"],
                "buckets": buckets,
            }
        return {
            "kind": "predictive_probe_snapshot",
            "ts": time.time(),
            "elapsed_sec": time.time() - _started_at,
            "phases": phases,
            "counters": dict(_COUNTERS),
            "counter_attrs": {k: dict(v) for k, v in _ATTRS.items()},
        }


def flush_snapshot(final: bool = False) -> None:
    try:
        snap = _snapshot()
        snap["final"] = final
        with open(_OUT_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(snap, separators=(",", ":")) + "\n")
    except Exception:
        _LOG_WARN("probe flush failed")


def _LOG_WARN(msg: str) -> None:
    print(f"[predictive-probe] {msg}", file=sys.stderr, flush=True)


def _pct_from_buckets(buckets: dict[int, int], total: int, pct: float) -> int | None:
    if not total:
        return None
    target = total * pct
    cumulative = 0
    for bucket in sorted(buckets):
        cumulative += buckets[bucket]
        if cumulative >= target:
            return 0 if bucket <= 1 else 1 << (bucket - 1)
    return None


def summarize(snapshot: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"phases": {}, "counters": snapshot.get("counters", {})}
    for name, stats in snapshot["phases"].items():
        total = stats["count"]
        out["phases"][name] = {
            "count": total,
            "total_ms": stats["total_ns"] / 1e6,
            "mean_ms": (stats["total_ns"] / total / 1e6) if total else 0.0,
            "max_ms": stats["max_ns"] / 1e6,
            "p50_ms": (_pct_from_buckets(stats["buckets"], total, 0.50) or 0) / 1e6,
            "p95_ms": (_pct_from_buckets(stats["buckets"], total, 0.95) or 0) / 1e6,
        }
    return out


def _wrap_commit(orig: Callable) -> Callable:
    """Time commit_dispatch_plan_prediction and track per-task GPU stability."""

    def wrapper(self, plan: Any, *args: Any, **kwargs: Any):
        start = time.perf_counter_ns()
        try:
            result = orig(self, plan, *args, **kwargs)
        finally:
            record_elapsed_ns("replan.commit_plan_prediction", time.perf_counter_ns() - start)
        try:
            task_id = str(getattr(plan, "task_id", "") or "")
            gpu_id = str(getattr(plan, "target_gpu_id", "") or "")
            if task_id and gpu_id:
                with _LOCK:
                    prev = _LAST_COMMIT_GPU.get(task_id)
                    _LAST_COMMIT_GPU[task_id] = gpu_id
                if prev is None:
                    count("decision.commit_first")
                elif prev == gpu_id:
                    count("decision.commit_same_gpu")
                else:
                    count("decision.commit_changed_gpu")
        except Exception:
            pass
        return result

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


def _wrap_dispatched(orig: Callable) -> Callable:
    """Track whether each dispatch lands on the last-committed GPU."""

    def wrapper(self, task_id: str, *args: Any, **kwargs: Any):
        try:
            gpu_id = str(args[3]) if len(args) > 3 else str(kwargs.get("gpu_id", "") or "")
            if task_id and gpu_id:
                with _LOCK:
                    prev = _LAST_COMMIT_GPU.get(str(task_id))
                if prev is None:
                    count("decision.dispatch_no_prior_commit")
                elif prev == gpu_id:
                    count("decision.dispatch_on_last_committed_gpu")
                else:
                    count("decision.dispatch_gpu_changed_since_commit")
        except Exception:
            pass
        return orig(self, task_id, *args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


def _wrap_sync(orig: Callable, phase: str, result_counter: str | None = None) -> Callable:
    def wrapper(self, *args: Any, **kwargs: Any):
        start = time.perf_counter_ns()
        try:
            result = orig(self, *args, **kwargs)
        finally:
            record_elapsed_ns(phase, time.perf_counter_ns() - start)
        if result_counter is not None:
            try:
                value = int(result)
                if value:
                    count(result_counter, value)
            except (TypeError, ValueError):
                pass
        return result

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


def _wrap_async(orig: Callable, phase: str) -> Callable:
    async def wrapper(self, *args: Any, **kwargs: Any):
        start = time.perf_counter_ns()
        try:
            return await orig(self, *args, **kwargs)
        finally:
            record_elapsed_ns(phase, time.perf_counter_ns() - start)

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


def _wrap_counter(orig: Callable, counter: str, attr_from_args: Callable | None = None) -> Callable:
    def wrapper(self, *args: Any, **kwargs: Any):
        count(counter)
        if attr_from_args is not None:
            try:
                attrs = attr_from_args(args, kwargs)
                if attrs:
                    count(counter, 0, **attrs)
            except Exception:
                pass
        return orig(self, *args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", "wrapped")
    return wrapper


_TARGETS: tuple[tuple[str, str, str, str, str, str | None], ...] = (
    ("gateway.planning.scenario", "SchedulingScenario", "add_predicted_entry", "maint.add_predicted_entry", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "find_entry", "maint.find_entry", "time", None),
    ("gateway.planning.scenario", "GpuTimeline", "available_vram_at", "maint.gpu_timeline_available_vram_at", "time", None),
    ("gateway.planning.scenario", "GpuTimeline", "earliest_fit_time", "maint.gpu_timeline_earliest_fit_time", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "earliest_dual_fit_time", "maint.earliest_dual_fit_time", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "candidate_interval_fits", "maint.candidate_interval_fits", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "best_gpu_for", "maint.best_gpu_for", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "next_predicted_completion", "maint.next_predicted_completion", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "available_vram_at", "maint.available_vram_at", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "on_task_submit", "maint.on_task_submit", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_project_task", "maint.project_task", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_project_downstream", "maint.project_downstream", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "on_task_dispatched", "maint.on_task_dispatched", "dispatched", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "on_task_complete", "maint.on_task_complete", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "plan_fan_out", "maint.plan_fan_out", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_check_pre_init", "maint.check_pre_init", "time", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "_refresh_phase_model", "maint.refresh_phase_model", "time", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "_build_phase_states", "maint.build_phase_states", "time", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "_build_primary_envelope", "maint.build_primary_envelope", "time", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "_update_estimates", "maint.update_estimates", "time", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "_find_backfill_slack", "maint.find_backfill_slack", "time", None),
    ("gateway.planning.scenario", "SchedulingScenario", "invalidate_component", "inval.invalidate_component", "time", "pred_invalidated_entries"),
    ("gateway.planning.scenario", "SchedulingScenario", "prune_stale_predicted", "inval.prune_stale_predicted", "time", "pred_pruned_entries"),
    ("gateway.planning.scenario", "SchedulingScenario", "gc_orphaned_predicted_entries", "inval.gc_orphaned_predicted", "time", "pred_gc_orphaned"),
    ("gateway.planning.scenario", "SchedulingScenario", "remove_predicted_entries_for_gpu", "inval.remove_predicted_for_gpu", "time", "pred_removed_for_gpu"),
    ("gateway.planning.scenario", "SchedulingScenario", "remove_predicted_entries_for_task", "inval.remove_predicted_for_task", "time", "pred_removed_for_task"),
    ("gateway.planning.scenario", "SchedulingScenario", "remove_predicted_entries_for_tasks", "inval.remove_predicted_for_tasks", "time", "pred_removed_for_tasks"),
    ("gateway.planning.scenario", "SchedulingScenario", "remove_predicted_entries_for_worker", "inval.remove_predicted_for_worker", "time", "pred_removed_for_worker"),
    ("gateway.planning.scenario", "SchedulingScenario", "remove_all_entries", "inval.remove_all_entries", "time", "pred_removed_all"),
    ("gateway.planning.scenario", "SchedulingScenario", "promote_predicted_to_active", "inval.promote_predicted_to_active", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_on_scenario_changed_impl", "inval.on_scenario_changed", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_drift_apply_per_component", "inval.drift_apply_per_component", "time", None),
    ("gateway.planning.campaign_scheduler", "CampaignScheduler", "_flush_profile_drift_batch", "inval.flush_profile_drift_batch", "time", None),
    ("gateway.planning.global_planner", "GlobalPlanner", "solve", "replan.solve", "time", None),
    ("gateway.planning.global_planner", "GlobalPlanner", "solve_admissible", "replan.solve_admissible", "time_async", None),
    ("gateway.planning.global_planner", "GlobalPlanner", "_solve_impl", "replan.solve_impl", "time", None),
    ("gateway.planning.global_planner", "GlobalPlanner", "_place_single", "replan.place_single", "time", None),
    ("gateway.planning.global_planner", "GlobalPlanner", "commit_dispatch_plan_prediction", "replan.commit_plan_prediction", "commit", None),
    ("gateway.planning.phase_campaign_planner", "PhaseCampaignPlanner", "solve_admissible", "replan.phase_solve_admissible", "time_async", None),
    ("gateway.planning.scheduling_supervisor", "SchedulingSupervisor", "notify_wake", "wake.notify_wake", "count", None),
    ("gateway.planning.scheduling_supervisor", "SchedulingSupervisor", "submit", "wake.supervisor_submit", "count", None),
)


def maybe_install() -> bool:
    """Install wrappers when PROTON_PREDICTIVE_PROBE=1.  Idempotent."""
    global _installed
    if not _ENABLED or _installed:
        return False
    _installed = True
    installed_count = 0
    for module_name, class_name, method_name, phase, kind, result_counter in _TARGETS:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is None:
                count("probe_install_misses")
                continue
            orig = getattr(cls, method_name, None)
            if orig is None or getattr(orig, "_predictive_probe_wrapped", False):
                count("probe_install_misses")
                continue
            if kind == "time":
                wrapped = _wrap_sync(orig, phase, result_counter)
            elif kind == "time_async":
                wrapped = _wrap_async(orig, phase)
            elif kind == "commit":
                wrapped = _wrap_commit(orig)
            elif kind == "dispatched":
                wrapped = _wrap_dispatched(orig)
            else:
                wrapped = _wrap_counter(orig, phase)
            wrapped._predictive_probe_wrapped = True
            setattr(cls, method_name, wrapped)
            installed_count += 1
        except Exception:
            count("probe_install_errors")
            _LOG_WARN(f"install failed for {class_name}.{method_name}:\n{traceback.format_exc()}")
    count("probe_installed_targets", installed_count)

    import atexit
    import signal

    atexit.register(flush_snapshot, True)
    _prev_term = signal.getsignal(signal.SIGTERM)

    def _on_term(signum, frame):
        flush_snapshot(final=True)
        if callable(_prev_term):
            _prev_term(signum, frame)
        else:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, _on_term)

    def _flush_loop() -> None:
        while True:
            time.sleep(_FLUSH_SEC)
            flush_snapshot()

    thread = threading.Thread(target=_flush_loop, name="predictive-probe-flush", daemon=True)
    thread.start()
    _LOG_WARN(f"installed {installed_count} wrappers, output={_OUT_PATH}")
    return True
