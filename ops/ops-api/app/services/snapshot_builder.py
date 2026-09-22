from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping


def _source_stale(source: Mapping[str, Any]) -> bool:
    return str(source.get("status") or "") != "ok"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_epoch(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return 0.0


def build_queue_section(filtered_workers: List[Mapping[str, Any]]) -> Dict[str, Any]:
    total_depth = 0
    total_inflight = 0
    by_component: Dict[str, Dict[str, Any]] = {}

    for row in filtered_workers:
        component = str(row.get("component") or "")
        in_queue = int(row.get("in_queue") or 0)
        prepared_queue = int(row.get("prepared_queue") or 0)
        output_queue = int(row.get("output_queue") or 0)
        prepare_inflight = int(row.get("prepare_inflight") or 0)
        execute_inflight = int(row.get("execute_inflight") or 0)
        finalize_inflight = int(row.get("finalize_inflight") or 0)

        depth = in_queue + prepared_queue + output_queue
        inflight = prepare_inflight + execute_inflight + finalize_inflight

        total_depth += depth
        total_inflight += inflight

        bucket = by_component.setdefault(
            component,
            {
                "component": component,
                "workers": 0,
                "queue_depth": 0,
                "inflight": 0,
            },
        )
        bucket["workers"] = int(bucket["workers"]) + 1
        bucket["queue_depth"] = int(bucket["queue_depth"]) + depth
        bucket["inflight"] = int(bucket["inflight"]) + inflight

    return {
        "total_queue_depth": total_depth,
        "total_inflight": total_inflight,
        "by_component": sorted(by_component.values(), key=lambda item: item["component"]),
    }


def _build_signal_summary(
    workers: List[Mapping[str, Any]],
    signals: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a signal summary section for the fleet overview.

    Aggregates per-worker signal data and includes interference model state.
    """
    workload_classes: Dict[str, str] = {}
    co_location_map: Dict[str, List[str]] = {}
    total_signal_tracked = 0

    for w in workers:
        component = str(w.get("component") or "")
        wclass = str(w.get("workload_class") or "unknown")
        addr = str(w.get("addr") or "")
        if component and wclass != "unknown":
            workload_classes[component] = wclass
        co_located = w.get("co_located_components")
        if isinstance(co_located, list) and co_located:
            co_location_map[addr] = [str(c) for c in co_located]
        active = int(w.get("signal_active_task_count") or 0)
        total_signal_tracked += active

    interference_summary: Dict[str, Any] = {}
    if signals:
        interference = signals.get("interference") if isinstance(signals.get("interference"), dict) else {}
        if interference:
            pairwise = interference.get("pairwise") if isinstance(interference.get("pairwise"), dict) else {}
            solo_baselines = interference.get("solo_baselines") if isinstance(interference.get("solo_baselines"), dict) else {}
            solo_vram = interference.get("solo_vram") if isinstance(interference.get("solo_vram"), dict) else {}
            interference_summary = {
                "components_with_solo_baseline": len(solo_baselines),
                "components_with_vram_baseline": len(solo_vram),
                "pairwise_records": len(pairwise),
                "config": interference.get("config", {}),
            }

        profiles = signals.get("workload_profiles") if isinstance(signals.get("workload_profiles"), dict) else {}
        if profiles:
            workload_classes.update({
                comp: str(p.get("workload_class") or "unknown")
                for comp, p in profiles.items()
                if isinstance(p, dict) and str(p.get("workload_class") or "unknown") != "unknown"
            })

    class_counts: Dict[str, int] = {}
    for wclass in workload_classes.values():
        class_counts[wclass] = class_counts.get(wclass, 0) + 1

    return {
        "total_tasks_tracked": total_signal_tracked,
        "co_located_workers": len(co_location_map),
        "workload_classes": workload_classes,
        "workload_class_counts": class_counts,
        "interference": interference_summary,
    }


def build_fleet_section(
    *,
    workers: List[Mapping[str, Any]],
    signals: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    filtered_workers = [dict(row) for row in workers if isinstance(row, Mapping)]
    queue_section = build_queue_section(filtered_workers)
    signal_summary = _build_signal_summary(filtered_workers, signals or {})
    return {"queue": queue_section, "signals": signal_summary}


def build_meta(snapshot: Mapping[str, Any], *, ttl_seconds: float, poll_interval_seconds: float) -> Dict[str, Any]:
    generated_at = str(snapshot.get("generated_at") or _iso_now())
    age_seconds = max(0.0, datetime.now(timezone.utc).timestamp() - _to_epoch(generated_at))

    source_status = snapshot.get("source_status") if isinstance(snapshot.get("source_status"), Mapping) else {}
    stale_reasons: List[str] = []
    for source_name, source in source_status.items():
        if isinstance(source, Mapping) and bool(source.get("required", False)) and _source_stale(source):
            stale_reasons.append(f"source_error:{source_name}")

    if age_seconds > ttl_seconds:
        stale_reasons.append("ttl_expired")

    for source_name, source in source_status.items():
        if not isinstance(source, Mapping):
            continue
        is_stale = age_seconds > ttl_seconds or (bool(source.get("required", False)) and _source_stale(source))
        source["stale"] = bool(is_stale)

    return {
        "generated_at": generated_at,
        "data_age_seconds": round(age_seconds, 3),
        "stale": len(stale_reasons) > 0,
        "stale_reason": stale_reasons,
        "poll_interval_seconds": poll_interval_seconds,
        "ttl_seconds": ttl_seconds,
        "sources": source_status,
    }


def build_overview(
    snapshot: Mapping[str, Any],
    *,
    ttl_seconds: float,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    workers = list(snapshot.get("workers") or [])
    signals = snapshot.get("signals") if isinstance(snapshot.get("signals"), dict) else {}
    fleet_section = build_fleet_section(
        workers=workers,
        signals=signals,
    )

    return {
        "meta": build_meta(
            snapshot,
            ttl_seconds=ttl_seconds,
            poll_interval_seconds=poll_interval_seconds,
        ),
        "fleet": fleet_section,
    }
