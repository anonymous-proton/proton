from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import quote, urlencode

import httpx
from fastapi import FastAPI, HTTPException, Query

from app.adapters.gateway_runs import normalize_run_payload, normalize_runs_payload
from app.config import Settings, load_settings
from app.models import OverviewResponse, RecentRunsResponse, RunDetailResponse
from app.services.poller import OpsPoller
from app.services.run_views import build_recent_runs_payload, build_run_detail_payload
from app.services.snapshot_builder import build_meta, build_overview
from app.state import SnapshotStore


def _normalize_limit(value: int, *, default: int, min_v: int, max_v: int) -> int:
    if value < min_v:
        return default
    if value > max_v:
        return max_v
    return value


def _clean_repeated(values: Sequence[str] | None, *, lower: bool = False) -> list[str]:
    out: list[str] = []
    for raw in list(values or []):
        text = str(raw or "").strip()
        if not text:
            continue
        token = text.lower() if lower else text
        out.append(token)
    return list(dict.fromkeys(out))


def _add_repeated(params: Dict[str, Any], key: str, values: Sequence[str]) -> None:
    cleaned = [str(item).strip() for item in list(values or []) if str(item).strip()]
    if cleaned:
        params[key] = cleaned


def create_app(*, settings: Optional[Settings] = None, start_poller: bool = True) -> FastAPI:
    settings_obj = settings or load_settings()
    store = SnapshotStore()
    poller = OpsPoller(settings_obj, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings_obj
        app.state.store = store
        app.state.poller = poller
        if start_poller:
            poller.start()
        try:
            yield
        finally:
            if start_poller:
                await poller.stop()

    app = FastAPI(title="ops-api", version="1.0.0", lifespan=lifespan)

    async def _fetch_gateway_json(path: str) -> tuple[Optional[Dict[str, Any]], int, Optional[str]]:
        url = f"{settings_obj.gateway_base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=settings_obj.gateway_timeout_seconds) as client:
                resp = await client.get(url)
        except Exception as exc:
            return None, 0, f"gateway request failed: {type(exc).__name__}: {exc}"

        status = int(resp.status_code)
        try:
            payload = resp.json()
        except Exception as exc:
            return None, status, f"gateway returned invalid json for {path}: {exc}"
        if not isinstance(payload, dict):
            return None, status, f"gateway returned non-object json for {path}"
        if status != 200:
            detail = payload.get("error") if isinstance(payload, Mapping) else None
            return payload, status, f"gateway HTTP {status} for {path}: {detail or 'request failed'}"
        return payload, status, None

    def _gateway_runs_path(
        *,
        limit: int,
        campaign_id: str = "",
        component: str = "",
        worker_name: str = "",
        from_ts: float | None = None,
        include_gateway_instance_id: Sequence[str] = (),
        exclude_gateway_instance_id: Sequence[str] = (),
        include_gateway_git_commit: Sequence[str] = (),
        exclude_gateway_git_commit: Sequence[str] = (),
    ) -> str:
        params: Dict[str, Any] = {
            "run_source": "task",
            "sort": "updated_at:desc",
            "limit": int(limit),
            "offset": 0,
        }
        if campaign_id:
            params["campaign_id"] = campaign_id
        if component:
            params["component"] = component
        if worker_name:
            params["worker_name"] = worker_name
        if from_ts is not None:
            params["from_ts"] = from_ts
        _add_repeated(params, "include_gateway_instance_id", include_gateway_instance_id)
        _add_repeated(params, "exclude_gateway_instance_id", exclude_gateway_instance_id)
        _add_repeated(params, "include_gateway_git_commit", include_gateway_git_commit)
        _add_repeated(params, "exclude_gateway_git_commit", exclude_gateway_git_commit)
        return f"/api/v1/profile/runs?{urlencode(params, doseq=True)}"

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        snap = store.get()
        meta = build_meta(
            snap,
            ttl_seconds=settings_obj.snapshot_ttl_seconds,
            poll_interval_seconds=settings_obj.poll_interval_seconds,
        )
        return {
            "status": "ok",
            "stale": meta.get("stale", True),
            "generated_at": meta.get("generated_at"),
            "data_age_seconds": meta.get("data_age_seconds"),
            "sources": meta.get("sources", {}),
        }

    @app.get("/api/v1/ops/overview", response_model=OverviewResponse)
    async def get_overview() -> Dict[str, Any]:
        snap = store.get()
        return build_overview(
            snap,
            ttl_seconds=settings_obj.snapshot_ttl_seconds,
            poll_interval_seconds=settings_obj.poll_interval_seconds,
        )

    @app.get("/api/v1/ops/campaigns")
    async def get_campaigns(
        limit: int = Query(default=200),
        offset: int = Query(default=0),
        include_gateway_instance_id: list[str] = Query(default=[]),
        exclude_gateway_instance_id: list[str] = Query(default=[]),
        include_gateway_git_commit: list[str] = Query(default=[]),
        exclude_gateway_git_commit: list[str] = Query(default=[]),
    ) -> Dict[str, Any]:
        capped_limit = _normalize_limit(limit, default=200, min_v=1, max_v=1000)
        capped_offset = max(0, int(offset))
        params: Dict[str, Any] = {"limit": int(capped_limit), "offset": int(capped_offset)}
        _add_repeated(params, "include_gateway_instance_id", _clean_repeated(include_gateway_instance_id))
        _add_repeated(params, "exclude_gateway_instance_id", _clean_repeated(exclude_gateway_instance_id))
        _add_repeated(params, "include_gateway_git_commit", _clean_repeated(include_gateway_git_commit, lower=True))
        _add_repeated(params, "exclude_gateway_git_commit", _clean_repeated(exclude_gateway_git_commit, lower=True))
        payload, status, error = await _fetch_gateway_json(f"/api/v1/campaigns?{urlencode(params, doseq=True)}")
        if status == 200 and payload is not None:
            return payload
        raise HTTPException(status_code=status or 502, detail=error or "failed to fetch campaigns")

    @app.get("/api/v1/ops/campaigns/{campaign_id}")
    async def get_campaign_detail(
        campaign_id: str,
        include_gateway_instance_id: list[str] = Query(default=[]),
        exclude_gateway_instance_id: list[str] = Query(default=[]),
        include_gateway_git_commit: list[str] = Query(default=[]),
        exclude_gateway_git_commit: list[str] = Query(default=[]),
    ) -> Dict[str, Any]:
        cleaned_campaign_id = str(campaign_id or "").strip()
        if not cleaned_campaign_id:
            raise HTTPException(status_code=400, detail="campaign_id is required")
        params: Dict[str, Any] = {}
        _add_repeated(params, "include_gateway_instance_id", _clean_repeated(include_gateway_instance_id))
        _add_repeated(params, "exclude_gateway_instance_id", _clean_repeated(exclude_gateway_instance_id))
        _add_repeated(params, "include_gateway_git_commit", _clean_repeated(include_gateway_git_commit, lower=True))
        _add_repeated(params, "exclude_gateway_git_commit", _clean_repeated(exclude_gateway_git_commit, lower=True))
        suffix = f"?{urlencode(params, doseq=True)}" if params else ""
        payload, status, error = await _fetch_gateway_json(f"/api/v1/campaigns/{cleaned_campaign_id}{suffix}")
        if status == 200 and payload is not None:
            return payload
        raise HTTPException(status_code=status or 502, detail=error or "failed to fetch campaign detail")

    @app.get("/api/v1/ops/runs/recent", response_model=RecentRunsResponse)
    async def get_recent_runs(
        limit: int = Query(default=50),
        campaign_id: str = Query(default=""),
        component: str = Query(default=""),
        worker_name: str = Query(default=""),
        include_gateway_instance_id: list[str] = Query(default=[]),
        exclude_gateway_instance_id: list[str] = Query(default=[]),
        include_gateway_git_commit: list[str] = Query(default=[]),
        exclude_gateway_git_commit: list[str] = Query(default=[]),
    ) -> Dict[str, Any]:
        capped_limit = _normalize_limit(limit, default=50, min_v=1, max_v=200)
        cleaned_campaign_id = str(campaign_id or "").strip()
        cleaned_component = str(component or "").strip().lower()
        cleaned_worker_name = str(worker_name or "").strip()
        include_instance_ids = _clean_repeated(include_gateway_instance_id)
        exclude_instance_ids = _clean_repeated(exclude_gateway_instance_id)
        include_git_commits = _clean_repeated(include_gateway_git_commit, lower=True)
        exclude_git_commits = _clean_repeated(exclude_gateway_git_commit, lower=True)

        table_path = _gateway_runs_path(
            limit=capped_limit,
            campaign_id=cleaned_campaign_id,
            component=cleaned_component,
            worker_name=cleaned_worker_name,
            include_gateway_instance_id=include_instance_ids,
            exclude_gateway_instance_id=exclude_instance_ids,
            include_gateway_git_commit=include_git_commits,
            exclude_gateway_git_commit=exclude_git_commits,
        )
        window_path = _gateway_runs_path(
            limit=500,
            campaign_id=cleaned_campaign_id,
            component=cleaned_component,
            worker_name=cleaned_worker_name,
            from_ts=time.time() - 3600.0,
            include_gateway_instance_id=include_instance_ids,
            exclude_gateway_instance_id=exclude_instance_ids,
            include_gateway_git_commit=include_git_commits,
            exclude_gateway_git_commit=exclude_git_commits,
        )
        table_payload, table_status, table_error = await _fetch_gateway_json(table_path)
        if table_status != 200 or table_payload is None:
            raise HTTPException(status_code=table_status or 502, detail=table_error or "failed to fetch recent runs")

        window_payload, window_status, window_error = await _fetch_gateway_json(window_path)
        if window_status != 200 or window_payload is None:
            raise HTTPException(status_code=window_status or 502, detail=window_error or "failed to fetch run summary")

        normalized_table = normalize_runs_payload(table_payload)
        normalized_window = normalize_runs_payload(window_payload)
        return build_recent_runs_payload(
            rows=normalized_table["runs"],
            summary_rows=normalized_window["runs"],
            filters={
                "campaign_id": cleaned_campaign_id,
                "component": cleaned_component,
                "worker_name": cleaned_worker_name,
                "include_gateway_instance_id": include_instance_ids,
                "exclude_gateway_instance_id": exclude_instance_ids,
                "include_gateway_git_commit": include_git_commits,
                "exclude_gateway_git_commit": exclude_git_commits,
            },
            total=int(normalized_table["total"]),
            limit=int(normalized_table["limit"]),
            offset=int(normalized_table["offset"]),
        )

    @app.get("/api/v1/ops/signals")
    async def get_signals() -> Dict[str, Any]:
        """Full SignalService state: activation peaks, latency trackers,
        interference registry (baselines, pairwise matrix, workload classes)."""
        snap = store.get()
        signals = snap.get("signals") if isinstance(snap.get("signals"), dict) else {}
        meta = build_meta(
            snap,
            ttl_seconds=settings_obj.snapshot_ttl_seconds,
            poll_interval_seconds=settings_obj.poll_interval_seconds,
        )
        return {"meta": meta, "signals": signals}

    @app.get("/api/v1/ops/runs/{run_key}", response_model=RunDetailResponse)
    async def get_run_detail(run_key: str) -> Dict[str, Any]:
        cleaned_run_key = str(run_key or "").strip()
        if not cleaned_run_key:
            raise HTTPException(status_code=400, detail="run_key is required")
        encoded_run_key = quote(cleaned_run_key, safe="")
        payload, status, error = await _fetch_gateway_json(f"/api/v1/profile/runs/{encoded_run_key}")
        if status != 200 or payload is None:
            raise HTTPException(status_code=status or 502, detail=error or "failed to fetch run detail")
        normalized = normalize_run_payload(payload)
        return build_run_detail_payload(normalized)

    return app


app = create_app()
