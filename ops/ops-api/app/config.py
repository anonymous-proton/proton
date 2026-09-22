from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gateway_base_url: str
    poll_interval_seconds: float
    snapshot_ttl_seconds: float
    gateway_timeout_seconds: float
    ops_api_port: int


REQUIRED_SOURCES: tuple[str, ...] = (
    "gateway_health",
    "gateway_workers",
)



def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value



def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value



def load_settings() -> Settings:
    gateway_base_url = os.getenv("GATEWAY_BASE_URL", "http://127.0.0.7:8098").strip().rstrip("/")

    return Settings(
        gateway_base_url=gateway_base_url,
        poll_interval_seconds=_as_float("POLL_INTERVAL_SECONDS", 2.0),
        snapshot_ttl_seconds=_as_float("SNAPSHOT_TTL_SECONDS", 5.0),
        gateway_timeout_seconds=_as_float("GATEWAY_TIMEOUT_SECONDS", 2.0),
        ops_api_port=_as_int("OPS_API_PORT", 18098),
    )
