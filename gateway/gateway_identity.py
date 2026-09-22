from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Optional


UNKNOWN_GATEWAY_VALUE = "unknown"


def normalize_gateway_instance_id(value: Any) -> str:
    token = str(value or "").strip()
    return token or UNKNOWN_GATEWAY_VALUE


def normalize_gateway_bind_addr(value: Any) -> str:
    token = str(value or "").strip()
    return token or UNKNOWN_GATEWAY_VALUE


def normalize_gateway_git_commit(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token or UNKNOWN_GATEWAY_VALUE


def gateway_label(*, bind_addr: Any, git_commit: Any) -> str:
    normalized_bind_addr = normalize_gateway_bind_addr(bind_addr)
    normalized_git_commit = normalize_gateway_git_commit(git_commit)
    if normalized_bind_addr == UNKNOWN_GATEWAY_VALUE and normalized_git_commit == UNKNOWN_GATEWAY_VALUE:
        return UNKNOWN_GATEWAY_VALUE
    parts = []
    if normalized_bind_addr != UNKNOWN_GATEWAY_VALUE:
        parts.append(normalized_bind_addr)
    if normalized_git_commit != UNKNOWN_GATEWAY_VALUE:
        parts.append(normalized_git_commit[:7])
    return " | ".join(parts) or UNKNOWN_GATEWAY_VALUE


def gateway_payload_from_mapping(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    started_at = raw.get("started_at")
    try:
        parsed_started_at = float(started_at)
    except Exception:
        parsed_started_at = 0.0
    if parsed_started_at < 0:
        parsed_started_at = 0.0
    bind_addr = normalize_gateway_bind_addr(raw.get("bind_addr"))
    git_commit = normalize_gateway_git_commit(raw.get("git_commit"))
    return {
        "instance_id": normalize_gateway_instance_id(raw.get("instance_id")),
        "bind_addr": bind_addr,
        "git_commit": git_commit,
        "started_at": parsed_started_at if parsed_started_at > 0 else None,
        "label": gateway_label(bind_addr=bind_addr, git_commit=git_commit),
    }


@dataclass(frozen=True)
class GatewayIdentity:
    instance_id: str
    bind_addr: str
    git_commit: str
    started_at: float

    def as_dict(self) -> dict[str, Any]:
        return gateway_payload_from_mapping(
            {
                "instance_id": self.instance_id,
                "bind_addr": self.bind_addr,
                "git_commit": self.git_commit,
                "started_at": self.started_at,
            }
        )


def _startup_tag(started_at: float) -> str:
    return datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_bind_fragment(bind_addr: str) -> str:
    return bind_addr.replace(":", "_").replace("/", "_")


def discover_git_commit(*, repo_root: Optional[Path] = None) -> str:
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return UNKNOWN_GATEWAY_VALUE
    return normalize_gateway_git_commit(output)


def build_gateway_identity(
    *,
    host: str,
    port: int,
    repo_root: Optional[Path] = None,
    started_at: Optional[float] = None,
    git_commit: Optional[str] = None,
) -> GatewayIdentity:
    bind_addr = normalize_gateway_bind_addr(f"{str(host).strip()}:{int(port)}")
    started = float(started_at if started_at is not None else time.time())
    commit = normalize_gateway_git_commit(git_commit or discover_git_commit(repo_root=repo_root))
    startup_tag = _startup_tag(started)
    safe_bind = _safe_bind_fragment(bind_addr)
    digest = hashlib.sha1(f"{bind_addr}|{commit}|{started:.6f}".encode("utf-8")).hexdigest()[:10]
    instance_id = normalize_gateway_instance_id(f"gw-{safe_bind}-{startup_tag}-{digest}")
    return GatewayIdentity(
        instance_id=instance_id,
        bind_addr=bind_addr,
        git_commit=commit,
        started_at=started,
    )
