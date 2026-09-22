from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence


UNKNOWN_GATEWAY_VALUE = "unknown"


def mapping_payload(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(value or []) if isinstance(item, Mapping)]


def text_payload(value: Any) -> str:
    return str(value or "").strip()


def float_payload(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or math.isinf(parsed):
        return None
    return float(parsed)


def int_payload(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def sum_payload(values: Sequence[float | None]) -> float | None:
    existing = [float(value) for value in values if value is not None]
    if not existing:
        return None
    return float(sum(existing))


def normalize_gateway_payload(value: Mapping[str, Any] | None) -> Dict[str, Any]:
    gateway = mapping_payload(value)
    instance_id = text_payload(gateway.get("instance_id")) or UNKNOWN_GATEWAY_VALUE
    bind_addr = text_payload(gateway.get("bind_addr")) or UNKNOWN_GATEWAY_VALUE
    git_commit = text_payload(gateway.get("git_commit")).lower() or UNKNOWN_GATEWAY_VALUE
    label = text_payload(gateway.get("label"))
    if not label:
        if bind_addr == UNKNOWN_GATEWAY_VALUE and git_commit == UNKNOWN_GATEWAY_VALUE:
            label = UNKNOWN_GATEWAY_VALUE
        elif bind_addr == UNKNOWN_GATEWAY_VALUE:
            label = git_commit[:7]
        elif git_commit == UNKNOWN_GATEWAY_VALUE:
            label = bind_addr
        else:
            label = f"{bind_addr} | {git_commit[:7]}"
    return {
        "instance_id": instance_id,
        "bind_addr": bind_addr,
        "git_commit": git_commit,
        "started_at": float_payload(gateway.get("started_at")),
        "label": label,
    }
