from __future__ import annotations

from typing import Any, Dict, Mapping

from app.services.payload_utils import int_payload, mapping_payload, text_payload


MEMORY_UNIT = "mib"
SUPPORT_ORDER = {"none": 0, "coarse": 1, "nearby": 2, "exact": 3}
FALLBACK_ORDER = {"exact": 0, "nearby": 1, "coarse": 2, "none": 3}


def display_unit(unit: str) -> str:
    return "MiB" if unit == MEMORY_UNIT else unit


def support_rank(level: str) -> int:
    return int(SUPPORT_ORDER.get(str(level or "none"), 0))


def fallback_rank(level: str) -> int:
    return int(FALLBACK_ORDER.get(str(level or "none"), FALLBACK_ORDER["none"]))


def support_level(target: Mapping[str, Any]) -> str:
    support = mapping_payload(target.get("support"))
    return text_payload(support.get("level")).lower() or "none"


def fallback_level(target: Mapping[str, Any]) -> str:
    support = mapping_payload(target.get("support"))
    return text_payload(support.get("fallback_level")).lower() or "none"


def effective_support(target: Mapping[str, Any]) -> int:
    support = mapping_payload(target.get("support"))
    if support.get("effective_support") is not None:
        return int_payload(support.get("effective_support"))
    evidence = mapping_payload(target.get("evidence"))
    return int_payload(evidence.get("n_selected_rows"))


def estimate_center(target: Mapping[str, Any]) -> float | None:
    from app.services.payload_utils import float_payload

    estimate = mapping_payload(target.get("estimate"))
    return float_payload(estimate.get("center"))


def estimate_upper(target: Mapping[str, Any]) -> float | None:
    from app.services.payload_utils import float_payload

    estimate = mapping_payload(target.get("estimate"))
    return float_payload(estimate.get("upper"))


def estimate_scope(target: Mapping[str, Any]) -> str | None:
    estimate = mapping_payload(target.get("estimate"))
    value = text_payload(estimate.get("scope"))
    return value or None


def normalized_evidence(target: Mapping[str, Any]) -> Dict[str, int]:
    evidence = mapping_payload(target.get("evidence"))
    return {
        "n_history_rows": int_payload(evidence.get("n_history_rows")),
        "n_exact_rows": int_payload(evidence.get("n_exact_rows")),
        "n_nearby_rows": int_payload(evidence.get("n_nearby_rows")),
        "n_coarse_rows": int_payload(evidence.get("n_coarse_rows")),
        "n_selected_rows": int_payload(evidence.get("n_selected_rows")),
        "n_campaign_correction_rows": int_payload(evidence.get("n_campaign_correction_rows")),
    }
