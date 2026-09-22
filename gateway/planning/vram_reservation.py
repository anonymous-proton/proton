"""Pluggable GPU VRAM reservation models.

The selector changes only how a timeline entry reports its reservation.  The
scheduler, placement policy, and hard-capacity path remain shared by all models.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

_LOG = logging.getLogger(__name__)

FULL_WALL = "full_wall"
TEMPORAL_PEAK_INTERVAL = "temporal_peak_interval"
_ALLOWED_MODELS = {FULL_WALL, TEMPORAL_PEAK_INTERVAL}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class VramReservation:
    """Optional temporal fields for one timeline entry."""

    allow_vram_mb: float | None = None
    peak_start_time: float | None = None
    peak_end_time: float | None = None


class VramReservationModel(ABC):
    name: str = FULL_WALL

    def __init__(self, signal_service: Any = None, resource_z: float = 1.96) -> None:
        self._signal_service = signal_service
        parsed_z = _finite_float(resource_z)
        self._resource_z = parsed_z if parsed_z is not None and parsed_z > 0 else 1.96

    @abstractmethod
    def reservation_for(
        self,
        *,
        component: str,
        config_fingerprint: str,
        gpu_id: str,
        start_time: float,
        predicted_end_time: float,
        predicted_vram_mb: float,
        input_size: float = 0.0,
    ) -> VramReservation:
        raise NotImplementedError


class FullWallVramReservationModel(VramReservationModel):
    name = FULL_WALL

    def reservation_for(self, **_: Any) -> VramReservation:
        return VramReservation()


class TemporalPeakIntervalVramReservationModel(VramReservationModel):
    name = TEMPORAL_PEAK_INTERVAL

    def __init__(self, signal_service: Any = None, resource_z: float = 1.96) -> None:
        super().__init__(signal_service=signal_service, resource_z=resource_z)
        self._temporal_profile_cache: dict[
            tuple[str, str, str, float], dict[str, float] | None
        ] = {}

    def _query_temporal_profile(
        self,
        *,
        component: str,
        config_fingerprint: str,
        gpu_id: str,
        input_size: float,
    ) -> dict[str, float] | None:
        parsed_input = _finite_float(input_size)
        if parsed_input is None:
            return None
        key = (
            str(component),
            str(config_fingerprint or ""),
            str(gpu_id),
            parsed_input,
        )
        if key in self._temporal_profile_cache:
            return self._temporal_profile_cache[key]
        profile_registry = getattr(self._signal_service, "resource_profiles", None)
        query = getattr(profile_registry, "predict_temporal_vram", None)
        raw_result: Any = None
        if callable(query):
            try:
                raw_result = query(
                    component,
                    config_fingerprint,
                    input_size=parsed_input,
                    gpu_id=gpu_id,
                    z=self._resource_z,
                )
            except Exception:
                _LOG.warning(
                    "[vram-reservation] temporal profile query failed; using full wall",
                    exc_info=True,
                )
        result: dict[str, float] | None
        if isinstance(raw_result, dict):
            low = _finite_float(raw_result.get("low_vram"))
            start_sec = _finite_float(raw_result.get("peak_start_sec"))
            duration_sec = _finite_float(raw_result.get("peak_duration_sec"))
            if (
                low is None
                or start_sec is None
                or duration_sec is None
                or start_sec <= 0
                or duration_sec <= 0
                or low < 0
            ):
                result = None
            else:
                result = {
                    "low_vram": low,
                    "peak_start_sec": start_sec,
                    "peak_duration_sec": duration_sec,
                }
        else:
            result = None
        self._temporal_profile_cache[key] = result
        return result

    def invalidate_temporal_profile(
        self, component: str = "", config_fingerprint: str = ""
    ) -> None:
        """Drop cached profiles after a new observation."""
        comp = str(component or "")
        cfg = str(config_fingerprint or "")
        if not comp and not cfg:
            self._temporal_profile_cache.clear()
            return
        self._temporal_profile_cache = {
            key: value
            for key, value in self._temporal_profile_cache.items()
            if (comp and key[0] != comp) or (cfg and key[1] != cfg)
        }

    def reservation_for(
        self,
        *,
        component: str,
        config_fingerprint: str,
        gpu_id: str,
        start_time: float,
        predicted_end_time: float,
        predicted_vram_mb: float,
        input_size: float = 0.0,
    ) -> VramReservation:
        summary = self._query_temporal_profile(
            component=component,
            config_fingerprint=config_fingerprint,
            gpu_id=gpu_id,
            input_size=input_size,
        )
        if not isinstance(summary, dict):
            return VramReservation()
        start = _finite_float(start_time)
        end = _finite_float(predicted_end_time)
        peak = _finite_float(predicted_vram_mb)
        low = summary.get("low_vram")
        start_sec = summary.get("peak_start_sec")
        duration_sec = summary.get("peak_duration_sec")
        if (
            start is None
            or end is None
            or peak is None
            or not isinstance(low, (int, float))
            or not isinstance(start_sec, (int, float))
            or not isinstance(duration_sec, (int, float))
        ):
            return VramReservation()
        if end <= start or low < 0 or peak <= low:
            return VramReservation()
        peak_start = start + start_sec
        peak_end = peak_start + duration_sec
        if not start < peak_start < peak_end < end:
            return VramReservation()
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug(
                "[vram-reservation] temporal component=%s gpu=%s low=%.1f "
                "start_sec=%.6f duration_sec=%.6f",
                component,
                gpu_id,
                low,
                start_sec,
                duration_sec,
            )
        return VramReservation(
            allow_vram_mb=low,
            peak_start_time=peak_start,
            peak_end_time=peak_end,
        )


_MODEL_TYPES: dict[str, type[VramReservationModel]] = {
    FULL_WALL: FullWallVramReservationModel,
    TEMPORAL_PEAK_INTERVAL: TemporalPeakIntervalVramReservationModel,
}


def create_vram_reservation_model(
    name: str | None,
    *,
    signal_service: Any = None,
    resource_z: float = 1.96,
) -> VramReservationModel:
    selected = str(name or FULL_WALL).strip().lower()
    model_type = _MODEL_TYPES.get(selected)
    if model_type is None:
        _LOG.warning(
            "[vram-reservation] invalid model=%r; falling back to %s",
            name,
            FULL_WALL,
        )
        model_type = _MODEL_TYPES[FULL_WALL]
    return model_type(signal_service=signal_service, resource_z=resource_z)


def allowed_vram_reservation_models() -> tuple[str, ...]:
    return tuple(sorted(_ALLOWED_MODELS))
