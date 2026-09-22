"""Interference-aware initialization (cold-start) latency tracking.

Measures the time from container start to first successful inference for
each (component, gpu_id) pair.  Uses the same ``GPEstimate`` class and
three-case interference correction pipeline as inference latency
(see <docs> ).

Since initialization latency is **input-size invariant** (model init doesn't
depend on data), all observations use ``x = 0``.  The GP effectively
degenerates to a scalar estimator with heteroscedastic noise — but using
``GPEstimate`` directly gives us interference correction, drift detection,
and GPU-specific → cross-GPU fallback for free.

See <docs>  for design rationale.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .resource_profile import GPEstimate

_LOG = logging.getLogger(__name__)

_DEFAULT_INIT_SEC = 10.0
_DEFAULT_INIT_VAR = 1.0

_CROSS_GPU = "*"


class InitProfile:
    """Per-(component, gpu_id) initialization latency with interference correction.

    Each (component, gpu_id) pair gets its own ``GPEstimate`` with all
    observations at ``x = 0`` (initialization is input-size invariant).

    Three-case R_t computation (same as inference latency):
      - Case 1 (solo): R = None → default σ²_solo
      - Case 2 (known pair): corrected via delta method
      - Case 3 (unknown pair): inflated noise
    """

    def __init__(
        self,
        interference_registry: Any = None,
        *,
        default_init_sec: float = _DEFAULT_INIT_SEC,
        default_init_var: float = _DEFAULT_INIT_VAR,
    ) -> None:
        self._interference_registry = interference_registry
        self._gps: Dict[Tuple[str, str], GPEstimate] = {}
        self._drift_callbacks: List[Any] = []
        self._default_init_sec = float(default_init_sec)
        self._default_init_var = float(default_init_var)

    def _key(self, component: str, gpu_id: str) -> Tuple[str, str]:
        return (str(component).strip().lower(), str(gpu_id).strip())

    def _get_or_create(self, component: str, gpu_id: str) -> GPEstimate:
        key = self._key(component, gpu_id)
        gp = self._gps.get(key)
        if gp is None:
            gp = GPEstimate(sigma2_f=100.0, lengthscale=50.0, n_max=50, max_age=7200.0)
            self._gps[key] = gp
        return gp


    def record(
        self,
        component: str,
        gpu_id: str,
        init_sec: float,
        co_located: Optional[List[str]] = None,
    ) -> None:
        """Record a initialization latency observation.

        Args:
            component: The component that was cold-started.
            gpu_id: The GPU where the cold start happened.
            init_sec: Observed time from container start to first inference.
            co_located: List of component names on the same GPU during cold start.
                        Empty or None means solo cold start.
        """
        if init_sec <= 0:
            return

        comp = str(component).strip().lower()
        gid = str(gpu_id).strip()
        co_located = co_located or []

        corrected, R = self._correct_observation(comp, init_sec, co_located)

        gp = self._get_or_create(comp, gid)
        effective_R = R if R is not None else max(gp.sigma2_solo if gp.n > 0 else (0.1 * corrected) ** 2, (0.1 * abs(corrected)) ** 2)
        gp.update(x=0.0, y=corrected, R=effective_R)

        cross_gp = self._get_or_create(comp, _CROSS_GPU)
        cross_effective_R = R if R is not None else max(cross_gp.sigma2_solo if cross_gp.n > 0 else (0.1 * corrected) ** 2, (0.1 * abs(corrected)) ** 2)
        cross_gp.update(x=0.0, y=corrected, R=cross_effective_R)

        _LOG.debug(
            "[init] recorded %s on GPU %s: %.2fs (corrected=%.2fs, R=%s, co=%s, n=%d)",
            comp, gid, init_sec, corrected,
            f"{R:.4f}" if R is not None else "solo",
            co_located, gp.n,
        )

    def _correct_observation(
        self,
        component: str,
        raw_init: float,
        co_located: List[str],
    ) -> Tuple[float, Optional[float]]:
        """Apply three-case interference correction.

        Returns (corrected_value, R_noise_variance).
        R = None means solo observation (default σ²_solo in GP).
        """
        if not co_located:
            return raw_init, None

        if self._interference_registry is None:
            return raw_init, max(raw_init * 0.5, 10.0) ** 2

        all_known = True
        total_delta = 0.0
        for other in co_located:
            delta = self._interference_registry.get_pairwise_slowdown(component, other)
            if delta is None:
                all_known = False
                break
            total_delta += delta

        if all_known and total_delta > 0:
            corrected = raw_init / (1.0 + total_delta)
            var_delta = (0.1 * total_delta) ** 2
            R = (raw_init / (1.0 + total_delta) ** 2) ** 2 * var_delta
            R = max(R, 0.01)
            return corrected, R
        else:
            gp = self._gps.get(self._key(component, _CROSS_GPU))
            sigma2_solo = gp.sigma2_solo if gp and gp.n > 0 else (raw_init * 0.3) ** 2
            sigma2_bias = max(raw_init * 0.5, 10.0) ** 2
            return raw_init, sigma2_solo + sigma2_bias


    def predict(self, component: str, gpu_id: str) -> Tuple[float, float]:
        """Returns (μ*_init, σ²*_init) — estimated solo initialization time.

        Falls back: GPU-specific → cross-GPU pool → conservative default.
        """
        comp = str(component).strip().lower()
        gid = str(gpu_id).strip()

        gp = self._gps.get(self._key(comp, gid))
        if gp and gp.n > 0:
            return gp.predict(0.0)

        cross = self._gps.get(self._key(comp, _CROSS_GPU))
        if cross and cross.n > 0:
            return cross.predict(0.0)

        return (self._default_init_sec, self._default_init_var)

    def confidence(self, component: str, gpu_id: str) -> float:
        """Return confidence in [0, 1] for the initialization estimate."""
        comp = str(component).strip().lower()
        gid = str(gpu_id).strip()

        gp = self._gps.get(self._key(comp, gid))
        if gp and gp.n > 0:
            return gp.confidence(0.0)

        cross = self._gps.get(self._key(comp, _CROSS_GPU))
        if cross and cross.n > 0:
            return cross.confidence(0.0)

        return 0.0


    def observation_count(self, component: str, gpu_id: Optional[str] = None) -> int:
        """Number of initialization observations for a component."""
        comp = str(component).strip().lower()
        if gpu_id:
            gp = self._gps.get(self._key(comp, str(gpu_id).strip()))
            return gp.n if gp else 0
        cross = self._gps.get(self._key(comp, _CROSS_GPU))
        return cross.n if cross else 0

    def all_components(self) -> List[str]:
        """Return all components with at least one initialization observation."""
        seen: set = set()
        for (comp, gid), gp in self._gps.items():
            if gp.n > 0 and gid != _CROSS_GPU:
                seen.add(comp)
        return sorted(seen)


    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for (comp, gid), gp in sorted(self._gps.items()):
            if gp.n == 0:
                continue
            mu, var = gp.predict(0.0)
            entry = {
                "n": gp.n,
                "mu_sec": round(mu, 3),
                "sigma_sec": round(var ** 0.5, 3),
                "confidence": round(gp.confidence(0.0), 3),
            }
            result.setdefault(comp, {})[gid] = entry
        return result

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema": "init_profile_v1",
            "default_init_sec": float(self._default_init_sec),
            "default_init_var": float(self._default_init_var),
            "gps": {
                f"{comp}|{gid}": gp.export_state()
                for (comp, gid), gp in sorted(self._gps.items())
            },
        }

    def import_state(self, data: Mapping[str, Any]) -> None:
        self._default_init_sec = float(
            data.get("default_init_sec", self._default_init_sec)
            or self._default_init_sec
        )
        self._default_init_var = float(
            data.get("default_init_var", self._default_init_var)
            or self._default_init_var
        )
        self._gps = {}
        gps = data.get("gps") or {}
        for raw_key, payload in (gps.items() if isinstance(gps, Mapping) else []):
            comp, _, gid = str(raw_key or "").partition("|")
            if not comp or not gid or not isinstance(payload, Mapping):
                continue
            gp = GPEstimate(sigma2_f=100.0, lengthscale=50.0, n_max=50, max_age=7200.0)
            gp.import_state(payload)
            self._gps[self._key(comp, gid)] = gp
