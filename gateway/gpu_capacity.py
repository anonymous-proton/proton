"""GPU capacity fallback policy — single source of truth.

The gateway probes real per-GPU capacity via nvidia-smi at startup
(supervisor ResourceAdmissionTracker). The constants in this module are the
FALLBACK values used only when the probe is unavailable or fails, and for
pre-init defaults.

Historically the fallback was hard-coded to 24576 MiB (RTX 4090 default),
which is correct for the L4-8-24G cloud target but would silently UNDER-USE
an H100 80 GiB host whenever the probe path is unavailable. Deployment
surfaces (cloud launchers) can now override the fallback via:

    PROTON_GPU_VRAM_FALLBACK_MIB=<per-GPU total MiB>

Default behaviour is unchanged (24576) when the env var is unset, so local
benchmarks keep identical scheduler semantics.
"""

from __future__ import annotations

import os

_DEFAULT_GPU_VRAM_FALLBACK_MIB = 24576


def gpu_vram_fallback_mib() -> int:
    """Per-GPU total-VRAM fallback in MiB (env-overridable, default 24576)."""
    raw = (os.environ.get("PROTON_GPU_VRAM_FALLBACK_MIB") or "").strip()
    if raw:
        try:
            value = int(float(raw))
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _DEFAULT_GPU_VRAM_FALLBACK_MIB


def default_mem_safe_limit_mib() -> float:
    """Default per-GPU mem-safe admission limit (90% of fallback capacity)."""
    return 0.9 * gpu_vram_fallback_mib()
