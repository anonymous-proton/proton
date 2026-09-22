"""PyTorch-specific concurrency utilities for the worker engine.

Non-PyTorch adapters (Vina-GPU, MMSeqs2, etc.) do not use this module.
If JAX/TF/ONNX adapters are added in the future, they should implement
their own framework-specific copy and GC logic.
"""

from __future__ import annotations

import copy
import gc
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        torch = None


_last_gc_time = 0.0
_gc_lock = threading.Lock()
_STICKY_CUDA_ERROR_SIGNATURES = (
    (
        "mps_rpc_failure",
        "remote procedural call between the mps server and the mps client failed",
    ),
    ("illegal_memory_access", "illegal memory access"),
    ("device_side_assert", "device side assert"),
    ("misaligned_address", "misaligned address"),
    ("unspecified_launch_failure", "unspecified launch failure"),
)


def classify_sticky_cuda_error(error: BaseException | str) -> str | None:
    """Return a stable class for CUDA errors that poison a process context."""
    message = " ".join(str(error).casefold().replace("-", " ").split())
    for error_class, signature in _STICKY_CUDA_ERROR_SIGNATURES:
        if signature in message:
            return error_class
    return None


def maybe_gc(interval_s: float = 5.0) -> None:
    """Debounced ``gc.collect()`` + ``torch.cuda.empty_cache()``.

    Under concurrent execution, multiple threads calling gc/empty_cache
    simultaneously causes process-wide pauses.  This helper ensures the
    cleanup runs at most once every *interval_s* seconds.
    """
    global _last_gc_time
    now = time.time()
    if now - _last_gc_time < interval_s:
        return
    with _gc_lock:
        if time.time() - _last_gc_time < interval_s:
            return
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _last_gc_time = time.time()


def safe_model_copy(model: torch.nn.Module) -> torch.nn.Module:
    """Shallow-copy a PyTorch model recursively, sharing parameters but cloning buffers.

    ``copy.copy(model)`` shares **both** parameters and submodules with the original.
    This recursive implementation clones the module hierarchy and replicates the
    ``_modules`` and ``_buffers`` dictionaries at each level, ensuring that buffers
    (e.g., BatchNorm running stats) are isolated per request to avoid corruption under
    concurrent forward passes, while parameters (weights) remain shared.
    """
    if torch is None:
        raise RuntimeError("safe_model_copy requires PyTorch")

    memo = {}

    def memoized_copy(m: torch.nn.Module) -> torch.nn.Module:
        if m in memo:
            return memo[m]

        copied = copy.copy(m)
        memo[m] = copied

        if hasattr(copied, "_modules"):
            copied._modules = copy.copy(copied._modules)
            for name, submodule in copied._modules.items():
                if submodule is not None:
                    copied._modules[name] = memoized_copy(submodule)

        if hasattr(copied, "_buffers"):
            copied._buffers = copy.copy(copied._buffers)
            for name, buf in copied._buffers.items():
                if buf is not None:
                    copied._buffers[name] = buf.clone()

        return copied

    return memoized_copy(model)
