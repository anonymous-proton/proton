"""Serving helpers for long-lived model workers."""

from modelworker.model_adapter import ModelAdapter

__all__ = ["ModelAdapter", "ProtenixAdapter"]


def __getattr__(name):
    if name == "ProtenixAdapter":
        from protenix.serving.protenix_adapter import ProtenixAdapter

        return ProtenixAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
