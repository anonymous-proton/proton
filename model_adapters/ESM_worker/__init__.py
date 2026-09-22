"""Serving helpers for long-lived model workers."""

from modelworker.model_adapter import ModelAdapter

__all__ = ["ModelAdapter", "ESMAdapter"]


def __getattr__(name):
    if name == "ESMAdapter":
        from serving.esm_adapter import ESMAdapter

        return ESMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
