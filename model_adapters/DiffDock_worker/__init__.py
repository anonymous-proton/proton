"""Serving helpers for long-lived model workers."""

from modelworker.model_adapter import ModelAdapter

__all__ = ["ModelAdapter", "DiffDockAdapter"]


def __getattr__(name):
    if name == "DiffDockAdapter":
        from serving.diffdock_adapter import DiffDockAdapter

        return DiffDockAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
