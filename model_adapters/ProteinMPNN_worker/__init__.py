"""Serving helpers for long-lived model workers."""

from modelworker.model_adapter import ModelAdapter

__all__ = ["ModelAdapter", "ProteinMPNNAdapter"]


def __getattr__(name):
    if name == "ProteinMPNNAdapter":
        from serving.proteinmpnn_adapter import ProteinMPNNAdapter

        return ProteinMPNNAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
