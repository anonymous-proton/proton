"""Serving helpers for long-lived model workers."""

from modelworker.model_adapter import ModelAdapter

__all__ = ["ModelAdapter", "RFDiffusionAdapter"]


def __getattr__(name):
    if name == "RFDiffusionAdapter":
        from rfdiffusion.serving.rfdiffusion_adapter import RFDiffusionAdapter

        return RFDiffusionAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
