"""Pluggable OOM Guard policies for handling unknown VRAM requirements."""

import abc
from typing import Dict, Optional


class WeightOOMPolicy(abc.ABC):
    """Policy for determining VRAM reservation when a component's model weight footprint is unknown."""
    _REGISTRY: Dict[str, type['WeightOOMPolicy']] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(policy_cls: type['WeightOOMPolicy']):
            cls._REGISTRY[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type['WeightOOMPolicy']]:
        return cls._REGISTRY.get(name)

    @abc.abstractmethod
    def get_reservation_mb(self, component: str, total_gpu_vram_mb: int) -> int:
        pass


@WeightOOMPolicy.register("pessimistic")
class PessimisticWeightGuard(WeightOOMPolicy):
    """Monopolizes the GPU by reserving almost all available VRAM for the first run."""
    def get_reservation_mb(self, component: str, total_gpu_vram_mb: int) -> int:
        return int(total_gpu_vram_mb * 0.9)


@WeightOOMPolicy.register("optimistic")
class OptimisticWeightGuard(WeightOOMPolicy):
    """Assumes a small baseline and packs normally, relying on strict monitoring to catch OOMs."""
    def __init__(self, baseline_mb: int = 4096):
        self._baseline_mb = baseline_mb

    def get_reservation_mb(self, component: str, total_gpu_vram_mb: int) -> int:
        return self._baseline_mb
