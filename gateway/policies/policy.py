from __future__ import annotations

import abc
import csv
import io
import shutil
from typing import List, Optional, Dict
import subprocess

class PolicyRegistry:
    _REGISTRY: Dict[str, type[GpuPolicy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(policy_cls: type[GpuPolicy]):
            cls._REGISTRY[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type[GpuPolicy]]:
        return cls._REGISTRY.get(name)

    @classmethod
    def list_policies(cls) -> List[str]:
        return list(cls._REGISTRY.keys())

class GpuPolicy(abc.ABC):
    """Abstract base class for GPU allocation policies."""

    def __init__(self, pool: List[str]):
        """
        Args:
            pool: List of GPU IDs (strings) available for allocation.
        """
        self.pool = pool

    @abc.abstractmethod
    def allocate(self, count: int = 1, pending_usage: Optional[Dict[str, int]] = None) -> List[str]:
        """
        Allocates `count` GPUs from the pool based on the policy.

        Args:
            count: Number of GPUs to allocate.
            pending_usage: Dictionary mapping GPU ID to amount of memory (MiB) 
                           that is reserved/pending but not yet visible in nvidia-smi.

        Returns:
            List of allocated GPU IDs.

        Raises:
            RuntimeError: If allocation fails (e.g. not enough resources).
        """
        pass


@PolicyRegistry.register("least_memory_used")
class LeastMemoryPolicy(GpuPolicy):
    """Allocates GPUs with the least current memory usage (including pending)."""

    def allocate(self, count: int = 1, pending_usage: Optional[Dict[str, int]] = None) -> List[str]:
        if count <= 0:
            return []
        
        pending_usage = pending_usage or {}

        if count > len(self.pool):
            raise RuntimeError(f"Requested {count} GPUs, but pool only has {len(self.pool)}")

        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            raise RuntimeError("nvidia-smi not found")

        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPU status: {e}")

        gpu_memory = []
        reader = csv.reader(io.StringIO(result.stdout))
        for row in reader:
            if not row:
                continue
            idx = row[0].strip()
            mem_used = int(row[1].strip())
            
            if idx in self.pool:
                pending = pending_usage.get(idx, 0)
                gpu_memory.append((idx, mem_used + pending))

        if len(gpu_memory) < count:
            raise RuntimeError(
                f"Requested {count} GPUs from pool {self.pool}, but only found {len(gpu_memory)} available via nvidia-smi"
            )

        gpu_memory.sort(key=lambda x: x[1])

        allocated = [x[0] for x in gpu_memory[:count]]
        return allocated


@PolicyRegistry.register("on_demand")
class OnDemandPolicy(GpuPolicy):
    """Allocates from pool using supervisor-side pending usage only.

    This policy intentionally avoids querying runtime GPU memory from nvidia-smi.
    Worker lifecycle (lazy spawn / startup readiness) is handled by the supervisor.
    This class only determines which GPU IDs are chosen when allocation is needed.
    """

    def allocate(self, count: int = 1, pending_usage: Optional[Dict[str, int]] = None) -> List[str]:
        if count <= 0:
            return []
        if count > len(self.pool):
            raise RuntimeError(f"Requested {count} GPUs, but pool only has {len(self.pool)}")

        pending = pending_usage or {}
        ranked = [
            (gpu_id, int(pending.get(gpu_id, 0)), idx)
            for idx, gpu_id in enumerate(self.pool)
        ]
        ranked.sort(key=lambda item: (item[1], item[2]))
        return [gpu_id for gpu_id, _, _ in ranked[:count]]
@PolicyRegistry.register("exclusive")
class ExclusivePolicy(GpuPolicy):
    """Allocates GPUs only if they are completely empty (no other containers)."""

    def allocate(self, count: int = 1, pending_usage: Optional[Dict[str, int]] = None) -> List[str]:
        if count <= 0:
            return []
        
        pending_usage = pending_usage or {}

        if count > len(self.pool):
            raise RuntimeError(f"Requested {count} GPUs, but pool only has {len(self.pool)}")

        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            raise RuntimeError("nvidia-smi not found")

        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to query GPU status: {e}")

        gpu_stats = []
        reader = csv.reader(io.StringIO(result.stdout))
        for row in reader:
            if not row:
                continue
            idx = row[0].strip()
            mem_used = int(row[1].strip())
            util = int(row[2].strip())
            
            if idx in self.pool:
                pending = pending_usage.get(idx, 0)
                gpu_stats.append((idx, mem_used + pending, util))

        empty_gpus = [idx for idx, mem, util in gpu_stats if mem < 100 and util == 0]

        if len(empty_gpus) < count:
            raise RuntimeError(
                f"Exclusive allocation failed: Requested {count} empty GPUs, but only {len(empty_gpus)} are empty "
                f"(Total pool: {self.pool}, Stats: {gpu_stats})"
            )

        return empty_gpus[:count]
