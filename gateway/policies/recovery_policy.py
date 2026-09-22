from __future__ import annotations

import abc
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.supervisor import WorkerState, GpuResourceManager

class RecoveryPolicyRegistry:
    _REGISTRY: Dict[str, type[RecoveryPolicy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(policy_cls: type[RecoveryPolicy]):
            cls._REGISTRY[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type[RecoveryPolicy]]:
        return cls._REGISTRY.get(name)

    @classmethod
    def list_policies(cls) -> List[str]:
        return list(cls._REGISTRY.keys())


class RecoveryPolicy(abc.ABC):
    """Abstract base class for worker recovery policies."""

    @abc.abstractmethod
    def select_for_recovery(
        self, 
        candidates: List[WorkerState], 
        resource_manager: GpuResourceManager,
        all_states: List[WorkerState] = None
    ) -> List[WorkerState]:
        pass

    def _select_with_sort(
        self, 
        candidates: List[WorkerState], 
        resource_manager: GpuResourceManager,
        sort_key: Any,
        reverse: bool = False
    ) -> List[WorkerState]:
        """Helper to recover workers based on a sorted order."""
        if not candidates:
            return []

        sorted_candidates = sorted(candidates, key=sort_key, reverse=reverse)
        to_recover = []
        available_memo: Dict[str, int] = {}

        for st in sorted_candidates:
            needed_mb = getattr(st, "memory_reserved_mb", 0)
            if needed_mb <= 0:
                to_recover.append(st)
                continue

            target_gpu = None
            is_static = False
            
            if st.spec.gpus:
                target_gpu = str(st.spec.gpus[0])
                is_static = True
            elif st.assigned_gpus:
                target_gpu = str(st.assigned_gpus[0])
            
            if not is_static:
                search_order = list(resource_manager.pool)
                if target_gpu and target_gpu in search_order:
                    search_order.remove(target_gpu)
                    search_order.insert(0, target_gpu)
                
                final_gpu = None
                for gpu_id in search_order:
                    if gpu_id not in available_memo:
                        available_memo[gpu_id] = resource_manager.get_available_memory(gpu_id)
                    if available_memo[gpu_id] >= needed_mb:
                        final_gpu = gpu_id
                        break
                target_gpu = final_gpu

            if target_gpu:
                if target_gpu not in available_memo:
                    available_memo[target_gpu] = resource_manager.get_available_memory(target_gpu)
                
                avail = available_memo[target_gpu]
                if avail >= needed_mb:
                    to_recover.append(st)
                    available_memo[target_gpu] -= needed_mb
        
        return to_recover


@RecoveryPolicyRegistry.register("mru")
class MruRecoveryPolicy(RecoveryPolicy):
    """Restarts workers that were most recently used first."""

    def select_for_recovery(
        self, 
        candidates: List[WorkerState], 
        resource_manager: GpuResourceManager,
        all_states: List[WorkerState] = None
    ) -> List[WorkerState]:
        return self._select_with_sort(
            candidates, resource_manager, lambda x: x.last_used_at, reverse=True
        )


@RecoveryPolicyRegistry.register("fifo")
class FifoRecoveryPolicy(RecoveryPolicy):
    """Restarts workers in the order they were stopped (First-In, First-Out)."""

    def select_for_recovery(
        self, 
        candidates: List[WorkerState], 
        resource_manager: GpuResourceManager,
        all_states: List[WorkerState] = None
    ) -> List[WorkerState]:
        return self._select_with_sort(
            candidates, resource_manager, lambda x: x.last_used_at, reverse=False
        )


@RecoveryPolicyRegistry.register("fair_share")
class FairShareRecoveryPolicy(RecoveryPolicy):
    """Ensures all components get a fair share of recovery attempts.
    
    Prioritizes components that have fewer 'Ready' instances relative to their
    total configured instances across the cluster.
    """

    def select_for_recovery(
        self, 
        candidates: List[WorkerState], 
        resource_manager: GpuResourceManager,
        all_states: List[WorkerState] = None
    ) -> List[WorkerState]:
        if not candidates:
            return []

        all_states = all_states or []
        
        comp_total: Dict[str, int] = {}
        comp_ready: Dict[str, int] = {}
        
        for st in all_states:
            c = st.spec.component
            comp_total[c] = comp_total.get(c, 0) + 1
            if st.ready or st.lifecycle_state == "starting":
                comp_ready[c] = comp_ready.get(c, 0) + 1
        
        def _fair_share_key(st: WorkerState) -> tuple:
            c = st.spec.component
            ready = comp_ready.get(c, 0)
            total = comp_total.get(c, 1)
            ratio = float(ready) / float(total)
            return (ratio, st.last_used_at)

        return self._select_with_sort(
            candidates, resource_manager, _fair_share_key
        )
