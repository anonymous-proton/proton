from __future__ import annotations

import abc
import time
from typing import List, Optional, Dict, Any, Callable, Tuple

class EvictionPolicyRegistry:
    _REGISTRY: Dict[str, type[EvictionPolicy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(policy_cls: type[EvictionPolicy]):
            cls._REGISTRY[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type[EvictionPolicy]]:
        return cls._REGISTRY.get(name)

    @classmethod
    def list_policies(cls) -> List[str]:
        return list(cls._REGISTRY.keys())


class EvictionPolicy(abc.ABC):
    """Abstract base class for worker eviction policies."""

    @abc.abstractmethod
    def select_for_eviction(
        self, 
        candidates: List[Any], 
        needed_mb: int,
        freeable_fn: Optional[Callable[[Any], int]] = None,
    ) -> List[Any]:
        """
        Selects workers to evict from the list of candidates.
        
        Args:
            candidates: List of WorkerState objects eligible for eviction.
            needed_mb: Total memory (MiB) needed to be freed.
            
        Returns:
            List of WorkerState objects to stop.
        """
        pass


@EvictionPolicyRegistry.register("lru")
class LruEvictionPolicy(EvictionPolicy):
    """Evicts workers that haven't been used for the longest time."""

    def select_for_eviction(
        self, 
        candidates: List[Any], 
        needed_mb: int,
        freeable_fn: Optional[Callable[[Any], int]] = None,
    ) -> List[Any]:
        if not candidates or needed_mb <= 0:
            return []

        freeable = freeable_fn or (lambda st: getattr(st, "memory_reserved_mb", 0))
        sorted_candidates = sorted(candidates, key=lambda x: x.last_used_at)
        
        to_evict = []
        freed_mb = 0
        for st in sorted_candidates:
            to_evict.append(st)
            freed_mb += int(freeable(st) or 0)
            if freed_mb >= needed_mb:
                break
        
        return to_evict


@EvictionPolicyRegistry.register("priority_preemption")
class PriorityEvictionPolicy(EvictionPolicy):
    """Evicts workers with the lowest priority first.
    
    Ties are broken by LRU.
    """

    def select_for_eviction(
        self, 
        candidates: List[Any], 
        needed_mb: int,
        freeable_fn: Optional[Callable[[Any], int]] = None,
    ) -> List[Any]:
        if not candidates or needed_mb <= 0:
            return []

        freeable = freeable_fn or (lambda st: getattr(st, "memory_reserved_mb", 0))
        def _evict_key(st: Any) -> tuple:
            priority = getattr(st.spec, "priority", 100) or 100
            return (priority, st.last_used_at)

        sorted_candidates = sorted(candidates, key=_evict_key)
        
        to_evict = []
        freed_mb = 0
        for st in sorted_candidates:
            to_evict.append(st)
            freed_mb += int(freeable(st) or 0)
            if freed_mb >= needed_mb:
                break
        
        return to_evict


@EvictionPolicyRegistry.register("cost_sensitive")
class CostSensitiveEvictionPolicy(EvictionPolicy):
    """Protects workers with high memory requirements (high cold-start cost).
    
    Evicts workers with SMALLER memory requirements or LOW priority first.
    Formula: priority / memory (higher is better protected)
    Or simpler: Sort by (priority, memory_required)
    """

    def select_for_eviction(
        self, 
        candidates: List[Any], 
        needed_mb: int,
        freeable_fn: Optional[Callable[[Any], int]] = None,
    ) -> List[Any]:
        if not candidates or needed_mb <= 0:
            return []

        freeable = freeable_fn or (lambda st: getattr(st, "memory_reserved_mb", 0))
        def _cost_key(st: Any) -> tuple:
            priority = getattr(st.spec, "priority", 100) or 100
            mem = int(freeable(st) or 0) or 1
            return (priority, mem, st.last_used_at)

        sorted_candidates = sorted(candidates, key=_cost_key)

        to_evict = []
        freed_mb = 0
        for st in sorted_candidates:
            to_evict.append(st)
            freed_mb += int(freeable(st) or 0)
            if freed_mb >= needed_mb:
                break

        return to_evict


@EvictionPolicyRegistry.register("campaign_aware")
class CampaignAwareEvictionPolicy(EvictionPolicy):
    """Multi-phase eviction: backfill first, then selective non-backfill.

    Phase 1: Evict backfill workers (lowest eviction cost first).
    Phase 2: If backfill is insufficient, check whether a single large
             non-backfill worker is cheaper to evict than all backfill
             workers combined. If so, evict only that worker.
             Otherwise, evict all backfill + cheapest non-backfill.

    Eviction cost = elapsed_time + init_estimate + predicted_latency
    (lower cost = cheaper to evict).

    See <docs> 
    """

    def __init__(
        self,
        init_fn: Optional[Callable[[str, str], Tuple[float, float]]] = None,
        primary_campaign_id: Optional[str] = None,
        grace_period_sec: float = 5.0,
    ) -> None:
        self._init_fn = init_fn
        self._primary_campaign_id = primary_campaign_id
        self._grace_period_sec = grace_period_sec

    def set_primary_campaign(self, campaign_id: Optional[str]) -> None:
        """Update the primary campaign ID (called by CampaignScheduler)."""
        self._primary_campaign_id = campaign_id

    def _eviction_cost(self, st: Any) -> float:
        """Compute eviction cost for a worker. Lower = cheaper to evict."""
        elapsed = time.time() - st.last_used_at if st.last_used_at > 0 else 0.0
        init_latency = 10.0
        if self._init_fn:
            try:
                comp = getattr(st.spec, "component", "")
                gpus = getattr(st, "assigned_gpus", [])
                gpu_id = str(gpus[0]) if gpus else "0"
                mu, _ = self._init_fn(comp, gpu_id)
                if mu > 0:
                    init_latency = mu
            except Exception:
                pass
        return elapsed + init_latency

    def _is_backfill(self, st: Any) -> bool:
        """Check if a worker is serving backfill tasks."""
        if not self._primary_campaign_id:
            return False
        worker_campaigns = getattr(st, "active_campaign_ids", None)
        if worker_campaigns:
            return self._primary_campaign_id not in worker_campaigns
        return False

    def select_for_eviction(
        self,
        candidates: List[Any],
        needed_mb: int,
        freeable_fn: Optional[Callable[[Any], int]] = None,
    ) -> List[Any]:
        if not candidates or needed_mb <= 0:
            return []

        freeable = freeable_fn or (lambda st: getattr(st, "memory_reserved_mb", 0))
        now = time.time()
        eligible = [
            w for w in candidates
            if (now - getattr(w, "last_ready_at", 0)) >= self._grace_period_sec
            and (now - getattr(w, "last_used_at", 0)) >= self._grace_period_sec
        ]
        if not eligible:
            eligible = candidates

        backfill = [w for w in eligible if self._is_backfill(w)]
        non_backfill = [w for w in eligible if not self._is_backfill(w)]

        sorted_bf = sorted(backfill, key=self._eviction_cost)
        to_evict: List[Any] = []
        freed = 0
        for w in sorted_bf:
            to_evict.append(w)
            freed += int(freeable(w) or 0)
            if freed >= needed_mb:
                return to_evict

        sorted_nbf = sorted(non_backfill, key=self._eviction_cost)

        if sorted_bf:
            all_bf_cost = sum(self._eviction_cost(b) for b in sorted_bf)
            for w in sorted_nbf:
                w_mem = int(freeable(w) or 0)
                if w_mem >= needed_mb and self._eviction_cost(w) < all_bf_cost:
                    return [w]

        remaining = needed_mb - freed
        for w in sorted_nbf:
            to_evict.append(w)
            remaining -= int(freeable(w) or 0)
            if remaining <= 0:
                return to_evict

        return to_evict
