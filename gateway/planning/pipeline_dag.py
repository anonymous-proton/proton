"""Pipeline DAG model built from Nextflow session DAG topology.

The DAG captures the runtime topology of the proton pipeline — process
dependencies and per-component fan-out factors.  The topology is extracted by
the nf-gw plugin from Nextflow's ``session.dag`` and sent to the gateway as
part of each task submission payload.

Two sources of fan-out information:

1. **``ext.output_sample_count``** (runtime, per-task): Declared in
   ``nextflow.config`` from pipeline params.  Highest priority.

2. **``workers.yaml workload.fan_out_args``** (argv extraction, per-component):
   Extracted from argv at dispatch time by ``component_features.extract_fan_out``.
   Fallback when ``ext.output_sample_count`` is not set.

The PipelineDAG is built incrementally:

- On the first task submission that includes ``dag_topology``, the gateway
  builds the process-level graph (process names + edges).
- Each task submission also carries ``process_name`` and ``component``,
  allowing the gateway to accumulate a ``process_name → component`` mapping.
- The component-level DAG is derived by projecting the process-level graph
  through this mapping.

Typical usage::

    from gateway.pipeline_dag import PipelineDAG

    dag = PipelineDAG()
    dag.register_topology(dag_topology_from_nextflow)
    dag.register_process_component("RUN_RFDIFFUSION", "rfdiffusion")

    node = dag.get_component_node("rfdiffusion")
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any, Dict, List, Optional, Set

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class ComponentDAGNode:
    """A component in the resolved component-level DAG.

    ``join_type`` (fix): fan-out synchronization semantics.
    - ``"independent"`` (default): each sibling's output flows through its
      own downstream chain independently.  b-rank uses ``Φ⁻¹(α)`` per
      sibling (N=1).  Matches Nextflow channel semantics (``.transpose()``).
    - ``"barrier"``: explicit collector/join node (e.g., ``collect()``,
      ``groupTuple()``).  N siblings must all complete before the collector
      fires.  b-rank for the collector uses ``Φ⁻¹(α^{1/N})`` (max of N
      i.i.d. normals, David-Nagaraja 2003).

    Only nodes explicitly tagged ``barrier`` apply the max quantile.  The
    default independent mode avoids over-padding that would inflate b-rank
    and distort HEFT priority (PERT merge-bias analogue — Fulkerson 1962,
    Clark 1961, MacCrimmon-Ryavec 1964).
    """

    component: str
    upstream_components: List[str]
    downstream_components: List[str]
    join_type: str = "independent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "upstream_components": list(self.upstream_components),
            "downstream_components": list(self.downstream_components),
            "join_type": self.join_type,
        }


@dataclasses.dataclass
class DAGContext:
    """Snapshot of the DAG state for a single campaign.

    The Planner operates **per-campaign** (one inference graph = one campaign).
    Each campaign may have different fan-out values because pipeline params
    differ across runs (e.g., ``num_designs=3`` in one campaign, ``10`` in another).

    The DAG topology (component ordering, edges) is shared across campaigns
    — it's the same Nextflow pipeline.  Fan-out is campaign-specific.

    ``fan_out_map`` uses ``None`` to distinguish "no hint provided" from
    "known to be 1".  The Planner must treat ``None`` as *unknown* rather
    than assuming any default.
    """

    topology_registered: bool
    campaign_id: Optional[str]
    component_order: List[str]
    fan_out_map: Dict[str, Optional[int]]
    upstream_map: Dict[str, List[str]]
    downstream_map: Dict[str, List[str]]
    join_type_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    is_component_coverage_complete: bool = True
    unresolved_process_count: int = 0

    def join_type(self, component: str) -> str:
        """Return join_type for a component ("independent" default)."""
        return self.join_type_map.get(component, "independent")

    @property
    def known_fan_out_components(self) -> List[str]:
        """Components whose fan-out is known (hint was provided)."""
        return [c for c, v in self.fan_out_map.items() if v is not None]

    @property
    def unknown_fan_out_components(self) -> List[str]:
        """Components whose fan-out is unknown (no hint provided)."""
        return [c for c, v in self.fan_out_map.items() if v is None]

    def cumulative_fan_out_from(self, component: str) -> Optional[int]:
        """Product of fan-outs along the longest downstream path.

        Returns ``None`` if any component on the path has unknown fan-out,
        since the total cannot be computed reliably.
        """
        return self._cumulative(component, set())

    def _cumulative(self, comp: str, visited: set) -> Optional[int]:
        if comp in visited:
            return 1
        visited.add(comp)
        my_fan = self.fan_out_map.get(comp)
        if my_fan is None:
            return None
        downstream = self.downstream_map.get(comp, [])
        if not downstream:
            return my_fan
        child_values = [self._cumulative(d, visited) for d in downstream]
        if any(v is None for v in child_values):
            return None
        return my_fan * max(v for v in child_values if v is not None)


_DEFAULT_CAMPAIGN = "__default__"


class PipelineDAG:
    """Directed acyclic graph built from Nextflow's runtime DAG.

    Thread-safe: topology and process→component mappings can be registered
    concurrently from multiple task submission handlers.

    **Topology** (graph structure) is shared across all campaigns — it's the
    same Nextflow pipeline regardless of which params were used.

    **Fan-out** is tracked **per-campaign** because different campaigns may run
    with different params (e.g., ``num_designs=3`` in one run, ``10`` in another).
    Within each campaign, fan-out per component is ``Optional[int]``:
    ``None`` means "no hint was provided" — the Planner must treat it as
    unknown rather than assuming 1.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process_vertices: List[str] = []
        self._process_edges: List[Dict[str, str]] = []
        self._topology_registered: bool = False
        self._process_to_component: Dict[str, str] = {}
        self._eager_component_mapping_complete: bool = False
        self._process_ext_components: Dict[str, str] = {}
        self._campaign_fan_out: Dict[str, Dict[str, Optional[int]]] = {}
        self._component_join_types: Dict[str, str] = {}
        self._component_nodes: Optional[Dict[str, ComponentDAGNode]] = None


    def register_topology(self, dag_topology: Dict[str, Any]) -> None:
        """Register the process-level DAG topology from Nextflow.

        Called on each task submission.  Only the first non-empty topology
        is accepted; subsequent calls are no-ops (the DAG is static per
        Nextflow session).

        Plan fix — eager process_name → component mapping
        sourced **authoritatively** from Nextflow's ``ext.component``
        declaration.  Every process in ``nextflow.config`` declares
        ``ext.component = 'vina_gpu'`` (or equivalent), and the
        ``nf-gw`` plugin forwards that value in each vertex's
        ``component`` field.  Gateway **trusts** the declaration and
        refuses to fall back to heuristic substring matching:
        process-name patterns are fragile (opaque names, renames, name
        collisions across workers) and would silently mis-route
        dispatches or pre-init warms.  Processes whose vertex lacks a
        ``component`` field remain unresolved → surface via
        ``is_component_coverage_complete=False`` → partial-DAG guard
        (fix) forces WSJF to FIFO until the first task's
        ``register_process_component`` fills the mapping explicitly.

        Explicit ``register_process_component`` (task-body override)
        still takes precedence and can correct a mis-declared vertex.
        """
        if not dag_topology or not isinstance(dag_topology, dict):
            return
        vertices = dag_topology.get("vertices")
        edges = dag_topology.get("edges")
        if not vertices:
            return

        with self._lock:
            if self._topology_registered:
                return
            raw_vertices = [
                v for v in (vertices or [])
                if isinstance(v, dict) and v.get("name")
            ]
            self._process_vertices = [str(v["name"]) for v in raw_vertices]
            self._process_edges = [
                {"from": str(e["from"]), "to": str(e["to"])}
                for e in (edges or [])
                if isinstance(e, dict) and e.get("from") and e.get("to")
            ]
            self._topology_registered = True
            self._component_nodes = None

            authoritative = 0
            for v in raw_vertices:
                comp = v.get("component")
                if not comp:
                    continue
                pn = str(v["name"]).strip()
                c = str(comp).strip().lower()
                if not pn or not c:
                    continue
                if self._process_to_component.get(pn) != c:
                    self._process_to_component[pn] = c
                authoritative += 1
            self._eager_component_mapping_complete = authoritative > 0
            _LOG.info(
                "Pipeline DAG topology registered: %d processes, %d edges "
                "(%d authoritative ext.component mappings via Config Model; "
                "remaining %d vertices %s)",
                len(self._process_vertices),
                len(self._process_edges),
                authoritative,
                len(self._process_vertices) - authoritative,
                "will resolve lazily at first task submission if declared "
                "inside a .nf process block" if authoritative > 0 else
                "— NO eager mappings arrived; pipeline must declare "
                "``ext.component`` in nextflow.config for WSJF to engage "
                "(see <docs>)",
            )

    def register_process_component(self, process_name: str, component: str) -> None:
        """Record a process_name → component mapping from a task submission."""
        pn = str(process_name).strip()
        comp = str(component).strip().lower()
        if not pn or not comp:
            return
        with self._lock:
            if self._process_to_component.get(pn) != comp:
                self._process_to_component[pn] = comp
                self._component_nodes = None

    def register_join_type(self, component: str, join_type: str) -> None:
        """Record the fan-out semantics of *component* (Plan fix).

        ``join_type`` must be ``"independent"`` (each sibling proceeds on
        its own downstream chain — Nextflow's streaming default) or
        ``"barrier"`` (N siblings all required before downstream proceeds —
        Nextflow's ``collect()``/``groupTuple()`` operators).

        D1 b-rank uses this tag to decide whether to apply the
        ``Φ⁻¹(α^{1/N})`` order-statistics quantile.  Independent fan-out
        uses the single-task ``Φ⁻¹(α)`` regardless of N.
        """
        comp = str(component).strip().lower()
        jt = str(join_type).strip().lower()
        if comp not in ("",) and jt in ("independent", "barrier"):
            with self._lock:
                if self._component_join_types.get(comp) != jt:
                    self._component_join_types[comp] = jt
                    self._component_nodes = None

    def register_fan_out(
        self,
        component: str,
        output_sample_count: int,
        campaign_id: Optional[str] = None,
    ) -> None:
        """Record a known fan-out for a component within a campaign.

        Called when a task submission provides ``output_sample_count``
        (from ``ext.output_sample_count`` or ``fan_out_args`` extraction).
        Subsequent calls for the same component+campaign update the value.

        Components that never call this remain ``None`` (unknown) in the
        campaign's fan-out map — the Planner can distinguish "known 1"
        from "unknown".

        Different campaigns may have different fan-out values for the same
        component (e.g., ``num_designs=3`` in campaign A, ``10`` in campaign B).
        """
        comp = str(component).strip().lower()
        cid = str(campaign_id or "").strip() or _DEFAULT_CAMPAIGN
        if not comp or output_sample_count is None or output_sample_count < 1:
            return
        with self._lock:
            campaign_map = self._campaign_fan_out.setdefault(cid, {})
            campaign_map[comp] = output_sample_count


    def get_component_node(self, component: str) -> Optional[ComponentDAGNode]:
        """Return the DAG node for a component, or None if not yet resolved."""
        nodes = self._resolve_component_dag()
        return nodes.get(str(component).strip().lower())

    @property
    def component_nodes(self) -> Dict[str, ComponentDAGNode]:
        return dict(self._resolve_component_dag())

    @property
    def is_topology_registered(self) -> bool:
        return self._topology_registered

    @property
    def is_eager_component_mapping_complete(self) -> bool:
        """True iff ``register_topology`` received an authoritative
        ``component`` field on at least one vertex (Plan 
        fix / W1).  Gates WSJF ``refresh_remaining_est``: True
        ⇒ gateway-bound process set is known at topology time and
        per-component strict GP-observation check governs; False ⇒
        pipeline did not follow the Config Model authoring contract
        (``ext.component`` must be declared inside ``nextflow.config``)
        and WSJF stays in FIFO fallback per """
        with self._lock:
            return self._eager_component_mapping_complete

    @property
    def process_to_component(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._process_to_component)

    def downstream_components(self, component: str) -> List[str]:
        """Return components that depend on *component*."""
        node = self.get_component_node(component)
        return list(node.downstream_components) if node else []

    def upstream_components(self, component: str) -> List[str]:
        """Return components that *component* depends on."""
        node = self.get_component_node(component)
        return list(node.upstream_components) if node else []

    def cumulative_fan_in_to(
        self,
        component: str,
        fan_out_resolver,
    ) -> Optional[int]:
        """Plan fix (P-3) — expected total task count of
        ``component`` when the pipeline runs through its DAG chain.

        Recursive definition::

            total_instances(s) = 1                                   if no upstream
                               = 1                                   if join_type(s) == "barrier"
                               = total_instances(parent) × resolver(parent)   otherwise

        ``resolver(comp)`` is a callable that returns the per-task fan-out
        (output sample count) for *comp* — e.g.
        ``CampaignScheduler.effective_fan_out_with_global``.  Returns
        ``None`` when any parent fan-out is unavailable so callers can
        gracefully fall back to local heuristics.

        Single-parent chain assumption.  Multi-parent merge stages pick
        the first listed upstream component; tighter multi-parent
        support is deferred.
        """
        return self._cumulative_in(component, set(), fan_out_resolver)

    def _cumulative_in(self, comp: str, visited: set, resolver) -> Optional[int]:
        key = str(comp).strip().lower()
        if key in visited:
            return 1
        visited.add(key)
        node = self.get_component_node(key)
        if node is None or not node.upstream_components:
            return 1
        if self.join_type(key) == "barrier":
            return 1
        parent = node.upstream_components[0]
        parent_total = self._cumulative_in(parent, visited, resolver)
        if parent_total is None:
            return None
        try:
            parent_fan = resolver(parent)
        except Exception:
            return None
        if parent_fan is None or parent_fan <= 0:
            return None
        return int(parent_total * int(round(float(parent_fan))))

    def join_type(self, component: str) -> str:
        """Plan — delegate to the resolved-topology join-type map so
        ``cumulative_fan_in_to`` can treat barrier joins as cardinality 1.
        Callers that hit a component before topology resolution get the
        ``"independent"`` default (matches ``DAGContext.join_type``).
        """
        nodes = self._resolve_component_dag()
        node = nodes.get(str(component).strip().lower())
        if node is None:
            return "independent"
        return getattr(node, "join_type", "independent") or "independent"

    def topological_order(self) -> List[str]:
        """Return components in dependency order (Kahn's algorithm)."""
        nodes = self._resolve_component_dag()
        in_degree: Dict[str, int] = {c: 0 for c in nodes}
        for comp, node in nodes.items():
            for up in node.upstream_components:
                if up in nodes:
                    in_degree[comp] = in_degree.get(comp, 0) + 1

        result: List[str] = []
        queue = [c for c, d in sorted(in_degree.items()) if d == 0]
        while queue:
            c = queue.pop(0)
            result.append(c)
            node = nodes.get(c)
            if node:
                for down in node.downstream_components:
                    if down in in_degree:
                        in_degree[down] -= 1
                        if in_degree[down] == 0:
                            queue.append(down)
        for c in nodes:
            if c not in result:
                result.append(c)
        return result

    def get_dag_context(self, campaign_id: Optional[str] = None) -> DAGContext:
        """Build a snapshot of the DAG state for a specific campaign.

        Returns a ``DAGContext`` scoped to *campaign_id*.  The topology
        (component ordering, edges) is shared, but the fan-out map is
        campaign-specific: components without a fan-out hint for this
        campaign appear as ``None``.

        The Planner operates per-campaign.  For inter-campaign backfill
        scheduling, it can call this method for each active campaign and
        compare their DAGContexts.
        """
        cid = str(campaign_id or "").strip() or _DEFAULT_CAMPAIGN
        nodes = self._resolve_component_dag()
        order = self.topological_order()
        with self._lock:
            campaign_map = dict(self._campaign_fan_out.get(cid, {}))
            unresolved_count = 0
            coverage_complete = True
        for comp in nodes:
            campaign_map.setdefault(comp, None)
        return DAGContext(
            topology_registered=self._topology_registered,
            campaign_id=campaign_id,
            component_order=order,
            fan_out_map=campaign_map,
            upstream_map={c: list(n.upstream_components) for c, n in nodes.items()},
            downstream_map={c: list(n.downstream_components) for c, n in nodes.items()},
            join_type_map={c: n.join_type for c, n in nodes.items()},
            is_component_coverage_complete=coverage_complete,
            unresolved_process_count=unresolved_count,
        )

    @property
    def campaign_ids(self) -> List[str]:
        """Return all campaign IDs that have registered fan-out data."""
        with self._lock:
            return [
                cid for cid in self._campaign_fan_out
                if cid != _DEFAULT_CAMPAIGN
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging or API responses."""
        nodes = self._resolve_component_dag()
        with self._lock:
            campaigns = {
                cid: dict(fan_map)
                for cid, fan_map in self._campaign_fan_out.items()
            }
        return {
            "topology_registered": self._topology_registered,
            "process_count": len(self._process_vertices),
            "edge_count": len(self._process_edges),
            "process_to_component": dict(self._process_to_component),
            "campaign_fan_out": campaigns,
            "component_dag": {
                comp: node.to_dict()
                for comp, node in sorted(nodes.items())
            },
        }


    def _resolve_component_dag(self) -> Dict[str, ComponentDAGNode]:
        """Build the component-level DAG from process-level graph + mapping."""
        with self._lock:
            if self._component_nodes is not None:
                return self._component_nodes

            if not self._topology_registered:
                self._component_nodes = {}
                return self._component_nodes

            component_upstream: Dict[str, Set[str]] = {}
            component_downstream: Dict[str, Set[str]] = {}

            for comp in set(self._process_to_component.values()):
                component_upstream.setdefault(comp, set())
                component_downstream.setdefault(comp, set())

            adj_fwd: Dict[str, Set[str]] = {}
            for edge in self._process_edges:
                adj_fwd.setdefault(edge["from"], set()).add(edge["to"])

            def _reachable_mapped_descendants(start: str) -> Set[str]:
                """BFS forward from ``start``; return components of every
                mapped vertex reached.  Mapped vertices terminate the
                walk on that branch (they don't propagate further) so we
                only get the *nearest* downstream component along each
                path — a contracted edge in the component graph."""
                out: Set[str] = set()
                seen: Set[str] = {start}
                queue: List[str] = list(adj_fwd.get(start, ()))
                while queue:
                    nxt = queue.pop(0)
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    nxt_comp = self._process_to_component.get(nxt)
                    if nxt_comp:
                        out.add(nxt_comp)
                        continue
                    queue.extend(adj_fwd.get(nxt, ()))
                return out

            for proc, comp in self._process_to_component.items():
                for d_comp in _reachable_mapped_descendants(proc):
                    if d_comp == comp:
                        continue
                    component_upstream.setdefault(d_comp, set()).add(comp)
                    component_downstream.setdefault(comp, set()).add(d_comp)

            nodes: Dict[str, ComponentDAGNode] = {}
            for comp in set(self._process_to_component.values()):
                nodes[comp] = ComponentDAGNode(
                    component=comp,
                    upstream_components=sorted(component_upstream.get(comp, set())),
                    downstream_components=sorted(component_downstream.get(comp, set())),
                    join_type=self._component_join_types.get(comp, "independent"),
                )

            self._component_nodes = nodes
            return nodes

    def __len__(self) -> int:
        return len(self._resolve_component_dag())

    def __bool__(self) -> bool:
        return self._topology_registered
