"""Computational graph schema used by SPARK."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from itertools import combinations


@dataclass(frozen=True, slots=True)
class NodeLabel:
    op_type: str
    shape_summary: str | None = None
    dtype_summary: str | None = None
    attr_summary: tuple[tuple[str, str], ...] = ()

    def as_key(self) -> tuple[str, str | None, str | None, tuple[tuple[str, str], ...]]:
        return (self.op_type, self.shape_summary, self.dtype_summary, tuple(sorted(self.attr_summary)))


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    label: NodeLabel


@dataclass(slots=True)
class ComputationGraph:
    graph_id: str
    case_id: str
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str) -> None:
        if src == dst:
            return
        if src not in self.nodes or dst not in self.nodes:
            return
        if (src, dst) not in self.edges:
            self.edges.append((src, dst))

    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def successors_map(self) -> dict[str, list[str]]:
        succ: dict[str, list[str]] = defaultdict(list)
        for src, dst in self.edges:
            succ[src].append(dst)
        for node_id in self.nodes:
            succ.setdefault(node_id, [])
        return {node_id: sorted(neighbors) for node_id, neighbors in succ.items()}

    def predecessors_map(self) -> dict[str, list[str]]:
        pred: dict[str, list[str]] = defaultdict(list)
        for src, dst in self.edges:
            pred[dst].append(src)
        for node_id in self.nodes:
            pred.setdefault(node_id, [])
        return {node_id: sorted(neighbors) for node_id, neighbors in pred.items()}

    def successors(self, node_id: str) -> list[str]:
        return self.successors_map().get(node_id, [])

    def predecessors(self, node_id: str) -> list[str]:
        return self.predecessors_map().get(node_id, [])

    def indegree(self, node_id: str) -> int:
        return len(self.predecessors(node_id))

    def outdegree(self, node_id: str) -> int:
        return len(self.successors(node_id))

    def sources(self) -> list[str]:
        preds = self.predecessors_map()
        return [node_id for node_id, parents in preds.items() if not parents]

    def sinks(self) -> list[str]:
        succs = self.successors_map()
        return [node_id for node_id, children in succs.items() if not children]

    def topological_order(self) -> list[str]:
        preds = self.predecessors_map()
        succs = self.successors_map()
        indegree = {node_id: len(preds[node_id]) for node_id in self.nodes}
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for nxt in succs[node_id]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            raise ValueError(f"Graph {self.graph_id} is not a DAG.")
        return order

    def validate_dag(self) -> None:
        self.topological_order()

    def induced_edges(self, node_ids: set[str]) -> list[tuple[str, str]]:
        return [(src, dst) for src, dst in self.edges if src in node_ids and dst in node_ids]

    def local_operator_signature(self, node_id: str) -> str:
        label = self.nodes[node_id].label
        signature = {
            "label": label.as_key(),
            "indegree": self.indegree(node_id),
            "outdegree": self.outdegree(node_id),
        }
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    def ordered_subgraph_code(self, node_ids: tuple[str, ...]) -> str:
        subset = set(node_ids)
        full_order = self.topological_order()
        ordered_nodes = [node_id for node_id in full_order if node_id in subset]
        pos = {node_id: index for index, node_id in enumerate(ordered_nodes)}
        payload = {
            "nodes": [self.local_operator_signature(node_id) for node_id in ordered_nodes],
            "edges": sorted((pos[src], pos[dst]) for src, dst in self.induced_edges(subset)),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def unordered_subgraph_code(self, node_ids: tuple[str, ...]) -> str:
        subset = set(node_ids)
        node_codes = sorted((self.local_operator_signature(node_id), node_id) for node_id in subset)
        sig_lookup = {node_id: code for code, node_id in node_codes}
        edge_codes = sorted((sig_lookup[src], sig_lookup[dst]) for src, dst in self.induced_edges(subset))
        payload = {
            "nodes": [code for code, _ in node_codes],
            "edges": edge_codes,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def connected_pairs(self) -> list[tuple[str, str]]:
        return sorted(self.edges)

    def sampled_triplets(self, wedge_cap: int) -> list[tuple[str, str, str]]:
        pred = self.predecessors_map()
        succ = self.successors_map()
        triplets: set[tuple[str, str, str]] = set()
        for center in sorted(self.nodes):
            neighborhood = sorted(set(pred[center]) | set(succ[center]))
            for left, right in list(combinations(neighborhood, 2))[:wedge_cap]:
                node_tuple = tuple(sorted((left, center, right)))
                triplets.add(node_tuple)
        return sorted(triplets)

    def reachable_pairs(self) -> set[tuple[str, str]]:
        succ = self.successors_map()
        pairs: set[tuple[str, str]] = set()
        for src in self.topological_order():
            queue = deque(succ[src])
            seen = set(queue)
            while queue:
                dst = queue.popleft()
                pairs.add((src, dst))
                for nxt in succ[dst]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        return pairs

    def cooccurring_operator_type_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for src, dst in self.reachable_pairs():
            src_type = self.nodes[src].label.op_type
            dst_type = self.nodes[dst].label.op_type
            pairs.add((src_type, dst_type))
        return pairs

    def source_to_sink_path_lengths(self) -> set[int]:
        order = self.topological_order()
        succ = self.successors_map()
        lengths_to: dict[str, set[int]] = {node_id: set() for node_id in self.nodes}
        for source in self.sources():
            lengths_to[source].add(0)
        for node_id in order:
            current = lengths_to[node_id] or {0}
            for nxt in succ[node_id]:
                lengths_to[nxt].update(length + 1 for length in current)
        path_lengths: set[int] = set()
        for sink in self.sinks():
            path_lengths.update(lengths_to[sink] or {0})
        return path_lengths or {0}

    def operator_types(self) -> set[str]:
        return {node.label.op_type for node in self.nodes.values()}
