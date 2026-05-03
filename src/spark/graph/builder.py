"""Graph builders that convert tool-specific tests into computational graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..core.types import TestCase
from .normalization import LabelNormalizer
from .schema import ComputationGraph, GraphNode, NodeLabel


class GraphBuilder(Protocol):
    def build(self, case: TestCase) -> ComputationGraph:
        """Build a computational graph from one test case."""


@dataclass(slots=True)
class GenericGraphBuilder:
    normalizer: LabelNormalizer = field(default_factory=LabelNormalizer)

    def build(self, case: TestCase) -> ComputationGraph:
        payload = case.payload()
        if payload:
            return self._build_from_payload(case, payload)
        if case.operator_sequence:
            return self._build_from_sequence(case, list(case.operator_sequence))
        metadata_sequence = case.metadata.get("operator_sequence")
        if isinstance(metadata_sequence, list):
            return self._build_from_sequence(case, [str(item) for item in metadata_sequence])
        operators = case.metadata.get("operators")
        if isinstance(operators, list):
            return self._build_from_operator_records(case, operators)
        return self._build_placeholder(case)

    def _build_from_payload(self, case: TestCase, payload: dict[str, object]) -> ComputationGraph:
        graph = ComputationGraph(graph_id=f"{case.case_id}:graph", case_id=case.case_id, metadata={"source": "payload"})
        node_specs = payload.get("nodes", [])
        edge_specs = payload.get("edges", [])
        if isinstance(node_specs, list):
            for index, spec in enumerate(node_specs):
                if not isinstance(spec, dict):
                    continue
                node_id = str(spec.get("id", f"n{index}"))
                graph.add_node(GraphNode(node_id=node_id, label=self._make_label(spec)))
        if not graph.nodes:
            operators = payload.get("operators")
            if isinstance(operators, list):
                return self._build_from_operator_records(case, operators)
        if isinstance(edge_specs, list):
            for edge in edge_specs:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    graph.add_edge(str(edge[0]), str(edge[1]))
                elif isinstance(edge, dict):
                    src = edge.get("src") or edge.get("from")
                    dst = edge.get("dst") or edge.get("to")
                    if src is not None and dst is not None:
                        graph.add_edge(str(src), str(dst))
        if not graph.edges and len(graph.nodes) > 1:
            ordered = sorted(graph.nodes)
            for src, dst in zip(ordered, ordered[1:]):
                graph.add_edge(src, dst)
        return graph

    def _build_from_sequence(self, case: TestCase, operators: list[str]) -> ComputationGraph:
        graph = ComputationGraph(graph_id=f"{case.case_id}:graph", case_id=case.case_id, metadata={"source": "sequence"})
        previous_id: str | None = None
        for index, operator_name in enumerate(operators):
            node_id = f"n{index}"
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    label=self.normalizer.normalize_label(NodeLabel(op_type=operator_name)),
                )
            )
            if previous_id is not None:
                graph.add_edge(previous_id, node_id)
            previous_id = node_id
        return graph

    def _build_from_operator_records(self, case: TestCase, records: list[object]) -> ComputationGraph:
        graph = ComputationGraph(graph_id=f"{case.case_id}:graph", case_id=case.case_id, metadata={"source": "records"})
        for index, raw_record in enumerate(records):
            if not isinstance(raw_record, dict):
                continue
            node_id = str(raw_record.get("id", f"n{index}"))
            graph.add_node(GraphNode(node_id=node_id, label=self._make_label(raw_record)))
        for index, raw_record in enumerate(records):
            if not isinstance(raw_record, dict):
                continue
            node_id = str(raw_record.get("id", f"n{index}"))
            inputs = raw_record.get("inputs") or raw_record.get("parents") or raw_record.get("deps") or []
            if isinstance(inputs, list):
                for parent in inputs:
                    parent_id = str(parent)
                    if parent_id in graph.nodes and node_id in graph.nodes:
                        graph.add_edge(parent_id, node_id)
        if not graph.edges and len(graph.nodes) > 1:
            ordered = sorted(graph.nodes)
            for src, dst in zip(ordered, ordered[1:]):
                graph.add_edge(src, dst)
        return graph

    def _build_placeholder(self, case: TestCase) -> ComputationGraph:
        graph = ComputationGraph(graph_id=f"{case.case_id}:graph", case_id=case.case_id, metadata={"source": "placeholder"})
        op_name = str(case.metadata.get("model_type", case.metadata.get("entry_op", "UnknownOp")))
        graph.add_node(GraphNode(node_id="n0", label=self.normalizer.normalize_label(NodeLabel(op_type=op_name))))
        return graph

    def _make_label(self, spec: dict[str, object]) -> NodeLabel:
        attrs = spec.get("attrs") or spec.get("attributes") or {}
        attr_summary: tuple[tuple[str, str], ...] = ()
        if isinstance(attrs, dict):
            attr_summary = tuple((str(key), str(value)) for key, value in sorted(attrs.items()))
        label = NodeLabel(
            op_type=str(spec.get("op") or spec.get("type") or spec.get("name") or "UnknownOp"),
            shape_summary=None if spec.get("shape") is None else str(spec.get("shape")),
            dtype_summary=None if spec.get("dtype") is None else str(spec.get("dtype")),
            attr_summary=attr_summary,
        )
        return self.normalizer.normalize_label(label)


@dataclass(slots=True)
class GraphBuilderRegistry:
    default_builder: GraphBuilder = field(default_factory=GenericGraphBuilder)
    builders: dict[str, GraphBuilder] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for tool_name in ("COMET", "DevMuT", "ModelMeta", "Muffin", "NEURI", "GenCoG", "LEMON", "Gandalf", "NNSmith"):
            self.builders.setdefault(tool_name.lower(), self.default_builder)

    def register(self, tool_name: str, builder: GraphBuilder) -> None:
        self.builders[tool_name.lower()] = builder

    def get(self, tool_name: str) -> GraphBuilder:
        return self.builders.get(tool_name.lower(), self.default_builder)
