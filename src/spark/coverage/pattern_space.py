"""Pattern-space extraction for SPARK coverage metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ..graph.schema import ComputationGraph


@dataclass(slots=True)
class GraphPatternCounts:
    operator_patterns: Counter[str] = field(default_factory=Counter)
    ordered_subgraphs: Counter[str] = field(default_factory=Counter)
    unordered_subgraphs: Counter[str] = field(default_factory=Counter)
    path_lengths: Counter[int] = field(default_factory=Counter)
    cooccurring_operator_pairs: Counter[tuple[str, str]] = field(default_factory=Counter)

    def clone(self) -> "GraphPatternCounts":
        return GraphPatternCounts(
            operator_patterns=Counter(self.operator_patterns),
            ordered_subgraphs=Counter(self.ordered_subgraphs),
            unordered_subgraphs=Counter(self.unordered_subgraphs),
            path_lengths=Counter(self.path_lengths),
            cooccurring_operator_pairs=Counter(self.cooccurring_operator_pairs),
        )

    def merge(self, other: "GraphPatternCounts") -> None:
        self.operator_patterns.update(other.operator_patterns)
        self.ordered_subgraphs.update(other.ordered_subgraphs)
        self.unordered_subgraphs.update(other.unordered_subgraphs)
        self.path_lengths.update(other.path_lengths)
        self.cooccurring_operator_pairs.update(other.cooccurring_operator_pairs)


def merge_pattern_counts(parts: Iterable[GraphPatternCounts]) -> GraphPatternCounts:
    merged = GraphPatternCounts()
    for part in parts:
        merged.merge(part)
    return merged


@dataclass(slots=True)
class PatternSpaces:
    operator_patterns: set[str] = field(default_factory=set)
    ordered_subgraphs: set[str] = field(default_factory=set)
    unordered_subgraphs: set[str] = field(default_factory=set)
    path_lengths: set[int] = field(default_factory=set)
    cooccurring_operator_pairs: set[tuple[str, str]] = field(default_factory=set)
    saturation_thresholds: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedPatternSuite:
    spaces: PatternSpaces
    by_graph_id: dict[str, GraphPatternCounts] = field(default_factory=dict)


@dataclass(slots=True)
class PatternSpaceExtractor:
    wedge_cap: int = 16

    def extract_graph_patterns(self, graph: ComputationGraph) -> GraphPatternCounts:
        counts = GraphPatternCounts()

        for node_id in graph.node_ids():
            operator_signature = graph.local_operator_signature(node_id)
            counts.operator_patterns[operator_signature] += 1
            counts.ordered_subgraphs[graph.ordered_subgraph_code((node_id,))] += 1
            counts.unordered_subgraphs[graph.unordered_subgraph_code((node_id,))] += 1

        for src, dst in graph.connected_pairs():
            pair = (src, dst)
            counts.ordered_subgraphs[graph.ordered_subgraph_code(pair)] += 1
            counts.unordered_subgraphs[graph.unordered_subgraph_code(pair)] += 1

        for triplet in graph.sampled_triplets(self.wedge_cap):
            counts.ordered_subgraphs[graph.ordered_subgraph_code(triplet)] += 1
            counts.unordered_subgraphs[graph.unordered_subgraph_code(triplet)] += 1

        for path_length in graph.source_to_sink_path_lengths():
            counts.path_lengths[path_length] += 1

        for pair in graph.cooccurring_operator_type_pairs():
            counts.cooccurring_operator_pairs[pair] += 1

        return counts

    def extract_many(self, graphs: Iterable[ComputationGraph]) -> dict[str, GraphPatternCounts]:
        return {graph.graph_id: self.extract_graph_patterns(graph) for graph in graphs}

    def build_spaces(self, pattern_counts: dict[str, GraphPatternCounts]) -> PatternSpaces:
        spaces = PatternSpaces()
        for counts in pattern_counts.values():
            spaces.operator_patterns.update(counts.operator_patterns.keys())
            spaces.ordered_subgraphs.update(counts.ordered_subgraphs.keys())
            spaces.unordered_subgraphs.update(counts.unordered_subgraphs.keys())
            spaces.path_lengths.update(counts.path_lengths.keys())
            spaces.cooccurring_operator_pairs.update(counts.cooccurring_operator_pairs.keys())
            for pattern, count in counts.operator_patterns.items():
                spaces.saturation_thresholds[pattern] = max(spaces.saturation_thresholds.get(pattern, 0), count)
        return spaces

    def extract_suite(self, graphs: Iterable[ComputationGraph]) -> ExtractedPatternSuite:
        by_graph_id = self.extract_many(graphs)
        return ExtractedPatternSuite(spaces=self.build_spaces(by_graph_id), by_graph_id=by_graph_id)

    def extract(self, graphs: Iterable[ComputationGraph]) -> PatternSpaces:
        return self.extract_suite(graphs).spaces
