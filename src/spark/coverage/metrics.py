"""Definitions for the six SPARK coverage metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p

from ..graph.schema import ComputationGraph
from .pattern_space import GraphPatternCounts, PatternSpaceExtractor, PatternSpaces, merge_pattern_counts


@dataclass(slots=True)
class CoverageVector:
    opc: float | None = None
    osc: float | None = None
    usc: float | None = None
    nrc: float | None = None
    plc: float | None = None
    coc: float | None = None
    operator_component: float | None = None
    subgraph_component: float | None = None
    global_component: float | None = None
    ocs: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "opc": self.opc,
            "osc": self.osc,
            "usc": self.usc,
            "nrc": self.nrc,
            "plc": self.plc,
            "coc": self.coc,
            "operator_component": self.operator_component,
            "subgraph_component": self.subgraph_component,
            "global_component": self.global_component,
            "ocs": self.ocs,
        }


@dataclass(slots=True)
class CoverageMetricCalculator:
    extractor: PatternSpaceExtractor

    def _counts_for(self, graph: ComputationGraph, counts: GraphPatternCounts | None = None) -> GraphPatternCounts:
        return counts if counts is not None else self.extractor.extract_graph_patterns(graph)

    def _presence_ratio(self, present: set[object], space: set[object]) -> float:
        if not space:
            return 0.0
        return len(present & space) / float(len(space))

    def operator_pattern_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        return self._presence_ratio(set(counts.operator_patterns), spaces.operator_patterns)

    def ordered_subgraph_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        return self._presence_ratio(set(counts.ordered_subgraphs), spaces.ordered_subgraphs)

    def unordered_subgraph_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        return self._presence_ratio(set(counts.unordered_subgraphs), spaces.unordered_subgraphs)

    def non_redundant_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        if not spaces.saturation_thresholds:
            return 0.0
        denominator = 0.0
        numerator = 0.0
        for pattern, max_count in spaces.saturation_thresholds.items():
            denominator += log1p(max_count)
            numerator += log1p(min(counts.operator_patterns.get(pattern, 0), max_count))
        return 0.0 if denominator == 0 else numerator / denominator

    def path_length_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        return self._presence_ratio(set(counts.path_lengths), spaces.path_lengths)

    def cooccurring_operator_coverage(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> float:
        return self._presence_ratio(set(counts.cooccurring_operator_pairs), spaces.cooccurring_operator_pairs)

    def compute_from_counts(self, counts: GraphPatternCounts, spaces: PatternSpaces) -> CoverageVector:
        opc = self.operator_pattern_coverage(counts, spaces)
        osc = self.ordered_subgraph_coverage(counts, spaces)
        usc = self.unordered_subgraph_coverage(counts, spaces)
        nrc = self.non_redundant_coverage(counts, spaces)
        plc = self.path_length_coverage(counts, spaces)
        coc = self.cooccurring_operator_coverage(counts, spaces)
        operator_component = (opc + nrc) / 2.0
        subgraph_component = (osc + usc + coc) / 3.0
        global_component = plc
        ocs = (operator_component + subgraph_component + global_component) / 3.0
        return CoverageVector(
            opc=opc,
            osc=osc,
            usc=usc,
            nrc=nrc,
            plc=plc,
            coc=coc,
            operator_component=operator_component,
            subgraph_component=subgraph_component,
            global_component=global_component,
            ocs=ocs,
        )

    def compute(self, graph: ComputationGraph, spaces: PatternSpaces, counts: GraphPatternCounts | None = None) -> CoverageVector:
        return self.compute_from_counts(self._counts_for(graph, counts), spaces)

    def marginal_gain(
        self,
        aggregate_counts: GraphPatternCounts,
        candidate_counts: GraphPatternCounts,
        spaces: PatternSpaces,
    ) -> float:
        before = self.compute_from_counts(aggregate_counts, spaces).ocs or 0.0
        after_counts = merge_pattern_counts((aggregate_counts, candidate_counts))
        after = self.compute_from_counts(after_counts, spaces).ocs or 0.0
        return after - before
