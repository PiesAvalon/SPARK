"""Coverage scoring entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..graph.schema import ComputationGraph
from .metrics import CoverageMetricCalculator, CoverageVector
from .pattern_space import GraphPatternCounts, PatternSpaceExtractor, PatternSpaces


@dataclass(slots=True)
class CoverageScorer:
    extractor: PatternSpaceExtractor = field(default_factory=PatternSpaceExtractor)
    calculator: CoverageMetricCalculator | None = None

    def __post_init__(self) -> None:
        if self.calculator is None:
            self.calculator = CoverageMetricCalculator(extractor=self.extractor)

    def score_graph(
        self,
        graph: ComputationGraph,
        spaces: PatternSpaces,
        counts: GraphPatternCounts | None = None,
    ) -> CoverageVector:
        assert self.calculator is not None
        return self.calculator.compute(graph=graph, spaces=spaces, counts=counts)

    def score_suite(
        self,
        graphs: Iterable[ComputationGraph],
        spaces: PatternSpaces,
        counts_by_graph_id: dict[str, GraphPatternCounts] | None = None,
    ) -> dict[str, CoverageVector]:
        scores: dict[str, CoverageVector] = {}
        for graph in graphs:
            counts = None if counts_by_graph_id is None else counts_by_graph_id.get(graph.graph_id)
            scores[graph.graph_id] = self.score_graph(graph, spaces, counts)
        return scores
