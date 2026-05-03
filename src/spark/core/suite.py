"""Suite-level containers for graphs, clusters, and prioritized orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..coverage.metrics import CoverageVector
    from ..coverage.pattern_space import GraphPatternCounts, PatternSpaces
    from ..graph.schema import ComputationGraph
    from .types import ExecutionResult, TestCase


@dataclass(slots=True)
class GraphCase:
    case_id: str
    tool_name: str
    graph_id: str
    test_case: TestCase | None = None
    graph: ComputationGraph | None = None
    pattern_counts: GraphPatternCounts | None = None
    coverage: CoverageVector | None = None
    coverage_score: float = 0.0
    wl_features: dict[str, int] = field(default_factory=dict)
    simhash_signature: str | None = None
    cluster_id: str | None = None
    original_order: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def static_score(self) -> float:
        if self.coverage is not None and self.coverage.ocs is not None:
            return float(self.coverage.ocs)
        return float(self.coverage_score)


@dataclass(slots=True)
class Cluster:
    cluster_id: str
    case_ids: list[str] = field(default_factory=list)
    representative_id: str | None = None
    member_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedSuite:
    cases: list[GraphCase] = field(default_factory=list)
    pattern_spaces: PatternSpaces | None = None
    clusters: list[Cluster] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def case_lookup(self) -> dict[str, GraphCase]:
        return {case.case_id: case for case in self.cases}


@dataclass(slots=True)
class PrioritizedSuite:
    strategy_name: str
    ordered_case_ids: list[str] = field(default_factory=list)
    case_lookup: dict[str, GraphCase] = field(default_factory=dict)
    clusters: list[Cluster] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def ordered_cases(self) -> list[GraphCase]:
        return [self.case_lookup[case_id] for case_id in self.ordered_case_ids if case_id in self.case_lookup]

    def attach_results(self, results: list[ExecutionResult]) -> None:
        self.execution_results = results
