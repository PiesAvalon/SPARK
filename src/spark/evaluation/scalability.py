"""Scalability and runtime profiling hooks."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Sequence

from ..core.suite import GraphCase


@dataclass(slots=True)
class RuntimeProfile:
    suite_size: int
    average_graph_size: float = 0.0
    total_runtime_seconds: float | None = None
    core_runtime_seconds: float | None = None
    estimated_spark_cost: float | None = None
    estimated_pairwise_cost: float | None = None


@dataclass(slots=True)
class ScalabilityProfiler:
    def profile(self, suite_size: int, average_graph_size: float = 1.0) -> RuntimeProfile:
        spark_cost = (suite_size * average_graph_size) + (suite_size * log2(max(2, suite_size)))
        pairwise_cost = (suite_size * suite_size) * average_graph_size
        return RuntimeProfile(
            suite_size=suite_size,
            average_graph_size=average_graph_size,
            estimated_spark_cost=spark_cost,
            estimated_pairwise_cost=pairwise_cost,
        )

    def profile_cases(self, cases: Sequence[GraphCase]) -> RuntimeProfile:
        if not cases:
            return self.profile(0, 0.0)
        graph_sizes = []
        for case in cases:
            if case.graph is None:
                graph_sizes.append(1.0)
            else:
                graph_sizes.append(float(len(case.graph.nodes) + len(case.graph.edges)))
        avg_graph_size = sum(graph_sizes) / len(graph_sizes)
        return self.profile(len(cases), avg_graph_size)
