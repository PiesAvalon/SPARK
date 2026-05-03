"""Baseline strategies described in the paper."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence

from ..core.suite import Cluster, GraphCase, PrioritizedSuite
from ..coverage.metrics import CoverageMetricCalculator
from ..coverage.pattern_space import GraphPatternCounts, PatternSpaceExtractor
from ..similarity.hashing import SimHashEncoder
from ..similarity.wl import WLFeatureExtractor


def _case_lookup(suite: Sequence[GraphCase]) -> dict[str, GraphCase]:
    return {case.case_id: case for case in suite}


def _original_key(case: GraphCase) -> tuple[int, str]:
    return (case.original_order, case.case_id)


def _score_key(case: GraphCase) -> tuple[float, int, str]:
    return (-case.static_score(), case.original_order, case.case_id)


def _cluster_groups(suite: Sequence[GraphCase]) -> dict[str, list[GraphCase]]:
    groups: dict[str, list[GraphCase]] = {}
    for case in suite:
        groups.setdefault(case.cluster_id or f"singleton:{case.case_id}", []).append(case)
    return groups


def _clusters_from_suite(suite: Sequence[GraphCase]) -> list[Cluster]:
    clusters: list[Cluster] = []
    for cluster_id, cases in sorted(_cluster_groups(suite).items()):
        ranked = sorted(cases, key=_score_key)
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                case_ids=[case.case_id for case in ranked],
                representative_id=ranked[0].case_id if ranked else None,
                member_ids=[case.case_id for case in ranked[1:]],
            )
        )
    return clusters


def _materialize(strategy_name: str, ordered_cases: list[GraphCase], suite: Sequence[GraphCase]) -> PrioritizedSuite:
    return PrioritizedSuite(
        strategy_name=strategy_name,
        ordered_case_ids=[case.case_id for case in ordered_cases],
        case_lookup=_case_lookup(suite),
        clusters=_clusters_from_suite(suite),
    )


def _reference_spaces(suite: Sequence[GraphCase]):
    if not suite:
        return None
    return suite[0].metadata.get("pattern_spaces")


@dataclass(slots=True)
class OriginalOrderStrategy:
    name: str = "original-order"

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        ordered = sorted(suite, key=_original_key)
        return _materialize(self.name, ordered, suite)


@dataclass(slots=True)
class RandomStrategy:
    seed: int = 20260425
    name: str = "random"

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        ordered = list(suite)
        random.Random(self.seed).shuffle(ordered)
        return _materialize(self.name, ordered, suite)


@dataclass(slots=True)
class CoverageOnlyStrategy:
    name: str = "coverage-only"

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        ordered = sorted(suite, key=_score_key)
        return _materialize(self.name, ordered, suite)


@dataclass(slots=True)
class CoverageGreedyStrategy:
    name: str = "coverage-greedy"
    extractor: PatternSpaceExtractor = field(default_factory=PatternSpaceExtractor)

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        spaces = _reference_spaces(suite)
        if spaces is None or any(case.pattern_counts is None for case in suite):
            return CoverageOnlyStrategy().prioritize(suite)

        calculator = CoverageMetricCalculator(extractor=self.extractor)
        aggregate = GraphPatternCounts()
        remaining = list(suite)
        ordered: list[GraphCase] = []

        while remaining:
            best_case = max(
                remaining,
                key=lambda case: (
                    calculator.marginal_gain(aggregate, case.pattern_counts or GraphPatternCounts(), spaces),
                    case.static_score(),
                    -case.original_order,
                ),
            )
            ordered.append(best_case)
            if best_case.pattern_counts is not None:
                aggregate.merge(best_case.pattern_counts)
            remaining.remove(best_case)

        return _materialize(self.name, ordered, suite)


@dataclass(slots=True)
class StructureOnlyStrategy:
    name: str = "structure-only"

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        groups = _cluster_groups(suite)
        cluster_order = sorted(
            groups.items(),
            key=lambda item: min(_original_key(case) for case in item[1]),
        )

        ordered: list[GraphCase] = []
        for _, cases in cluster_order:
            representative = min(cases, key=_original_key)
            ordered.append(representative)
        for _, cases in cluster_order:
            representative = min(cases, key=_original_key)
            residual = [case for case in sorted(cases, key=_original_key) if case.case_id != representative.case_id]
            ordered.extend(residual)

        return _materialize(self.name, ordered, suite)


@dataclass(slots=True)
class PairwiseGreedyStrategy:
    name: str = "pairwise-greedy"

    def _distance(self, left: GraphCase, right: GraphCase) -> float:
        if left.wl_features and right.wl_features:
            return 1.0 - WLFeatureExtractor.cosine_similarity(left.wl_features, right.wl_features)
        if left.simhash_signature and right.simhash_signature:
            return 1.0 - SimHashEncoder.hamming_similarity(left.simhash_signature, right.simhash_signature)
        return abs(left.static_score() - right.static_score())

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        remaining = list(suite)
        if not remaining:
            return _materialize(self.name, [], suite)

        ordered = [max(remaining, key=lambda case: (case.static_score(), -case.original_order))]
        remaining.remove(ordered[0])

        while remaining:
            candidate = max(
                remaining,
                key=lambda case: (
                    min(self._distance(case, selected) for selected in ordered),
                    case.static_score(),
                    -case.original_order,
                ),
            )
            ordered.append(candidate)
            remaining.remove(candidate)

        return _materialize(self.name, ordered, suite)
