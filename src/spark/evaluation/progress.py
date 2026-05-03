"""Progress-curve evaluation for bugs and coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core.suite import GraphCase
from ..core.types import ExecutionResult
from ..coverage.metrics import CoverageMetricCalculator
from ..coverage.pattern_space import GraphPatternCounts, PatternSpaceExtractor


@dataclass(slots=True)
class ProgressSeries:
    checkpoints: list[float] = field(default_factory=list)
    bug_progress: list[float] = field(default_factory=list)
    coverage_progress: list[float] = field(default_factory=list)
    raw_values: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ProgressCurveEvaluator:
    checkpoints: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)
    extractor: PatternSpaceExtractor = field(default_factory=PatternSpaceExtractor)
    calculator: CoverageMetricCalculator | None = None

    def __post_init__(self) -> None:
        if self.calculator is None:
            self.calculator = CoverageMetricCalculator(extractor=self.extractor)

    def _sample(self, values: list[float]) -> list[float]:
        if not values:
            return [0.0 for _ in self.checkpoints]
        total = len(values)
        sampled: list[float] = []
        for checkpoint in self.checkpoints:
            index = min(total - 1, max(0, int(total * checkpoint + 0.999999) - 1))
            sampled.append(values[index])
        return sampled

    def bug_progress(self, ordered_results: Sequence[ExecutionResult]) -> ProgressSeries:
        seen: set[str] = set()
        all_bugs: set[str] = set()
        for result in ordered_results:
            all_bugs.update(result.normalized_bug_ids())
        total_bugs = max(1, len(all_bugs))

        raw: list[float] = []
        for result in ordered_results:
            seen.update(result.normalized_bug_ids())
            raw.append(len(seen) / total_bugs)

        return ProgressSeries(
            checkpoints=list(self.checkpoints),
            bug_progress=self._sample(raw),
            raw_values=raw,
        )

    def coverage_progress(self, ordered_suite: Sequence[GraphCase]) -> ProgressSeries:
        raw: list[float] = []
        if not ordered_suite:
            return ProgressSeries(checkpoints=list(self.checkpoints), coverage_progress=self._sample(raw), raw_values=raw)

        spaces = ordered_suite[0].metadata.get("pattern_spaces")
        if spaces is not None and self.calculator is not None and all(case.pattern_counts is not None for case in ordered_suite):
            aggregate = GraphPatternCounts()
            for case in ordered_suite:
                if case.pattern_counts is not None:
                    aggregate.merge(case.pattern_counts)
                raw.append(self.calculator.compute_from_counts(aggregate, spaces).ocs or 0.0)
        else:
            running_sum = 0.0
            for index, case in enumerate(ordered_suite, start=1):
                running_sum += case.static_score()
                raw.append(running_sum / index)

        return ProgressSeries(
            checkpoints=list(self.checkpoints),
            coverage_progress=self._sample(raw),
            raw_values=raw,
        )
