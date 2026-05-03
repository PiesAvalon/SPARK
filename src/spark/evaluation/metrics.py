"""Prioritization effectiveness metrics such as APFD and APFDc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core.types import ExecutionResult


@dataclass(slots=True)
class EvaluationSummary:
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PrioritizationEvaluator:
    checkpoints: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)
    use_root_causes: bool = True

    def _first_detection_positions(self, ordered_results: Sequence[ExecutionResult]) -> dict[str, int]:
        positions: dict[str, int] = {}
        for index, result in enumerate(ordered_results, start=1):
            for bug_id in result.normalized_bug_ids(use_root_causes=self.use_root_causes):
                positions.setdefault(bug_id, index)
        return positions

    def _total_bugs(self, ordered_results: Sequence[ExecutionResult]) -> int:
        return len(self._first_detection_positions(ordered_results))

    def compute_apfd(self, ordered_results: Sequence[ExecutionResult]) -> float:
        num_tests = len(ordered_results)
        if num_tests == 0:
            return 0.0
        first_positions = self._first_detection_positions(ordered_results)
        num_bugs = len(first_positions)
        if num_bugs == 0:
            return 0.0
        return 1.0 - (sum(first_positions.values()) / (num_tests * num_bugs)) + (1.0 / (2.0 * num_tests))

    def compute_apfdc(self, ordered_results: Sequence[ExecutionResult]) -> float:
        total_bugs = self._total_bugs(ordered_results)
        if total_bugs == 0:
            return 0.0

        costs = [float(result.runtime_seconds or 1.0) for result in ordered_results]
        total_cost = sum(costs)
        if total_cost == 0:
            return 0.0

        area = 0.0
        prev_cost_ratio = 0.0
        prev_bug_ratio = 0.0
        seen: set[str] = set()

        for result, cost in zip(ordered_results, costs):
            current_cost_ratio = prev_cost_ratio + (cost / total_cost)
            seen.update(result.normalized_bug_ids(use_root_causes=self.use_root_causes))
            current_bug_ratio = len(seen) / total_bugs
            area += ((prev_bug_ratio + current_bug_ratio) / 2.0) * (current_cost_ratio - prev_cost_ratio)
            prev_cost_ratio = current_cost_ratio
            prev_bug_ratio = current_bug_ratio

        return area

    def _checkpoint_bug_ratios(self, ordered_results: Sequence[ExecutionResult]) -> dict[str, float]:
        total_bugs = self._total_bugs(ordered_results)
        if total_bugs == 0:
            return {f"bugs_at_{int(checkpoint * 100)}pct": 0.0 for checkpoint in self.checkpoints}

        seen: set[str] = set()
        ratios_by_index: list[float] = []
        for result in ordered_results:
            seen.update(result.normalized_bug_ids(use_root_causes=self.use_root_causes))
            ratios_by_index.append(len(seen) / total_bugs)

        metrics: dict[str, float] = {}
        total_tests = max(1, len(ordered_results))
        for checkpoint in self.checkpoints:
            index = min(total_tests - 1, max(0, int(total_tests * checkpoint + 0.999999) - 1))
            metrics[f"bugs_at_{int(checkpoint * 100)}pct"] = ratios_by_index[index] if ratios_by_index else 0.0
        return metrics

    def summarize(self, ordered_results: Sequence[ExecutionResult]) -> EvaluationSummary:
        total_runtime = sum(float(result.runtime_seconds or 1.0) for result in ordered_results)
        summary = EvaluationSummary()
        summary.scalar_metrics = {
            "apfd": self.compute_apfd(ordered_results),
            "apfdc": self.compute_apfdc(ordered_results),
            "num_tests": float(len(ordered_results)),
            "num_bugs": float(self._total_bugs(ordered_results)),
            "total_runtime": float(total_runtime),
        }
        summary.checkpoint_metrics = self._checkpoint_bug_ratios(ordered_results)
        return summary
