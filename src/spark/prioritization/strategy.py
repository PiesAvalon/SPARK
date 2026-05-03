"""Strategy interfaces for SPARK and its baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..config import SchedulerConfig
from ..core.suite import GraphCase, PrioritizedSuite
from .baselines import (
    CoverageGreedyStrategy,
    CoverageOnlyStrategy,
    OriginalOrderStrategy,
    PairwiseGreedyStrategy,
    RandomStrategy,
    StructureOnlyStrategy,
)
from .scheduler import AdaptiveScheduler


class PrioritizationStrategy(Protocol):
    name: str

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        """Return a complete execution order."""


@dataclass(slots=True)
class SparkStrategy:
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    name: str = "spark"

    def prioritize(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        scheduler = AdaptiveScheduler(alpha=self.scheduler_config.alpha, gamma=self.scheduler_config.gamma)
        prioritized = scheduler.plan_without_feedback(suite)
        prioritized.strategy_name = self.name
        return prioritized


@dataclass(slots=True)
class StrategyRegistry:
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    def create(self, strategy_name: str) -> PrioritizationStrategy:
        normalized = strategy_name.lower()
        if normalized == "spark":
            return SparkStrategy(scheduler_config=self.scheduler_config)
        if normalized == "original-order":
            return OriginalOrderStrategy()
        if normalized == "random":
            return RandomStrategy()
        if normalized == "coverage-only":
            return CoverageOnlyStrategy()
        if normalized == "coverage-greedy":
            return CoverageGreedyStrategy()
        if normalized == "structure-only":
            return StructureOnlyStrategy()
        if normalized == "pairwise-greedy":
            return PairwiseGreedyStrategy()
        raise KeyError(f"Unknown prioritization strategy: {strategy_name}")
