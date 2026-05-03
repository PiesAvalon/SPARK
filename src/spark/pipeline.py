"""Top-level SPARK pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .config import SparkConfig
from .core.suite import Cluster, GraphCase, PreparedSuite, PrioritizedSuite
from .core.types import ExecutionResult, TestCase
from .coverage.pattern_space import PatternSpaceExtractor
from .coverage.scorer import CoverageScorer
from .evaluation.metrics import PrioritizationEvaluator
from .evaluation.progress import ProgressCurveEvaluator
from .evaluation.scalability import ScalabilityProfiler
from .execution.runner import ExecutionRunner
from .graph.builder import GraphBuilderRegistry
from .prioritization.scheduler import AdaptiveScheduler
from .prioritization.strategy import StrategyRegistry
from .similarity.clustering import StructuralClusterer
from .similarity.hashing import SimHashEncoder
from .similarity.wl import WLFeatureExtractor


@dataclass(slots=True)
class SparkPipeline:
    config: SparkConfig = field(default_factory=SparkConfig)
    builder_registry: GraphBuilderRegistry = field(default_factory=GraphBuilderRegistry)
    pattern_extractor: PatternSpaceExtractor | None = None
    coverage_scorer: CoverageScorer | None = None
    wl_extractor: WLFeatureExtractor = field(default_factory=WLFeatureExtractor)
    simhash_encoder: SimHashEncoder = field(default_factory=SimHashEncoder)
    clusterer: StructuralClusterer = field(default_factory=StructuralClusterer)
    strategy_registry: StrategyRegistry | None = None
    execution_runner: ExecutionRunner = field(default_factory=ExecutionRunner)
    evaluator: PrioritizationEvaluator | None = None
    progress_evaluator: ProgressCurveEvaluator | None = None
    scalability_profiler: ScalabilityProfiler = field(default_factory=ScalabilityProfiler)

    def __post_init__(self) -> None:
        if self.pattern_extractor is None:
            self.pattern_extractor = PatternSpaceExtractor(wedge_cap=self.config.coverage.wedge_cap)
        if self.coverage_scorer is None:
            self.coverage_scorer = CoverageScorer(extractor=self.pattern_extractor)
        if self.strategy_registry is None:
            self.strategy_registry = StrategyRegistry(scheduler_config=self.config.scheduler)
        if self.evaluator is None:
            self.evaluator = PrioritizationEvaluator(
                checkpoints=self.config.evaluation.checkpoints,
                use_root_causes=self.config.evaluation.use_root_causes_when_available,
            )
        if self.progress_evaluator is None:
            self.progress_evaluator = ProgressCurveEvaluator(
                checkpoints=self.config.evaluation.checkpoints,
                extractor=self.pattern_extractor,
            )

    def prepare_suite(self, cases: Sequence[TestCase]) -> PreparedSuite:
        assert self.pattern_extractor is not None
        assert self.coverage_scorer is not None

        built_graphs = []
        case_by_graph_id: dict[str, TestCase] = {}
        for case in cases:
            builder = self.builder_registry.get(case.tool_name)
            graph = builder.build(case)
            built_graphs.append(graph)
            case_by_graph_id[graph.graph_id] = case

        extracted = self.pattern_extractor.extract_suite(built_graphs)
        coverage_vectors = self.coverage_scorer.score_suite(built_graphs, extracted.spaces, extracted.by_graph_id)
        wl_features = self.wl_extractor.extract_many(built_graphs, self.config.similarity.wl_iterations)
        signatures = self.simhash_encoder.encode_many(wl_features, self.config.similarity.simhash_bits)
        structural_clusters = self.clusterer.cluster(
            wl_features=wl_features,
            signatures=signatures,
            bands=self.config.similarity.lsh_bands,
        )

        graph_cases: list[GraphCase] = []
        graph_id_to_case_id: dict[str, str] = {}
        for graph in built_graphs:
            test_case = case_by_graph_id[graph.graph_id]
            pattern_counts = extracted.by_graph_id[graph.graph_id]
            coverage = coverage_vectors[graph.graph_id]
            graph_case = GraphCase(
                case_id=test_case.case_id,
                tool_name=test_case.tool_name,
                graph_id=graph.graph_id,
                test_case=test_case,
                graph=graph,
                pattern_counts=pattern_counts,
                coverage=coverage,
                coverage_score=coverage.ocs or 0.0,
                wl_features=wl_features.get(graph.graph_id, {}),
                simhash_signature=signatures.get(graph.graph_id),
                cluster_id=structural_clusters.case_to_cluster.get(graph.graph_id),
                original_order=test_case.original_order,
                metadata={"pattern_spaces": extracted.spaces},
            )
            graph_cases.append(graph_case)
            graph_id_to_case_id[graph.graph_id] = test_case.case_id

        clusters: list[Cluster] = []
        for cluster_id, members in sorted(structural_clusters.cluster_members.items()):
            case_ids = [graph_id_to_case_id[graph_id] for graph_id in members if graph_id in graph_id_to_case_id]
            ranked = sorted(
                (case for case in graph_cases if case.case_id in case_ids),
                key=lambda item: (-item.static_score(), item.original_order, item.case_id),
            )
            if ranked:
                clusters.append(
                    Cluster(
                        cluster_id=cluster_id,
                        case_ids=[case.case_id for case in ranked],
                        representative_id=ranked[0].case_id,
                        member_ids=[case.case_id for case in ranked[1:]],
                    )
                )

        return PreparedSuite(
            cases=graph_cases,
            pattern_spaces=extracted.spaces,
            clusters=clusters,
            metadata={
                "cluster_quality": self.clusterer.quality_statistics(wl_features, signatures, structural_clusters),
                "scalability_profile": self.scalability_profiler.profile_cases(graph_cases),
            },
        )

    def prioritize(self, suite: PreparedSuite, strategy_name: str | None = None) -> PrioritizedSuite:
        assert self.strategy_registry is not None
        name = strategy_name or self.config.experiment.default_strategy
        strategy = self.strategy_registry.create(name)
        prioritized = strategy.prioritize(suite.cases)
        prioritized.metadata.setdefault("prepared_cluster_count", len(suite.clusters))
        return prioritized

    def execute(self, prioritized: PrioritizedSuite) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for case in prioritized.ordered_cases():
            result = self.execution_runner.run_graph_case(case)
            result.strategy_name = prioritized.strategy_name
            results.append(result)
        prioritized.attach_results(results)
        return results

    def evaluate(
        self,
        prioritized: PrioritizedSuite,
        results: Sequence[ExecutionResult] | None = None,
    ) -> dict[str, float]:
        assert self.evaluator is not None
        assert self.progress_evaluator is not None

        ordered_results = list(results if results is not None else prioritized.execution_results)
        if not ordered_results:
            return {
                "planned_cases": float(len(prioritized.ordered_case_ids)),
                "cluster_count": float(len(prioritized.clusters)),
            }

        summary = self.evaluator.summarize(ordered_results)
        metrics = dict(summary.scalar_metrics)
        metrics.update(summary.checkpoint_metrics)

        bug_progress = self.progress_evaluator.bug_progress(ordered_results)
        coverage_progress = self.progress_evaluator.coverage_progress(prioritized.ordered_cases())
        for checkpoint, value in zip(bug_progress.checkpoints, bug_progress.bug_progress):
            metrics[f"progress_bug_{int(checkpoint * 100)}pct"] = value
        for checkpoint, value in zip(coverage_progress.checkpoints, coverage_progress.coverage_progress):
            metrics[f"progress_coverage_{int(checkpoint * 100)}pct"] = value
        return metrics

    def run(
        self,
        cases: Sequence[TestCase],
        strategy_name: str | None = None,
        adaptive: bool = True,
    ) -> tuple[PrioritizedSuite, list[ExecutionResult], dict[str, float]]:
        prepared = self.prepare_suite(cases)
        strategy_name = strategy_name or self.config.experiment.default_strategy
        results: list[ExecutionResult] = []

        if strategy_name == "spark" and adaptive and not self.config.execution.plan_only:
            scheduler = AdaptiveScheduler(alpha=self.config.scheduler.alpha, gamma=self.config.scheduler.gamma)
            prioritized = scheduler.run(prepared.cases, self.execution_runner.run_graph_case)
            results = list(prioritized.execution_results)
        else:
            prioritized = self.prioritize(prepared, strategy_name)

        if self.config.execution.plan_only:
            metrics = self.evaluate(prioritized, [])
            return (prioritized, [], metrics)

        if not results:
            results = self.execute(prioritized)
        metrics = self.evaluate(prioritized, results)
        return (prioritized, results, metrics)
