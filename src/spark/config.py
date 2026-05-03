"""Configuration objects for the SPARK dataset release package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphConfig:
    include_runtime_details: bool = False
    normalize_operator_names: bool = True
    summarize_tensor_metadata: bool = True
    max_inline_attributes: int = 8
    placeholder_operator_name: str = "UnknownOp"


@dataclass(slots=True)
class CoverageConfig:
    wedge_cap: int = 16
    ordered_subgraph_sizes: tuple[int, ...] = (1, 2, 3)
    unordered_subgraph_sizes: tuple[int, ...] = (1, 2, 3)
    use_suite_relative_spaces: bool = True


@dataclass(slots=True)
class SimilarityConfig:
    wl_iterations: int = 2
    simhash_bits: int = 64
    lsh_bands: int = 8
    random_seed: int = 20260425


@dataclass(slots=True)
class SchedulerConfig:
    alpha: float = 0.7
    gamma: float = 0.1
    enable_duplicate_suppression: bool = True


@dataclass(slots=True)
class ExecutionConfig:
    plan_only: bool = True
    use_metadata_oracle: bool = True
    default_failure_kind: str = "unknown"


@dataclass(slots=True)
class EvaluationConfig:
    checkpoints: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)
    use_root_causes_when_available: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    tool_name: str = "unknown"
    run_id: str = "run-0"
    suite_size: int = 100
    repetitions: int = 50
    default_strategy: str = "spark"


@dataclass(slots=True)
class SparkConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
