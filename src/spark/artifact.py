"""Dataset-facing helpers for loading release bundles and project-backed integrations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Sequence

from .benchmarks.adapters import AdapterRegistry
from .benchmarks.manifest import BenchmarkManifest, ToolRunManifest
from .config import SparkConfig
from .pipeline import SparkPipeline
from .prioritization.scheduler import AdaptiveScheduler

RELEASE_NAME = "SPARK-benchmark-release-v1"

TOOL_PROJECTS: dict[str, str] = {
    "COMET": "COMET",
    "DevMuT": "DevMuT",
    "Gandalf": "Gandalf",
    "GenCoG": "GenCoG",
    "LEMON": "LEMON",
    "ModelMeta": "ModelMeta",
    "Muffin": "Muffin",
    "NEURI": "neuri-artifact",
    "NNSmith": "nnsmith",
}

TOOL_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "COMET": {"frameworks": ("PyTorch",), "backends": ("ONNXRuntime",)},
    "DevMuT": {"frameworks": ("PyTorch",), "backends": ("MindSpore",)},
    "Gandalf": {"frameworks": ("TensorFlow",), "backends": ("TFLite",)},
    "GenCoG": {"frameworks": ("TVM Relay",), "backends": ("TVM",)},
    "LEMON": {"frameworks": ("PyTorch",), "backends": ("TensorFlow",)},
    "ModelMeta": {"frameworks": ("PyTorch", "MindSpore"), "backends": ("CUDA",)},
    "Muffin": {"frameworks": ("PyTorch",), "backends": ("TensorFlow",)},
    "NEURI": {"frameworks": ("PyTorch", "TensorFlow"), "backends": ("TensorRT",)},
    "NNSmith": {"frameworks": ("ONNX",), "backends": ("TVM",)},
}

DEFAULT_STRATEGIES: tuple[str, ...] = (
    "spark",
    "original-order",
    "random",
    "coverage-only",
    "coverage-greedy",
    "structure-only",
    "pairwise-greedy",
)


@dataclass(slots=True)
class StrategyRunRecord:
    tool_name: str
    run_id: str
    strategy: str
    metrics: dict[str, float] = field(default_factory=dict)
    ordered_case_ids: list[str] = field(default_factory=list)
    root_cause_ids: list[str] = field(default_factory=list)
    source_file: str | None = None

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "tool_name": self.tool_name,
            "run_id": self.run_id,
            "strategy": self.strategy,
            "source_file": self.source_file or "",
            "num_cases": len(self.ordered_case_ids),
            "num_root_causes": len(self.root_cause_ids),
        }
        row.update(self.metrics)
        return row


def _coerce_case(raw_case: dict[str, object], run_payload: dict[str, object]) -> dict[str, object]:
    case = dict(raw_case)
    device = str(case.get("device", run_payload.get("device", "unknown")))
    backend = str(case.get("backend", run_payload.get("backend", "unknown")))
    case["env"] = {"device": device, "backend": backend}

    runtime_ms = case.get("runtime_ms")
    if isinstance(runtime_ms, (int, float)) and "runtime_seconds" not in case:
        case["runtime_seconds"] = float(runtime_ms) / 1000.0

    if case.get("is_bug"):
        case.setdefault(
            "failure",
            {
                "failure_kind": str(case.get("failure_kind", "unknown")),
                "message": None if case.get("result_summary") is None else str(case.get("result_summary")),
                "env": {"device": device, "backend": backend},
            },
        )
        bug_report_id = case.get("bug_report_id")
        if bug_report_id not in {None, ""} and "root_causes" not in case:
            case["root_causes"] = [str(bug_report_id)]

    return case


def _tool_project_root(tool_name: str, tools_root: Path) -> Path | None:
    project_dir = TOOL_PROJECTS.get(tool_name)
    if project_dir is None:
        return None
    candidate = (tools_root / project_dir).resolve()
    return candidate if candidate.exists() else None


def discover_benchmark_dataset(dataset_root: Path, tools_root: Path = Path("tools")) -> BenchmarkManifest:
    dataset_root = dataset_root.resolve()
    tools_root = tools_root.resolve()
    manifest = BenchmarkManifest()
    for path in sorted(dataset_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        raw_cases = payload.get("cases", [])
        cases = [_coerce_case(case, payload) for case in raw_cases if isinstance(case, dict)]
        tool_name = str(payload.get("tool_name", path.stem))
        framework = payload.get("framework")
        backend = payload.get("backend")
        project_root = _tool_project_root(tool_name, tools_root)
        manifest.add_run(
            ToolRunManifest(
                tool_name=tool_name,
                run_id=str(payload.get("run_id", path.stem)),
                case_root=dataset_root,
                expected_suite_size=len(cases),
                frameworks=(str(framework),) if framework is not None else (),
                backends=(str(backend),) if backend is not None else (),
                metadata={
                    "cases": cases,
                    "source_file": str(path),
                    "dataset_kind": "benchmark-release",
                    "release_name": RELEASE_NAME,
                    "schema_version": payload.get("schema_version"),
                    "exported_at": payload.get("exported_at"),
                    "project_root": str(project_root) if project_root is not None else None,
                    "prefer_project_files": False,
                },
            )
        )
    return manifest


def discover_repository_projects(tools_root: Path = Path("tools"), expected_suite_size: int = 100) -> BenchmarkManifest:
    tools_root = tools_root.resolve()
    manifest = BenchmarkManifest()
    for tool_name, project_dir in sorted(TOOL_PROJECTS.items()):
        project_root = tools_root / project_dir
        hints = TOOL_HINTS.get(tool_name, {})
        manifest.add_run(
            ToolRunManifest(
                tool_name=tool_name,
                run_id=f"{tool_name.lower()}-project-catalog",
                case_root=project_root,
                expected_suite_size=expected_suite_size,
                frameworks=hints.get("frameworks", ()),
                backends=hints.get("backends", ()),
                metadata={
                    "dataset_kind": "tool-project-catalog",
                    "release_name": RELEASE_NAME,
                    "project_root": str(project_root),
                    "prefer_project_files": True,
                },
            )
        )
    return manifest


def _default_pipeline(tool_name: str, run_id: str) -> SparkPipeline:
    config = SparkConfig()
    config.execution.plan_only = False
    config.experiment.tool_name = tool_name
    config.experiment.run_id = run_id
    return SparkPipeline(config=config)


def _run_one_strategy(pipeline: SparkPipeline, run: ToolRunManifest, strategy: str) -> StrategyRunRecord:
    adapter = AdapterRegistry().create(run)
    cases = list(adapter.discover_cases())
    prepared = pipeline.prepare_suite(cases)

    if strategy == "spark":
        scheduler = AdaptiveScheduler(
            alpha=pipeline.config.scheduler.alpha,
            gamma=pipeline.config.scheduler.gamma,
        )
        prioritized = scheduler.run(prepared.cases, pipeline.execution_runner.run_graph_case)
        metrics = pipeline.evaluate(prioritized, prioritized.execution_results)
    else:
        prioritized = pipeline.prioritize(prepared, strategy)
        results = pipeline.execute(prioritized)
        metrics = pipeline.evaluate(prioritized, results)

    root_cause_ids = sorted(
        {
            bug_id
            for result in prioritized.execution_results
            for bug_id in result.normalized_bug_ids(
                use_root_causes=pipeline.config.evaluation.use_root_causes_when_available
            )
        }
    )
    return StrategyRunRecord(
        tool_name=run.tool_name,
        run_id=run.run_id,
        strategy=strategy,
        metrics=metrics,
        ordered_case_ids=list(prioritized.ordered_case_ids),
        root_cause_ids=root_cause_ids,
        source_file=str(run.metadata.get("source_file")) if run.metadata.get("source_file") is not None else None,
    )


def _aggregate(records: Sequence[StrategyRunRecord]) -> dict[str, dict[str, float]]:
    by_strategy: dict[str, list[StrategyRunRecord]] = {}
    for record in records:
        by_strategy.setdefault(record.strategy, []).append(record)

    aggregated: dict[str, dict[str, float]] = {}
    for strategy, items in sorted(by_strategy.items()):
        metric_names = sorted({name for item in items for name in item.metrics})
        aggregated[strategy] = {
            metric_name: mean(item.metrics.get(metric_name, 0.0) for item in items)
            for metric_name in metric_names
        }
    return aggregated


def _write_metrics_csv(path: Path, records: Sequence[StrategyRunRecord]) -> None:
    fieldnames = sorted({key for record in records for key in record.as_row()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())


def _write_order_exports(output_dir: Path, records: Sequence[StrategyRunRecord]) -> None:
    orders_dir = output_dir / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        payload = {
            "tool_name": record.tool_name,
            "run_id": record.run_id,
            "strategy": record.strategy,
            "metrics": record.metrics,
            "root_cause_ids": record.root_cause_ids,
            "ordered_case_ids": record.ordered_case_ids,
            "top_10_case_ids": record.ordered_case_ids[:10],
            "source_file": record.source_file,
        }
        filename = f"{record.tool_name.lower()}__{record.strategy}.json"
        (orders_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def export_benchmark_results(
    dataset_root: Path,
    output_dir: Path,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    tools_root: Path = Path("tools"),
) -> dict[str, object]:
    manifest = discover_benchmark_dataset(dataset_root, tools_root=tools_root)
    if not manifest.runs:
        raise FileNotFoundError(f"No benchmark dataset JSON files were found under {dataset_root}.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[StrategyRunRecord] = []
    for run in manifest.runs:
        pipeline = _default_pipeline(tool_name=run.tool_name, run_id=run.run_id)
        for strategy in strategies:
            records.append(_run_one_strategy(pipeline, run, strategy))

    aggregated = _aggregate(records)
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "output_dir": str(output_dir),
        "num_runs": len(manifest.runs),
        "num_strategies": len(tuple(strategies)),
        "strategies": list(strategies),
        "release_name": RELEASE_NAME,
        "aggregate_metrics": aggregated,
        "records": [record.as_row() for record in records],
    }

    _write_metrics_csv(output_dir / "metrics.csv", records)
    _write_order_exports(output_dir, records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def format_summary(summary: dict[str, object]) -> str:
    aggregate_metrics = summary.get("aggregate_metrics", {})
    lines = [
        f"Dataset: {summary.get('dataset_root', '')}",
        f"Runs: {summary.get('num_runs', 0)}",
        f"Strategies: {', '.join(summary.get('strategies', []))}",
        "",
        "Mean APFD / APFDc by strategy:",
    ]
    if isinstance(aggregate_metrics, dict):
        for strategy, metrics in sorted(aggregate_metrics.items()):
            if not isinstance(metrics, dict):
                continue
            apfd = float(metrics.get("apfd", 0.0))
            apfdc = float(metrics.get("apfdc", 0.0))
            lines.append(f"  {strategy:16s} apfd={apfd:.4f}  apfdc={apfdc:.4f}")
    return "\n".join(lines)
