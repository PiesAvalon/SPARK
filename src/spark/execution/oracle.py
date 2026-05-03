"""Tool-specific bug oracle interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..core.types import ExecutionEnv, ExecutionResult, FailureSummary, InputFeature, TestCase


class BugOracle(Protocol):
    def evaluate(self, case: TestCase) -> ExecutionResult:
        """Execute a tool-specific oracle over one test case."""


@dataclass(slots=True)
class MetadataBugOracle:
    default_runtime_seconds: float = 1.0

    def _build_failure(self, case: TestCase) -> FailureSummary | None:
        if case.expected_failure is not None:
            return case.expected_failure

        raw = case.metadata.get("failure")
        if isinstance(raw, FailureSummary):
            return raw
        if isinstance(raw, dict):
            raw_features = raw.get("input_features", [])
            input_features = tuple(
                InputFeature(dtype=str(item.get("dtype", "unknown")), rank=int(item.get("rank", -1)))
                for item in raw_features
                if isinstance(item, dict)
            )
            env = None
            raw_env = raw.get("env")
            if isinstance(raw_env, dict):
                env = ExecutionEnv(
                    device=str(raw_env.get("device", "unknown")),
                    backend=str(raw_env.get("backend", "unknown")),
                )
            return FailureSummary(
                failure_kind=str(raw.get("failure_kind", "unknown")),
                operator_types=tuple(str(item) for item in raw.get("operator_types", [])),
                input_features=input_features,
                env=env,
                message=None if raw.get("message") is None else str(raw.get("message")),
            )

        if case.metadata.get("is_bug"):
            return FailureSummary(
                failure_kind=str(case.metadata.get("failure_kind", "unknown")),
                operator_types=tuple(str(item) for item in case.metadata.get("operator_types", [])),
                input_features=case.input_features,
                env=case.env,
                message=None if case.metadata.get("message") is None else str(case.metadata.get("message")),
            )
        return None

    def evaluate(self, case: TestCase) -> ExecutionResult:
        embedded = case.metadata.get("execution_result")
        if isinstance(embedded, ExecutionResult):
            return embedded

        failure = self._build_failure(case)
        runtime_raw = case.metadata.get("runtime_seconds", self.default_runtime_seconds)
        runtime = float(runtime_raw) if isinstance(runtime_raw, (int, float)) else self.default_runtime_seconds
        is_bug = bool(case.metadata.get("is_bug", failure is not None))
        root_causes = case.expected_root_causes
        if not root_causes:
            raw_root_causes = case.metadata.get("root_causes", [])
            if isinstance(raw_root_causes, list):
                root_causes = tuple(str(item) for item in raw_root_causes)

        return ExecutionResult(
            case_id=case.case_id,
            is_bug=is_bug,
            runtime_seconds=runtime,
            failure=failure,
            root_cause_ids=root_causes,
            raw_artifacts={"oracle": "metadata"},
        )


@dataclass(slots=True)
class OracleRegistry:
    default_oracle: BugOracle = field(default_factory=MetadataBugOracle)
    oracles: dict[str, BugOracle] = field(default_factory=dict)

    def register(self, tool_name: str, oracle: BugOracle) -> None:
        self.oracles[tool_name.lower()] = oracle

    def get(self, tool_name: str) -> BugOracle:
        return self.oracles.get(tool_name.lower(), self.default_oracle)
