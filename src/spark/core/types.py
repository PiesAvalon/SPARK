"""Shared types used across the SPARK dataset release package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InputFeature:
    dtype: str = "unknown"
    rank: int = -1

    def as_key(self) -> tuple[str, int]:
        return (self.dtype, self.rank)


@dataclass(frozen=True, slots=True)
class ExecutionEnv:
    device: str = "unknown"
    backend: str = "unknown"

    def as_key(self) -> tuple[str, str]:
        return (self.device, self.backend)


@dataclass(frozen=True, slots=True)
class FailureSummary:
    failure_kind: str
    operator_types: tuple[str, ...] = ()
    input_features: tuple[InputFeature, ...] = ()
    env: ExecutionEnv | None = None
    message: str | None = None

    def canonical_operator_types(self) -> tuple[str, ...]:
        return tuple(sorted(self.operator_types))

    def canonical_input_features(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(feature.as_key() for feature in self.input_features))


@dataclass(slots=True)
class TestCase:
    case_id: str
    tool_name: str
    source_path: Path | None = None
    original_order: int = 0
    graph_payload: dict[str, object] | None = None
    operator_sequence: tuple[str, ...] = ()
    input_features: tuple[InputFeature, ...] = ()
    env: ExecutionEnv | None = None
    expected_failure: FailureSummary | None = None
    expected_root_causes: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object] | None:
        if self.graph_payload is not None:
            return self.graph_payload
        payload = self.metadata.get("graph_payload")
        if isinstance(payload, dict):
            return payload
        return None


@dataclass(slots=True)
class ExecutionResult:
    case_id: str
    is_bug: bool
    runtime_seconds: float | None = None
    failure: FailureSummary | None = None
    bug_signature: str | None = None
    root_cause_ids: tuple[str, ...] = ()
    strategy_name: str | None = None
    raw_artifacts: dict[str, object] = field(default_factory=dict)

    def normalized_bug_ids(self, use_root_causes: bool = True) -> tuple[str, ...]:
        if not self.is_bug:
            return ()
        if use_root_causes and self.root_cause_ids:
            return tuple(sorted(set(self.root_cause_ids)))
        if self.bug_signature:
            return (self.bug_signature,)
        return (self.case_id,)
