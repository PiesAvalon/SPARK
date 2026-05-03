"""Execution runner that delegates to tool adapters and oracles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core.suite import GraphCase
from ..core.types import ExecutionResult, TestCase
from .oracle import MetadataBugOracle, OracleRegistry
from .signatures import BugSignatureBuilder


@dataclass(slots=True)
class ExecutionRunner:
    signature_builder: BugSignatureBuilder = field(default_factory=BugSignatureBuilder)
    oracle_registry: OracleRegistry = field(default_factory=OracleRegistry)

    def __post_init__(self) -> None:
        if not self.oracle_registry.oracles:
            self.oracle_registry.default_oracle = MetadataBugOracle()

    def run_case(self, case: TestCase) -> ExecutionResult:
        oracle = self.oracle_registry.get(case.tool_name)
        result = oracle.evaluate(case)
        if result.is_bug and not result.bug_signature:
            result.bug_signature = self.signature_builder.build(result)
        return result

    def run_graph_case(self, case: GraphCase) -> ExecutionResult:
        if case.test_case is None:
            placeholder = TestCase(case_id=case.case_id, tool_name=case.tool_name, original_order=case.original_order)
            result = self.run_case(placeholder)
        else:
            result = self.run_case(case.test_case)
        return result

    def run_suite(self, cases: Sequence[TestCase]) -> list[ExecutionResult]:
        return [self.run_case(case) for case in cases]
