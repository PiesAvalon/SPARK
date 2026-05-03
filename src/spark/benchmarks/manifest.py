"""Benchmark metadata for dataset releases and project-backed catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ToolRunManifest:
    tool_name: str
    run_id: str
    case_root: Path
    expected_suite_size: int = 100
    frameworks: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def inline_cases(self) -> list[dict[str, object]]:
        raw = self.metadata.get("cases", [])
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    def project_root(self) -> Path | None:
        raw = self.metadata.get("project_root")
        if isinstance(raw, Path):
            return raw
        if isinstance(raw, str) and raw:
            return Path(raw)
        return None

    def prefer_project_files(self) -> bool:
        return bool(self.metadata.get("prefer_project_files", False))


@dataclass(slots=True)
class BenchmarkManifest:
    runs: list[ToolRunManifest] = field(default_factory=list)

    def add_run(self, run: ToolRunManifest) -> None:
        self.runs.append(run)

    def by_tool(self) -> dict[str, list[ToolRunManifest]]:
        grouped: dict[str, list[ToolRunManifest]] = {}
        for run in self.runs:
            grouped.setdefault(run.tool_name, []).append(run)
        return grouped
