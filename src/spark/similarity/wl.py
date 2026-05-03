"""Weisfeiler-Lehman feature extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from math import sqrt
from typing import Iterable

from ..graph.schema import ComputationGraph


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class WLFeatureExtractor:
    def extract_one(self, graph: ComputationGraph, iterations: int) -> dict[str, int]:
        pred = graph.predecessors_map()
        succ = graph.successors_map()
        colors = {node_id: graph.nodes[node_id].label.as_key() for node_id in graph.nodes}
        features: Counter[str] = Counter(_stable_hash(repr(value)) for value in colors.values())

        for _ in range(iterations):
            next_colors: dict[str, str] = {}
            for node_id in sorted(graph.nodes):
                pred_colors = sorted(repr(colors[parent]) for parent in pred[node_id])
                succ_colors = sorted(repr(colors[child]) for child in succ[node_id])
                serialized = repr((colors[node_id], tuple(pred_colors), tuple(succ_colors)))
                next_colors[node_id] = _stable_hash(serialized)
            colors = next_colors
            features.update(colors.values())

        return dict(features)

    def extract_many(self, graphs: Iterable[ComputationGraph], iterations: int) -> dict[str, dict[str, int]]:
        return {graph.graph_id: self.extract_one(graph, iterations) for graph in graphs}

    @staticmethod
    def cosine_similarity(left: dict[str, int], right: dict[str, int]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        dot = float(sum(left[key] * right[key] for key in shared))
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
