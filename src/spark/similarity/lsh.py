"""Locality-sensitive hashing primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass(slots=True)
class LSHBuckets:
    bands: int
    buckets: dict[str, list[str]] = field(default_factory=dict)

    def candidate_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for members in self.buckets.values():
            for left, right in combinations(sorted(set(members)), 2):
                pairs.add((left, right))
        return pairs


@dataclass(slots=True)
class LSHBandIndex:
    def _segments(self, signature: str, bands: int) -> list[str]:
        if not signature:
            return []
        bands = max(1, bands)
        width = max(1, len(signature) // bands)
        segments: list[str] = []
        for band_index in range(bands):
            start = band_index * width
            end = len(signature) if band_index == bands - 1 else min(len(signature), start + width)
            if start >= len(signature):
                break
            segments.append(signature[start:end])
        return segments

    def bucketize(self, signatures: dict[str, str], bands: int) -> LSHBuckets:
        buckets = LSHBuckets(bands=bands)
        for graph_id, signature in signatures.items():
            for band_index, segment in enumerate(self._segments(signature, bands)):
                key = f"band:{band_index}:{segment}"
                buckets.buckets.setdefault(key, []).append(graph_id)
        return buckets
