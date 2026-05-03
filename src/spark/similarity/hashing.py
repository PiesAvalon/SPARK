"""Hash-based compression for structural features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


def _bit_sign(feature: str, bit_index: int) -> int:
    digest = hashlib.sha256(f"{feature}:{bit_index}".encode("utf-8")).digest()
    return 1 if (digest[0] & 1) else -1


@dataclass(slots=True)
class SimHashEncoder:
    def encode(self, features: dict[str, int], bits: int) -> str:
        if bits <= 0:
            return ""
        accumulator = [0] * bits
        for feature, weight in sorted(features.items()):
            for bit_index in range(bits):
                accumulator[bit_index] += weight * _bit_sign(feature, bit_index)
        return "".join("1" if value >= 0 else "0" for value in accumulator)

    def encode_many(self, features_by_graph_id: dict[str, dict[str, int]], bits: int) -> dict[str, str]:
        return {graph_id: self.encode(features, bits) for graph_id, features in features_by_graph_id.items()}

    @staticmethod
    def hamming_distance(left: str, right: str) -> int:
        return sum(1 for left_bit, right_bit in zip(left, right) if left_bit != right_bit)

    @staticmethod
    def hamming_similarity(left: str, right: str) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        distance = SimHashEncoder.hamming_distance(left, right)
        return 1.0 - (distance / len(left))
