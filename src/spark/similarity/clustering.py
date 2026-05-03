"""Approximate clustering over structural signatures."""

from __future__ import annotations

from dataclasses import dataclass, field

from .hashing import SimHashEncoder
from .lsh import LSHBandIndex, LSHBuckets
from .wl import WLFeatureExtractor


@dataclass(slots=True)
class StructuralClusters:
    cluster_members: dict[str, list[str]] = field(default_factory=dict)
    case_to_cluster: dict[str, str] = field(default_factory=dict)
    buckets: LSHBuckets | None = None


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(slots=True)
class StructuralClusterer:
    band_index: LSHBandIndex = field(default_factory=LSHBandIndex)

    def cluster(
        self,
        wl_features: dict[str, dict[str, int]],
        signatures: dict[str, str],
        bands: int = 8,
    ) -> StructuralClusters:
        graph_ids = sorted(set(wl_features) | set(signatures))
        union_find = _UnionFind(graph_ids)
        buckets = self.band_index.bucketize(signatures, bands)
        for left, right in buckets.candidate_pairs():
            union_find.union(left, right)

        members_by_root: dict[str, list[str]] = {}
        for graph_id in graph_ids:
            root = union_find.find(graph_id)
            members_by_root.setdefault(root, []).append(graph_id)

        cluster_members: dict[str, list[str]] = {}
        case_to_cluster: dict[str, str] = {}
        for index, (_, members) in enumerate(sorted(members_by_root.items(), key=lambda item: (len(item[1]) * -1, item[0]))):
            cluster_id = f"cluster-{index:04d}"
            cluster_members[cluster_id] = sorted(members)
            for graph_id in members:
                case_to_cluster[graph_id] = cluster_id

        return StructuralClusters(cluster_members=cluster_members, case_to_cluster=case_to_cluster, buckets=buckets)

    def quality_statistics(
        self,
        wl_features: dict[str, dict[str, int]],
        signatures: dict[str, str],
        clusters: StructuralClusters,
    ) -> dict[str, float]:
        wl_extractor = WLFeatureExtractor()
        simhash = SimHashEncoder()
        intra_wl_total = 0.0
        intra_wl_count = 0
        inter_wl_total = 0.0
        inter_wl_count = 0
        intra_sh_total = 0.0
        intra_sh_count = 0
        inter_sh_total = 0.0
        inter_sh_count = 0

        cluster_lookup = clusters.case_to_cluster
        graph_ids = sorted(set(wl_features) | set(signatures))
        for left_index, left_id in enumerate(graph_ids):
            for right_id in graph_ids[left_index + 1 :]:
                same_cluster = cluster_lookup.get(left_id) == cluster_lookup.get(right_id)
                wl_similarity = wl_extractor.cosine_similarity(wl_features.get(left_id, {}), wl_features.get(right_id, {}))
                sh_similarity = simhash.hamming_similarity(signatures.get(left_id, ""), signatures.get(right_id, ""))
                if same_cluster:
                    intra_wl_total += wl_similarity
                    intra_wl_count += 1
                    intra_sh_total += sh_similarity
                    intra_sh_count += 1
                else:
                    inter_wl_total += wl_similarity
                    inter_wl_count += 1
                    inter_sh_total += sh_similarity
                    inter_sh_count += 1

        return {
            "wl_intra": 0.0 if intra_wl_count == 0 else intra_wl_total / intra_wl_count,
            "wl_inter": 0.0 if inter_wl_count == 0 else inter_wl_total / inter_wl_count,
            "sh_intra": 0.0 if intra_sh_count == 0 else intra_sh_total / intra_sh_count,
            "sh_inter": 0.0 if inter_sh_count == 0 else inter_sh_total / inter_sh_count,
        }
