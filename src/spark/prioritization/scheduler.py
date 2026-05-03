"""Adaptive scheduler state for SPARK."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Callable, Sequence

from ..core.suite import Cluster, GraphCase, PrioritizedSuite
from ..core.types import ExecutionResult


@dataclass(slots=True)
class ClusterState:
    cluster_id: str
    representative_id: str
    member_ids: list[str] = field(default_factory=list)
    mode: str = "unexplored"
    pointer: int = 0
    bug_count: int = 0
    factor: float = 1.0
    version: int = 0

    def current_member(self) -> str | None:
        if 0 <= self.pointer < len(self.member_ids):
            return self.member_ids[self.pointer]
        return None

    def remaining_suffix(self) -> list[str]:
        return list(self.member_ids[self.pointer :])


@dataclass(slots=True)
class AdaptiveScheduler:
    alpha: float = 0.7
    gamma: float = 0.1
    case_lookup: dict[str, GraphCase] = field(default_factory=dict, init=False)
    cluster_states: dict[str, ClusterState] = field(default_factory=dict, init=False)
    cluster_descriptors: list[Cluster] = field(default_factory=list, init=False)
    representative_heap: list[tuple[float, str]] = field(default_factory=list, init=False)
    active_heap: list[tuple[float, str, int]] = field(default_factory=list, init=False)
    ordered_case_ids: list[str] = field(default_factory=list, init=False)
    executed_case_ids: set[str] = field(default_factory=set, init=False)
    observed_bug_signatures: set[str] = field(default_factory=set, init=False)
    last_selection: tuple[str, str, str] | None = field(default=None, init=False)

    def initialize(self, suite: Sequence[GraphCase]) -> None:
        self.case_lookup = {case.case_id: case for case in suite}
        self.cluster_states.clear()
        self.cluster_descriptors.clear()
        self.representative_heap.clear()
        self.active_heap.clear()
        self.ordered_case_ids.clear()
        self.executed_case_ids.clear()
        self.observed_bug_signatures.clear()
        self.last_selection = None

        grouped: dict[str, list[GraphCase]] = {}
        for case in suite:
            cluster_id = case.cluster_id or f"singleton:{case.case_id}"
            grouped.setdefault(cluster_id, []).append(case)

        for cluster_id, cases in sorted(grouped.items()):
            ranked = sorted(cases, key=lambda item: (-item.static_score(), item.original_order, item.case_id))
            representative = ranked[0]
            members = ranked[1:]
            state = ClusterState(
                cluster_id=cluster_id,
                representative_id=representative.case_id,
                member_ids=[case.case_id for case in members],
            )
            self.cluster_states[cluster_id] = state
            self.cluster_descriptors.append(
                Cluster(
                    cluster_id=cluster_id,
                    case_ids=[case.case_id for case in ranked],
                    representative_id=representative.case_id,
                    member_ids=[case.case_id for case in members],
                )
            )
            heapq.heappush(self.representative_heap, (-representative.static_score(), cluster_id))

    def _case_score(self, case_id: str) -> float:
        return self.case_lookup[case_id].static_score()

    def _push_active(self, cluster_id: str) -> None:
        state = self.cluster_states[cluster_id]
        current = state.current_member()
        if current is None:
            return
        state.version += 1
        priority = state.factor * self._case_score(current)
        heapq.heappush(self.active_heap, (-priority, cluster_id, state.version))

    def _peek_representative(self) -> tuple[str, str, float] | None:
        while self.representative_heap:
            _, cluster_id = self.representative_heap[0]
            state = self.cluster_states[cluster_id]
            if state.mode != "unexplored" or state.representative_id in self.executed_case_ids:
                heapq.heappop(self.representative_heap)
                continue
            priority = self._case_score(state.representative_id) + (self.gamma * len(self.observed_bug_signatures))
            return (cluster_id, state.representative_id, priority)
        return None

    def _peek_active(self) -> tuple[str, str, float] | None:
        while self.active_heap:
            neg_priority, cluster_id, version = self.active_heap[0]
            state = self.cluster_states[cluster_id]
            current = state.current_member()
            if state.mode != "active" or current is None or version != state.version:
                heapq.heappop(self.active_heap)
                continue
            return (cluster_id, current, -neg_priority)
        return None

    def has_online_work(self) -> bool:
        return self._peek_representative() is not None or self._peek_active() is not None

    def select_next(self) -> str:
        rep_candidate = self._peek_representative()
        active_candidate = self._peek_active()
        if rep_candidate is None and active_candidate is None:
            raise RuntimeError("No remaining online work.")

        if active_candidate is None or (rep_candidate is not None and rep_candidate[2] >= active_candidate[2]):
            cluster_id, case_id, _ = rep_candidate  # type: ignore[misc]
            heapq.heappop(self.representative_heap)
            kind = "rep"
        else:
            cluster_id, case_id, _ = active_candidate  # type: ignore[misc]
            heapq.heappop(self.active_heap)
            kind = "member"

        self.ordered_case_ids.append(case_id)
        self.executed_case_ids.add(case_id)
        self.last_selection = (cluster_id, case_id, kind)
        return case_id

    def _classify_bug(self, result: ExecutionResult) -> tuple[str, str | None]:
        if not result.is_bug:
            return ("none", None)
        signature = result.bug_signature or f"bug:{result.case_id}"
        if signature in self.observed_bug_signatures:
            return ("dup", signature)
        return ("new", signature)

    def update(self, result: ExecutionResult) -> None:
        if self.last_selection is None:
            raise RuntimeError("update() called before select_next().")

        cluster_id, case_id, kind = self.last_selection
        if result.case_id != case_id:
            raise ValueError(f"Execution result mismatch: expected {case_id}, got {result.case_id}.")

        state = self.cluster_states[cluster_id]
        bug_state, signature = self._classify_bug(result)

        if kind == "rep":
            if bug_state == "none":
                if state.member_ids:
                    state.mode = "active"
                    state.factor = self.gamma
                    self._push_active(cluster_id)
                else:
                    state.mode = "finished"
            else:
                state.mode = "suppressed"
                if bug_state == "new":
                    state.bug_count += 1
                    state.factor = self.alpha ** state.bug_count
                    if signature is not None:
                        self.observed_bug_signatures.add(signature)
                else:
                    state.factor = self.alpha ** max(2, state.bug_count)
        else:
            state.pointer += 1
            if bug_state == "none":
                if state.current_member() is not None:
                    state.mode = "active"
                    self._push_active(cluster_id)
                else:
                    state.mode = "finished"
            else:
                state.mode = "suppressed"
                if bug_state == "new":
                    state.bug_count += 1
                    state.factor = self.alpha ** state.bug_count
                    if signature is not None:
                        self.observed_bug_signatures.add(signature)
                else:
                    state.factor = self.alpha ** max(2, state.bug_count)
                state.version += 1

        self.last_selection = None

    def _tail_merge(self) -> list[str]:
        suffixes = {
            cluster_id: state.remaining_suffix()
            for cluster_id, state in self.cluster_states.items()
            if state.mode == "suppressed" and state.remaining_suffix()
        }
        heap: list[tuple[float, str, int]] = []
        for cluster_id, remaining in suffixes.items():
            state = self.cluster_states[cluster_id]
            first_case_id = remaining[0]
            heapq.heappush(heap, (-(state.factor * self._case_score(first_case_id)), cluster_id, 0))

        order: list[str] = []
        while heap:
            _, cluster_id, index = heapq.heappop(heap)
            remaining = suffixes[cluster_id]
            case_id = remaining[index]
            order.append(case_id)
            next_index = index + 1
            if next_index < len(remaining):
                next_case_id = remaining[next_index]
                state = self.cluster_states[cluster_id]
                heapq.heappush(heap, (-(state.factor * self._case_score(next_case_id)), cluster_id, next_index))
        return order

    def finalize(self) -> PrioritizedSuite:
        tail_order = self._tail_merge()
        complete_order = list(self.ordered_case_ids) + [case_id for case_id in tail_order if case_id not in self.executed_case_ids]
        return PrioritizedSuite(
            strategy_name="spark",
            ordered_case_ids=complete_order,
            case_lookup=self.case_lookup,
            clusters=self.cluster_descriptors,
            metadata={"bug_signatures": sorted(self.observed_bug_signatures)},
        )

    def plan_without_feedback(self, suite: Sequence[GraphCase]) -> PrioritizedSuite:
        self.initialize(suite)
        representatives = sorted(
            (state.representative_id for state in self.cluster_states.values()),
            key=lambda case_id: (-self._case_score(case_id), self.case_lookup[case_id].original_order, case_id),
        )
        residuals = {state.cluster_id: list(state.member_ids) for state in self.cluster_states.values() if state.member_ids}
        heap: list[tuple[float, str, int]] = []
        for cluster_id, member_ids in residuals.items():
            first_case_id = member_ids[0]
            heapq.heappush(heap, (-(self.gamma * self._case_score(first_case_id)), cluster_id, 0))

        members_order: list[str] = []
        while heap:
            _, cluster_id, index = heapq.heappop(heap)
            member_ids = residuals[cluster_id]
            members_order.append(member_ids[index])
            next_index = index + 1
            if next_index < len(member_ids):
                next_case_id = member_ids[next_index]
                heapq.heappush(heap, (-(self.gamma * self._case_score(next_case_id)), cluster_id, next_index))

        return PrioritizedSuite(
            strategy_name="spark",
            ordered_case_ids=representatives + members_order,
            case_lookup=self.case_lookup,
            clusters=self.cluster_descriptors,
            metadata={"mode": "offline-plan"},
        )

    def run(
        self,
        suite: Sequence[GraphCase],
        execute: Callable[[GraphCase], ExecutionResult],
    ) -> PrioritizedSuite:
        self.initialize(suite)
        results: list[ExecutionResult] = []
        while self.has_online_work():
            case_id = self.select_next()
            result = execute(self.case_lookup[case_id])
            results.append(result)
            self.update(result)
        prioritized = self.finalize()
        for case_id in prioritized.ordered_case_ids:
            if case_id in self.executed_case_ids:
                continue
            results.append(execute(self.case_lookup[case_id]))
        prioritized.attach_results(results)
        return prioritized
