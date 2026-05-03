"""Lightweight bug signature construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ..core.types import ExecutionResult


@dataclass(slots=True)
class BugSignatureBuilder:
    null_token: str = "__null__"

    def build(self, result: ExecutionResult) -> str | None:
        if not result.is_bug:
            return None

        if result.failure is None:
            payload = ("bug", result.case_id)
        else:
            failure = result.failure
            payload = (
                failure.failure_kind or self.null_token,
                failure.canonical_operator_types() or (self.null_token,),
                failure.canonical_input_features() or ((self.null_token, -1),),
                failure.env.as_key() if failure.env is not None else (self.null_token, self.null_token),
            )

        digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        return digest[:24]
