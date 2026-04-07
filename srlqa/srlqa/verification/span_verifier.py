"""Evidence-based verifier for candidate SRL-QA spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..decoding.span_rules import SpanCandidate
from ..retrieval.propbank_index import FrameRecord


@dataclass(slots=True)
class VerificationResult:
    candidate: SpanCandidate
    score: float
    accepted: bool
    reasons: list[str]


class SpanVerifier:
    """Conservative verifier that can only score extracted candidates."""

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold

    @staticmethod
    def _frame_role_compatible(role: str, frames: Sequence[FrameRecord]) -> bool:
        if not frames:
            return True
        frame_roles = {definition.role for frame in frames for definition in frame.roles}
        return role in frame_roles or role.startswith("ARGM")

    def verify(
        self,
        candidate: SpanCandidate,
        question: str,
        context: str,
        frames: Sequence[FrameRecord] | None = None,
    ) -> VerificationResult:
        reasons: list[str] = []
        score = 0.50
        normalized = candidate.text.strip().lower()
        if normalized and normalized in context.lower():
            score += 0.20
            reasons.append("candidate_is_extractable")
        if self._frame_role_compatible(candidate.role, frames or []):
            score += 0.15
            reasons.append("role_frame_compatible")
        else:
            score -= 0.25
            reasons.append("role_frame_mismatch")
        q = question.lower()
        if "where" in q and candidate.role == "ARGM-LOC":
            score += 0.10
        if "when" in q and candidate.role == "ARGM-TMP":
            score += 0.10
        if "why" in q and candidate.role in {"ARGM-CAU", "ARGM-PRP", "ARGM-PNC"}:
            score += 0.10
        if "who" in q and candidate.role in {"ARG0", "ARG2"}:
            score += 0.05
        score += min(max(candidate.score, -1.0), 1.0) * 0.05
        score = max(0.0, min(1.0, score))
        return VerificationResult(
            candidate=candidate,
            score=score,
            accepted=score >= self.threshold,
            reasons=reasons,
        )
