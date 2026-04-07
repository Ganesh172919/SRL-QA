"""Span validation rules for constrained SRL-QA decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .role_priors import length_penalty


TEMPORAL_MARKERS = {
    "today",
    "yesterday",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "noon",
    "midnight",
    "morning",
    "evening",
    "afternoon",
    "night",
}
LOCATION_PREPOSITIONS = {"in", "at", "near", "inside", "outside", "through", "across", "into", "onto", "under", "over", "to"}
CAUSE_MARKERS = {"because", "since", "as", "due"}


@dataclass(slots=True)
class SpanCandidate:
    text: str
    start_token: int
    end_token: int
    role: str
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end_token - self.start_token + 1


def span_text(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[start : end + 1]).strip()


def violates_question_boundary(candidate: SpanCandidate, question_type: str) -> bool:
    tokens = candidate.text.lower().split()
    if question_type == "WHERE" and any(token in TEMPORAL_MARKERS for token in tokens):
        return True
    if question_type == "WHEN" and any(token in LOCATION_PREPOSITIONS for token in tokens) and not any(token in TEMPORAL_MARKERS for token in tokens):
        return True
    if question_type == "WHY" and tokens and tokens[0] not in CAUSE_MARKERS:
        return False
    return False


def apply_span_rules(
    candidate: SpanCandidate,
    context_tokens: Sequence[str],
    question_type: str,
    predicate_index: int | None = None,
    max_span_length: int = 18,
) -> SpanCandidate | None:
    if candidate.start_token < 0 or candidate.end_token >= len(context_tokens):
        return None
    if candidate.start_token > candidate.end_token:
        return None
    if candidate.length > max_span_length:
        return None
    if predicate_index is not None and candidate.start_token <= predicate_index <= candidate.end_token:
        candidate.reasons.append("predicate_inside_span_penalty")
        candidate.score -= 0.20
    if violates_question_boundary(candidate, question_type):
        return None
    penalty = length_penalty(candidate.role, candidate.length)
    if penalty:
        candidate.reasons.append(f"length_penalty={penalty:.3f}")
        candidate.score -= penalty
    return candidate
