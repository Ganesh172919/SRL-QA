"""Self-correction loop constrained to extracted spans."""

from __future__ import annotations

from typing import Sequence

from ..decoding.span_rules import SpanCandidate
from ..retrieval.propbank_index import FrameRecord
from .span_verifier import SpanVerifier, VerificationResult


class SelfCorrectionLoop:
    """Pick a verified candidate without allowing hallucinated answers."""

    def __init__(self, verifier: SpanVerifier) -> None:
        self.verifier = verifier

    def correct(
        self,
        candidates: Sequence[SpanCandidate],
        question: str,
        context: str,
        frames: Sequence[FrameRecord] | None = None,
    ) -> VerificationResult | None:
        verified = [self.verifier.verify(candidate, question, context, frames) for candidate in candidates]
        if not verified:
            return None
        accepted = [item for item in verified if item.accepted]
        pool = accepted or verified
        pool.sort(key=lambda item: (item.score, item.candidate.score), reverse=True)
        return pool[0]
