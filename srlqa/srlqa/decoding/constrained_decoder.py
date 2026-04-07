"""Constrained span decoder for MRC start/end logits."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..config import DecodingConfig
from .span_rules import SpanCandidate, apply_span_rules, span_text


class ConstrainedDecoder:
    def __init__(self, config: DecodingConfig) -> None:
        self.config = config

    def decode(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        context_tokens: Sequence[str],
        role: str,
        question_type: str,
        predicate_index: int | None = None,
    ) -> list[SpanCandidate]:
        start_scores, start_indices = torch.topk(start_logits, k=min(self.config.top_k, start_logits.numel()))
        end_scores, end_indices = torch.topk(end_logits, k=min(self.config.top_k, end_logits.numel()))
        candidates: list[SpanCandidate] = []
        for start_score, start in zip(start_scores.tolist(), start_indices.tolist(), strict=False):
            for end_score, end in zip(end_scores.tolist(), end_indices.tolist(), strict=False):
                if start > end:
                    continue
                if end - start + 1 > self.config.max_span_length:
                    continue
                if end >= len(context_tokens):
                    continue
                score = float(start_score + end_score - self.config.length_penalty * (end - start + 1))
                candidate = SpanCandidate(
                    text=span_text(context_tokens, start, end),
                    start_token=start,
                    end_token=end,
                    role=role,
                    score=score,
                )
                filtered = apply_span_rules(
                    candidate,
                    context_tokens=context_tokens,
                    question_type=question_type,
                    predicate_index=predicate_index,
                    max_span_length=self.config.max_span_length,
                )
                if filtered is not None:
                    candidates.append(filtered)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.config.top_k]
