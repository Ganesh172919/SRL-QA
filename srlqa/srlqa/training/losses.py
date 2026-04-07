"""Standalone losses used by the RAISE-SRL-QA roadmap."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def span_cross_entropy(start_logits: Tensor, end_logits: Tensor, starts: Tensor, ends: Tensor) -> Tensor:
    ignored_index = start_logits.size(1)
    return (
        F.cross_entropy(start_logits, starts.clamp(0, ignored_index), ignore_index=ignored_index)
        + F.cross_entropy(end_logits, ends.clamp(0, ignored_index), ignore_index=ignored_index)
    )


def multi_task_srlqa_loss(
    start_logits: Tensor,
    end_logits: Tensor,
    role_logits: Tensor,
    answerable_logits: Tensor,
    starts: Tensor,
    ends: Tensor,
    roles: Tensor,
    answerable: Tensor,
    role_weight: float = 0.35,
    answerable_weight: float = 0.20,
) -> Tensor:
    span_loss = span_cross_entropy(start_logits, end_logits, starts, ends)
    role_loss = F.cross_entropy(role_logits, roles)
    answerable_loss = F.cross_entropy(answerable_logits, answerable)
    return span_loss + role_weight * role_loss + answerable_weight * answerable_loss


def pairwise_margin_ranking_loss(gold_scores: Tensor, negative_scores: Tensor, margin: float = 0.25) -> Tensor:
    target = torch.ones_like(gold_scores)
    return F.margin_ranking_loss(gold_scores, negative_scores, target, margin=margin)
