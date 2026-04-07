"""Ranking loss helpers for hard-negative span discrimination."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def gold_over_negative_loss(gold_score: Tensor, negative_score: Tensor, margin: float = 0.25) -> Tensor:
    return F.relu(margin - gold_score + negative_score).mean()


def batched_gold_over_negatives(gold_scores: Tensor, negative_scores: Tensor, margin: float = 0.25) -> Tensor:
    if negative_scores.ndim == 1:
        negative_scores = negative_scores.unsqueeze(0)
    expanded_gold = gold_scores.unsqueeze(-1).expand_as(negative_scores)
    return torch.relu(margin - expanded_gold + negative_scores).mean()
