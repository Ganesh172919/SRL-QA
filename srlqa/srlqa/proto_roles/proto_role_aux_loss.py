"""Auxiliary proto-role loss."""

from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def proto_role_aux_loss(logits: Tensor, labels: Tensor, weight: float = 0.15) -> Tensor:
    return weight * F.binary_cross_entropy_with_logits(logits, labels.float())
