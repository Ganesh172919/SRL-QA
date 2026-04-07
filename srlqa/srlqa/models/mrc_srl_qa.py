"""Multi-task MRC encoder for SRL question answering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..config import ProjectConfig


@dataclass(slots=True)
class MrcSrlQaOutput:
    """Structured model output."""

    loss: Tensor | None
    start_logits: Tensor
    end_logits: Tensor
    role_logits: Tensor
    answerable_logits: Tensor
    boundary_logits: Tensor


class MrcSrlQaModel(nn.Module):
    """Transformer encoder with SRL-QA heads.

    Heads:
    - start/end span extraction
    - role classification
    - answerability classification
    - boundary quality score
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, role_labels: list[str], dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.role_labels = role_labels
        self.dropout = nn.Dropout(dropout)
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)
        self.role_head = nn.Linear(hidden_size, len(role_labels))
        self.answerable_head = nn.Linear(hidden_size, 2)
        self.boundary_head = nn.Linear(hidden_size * 2, 1)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "MrcSrlQaModel":
        """Download/load the configured encoder through Transformers."""

        encoder_config = AutoConfig.from_pretrained(
            config.model.encoder_name,
            cache_dir=str(config.paths.hf_cache_dir / "models"),
        )
        encoder = AutoModel.from_pretrained(
            config.model.encoder_name,
            config=encoder_config,
            cache_dir=str(config.paths.hf_cache_dir / "models"),
        )
        return cls(
            encoder=encoder,
            hidden_size=int(encoder_config.hidden_size),
            role_labels=list(config.model.role_labels),
            dropout=config.model.dropout,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        start_positions: Tensor | None = None,
        end_positions: Tensor | None = None,
        role_labels: Tensor | None = None,
        answerable_labels: Tensor | None = None,
        boundary_labels: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor | None]:
        encoder_inputs: dict[str, Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        sequence = self.dropout(outputs.last_hidden_state)
        pooled = sequence[:, 0]

        start_logits = self.start_head(sequence).squeeze(-1)
        end_logits = self.end_head(sequence).squeeze(-1)
        role_logits = self.role_head(pooled)
        answerable_logits = self.answerable_head(pooled)

        if start_positions is not None and end_positions is not None:
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            start_states = sequence[batch_indices, start_positions.clamp_min(0)]
            end_states = sequence[batch_indices, end_positions.clamp_min(0)]
            boundary_inputs = torch.cat([start_states, end_states], dim=-1)
        else:
            max_start = start_logits.argmax(dim=-1)
            max_end = end_logits.argmax(dim=-1)
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            boundary_inputs = torch.cat([sequence[batch_indices, max_start], sequence[batch_indices, max_end]], dim=-1)
        boundary_logits = self.boundary_head(boundary_inputs).squeeze(-1)

        loss: Tensor | None = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_targets = start_positions.clamp(0, ignored_index)
            end_targets = end_positions.clamp(0, ignored_index)
            losses = [
                F.cross_entropy(start_logits, start_targets, ignore_index=ignored_index),
                F.cross_entropy(end_logits, end_targets, ignore_index=ignored_index),
            ]
            if role_labels is not None:
                losses.append(F.cross_entropy(role_logits, role_labels))
            if answerable_labels is not None:
                losses.append(F.cross_entropy(answerable_logits, answerable_labels))
            if boundary_labels is not None:
                losses.append(F.binary_cross_entropy_with_logits(boundary_logits, boundary_labels.float()))
            loss = sum(losses)

        return {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "role_logits": role_logits,
            "answerable_logits": answerable_logits,
            "boundary_logits": boundary_logits,
        }

    def role_id_to_label(self, role_id: int) -> str:
        if 0 <= role_id < len(self.role_labels):
            return self.role_labels[role_id]
        return "O"
