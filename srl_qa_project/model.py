"""PropQA-Net model definition.

ASCII architecture diagram
==========================

    Context tokens ----> [Word Embedding] --\
    POS tags ---------> [POS Embedding] -----+--> [BiLSTM Context Encoder] --> [Dropout]
    Predicate flags --> [Predicate Emb.] ----/                               |          \
                                                                             |           \
                                                                             v            v
                                                                    [SRL BIO Classifier]  [QA Span Projections]
                                                                             |            /
    Question tokens --> [Shared Word Embedding] --> [BiLSTM Question Encoder] --> [Question Vector]
                                                                                           |
                                                                                           v
                                                                     [Argument-Question Matching Layer]
                                                                                           |
                                                                                           v
                                                                               Best answer span + score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import ProjectConfig


def strip_bio_prefix(label: str) -> str:
    """Remove BIO prefixes from a label."""

    if label == "O" or "-" not in label:
        return label
    return label.split("-", maxsplit=1)[1]


def masked_mean_pooling(sequence: Tensor, mask: Tensor) -> Tensor:
    """Compute a mask-aware mean over the time dimension."""

    mask = mask.unsqueeze(-1).float()
    masked_sequence = sequence * mask
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return masked_sequence.sum(dim=1) / denominator


def decode_bio_spans(labels: Sequence[str]) -> List[Dict[str, int | str]]:
    """Decode BIO tags into argument spans."""

    spans: List[Dict[str, int | str]] = []
    start = None
    role = None

    for index, label in enumerate(labels):
        if label == "O":
            if start is not None and role is not None:
                spans.append({"start": start, "end": index - 1, "role": role})
                start = None
                role = None
            continue

        prefix, current_role = label.split("-", maxsplit=1)
        if prefix == "B" or current_role != role:
            if start is not None and role is not None:
                spans.append({"start": start, "end": index - 1, "role": role})
            start = index
            role = current_role
        elif prefix == "I":
            continue

    if start is not None and role is not None:
        spans.append({"start": start, "end": len(labels) - 1, "role": role})
    return spans


def majority_role(labels: Sequence[str]) -> str:
    """Return the most frequent non-``O`` role in a label window."""

    counts: Dict[str, int] = {}
    for label in labels:
        role = strip_bio_prefix(label)
        if role == "O":
            continue
        counts[role] = counts.get(role, 0) + 1
    if not counts:
        return "O"
    return max(counts.items(), key=lambda item: item[1])[0]


@dataclass(slots=True)
class PredictionResult:
    """Container for model predictions on a single example."""

    start: int
    end: int
    role: str
    confidence: float
    decoded_labels: List[str]


class PropQANet(nn.Module):
    """Proposition-anchored QA network for SRL-based extractive QA.

    Pseudocode
    ----------

    1. Embed each context token with word, POS, and predicate-indicator vectors.
    2. Run a BiLSTM over the context to obtain contextualized token states.
    3. Predict BIO SRL tags from each contextualized token state.
    4. Embed question tokens and encode them with a second BiLSTM.
    5. Pool question states into a fixed-size question vector.
    6. Score answer boundaries by combining each context state with the question vector.
    7. Decode SRL spans, compute cosine similarity between each candidate span and
       the question vector, and combine that score with boundary confidence.
    8. Return the highest-scoring span as the final answer.
    """

    def __init__(
        self,
        vocab_size: int,
        pos_vocab_size: int,
        num_labels: int,
        config: ProjectConfig,
    ) -> None:
        """Initialize the network modules."""

        super().__init__()
        self.config = config
        self.alpha = config.model.alpha
        context_input_size = (
            config.model.word_embedding_dim
            + config.model.pos_embedding_dim
            + config.model.predicate_embedding_dim
        )
        context_hidden = config.model.hidden_size
        question_hidden = config.model.question_hidden_size

        self.word_embeddings = nn.Embedding(
            vocab_size,
            config.model.word_embedding_dim,
            padding_idx=0,
        )
        self.pos_embeddings = nn.Embedding(
            pos_vocab_size,
            config.model.pos_embedding_dim,
            padding_idx=0,
        )
        self.predicate_embeddings = nn.Embedding(
            2,
            config.model.predicate_embedding_dim,
        )

        self.context_encoder = nn.LSTM(
            input_size=context_input_size,
            hidden_size=context_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.question_encoder = nn.LSTM(
            input_size=config.model.word_embedding_dim,
            hidden_size=question_hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.model.dropout)
        self.srl_classifier = nn.Linear(context_hidden * 2, num_labels)
        self.question_projection = nn.Linear(question_hidden * 2, context_hidden * 2)

        interaction_input = (context_hidden * 2) * 4
        self.start_projection = nn.Linear(interaction_input, 1)
        self.end_projection = nn.Linear(interaction_input, 1)

    def _encode_lstm(
        self,
        encoder: nn.LSTM,
        embeddings: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode a padded batch with a BiLSTM."""

        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = encoder(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=embeddings.size(1),
        )
        return outputs

    def encode_context(
        self,
        context_ids: Tensor,
        pos_ids: Tensor,
        predicate_flags: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        """Encode context tokens into contextualized states."""

        token_embeddings = self.word_embeddings(context_ids)
        pos_embeddings = self.pos_embeddings(pos_ids)
        predicate_embeddings = self.predicate_embeddings(predicate_flags)
        context_inputs = torch.cat(
            [token_embeddings, pos_embeddings, predicate_embeddings],
            dim=-1,
        )
        context_outputs = self._encode_lstm(
            encoder=self.context_encoder,
            embeddings=context_inputs,
            mask=context_mask,
        )
        return self.dropout(context_outputs)

    def encode_question(self, question_ids: Tensor, question_mask: Tensor) -> Tensor:
        """Encode question tokens into a pooled question vector."""

        question_embeddings = self.word_embeddings(question_ids)
        question_outputs = self._encode_lstm(
            encoder=self.question_encoder,
            embeddings=question_embeddings,
            mask=question_mask,
        )
        return masked_mean_pooling(question_outputs, question_mask)

    def forward(
        self,
        context_ids: Tensor,
        pos_ids: Tensor,
        predicate_flags: Tensor,
        context_mask: Tensor,
        question_ids: Tensor,
        question_mask: Tensor,
        label_ids: Tensor | None = None,
        answer_starts: Tensor | None = None,
        answer_ends: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Run a full forward pass through PropQA-Net."""

        context_outputs = self.encode_context(
            context_ids=context_ids,
            pos_ids=pos_ids,
            predicate_flags=predicate_flags,
            context_mask=context_mask,
        )
        question_vector = self.encode_question(
            question_ids=question_ids,
            question_mask=question_mask,
        )
        question_vector = self.question_projection(question_vector)

        question_expanded = question_vector.unsqueeze(1).expand(
            -1,
            context_outputs.size(1),
            -1,
        )
        interaction = torch.cat(
            [
                context_outputs,
                question_expanded,
                context_outputs * question_expanded,
                torch.abs(context_outputs - question_expanded),
            ],
            dim=-1,
        )

        srl_logits = self.srl_classifier(context_outputs)
        start_logits = self.start_projection(interaction).squeeze(-1)
        end_logits = self.end_projection(interaction).squeeze(-1)

        mask_value = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(~context_mask, mask_value)
        end_logits = end_logits.masked_fill(~context_mask, mask_value)

        outputs: Dict[str, Tensor] = {
            "context_outputs": context_outputs,
            "question_vector": question_vector,
            "srl_logits": srl_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        if label_ids is not None and answer_starts is not None and answer_ends is not None:
            label_loss = F.cross_entropy(
                srl_logits.view(-1, srl_logits.size(-1)),
                label_ids.view(-1),
                reduction="none",
            )
            flat_mask = context_mask.view(-1).float()
            srl_loss = (label_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
            start_loss = F.cross_entropy(start_logits, answer_starts)
            end_loss = F.cross_entropy(end_logits, answer_ends)
            qa_loss = 0.5 * (start_loss + end_loss)
            outputs["srl_loss"] = srl_loss
            outputs["qa_loss"] = qa_loss
            outputs["loss"] = self.alpha * srl_loss + (1.0 - self.alpha) * qa_loss

        return outputs

    def predict(
        self,
        batch: Dict[str, Tensor | Sequence[Dict[str, Any]]],
        id_to_label: Sequence[str],
    ) -> List[PredictionResult]:
        """Decode model predictions for a batched input."""

        outputs = self.forward(
            context_ids=batch["context_ids"],  # type: ignore[arg-type]
            pos_ids=batch["pos_ids"],  # type: ignore[arg-type]
            predicate_flags=batch["predicate_flags"],  # type: ignore[arg-type]
            context_mask=batch["context_mask"],  # type: ignore[arg-type]
            question_ids=batch["question_ids"],  # type: ignore[arg-type]
            question_mask=batch["question_mask"],  # type: ignore[arg-type]
        )

        context_mask = batch["context_mask"]  # type: ignore[assignment]
        srl_logits = outputs["srl_logits"]
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        context_outputs = outputs["context_outputs"]
        question_vector = outputs["question_vector"]

        srl_predictions = srl_logits.argmax(dim=-1)
        start_probabilities = torch.softmax(start_logits, dim=-1)
        end_probabilities = torch.softmax(end_logits, dim=-1)

        decoded: List[PredictionResult] = []
        for batch_index in range(srl_predictions.size(0)):
            valid_length = int(context_mask[batch_index].sum().item())
            label_sequence = [
                id_to_label[label_id]
                for label_id in srl_predictions[batch_index, :valid_length].tolist()
            ]
            candidate_spans = decode_bio_spans(label_sequence)

            best_start = int(torch.argmax(start_probabilities[batch_index, :valid_length]).item())
            best_end = int(
                torch.argmax(end_probabilities[batch_index, best_start:valid_length]).item()
                + best_start
            )

            if candidate_spans:
                best_candidate = None
                for candidate in candidate_spans:
                    start = int(candidate["start"])
                    end = int(candidate["end"])
                    span_vector = context_outputs[batch_index, start : end + 1].mean(dim=0)
                    cosine = F.cosine_similarity(
                        span_vector.unsqueeze(0),
                        question_vector[batch_index].unsqueeze(0),
                    ).item()
                    cosine = (cosine + 1.0) / 2.0
                    boundary = float(
                        0.5
                        * (
                            start_probabilities[batch_index, start].item()
                            + end_probabilities[batch_index, end].item()
                        )
                    )
                    score = 0.60 * cosine + 0.40 * boundary
                    if best_candidate is None or score > best_candidate["score"]:
                        best_candidate = {
                            "start": start,
                            "end": end,
                            "role": str(candidate["role"]),
                            "score": score,
                        }

                assert best_candidate is not None
                decoded.append(
                    PredictionResult(
                        start=int(best_candidate["start"]),
                        end=int(best_candidate["end"]),
                        role=str(best_candidate["role"]),
                        confidence=float(best_candidate["score"]),
                        decoded_labels=label_sequence,
                    )
                )
                continue

            fallback_labels = label_sequence[best_start : best_end + 1]
            decoded.append(
                PredictionResult(
                    start=best_start,
                    end=best_end,
                    role=majority_role(fallback_labels),
                    confidence=float(
                        0.5
                        * (
                            start_probabilities[batch_index, best_start].item()
                            + end_probabilities[batch_index, best_end].item()
                        )
                    ),
                    decoded_labels=label_sequence,
                )
            )

        return decoded

    def model_summary(self) -> Dict[str, Any]:
        """Return a lightweight model summary."""

        parameter_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "name": "PropQA-Net",
            "trainable_parameters": parameter_count,
            "hidden_size": self.config.model.hidden_size,
            "question_hidden_size": self.config.model.question_hidden_size,
            "dropout": self.config.model.dropout,
            "alpha": self.config.model.alpha,
        }
