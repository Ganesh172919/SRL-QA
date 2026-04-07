"""Tokenization collators for SRL-QA MRC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..config import ProjectConfig


@dataclass(slots=True)
class RoleVocab:
    labels: list[str]

    @property
    def token_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}

    def encode(self, label: str) -> int:
        return self.token_to_id.get(label, self.token_to_id.get("O", 0))


class SrlQaDataCollator:
    """Build model tensors and align character answer spans to token spans."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: ProjectConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.role_vocab = RoleVocab(list(config.model.role_labels))

    @classmethod
    def from_config(cls, config: ProjectConfig) -> "SrlQaDataCollator":
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.encoder_name,
            use_fast=True,
            cache_dir=str(config.paths.hf_cache_dir / "models"),
        )
        return cls(tokenizer=tokenizer, config=config)

    def _align_span(self, encoding: Any, batch_index: int, example: Mapping[str, Any]) -> tuple[int, int, int]:
        answer_start = int(example.get("answer_start_char", -1))
        answer_end = int(example.get("answer_end_char", -1))
        if answer_start < 0 or answer_end <= answer_start:
            return 0, 0, 0

        sequence_ids = encoding.sequence_ids(batch_index)
        offsets = encoding["offset_mapping"][batch_index].tolist()
        token_start = None
        token_end = None
        for token_index, (start, end) in enumerate(offsets):
            if sequence_ids[token_index] != 1:
                continue
            if token_start is None and start <= answer_start < end:
                token_start = token_index
            if start < answer_end <= end:
                token_end = token_index
                break
        if token_start is None or token_end is None:
            return 0, 0, 0
        return token_start, token_end, 1

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        questions = [
            f"{example.get('question', '')} [ROLE_HINT] {example.get('frame_hint', '')}".strip()
            for example in examples
        ]
        contexts = [str(example.get("context", "")) for example in examples]
        encoding = self.tokenizer(
            questions,
            contexts,
            max_length=self.config.model.max_length,
            truncation="only_second",
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        starts: list[int] = []
        ends: list[int] = []
        answerable: list[int] = []
        role_ids: list[int] = []
        for index, example in enumerate(examples):
            start, end, is_answerable = self._align_span(encoding, index, example)
            starts.append(start)
            ends.append(end)
            answerable.append(is_answerable)
            role_ids.append(self.role_vocab.encode(str(example.get("role", "O"))))

        encoding.pop("offset_mapping")
        encoding["start_positions"] = torch.tensor(starts, dtype=torch.long)
        encoding["end_positions"] = torch.tensor(ends, dtype=torch.long)
        encoding["role_labels"] = torch.tensor(role_ids, dtype=torch.long)
        encoding["answerable_labels"] = torch.tensor(answerable, dtype=torch.long)
        encoding["boundary_labels"] = torch.tensor(answerable, dtype=torch.float)
        return dict(encoding)
