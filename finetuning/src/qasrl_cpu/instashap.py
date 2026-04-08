from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class InstaShapExplanation:
    tokens: list[str]
    scores: list[float]
    normalized_scores: list[float]
    target_text: str
    confidence_drop: float


class InstaShapExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def explain(self, input_text: str, target_text: str, top_k: int = 5) -> InstaShapExplanation:
        self.model.zero_grad(set_to_none=True)
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=196,
        )
        labels = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )["input_ids"]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
        embeddings = embeddings.detach().clone().requires_grad_(True)
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=encoded["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        gradients = embeddings.grad[0].detach().cpu().numpy()
        embed_values = embeddings[0].detach().cpu().numpy()
        raw_scores = -(gradients * embed_values).sum(axis=1)
        raw_scores = raw_scores.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
        abs_sum = float(np.sum(np.abs(raw_scores))) or 1.0
        normalized = [float(score / abs_sum) for score in raw_scores]
        confidence_drop = self._compute_confidence_drop(encoded, labels, raw_scores, top_k)
        return InstaShapExplanation(
            tokens=tokens,
            scores=raw_scores,
            normalized_scores=normalized,
            target_text=target_text,
            confidence_drop=confidence_drop,
        )

    def _compute_confidence_drop(self, encoded: dict, labels: torch.Tensor, raw_scores: list[float], top_k: int) -> float:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            baseline = self.model(**encoded, labels=labels).loss.item()
        token_count = len(raw_scores)
        keep = max(1, min(top_k, token_count))
        top_indices = np.argsort(np.abs(raw_scores))[-keep:]
        masked_input_ids = encoded["input_ids"].clone()
        masked_input_ids[:, top_indices] = self.tokenizer.pad_token_id
        masked_inputs = {
            "input_ids": masked_input_ids.to(device),
            "attention_mask": encoded["attention_mask"].clone().to(device),
        }
        with torch.no_grad():
            masked_loss = self.model(**masked_inputs, labels=labels).loss.item()
        return round(masked_loss - baseline, 4)

    def compute_plausibility(self, explanation: InstaShapExplanation, gold_tokens: set[str], top_k: int = 5) -> float:
        ranked = np.argsort(np.abs(explanation.scores))[-top_k:]
        predicted = {explanation.tokens[index].replace("▁", "").lower() for index in ranked}
        gold_tokens = {token.lower() for token in gold_tokens if token}
        if not gold_tokens:
            return 0.0
        return round(len(predicted & gold_tokens) / len(gold_tokens), 4)

    def plot(self, explanation: InstaShapExplanation, max_tokens: int = 40):
        tokens = explanation.tokens[:max_tokens]
        scores = explanation.scores[:max_tokens]
        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ["#166534" if score >= 0 else "#991b1b" for score in scores]
        ax.bar(range(len(tokens)), scores, color=colors, alpha=0.85)
        ax.axhline(y=0.0, color="black", linewidth=0.7)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
        ax.set_title("InstaShap-style token attribution")
        ax.set_ylabel("Contribution")
        fig.tight_layout()
        return fig
