"""Training utilities for PropQA-Net."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import ProjectConfig
from model import PropQANet


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Move tensor fields in a batch to the target device."""

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def token_level_f1(predicted_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    """Compute token-overlap F1 for extractive QA answers."""

    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0

    predicted_counter: Dict[str, int] = {}
    gold_counter: Dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counter[token.lower()] = predicted_counter.get(token.lower(), 0) + 1
    for token in gold_tokens:
        gold_counter[token.lower()] = gold_counter.get(token.lower(), 0) + 1

    overlap = 0
    for token, count in predicted_counter.items():
        overlap += min(count, gold_counter.get(token, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / max(len(predicted_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def evaluate_validation_split(
    model: PropQANet,
    validation_loader: Any,
    id_to_label: Sequence[str],
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate validation loss, EM, and token F1."""

    model.eval()
    losses: List[float] = []
    exact_matches: List[float] = []
    f1_scores: List[float] = []

    with torch.no_grad():
        for batch in validation_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                context_ids=batch["context_ids"],
                pos_ids=batch["pos_ids"],
                predicate_flags=batch["predicate_flags"],
                context_mask=batch["context_mask"],
                question_ids=batch["question_ids"],
                question_mask=batch["question_mask"],
                label_ids=batch["label_ids"],
                answer_starts=batch["answer_starts"],
                answer_ends=batch["answer_ends"],
            )
            losses.append(float(outputs["loss"].item()))

            predictions = model.predict(batch, id_to_label)
            raw_examples = batch["raw_examples"]

            for prediction, raw_example in zip(predictions, raw_examples, strict=True):
                predicted_tokens = raw_example["context_tokens"][
                    prediction.start : prediction.end + 1
                ]
                gold_tokens = raw_example["answer_tokens"]
                exact_matches.append(
                    float(
                        " ".join(predicted_tokens).lower()
                        == " ".join(gold_tokens).lower()
                    )
                )
                f1_scores.append(token_level_f1(predicted_tokens, gold_tokens))

    validation_loss = float(sum(losses) / max(len(losses), 1))
    validation_em = float(sum(exact_matches) / max(len(exact_matches), 1))
    validation_f1 = float(sum(f1_scores) / max(len(f1_scores), 1))
    return validation_loss, validation_em, validation_f1


def serialize_vocabularies(vocabularies: Dict[str, Any]) -> Dict[str, Any]:
    """Convert vocabulary objects into plain dictionaries."""

    return {name: vocabulary.to_dict() for name, vocabulary in vocabularies.items()}


def train_model(
    train_loader: Any,
    validation_loader: Any,
    vocabularies: Dict[str, Any],
    config: ProjectConfig,
) -> Tuple[PropQANet, Dict[str, Any]]:
    """Train PropQA-Net and save the best checkpoint."""

    set_random_seed(config.data.random_seed)
    device = torch.device(config.runtime.device)
    label_vocab = vocabularies["label_vocab"]

    model = PropQANet(
        vocab_size=len(vocabularies["token_vocab"].id_to_token),
        pos_vocab_size=len(vocabularies["pos_vocab"].id_to_token),
        num_labels=len(label_vocab.id_to_token),
        config=config,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_validation_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("[train] model summary:", json.dumps(model.model_summary(), indent=2))

    for epoch in range(1, config.training.max_epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            outputs = model(
                context_ids=batch["context_ids"],
                pos_ids=batch["pos_ids"],
                predicate_flags=batch["predicate_flags"],
                context_mask=batch["context_mask"],
                question_ids=batch["question_ids"],
                question_mask=batch["question_mask"],
                label_ids=batch["label_ids"],
                answer_starts=batch["answer_starts"],
                answer_ends=batch["answer_ends"],
            )
            loss = outputs["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        average_train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        validation_loss, validation_em, validation_f1 = evaluate_validation_split(
            model=model,
            validation_loader=validation_loader,
            id_to_label=label_vocab.id_to_token,
            device=device,
        )

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": average_train_loss,
            "validation_loss": validation_loss,
            "validation_em": validation_em,
            "validation_f1": validation_f1,
        }
        history.append(epoch_record)

        print(
            f"[train] epoch={epoch} train_loss={average_train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_em={validation_em:.4f} "
            f"val_f1={validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_f1": best_validation_f1,
                "vocabularies": serialize_vocabularies(vocabularies),
                "model_summary": model.model_summary(),
            }
            torch.save(checkpoint_payload, config.paths.checkpoint_path)
            print(
                f"[train] saved new best checkpoint to {config.paths.checkpoint_path}"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print("[train] early stopping triggered")
                break

    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    training_summary = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "model_summary": model.model_summary(),
    }
    return model, training_summary
