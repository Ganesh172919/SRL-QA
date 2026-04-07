"""Evaluation and reporting utilities for PropQA-Net."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import ProjectConfig
from model import PredictionResult, PropQANet
from trainer import move_batch_to_device, token_level_f1


def vocabulary_from_serialized(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Load serialized vocabulary mappings from a checkpoint."""

    return serialized


def strip_bio_prefix(label: str) -> str:
    """Remove BIO prefixes from a label."""

    if label == "O" or "-" not in label:
        return label
    return label.split("-", maxsplit=1)[1]


def normalize_text(text: str) -> str:
    """Normalize text for exact-match comparison."""

    return " ".join(text.lower().split())


def load_trained_model(
    config: ProjectConfig,
    device: torch.device,
) -> Tuple[PropQANet, Dict[str, Any]]:
    """Load the best checkpoint and rebuild the model."""

    checkpoint = torch.load(config.paths.checkpoint_path, map_location=device)
    vocabularies = vocabulary_from_serialized(checkpoint["vocabularies"])

    model = PropQANet(
        vocab_size=len(vocabularies["token_vocab"]["id_to_token"]),
        pos_vocab_size=len(vocabularies["pos_vocab"]["id_to_token"]),
        num_labels=len(vocabularies["label_vocab"]["id_to_token"]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def prediction_records(
    model: PropQANet,
    data_loader: Any,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Generate prediction records for every example in a data loader."""

    label_vocab = checkpoint["vocabularies"]["label_vocab"]
    id_to_label = label_vocab["id_to_token"]
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            predictions = model.predict(batch, id_to_label)
            raw_examples = batch["raw_examples"]

            for prediction, raw_example in zip(predictions, raw_examples, strict=True):
                predicted_tokens = raw_example["context_tokens"][
                    prediction.start : prediction.end + 1
                ]
                predicted_text = " ".join(predicted_tokens)
                gold_text = raw_example["answer_text"]
                gold_tokens = raw_example["answer_tokens"]
                role_prediction = prediction.role
                gold_role = raw_example["target_role"]
                qa_f1 = token_level_f1(predicted_tokens, gold_tokens)
                exact_match = float(
                    normalize_text(predicted_text) == normalize_text(gold_text)
                )

                records.append(
                    {
                        "example_id": raw_example["example_id"],
                        "context": raw_example["context"],
                        "question": raw_example["question"],
                        "question_type": raw_example["question_type"],
                        "gold_text": gold_text,
                        "gold_tokens": gold_tokens,
                        "gold_role": gold_role,
                        "gold_bio": raw_example["srl_tags"],
                        "predicted_text": predicted_text,
                        "predicted_tokens": predicted_tokens,
                        "predicted_role": role_prediction,
                        "predicted_bio": prediction.decoded_labels,
                        "start": prediction.start,
                        "end": prediction.end,
                        "confidence": prediction.confidence,
                        "exact_match": exact_match,
                        "token_f1": qa_f1,
                        "answer_length_difference": len(predicted_tokens) - len(gold_tokens),
                        "sentence_length": len(raw_example["context_tokens"]),
                    }
                )

    return records


def role_metrics_from_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute token-level SRL metrics from prediction records."""

    gold_roles: List[str] = []
    predicted_roles: List[str] = []
    gold_bio: List[str] = []
    predicted_bio: List[str] = []

    for record in records:
        gold_labels = record["gold_bio"]
        predicted_labels = record["predicted_bio"]
        gold_bio.extend(gold_labels)
        predicted_bio.extend(predicted_labels)
        gold_roles.extend(strip_bio_prefix(label) for label in gold_labels)
        predicted_roles.extend(strip_bio_prefix(label) for label in predicted_labels)

    role_labels = sorted(set(gold_roles) | set(predicted_roles))
    evaluation_labels = [label for label in role_labels if label != "O"]

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_roles,
        predicted_roles,
        labels=evaluation_labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold_roles,
        predicted_roles,
        labels=evaluation_labels,
        average="macro",
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        gold_roles,
        predicted_roles,
        labels=evaluation_labels,
        average="micro",
        zero_division=0,
    )

    per_role = {}
    for label, prc, rec, score, sup in zip(
        evaluation_labels,
        precision,
        recall,
        f1,
        support,
        strict=True,
    ):
        per_role[label] = {
            "precision": float(prc),
            "recall": float(rec),
            "f1": float(score),
            "support": int(sup),
        }

    confusion = confusion_matrix(
        gold_roles,
        predicted_roles,
        labels=role_labels,
    )
    bio_accuracy = accuracy_score(gold_bio, predicted_bio)

    return {
        "per_role": per_role,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "bio_accuracy": float(bio_accuracy),
        "confusion_matrix_labels": role_labels,
        "confusion_matrix": confusion.tolist(),
    }


def qa_metrics_from_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute QA metrics from prediction records."""

    exact_matches = [record["exact_match"] for record in records]
    token_f1s = [record["token_f1"] for record in records]
    length_differences = [record["answer_length_difference"] for record in records]

    qtype_metrics: Dict[str, Dict[str, float]] = {}
    for question_type in sorted({record["question_type"] for record in records}):
        subset = [record for record in records if record["question_type"] == question_type]
        qtype_metrics[question_type] = {
            "em": float(sum(item["exact_match"] for item in subset) / max(len(subset), 1)),
            "f1": float(sum(item["token_f1"] for item in subset) / max(len(subset), 1)),
            "count": float(len(subset)),
        }

    return {
        "exact_match": float(sum(exact_matches) / max(len(exact_matches), 1)),
        "token_f1": float(sum(token_f1s) / max(len(token_f1s), 1)),
        "answer_length_deviation_mean": float(np.mean(length_differences)) if length_differences else 0.0,
        "answer_length_deviation_abs_mean": float(np.mean(np.abs(length_differences))) if length_differences else 0.0,
        "per_question_type": qtype_metrics,
    }


def classify_error(record: Dict[str, Any]) -> str:
    """Assign an error category to a prediction record."""

    if record["exact_match"] == 1.0:
        return "correct"
    if record["predicted_role"] == "O":
        return "predicate miss"
    if record["predicted_role"] != record["gold_role"]:
        return "wrong role"
    if record["token_f1"] > 0.0:
        return "span boundary error"
    return "other"


def error_analysis(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce error-focused summaries from test predictions."""

    error_records = []
    taxonomy_counter: Dict[str, int] = {}
    sentence_bucket_counter: Dict[str, Dict[str, int]] = {}
    role_error_counter: Dict[str, Dict[str, int]] = {}

    for record in records:
        category = classify_error(record)
        taxonomy_counter[category] = taxonomy_counter.get(category, 0) + 1
        if category == "correct":
            continue

        enriched = dict(record)
        enriched["error_category"] = category
        error_records.append(enriched)

        length = record["sentence_length"]
        if length <= 10:
            bucket = "0-10"
        elif length <= 20:
            bucket = "11-20"
        elif length <= 30:
            bucket = "21-30"
        else:
            bucket = "31+"

        sentence_bucket_counter.setdefault(bucket, {"errors": 0, "total": 0})
        sentence_bucket_counter[bucket]["errors"] += 1

        role = record["gold_role"]
        role_error_counter.setdefault(role, {"errors": 0, "total": 0})
        role_error_counter[role]["errors"] += 1

    for record in records:
        length = record["sentence_length"]
        if length <= 10:
            bucket = "0-10"
        elif length <= 20:
            bucket = "11-20"
        elif length <= 30:
            bucket = "21-30"
        else:
            bucket = "31+"

        sentence_bucket_counter.setdefault(bucket, {"errors": 0, "total": 0})
        sentence_bucket_counter[bucket]["total"] += 1

        role = record["gold_role"]
        role_error_counter.setdefault(role, {"errors": 0, "total": 0})
        role_error_counter[role]["total"] += 1

    top_errors = sorted(
        error_records,
        key=lambda item: (item["token_f1"], -item["confidence"]),
    )[:20]

    sentence_bucket_rates = {
        bucket: {
            "errors": values["errors"],
            "total": values["total"],
            "rate": float(values["errors"] / max(values["total"], 1)),
        }
        for bucket, values in sorted(sentence_bucket_counter.items())
    }
    role_error_rates = {
        role: {
            "errors": values["errors"],
            "total": values["total"],
            "rate": float(values["errors"] / max(values["total"], 1)),
        }
        for role, values in sorted(role_error_counter.items())
    }

    return {
        "top_20_errors": top_errors,
        "taxonomy": taxonomy_counter,
        "sentence_length_error_rates": sentence_bucket_rates,
        "role_error_rates": role_error_rates,
    }


def plot_loss_curve(history: Sequence[Dict[str, float]], output_path: Path) -> None:
    """Plot training and validation loss curves."""

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [item["train_loss"] for item in history]
    validation_loss = [item["validation_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Training loss")
    plt.plot(epochs, validation_loss, marker="s", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_role_f1(role_metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot per-role F1 scores."""

    labels = list(role_metrics["per_role"].keys())
    scores = [role_metrics["per_role"][label]["f1"] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color="#3C6E71")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1")
    plt.xlabel("Argument type")
    plt.title("Per-Argument-Type F1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion(role_metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot a confusion matrix heatmap."""

    matrix = np.array(role_metrics["confusion_matrix"])
    labels = role_metrics["confusion_matrix_labels"]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted role")
    plt.ylabel("Gold role")
    plt.title("SRL Role Confusion Matrix")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_qtype_metrics(qa_metrics: Dict[str, Any], output_path: Path) -> None:
    """Plot EM and F1 by question type."""

    question_types = list(qa_metrics["per_question_type"].keys())
    em_scores = [qa_metrics["per_question_type"][key]["em"] for key in question_types]
    f1_scores = [qa_metrics["per_question_type"][key]["f1"] for key in question_types]

    positions = np.arange(len(question_types))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(positions - width / 2, em_scores, width=width, label="EM", color="#D9BF77")
    plt.bar(positions + width / 2, f1_scores, width=width, label="F1", color="#284B63")
    plt.xticks(positions, question_types)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("Question type")
    plt.title("EM and F1 by Question Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_answer_length_distribution(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    """Plot predicted vs. gold answer-length distributions."""

    gold_lengths = [len(record["gold_tokens"]) for record in records]
    predicted_lengths = [len(record["predicted_tokens"]) for record in records]

    plt.figure(figsize=(8, 5))
    bins = np.arange(0, max(gold_lengths + predicted_lengths + [1]) + 2) - 0.5
    plt.hist(gold_lengths, bins=bins, alpha=0.6, label="Gold", color="#5C946E")
    plt.hist(predicted_lengths, bins=bins, alpha=0.6, label="Predicted", color="#A44A3F")
    plt.xlabel("Answer length (tokens)")
    plt.ylabel("Frequency")
    plt.title("Answer Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_error_taxonomy(error_report: Dict[str, Any], output_path: Path) -> None:
    """Plot the error taxonomy pie chart."""

    taxonomy = {k: v for k, v in error_report["taxonomy"].items() if k != "correct"}
    if not taxonomy:
        taxonomy = {"no errors": 1}

    plt.figure(figsize=(6, 6))
    plt.pie(
        list(taxonomy.values()),
        labels=list(taxonomy.keys()),
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Error Taxonomy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Write metrics to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_pointer:
        json.dump(metrics, file_pointer, indent=2)


def evaluate_model(
    test_loader: Any,
    config: ProjectConfig,
) -> Dict[str, Any]:
    """Evaluate the best checkpoint on the test set and save all reports."""

    device = torch.device(config.runtime.device)
    model, checkpoint = load_trained_model(config, device)
    records = prediction_records(
        model=model,
        data_loader=test_loader,
        checkpoint=checkpoint,
        device=device,
    )
    srl_metrics = role_metrics_from_records(records)
    qa_metrics = qa_metrics_from_records(records)
    error_report = error_analysis(records)

    history = checkpoint.get("history", [])
    best_validation_f1 = float(checkpoint.get("best_validation_f1", 0.0))
    convergence_epoch = None
    for item in history:
        if item["validation_f1"] >= 0.90 * best_validation_f1:
            convergence_epoch = int(item["epoch"])
            break
    if convergence_epoch is None and history:
        convergence_epoch = int(history[-1]["epoch"])

    diagnostics = {
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "best_validation_f1": best_validation_f1,
        "epochs_to_90pct_best_f1": int(convergence_epoch or 0),
        "parameter_count": int(checkpoint.get("model_summary", {}).get("trainable_parameters", 0)),
        "history": history,
    }

    metrics = {
        "srl_performance": srl_metrics,
        "qa_performance": qa_metrics,
        "training_diagnostics": diagnostics,
        "error_analysis": error_report,
        "prediction_sample": records[:10],
    }

    config.paths.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, config.paths.plots_dir / "loss_curve.png")
    plot_role_f1(srl_metrics, config.paths.plots_dir / "f1_by_argtype.png")
    plot_confusion(srl_metrics, config.paths.plots_dir / "confusion_matrix.png")
    plot_qtype_metrics(qa_metrics, config.paths.plots_dir / "qa_accuracy_by_qtype.png")
    plot_answer_length_distribution(records, config.paths.plots_dir / "answer_length_dist.png")
    plot_error_taxonomy(error_report, config.paths.plots_dir / "error_taxonomy.png")

    save_metrics(metrics, config.paths.metrics_path)
    print(f"[eval] metrics written to {config.paths.metrics_path}")
    return metrics
