from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qasrl_cpu.data import prepare_grouped_dataset
from qasrl_cpu.inference import predict_dataset
from qasrl_cpu.metrics import compute_dataset_metrics
from qasrl_cpu.modeling import create_lora_model, save_training_bundle


class TextPairDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class Seq2SeqCollator:
    def __init__(self, tokenizer, max_input_length: int, max_target_length: int):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict]) -> dict:
        inputs = self.tokenizer(
            [item["input_text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        labels = self.tokenizer(
            [item["target_text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_loss(model, dataloader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    return float(np.mean(losses)) if losses else 0.0


def train_model(args) -> dict:
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(args.num_threads, os.cpu_count() or 1)))
    device = torch.device("cpu")

    dataset = prepare_grouped_dataset(
        data_dir=args.data_dir,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    train_records = list(dataset["train"])
    validation_records = list(dataset["validation"])
    test_records = list(dataset["test"])

    baseline_records = test_records[: min(args.baseline_limit, len(test_records))]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    zero_shot_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    zero_shot_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    zero_shot_model.to(device)
    zero_shot_predictions = predict_dataset(
        zero_shot_model,
        zero_shot_tokenizer,
        baseline_records,
        description="Zero-shot baseline",
        max_new_tokens=args.max_target_length,
        num_beams=args.num_beams,
        use_fallback=False,
    )
    zero_shot_metrics = compute_dataset_metrics(baseline_records, zero_shot_predictions)
    del zero_shot_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model, tokenizer = create_lora_model(
        args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

    train_loader = DataLoader(
        TextPairDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Seq2SeqCollator(tokenizer, args.max_input_length, args.max_target_length),
    )
    validation_loader = DataLoader(
        TextPairDataset(validation_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Seq2SeqCollator(tokenizer, args.max_input_length, args.max_target_length),
    )

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state = None
    best_metric = -1.0
    patience_left = args.patience
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_losses.append(loss.item() * args.gradient_accumulation_steps)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        validation_loss = evaluate_loss(model, validation_loader, device)
        selection_records = validation_records[: min(args.selection_limit, len(validation_records))]
        selection_predictions = predict_dataset(
            model,
            tokenizer,
            selection_records,
            description=f"Validation generation epoch {epoch}",
            max_new_tokens=args.max_target_length,
            num_beams=args.num_beams,
        )
        selection_metrics = compute_dataset_metrics(selection_records, selection_predictions)
        epoch_metric = selection_metrics["token_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(epoch_losses)), 4),
                "validation_loss": round(validation_loss, 4),
                "selection_token_f1": epoch_metric,
                "selection_exact_match": selection_metrics["exact_match"],
                "selection_role_coverage": selection_metrics["role_coverage"],
            }
        )

        if epoch_metric > best_metric:
            best_metric = epoch_metric
            best_state = deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_seconds = round(time.time() - training_start, 2)
    artifact_dir = Path(args.output_dir)
    save_training_bundle(
        model,
        tokenizer,
        artifact_dir,
        base_model_name=args.model_name,
        extra_metadata={
            "train_limit": len(train_records),
            "validation_limit": len(validation_records),
            "test_limit": len(test_records),
            "epochs_completed": len(history),
            "best_validation_token_f1": best_metric,
            "training_seconds": training_seconds,
        },
    )

    final_predictions = predict_dataset(
        model,
        tokenizer,
        test_records,
        description="Fine-tuned evaluation",
        max_new_tokens=args.max_target_length,
        num_beams=args.num_beams,
    )
    fine_tuned_metrics = compute_dataset_metrics(test_records, final_predictions)

    summary = {
        "config": vars(args),
        "dataset": {
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
            "test_examples": len(test_records),
        },
        "zero_shot_baseline": zero_shot_metrics,
        "training_history": history,
        "fine_tuned_metrics": fine_tuned_metrics,
        "training_seconds": training_seconds,
    }
    with (results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (results_dir / "test_predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "id": record["id"],
                    "sentence": record["sentence"],
                    "predicate": record["predicate"],
                    "gold": record["roles"],
                    "prediction": prediction,
                }
                for record, prediction in zip(test_records, final_predictions)
            ],
            handle,
            indent=2,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a CPU-friendly QA-SRL model with LoRA.")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "flan_t5_small_lora"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--baseline-limit", type=int, default=50)
    parser.add_argument("--selection-limit", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-length", type=int, default=196)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = train_model(args)
    print(json.dumps(summary["fine_tuned_metrics"], indent=2))


if __name__ == "__main__":
    main()
