"""Training entrypoint for the DeBERTa-style MRC SRL-QA model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from ..config import get_config
from ..data.dataset_library import load_library_dataset
from ..data.convert_to_mrc import normalize_records
from ..models.mrc_srl_qa import MrcSrlQaModel
from .collators import SrlQaDataCollator


def _dataset_to_records(dataset: Any) -> list[dict[str, Any]]:
    return [dict(dataset[index]) for index in range(len(dataset))]


def train(output_dir: Path | None = None) -> dict[str, Any]:
    """Run a small supervised MRC fine-tuning job.

    This is intentionally conservative: it creates a real Trainer setup, but
    the caller controls dataset size through `ProjectConfig.dataset.max_examples`
    before launching expensive experiments.
    """

    config = get_config()
    train_dataset = load_library_dataset(config, split=config.dataset.split, max_examples=config.dataset.max_examples)
    records = normalize_records(_dataset_to_records(train_dataset), source=config.dataset.name)
    model = MrcSrlQaModel.from_config(config)
    collator = SrlQaDataCollator.from_config(config)

    target_dir = output_dir or (config.paths.model_dir / "mrc_srlqa")
    args = TrainingArguments(
        output_dir=str(target_dir),
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=records, data_collator=collator)
    result = trainer.train()
    trainer.save_model(str(target_dir))
    return {"output_dir": str(target_dir), "training_loss": float(result.training_loss)}
