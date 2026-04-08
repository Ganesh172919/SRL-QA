from __future__ import annotations

import json
from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def create_lora_model(
    model_name: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def save_training_bundle(
    model,
    tokenizer,
    output_dir: str | Path,
    base_model_name: str,
    extra_metadata: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {"base_model_name": base_model_name}
    if extra_metadata:
        metadata.update(extra_metadata)
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_trained_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    base_model_name = metadata["base_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer, metadata
