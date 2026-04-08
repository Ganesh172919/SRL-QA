from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

from .roles import fallback_role_mapping, format_role_output, parse_role_output, refine_role_mapping, render_qa_pairs


def _to_device(inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in inputs.items()}


def generate_text(
    model,
    tokenizer,
    input_text: str,
    max_input_length: int = 196,
    max_new_tokens: int = 96,
    num_beams: int = 2,
) -> tuple[str, float]:
    device = next(model.parameters()).device
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = _to_device(encoded, device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.15,
    }
    if num_beams and num_beams > 1:
        generate_kwargs["length_penalty"] = 0.9
    with torch.no_grad():
        outputs = model.generate(**encoded, **generate_kwargs)
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    score = 0.0
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        score = float(math.exp(outputs.sequences_scores[0].item()))
    return text, score


def predict_single(model, tokenizer, sentence: str, predicate: str, input_text: str) -> dict:
    return predict_single_with_options(model, tokenizer, sentence, predicate, input_text, use_fallback=True)


def predict_single_with_options(
    model,
    tokenizer,
    sentence: str,
    predicate: str,
    input_text: str,
    use_fallback: bool = True,
) -> dict:
    prediction_text, confidence = generate_text(model, tokenizer, input_text)
    role_map = refine_role_mapping(parse_role_output(prediction_text), sentence)
    if use_fallback and not role_map:
        role_map = fallback_role_mapping(sentence, predicate)
    prediction_text = format_role_output(role_map) if role_map else prediction_text
    qa_pairs = render_qa_pairs(role_map, predicate)
    return {
        "sentence": sentence,
        "predicate": predicate,
        "prediction_text": prediction_text,
        "roles": role_map,
        "qa_pairs": qa_pairs,
        "confidence": round(confidence, 4),
    }


def predict_dataset(
    model,
    tokenizer,
    records: list[dict],
    description: str = "Generating predictions",
    max_new_tokens: int = 96,
    num_beams: int = 2,
    use_fallback: bool = True,
) -> list[str]:
    predictions = []
    for record in tqdm(records, desc=description):
        prediction_text, _ = generate_text(
            model,
            tokenizer,
            record["input_text"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        role_map = refine_role_mapping(parse_role_output(prediction_text), record["sentence"])
        if use_fallback and not role_map:
            role_map = fallback_role_mapping(record["sentence"], record["predicate"])
        predictions.append(format_role_output(role_map) if role_map else prediction_text)
    return predictions
