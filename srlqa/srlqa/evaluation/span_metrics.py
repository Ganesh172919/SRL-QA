"""Span metrics for SRL-QA."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Mapping, Sequence


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0
    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def role_accuracy(predicted_role: str, gold_role: str) -> float:
    return float(predicted_role == gold_role)


def aggregate_records(records: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not records:
        return {"count": 0.0, "exact_match": 0.0, "token_f1": 0.0, "role_accuracy": 0.0}
    return {
        "count": float(len(records)),
        "exact_match": sum(exact_match(item["prediction"], item["gold"]) for item in records) / len(records),
        "token_f1": sum(token_f1(item["prediction"], item["gold"]) for item in records) / len(records),
        "role_accuracy": sum(role_accuracy(item.get("predicted_role", ""), item.get("gold_role", "")) for item in records) / len(records),
    }


def bootstrap_ci(values: Sequence[float], samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[math.floor(0.025 * (len(means) - 1))]
    high = means[math.floor(0.975 * (len(means) - 1))]
    return low, high
