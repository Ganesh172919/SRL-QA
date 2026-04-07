"""Offline prediction-file evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calibration import expected_calibration_error
from .span_metrics import aggregate_records, bootstrap_ci, exact_match, token_f1


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    return payload.get("records", [])


def evaluate_prediction_file(path: Path) -> dict[str, Any]:
    records = _load_records(path)
    normalized = []
    f1_values = []
    correctness = []
    confidences = []
    for record in records:
        prediction = record.get("prediction", record.get("predicted_answer", ""))
        gold = record.get("gold", record.get("gold_answer", record.get("expected_answer", "")))
        normalized.append(
            {
                "prediction": prediction,
                "gold": gold,
                "predicted_role": record.get("predicted_role", ""),
                "gold_role": record.get("gold_role", record.get("target_role", "")),
            }
        )
        f1 = token_f1(prediction, gold)
        f1_values.append(f1)
        correctness.append(exact_match(prediction, gold))
        confidences.append(float(record.get("confidence", 0.0)))
    low, high = bootstrap_ci(f1_values, samples=500)
    summary = aggregate_records(normalized)
    summary["token_f1_ci95_low"] = low
    summary["token_f1_ci95_high"] = high
    summary["expected_calibration_error"] = expected_calibration_error(confidences, correctness)
    return summary
