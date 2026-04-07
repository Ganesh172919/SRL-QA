"""Calibration metrics."""

from __future__ import annotations

from typing import Sequence


def expected_calibration_error(confidences: Sequence[float], correctness: Sequence[float], bins: int = 10) -> float:
    if not confidences or not correctness:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if low <= confidence < high or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        avg_confidence = sum(confidences[index] for index in members) / len(members)
        avg_accuracy = sum(correctness[index] for index in members) / len(members)
        ece += (len(members) / total) * abs(avg_confidence - avg_accuracy)
    return ece


def temperature_scale(confidence: float, temperature: float) -> float:
    if temperature <= 0:
        return confidence
    confidence = min(max(confidence, 1e-6), 1 - 1e-6)
    odds = confidence / (1 - confidence)
    scaled_odds = odds ** (1 / temperature)
    return scaled_odds / (1 + scaled_odds)
