"""Confidence calibration helpers."""

from __future__ import annotations

from ..evaluation.calibration import temperature_scale


class CalibratedConfidence:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def __call__(self, confidence: float) -> float:
        return temperature_scale(confidence, self.temperature)
