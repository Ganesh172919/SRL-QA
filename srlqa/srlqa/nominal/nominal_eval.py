"""Separate nominal evaluation hook."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..evaluation.span_metrics import exact_match, token_f1


def evaluate_nominal(records: Sequence[Mapping[str, str]]) -> dict[str, float]:
    if not records:
        return {"count": 0.0, "exact_match": 0.0, "token_f1": 0.0}
    return {
        "count": float(len(records)),
        "exact_match": sum(exact_match(item["prediction"], item["gold"]) for item in records) / len(records),
        "token_f1": sum(token_f1(item["prediction"], item["gold"]) for item in records) / len(records),
    }
