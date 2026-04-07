"""Strict silver-label filtering."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def accept_teacher_label(
    teacher_prediction: Mapping[str, Any],
    context: str,
    verifier_score: float,
    min_teacher_score: float = 0.80,
    min_verifier_score: float = 0.90,
) -> bool:
    answer = str(teacher_prediction.get("answer", "")).strip()
    if not answer:
        return False
    if answer.lower() not in context.lower():
        return False
    if float(teacher_prediction.get("score", 0.0)) < min_teacher_score:
        return False
    if verifier_score < min_verifier_score:
        return False
    return True


def filter_silver_labels(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    accepted = []
    for record in records:
        if accept_teacher_label(
            record["teacher_prediction"],
            record["context"],
            float(record.get("verifier_score", 0.0)),
        ):
            accepted.append(dict(record))
    return accepted
