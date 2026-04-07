"""Build verifier training pairs from gold and near-miss candidates."""

from __future__ import annotations

from typing import Any, Mapping

from ..training.hard_negative_mining import boundary_near_misses


def make_verifier_examples(example: Mapping[str, Any]) -> list[dict[str, Any]]:
    positives = [
        {
            "context": example.get("context", ""),
            "question": example.get("question", ""),
            "candidate_answer": example.get("answer_text", ""),
            "candidate_role": example.get("role", "O"),
            "label": 1,
            "source": "gold",
        }
    ]
    negatives = [
        {
            "context": example.get("context", ""),
            "question": example.get("question", ""),
            "candidate_answer": item["negative_text"],
            "candidate_role": example.get("role", "O"),
            "label": 0,
            "source": item["negative_type"],
        }
        for item in boundary_near_misses(example)
    ]
    return positives + negatives
