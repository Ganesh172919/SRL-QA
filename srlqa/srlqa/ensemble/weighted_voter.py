"""Weighted answer voter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


DEFAULT_WEIGHTS = {
    "baseline": 0.15,
    "mrc": 0.45,
    "retrieval_decoder": 0.20,
    "verifier": 0.20,
}


def weighted_vote(candidates: Sequence[Mapping[str, Any]], weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        answer = str(candidate.get("answer", candidate.get("text", ""))).strip()
        if not answer:
            continue
        source = str(candidate.get("source", "mrc"))
        score = float(candidate.get("confidence", candidate.get("score", 0.0)))
        scores[answer.lower()] += weights.get(source, 0.10) * score
        representatives.setdefault(answer.lower(), dict(candidate))
    if not scores:
        return {"answer": "", "confidence": 0.0, "source": "abstain"}
    best_key = max(scores.items(), key=lambda item: item[1])[0]
    result = representatives[best_key]
    result["confidence"] = min(1.0, scores[best_key])
    return result
