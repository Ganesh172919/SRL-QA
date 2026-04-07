"""Hard-negative generation for near-miss span training."""

from __future__ import annotations

from typing import Any, Mapping


def boundary_near_misses(example: Mapping[str, Any], max_width: int = 2) -> list[dict[str, Any]]:
    """Create overlapping negative spans around the gold answer."""

    context_tokens = str(example.get("context", "")).split()
    answer = str(example.get("answer_text", ""))
    answer_tokens = answer.split()
    if not context_tokens or not answer_tokens:
        return []

    joined = " ".join(context_tokens).lower()
    start_char = joined.find(answer.lower())
    if start_char < 0:
        return []

    prefix_tokens = joined[:start_char].split()
    start = len(prefix_tokens)
    end = start + len(answer_tokens) - 1
    negatives: list[dict[str, Any]] = []
    for left in range(max(0, start - max_width), start + 1):
        for right in range(end, min(len(context_tokens), end + max_width + 1)):
            if left == start and right == end:
                continue
            span = " ".join(context_tokens[left : right + 1])
            negatives.append(
                {
                    "example_id": example.get("example_id", ""),
                    "negative_text": span,
                    "negative_start_token": left,
                    "negative_end_token": right,
                    "negative_type": "boundary_near_miss",
                }
            )
    return negatives


def wrong_role_negatives(example: Mapping[str, Any], candidate_spans: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gold_role = str(example.get("role", ""))
    negatives = []
    for candidate in candidate_spans:
        if str(candidate.get("role", "")) != gold_role:
            payload = dict(candidate)
            payload["negative_type"] = "wrong_role"
            negatives.append(payload)
    return negatives
