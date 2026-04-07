"""Simple proto-role feature heuristics for ARG0/ARG1 confusion analysis."""

from __future__ import annotations

ANIMATE_WORDS = {"person", "people", "man", "woman", "child", "company", "team", "rahul", "maria", "john"}
CHANGE_WORDS = {"changed", "broken", "repaired", "destroyed", "improved", "affected"}


def proto_role_features(span_text: str) -> dict[str, float]:
    tokens = {token.lower().strip(".,;:!?") for token in span_text.split()}
    return {
        "sentient_or_org": float(bool(tokens & ANIMATE_WORDS)),
        "likely_affected": float(bool(tokens & CHANGE_WORDS)),
        "long_entity": float(len(tokens) > 4),
    }
