"""Lightweight nominal predicate detector."""

from __future__ import annotations

import re

EVENT_SUFFIXES = ("tion", "sion", "ment", "al", "ance", "ence")
EVENT_NOUNS = {"purchase", "sale", "explosion", "meeting", "approval", "delivery", "arrival"}


def detect_nominal_predicates(text: str) -> list[str]:
    candidates = []
    for token in re.findall(r"[A-Za-z][A-Za-z-]+", text):
        lower = token.lower()
        if lower in EVENT_NOUNS or lower.endswith(EVENT_SUFFIXES):
            candidates.append(token)
    return candidates
