"""Student training hook for accepted silver labels."""

from __future__ import annotations

from typing import Any, Sequence


def merge_gold_and_silver(gold: Sequence[dict[str, Any]], silver: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(item, supervision="gold") for item in gold]
    merged.extend(dict(item, supervision="silver") for item in silver)
    return merged
