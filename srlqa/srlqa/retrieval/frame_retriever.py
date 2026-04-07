"""Role-aware PropBank frame retrieval."""

from __future__ import annotations

import re
from pathlib import Path

from .propbank_index import FrameIndex, FrameRecord


def light_lemma(text: str) -> str:
    token = re.sub(r"[^A-Za-z-]", "", text.lower())
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


class FrameRetriever:
    """Retrieve PropBank frames for a predicate hint."""

    def __init__(self, index: FrameIndex) -> None:
        self.index = index

    @classmethod
    def from_store(cls, path: Path) -> "FrameRetriever":
        return cls(FrameIndex.load(path))

    def retrieve(self, predicate: str, limit: int = 3) -> list[FrameRecord]:
        lemma = light_lemma(predicate)
        direct = self.index.lookup(lemma)
        if direct:
            return direct[:limit]
        scored: list[tuple[int, FrameRecord]] = []
        for frame in self.index.frames:
            score = 0
            if frame.lemma.startswith(lemma) or lemma.startswith(frame.lemma):
                score += 2
            if lemma and lemma in frame.roleset_id.lower():
                score += 1
            if score:
                scored.append((score, frame))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [frame for _, frame in scored[:limit]]

    @staticmethod
    def format_frame_hint(frames: list[FrameRecord]) -> str:
        hints: list[str] = []
        for frame in frames:
            role_bits = [
                f"{role.role}: {role.description}"
                for role in frame.roles
                if role.description
            ]
            hint = f"{frame.roleset_id} {frame.name}. " + "; ".join(role_bits[:6])
            if frame.examples:
                hint += " Example: " + frame.examples[0]
            hints.append(hint.strip())
        return " ".join(hints)
