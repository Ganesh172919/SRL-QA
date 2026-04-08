"""PropBank frame-store helpers reused by the standalone demo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def light_lemma(text: str) -> str:
    token = re.sub(r"[^A-Za-z-]", "", text.lower())
    irregular = {
        "gave": "give",
        "sent": "send",
        "delivered": "deliver",
        "hired": "hire",
        "administered": "administer",
        "approved": "approve",
        "repaired": "repair",
    }
    if token in irregular:
        return irregular[token]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


@dataclass(slots=True)
class FrameStore:
    """Small in-memory index over the existing `srlqa` frame-store JSON."""

    frames: list[dict[str, Any]] = field(default_factory=list)
    by_lemma: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    by_roleset: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "FrameStore":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = list(payload.get("frames", []))
        store = cls(frames=frames)
        for frame in frames:
            lemma = str(frame.get("lemma", "")).lower()
            roleset_id = str(frame.get("roleset_id", ""))
            if lemma:
                store.by_lemma.setdefault(lemma, []).append(frame)
            if roleset_id:
                store.by_roleset[roleset_id] = frame
        return store

    def retrieve(self, predicate: str = "", roleset_id: str = "", limit: int = 3) -> list[dict[str, Any]]:
        if roleset_id and roleset_id in self.by_roleset:
            return [self.by_roleset[roleset_id]]
        lemma = light_lemma(predicate)
        if lemma in self.by_lemma:
            return self.by_lemma[lemma][:limit]
        scored: list[tuple[int, dict[str, Any]]] = []
        for frame in self.frames:
            frame_lemma = str(frame.get("lemma", "")).lower()
            roleset = str(frame.get("roleset_id", "")).lower()
            score = 0
            if lemma and (frame_lemma.startswith(lemma) or lemma.startswith(frame_lemma)):
                score += 2
            if lemma and lemma in roleset:
                score += 1
            if score:
                scored.append((score, frame))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [frame for _, frame in scored[:limit]]

    @staticmethod
    def format_hint(frames: list[dict[str, Any]], role_limit: int = 6) -> str:
        hints: list[str] = []
        for frame in frames:
            roles = [
                f"{role.get('role', '')}: {role.get('description', '')}"
                for role in frame.get("roles", [])[:role_limit]
                if role.get("description")
            ]
            hint = f"{frame.get('roleset_id', '')} {frame.get('name', '')}. " + "; ".join(roles)
            examples = frame.get("examples", [])
            if examples:
                hint += " Example: " + str(examples[0])
            hints.append(" ".join(hint.split()))
        return " ".join(hints).strip()

    def hint_for(self, predicate: str = "", roleset_id: str = "") -> str:
        return self.format_hint(self.retrieve(predicate=predicate, roleset_id=roleset_id))

    def role_compatible(self, role: str, predicate: str = "", roleset_id: str = "") -> bool:
        frames = self.retrieve(predicate=predicate, roleset_id=roleset_id)
        if not frames:
            return True
        frame_roles = {
            str(role_def.get("role", ""))
            for frame in frames
            for role_def in frame.get("roles", [])
        }
        return role in frame_roles or role.startswith("ARGM")
