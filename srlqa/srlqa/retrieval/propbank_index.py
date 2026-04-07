"""Build and query a local PropBank frame index."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RoleDefinition:
    role: str
    description: str


@dataclass(slots=True)
class FrameRecord:
    lemma: str
    roleset_id: str
    name: str
    roles: list[RoleDefinition]
    examples: list[str]
    source_file: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lemma": self.lemma,
            "roleset_id": self.roleset_id,
            "name": self.name,
            "roles": [asdict(role) for role in self.roles],
            "examples": self.examples,
            "source_file": self.source_file,
        }


class FrameIndex:
    """In-memory PropBank frame store."""

    def __init__(self, frames: list[FrameRecord]) -> None:
        self.frames = frames
        self.by_lemma: dict[str, list[FrameRecord]] = {}
        for frame in frames:
            self.by_lemma.setdefault(frame.lemma.lower(), []).append(frame)

    @staticmethod
    def _parse_frame(path: Path) -> list[FrameRecord]:
        tree = ET.parse(path)
        root = tree.getroot()
        frames: list[FrameRecord] = []
        for predicate in root.findall(".//predicate"):
            lemma = predicate.attrib.get("lemma", path.stem)
            for roleset in predicate.findall("roleset"):
                roles: list[RoleDefinition] = []
                for role in roleset.findall("./roles/role"):
                    number = role.attrib.get("n", "")
                    role_name = f"ARG{number}" if number else role.attrib.get("f", "ARG")
                    roles.append(
                        RoleDefinition(
                            role=role_name,
                            description=role.attrib.get("descr", "").strip(),
                        )
                    )
                examples = []
                for example in roleset.findall("./example"):
                    text = " ".join("".join(example.itertext()).split())
                    if text:
                        examples.append(text)
                frames.append(
                    FrameRecord(
                        lemma=lemma,
                        roleset_id=roleset.attrib.get("id", ""),
                        name=roleset.attrib.get("name", ""),
                        roles=roles,
                        examples=examples[:3],
                        source_file=str(path),
                    )
                )
        return frames

    @classmethod
    def from_directory(cls, frames_dir: Path) -> "FrameIndex":
        if not frames_dir.exists():
            raise FileNotFoundError(f"PropBank frames directory not found: {frames_dir}")
        frames: list[FrameRecord] = []
        for path in sorted(frames_dir.glob("*.xml")):
            try:
                frames.extend(cls._parse_frame(path))
            except ET.ParseError:
                continue
        return cls(frames)

    @classmethod
    def load(cls, path: Path) -> "FrameIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = [
            FrameRecord(
                lemma=item["lemma"],
                roleset_id=item["roleset_id"],
                name=item.get("name", ""),
                roles=[RoleDefinition(**role) for role in item.get("roles", [])],
                examples=list(item.get("examples", [])),
                source_file=item.get("source_file", ""),
            )
            for item in payload["frames"]
        ]
        return cls(frames)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"frames": [frame.to_dict() for frame in self.frames]}, indent=2),
            encoding="utf-8",
        )

    def lookup(self, lemma: str) -> list[FrameRecord]:
        return self.by_lemma.get(lemma.lower(), [])
