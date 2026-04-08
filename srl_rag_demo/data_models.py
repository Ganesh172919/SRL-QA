"""Shared data structures for SRL documents, retrieval, and QA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SRLArgument:
    role: str
    text: str
    description: str = ""
    start_token: int = -1
    end_token: int = -1
    is_contiguous: bool = True
    source: str = "propbank"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SRLArgument":
        return cls(**payload)


@dataclass(slots=True)
class SRLDocument:
    doc_id: str
    source: str
    context: str
    tokens: list[str]
    predicate: str = ""
    predicate_lemma: str = ""
    predicate_indices: list[int] = field(default_factory=list)
    roleset_id: str = ""
    roleset_name: str = ""
    frame_hint: str = ""
    arguments: list[SRLArgument] = field(default_factory=list)

    def role_triples(self) -> list[str]:
        predicate = self.predicate or self.predicate_lemma or "event"
        return [
            f"{predicate} -> {argument.role} -> {argument.text}"
            for argument in self.arguments
            if argument.text
        ]

    def retrieval_text(self) -> str:
        role_descriptions = [
            f"{argument.role}: {argument.description}"
            for argument in self.arguments
            if argument.description
        ]
        parts = [
            self.context,
            f"predicate: {self.predicate or self.predicate_lemma}",
            f"roleset: {self.roleset_id} {self.roleset_name}",
            " ".join(self.role_triples()),
            " ".join(role_descriptions),
            self.frame_hint,
        ]
        return " ".join(part for part in parts if part).strip()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["arguments"] = [argument.to_dict() for argument in self.arguments]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SRLDocument":
        copied = dict(payload)
        copied["arguments"] = [
            SRLArgument.from_dict(argument)
            for argument in copied.get("arguments", [])
        ]
        return cls(**copied)


@dataclass(slots=True)
class RetrievalHit:
    document: SRLDocument
    score: float
    backend: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "backend": self.backend,
            "doc_id": self.document.doc_id,
            "source": self.document.source,
            "predicate": self.document.predicate,
            "roleset_id": self.document.roleset_id,
            "context": self.document.context,
            "triples": self.document.role_triples(),
            "frame_hint": self.document.frame_hint,
        }


@dataclass(slots=True)
class AnswerCandidate:
    text: str
    role: str
    confidence: float
    source_doc_id: str
    source: str
    predicate: str = ""
    frame_hint: str = ""
    start_token: int = -1
    end_token: int = -1
    retrieval_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QAResult:
    question: str
    answer: str
    confidence: float
    role: str
    source_doc_id: str
    source: str
    predicate: str
    evidence_text: str
    frame_hint: str
    reasoning: list[str]
    candidates: list[AnswerCandidate]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload
