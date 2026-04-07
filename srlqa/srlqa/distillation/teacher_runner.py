"""Run optional teacher models for silver-label proposals."""

from __future__ import annotations

from typing import Any, Mapping


class ExtractiveQATeacher:
    """Wrapper around a Transformers QA pipeline."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self.model_name = model_name
        self.pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name, device=-1)

    def predict(self, example: Mapping[str, Any]) -> dict[str, Any]:
        result = self.pipeline(question=example["question"], context=example["context"])
        return {
            "example_id": example.get("example_id", ""),
            "teacher": self.model_name,
            "answer": result.get("answer", ""),
            "score": float(result.get("score", 0.0)),
            "start": int(result.get("start", -1)),
            "end": int(result.get("end", -1)),
        }
