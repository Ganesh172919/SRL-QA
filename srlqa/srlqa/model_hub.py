"""Unified access to every QA model family in this workspace."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProjectConfig, get_config
from .pipeline import RaiseSrlQaSystem


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    label: str
    description: str
    needs_external_weights: bool = False


MODEL_SPECS = [
    ModelSpec(
        key="raise_srlqa_fast",
        label="RAISE-SRL-QA Fast",
        description="New deterministic SRL rules + PropBank frame retrieval + verifier correction.",
    ),
    ModelSpec(
        key="raise_srlqa_model",
        label="RAISE-SRL-QA Model",
        description="New RAISE pipeline with DeBERTa/SQuAD-style QA candidates plus recursive SRL correction.",
        needs_external_weights=True,
    ),
    ModelSpec(
        key="legacy_hybrid",
        label="Legacy Hybrid",
        description="Existing srl_qa_project HybridQASystem with PropQA-Net, transformer QA, embeddings, and reranking.",
        needs_external_weights=True,
    ),
    ModelSpec(
        key="legacy_baseline",
        label="Legacy PropQA-Net Baseline",
        description="Existing trained BiLSTM PropQA-Net checkpoint from srl_qa_project.",
    ),
]


def model_choices(include_all: bool = True) -> list[str]:
    choices = [spec.key for spec in MODEL_SPECS]
    return ["all", *choices] if include_all else choices


def model_labels(include_all: bool = True) -> dict[str, str]:
    labels = {spec.key: spec.label for spec in MODEL_SPECS}
    if include_all:
        labels["all"] = "All Models"
    return labels


class ModelHub:
    """Lazy-load and run all model families without crashing the demo."""

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or get_config()
        self._instances: dict[str, Any] = {}

    @property
    def legacy_root(self) -> Path:
        return self.config.paths.existing_project_root

    def _ensure_legacy_path(self) -> None:
        root = str(self.legacy_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _legacy_config(self) -> Any:
        self._ensure_legacy_path()
        from config import get_config as get_legacy_config

        return get_legacy_config()

    def _get_raise_fast(self) -> RaiseSrlQaSystem:
        if "raise_srlqa_fast" not in self._instances:
            self._instances["raise_srlqa_fast"] = RaiseSrlQaSystem(self.config, use_teacher_qa=False)
        return self._instances["raise_srlqa_fast"]

    def _get_raise_model(self) -> RaiseSrlQaSystem:
        if "raise_srlqa_model" not in self._instances:
            self._instances["raise_srlqa_model"] = RaiseSrlQaSystem(self.config, use_teacher_qa=True)
        return self._instances["raise_srlqa_model"]

    def _get_legacy_hybrid(self) -> Any:
        if "legacy_hybrid" not in self._instances:
            self._ensure_legacy_path()
            from hybrid_qa import HybridQASystem

            self._instances["legacy_hybrid"] = HybridQASystem(
                self._legacy_config(),
                use_transformer_qa=True,
                use_sentence_embeddings=True,
            )
        return self._instances["legacy_hybrid"]

    def _run_legacy_baseline(self, context: str, question: str) -> dict[str, Any]:
        self._ensure_legacy_path()
        from qa_inference import ask_question

        result = ask_question(self._legacy_config(), context, question)
        return {
            "answer": result.get("predicted_answer", ""),
            "role": result.get("predicted_role", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": "Legacy PropQA-Net checkpoint prediction.",
            "raw": result,
        }

    @staticmethod
    def _normalize_legacy_hybrid(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": result.get("hybrid_answer", result.get("answer", "")),
            "role": result.get("role", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": result.get("reasoning_summary", ""),
            "raw": result,
        }

    @staticmethod
    def _normalize_raise(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": result.get("answer", ""),
            "role": result.get("role", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "reasoning": result.get("reasoning", ""),
            "raw": result,
        }

    def run_one(
        self,
        model_key: str,
        context: str,
        question: str,
        expected_answer: str | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            if model_key == "raise_srlqa_fast":
                raw = self._get_raise_fast().answer(context, question, expected_answer=expected_answer)
                normalized = self._normalize_raise(raw)
            elif model_key == "raise_srlqa_model":
                raw = self._get_raise_model().answer(context, question, expected_answer=expected_answer)
                normalized = self._normalize_raise(raw)
            elif model_key == "legacy_hybrid":
                raw = self._get_legacy_hybrid().answer_question(context, question)
                normalized = self._normalize_legacy_hybrid(raw)
            elif model_key == "legacy_baseline":
                normalized = self._run_legacy_baseline(context, question)
            else:
                raise ValueError(f"Unknown model key: {model_key}")
            normalized["ok"] = True
            normalized["error"] = ""
        except Exception as exc:  # pragma: no cover - demo resilience
            normalized = {
                "answer": "",
                "role": "",
                "confidence": 0.0,
                "reasoning": "",
                "raw": {},
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        spec = next((item for item in MODEL_SPECS if item.key == model_key), None)
        normalized["model_key"] = model_key
        normalized["model_label"] = spec.label if spec else model_key
        normalized["latency_ms"] = (time.perf_counter() - start) * 1000.0
        return normalized

    def run(
        self,
        model_key: str,
        context: str,
        question: str,
        expected_answer: str | None = None,
    ) -> list[dict[str, Any]]:
        keys = [spec.key for spec in MODEL_SPECS] if model_key == "all" else [model_key]
        return [self.run_one(key, context, question, expected_answer=expected_answer) for key in keys]
