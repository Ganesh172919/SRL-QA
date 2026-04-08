"""Controlled evaluation suite for the SRL + RAG demo.

This is intentionally small and transparent. It is a demo-suite accuracy check,
not a replacement for the full PropBank-derived benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

try:
    from .config import get_config
    from .frame_store import FrameStore
    from .qa import answer_question
    from .retrieval import SRLRetriever
    from .user_docs import build_user_documents
except ImportError:  # pragma: no cover - supports `python srl_rag_demo\demo_eval.py`
    from config import get_config
    from frame_store import FrameStore
    from qa import answer_question
    from retrieval import SRLRetriever
    from user_docs import build_user_documents


@dataclass(frozen=True, slots=True)
class DemoEvalExample:
    example_id: str
    context: str
    question: str
    expected_answer: str
    expected_role: str


DEMO_EVAL_EXAMPLES: tuple[DemoEvalExample, ...] = (
    DemoEvalExample(
        example_id="who_agent",
        context="The chef cooked a delicious meal in the kitchen.",
        question="Who cooked?",
        expected_answer="The chef",
        expected_role="ARG0",
    ),
    DemoEvalExample(
        example_id="what_theme",
        context="The nurse administered the medicine to the patient after dinner.",
        question="What was administered?",
        expected_answer="the medicine",
        expected_role="ARG1",
    ),
    DemoEvalExample(
        example_id="where_location",
        context="The courier delivered the package to the office at noon.",
        question="Where was the package delivered?",
        expected_answer="to the office",
        expected_role="ARGM-LOC",
    ),
    DemoEvalExample(
        example_id="when_time",
        context="She sent a letter to her friend at noon.",
        question="When did she send a letter?",
        expected_answer="at noon",
        expected_role="ARGM-TMP",
    ),
    DemoEvalExample(
        example_id="to_whom_recipient",
        context="She sent a letter to her friend at noon.",
        question="To whom did she send a letter?",
        expected_answer="to her friend",
        expected_role="ARG2",
    ),
    DemoEvalExample(
        example_id="what_object_boundary",
        context="The engineer repaired the machine carefully with a small screwdriver.",
        question="What did the engineer repair?",
        expected_answer="the machine",
        expected_role="ARG1",
    ),
    DemoEvalExample(
        example_id="how_manner",
        context="The engineer repaired the machine carefully with a small screwdriver.",
        question="How did the engineer repair the machine?",
        expected_answer="carefully",
        expected_role="ARGM-MNR",
    ),
    DemoEvalExample(
        example_id="why_cause",
        context="The company announced layoffs because of budget cuts.",
        question="Why did the company announce layoffs?",
        expected_answer="because of budget cuts",
        expected_role="ARGM-CAU",
    ),
)


def normalize_answer(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(normalized.split())


def answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(predicted: str, expected: str) -> float:
    return float(normalize_answer(predicted) == normalize_answer(expected))


def token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = answer_tokens(predicted)
    expected_tokens = answer_tokens(expected)
    if not predicted_tokens and not expected_tokens:
        return 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0

    remaining = list(expected_tokens)
    overlap = 0
    for token in predicted_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def role_match(predicted_role: str, expected_role: str) -> float:
    return float(predicted_role == expected_role)


def run_demo_evaluation(top_k: int = 3) -> dict[str, Any]:
    config = get_config()
    frame_store = FrameStore.load(config.frame_store_path)
    documents = build_user_documents((example.context for example in DEMO_EVAL_EXAMPLES), frame_store)

    retriever = SRLRetriever(config, mode="TF-IDF")
    retrieval_status = retriever.fit(documents)

    records: list[dict[str, Any]] = []
    for example in DEMO_EVAL_EXAMPLES:
        hits = retriever.search(example.question, top_k=top_k)
        result = answer_question(
            question=example.question,
            hits=hits,
            frame_store=frame_store,
            config=config,
            use_transformer=False,
        )
        record = {
            **asdict(example),
            "predicted_answer": result.answer,
            "predicted_role": result.role,
            "confidence": result.confidence,
            "source_doc_id": result.source_doc_id,
            "evidence_text": result.evidence_text,
            "reasoning": result.reasoning,
            "exact_match": exact_match(result.answer, example.expected_answer),
            "token_f1": token_f1(result.answer, example.expected_answer),
            "role_accuracy": role_match(result.role, example.expected_role),
            "retrieved_docs": [hit.to_dict() for hit in hits],
        }
        records.append(record)

    summary = {
        "examples": len(records),
        "exact_match": mean(record["exact_match"] for record in records) if records else 0.0,
        "token_f1": mean(record["token_f1"] for record in records) if records else 0.0,
        "role_accuracy": mean(record["role_accuracy"] for record in records) if records else 0.0,
        "mean_confidence": mean(record["confidence"] for record in records) if records else 0.0,
        "retrieval_backend": retrieval_status.backend,
        "embedding_ready": retrieval_status.embedding_ready,
        "embedding_error": retrieval_status.embedding_error,
    }
    return {
        "summary": summary,
        "records": records,
        "note": "Controlled SRL + RAG demo-suite metrics; not a full-corpus benchmark.",
    }


def format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def format_markdown_report(evaluation: dict[str, Any]) -> str:
    summary = evaluation["summary"]
    lines = [
        "# SRL + RAG Demo Evaluation",
        "",
        "This report is generated from `srl_rag_demo/demo_eval.py`.",
        "",
        "> Scope: controlled demo-suite metrics, not a full-corpus benchmark.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Examples | {summary['examples']} |",
        f"| Exact Match | {format_percent(summary['exact_match'])} |",
        f"| Token F1 | {format_percent(summary['token_f1'])} |",
        f"| Role Accuracy | {format_percent(summary['role_accuracy'])} |",
        f"| Mean Confidence | {format_percent(summary['mean_confidence'])} |",
        f"| Retrieval Backend | {summary['retrieval_backend']} |",
        "",
        "## Per-Example Results",
        "",
        "| Example | Question | Expected | Predicted | Expected Role | Predicted Role | Exact | Token F1 | Role |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for record in evaluation["records"]:
        lines.append(
            "| {example_id} | {question} | `{expected_answer}` | `{predicted_answer}` | `{expected_role}` | `{predicted_role}` | {exact} | {f1} | {role} |".format(
                example_id=record["example_id"],
                question=record["question"],
                expected_answer=record["expected_answer"],
                predicted_answer=record["predicted_answer"],
                expected_role=record["expected_role"],
                predicted_role=record["predicted_role"],
                exact=format_percent(record["exact_match"]),
                f1=format_percent(record["token_f1"]),
                role=format_percent(record["role_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "- Use this table to show the final demo path is functional.",
            "- Use the legacy full-test metrics for full-corpus baseline claims.",
            "- Use the RAISE seed-suite metrics only for curated-suite claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled SRL + RAG demo evaluation.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    args = parser.parse_args()

    evaluation = run_demo_evaluation()
    if args.json:
        print(json.dumps(evaluation, indent=2))
    else:
        print(format_markdown_report(evaluation))


if __name__ == "__main__":
    main()
