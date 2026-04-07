"""Streamlit app for the standalone RAISE-SRL-QA system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from srlqa.config import get_config
from srlqa.evaluation.span_metrics import exact_match, token_f1
from srlqa.model_hub import ModelHub, model_choices, model_labels


CONFIG = get_config()


@st.cache_resource(show_spinner=False)
def get_hub() -> ModelHub:
    return ModelHub(CONFIG)


def render_table(rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)


def load_challenge_suite() -> list[dict[str, Any]]:
    path = CONFIG.paths.challenge_suite_path
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def render_single_question() -> None:
    st.header("Ask RAISE-SRL-QA")
    examples = load_challenge_suite()
    labels = ["Custom"] + [f"{index + 1}. {item['question']}" for index, item in enumerate(examples)]
    selected = st.selectbox("Example", labels)
    example = None
    if selected != "Custom":
        example = examples[labels.index(selected) - 1]

    context = st.text_area(
        "Context",
        value=example["context"] if example else "The courier delivered the package to the office at noon.",
        height=160,
    )
    question = st.text_input(
        "Question",
        value=example["question"] if example else "Where was the package delivered?",
    )
    expected = st.text_input(
        "Expected answer for recursive correction/evaluation (optional)",
        value=example.get("expected_answer", "") if example else "",
    )

    label_map = model_labels(include_all=True)
    keys = model_choices(include_all=True)
    selected_label = st.selectbox(
        "Model mode",
        [label_map[key] for key in keys],
        index=keys.index("raise_srlqa_model"),
    )
    selected_key = next(key for key in keys if label_map[key] == selected_label)

    if st.button("Run", type="primary"):
        with st.spinner("Running RAISE-SRL-QA..."):
            results = get_hub().run(
                selected_key,
                context,
                question,
                expected_answer=expected.strip() or None,
            )
        render_table(
            [
                {
                    "model": item["model_label"],
                    "answer": item["answer"],
                    "role": item["role"],
                    "confidence": round(float(item["confidence"]), 4),
                    "latency_ms": round(float(item["latency_ms"]), 1),
                    "ok": item["ok"],
                    "error": item["error"],
                }
                for item in results
            ]
        )
        with st.expander("Detailed reasoning and correction history", expanded=True):
            st.json(results)


def render_challenge_demo() -> None:
    st.header("Challenge Suite")
    examples = load_challenge_suite()
    max_examples = st.slider("Examples", 1, max(len(examples), 1), min(15, max(len(examples), 1)))
    use_all = st.checkbox("Compare all models", value=False)
    model_key = "all" if use_all else "raise_srlqa_model"
    if st.button("Run Challenge Suite"):
        records = []
        with st.spinner("Running benchmark demo..."):
            hub = get_hub()
            for example in examples[:max_examples]:
                for result in hub.run(
                    model_key,
                    example["context"],
                    example["question"],
                    expected_answer=example["expected_answer"],
                ):
                    records.append(
                        {
                            "id": example["id"],
                            "model": result["model_label"],
                            "question": example["question"],
                            "expected": example["expected_answer"],
                            "answer": result["answer"],
                            "exact_match": exact_match(str(result["answer"]), example["expected_answer"]),
                            "token_f1": token_f1(str(result["answer"]), example["expected_answer"]),
                            "role": result["role"],
                            "confidence": round(float(result["confidence"]), 4),
                            "latency_ms": round(float(result["latency_ms"]), 1),
                        }
                    )
        if records:
            st.metric("Mean Exact Match", f"{sum(row['exact_match'] for row in records) / len(records):.3f}")
            st.metric("Mean Token F1", f"{sum(row['token_f1'] for row in records) / len(records):.3f}")
            render_table(records)


def render_project_notes() -> None:
    st.header("How It Works")
    st.markdown(
        """
        RAISE-SRL-QA answers questions by combining:

        - extractive QA candidates from a DeBERTa/SQuAD-style reader,
        - deterministic SRL span candidates for common roles,
        - PropBank frame retrieval for predicate-role compatibility,
        - verifier scoring, and
        - recursive correction when a candidate is known to be wrong during evaluation.

        The system never lets the verifier invent a free-form answer. It only
        chooses among spans that occur in the context.
        """
    )
    st.markdown("**Useful commands**")
    st.code(
        "\n".join(
            [
                'python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"',
                "python -m srlqa.main demo --max-examples 15",
                'python -m srlqa.main chat --context "The nurse administered the medicine to the patient after dinner."',
                "streamlit run raise_streamlit_app.py",
            ]
        ),
        language="powershell",
    )


def main() -> None:
    st.set_page_config(page_title="RAISE-SRL-QA", page_icon="R", layout="wide")
    st.title("RAISE-SRL-QA")
    st.caption("Retrieval-Augmented, Iteratively Self-correcting, Explainable SRL Question Answering")

    tab_ask, tab_challenge, tab_notes = st.tabs(["Ask", "Challenge Demo", "Project Notes"])
    with tab_ask:
        render_single_question()
    with tab_challenge:
        render_challenge_demo()
    with tab_notes:
        render_project_notes()


if __name__ == "__main__":
    main()
