"""Streamlit research app for the upgraded hybrid SRL-QA project."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from benchmark import load_latest_benchmark
from config import get_config
from hybrid_qa import HybridQASystem, load_challenge_suite

CONFIG = get_config()
PROJECT_ROOT = CONFIG.paths.project_root
WORKSPACE_ROOT = PROJECT_ROOT.parent
NEW_SRLQA_ROOT = WORKSPACE_ROOT / "srlqa"
if NEW_SRLQA_ROOT.exists() and str(NEW_SRLQA_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_SRLQA_ROOT))

try:
    from srlqa.model_hub import ModelHub, model_choices, model_labels
except Exception:  # pragma: no cover - optional new package may be absent
    ModelHub = None
    model_choices = None
    model_labels = None

LITERATURE_REFERENCES = [
    {
        "title": "Large-Scale QA-SRL Parsing",
        "venue_year": "ACL 2018",
        "url": "https://aclanthology.org/P18-1191/",
        "contribution": "Established large-scale QA-SRL parsing and motivates question-answer supervision over predicate structure.",
    },
    {
        "title": "PropBank Comes of Age: Larger, Smarter, and more Diverse",
        "venue_year": "*SEM 2022",
        "url": "https://aclanthology.org/2022.starsem-1.24/",
        "contribution": "Documents the modern PropBank resource used as the semantic backbone for this project.",
    },
    {
        "title": "Potential and Limitations of LLMs in Capturing Structured Semantics: A Case Study on SRL",
        "venue_year": "arXiv 2024",
        "url": "https://arxiv.org/abs/2405.06410",
        "contribution": "Frames where LLM-style reasoning helps SRL and where deterministic structure is still needed.",
    },
    {
        "title": "LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models",
        "venue_year": "ACL Findings 2025",
        "url": "https://aclanthology.org/2025.findings-acl.1189/",
        "contribution": "Provides a recent anchor for the project's LLM-assisted reasoning discussion and future work framing.",
    },
]


def load_json(path: Path, default: Any) -> Any:
    """Load JSON from disk with a fallback default."""

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def top_items(mapping: Dict[str, Any], limit: int = 12) -> pd.DataFrame:
    """Convert a frequency mapping into a sorted dataframe."""

    rows = sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]
    return pd.DataFrame(rows, columns=["label", "count"])


def render_table(data: pd.DataFrame) -> None:
    """Render a dataframe using the non-deprecated Streamlit width API."""

    normalized = data.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else ""
                    if pd.isna(value)
                    else str(value)
                )
            )
    st.dataframe(normalized, width="stretch", hide_index=True)


def render_image(image_path: Path, caption: str) -> None:
    """Render an image using the non-deprecated Streamlit width API."""

    st.image(str(image_path), width="stretch", caption=caption)


@st.cache_resource(show_spinner=False)
def get_hybrid_system() -> HybridQASystem:
    """Cache the hybrid system so the app stays responsive."""

    return HybridQASystem(CONFIG, use_transformer_qa=True, use_sentence_embeddings=True)


@st.cache_resource(show_spinner=False)
def get_all_model_hub() -> Any:
    """Cache the all-model hub from the new RAISE-SRL-QA package."""

    if ModelHub is None:
        return None
    return ModelHub()


def highlight_answer(context: str, answer: str) -> str:
    """Highlight the first matching answer span in the context."""

    escaped_context = html.escape(context)
    if not answer.strip():
        return escaped_context
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    return pattern.sub(lambda match: f"<mark>{html.escape(match.group(0))}</mark>", escaped_context, count=1)


def render_metric_row(prediction: Dict[str, Any]) -> None:
    """Show the core answer metrics."""

    st.success(f"Answer: {prediction['hybrid_answer'] or 'No answer produced'}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Role", prediction["role"])
    col2.metric("Confidence", f"{prediction['confidence']:.3f}")
    col3.metric("Predicate", prediction["predicate"] or "N/A")
    col4.metric("Question Type", prediction["question_type"])
    col5.metric("Latency (ms)", f"{prediction['latency_ms']:.1f}")
    availability = prediction.get("model_availability", {})
    enabled = [name for name, is_ready in availability.items() if is_ready]
    st.caption(
        "Enabled local helpers: "
        + (", ".join(enabled) if enabled else "baseline-only fallback")
    )


def render_ask_section(challenge_suite: List[Dict[str, Any]]) -> None:
    """Render the chat-style QA interface."""

    st.title("Ask the Model")
    st.caption("Question answering with PropQA-Net baseline signals, role-aware heuristics, and optional local transformer support.")

    labels = ["Custom question"] + [f"{index + 1}. {item['question']}" for index, item in enumerate(challenge_suite)]
    selected_label = st.selectbox("Sample Questions", labels)
    selected_example = None
    if selected_label != "Custom question":
        selected_index = labels.index(selected_label) - 1
        selected_example = challenge_suite[selected_index]

    default_context = selected_example["context"] if selected_example else "The scientist explained the process with remarkable clarity."
    default_question = selected_example["question"] if selected_example else "How did the scientist explain the process?"

    context = st.text_area("Context", value=default_context, height=180)
    question = st.text_input("Question", value=default_question)

    with st.expander("Starter Question Bank", expanded=selected_example is None):
        starter_rows = [
            {
                "question": item["question"],
                "type": item["question_type"],
                "expected_role": item["target_role"],
                "gold_answer": item.get("expected_answer", item.get("answer", "")),
            }
            for item in challenge_suite
        ]
        render_table(pd.DataFrame(starter_rows))

    if st.button("Ask Hybrid Model", type="primary"):
        with st.spinner("Running hybrid inference..."):
            prediction = get_hybrid_system().answer_question(context, question)
        render_metric_row(prediction)
        st.markdown("**Reasoning Trace**")
        st.write(prediction["reasoning_summary"])
        st.markdown("**Inference Diagnostics**")
        render_table(
            pd.DataFrame(
                [
                    {
                        "field": "expected_role_from_question",
                        "value": prediction["expected_role"],
                    },
                    {
                        "field": "semantic_alignment",
                        "value": round(prediction["semantic_alignment"], 4),
                    },
                    {
                        "field": "baseline_confidence",
                        "value": round(prediction["baseline_confidence"], 4),
                    },
                ]
            )
        )
        st.markdown("**Context with Evidence Highlight**")
        st.markdown(
            f"<div style='padding:0.8rem;border:1px solid #d6d6d6;border-radius:0.6rem;background:#fafafa'>{highlight_answer(context, prediction['hybrid_answer'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Baseline vs Hybrid**")
        render_table(
            pd.DataFrame(
                [
                    {
                        "channel": "baseline",
                        "answer": prediction["baseline_answer"],
                        "role": prediction["baseline_role"],
                        "confidence": round(prediction["baseline_confidence"], 4),
                    },
                    {
                        "channel": "hybrid",
                        "answer": prediction["hybrid_answer"],
                        "role": prediction["role"],
                        "confidence": round(prediction["confidence"], 4),
                    },
                ]
            )
        )
        st.markdown("**Predicate-Argument Evidence Table**")
        evidence_df = pd.DataFrame(prediction["evidence_spans"])
        if not evidence_df.empty:
            render_table(evidence_df)
        else:
            st.info("No evidence spans were returned for this query.")


def render_all_model_qa_section(challenge_suite: List[Dict[str, Any]]) -> None:
    """Render model selection and all-model comparison."""

    st.title("All Model QA")
    st.caption("Run the legacy PropQA-Net baseline, legacy hybrid, RAISE fast, and RAISE model-backed pipelines from one screen.")
    if ModelHub is None or model_choices is None or model_labels is None:
        st.error("The new `srlqa` package was not found. Create/run the top-level `srlqa/` folder first.")
        return

    labels = ["Custom question"] + [f"{index + 1}. {item['question']}" for index, item in enumerate(challenge_suite)]
    selected_label = st.selectbox("Sample Questions", labels, key="all_model_sample")
    selected_example = None
    if selected_label != "Custom question":
        selected_index = labels.index(selected_label) - 1
        selected_example = challenge_suite[selected_index]

    default_context = selected_example["context"] if selected_example else "The courier delivered the package to the office at noon."
    default_question = selected_example["question"] if selected_example else "Where was the package delivered?"
    default_expected = selected_example.get("expected_answer", "") if selected_example else "to the office"

    context = st.text_area("Context", value=default_context, height=170, key="all_model_context")
    question = st.text_input("Question", value=default_question, key="all_model_question")
    expected_answer = st.text_input("Expected answer for recursive correction/evaluation (optional)", value=default_expected, key="all_model_expected")

    labels_by_key = model_labels(include_all=True)
    choice_keys = model_choices(include_all=True)
    selected_model_label = st.selectbox(
        "Model",
        [labels_by_key[key] for key in choice_keys],
        index=0,
        help="Choose All Models to compare every available system.",
    )
    selected_model = next(key for key in choice_keys if labels_by_key[key] == selected_model_label)

    if st.button("Run Selected Model(s)", type="primary"):
        hub = get_all_model_hub()
        if hub is None:
            st.error("Model hub unavailable.")
            return
        with st.spinner("Running model(s)... first run can download/load local weights."):
            results = hub.run(
                selected_model,
                context,
                question,
                expected_answer=expected_answer.strip() or None,
            )

        rows = []
        for result in results:
            rows.append(
                {
                    "model": result["model_label"],
                    "ok": result["ok"],
                    "answer": result["answer"],
                    "role": result["role"],
                    "confidence": round(float(result["confidence"]), 4),
                    "latency_ms": round(float(result["latency_ms"]), 1),
                    "error": result["error"],
                    "reasoning": result["reasoning"],
                }
            )
        render_table(pd.DataFrame(rows))

        st.markdown("**Context With Best Highlight**")
        best = max(results, key=lambda item: float(item["confidence"]) if item["ok"] else -1.0)
        st.markdown(
            f"<div style='padding:0.8rem;border:1px solid #d6d6d6;border-radius:0.5rem;background:#fafafa'>{highlight_answer(context, str(best.get('answer', '')))}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Raw model outputs"):
            st.json(results)


def render_architecture_section(plots_dir: Path) -> None:
    """Render the architecture diagrams."""

    st.title("Architecture")
    st.write(
        "The upgraded system keeps the classical PropQA-Net checkpoint as a reproducible SRL-QA backbone and layers role heuristics, transformer QA spans, semantic reranking, and a research-facing Streamlit interface on top."
    )
    for image_name in ["research_architecture.png", "propqa_architecture.png", "hybridpropqa.png", "srl_pipeline.png"]:
        image_path = plots_dir / image_name
        if image_path.exists():
            render_image(image_path, image_name.replace("_", " ").replace(".png", "").title())


def render_dataset_section(stats: Dict[str, Any], test_examples: List[Dict[str, Any]]) -> None:
    """Render the dataset explorer."""

    st.title("Dataset & PropBank Explorer")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PropBank Instances", f"{stats.get('total_propbank_instances', 0):,}")
    col2.metric("Usable Instances", f"{stats.get('usable_propbank_instances', 0):,}")
    col3.metric("QA Pairs", f"{stats.get('qa_pair_count', 0):,}")
    col4.metric("Unique Rolesets", f"{stats.get('unique_rolesets', 0):,}")

    st.markdown("**Question-Type Distribution**")
    if stats.get("qa_pairs_per_question_type"):
        question_type_df = top_items(stats["qa_pairs_per_question_type"], limit=20)
        render_table(question_type_df.rename(columns={"label": "question_type"}))

    st.markdown("**Top Argument Types**")
    if stats.get("argument_type_distribution"):
        role_df = top_items(stats["argument_type_distribution"], limit=15)
        render_table(role_df)

    st.markdown("**Sample PropBank-Derived QA Pairs**")
    sample_rows = [
        {
            "question": example["question"],
            "answer": example["answer_text"],
            "role": example["target_role"],
            "predicate": example["predicate_text"],
        }
        for example in test_examples[:12]
    ]
    render_table(pd.DataFrame(sample_rows))

    st.markdown("**Corpus Snapshot**")
    if stats.get("split_sizes"):
        split_df = pd.DataFrame(
            [{"split": key, "count": value} for key, value in stats["split_sizes"].items()]
        )
        render_table(split_df)


def render_experiments_section(plots_dir: Path, benchmark_payload: Dict[str, Any] | None) -> None:
    """Render the experiment dashboard."""

    st.title("Experiments")
    if benchmark_payload is None:
        st.warning("No benchmark results found yet. Run `python main.py --mode benchmark` to generate them.")
        return

    summary_rows = []
    for track_name, payload in benchmark_payload["tracks"].items():
        summary_rows.append(
            {
                "track": track_name,
                "challenge_em": round(payload["challenge"]["exact_match"], 4),
                "challenge_f1": round(payload["challenge"]["token_f1"], 4),
                "test_em": round(payload["test_subset"]["exact_match"], 4),
                "test_f1": round(payload["test_subset"]["token_f1"], 4),
                "mean_latency_ms": round(payload["combined"]["mean_latency_ms"], 2),
            }
        )
    render_table(pd.DataFrame(summary_rows))

    track_names = list(benchmark_payload["tracks"].keys())
    selected_track = st.selectbox("Inspect Track", track_names, index=track_names.index("full_hybrid") if "full_hybrid" in track_names else 0)
    selected_payload = benchmark_payload["tracks"][selected_track]

    st.markdown("**Per-Question-Type Metrics**")
    question_df = pd.DataFrame(
        [
            {
                "question_type": key,
                "exact_match": round(value["exact_match"], 4),
                "token_f1": round(value["token_f1"], 4),
                "count": int(value["count"]),
            }
            for key, value in selected_payload["combined"]["per_question_type"].items()
        ]
    ).sort_values("question_type")
    render_table(question_df)

    st.markdown("**Per-Role Metrics**")
    role_df = pd.DataFrame(
        [
            {
                "role": key,
                "role_accuracy": round(value["role_accuracy"], 4),
                "token_f1": round(value["token_f1"], 4),
                "count": int(value["count"]),
            }
            for key, value in selected_payload["combined"]["per_role"].items()
        ]
    ).sort_values("role")
    render_table(role_df)

    st.markdown("**Challenge Suite Samples**")
    sample_df = pd.DataFrame(selected_payload["combined"]["samples"])
    sample_columns = [
        "example_id",
        "question_type",
        "question",
        "expected_answer",
        "predicted_answer",
        "target_role",
        "predicted_role",
        "exact_match",
        "token_f1",
    ]
    render_table(sample_df[sample_columns])

    for image_name in [
        "ablation_summary.png",
        "latency_accuracy_tradeoff.png",
        "question_type_heatmap.png",
        "role_heatmap.png",
        "confidence_histogram.png",
        "dataset_balance.png",
        "challenge_table.png",
        "error_gallery.png",
    ]:
        image_path = plots_dir / image_name
        if image_path.exists():
            render_image(image_path, image_name.replace("_", " ").replace(".png", "").title())


def render_tradeoffs_section(benchmark_payload: Dict[str, Any] | None) -> None:
    """Render the trade-off discussion."""

    st.title("Tradeoffs")
    if benchmark_payload is not None:
        tradeoff_df = pd.DataFrame(
            [
                {
                    "track": name,
                    "exact_match": round(payload["combined"]["exact_match"], 4),
                    "token_f1": round(payload["combined"]["token_f1"], 4),
                    "role_accuracy": round(payload["combined"]["role_accuracy"], 4),
                    "mean_latency_ms": round(payload["combined"]["mean_latency_ms"], 2),
                    "load_time_sec": round(payload["combined"]["load_time_sec"], 2),
                }
                for name, payload in benchmark_payload["tracks"].items()
            ]
        )
        render_table(tradeoff_df)

    st.markdown(
        """
        - The classical baseline is the most reproducible component and remains the authoritative SRL backbone.
        - The heuristic reranker improves role-sensitive questions quickly, but still depends on surface cues.
        - Transformer QA adds stronger span proposals, but first-run downloads and CPU latency increase local cost.
        - Sentence embeddings help semantic matching and explanation quality, but they are still subordinate to role-aware evidence selection.
        - The project intentionally treats literature numbers as references and local results as the official reproducible findings.
        """
    )


def render_documentation_section(stats: Dict[str, Any], benchmark_payload: Dict[str, Any] | None) -> None:
    """Render the long-form website documentation."""

    st.title("Documentation")
    sections = {
        "1. Motivation": "Traditional extractive QA often returns a span without revealing the event slot it represents. This project reframes QA as semantic slot filling over PropBank predicate-argument structures so that answers are both extractive and explainable.",
        "2. Prior Work": "The documentation follows the uploaded presentation structure: classical SRL, neural SRL, QA-SRL, QA-based semantics, and recent 2024-2025 work on LLM-guided structured semantics. Literature claims are shown as external references, not local reproduced scores.",
        "3. PropBank + NLTK": "The repo uses the bundled `nltk_data/` folder to reconstruct authentic PropBank examples offline. Only Treebank-backed instances are kept so answer spans remain deterministic and reproducible.",
        "4. Hybrid System": "The final system combines baseline PropQA-Net predictions, improved question-role parsing, role-aware candidate extraction, transformer span proposals, semantic reranking, and short explanation traces.",
        "5. Experiments": "The benchmark suite compares four tracks: classical baseline, heuristic reranker, transformer QA assist, and full hybrid. The app and PDFs present these side by side with latency, EM, token-F1, role accuracy, and per-question-type analysis.",
        "6. Results and Limits": "Repo metrics remain the official baseline. The upgraded hybrid layer is evaluated as an inference-time improvement, not a replacement of the original checkpoint. This keeps the project transparent about what was trained, what was added later, and what was measured locally.",
    }
    for title, body in sections.items():
        with st.expander(title, expanded=title == "1. Motivation"):
            st.write(body)

    st.markdown("**Core Project Snapshot**")
    snapshot_rows = [
        {"item": "Usable PropBank instances", "value": f"{stats.get('usable_propbank_instances', 0):,}"},
        {"item": "QA pairs", "value": f"{stats.get('qa_pair_count', 0):,}"},
        {"item": "Unique predicates", "value": f"{stats.get('unique_predicates', 0):,}"},
        {"item": "Unique rolesets", "value": f"{stats.get('unique_rolesets', 0):,}"},
    ]
    if benchmark_payload is not None and "full_hybrid" in benchmark_payload["tracks"]:
        combined = benchmark_payload["tracks"]["full_hybrid"]["combined"]
        snapshot_rows.extend(
            [
                {"item": "Hybrid combined exact match", "value": f"{combined['exact_match']:.4f}"},
                {"item": "Hybrid combined token-F1", "value": f"{combined['token_f1']:.4f}"},
                {"item": "Hybrid role accuracy", "value": f"{combined['role_accuracy']:.4f}"},
            ]
        )
    render_table(pd.DataFrame(snapshot_rows))

    st.markdown("**Key Innovations**")
    innovation_rows = [
        {
            "innovation": "Role-aware question parsing",
            "impact": "Maps who/what/when/where/how/why/to-whom questions to SRL roles before answer selection.",
        },
        {
            "innovation": "Heuristic evidence extraction",
            "impact": "Adds robust local coverage for ARGM-TMP, ARGM-LOC, ARGM-MNR, ARGM-CAU, and ARG2 recipient cases.",
        },
        {
            "innovation": "Transformer span assist",
            "impact": "Uses a compact local SQuAD-style QA model to propose answer spans without replacing the baseline checkpoint.",
        },
        {
            "innovation": "Semantic reranking",
            "impact": "Uses sentence embeddings to prefer evidence that best matches the question intent and predicate context.",
        },
        {
            "innovation": "Deterministic reasoning trace",
            "impact": "Explains which evidence source won and why, while keeping answer selection evidence-based rather than free-form.",
        },
    ]
    render_table(pd.DataFrame(innovation_rows))

    with st.expander("Reproducibility Workflow", expanded=False):
        st.code(
            "\n".join(
                [
                    "python main.py --mode ask --engine hybrid",
                    "python main.py --mode benchmark --max-examples 40",
                    "python main.py --mode report",
                    "python main.py --mode app --port 8501",
                ]
            ),
            language="bash",
        )
        st.write(
            "The benchmark and report modes regenerate JSON summaries, plots, and PDF deliverables from the current local environment."
        )

    with st.expander("Research Anchors", expanded=False):
        for item in LITERATURE_REFERENCES:
            st.markdown(
                f"- [{item['title']}]({item['url']}) ({item['venue_year']}): {item['contribution']}"
            )


def render_downloads_section(outputs_dir: Path) -> None:
    """Render download buttons for generated outputs."""

    st.title("Downloads")
    download_targets = [
        outputs_dir / "survey.pdf",
        outputs_dir / "analysis.pdf",
        outputs_dir / "innovation.pdf",
        outputs_dir / "research_paper.pdf",
        outputs_dir / "implementation_code.py",
        CONFIG.paths.results_dir / "metrics.json",
        CONFIG.paths.results_dir / "data_statistics.json",
        CONFIG.paths.results_dir / "benchmarks" / "benchmark_results.json",
    ]
    for path in download_targets:
        file_name = path.name
        if not path.exists():
            continue
        st.download_button(
            label=f"Download {file_name}",
            data=path.read_bytes(),
            file_name=file_name,
            mime="application/json" if path.suffix == ".json" else "application/octet-stream",
        )


def main() -> None:
    """Launch the multi-page Streamlit research dashboard."""

    st.set_page_config(page_title="Hybrid SRL-QA Research App", page_icon="📘", layout="wide")
    challenge_suite = load_challenge_suite(PROJECT_ROOT)
    stats = load_json(CONFIG.paths.results_dir / "data_statistics.json", {})
    test_examples = load_json(CONFIG.paths.test_json, [])
    benchmark_payload = load_latest_benchmark(CONFIG)

    st.sidebar.title("Hybrid SRL-QA")
    st.sidebar.caption("Research-grade Streamlit interface for the upgraded PropBank SRL-QA project.")
    st.sidebar.markdown("**Baseline Snapshot**")
    st.sidebar.write(
        f"{stats.get('qa_pair_count', 0):,} QA pairs from {stats.get('usable_propbank_instances', 0):,} usable PropBank instances."
    )
    if benchmark_payload is not None and "full_hybrid" in benchmark_payload["tracks"]:
        combined = benchmark_payload["tracks"]["full_hybrid"]["combined"]
        st.sidebar.write(
            f"Hybrid combined EM {combined['exact_match']:.3f}, token-F1 {combined['token_f1']:.3f}, role accuracy {combined['role_accuracy']:.3f}."
        )
    section = st.sidebar.radio(
        "Navigate",
        [
            "All Model QA",
            "Ask the Model",
            "Architecture",
            "Dataset & PropBank Explorer",
            "Experiments",
            "Tradeoffs",
            "Documentation",
            "Downloads",
        ],
    )

    if section == "All Model QA":
        render_all_model_qa_section(challenge_suite)
    elif section == "Ask the Model":
        render_ask_section(challenge_suite)
    elif section == "Architecture":
        render_architecture_section(CONFIG.paths.plots_dir)
    elif section == "Dataset & PropBank Explorer":
        render_dataset_section(stats, test_examples)
    elif section == "Experiments":
        render_experiments_section(CONFIG.paths.plots_dir, benchmark_payload)
    elif section == "Tradeoffs":
        render_tradeoffs_section(benchmark_payload)
    elif section == "Documentation":
        render_documentation_section(stats, benchmark_payload)
    else:
        render_downloads_section(CONFIG.paths.outputs_dir)


if __name__ == "__main__":
    main()
