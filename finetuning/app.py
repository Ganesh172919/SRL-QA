from __future__ import annotations

import json
import string
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"In the future `np\.object` will be defined as the corresponding NumPy scalar\.",
    category=FutureWarning,
)

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def bootstrap_streamlit() -> None:
    if __name__ != "__main__" or st.runtime.exists():
        return

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=str(ROOT), check=False).returncode)


bootstrap_streamlit()

from qasrl_cpu.data import build_input_text, mark_predicate
from qasrl_cpu.instashap import InstaShapExplainer
from qasrl_cpu.inference import predict_single
from qasrl_cpu.modeling import load_trained_model


def find_predicate_index(sentence: str, predicate: str) -> int:
    tokens = sentence.split()
    normalized_predicate = predicate.strip().lower().strip(string.punctuation)
    for index, token in enumerate(tokens):
        if token.lower().strip(string.punctuation) == normalized_predicate:
            return index
    return 0


@st.cache_resource
def load_resources(model_dir: str):
    return load_trained_model(model_dir)


def render_app() -> None:
    st.set_page_config(page_title="QA-SRL InstaShap Demo", layout="wide")
    st.title("QA-SRL Fine-Tuning + InstaShap")
    st.caption("CPU-friendly QA-SRL parser fine-tuned on QA-SRL Bank 2.1 and explained with InstaShap-style token attributions.")

    default_model_dir = str(ROOT / "artifacts" / "flan_t5_small_lora")
    model_dir = st.sidebar.text_input("Model directory", value=default_model_dir)
    results_path = ROOT / "results" / "evaluation_report.json"

    if results_path.exists():
        report = json.loads(results_path.read_text(encoding="utf-8"))
        st.sidebar.subheader("Latest evaluation")
        st.sidebar.write(report["fine_tuned_metrics"])
        st.sidebar.write(report["xai_metrics"])

    if not Path(model_dir).exists():
        st.warning("Train the model first with `python run_project.py` or `python train.py`.")
        st.stop()

    model, tokenizer, metadata = load_resources(model_dir)

    sentence = st.text_area(
        "Sentence",
        value="The company successfully launched its new product last Tuesday.",
        height=120,
    )
    predicate = st.text_input("Predicate", value="launched")

    if st.button("Analyze"):
        predicate_idx = find_predicate_index(sentence, predicate)
        tokens = sentence.split()
        marked_sentence = mark_predicate(tokens, predicate_idx)
        input_text = build_input_text(sentence, predicate, marked_sentence)
        prediction = predict_single(model, tokenizer, sentence, predicate, input_text)
        explainer = InstaShapExplainer(model, tokenizer)
        explanation = explainer.explain(input_text, prediction["prediction_text"] or "")
        fig = explainer.plot(explanation)

        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            st.subheader("Structured prediction")
            st.code(prediction["prediction_text"] or "No roles predicted", language="text")
            st.subheader("Natural-language QA pairs")
            if prediction["qa_pairs"]:
                for qa in prediction["qa_pairs"]:
                    st.markdown(f"**{qa['role']}**  \nQ: {qa['question']}  \nA: {qa['answer']}")
            else:
                st.write("No QA pairs were produced.")
        with col2:
            st.subheader("InstaShap explanation")
            st.pyplot(fig, clear_figure=True)
            st.write({"confidence": prediction["confidence"], "training_metadata": metadata})


if st.runtime.exists():
    render_app()
