"""Streamlit app for SRL + RAG explainable QA."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from data_models import SRLDocument
from frame_store import FrameStore
from graphing import build_reasoning_graph, graph_to_json, graph_to_plotly
from propbank_loader import inspect_corpus, load_or_build_propbank_documents
from qa import answer_question
from retrieval import SRLRetriever
from user_docs import build_user_documents


CONFIG = get_config()
DEFAULT_USER_TEXT = "The courier delivered the package to the office at noon."


@st.cache_resource(show_spinner=False)
def load_frame_store() -> FrameStore:
    return FrameStore.load(CONFIG.frame_store_path)


@st.cache_data(show_spinner=False)
def load_propbank(limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents, stats = load_or_build_propbank_documents(CONFIG, load_frame_store(), limit=limit, use_cache=True)
    return [document.to_dict() for document in documents], stats


@st.cache_data(show_spinner=False)
def corpus_overview() -> dict[str, Any]:
    return inspect_corpus(CONFIG)


def decode_uploads(uploaded_files: list[Any] | None) -> list[str]:
    texts: list[str] = []
    for uploaded in uploaded_files or []:
        try:
            texts.append(uploaded.read().decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return texts


def dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    return pd.DataFrame(normalized)


def run_pipeline(
    question: str,
    propbank_limit: int,
    user_text: str,
    uploaded_files: list[Any] | None,
    mode: str,
    top_k: int,
    use_transformer: bool,
) -> None:
    frame_store = load_frame_store()
    propbank_payloads, stats = load_propbank(propbank_limit)
    propbank_docs = [SRLDocument.from_dict(payload) for payload in propbank_payloads]

    user_texts = [user_text] if user_text.strip() else []
    user_texts.extend(decode_uploads(uploaded_files))
    user_docs = build_user_documents(user_texts, frame_store)
    documents = propbank_docs + user_docs

    retriever = SRLRetriever(CONFIG, mode=mode)
    retrieval_status = retriever.fit(documents)
    hits = retriever.search(question, top_k=top_k)
    result = answer_question(question, hits, frame_store, CONFIG, use_transformer=use_transformer)
    graph = build_reasoning_graph(question, hits, result)

    st.session_state["last_run"] = {
        "question": question,
        "stats": stats,
        "retrieval_status": retrieval_status,
        "hits": hits,
        "result": result,
        "graph": graph,
        "graph_json": graph_to_json(graph),
        "doc_count": len(documents),
        "user_doc_count": len(user_docs),
    }


def render_qa_tab() -> None:
    st.header("Ask With SRL + RAG")
    st.write("Ground answers in PropBank roles, retrieved evidence, and a visible reasoning graph.")

    question = st.text_input("Question", value="Where was the package delivered?")
    user_text = st.text_area("Paste demo documents", value=DEFAULT_USER_TEXT, height=160)
    uploaded_files = st.file_uploader("Upload text documents", type=["txt", "md"], accept_multiple_files=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    propbank_limit = col_a.number_input(
        "PropBank sample size",
        min_value=25,
        max_value=2500,
        value=CONFIG.default_propbank_limit,
        step=25,
    )
    top_k = col_b.slider("Retrieved documents", 1, 10, CONFIG.default_top_k)
    mode = col_c.selectbox("Retrieval mode", ["Hybrid", "Embeddings", "TF-IDF fallback"])
    use_transformer = col_d.checkbox("Optional transformer QA", value=False)

    if st.button("Run SRL + RAG QA", type="primary"):
        with st.spinner("Building SRL corpus, retrieving evidence, and answering..."):
            run_pipeline(question, int(propbank_limit), user_text, uploaded_files, mode, int(top_k), use_transformer)

    last = st.session_state.get("last_run")
    if not last:
        st.info("Run a question to see the answer, retrieved evidence, and graph.")
        return

    result = last["result"]
    retrieval_status = last["retrieval_status"]
    st.subheader("Answer")
    st.success(result.answer or "No answer found")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{result.confidence:.3f}")
    col2.metric("Role", result.role or "N/A")
    col3.metric("Predicate", result.predicate or "N/A")
    col4.metric("Retrieval", retrieval_status.backend)
    if retrieval_status.embedding_error:
        st.warning(f"Embedding backend unavailable, using TF-IDF fallback: {retrieval_status.embedding_error}")
    st.markdown("**Evidence**")
    st.write(result.evidence_text or "No evidence text available.")
    st.markdown("**Reasoning**")
    st.write(" -> ".join(result.reasoning))


def render_corpus_tab() -> None:
    st.header("Corpus And Index Status")
    overview = corpus_overview()
    col1, col2, col3 = st.columns(3)
    col1.metric("PropBank instances", f"{overview.get('total_instances', 0):,}")
    col2.metric("Treebank files", f"{overview.get('treebank_file_count', 0):,}")
    col3.metric("Treebank-backed", f"{overview.get('usable_treebank_backed', 0):,}")
    st.write("Local NLTK data:", overview.get("nltk_data_dir", ""))
    st.write("Frame store:", str(CONFIG.frame_store_path))
    if CONFIG.frame_store_path.exists():
        st.write("Frame store size:", f"{CONFIG.frame_store_path.stat().st_size:,} bytes")

    last = st.session_state.get("last_run")
    if last:
        st.subheader("Last Run Index")
        st.json(
            {
                "total_indexed_documents": last["doc_count"],
                "user_documents": last["user_doc_count"],
                "propbank_stats": last["stats"],
                "retrieval_backend": last["retrieval_status"].backend,
            }
        )


def render_evidence_tab() -> None:
    st.header("Retrieved SRL Evidence")
    last = st.session_state.get("last_run")
    if not last:
        st.info("Run a question first.")
        return
    hits = last["hits"]
    if hits:
        st.dataframe(dataframe([hit.to_dict() for hit in hits]), width="stretch", hide_index=True)
    result = last["result"]
    if result.candidates:
        st.subheader("Answer Candidates")
        st.dataframe(dataframe([candidate.to_dict() for candidate in result.candidates]), width="stretch", hide_index=True)


def render_graph_tab() -> None:
    st.header("Explainable Semantic Graph")
    last = st.session_state.get("last_run")
    if not last:
        st.info("Run a question first.")
        return
    st.plotly_chart(graph_to_plotly(last["graph"]), use_container_width=True)
    st.download_button(
        "Download graph JSON",
        data=json.dumps(last["graph_json"], indent=2),
        file_name="srl_rag_reasoning_graph.json",
        mime="application/json",
    )
    with st.expander("Graph JSON"):
        st.json(last["graph_json"])


def main() -> None:
    st.set_page_config(page_title="SRL + RAG Explainable QA", page_icon="S", layout="wide")
    st.title("SRL + RAG Explainable QA")
    st.caption("Local PropBank SRL, hybrid retrieval, role-aware QA, and graph reasoning.")

    tab_qa, tab_corpus, tab_evidence, tab_graph = st.tabs(
        ["QA", "Corpus / Index", "Retrieved SRL Evidence", "Explainable Graph Reasoning"]
    )
    with tab_qa:
        render_qa_tab()
    with tab_corpus:
        render_corpus_tab()
    with tab_evidence:
        render_evidence_tab()
    with tab_graph:
        render_graph_tab()


if __name__ == "__main__":
    main()
