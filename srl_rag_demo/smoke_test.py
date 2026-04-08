"""Smoke tests for the standalone SRL + RAG demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from frame_store import FrameStore
from graphing import build_reasoning_graph, graph_to_json
from propbank_loader import inspect_corpus, load_or_build_propbank_documents
from qa import answer_question
from retrieval import SRLRetriever
from user_docs import build_user_documents


def main() -> None:
    config = get_config()
    frame_store = FrameStore.load(config.frame_store_path)

    corpus_stats = inspect_corpus(config)
    print("PropBank instances:", corpus_stats["total_instances"])
    print("Treebank-backed:", corpus_stats["usable_treebank_backed"])
    assert corpus_stats["total_instances"] >= 100000
    assert corpus_stats["usable_treebank_backed"] >= 9000

    propbank_docs, build_stats = load_or_build_propbank_documents(config, frame_store, limit=40, use_cache=False)
    print("Built PropBank docs:", len(propbank_docs))
    assert propbank_docs
    assert build_stats["indexed_documents"] == len(propbank_docs)

    user_docs = build_user_documents(["The courier delivered the package to the office at noon."], frame_store)
    docs = propbank_docs + user_docs
    retriever = SRLRetriever(config, mode="TF-IDF fallback")
    status = retriever.fit(docs)
    hits = retriever.search("Where was the package delivered?", top_k=5)
    print("Retrieval backend:", status.backend)
    print("Hits:", len(hits))
    assert hits

    result = answer_question("Where was the package delivered?", hits, frame_store, config, use_transformer=False)
    print("Answer:", result.answer)
    print("Role:", result.role)
    assert result.answer
    assert "office" in result.answer.lower()

    graph = build_reasoning_graph("Where was the package delivered?", hits, result)
    graph_json = graph_to_json(graph)
    print("Graph nodes:", len(graph_json["nodes"]))
    print("Graph edges:", len(graph_json["edges"]))
    assert graph_json["nodes"]
    assert graph_json["edges"]
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
