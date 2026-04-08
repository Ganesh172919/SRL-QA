"""Hybrid local retrieval over SRL-structured documents."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import DemoConfig
    from .data_models import RetrievalHit, SRLDocument
except ImportError:  # pragma: no cover
    from config import DemoConfig
    from data_models import RetrievalHit, SRLDocument


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


@dataclass(slots=True)
class RetrievalStatus:
    backend: str
    embedding_ready: bool
    embedding_error: str = ""


class SRLRetriever:
    """TF-IDF plus optional sentence-transformer retrieval."""

    def __init__(self, config: DemoConfig, mode: str = "Hybrid") -> None:
        self.config = config
        self.mode = mode
        self.documents: list[SRLDocument] = []
        self.texts: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: object | None = None
        self.embedder: object | None = None
        self.embedding_matrix: np.ndarray | None = None
        self.embedding_error = ""
        self.backend = "not-fit"

    def fit(self, documents: list[SRLDocument]) -> RetrievalStatus:
        self.documents = documents
        self.texts = [document.retrieval_text() for document in documents]
        if not self.texts:
            self.backend = "empty"
            return RetrievalStatus(self.backend, False, "No documents were available.")

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        self.backend = "tfidf"

        if self.mode.lower().startswith("tf-idf"):
            return RetrievalStatus(self.backend, False)

        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.config.embedding_model_name)
            self.embedding_matrix = np.asarray(
                self.embedder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)
            )
            self.backend = "embeddings" if self.mode.lower().startswith("embedding") else "hybrid"
            return RetrievalStatus(self.backend, True)
        except Exception as exc:  # pragma: no cover - environment and cache dependent
            self.embedding_error = f"{type(exc).__name__}: {exc}"
            self.backend = "tfidf-fallback"
            if self.mode.lower().startswith("embedding"):
                self.backend = "tfidf-fallback-from-embeddings"
            return RetrievalStatus(self.backend, False, self.embedding_error)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not self.documents or self.vectorizer is None or self.tfidf_matrix is None:
            return []
        tfidf_query = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix).reshape(-1)

        if self.embedding_matrix is not None and self.embedder is not None:
            embedding_query = np.asarray(
                self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
            )
            embedding_scores = (embedding_query @ self.embedding_matrix.T).reshape(-1)
            if self.backend == "hybrid":
                scores = (0.65 * embedding_scores) + (0.35 * tfidf_scores)
            else:
                scores = embedding_scores
        else:
            scores = tfidf_scores

        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][: max(1, top_k)]
        return [
            RetrievalHit(
                document=self.documents[int(index)],
                score=float(scores[int(index)]),
                backend=self.backend,
                rank=rank,
            )
            for rank, index in enumerate(top_indices, start=1)
            if float(scores[int(index)]) > 0.0 or rank == 1
        ]
