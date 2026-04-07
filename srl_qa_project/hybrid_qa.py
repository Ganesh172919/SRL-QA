"""Hybrid SRL-QA inference with role-aware reranking and optional local models."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from nltk.tokenize.treebank import TreebankWordDetokenizer

from config import ProjectConfig
from evaluator import normalize_text
from qa_inference import InferenceEngine, infer_predicate_index, simple_lemmatize, simple_word_tokenize
from trainer import token_level_f1

DETOKENIZER = TreebankWordDetokenizer()
QUESTION_ROLE_MAP = {
    "who": ("WHO", "ARG0"),
    "what": ("WHAT", "ARG1"),
    "when": ("WHEN", "ARGM-TMP"),
    "where": ("WHERE", "ARGM-LOC"),
    "how": ("HOW", "ARGM-MNR"),
    "why": ("WHY", "ARGM-CAU"),
    "whom": ("TO-WHOM", "ARG2"),
}
STOPWORDS = {
    "a", "an", "the", "did", "do", "does", "was", "were", "is", "are", "am", "be", "been", "being",
    "will", "would", "should", "could", "can", "who", "what", "when", "where", "how", "why", "whom",
    "to", "for", "of", "in", "on", "at", "with", "by", "from", "after", "before", "during", "through",
    "their", "his", "her", "its", "our", "your", "my", "they", "them", "he", "she", "it", "we", "you", "i",
}
BOUNDARY_TOKENS = {",", ";", ":", ".", "?", "!", "and", "but", "or"}
TEMPORAL_MARKERS = {
    "today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "noon", "midnight", "morning", "evening", "afternoon", "night",
    "week", "month", "year", "hours", "hour", "days", "day",
}
LOCATION_PREPOSITIONS = {"in", "at", "on", "near", "inside", "outside", "through", "across", "into", "onto", "under", "over", "to"}
TEMPORAL_PREPOSITIONS = {"on", "in", "at", "during", "after", "before", "since", "until"}
TRANSFER_VERBS = {"give", "gave", "send", "sent", "hand", "handed", "deliver", "delivered", "offer", "offered", "award", "awarded", "grant", "granted", "present", "presented", "distribute", "distributed"}
ANIMATE_HINTS = {"friend", "intern", "patient", "family", "families", "committee", "rahul", "mary", "john", "maria", "buyer", "recipient", "supplier", "visitor", "visitors"}
INSTRUCTION_MODEL_ENV = "SRL_QA_ENABLE_REASONER"


@dataclass(slots=True)
class QuestionIntent:
    """Parsed intent for a natural-language question."""

    question_type: str
    expected_role: str
    predicate_hint: str
    target_terms: List[str]
    raw_question: str


@dataclass(slots=True)
class CandidateSpan:
    """Answer candidate proposed by one of the hybrid components."""

    text: str
    start_token: int
    end_token: int
    role: str
    source: str
    base_score: float
    char_start: int
    char_end: int
    features: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable form of the candidate."""

        payload = asdict(self)
        payload["score"] = payload.pop("final_score")
        return payload


@dataclass(slots=True)
class HybridPrediction:
    """Structured result returned by the hybrid system."""

    context: str
    question: str
    answer: str
    role: str
    confidence: float
    predicate: str
    evidence_spans: List[Dict[str, Any]]
    baseline_answer: str
    baseline_role: str
    baseline_confidence: float
    hybrid_answer: str
    reasoning_summary: str
    question_type: str
    expected_role: str
    latency_ms: float
    model_availability: Dict[str, bool]
    semantic_alignment: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert the prediction to a JSON-serializable payload."""

        return asdict(self)


def _detokenize(tokens: Sequence[str]) -> str:
    """Detokenize a token span into readable text."""

    return DETOKENIZER.detokenize(list(tokens)).strip()


def _token_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Map token indices back into character spans."""

    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def _safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero in small heuristic scores."""

    return numerator / denominator if denominator else 0.0


def _clamp(score: float) -> float:
    """Clamp a score into a conservative confidence interval."""

    return max(0.01, min(0.99, score))


class ExternalModelBundle:
    """Lazy loader for small optional local models."""

    def __init__(
        self,
        qa_model_name: str = "distilbert-base-cased-distilled-squad",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reasoning_model_name: str = "google/flan-t5-small",
    ) -> None:
        self.qa_model_name = qa_model_name
        self.embedding_model_name = embedding_model_name
        self.reasoning_model_name = reasoning_model_name
        self.qa_pipeline: Any | None = None
        self.embedder: Any | None = None
        self.reasoner: Any | None = None
        self.availability = {
            "transformer_qa": False,
            "sentence_embeddings": False,
            "local_reasoner": False,
        }
        self.load_messages: List[str] = []
        self.load_time_sec = 0.0
        self._embedding_cache: Dict[str, Any] = {}

    def ensure_qa_pipeline(self) -> Any | None:
        """Load the extractive QA pipeline if available."""

        if self.qa_pipeline is not None or self.availability["transformer_qa"]:
            return self.qa_pipeline

        start_time = time.perf_counter()
        try:
            from transformers import pipeline

            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model_name,
                tokenizer=self.qa_model_name,
                device=-1,
            )
            self.availability["transformer_qa"] = True
        except Exception as error:  # pragma: no cover - availability varies by environment
            self.load_messages.append(f"transformer_qa_unavailable: {error}")
            self.qa_pipeline = None
        self.load_time_sec += time.perf_counter() - start_time
        return self.qa_pipeline

    def ensure_embedder(self) -> Any | None:
        """Load the sentence embedding model if available."""

        if self.embedder is not None or self.availability["sentence_embeddings"]:
            return self.embedder

        start_time = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_model_name)
            self.availability["sentence_embeddings"] = True
        except Exception as error:  # pragma: no cover - availability varies by environment
            self.load_messages.append(f"sentence_embeddings_unavailable: {error}")
            self.embedder = None
        self.load_time_sec += time.perf_counter() - start_time
        return self.embedder

    def ensure_reasoner(self) -> Any | None:
        """Load a small reasoning model only when explicitly enabled."""

        if os.environ.get(INSTRUCTION_MODEL_ENV, "0") != "1":
            return None
        if self.reasoner is not None or self.availability["local_reasoner"]:
            return self.reasoner

        start_time = time.perf_counter()
        try:
            from transformers import pipeline

            self.reasoner = pipeline(
                "text2text-generation",
                model=self.reasoning_model_name,
                tokenizer=self.reasoning_model_name,
                device=-1,
            )
            self.availability["local_reasoner"] = True
        except Exception as error:  # pragma: no cover - availability varies by environment
            self.load_messages.append(f"local_reasoner_unavailable: {error}")
            self.reasoner = None
        self.load_time_sec += time.perf_counter() - start_time
        return self.reasoner

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Compute sentence-level similarity with a model or lexical fallback."""

        embedder = self.ensure_embedder()
        if embedder is None:
            return self.lexical_similarity(text_a, text_b)

        key_a = f"embed::{text_a}"
        key_b = f"embed::{text_b}"
        if key_a not in self._embedding_cache:
            self._embedding_cache[key_a] = embedder.encode(text_a, normalize_embeddings=True)
        if key_b not in self._embedding_cache:
            self._embedding_cache[key_b] = embedder.encode(text_b, normalize_embeddings=True)
        vector_a = self._embedding_cache[key_a]
        vector_b = self._embedding_cache[key_b]
        try:
            similarity = float(vector_a @ vector_b)
        except Exception:
            similarity = self.lexical_similarity(text_a, text_b)
        return _clamp((similarity + 1.0) / 2.0 if similarity < 0.0 else similarity)

    @staticmethod
    def lexical_similarity(text_a: str, text_b: str) -> float:
        """Compute a light lexical-overlap similarity score."""

        tokens_a = {simple_lemmatize(token) for token in simple_word_tokenize(text_a.lower()) if token.isalpha()}
        tokens_b = {simple_lemmatize(token) for token in simple_word_tokenize(text_b.lower()) if token.isalpha()}
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return _safe_divide(overlap, union)


class HybridQASystem:
    """Role-aware hybrid QA wrapper built on top of PropQA-Net."""

    def __init__(
        self,
        config: ProjectConfig,
        use_transformer_qa: bool = True,
        use_sentence_embeddings: bool = True,
        use_reasoner: bool = False,
    ) -> None:
        self.config = config
        self.baseline_engine = InferenceEngine(config)
        self.use_transformer_qa = use_transformer_qa
        self.use_sentence_embeddings = use_sentence_embeddings
        self.use_reasoner = use_reasoner
        self.external_models = ExternalModelBundle()
        self.load_time_sec = self.external_models.load_time_sec

    def answer_question(self, context: str, question: str) -> Dict[str, Any]:
        """Answer a question with a structured hybrid prediction."""

        start_time = time.perf_counter()
        cleaned_context = context.strip()
        cleaned_question = question.strip()
        if not cleaned_context:
            raise ValueError("Context must not be empty.")
        if not cleaned_question:
            raise ValueError("Question must not be empty.")

        intent = self._analyze_question(cleaned_question)
        baseline = self.baseline_engine.infer(cleaned_context, cleaned_question)
        tokens = simple_word_tokenize(cleaned_context)
        offsets = _token_offsets(cleaned_context, tokens)
        predicate_index = self._infer_predicate_index(tokens, intent)
        predicate_text = tokens[predicate_index] if tokens else ""

        candidates = self._generate_candidates(
            context=cleaned_context,
            question=cleaned_question,
            tokens=tokens,
            offsets=offsets,
            predicate_index=predicate_index,
            intent=intent,
            baseline=baseline,
        )
        scored_candidates = self._score_candidates(
            candidates=candidates,
            question=cleaned_question,
            baseline=baseline,
            intent=intent,
            predicate_text=predicate_text,
        )
        best_candidate = self._select_best_candidate(scored_candidates, baseline, tokens, offsets, intent)
        semantic_alignment = best_candidate.features.get("semantic_alignment", 0.0)
        reasoning_summary = self._build_reasoning_summary(
            question=cleaned_question,
            predicate_text=predicate_text,
            baseline=baseline,
            best_candidate=best_candidate,
            intent=intent,
        )
        if self.use_reasoner:
            reasoning_summary = self._maybe_rewrite_reasoning(reasoning_summary)

        prediction = HybridPrediction(
            context=cleaned_context,
            question=cleaned_question,
            answer=best_candidate.text,
            role=best_candidate.role,
            confidence=_clamp(best_candidate.final_score),
            predicate=predicate_text,
            evidence_spans=[candidate.as_dict() for candidate in scored_candidates[:3]],
            baseline_answer=baseline.answer_text,
            baseline_role=baseline.predicted_role,
            baseline_confidence=float(baseline.confidence),
            hybrid_answer=best_candidate.text,
            reasoning_summary=reasoning_summary,
            question_type=intent.question_type,
            expected_role=intent.expected_role,
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
            model_availability=dict(self.external_models.availability),
            semantic_alignment=semantic_alignment,
        )
        return prediction.to_dict()

    def answer_examples(self, examples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch convenience wrapper used by the benchmark runner."""

        outputs = []
        for example in examples:
            result = self.answer_question(example["context"], example["question"])
            result["expected_answer"] = example.get("expected_answer", example.get("answer_text", ""))
            result["target_role"] = example.get("target_role", example.get("role", ""))
            result["question_type"] = example.get("question_type", result["question_type"])
            outputs.append(result)
        return outputs

    def _analyze_question(self, question: str) -> QuestionIntent:
        """Infer the target role and predicate hint from the question."""

        lower = question.lower().strip()
        tokens = simple_word_tokenize(lower)
        question_type = "WHAT"
        expected_role = "ARG1"

        if lower.startswith("to whom") or "who received" in lower or "whom did" in lower:
            question_type = "TO-WHOM"
            expected_role = "ARG2"
        else:
            for token in tokens:
                if token in QUESTION_ROLE_MAP:
                    question_type, expected_role = QUESTION_ROLE_MAP[token]
                    break

        if "how long" in lower:
            question_type = "WHEN"
            expected_role = "ARGM-TMP"
        elif "how much" in lower:
            question_type = "WHAT"
            expected_role = "ARG1"
        elif "who received" in lower or "recipient" in lower:
            expected_role = "ARG2"
            question_type = "TO-WHOM"

        predicate_hint = ""
        content_tokens = [token for token in simple_word_tokenize(question) if token.isalpha()]
        verb_like_tokens = [
            token
            for token in content_tokens
            if token.lower().endswith(("ed", "ing"))
            or simple_lemmatize(token) in {simple_lemmatize(item) for item in TRANSFER_VERBS}
            or token.lower() in {"receive", "received", "repair", "repaired", "deliver", "delivered", "announce", "announced", "approve", "approved", "present", "presented", "examine", "examined", "guide", "guided", "explain", "explained"}
        ]
        if verb_like_tokens:
            predicate_hint = simple_lemmatize(verb_like_tokens[-1])
        else:
            for token in reversed(content_tokens):
                lemma = simple_lemmatize(token)
                if lemma.lower() not in STOPWORDS:
                    predicate_hint = lemma
                    break
        if not predicate_hint and content_tokens:
            predicate_hint = simple_lemmatize(content_tokens[-1])

        target_terms = [
            simple_lemmatize(token)
            for token in simple_word_tokenize(question)
            if token.isalpha() and simple_lemmatize(token) not in STOPWORDS
        ]
        if predicate_hint and predicate_hint in target_terms:
            target_terms = [token for token in target_terms if token != predicate_hint]

        return QuestionIntent(
            question_type=question_type,
            expected_role=expected_role,
            predicate_hint=predicate_hint,
            target_terms=target_terms,
            raw_question=question,
        )

    def _infer_predicate_index(self, tokens: Sequence[str], intent: QuestionIntent) -> int:
        """Locate the context token most aligned with the question predicate."""

        if not tokens:
            return 0
        if intent.predicate_hint:
            for index, token in enumerate(tokens):
                if simple_lemmatize(token) == simple_lemmatize(intent.predicate_hint):
                    return index
        for index, token in enumerate(tokens):
            lemma = simple_lemmatize(token)
            if token.lower().endswith(("ed", "ing")) or lemma in {simple_lemmatize(item) for item in TRANSFER_VERBS}:
                return index
        return infer_predicate_index(tokens, intent.raw_question)

    def _generate_candidates(
        self,
        context: str,
        question: str,
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        predicate_index: int,
        intent: QuestionIntent,
        baseline: Any,
    ) -> List[CandidateSpan]:
        """Create candidate spans from the baseline, heuristics, and optional models."""

        candidates: List[CandidateSpan] = []
        self._add_baseline_candidate(candidates, tokens, offsets, baseline)
        candidates.extend(self._heuristic_role_candidates(tokens, offsets, predicate_index, intent))

        if self.use_transformer_qa:
            qa_pipeline = self.external_models.ensure_qa_pipeline()
            if qa_pipeline is not None:
                try:
                    transformer_outputs = qa_pipeline(
                        question=question,
                        context=context,
                        top_k=3,
                        handle_impossible_answer=False,
                    )
                    if isinstance(transformer_outputs, dict):
                        transformer_outputs = [transformer_outputs]
                    for output in transformer_outputs:
                        candidates.extend(
                            self._transformer_candidates_from_output(
                                context=context,
                                tokens=tokens,
                                offsets=offsets,
                                predicate_index=predicate_index,
                                intent=intent,
                                output=output,
                            )
                        )
                except Exception as error:  # pragma: no cover - model behavior is environment-specific
                    self.external_models.load_messages.append(f"transformer_inference_failed: {error}")

        deduplicated: Dict[Tuple[str, str, int, int], CandidateSpan] = {}
        for candidate in candidates:
            if not candidate.text.strip():
                continue
            key = (normalize_text(candidate.text), candidate.role, candidate.start_token, candidate.end_token)
            current = deduplicated.get(key)
            if current is None or candidate.base_score > current.base_score:
                deduplicated[key] = candidate
        return list(deduplicated.values())

    def _score_candidates(
        self,
        candidates: Sequence[CandidateSpan],
        question: str,
        baseline: Any,
        intent: QuestionIntent,
        predicate_text: str,
    ) -> List[CandidateSpan]:
        """Assign calibrated final scores to each candidate."""

        scored: List[CandidateSpan] = []
        for candidate in candidates:
            semantic_alignment = self.external_models.lexical_similarity(question, f"{predicate_text} {candidate.text}")
            if self.use_sentence_embeddings:
                semantic_alignment = self.external_models.semantic_similarity(
                    question,
                    f"{predicate_text} {candidate.role} {candidate.text}",
                )

            lexical_overlap = self.external_models.lexical_similarity(candidate.text, " ".join(intent.target_terms))
            shape_bonus = self._shape_bonus(candidate)
            role_match = self._role_match_score(intent.expected_role, candidate.role)
            baseline_bonus = 0.08 if normalize_text(candidate.text) == normalize_text(baseline.answer_text) else 0.0
            if candidate.source == "baseline":
                baseline_bonus += float(baseline.confidence) * 0.10

            candidate.features = {
                "semantic_alignment": semantic_alignment,
                "lexical_overlap": lexical_overlap,
                "shape_bonus": shape_bonus,
                "role_match": role_match,
                "baseline_bonus": baseline_bonus,
            }
            candidate.final_score = _clamp(
                candidate.base_score * 0.30
                + role_match * 0.32
                + semantic_alignment * 0.22
                + lexical_overlap * 0.06
                + shape_bonus * 0.10
                + baseline_bonus
            )
            if intent.expected_role in {"ARGM-TMP", "ARGM-LOC", "ARGM-MNR", "ARGM-CAU", "ARG2"} and len(simple_word_tokenize(candidate.text)) > 8:
                candidate.final_score = _clamp(candidate.final_score - 0.10)
            scored.append(candidate)

        scored.sort(key=lambda item: item.final_score, reverse=True)
        return scored

    def _select_best_candidate(
        self,
        candidates: Sequence[CandidateSpan],
        baseline: Any,
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        intent: QuestionIntent,
    ) -> CandidateSpan:
        """Prefer exact role matches and fall back safely to the baseline."""

        for candidate in candidates:
            if candidate.role == intent.expected_role:
                return candidate

        if candidates:
            return candidates[0]

        answer_tokens = simple_word_tokenize(baseline.answer_text)
        match = self._find_subsequence(tokens, answer_tokens) or (0, max(len(answer_tokens) - 1, 0))
        return CandidateSpan(
            text=baseline.answer_text,
            start_token=match[0],
            end_token=match[1],
            role=baseline.predicted_role,
            source="baseline_fallback",
            base_score=float(baseline.confidence),
            char_start=offsets[match[0]][0] if offsets else 0,
            char_end=offsets[match[1]][1] if offsets else len(baseline.answer_text),
            features={"semantic_alignment": 0.0},
            final_score=_clamp(float(baseline.confidence)),
        )

    def _build_reasoning_summary(
        self,
        question: str,
        predicate_text: str,
        baseline: Any,
        best_candidate: CandidateSpan,
        intent: QuestionIntent,
    ) -> str:
        """Create a short, deterministic reasoning trace."""

        role_phrase = intent.expected_role
        if best_candidate.role == intent.expected_role:
            return (
                f"Question pattern '{intent.question_type}' mapped to {role_phrase}. "
                f"The hybrid reranker aligned the predicate '{predicate_text}' and selected "
                f"'{best_candidate.text}' from {best_candidate.source} evidence because it best matched the expected role."
            )
        return (
            f"Question pattern '{intent.question_type}' suggested {role_phrase}, but the strongest available evidence "
            f"came from {best_candidate.source} with span '{best_candidate.text}'. The baseline answer was "
            f"'{baseline.answer_text}' ({baseline.predicted_role})."
        )

    def _maybe_rewrite_reasoning(self, reasoning_summary: str) -> str:
        """Optionally rewrite the explanation with a small local model."""

        reasoner = self.external_models.ensure_reasoner()
        if reasoner is None:
            return reasoning_summary
        try:
            prompt = (
                "Rewrite the following QA explanation in two precise sentences while preserving the decision:\n"
                f"{reasoning_summary}"
            )
            output = reasoner(prompt, max_new_tokens=96, num_beams=2)
            if output and isinstance(output, list):
                generated = str(output[0].get("generated_text", "")).strip()
                if generated:
                    return generated
        except Exception as error:  # pragma: no cover - optional runtime path
            self.external_models.load_messages.append(f"reasoner_failed: {error}")
        return reasoning_summary

    def _add_baseline_candidate(
        self,
        candidates: List[CandidateSpan],
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        baseline: Any,
    ) -> None:
        """Convert the baseline answer into a candidate span."""

        answer_tokens = simple_word_tokenize(baseline.answer_text)
        if not answer_tokens:
            return
        match = self._find_subsequence(tokens, answer_tokens)
        if match is None:
            return
        start_token, end_token = match
        candidates.append(
            CandidateSpan(
                text=_detokenize(tokens[start_token : end_token + 1]),
                start_token=start_token,
                end_token=end_token,
                role=baseline.predicted_role,
                source="baseline",
                base_score=float(baseline.confidence),
                char_start=offsets[start_token][0],
                char_end=offsets[end_token][1],
                features={"semantic_alignment": 0.0},
            )
        )

    def _heuristic_role_candidates(
        self,
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        predicate_index: int,
        intent: QuestionIntent,
    ) -> List[CandidateSpan]:
        """Extract role-like candidates directly from the sentence."""

        candidates: List[CandidateSpan] = []
        if not tokens:
            return candidates

        agent_span = self._agent_span(tokens, predicate_index)
        if agent_span is not None:
            candidates.append(self._build_candidate(tokens, offsets, agent_span, "ARG0", "heuristic_agent", 0.52))

        theme_span = self._theme_span(tokens, predicate_index)
        if theme_span is not None:
            candidates.append(self._build_candidate(tokens, offsets, theme_span, "ARG1", "heuristic_theme", 0.49))

        for span in self._recipient_spans(tokens, predicate_index):
            candidates.append(self._build_candidate(tokens, offsets, span, "ARG2", "heuristic_recipient", 0.58))
        for span in self._temporal_spans(tokens):
            candidates.append(self._build_candidate(tokens, offsets, span, "ARGM-TMP", "heuristic_temporal", 0.62))
        for span in self._location_spans(tokens):
            candidates.append(self._build_candidate(tokens, offsets, span, "ARGM-LOC", "heuristic_location", 0.61))
        for span in self._manner_spans(tokens):
            candidates.append(self._build_candidate(tokens, offsets, span, "ARGM-MNR", "heuristic_manner", 0.60))
        for span in self._cause_spans(tokens):
            candidates.append(self._build_candidate(tokens, offsets, span, "ARGM-CAU", "heuristic_cause", 0.63))

        if intent.expected_role == "ARG2":
            double_object_span = self._first_post_predicate_np(tokens, predicate_index)
            if double_object_span is not None:
                candidates.append(
                    self._build_candidate(tokens, offsets, double_object_span, "ARG2", "heuristic_double_object", 0.56)
                )
        return candidates

    def _transformer_candidates_from_output(
        self,
        context: str,
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        predicate_index: int,
        intent: QuestionIntent,
        output: Dict[str, Any],
    ) -> List[CandidateSpan]:
        """Turn one transformer QA answer into a scored candidate."""

        del context
        answer_text = str(output.get("answer", "")).strip()
        if not answer_text:
            return []
        char_start = int(output.get("start", -1))
        char_end = int(output.get("end", -1))
        span = self._char_span_to_token_span(tokens, offsets, char_start, char_end)
        if span is None:
            match = self._find_subsequence(tokens, simple_word_tokenize(answer_text))
            if match is None:
                return []
            span = match
            char_start = offsets[span[0]][0]
            char_end = offsets[span[1]][1]
        role = self._classify_span_role(tokens, span[0], span[1], predicate_index, intent)
        return [
            CandidateSpan(
                text=_detokenize(tokens[span[0] : span[1] + 1]),
                start_token=span[0],
                end_token=span[1],
                role=role,
                source="transformer_qa",
                base_score=float(output.get("score", 0.0)),
                char_start=char_start,
                char_end=char_end,
                features={"semantic_alignment": 0.0},
            )
        ]

    @staticmethod
    def _build_candidate(
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        span: Tuple[int, int],
        role: str,
        source: str,
        base_score: float,
    ) -> CandidateSpan:
        """Helper for heuristic span construction."""

        return CandidateSpan(
            text=_detokenize(tokens[span[0] : span[1] + 1]),
            start_token=span[0],
            end_token=span[1],
            role=role,
            source=source,
            base_score=base_score,
            char_start=offsets[span[0]][0],
            char_end=offsets[span[1]][1],
            features={"semantic_alignment": 0.0},
        )

    @staticmethod
    def _shape_bonus(candidate: CandidateSpan) -> float:
        """Score how well a candidate's surface form matches its semantic role."""

        text = candidate.text.lower()
        if candidate.role == "ARGM-TMP" and any(marker in text for marker in TEMPORAL_MARKERS):
            return 0.95
        if candidate.role == "ARGM-LOC" and re.match(r"^(in|at|on|near|inside|outside|through|into|to)\b", text):
            return 0.92
        if candidate.role == "ARGM-MNR" and (text.endswith("ly") or text.startswith("with ")):
            return 0.90
        if candidate.role == "ARGM-CAU" and (text.startswith("because") or text.startswith("due to")):
            return 0.96
        if candidate.role == "ARG2" and (text.startswith("to ") or text.startswith("for ")):
            return 0.88
        if candidate.role in {"ARG0", "ARG1"}:
            return 0.75
        return 0.40

    @staticmethod
    def _role_match_score(expected_role: str, candidate_role: str) -> float:
        """Reward exact matches and lightly reward compatible roles."""

        if expected_role == candidate_role:
            return 1.0
        compatibility = {
            "ARGM-CAU": {"ARGM-PRP", "ARGM-PNC"},
            "ARGM-LOC": {"ARGM-DIR"},
            "ARGM-MNR": {"ARGM-ADV", "ARGM-EXT"},
            "ARG2": {"ARG1"},
        }
        if candidate_role in compatibility.get(expected_role, set()):
            return 0.55
        if expected_role == "ARG1" and candidate_role.startswith("ARG"):
            return 0.35
        return 0.05

    @staticmethod
    def _find_subsequence(tokens: Sequence[str], answer_tokens: Sequence[str]) -> Tuple[int, int] | None:
        """Find an answer token subsequence inside the context tokens."""

        if not answer_tokens or not tokens:
            return None
        normalized_answer = [normalize_text(token) for token in answer_tokens]
        normalized_tokens = [normalize_text(token) for token in tokens]
        window = len(answer_tokens)
        for start in range(0, len(tokens) - window + 1):
            if normalized_tokens[start : start + window] == normalized_answer:
                return start, start + window - 1
        return None

    @staticmethod
    def _char_span_to_token_span(
        tokens: Sequence[str],
        offsets: Sequence[Tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> Tuple[int, int] | None:
        """Convert character offsets into token offsets."""

        del tokens
        start_token = None
        end_token = None
        for index, (token_start, token_end) in enumerate(offsets):
            if start_token is None and token_start <= char_start < token_end:
                start_token = index
            if token_start < char_end <= token_end:
                end_token = index
                break
        if start_token is None or end_token is None:
            return None
        return start_token, end_token

    @staticmethod
    def _scan_left_boundary(tokens: Sequence[str], start_index: int) -> int:
        """Expand a phrase leftward until a soft boundary is reached."""

        left = start_index
        while left > 0:
            previous = tokens[left - 1].lower()
            if previous in BOUNDARY_TOKENS:
                break
            if previous in LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS | {"because", "due"}:
                break
            if previous.isdigit() or previous.isalpha() or previous in {"'", "'s", "-"}:
                left -= 1
                continue
            break
        return left

    @staticmethod
    def _scan_right_boundary(tokens: Sequence[str], start_index: int) -> int:
        """Expand a phrase rightward until a soft boundary is reached."""

        right = start_index
        while right + 1 < len(tokens):
            next_token = tokens[right + 1].lower()
            if next_token in BOUNDARY_TOKENS:
                break
            if next_token in {"because"}:
                break
            if next_token.isdigit() or next_token.isalpha() or next_token in {"'", "'s", "-", "$"}:
                right += 1
                continue
            break
        return right

    def _agent_span(self, tokens: Sequence[str], predicate_index: int) -> Tuple[int, int] | None:
        """Approximate the ARG0 span immediately before the predicate."""

        if predicate_index <= 0:
            return None
        right = predicate_index - 1
        if tokens[right].lower() in {"to", "for", "because"}:
            return None
        left = right
        while left > 0 and tokens[left - 1].lower() not in BOUNDARY_TOKENS | {"because", "when", "after", "before"}:
            if tokens[left - 1].lower() in LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS:
                break
            left -= 1
        if left <= right:
            return left, right
        return None

    def _theme_span(self, tokens: Sequence[str], predicate_index: int) -> Tuple[int, int] | None:
        """Approximate the direct-object/theme span after the predicate."""

        if predicate_index + 1 >= len(tokens):
            return None
        left = predicate_index + 1
        while left < len(tokens) and tokens[left].lower() in {"to", "for", "with"}:
            left += 1
        if left >= len(tokens) or tokens[left] in BOUNDARY_TOKENS:
            return None
        right = left
        while right + 1 < len(tokens):
            next_lower = tokens[right + 1].lower()
            if next_lower in BOUNDARY_TOKENS | LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS | {"because", "to", "for", "with"}:
                break
            right += 1
        if left <= right:
            return left, right
        return None

    def _first_post_predicate_np(self, tokens: Sequence[str], predicate_index: int) -> Tuple[int, int] | None:
        """Approximate the first noun phrase after a predicate."""

        if predicate_index + 1 >= len(tokens):
            return None
        left = predicate_index + 1
        while left < len(tokens) and tokens[left].lower() in {"the", "a", "an", "to", "for"}:
            if tokens[left].lower() in {"to", "for"}:
                left += 1
                continue
            break
        if left >= len(tokens):
            return None
        right = left
        while right + 1 < len(tokens):
            next_lower = tokens[right + 1].lower()
            if next_lower in BOUNDARY_TOKENS | LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS | {"because", "to", "for", "with"}:
                break
            if next_lower in {"a", "an", "the"} and right + 1 > left:
                break
            right += 1
        return left, right

    def _recipient_spans(self, tokens: Sequence[str], predicate_index: int) -> List[Tuple[int, int]]:
        """Extract likely ARG2 recipient spans."""

        spans: List[Tuple[int, int]] = []
        if predicate_index < 0 or predicate_index >= len(tokens):
            return spans

        transfer_verb = simple_lemmatize(tokens[predicate_index]) in {simple_lemmatize(token) for token in TRANSFER_VERBS}
        for index, token in enumerate(tokens):
            lower = token.lower()
            if lower in {"to", "for"} and index + 1 < len(tokens):
                left = index + 1
                right = left
                while right + 1 < len(tokens):
                    next_lower = tokens[right + 1].lower()
                    if next_lower in BOUNDARY_TOKENS | LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS | {"because", "with"}:
                        break
                    right += 1
                candidate_tokens = tokens[left : right + 1]
                if lower == "for" and transfer_verb and not self._looks_person_like(candidate_tokens):
                    continue
                spans.append((left, right))

        if transfer_verb:
            start = predicate_index + 1
            if start < len(tokens):
                end = start
                while end + 1 < len(tokens):
                    next_lower = tokens[end + 1].lower()
                    if next_lower in BOUNDARY_TOKENS | LOCATION_PREPOSITIONS | TEMPORAL_PREPOSITIONS | {"because", "to", "for", "with"}:
                        break
                    if next_lower in {"a", "an", "the"} and end + 1 > start:
                        break
                    end += 1
                if start <= end:
                    spans.append((start, end))
        return spans

    @staticmethod
    def _looks_person_like(tokens: Sequence[str]) -> bool:
        """Approximate whether a span refers to a person or recipient-like entity."""

        for token in tokens:
            if token.istitle():
                return True
            if token.lower() in ANIMATE_HINTS | {"him", "her", "them", "us", "me", "you"}:
                return True
        return False

    def _temporal_spans(self, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        """Extract explicit time-like spans."""

        spans: List[Tuple[int, int]] = []
        for index, token in enumerate(tokens):
            lower = token.lower()
            if lower in TEMPORAL_MARKERS:
                left = index
                if index > 0 and (tokens[index - 1].lower() in {"last", "next", "this", "every", "early", "late"} or tokens[index - 1].isdigit()):
                    left = index - 1
                spans.append((left, self._scan_right_boundary(tokens, index)))
            elif lower in TEMPORAL_PREPOSITIONS and index + 1 < len(tokens):
                lookahead = tokens[index + 1].lower()
                if lookahead in TEMPORAL_MARKERS or re.fullmatch(r"\d{1,4}", lookahead):
                    spans.append((index, self._scan_right_boundary(tokens, index + 1)))
        return spans

    def _location_spans(self, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        """Extract explicit location-like spans."""

        spans: List[Tuple[int, int]] = []
        for index, token in enumerate(tokens):
            lower = token.lower()
            if lower in LOCATION_PREPOSITIONS and index + 1 < len(tokens):
                right = index + 1
                while right + 1 < len(tokens):
                    next_lower = tokens[right + 1].lower()
                    if next_lower in BOUNDARY_TOKENS | {"because"}:
                        break
                    if next_lower in TEMPORAL_MARKERS | TEMPORAL_PREPOSITIONS:
                        break
                    right += 1
                spans.append((index, right))
        return spans

    def _manner_spans(self, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        """Extract manner spans such as adverbs or instrumental phrases."""

        spans: List[Tuple[int, int]] = []
        for index, token in enumerate(tokens):
            lower = token.lower()
            if re.fullmatch(r"[a-z]+ly", lower):
                spans.append((index, index))
            elif lower in {"with", "using", "via"} and index + 1 < len(tokens):
                right = index + 1
                while right + 1 < len(tokens):
                    next_lower = tokens[right + 1].lower()
                    if next_lower in BOUNDARY_TOKENS | {"because"}:
                        break
                    if next_lower in TEMPORAL_PREPOSITIONS | LOCATION_PREPOSITIONS:
                        break
                    right += 1
                spans.append((index, right))
        return spans

    def _cause_spans(self, tokens: Sequence[str]) -> List[Tuple[int, int]]:
        """Extract causal or purpose-like spans."""

        spans: List[Tuple[int, int]] = []
        for index, token in enumerate(tokens):
            lower = token.lower()
            if lower == "because":
                spans.append((index, self._scan_right_boundary(tokens, index)))
            if lower == "due" and index + 1 < len(tokens) and tokens[index + 1].lower() == "to":
                spans.append((index, self._scan_right_boundary(tokens, index + 1)))
        return spans

    def _classify_span_role(
        self,
        tokens: Sequence[str],
        start_token: int,
        end_token: int,
        predicate_index: int,
        intent: QuestionIntent,
    ) -> str:
        """Assign a coarse role label to a span using surface cues."""

        span_text = _detokenize(tokens[start_token : end_token + 1]).lower()
        if any(marker in span_text for marker in ("because", "due to")):
            return "ARGM-CAU"
        if any(marker in span_text for marker in TEMPORAL_MARKERS):
            return "ARGM-TMP"
        if span_text.startswith(("in ", "at ", "on ", "near ", "inside ", "outside ", "through ", "to ")):
            if any(marker in span_text for marker in TEMPORAL_MARKERS):
                return "ARGM-TMP"
            if span_text.startswith("to ") and intent.expected_role == "ARG2":
                return "ARG2"
            return "ARGM-LOC"
        if span_text.startswith(("to ", "for ")) and intent.expected_role == "ARG2":
            return "ARG2"
        if span_text.endswith("ly") or span_text.startswith(("with ", "using ")):
            return "ARGM-MNR"
        if end_token < predicate_index:
            return "ARG0"
        if simple_lemmatize(tokens[predicate_index]) in {simple_lemmatize(token) for token in TRANSFER_VERBS} and start_token > predicate_index:
            if start_token == predicate_index + 1 and intent.expected_role == "ARG2":
                return "ARG2"
        return "ARG1"


def load_challenge_suite(project_root: Path | None = None) -> List[Dict[str, Any]]:
    """Load the curated challenge suite used by the app and benchmark runner."""

    base_path = project_root or Path(__file__).resolve().parent
    challenge_path = base_path / "data" / "challenge_suite.json"
    return json.loads(challenge_path.read_text(encoding="utf-8"))


def sample_questions() -> List[Dict[str, Any]]:
    """Return clickable sample questions for the web app."""

    return load_challenge_suite()


def evaluate_prediction(prediction: Dict[str, Any], expected_answer: str, target_role: str) -> Dict[str, Any]:
    """Attach accuracy metrics to a prediction payload."""

    predicted_answer = prediction["hybrid_answer"] if "hybrid_answer" in prediction else prediction.get("predicted_answer", "")
    predicted_role = prediction["role"] if "role" in prediction else prediction.get("predicted_role", "O")
    predicted_tokens = simple_word_tokenize(predicted_answer)
    gold_tokens = simple_word_tokenize(expected_answer)
    return {
        "exact_match": float(normalize_text(predicted_answer) == normalize_text(expected_answer)),
        "token_f1": token_level_f1(predicted_tokens, gold_tokens),
        "role_match": float(predicted_role == target_role),
    }
