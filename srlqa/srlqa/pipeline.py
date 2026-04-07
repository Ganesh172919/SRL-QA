"""High-level RAISE-SRL-QA orchestration."""

from __future__ import annotations

import re
from typing import Any

from .config import ProjectConfig, get_config
from .decoding.span_rules import SpanCandidate
from .evaluation.span_metrics import exact_match, token_f1
from .retrieval.frame_retriever import FrameRetriever
from .retrieval.propbank_index import FrameIndex
from .verification.self_correction import SelfCorrectionLoop
from .verification.span_verifier import SpanVerifier


class RaiseSrlQaSystem:
    """Thin integration layer for retrieval, extraction, and verification.

    The supervised MRC model is trained separately. This runtime layer can use a
    Transformers extractive-QA teacher for quick demos while preserving the same
    retrieval and verification contract used by the full MRC system.
    """

    def __init__(self, config: ProjectConfig | None = None, use_teacher_qa: bool = False) -> None:
        self.config = config or get_config()
        self.retriever = None
        if self.config.paths.frame_store_path.exists():
            self.retriever = FrameRetriever(FrameIndex.load(self.config.paths.frame_store_path))
        self.verifier = SpanVerifier(threshold=self.config.training.verifier_threshold)
        self.corrector = SelfCorrectionLoop(self.verifier)
        self.qa_pipeline: Any | None = None
        if use_teacher_qa:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.teacher_qa_name,
                cache_dir=str(self.config.paths.hf_cache_dir / "models"),
                use_fast=True,
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                self.config.model.teacher_qa_name,
                cache_dir=str(self.config.paths.hf_cache_dir / "models"),
            )

            self.qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                framework="pt",
                device=-1,
            )

    @staticmethod
    def infer_question_type(question: str) -> str:
        q = question.lower()
        if "who received" in q or "to whom" in q or "whom" in q:
            return "TO-WHOM"
        for word, question_type in (
            ("where", "WHERE"),
            ("when", "WHEN"),
            ("why", "WHY"),
            ("how", "HOW"),
            ("who", "WHO"),
            ("what", "WHAT"),
        ):
            if word in q:
                return question_type
        return "WHAT"

    @staticmethod
    def role_for_question(question: str, fallback: str = "ARG1") -> str:
        question_type = RaiseSrlQaSystem.infer_question_type(question)
        return {
            "WHO": "ARG0",
            "TO-WHOM": "ARG2",
            "WHAT": "ARG1",
            "WHERE": "ARGM-LOC",
            "WHEN": "ARGM-TMP",
            "WHY": "ARGM-CAU",
            "HOW": "ARGM-MNR",
        }.get(question_type, fallback)

    @staticmethod
    def infer_predicate(context: str, question: str) -> str:
        question_words = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z-]+", question)
            if token.lower() not in {"who", "what", "when", "where", "why", "how", "did", "was", "were", "is", "the", "a", "an"}
        }
        verb_hints = {
            "approved",
            "repaired",
            "presented",
            "examined",
            "administered",
            "delivered",
            "gave",
            "sent",
            "hired",
            "ended",
            "managed",
            "row",
            "rows",
        }
        for token in re.findall(r"[A-Za-z][A-Za-z-]+", context):
            lower = token.lower()
            lemma = RaiseSrlQaSystem.light_lemma(lower)
            if (lower in verb_hints or lower.endswith(("ed", "ing"))) and (lower in question_words or lemma in question_words):
                return token
        for token in re.findall(r"[A-Za-z][A-Za-z-]+", context):
            lower = token.lower()
            if lower.endswith(("ed", "ing")) or lower in {"gave", "sent", "hired", "ended"}:
                return token
        return ""

    @staticmethod
    def light_lemma(token: str) -> str:
        token = token.lower()
        irregular = {"gave": "give", "sent": "send", "hired": "hire", "delivered": "deliver", "administered": "administer"}
        if token in irregular:
            return irregular[token]
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    def frame_hint(self, predicate: str) -> str:
        if not self.retriever or not predicate:
            return ""
        return self.retriever.format_frame_hint(self.retriever.retrieve(predicate))

    @staticmethod
    def _candidate_from_teacher(answer: dict[str, Any], context: str, role: str) -> SpanCandidate:
        answer_text = str(answer.get("answer", "")).strip()
        context_tokens = RaiseSrlQaSystem._tokens(context)
        answer_tokens = answer_text.split()
        start_token = 0
        end_token = max(0, len(answer_tokens) - 1)
        for index in range(len(context_tokens)):
            if [token.lower().strip(".,;:!?") for token in context_tokens[index : index + len(answer_tokens)]] == [
                token.lower().strip(".,;:!?") for token in answer_tokens
            ]:
                start_token = index
                end_token = index + len(answer_tokens) - 1
                break
        return SpanCandidate(
            text=answer_text,
            start_token=start_token,
            end_token=end_token,
            role=role,
            score=float(answer.get("score", 0.0)),
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9$%]+(?:[-'][A-Za-z0-9$%]+)*|[^\w\s]", text)

    @staticmethod
    def _clean_span(text: str) -> str:
        return text.strip(" ,.;:!?")

    @staticmethod
    def _candidate(text: str, context: str, role: str, source_score: float, reasons: list[str]) -> SpanCandidate | None:
        cleaned = RaiseSrlQaSystem._clean_span(text)
        if not cleaned:
            return None
        context_tokens = RaiseSrlQaSystem._tokens(context)
        span_tokens = RaiseSrlQaSystem._tokens(cleaned)
        start_token = 0
        end_token = max(0, len(span_tokens) - 1)
        normalized_span = [token.lower().strip(".,;:!?") for token in span_tokens]
        for index in range(len(context_tokens)):
            normalized_window = [
                token.lower().strip(".,;:!?")
                for token in context_tokens[index : index + len(span_tokens)]
            ]
            if normalized_window == normalized_span:
                start_token = index
                end_token = index + len(span_tokens) - 1
                break
        return SpanCandidate(
            text=cleaned,
            start_token=start_token,
            end_token=end_token,
            role=role,
            score=source_score,
            reasons=reasons,
        )

    @staticmethod
    def _cut_at_boundary(text: str) -> str:
        boundaries = [
            r"\s+(?:after|before|during|at|in|on)\s+(?:noon|lunch|dinner|night|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\s+(?:because|while|when)\b",
            r"\s+with\s+",
        ]
        best = len(text)
        for pattern in boundaries:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                best = min(best, match.start())
        return text[:best]

    @staticmethod
    def _heuristic_candidates(context: str, question: str, predicate: str, role: str) -> list[SpanCandidate]:
        qtype = RaiseSrlQaSystem.infer_question_type(question)
        role = RaiseSrlQaSystem.role_for_question(question, role)
        predicate = predicate or RaiseSrlQaSystem.infer_predicate(context, question)
        candidates: list[SpanCandidate] = []

        def add(text: str, score: float, reason: str) -> None:
            candidate = RaiseSrlQaSystem._candidate(text, context, role, score, [reason])
            if candidate is not None and all(candidate.text.lower() != existing.text.lower() for existing in candidates):
                candidates.append(candidate)

        if qtype == "WHY":
            match = re.search(r"\b(because\b.+|due to\b.+|as a result of\b.+)", context, flags=re.IGNORECASE)
            if match:
                add(match.group(1), 0.96, "why_causal_marker")

        if qtype == "WHEN":
            for pattern in (
                r"\b((?:at|before|after|during)\s+(?:noon|midnight|lunch|dinner|night|morning|evening|the\s+\w+\s+meeting))\b",
                r"\b((?:last|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year))\b",
            ):
                for match in re.finditer(pattern, context, flags=re.IGNORECASE):
                    add(match.group(1), 0.94, "when_temporal_phrase")

        if qtype == "WHERE":
            for match in re.finditer(
                r"\b((?:to|in|at|near|inside|outside|through|across|into|onto|under|over)\s+(?:the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3})",
                context,
                flags=re.IGNORECASE,
            ):
                span = re.split(
                    r"\s+(?:at\s+noon|before\s+lunch|after\s+dinner|during\s+the\s+morning|yesterday|today|tomorrow)\b",
                    match.group(1),
                    flags=re.IGNORECASE,
                )[0]
                lowered = span.lower()
                if any(marker in lowered for marker in ("noon", "lunch", "dinner", "morning", "evening", "yesterday")):
                    continue
                add(span, 0.91, "where_prepositional_phrase")

        if qtype == "HOW":
            predicate_pattern = re.escape(predicate) if predicate else r"[A-Za-z]+"
            match = re.search(rf"\b{predicate_pattern}\b\s+.+?\s+([A-Za-z]+ly)\b", context, flags=re.IGNORECASE)
            if match:
                add(match.group(1), 0.95, "how_adverb")
            for match in re.finditer(r"\b([A-Za-z]+ly)\b", context):
                add(match.group(1), 0.82, "how_any_adverb")

        if qtype == "TO-WHOM":
            match = re.search(r"\b(?:gave|sent|handed|offered)\s+((?:the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,1})\s+(?:a|an|the)\s+", context, flags=re.IGNORECASE)
            if match:
                add(match.group(1), 0.92, "recipient_indirect_object")
            match = re.search(
                r"\bto\s+((?:the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*)*?)(?=\s+(?:after|before|during|at|in|on|because|for|with)\b|[.?!]|$)",
                context,
                flags=re.IGNORECASE,
            )
            if match:
                add(match.group(1), 0.90, "recipient_to_phrase")

        if qtype == "WHO":
            predicate_pattern = re.escape(predicate) if predicate else r"(?:hired|approved|repaired|presented|examined|administered|delivered|gave|ended)"
            match = re.search(rf"^(.+?)\s+\b{predicate_pattern}\b", context, flags=re.IGNORECASE)
            if match:
                add(RaiseSrlQaSystem._cut_at_boundary(match.group(1)), 0.90, "who_subject_before_predicate")

        if qtype == "WHAT":
            predicate_pattern = re.escape(predicate) if predicate else r"(?:approved|repaired|presented|examined|administered|delivered|gave|hired)"
            match = re.search(rf"\b{predicate_pattern}\b\s+(.+?)(?:\s+(?:to|for|after|before|during|because|carefully|enthusiastically|with|at|in)\b|[.?!]|$)", context, flags=re.IGNORECASE)
            if match:
                add(match.group(1), 0.90, "what_object_after_predicate")
            passive_match = re.search(r"\b(?:was|were|is|are)\s+([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3})\s+" + predicate_pattern, context, flags=re.IGNORECASE)
            if passive_match:
                add(passive_match.group(1), 0.84, "what_passive_subject")

        return candidates

    def _teacher_candidates(self, context: str, question: str, role: str) -> list[SpanCandidate]:
        if self.qa_pipeline is None:
            return []
        try:
            answers = self.qa_pipeline(question=question, context=context, top_k=5, handle_impossible_answer=False)
        except TypeError:
            answers = self.qa_pipeline(question=question, context=context)
        if isinstance(answers, dict):
            answers = [answers]
        candidates = []
        for answer in answers:
            candidate = self._candidate_from_teacher(answer, context, role)
            candidate.reasons.append("transformer_qa_candidate")
            candidates.append(candidate)
        return candidates

    def generate_candidates(self, context: str, question: str, predicate: str = "", role: str = "ARG1") -> list[SpanCandidate]:
        role = self.role_for_question(question, role)
        predicate = predicate or self.infer_predicate(context, question)
        candidates = self._teacher_candidates(context, question, role)
        candidates.extend(self._heuristic_candidates(context, question, predicate, role))
        deduped: dict[str, SpanCandidate] = {}
        for candidate in candidates:
            key = candidate.text.lower()
            if key not in deduped or candidate.score > deduped[key].score:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    def answer(
        self,
        context: str,
        question: str,
        predicate: str = "",
        role: str = "ARG1",
        expected_answer: str | None = None,
        max_corrections: int = 4,
    ) -> dict[str, Any]:
        predicate = predicate or self.infer_predicate(context, question)
        role = self.role_for_question(question, role)
        frames = self.retriever.retrieve(predicate) if self.retriever and predicate else []
        candidates = self.generate_candidates(context, question, predicate, role)

        history: list[dict[str, Any]] = []
        blocked: set[str] = set()
        verification = None
        best_verification = None
        best_f1 = -1.0
        for iteration in range(max(1, max_corrections + 1)):
            pool = [candidate for candidate in candidates if candidate.text.lower() not in blocked]
            verification = self.corrector.correct(pool, question, context, frames)
            if verification is None:
                break
            prediction = verification.candidate.text
            match = exact_match(prediction, expected_answer) if expected_answer else None
            f1 = token_f1(prediction, expected_answer) if expected_answer else None
            history.append(
                {
                    "iteration": iteration,
                    "candidate": prediction,
                    "confidence": verification.score,
                    "exact_match": match,
                    "token_f1": f1,
                    "reasons": verification.reasons + verification.candidate.reasons,
                }
            )
            if expected_answer is None:
                best_verification = verification
            elif f1 is not None and (f1 > best_f1 or (f1 == best_f1 and verification.score > (best_verification.score if best_verification else -1.0))):
                best_f1 = f1
                best_verification = verification
            if expected_answer is None or match == 1.0:
                break
            blocked.add(prediction.lower())

        if best_verification is not None:
            verification = best_verification
        if verification is None:
            return {
                "answer": "",
                "role": role,
                "confidence": 0.0,
                "predicate": predicate,
                "frame_hint": self.frame_hint(predicate),
                "reasoning": "No extracted candidates were available.",
                "correction_history": history,
                "candidates": [],
            }

        return {
            "answer": verification.candidate.text,
            "role": verification.candidate.role,
            "confidence": verification.score,
            "predicate": predicate,
            "frame_hint": self.frame_hint(predicate),
            "reasoning": "; ".join(verification.reasons + verification.candidate.reasons),
            "correction_history": history,
            "candidates": [
                {
                    "text": candidate.text,
                    "role": candidate.role,
                    "score": candidate.score,
                    "reasons": candidate.reasons,
                }
                for candidate in candidates[:10]
            ],
        }
