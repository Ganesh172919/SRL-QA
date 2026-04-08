"""Role-aware QA over retrieved SRL documents."""

from __future__ import annotations

import os
import re
from typing import Iterable

try:
    from .config import DemoConfig
    from .data_models import AnswerCandidate, QAResult, RetrievalHit, SRLArgument, SRLDocument
    from .frame_store import FrameStore
    from .propbank_loader import simple_word_tokenize
except ImportError:  # pragma: no cover
    from config import DemoConfig
    from data_models import AnswerCandidate, QAResult, RetrievalHit, SRLArgument, SRLDocument
    from frame_store import FrameStore
    from propbank_loader import simple_word_tokenize


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

QUESTION_ROLE_MAP = {
    "where": ("WHERE", {"ARGM-LOC", "ARGM-DIR", "ARGM-GOL"}),
    "when": ("WHEN", {"ARGM-TMP"}),
    "why": ("WHY", {"ARGM-CAU", "ARGM-PRP", "ARGM-PNC"}),
    "how": ("HOW", {"ARGM-MNR", "ARGM-ADV", "ARGM-EXT"}),
    "who": ("WHO", {"ARG0", "ARG2"}),
    "whom": ("TO-WHOM", {"ARG2"}),
    "what": ("WHAT", {"ARG1", "ARG2", "ARG3", "ARG4"}),
}


def infer_question_type_and_roles(question: str) -> tuple[str, set[str]]:
    lowered = " ".join(question.lower().split())
    if lowered.startswith(("to whom", "whom")):
        return QUESTION_ROLE_MAP["whom"]
    first_word = ""
    match = re.search(r"[A-Za-z]+", lowered)
    if match:
        first_word = match.group(0)
    return QUESTION_ROLE_MAP.get(first_word, ("WHAT", {"ARG1", "ARG2"}))


def _role_match_score(argument_role: str, expected_roles: set[str]) -> float:
    if argument_role in expected_roles:
        return 1.0
    if argument_role.startswith("ARGM") and any(role.startswith("ARGM") for role in expected_roles):
        return 0.55
    if argument_role.startswith("ARG") and any(role.startswith("ARG") for role in expected_roles):
        return 0.35
    return 0.0


def _question_overlap(question: str, text: str) -> float:
    q_tokens = {token.lower() for token in simple_word_tokenize(question) if token.isalpha()}
    t_tokens = {token.lower() for token in simple_word_tokenize(text) if token.isalpha()}
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens | t_tokens)


def _candidate_from_argument(
    argument: SRLArgument,
    document: SRLDocument,
    hit: RetrievalHit,
    question: str,
    expected_roles: set[str],
    frame_store: FrameStore,
) -> AnswerCandidate:
    role_score = _role_match_score(argument.role, expected_roles)
    frame_ok = frame_store.role_compatible(argument.role, predicate=document.predicate_lemma or document.predicate, roleset_id=document.roleset_id)
    score = (
        0.35 * max(hit.score, 0.0)
        + 0.35 * role_score
        + 0.12 * (1.0 if argument.text.lower() in document.context.lower() else 0.0)
        + 0.10 * (1.0 if frame_ok else 0.0)
        + 0.08 * _question_overlap(question, document.context)
    )
    reasons = [
        f"retrieval_score={hit.score:.3f}",
        f"role_match={role_score:.2f}",
        "frame_compatible" if frame_ok else "frame_mismatch",
        f"source={document.source}",
    ]
    return AnswerCandidate(
        text=argument.text,
        role=argument.role,
        confidence=max(0.0, min(1.0, score)),
        source_doc_id=document.doc_id,
        source=document.source,
        predicate=document.predicate or document.predicate_lemma,
        frame_hint=document.frame_hint,
        start_token=argument.start_token,
        end_token=argument.end_token,
        retrieval_score=hit.score,
        reasons=reasons,
    )


def _heuristic_candidates(document: SRLDocument, hit: RetrievalHit, question: str, expected_roles: set[str]) -> list[AnswerCandidate]:
    qtype, _ = infer_question_type_and_roles(question)
    context = document.context
    candidates: list[AnswerCandidate] = []

    def add(text: str, role: str, reason: str, base: float) -> None:
        cleaned = text.strip(" ,.;:!?")
        if not cleaned:
            return
        if any(candidate.text.lower() == cleaned.lower() for candidate in candidates):
            return
        candidates.append(
            AnswerCandidate(
                text=cleaned,
                role=role,
                confidence=max(0.0, min(1.0, base + 0.25 * max(hit.score, 0.0))),
                source_doc_id=document.doc_id,
                source=document.source,
                predicate=document.predicate or document.predicate_lemma,
                frame_hint=document.frame_hint,
                retrieval_score=hit.score,
                reasons=[reason, "heuristic_span"],
            )
        )

    if qtype == "WHERE":
        for match in re.finditer(
            r"\b((?:to|in|at|near|inside|outside|through|across|into|onto|under|over)\s+(?:the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3})",
            context,
            flags=re.IGNORECASE,
        ):
            span = re.split(r"\s+(?:at\s+noon|after\s+dinner|during\s+the\s+morning)\b", match.group(1), flags=re.IGNORECASE)[0]
            if not any(marker in span.lower() for marker in ("noon", "dinner", "morning", "evening")):
                add(span, "ARGM-LOC", "where_prepositional_phrase", 0.72)
    elif qtype == "WHEN":
        match = re.search(r"\b((?:at|before|after|during|on|in)\s+(?:noon|midnight|lunch|dinner|night|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", context, flags=re.IGNORECASE)
        if match:
            add(match.group(1), "ARGM-TMP", "when_temporal_phrase", 0.75)
    elif qtype == "WHY":
        match = re.search(r"\b(because\b.+|due to\b.+|as a result of\b.+)", context, flags=re.IGNORECASE)
        if match:
            add(match.group(1), "ARGM-CAU", "why_causal_marker", 0.78)
    elif qtype == "HOW":
        match = re.search(r"\b([A-Za-z]+ly)\b", context)
        if match:
            add(match.group(1), "ARGM-MNR", "how_adverb", 0.70)
    elif qtype == "WHO":
        predicate = re.escape(document.predicate) if document.predicate else r"[A-Za-z]+(?:ed|ing)?"
        match = re.search(rf"^(.+?)\s+\b{predicate}\b", context, flags=re.IGNORECASE)
        if match:
            add(match.group(1), "ARG0", "who_subject_before_predicate", 0.70)
    elif qtype == "WHAT":
        predicate = re.escape(document.predicate) if document.predicate else r"[A-Za-z]+(?:ed|ing)?"
        match = re.search(
            rf"\b{predicate}\b\s+(.+?)(?:\s+[A-Za-z]+ly\b|\s+(?:to|for|after|before|during|because|with|at|in)\b|[.?!]|$)",
            context,
            flags=re.IGNORECASE,
        )
        if match:
            add(match.group(1), "ARG1", "what_object_after_predicate", 0.70)

    return [
        candidate
        for candidate in candidates
        if _role_match_score(candidate.role, expected_roles) > 0.0
    ]


class OptionalTransformerQA:
    """Lazy local extractive QA model with graceful failure."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.pipeline = None
        self.error = ""

    def ensure(self) -> object | None:
        if self.pipeline is not None:
            return self.pipeline
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "question-answering",
                model=self.config.teacher_qa_model_name,
                tokenizer=self.config.teacher_qa_model_name,
                device=-1,
            )
        except Exception as exc:  # pragma: no cover - model cache/network dependent
            self.error = f"{type(exc).__name__}: {exc}"
            self.pipeline = None
        return self.pipeline

    def candidates(self, question: str, hits: Iterable[RetrievalHit]) -> list[AnswerCandidate]:
        pipe = self.ensure()
        if pipe is None:
            return []
        candidates: list[AnswerCandidate] = []
        qtype, expected_roles = infer_question_type_and_roles(question)
        role = sorted(expected_roles)[0]
        for hit in hits:
            try:
                output = pipe(question=question, context=hit.document.context)
            except Exception:
                continue
            answer = str(output.get("answer", "")).strip()
            if not answer:
                continue
            candidates.append(
                AnswerCandidate(
                    text=answer,
                    role=role,
                    confidence=min(1.0, float(output.get("score", 0.0)) + 0.20 * max(hit.score, 0.0)),
                    source_doc_id=hit.document.doc_id,
                    source=hit.document.source,
                    predicate=hit.document.predicate or hit.document.predicate_lemma,
                    frame_hint=hit.document.frame_hint,
                    retrieval_score=hit.score,
                    reasons=[f"transformer_qa_candidate={qtype}"],
                )
            )
        return candidates


def answer_question(
    question: str,
    hits: list[RetrievalHit],
    frame_store: FrameStore,
    config: DemoConfig,
    use_transformer: bool = False,
) -> QAResult:
    qtype, expected_roles = infer_question_type_and_roles(question)
    candidates: list[AnswerCandidate] = []
    for hit in hits:
        for argument in hit.document.arguments:
            if _role_match_score(argument.role, expected_roles) > 0.0:
                candidates.append(_candidate_from_argument(argument, hit.document, hit, question, expected_roles, frame_store))
        candidates.extend(_heuristic_candidates(hit.document, hit, question, expected_roles))

    if use_transformer:
        candidates.extend(OptionalTransformerQA(config).candidates(question, hits[:3]))

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    if not candidates:
        return QAResult(
            question=question,
            answer="",
            confidence=0.0,
            role=sorted(expected_roles)[0],
            source_doc_id="",
            source="",
            predicate="",
            evidence_text="No candidate span was found in the retrieved documents.",
            frame_hint="",
            reasoning=[f"question_type={qtype}", "no_candidate_span"],
            candidates=[],
        )

    best = candidates[0]
    source_doc = next((hit.document for hit in hits if hit.document.doc_id == best.source_doc_id), None)
    return QAResult(
        question=question,
        answer=best.text,
        confidence=best.confidence,
        role=best.role,
        source_doc_id=best.source_doc_id,
        source=best.source,
        predicate=best.predicate,
        evidence_text=source_doc.context if source_doc else "",
        frame_hint=best.frame_hint,
        reasoning=[f"question_type={qtype}", *best.reasons],
        candidates=candidates[:10],
    )
