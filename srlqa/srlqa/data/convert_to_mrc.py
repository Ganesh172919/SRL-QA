"""Convert QA-SRL-style records into extractive MRC examples."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


QUESTION_ROLE_HINTS = {
    "who": ("WHO", "ARG0"),
    "whom": ("TO-WHOM", "ARG2"),
    "what": ("WHAT", "ARG1"),
    "when": ("WHEN", "ARGM-TMP"),
    "where": ("WHERE", "ARGM-LOC"),
    "how": ("HOW", "ARGM-MNR"),
    "why": ("WHY", "ARGM-CAU"),
}


@dataclass(slots=True)
class MrcExample:
    """Canonical SRL-QA example used by the new pipeline."""

    example_id: str
    context: str
    question: str
    answers: list[str]
    answer_text: str
    answer_start_char: int
    answer_end_char: int
    predicate: str
    predicate_index: int | None
    role: str
    question_type: str
    source: str
    frame_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_text(value: Any) -> str:
    """Best-effort conversion of common dataset fields into text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "answer", "question", "sentence"):
            if key in value:
                text = _flatten_text(value[key])
                if text:
                    return text
        return " ".join(_flatten_text(item) for item in value.values()).strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = [_flatten_text(item) for item in value]
        parts = [part for part in parts if part]
        return " ".join(parts).strip()
    return str(value).strip()


def _normalize_tokens_to_text(tokens: Any) -> str:
    if isinstance(tokens, str):
        return tokens.strip()
    if isinstance(tokens, Sequence) and not isinstance(tokens, (bytes, bytearray)):
        return " ".join(str(token) for token in tokens).replace(" n't", "n't").strip()
    return _flatten_text(tokens)


def _field(example: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in example and example[name] not in (None, ""):
            return example[name]
    return None


def _pick_context(example: Mapping[str, Any]) -> str:
    value = _field(
        example,
        (
            "context",
            "sentence",
            "sent",
            "text",
            "tokens",
            "sentence_tokens",
            "words",
        ),
    )
    return _normalize_tokens_to_text(value)


def _pick_question(example: Mapping[str, Any]) -> str:
    value = _field(example, ("question", "query", "question_text", "question_tokens"))
    return _normalize_tokens_to_text(value)


def _answer_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        for key in ("text", "answer", "answers", "spans"):
            if key in value:
                answers = _answer_candidates(value[key])
                if answers:
                    return answers
        return []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        answers: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    answers.append(item.strip())
            else:
                text = _flatten_text(item)
                if text:
                    answers.append(text)
        return answers
    text = str(value).strip()
    return [text] if text else []


def _pick_answers(example: Mapping[str, Any]) -> list[str]:
    value = _field(
        example,
        (
            "answers",
            "answer",
            "answer_text",
            "answer_texts",
            "answer_spans",
            "gold_answers",
        ),
    )
    return _answer_candidates(value)


def _pick_predicate(example: Mapping[str, Any], context: str) -> tuple[str, int | None]:
    predicate = _field(example, ("predicate", "verb", "predicate_lemma", "verb_lemma"))
    predicate_index = _field(
        example,
        ("predicate_index", "predicate_idx", "verb_index", "verb_idx", "target_idx"),
    )
    try:
        parsed_index = int(predicate_index) if predicate_index is not None else None
    except (TypeError, ValueError):
        parsed_index = None

    if predicate:
        return _flatten_text(predicate), parsed_index

    tokens = context.split()
    if parsed_index is not None and 0 <= parsed_index < len(tokens):
        return tokens[parsed_index], parsed_index
    return "", parsed_index


def infer_question_type_and_role(question: str, role: str | None = None) -> tuple[str, str]:
    first_word_match = re.search(r"[A-Za-z]+", question.lower())
    first_word = first_word_match.group(0) if first_word_match else ""
    question_type, inferred_role = QUESTION_ROLE_HINTS.get(first_word, ("WHAT", "ARG1"))
    return question_type, role or inferred_role


def find_answer_span(context: str, answer: str) -> tuple[int, int]:
    if not context or not answer:
        return -1, -1
    start = context.lower().find(answer.lower())
    if start < 0:
        compact_context = re.sub(r"\s+", " ", context)
        compact_answer = re.sub(r"\s+", " ", answer)
        start = compact_context.lower().find(compact_answer.lower())
        if start >= 0:
            return start, start + len(compact_answer)
        return -1, -1
    return start, start + len(answer)


def normalize_record(
    example: Mapping[str, Any],
    index: int,
    source: str = "huggingface",
) -> MrcExample:
    """Normalize a single dataset record into an MRC example.

    The Hugging Face QA-SRL variants are not perfectly schema-aligned, so this
    function handles common field names and keeps unknown details in a stable
    default form instead of failing.
    """

    context = _pick_context(example)
    question = _pick_question(example)
    answers = _pick_answers(example)
    answer_text = answers[0] if answers else ""
    answer_start, answer_end = find_answer_span(context, answer_text)
    predicate, predicate_index = _pick_predicate(example, context)
    role_value = _field(example, ("role", "target_role", "arg_label", "label"))
    question_type, role = infer_question_type_and_role(question, _flatten_text(role_value) or None)
    example_id = _flatten_text(_field(example, ("id", "example_id", "qasrl_id", "sent_id")))
    if not example_id:
        example_id = f"{source}_{index:06d}"

    return MrcExample(
        example_id=example_id,
        context=context,
        question=question,
        answers=answers,
        answer_text=answer_text,
        answer_start_char=answer_start,
        answer_end_char=answer_end,
        predicate=predicate,
        predicate_index=predicate_index,
        role=role,
        question_type=question_type,
        source=source,
    )


def normalize_records(
    records: Sequence[Mapping[str, Any]],
    source: str = "huggingface",
) -> list[dict[str, Any]]:
    return [normalize_record(record, index=index, source=source).to_dict() for index, record in enumerate(records)]
