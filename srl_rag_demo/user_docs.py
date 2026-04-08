"""Build lightweight SRL documents from pasted or uploaded text."""

from __future__ import annotations

import re
from typing import Iterable

try:
    from .data_models import SRLArgument, SRLDocument
    from .frame_store import FrameStore, light_lemma
    from .propbank_loader import simple_word_tokenize
except ImportError:  # pragma: no cover
    from data_models import SRLArgument, SRLDocument
    from frame_store import FrameStore, light_lemma
    from propbank_loader import simple_word_tokenize


VERB_HINTS = {
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
    "cooked",
    "explained",
    "announced",
    "signed",
}
BOUNDARY_WORDS = {
    "to",
    "for",
    "after",
    "before",
    "during",
    "because",
    "with",
    "at",
    "in",
    "on",
    "near",
    "inside",
    "outside",
    "through",
    "across",
    "into",
    "onto",
    "under",
    "over",
}


def split_text_units(text: str) -> list[str]:
    units = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [unit.strip() for unit in units if unit.strip()]


def infer_predicate(tokens: list[str]) -> tuple[str, int]:
    for index, token in enumerate(tokens):
        lower = token.lower().strip(".,;:!?")
        if lower in VERB_HINTS or lower.endswith(("ed", "ing")):
            return token, index
    return (tokens[min(1, len(tokens) - 1)] if tokens else "", min(1, max(len(tokens) - 1, 0)))


def _span_text(tokens: list[str], start: int, end: int) -> str:
    return " ".join(tokens[start : end + 1]).strip(" ,.;:!?")


def _add_argument(arguments: list[SRLArgument], role: str, text: str, start: int, end: int, reason: str) -> None:
    cleaned = text.strip(" ,.;:!?")
    if not cleaned:
        return
    key = (role, cleaned.lower())
    if any((argument.role, argument.text.lower()) == key for argument in arguments):
        return
    arguments.append(
        SRLArgument(
            role=role,
            text=cleaned,
            description=reason,
            start_token=start,
            end_token=end,
            source="user",
        )
    )


def _find_token_window(tokens: list[str], window: list[str]) -> int:
    normalized = [token.lower().strip(".,;:!?") for token in tokens]
    target = [token.lower().strip(".,;:!?") for token in window]
    for index in range(len(normalized) - len(target) + 1):
        if normalized[index : index + len(target)] == target:
            return index
    return -1


def heuristic_arguments(tokens: list[str], predicate_index: int) -> list[SRLArgument]:
    arguments: list[SRLArgument] = []
    if not tokens:
        return arguments

    if predicate_index > 0:
        _add_argument(arguments, "ARG0", _span_text(tokens, 0, predicate_index - 1), 0, predicate_index - 1, "subject before predicate")

    object_start = predicate_index + 1
    object_end = object_start - 1
    while object_end + 1 < len(tokens):
        next_token = tokens[object_end + 1].lower().strip(".,;:!?")
        if next_token in BOUNDARY_WORDS or next_token.endswith("ly"):
            break
        object_end += 1
    if object_start <= object_end:
        _add_argument(arguments, "ARG1", _span_text(tokens, object_start, object_end), object_start, object_end, "object after predicate")

    joined = " ".join(tokens)
    location_match = re.search(
        r"\b(to|in|at|near|inside|outside|through|across|into|onto|under|over)\s+(the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,3}",
        joined,
        flags=re.IGNORECASE,
    )
    if location_match:
        loc_text = re.split(
            r"\s+(?:at|after|before|during|on|in)\s+(?:noon|midnight|lunch|dinner|night|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            location_match.group(0),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        loc_tokens = simple_word_tokenize(loc_text)
        start = _find_token_window(tokens, loc_tokens)
        end = start + len(loc_tokens) - 1
        if start >= 0:
            lowered = loc_text.lower()
            if lowered.startswith("to "):
                _add_argument(arguments, "ARG2", loc_text, start, end, "recipient or destination phrase")
            if not any(marker in lowered for marker in ("noon", "dinner", "morning", "evening", "monday", "yesterday")):
                _add_argument(arguments, "ARGM-LOC", loc_text, start, end, "location prepositional phrase")

    temporal_match = re.search(
        r"\b((?:at|before|after|during|on|in)\s+(?:noon|midnight|lunch|dinner|night|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:last|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|yesterday|today|tomorrow)\b",
        joined,
        flags=re.IGNORECASE,
    )
    if temporal_match:
        tmp_text = temporal_match.group(0)
        tmp_tokens = simple_word_tokenize(tmp_text)
        start = _find_token_window(tokens, tmp_tokens)
        if start >= 0:
            _add_argument(arguments, "ARGM-TMP", tmp_text, start, start + len(tmp_tokens) - 1, "temporal phrase")

    causal_match = re.search(r"\b(because\b.+|due to\b.+|as a result of\b.+)", joined, flags=re.IGNORECASE)
    if causal_match:
        cause_text = causal_match.group(1)
        cause_tokens = simple_word_tokenize(cause_text)
        start = _find_token_window(tokens, cause_tokens)
        if start >= 0:
            _add_argument(arguments, "ARGM-CAU", cause_text, start, start + len(cause_tokens) - 1, "causal marker")

    for index, token in enumerate(tokens):
        if token.lower().endswith("ly"):
            _add_argument(arguments, "ARGM-MNR", token, index, index, "manner adverb")

    return arguments


def build_user_documents(texts: Iterable[str], frame_store: FrameStore) -> list[SRLDocument]:
    documents: list[SRLDocument] = []
    for text_index, text in enumerate(texts):
        for unit_index, unit in enumerate(split_text_units(text)):
            tokens = simple_word_tokenize(unit)
            if not tokens:
                continue
            predicate, predicate_index = infer_predicate(tokens)
            predicate_lemma = light_lemma(predicate)
            arguments = heuristic_arguments(tokens, predicate_index)
            frame_hint = frame_store.hint_for(predicate=predicate_lemma)
            documents.append(
                SRLDocument(
                    doc_id=f"user:{text_index}:{unit_index}",
                    source="user",
                    context=unit,
                    tokens=tokens,
                    predicate=predicate,
                    predicate_lemma=predicate_lemma,
                    predicate_indices=[predicate_index] if predicate else [],
                    frame_hint=frame_hint,
                    arguments=arguments,
                )
            )
    return documents
