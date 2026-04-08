from __future__ import annotations

import re
from collections import OrderedDict


ROLE_ORDER = [
    "AGENT",
    "THEME",
    "LOCATION",
    "TIME",
    "MANNER",
    "REASON",
    "ATTRIBUTE",
    "SOURCE",
    "GOAL",
    "INSTRUMENT",
    "OBLIQUE",
    "OTHER",
]

EMPTY_SLOT_VALUES = {"", "_", "none", "null"}
LOCATION_PREPS = {"in", "on", "at", "inside", "outside", "within", "near", "under", "over", "around"}
TIME_PREPS = {"during", "after", "before", "since", "until"}
SOURCE_PREPS = {"from", "out", "off"}
GOAL_PREPS = {"into", "onto", "toward", "towards", "through", "across"}
TIME_WORDS = {"today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "year", "years", "month", "months", "day", "days", "pm", "am"}
CLAUSE_BOUNDARIES = {",", ";", ":", ".", "?", "!", "''", "``", "-lrb-", "-rrb-"}
LEADING_CLAUSE_WORDS = {"and", "but", "or", "that", "which", "who", "whom", "because", "since", "if", "when", "while", "although", "though", "as", "than"}
TRAILING_AUXILIARIES = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "must",
    "will",
    "shall",
}
PHRASE_STARTERS = {
    "in",
    "on",
    "at",
    "under",
    "over",
    "inside",
    "outside",
    "within",
    "near",
    "between",
    "among",
    "amongst",
    "with",
    "by",
    "for",
    "from",
    "into",
    "onto",
    "toward",
    "towards",
    "through",
    "across",
    "to",
    "during",
    "after",
    "before",
    "since",
    "until",
    "as",
    "because",
    "last",
    "next",
    "each",
    "every",
    "today",
    "yesterday",
    "tomorrow",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their", "its"}

QUESTION_TEMPLATES = {
    "AGENT": "Who or what {predicate} something?",
    "THEME": "What was {predicate}?",
    "LOCATION": "Where did {predicate} happen?",
    "TIME": "When did {predicate} happen?",
    "MANNER": "How did {predicate} happen?",
    "REASON": "Why did {predicate} happen?",
    "ATTRIBUTE": "As what was something {predicate}?",
    "SOURCE": "From where did {predicate} happen?",
    "GOAL": "Toward what did {predicate} happen?",
    "INSTRUMENT": "With what did {predicate} happen?",
    "OBLIQUE": "What other phrase was linked to {predicate}?",
    "OTHER": "What semantic role is linked to {predicate}?",
}


def normalize_slot(value: str | None) -> str:
    value = (value or "").strip().lower()
    return "_" if value in EMPTY_SLOT_VALUES else value


def normalize_question_slots(question_slots: dict) -> dict:
    return {
        "wh": normalize_slot(question_slots.get("wh")),
        "aux": normalize_slot(question_slots.get("aux")),
        "subj": normalize_slot(question_slots.get("subj")),
        "verb": normalize_slot(question_slots.get("verb")),
        "obj": normalize_slot(question_slots.get("obj")),
        "prep": normalize_slot(question_slots.get("prep")),
        "obj2": normalize_slot(question_slots.get("obj2")),
    }


def infer_role(question_slots: dict) -> str:
    slots = normalize_question_slots(question_slots)
    wh = slots["wh"]
    aux = slots["aux"]
    subj = slots["subj"]
    obj = slots["obj"]
    prep = slots["prep"]
    obj2 = slots["obj2"]
    passive = aux in {"is", "are", "was", "were", "been", "be", "being"} and obj != "_"

    if wh == "where":
        return "LOCATION"
    if wh == "when":
        return "TIME"
    if wh == "why":
        return "REASON"
    if wh == "how":
        if prep == "with" and obj != "_" and obj2 == "_":
            return "INSTRUMENT"
        return "MANNER"
    if subj == "_":
        return "THEME" if passive else "AGENT"
    if obj == "_":
        if prep in LOCATION_PREPS:
            return "LOCATION"
        if prep in TIME_PREPS:
            return "TIME"
        return "THEME"
    if prep == "as" and obj2 == "_":
        return "ATTRIBUTE"
    if prep in SOURCE_PREPS and obj2 == "_":
        return "SOURCE"
    if prep in GOAL_PREPS and obj2 == "_":
        return "GOAL"
    if prep == "with" and obj2 == "_":
        return "INSTRUMENT"
    if prep != "_" and obj2 == "_":
        return "OBLIQUE"
    return "OTHER"


def dedupe_answers(answers: list[str]) -> list[str]:
    seen = OrderedDict()
    for answer in answers:
        normalized = normalize_text(answer)
        if normalized:
            seen.setdefault(normalized, answer.strip())
    return list(seen.values())


def coerce_answer_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if item and str(item).strip()]
    return []


def normalize_role_mapping(role_mapping: dict | None) -> dict[str, list[str]]:
    normalized = OrderedDict()
    for role in ROLE_ORDER:
        answers = coerce_answer_list((role_mapping or {}).get(role))
        if answers:
            normalized[role] = dedupe_answers(answers)
    return normalized


def format_role_output(role_to_answers: dict[str, list[str]]) -> str:
    lines = []
    for role in ROLE_ORDER:
        answers = dedupe_answers(role_to_answers.get(role, []))
        if answers:
            lines.append(f"{role}: {' || '.join(answers)}")
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_token(token: str) -> str:
    raw = token.strip().lower()
    if raw in CLAUSE_BOUNDARIES:
        return raw
    stripped = raw.strip(".,;:!?\"'`()[]{}")
    return stripped or raw


def parse_role_output(output_text: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = OrderedDict()
    role_pattern = "|".join(ROLE_ORDER)
    matches = list(re.finditer(rf"(?P<role>{role_pattern})\s*:", output_text.upper()))
    if not matches:
        return parsed
    for index, match in enumerate(matches):
        role = match.group("role")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output_text)
        value = output_text[start:end].strip(" \n|")
        answers = [segment.strip() for segment in value.split("||")]
        answers = dedupe_answers([answer for answer in answers if answer])
        if answers:
            parsed[role] = answers
    return parsed


def render_qa_pairs(role_to_answers: dict[str, list[str]], predicate: str) -> list[dict[str, str]]:
    qa_pairs: list[dict[str, str]] = []
    predicate = predicate.strip()
    for role in ROLE_ORDER:
        answers = role_to_answers.get(role, [])
        if not answers:
            continue
        question = QUESTION_TEMPLATES.get(role, QUESTION_TEMPLATES["OTHER"]).format(predicate=predicate)
        for answer in answers:
            qa_pairs.append({"role": role, "question": question, "answer": answer})
    return qa_pairs


def _token_f1_local(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def align_answer_to_sentence(answer: str, sentence: str, max_extra_tokens: int = 4) -> str:
    answer = answer.replace("|", " ").strip()
    if not answer:
        return answer
    sentence_lower = sentence.lower()
    if answer.lower() in sentence_lower:
        start = sentence_lower.index(answer.lower())
        end = start + len(answer)
        return sentence[start:end].strip()
    sentence_tokens = sentence.split()
    answer_token_len = max(1, len(answer.split()))
    max_span_tokens = min(len(sentence_tokens), answer_token_len + max_extra_tokens)
    best_span = answer
    best_score = 0.0
    for start in range(len(sentence_tokens)):
        for end in range(start + 1, min(len(sentence_tokens), start + max_span_tokens) + 1):
            candidate = " ".join(sentence_tokens[start:end])
            score = _token_f1_local(candidate, answer)
            if score > best_score:
                best_score = score
                best_span = candidate
    return best_span if best_score >= 0.35 else answer


def guess_role_from_answer(answer: str) -> str | None:
    normalized = normalize_text(answer)
    if not normalized:
        return None
    if normalized.startswith("as of "):
        return "TIME"
    if normalized.startswith("because ") or normalized.startswith("so that ") or normalized.startswith("in order to "):
        return "REASON"
    if normalized.startswith("to "):
        tokens = normalized.split()
        if len(tokens) >= 3 and tokens[2] in DETERMINERS:
            return "REASON"
        return "LOCATION"
    if normalized.startswith("as "):
        return "ATTRIBUTE"
    if normalized.startswith("from "):
        return "SOURCE"
    if normalized.startswith("with ") or normalized.startswith("using "):
        return "INSTRUMENT"
    if normalized.startswith("by "):
        return "MANNER"
    if normalized.startswith(("in ", "on ", "at ", "under ", "over ", "inside ", "outside ", "within ", "near ", "between ", "among ", "amongst ", "to ")):
        return "LOCATION"
    if normalized.startswith(("before ", "after ", "during ", "since ", "until ", "as of ", "each ", "every ", "last ", "next ")):
        return "TIME"
    if any(time_word in normalized.split() for time_word in TIME_WORDS):
        return "TIME"
    if normalized.startswith(("for ", "about ", "of ")):
        return "OBLIQUE"
    if normalized[:1].isdigit() or normalized.startswith("$"):
        return "THEME"
    return None


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip()


def _strip_segment(tokens: list[str]) -> list[str]:
    cleaned = list(tokens)
    while cleaned and normalize_token(cleaned[0]) in LEADING_CLAUSE_WORDS | CLAUSE_BOUNDARIES:
        cleaned.pop(0)
    while cleaned and normalize_token(cleaned[-1]) in TRAILING_AUXILIARIES | CLAUSE_BOUNDARIES:
        cleaned.pop()
    return cleaned


def find_predicate_index(sentence: str, predicate: str) -> int:
    tokens = sentence.split()
    normalized_predicate = normalize_token(predicate)
    for index, token in enumerate(tokens):
        normalized = normalize_token(token)
        if normalized == normalized_predicate:
            return index
        if normalized_predicate and normalized.startswith(normalized_predicate):
            return index
    return -1


def _extract_subject(tokens: list[str], predicate_idx: int) -> str:
    if predicate_idx <= 0:
        return ""
    start = 0
    for index in range(predicate_idx - 1, -1, -1):
        if normalize_token(tokens[index]) in CLAUSE_BOUNDARIES:
            start = index + 1
            break
    segment = _strip_segment(tokens[start:predicate_idx])
    if not segment or normalize_token(segment[0]) in PHRASE_STARTERS:
        return ""
    if len(segment) > 8:
        segment = segment[-8:]
    return _join_tokens(segment)


def _extract_theme(tokens: list[str], predicate_idx: int) -> str:
    collected: list[str] = []
    for token in tokens[predicate_idx + 1 :]:
        normalized = normalize_token(token)
        if normalized in CLAUSE_BOUNDARIES:
            break
        if collected and normalized in PHRASE_STARTERS:
            break
        if not collected and normalized in {"to", "because", "as"}:
            break
        collected.append(token)
    collected = _strip_segment(collected)
    if not collected:
        return ""
    return _join_tokens(collected)


def _extract_tail_phrases(tokens: list[str], predicate_idx: int) -> list[str]:
    phrases: list[str] = []
    current: list[str] = []
    for token in tokens[predicate_idx + 1 :]:
        normalized = normalize_token(token)
        if normalized in CLAUSE_BOUNDARIES:
            if current:
                phrases.append(_join_tokens(current))
                current = []
            continue
        starts_new_phrase = normalized in PHRASE_STARTERS or (
            normalized == "to" and len(current) >= 1
        )
        if starts_new_phrase and current:
            phrases.append(_join_tokens(current))
            current = [token]
            continue
        if starts_new_phrase and not current:
            current = [token]
            continue
        if current:
            current.append(token)
    if current:
        phrases.append(_join_tokens(current))
    return [phrase for phrase in phrases if phrase]


def fallback_role_mapping(sentence: str, predicate: str) -> dict[str, list[str]]:
    tokens = sentence.split()
    predicate_idx = find_predicate_index(sentence, predicate)
    if predicate_idx < 0:
        return OrderedDict()

    fallback: dict[str, list[str]] = OrderedDict()
    subject = _extract_subject(tokens, predicate_idx)
    if subject:
        fallback["AGENT"] = [subject]

    theme = _extract_theme(tokens, predicate_idx)
    if theme:
        fallback.setdefault("THEME", []).append(theme)

    for phrase in _extract_tail_phrases(tokens, predicate_idx):
        guessed_role = guess_role_from_answer(phrase)
        if guessed_role and phrase not in fallback.get(guessed_role, []):
            fallback.setdefault(guessed_role, []).append(phrase)

    return normalize_role_mapping(fallback)


def refine_role_mapping(role_mapping: dict | None, sentence: str | None = None) -> dict[str, list[str]]:
    normalized = normalize_role_mapping(role_mapping)
    refined: dict[str, list[str]] = OrderedDict((role, []) for role in ROLE_ORDER)
    for role, answers in normalized.items():
        for index, answer in enumerate(answers):
            candidate = align_answer_to_sentence(answer, sentence) if sentence else answer
            guessed_role = guess_role_from_answer(candidate)
            target_role = role
            if role == "AGENT" and index > 0:
                target_role = guessed_role or "THEME"
            elif role in {"THEME", "OTHER", "OBLIQUE"} and guessed_role:
                target_role = guessed_role
            elif role == "AGENT" and guessed_role in {"LOCATION", "TIME", "MANNER", "SOURCE", "INSTRUMENT", "ATTRIBUTE", "OBLIQUE"}:
                target_role = guessed_role
            elif role in {"LOCATION", "TIME", "MANNER", "ATTRIBUTE", "SOURCE", "INSTRUMENT"} and guessed_role:
                target_role = guessed_role
            refined[target_role].append(candidate)
    return normalize_role_mapping(refined)
