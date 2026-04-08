from __future__ import annotations

import json
from collections import defaultdict

from .roles import ROLE_ORDER, normalize_role_mapping, normalize_text, parse_role_output


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred_roles: dict[str, list[str]], gold_roles: dict[str, list[str]]) -> float:
    pred_roles = normalize_role_mapping(pred_roles)
    gold_roles = normalize_role_mapping(gold_roles)
    normalized_pred = {
        role: sorted(normalize_text(answer) for answer in answers) for role, answers in pred_roles.items()
    }
    normalized_gold = {
        role: sorted(normalize_text(answer) for answer in answers) for role, answers in gold_roles.items()
    }
    return 1.0 if normalized_pred == normalized_gold else 0.0


def role_coverage(pred_roles: dict[str, list[str]], gold_roles: dict[str, list[str]]) -> float:
    pred_roles = normalize_role_mapping(pred_roles)
    gold_roles = normalize_role_mapping(gold_roles)
    gold_role_set = {role for role, answers in gold_roles.items() if answers}
    if not gold_role_set:
        return 1.0
    pred_role_set = {role for role, answers in pred_roles.items() if answers}
    return len(gold_role_set & pred_role_set) / len(gold_role_set)


def lcs_length(x_tokens: list[str], y_tokens: list[str]) -> int:
    if not x_tokens or not y_tokens:
        return 0
    previous = [0] * (len(y_tokens) + 1)
    for x_token in x_tokens:
        current = [0]
        for index, y_token in enumerate(y_tokens, start=1):
            if x_token == y_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_role_level_f1(pred_roles: dict[str, list[str]], gold_roles: dict[str, list[str]]) -> float:
    pred_roles = normalize_role_mapping(pred_roles)
    gold_roles = normalize_role_mapping(gold_roles)
    roles = sorted(set(pred_roles) | set(gold_roles) | set(ROLE_ORDER))
    scores = []
    for role in roles:
        pred_text = " | ".join(pred_roles.get(role, []))
        gold_text = " | ".join(gold_roles.get(role, []))
        if pred_text or gold_text:
            scores.append(token_f1(pred_text, gold_text))
    return sum(scores) / len(scores) if scores else 1.0


def compute_dataset_metrics(records: list[dict], prediction_texts: list[str]) -> dict:
    assert len(records) == len(prediction_texts)
    token_f1_scores = []
    exact_match_scores = []
    role_coverage_scores = []
    rouge_l_scores = []
    domain_buckets = defaultdict(list)
    per_example = []

    for record, prediction_text in zip(records, prediction_texts):
        pred_roles = parse_role_output(prediction_text)
        gold_roles = normalize_role_mapping(record["roles"])
        record_token_f1 = compute_role_level_f1(pred_roles, gold_roles)
        record_exact = exact_match(pred_roles, gold_roles)
        record_coverage = role_coverage(pred_roles, gold_roles)
        record_rouge_l = rouge_l_f1(prediction_text, record["target_text"])
        token_f1_scores.append(record_token_f1)
        exact_match_scores.append(record_exact)
        role_coverage_scores.append(record_coverage)
        rouge_l_scores.append(record_rouge_l)
        domain_buckets[record["domain"]].append(record_token_f1)
        per_example.append(
            {
                "id": record["id"],
                "sentence": record["sentence"],
                "predicate": record["predicate"],
                "domain": record["domain"],
                "gold": gold_roles,
                "prediction_text": prediction_text,
                "prediction_roles": pred_roles,
                "token_f1": record_token_f1,
                "exact_match": record_exact,
                "role_coverage": record_coverage,
                "rouge_l": record_rouge_l,
            }
        )

    domain_metrics = {
        domain: round(sum(scores) / len(scores), 4) for domain, scores in sorted(domain_buckets.items())
    }
    return {
        "token_f1": round(sum(token_f1_scores) / len(token_f1_scores), 4),
        "exact_match": round(sum(exact_match_scores) / len(exact_match_scores), 4),
        "role_coverage": round(sum(role_coverage_scores) / len(role_coverage_scores), 4),
        "rouge_l": round(sum(rouge_l_scores) / len(rouge_l_scores), 4),
        "domain_token_f1": domain_metrics,
        "per_example": per_example,
    }


def save_metrics(metrics: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
