from __future__ import annotations

import gzip
import io
import json
import random
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from .roles import dedupe_answers, format_role_output, infer_role


QASRL_V21_URL = "https://raw.githubusercontent.com/julianmichael/qasrl.org/master/data/qasrl-v2_1.tar"
ARCHIVE_NAME = "qasrl-v2_1.tar"
PROCESSED_VERSION = "v4"
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}


def ensure_archive(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / ARCHIVE_NAME
    if not archive_path.exists():
        urllib.request.urlretrieve(QASRL_V21_URL, archive_path)
    return archive_path


def mark_predicate(tokens: list[str], predicate_idx: int) -> str:
    marked_tokens = list(tokens)
    marked_tokens[predicate_idx] = f"<v> {marked_tokens[predicate_idx]} </v>"
    return " ".join(marked_tokens)


def detect_domain(sent_id: str) -> str:
    if sent_id.startswith("TQA"):
        return "TQA"
    parts = sent_id.split(":")
    return parts[1] if len(parts) > 1 else "unknown"


def extract_answers(question_obj: dict, tokens: list[str]) -> list[tuple[tuple[int, int], str]]:
    answer_spans: set[tuple[int, int]] = set()
    for judgment in question_obj.get("answerJudgments", []):
        if judgment.get("isValid"):
            for span in judgment.get("spans", []):
                answer_spans.add(tuple(span))
    ordered = sorted(answer_spans, key=lambda span: (span[0], span[1]))
    return [(span, " ".join(tokens[span[0] : span[1]])) for span in ordered]


def build_input_text(sentence: str, predicate: str, marked_sentence: str) -> str:
    return (
        "semantic role extraction\n"
        f"predicate: {predicate}\n"
        f"sentence: {sentence}\n"
        "labels:"
    )


def sentence_to_grouped_examples(sentence_obj: dict) -> list[dict]:
    tokens = sentence_obj["sentenceTokens"]
    sentence = " ".join(tokens)
    sent_id = sentence_obj["sentenceId"]
    domain = detect_domain(sent_id)
    examples: list[dict] = []

    for predicate_idx_str, verb_obj in sentence_obj.get("verbEntries", {}).items():
        predicate_idx = int(predicate_idx_str)
        predicate = tokens[predicate_idx]
        role_to_answers: dict[str, list[str]] = defaultdict(list)
        gold_questions: list[dict] = []

        for question_obj in verb_obj.get("questionLabels", {}).values():
            question_slots = question_obj.get("questionSlots", {})
            answers = extract_answers(question_obj, tokens)
            if not answers:
                continue
            role = infer_role(question_slots)
            for span, answer_text in answers:
                role_to_answers[role].append(answer_text)
                gold_questions.append(
                    {
                        "role": role,
                        "answer": answer_text,
                        "answer_span": span,
                        "question_slots": question_slots,
                    }
                )

        if not role_to_answers:
            continue

        for role, answers in role_to_answers.items():
            role_to_answers[role] = dedupe_answers(answers)

        marked_sentence = mark_predicate(tokens, predicate_idx)
        output_text = format_role_output(role_to_answers)
        examples.append(
            {
                "id": f"{sent_id}::{predicate_idx}",
                "sentence": sentence,
                "marked_sentence": marked_sentence,
                "predicate": predicate,
                "predicate_idx": predicate_idx,
                "sent_id": sent_id,
                "domain": domain,
                "input_text": build_input_text(sentence, predicate, marked_sentence),
                "target_text": output_text,
                "roles": role_to_answers,
                "gold_questions": gold_questions,
            }
        )

    return examples


def iter_split_examples(archive_path: str | Path, split_name: str):
    member_suffix = f"qasrl-v2_1/orig/{SPLIT_MAP[split_name]}.jsonl.gz"
    with tarfile.open(archive_path, "r") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(member_suffix))
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member_suffix)
        with gzip.GzipFile(fileobj=io.BytesIO(extracted.read())) as gz_file:
            for raw_line in gz_file:
                sentence_obj = json.loads(raw_line.decode("utf-8"))
                for example in sentence_to_grouped_examples(sentence_obj):
                    yield example


def write_processed_split(processed_path: Path, examples: list[dict]) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")


def read_processed_split(processed_path: Path) -> list[dict]:
    with processed_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def ensure_processed_dataset(data_dir: str | Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_paths = {
        split: processed_dir / f"{split}_grouped_{PROCESSED_VERSION}.jsonl"
        for split in ("train", "validation", "test")
    }
    if all(path.exists() for path in processed_paths.values()):
        return processed_paths

    archive_path = ensure_archive(data_dir / "cache")
    for split, output_path in processed_paths.items():
        examples = list(iter_split_examples(archive_path, split))
        write_processed_split(output_path, examples)
    return processed_paths


def sample_records(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:limit]


def prepare_grouped_dataset(
    data_dir: str | Path,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 42,
) -> DatasetDict:
    processed_paths = ensure_processed_dataset(data_dir)
    split_limits = {
        "train": train_limit,
        "validation": validation_limit,
        "test": test_limit,
    }
    dataset = {}
    for split, path in processed_paths.items():
        records = read_processed_split(path)
        sampled = sample_records(records, split_limits[split], seed)
        dataset[split] = Dataset.from_list(sampled)
    return DatasetDict(dataset)
