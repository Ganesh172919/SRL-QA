"""Data loading and preprocessing utilities for the PropBank SRL-QA project.

The preprocessing pipeline follows the project brief closely:

1. Load real PropBank instances through NLTK.
2. Align each instance with the locally available Penn Treebank sample.
3. Reconstruct sentence text and argument spans from Treebank leaves.
4. Generate SRL BIO tags and natural-language QA pairs.
5. Split the QA data deterministically into train/validation/test sets.
6. Build token, POS, and BIO-label vocabularies for downstream modeling.

The loader intentionally stays anchored to the local NLTK corpus assets under
``srl_qa_project/nltk_data`` so the pipeline remains fully reproducible inside
this repository.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import nltk
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.utils.data import DataLoader, Dataset

from config import ProjectConfig

ExampleDict = Dict[str, Any]

DETOKENIZER = TreebankWordDetokenizer()
MONTH_WORDS = {
    "jan.",
    "january",
    "feb.",
    "february",
    "mar.",
    "march",
    "apr.",
    "april",
    "may",
    "jun.",
    "june",
    "jul.",
    "july",
    "aug.",
    "august",
    "sep.",
    "sept.",
    "september",
    "oct.",
    "october",
    "nov.",
    "november",
    "dec.",
    "december",
}
DAY_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "yesterday",
    "today",
    "tomorrow",
}
LOCATION_PREPOSITIONS = {"in", "at", "on", "near", "inside", "outside", "across"}
ROLE_TO_QTYPE = {
    "ARG0": "WHO",
    "ARG1": "WHAT",
    "ARG2": "WHAT",
    "ARG3": "WHAT",
    "ARG4": "WHAT",
    "ARG5": "WHAT",
    "ARGA": "WHAT",
    "ARGM-TMP": "WHEN",
    "ARGM-LOC": "WHERE",
    "ARGM-DIR": "WHERE",
    "ARGM-GOL": "WHERE",
    "ARGM-MNR": "HOW",
    "ARGM-CAU": "WHY",
    "ARGM-PRP": "WHY",
    "ARGM-PNC": "WHY",
    "ARGM-ADV": "HOW",
    "ARGM-EXT": "HOW",
}


@dataclass(slots=True)
class Vocabulary:
    """Simple string-to-index vocabulary."""

    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @classmethod
    def build(
        cls,
        sequences: Iterable[Sequence[str]],
        min_frequency: int = 1,
        specials: Sequence[str] | None = None,
    ) -> "Vocabulary":
        """Build a vocabulary from token sequences."""

        counter: Counter[str] = Counter()
        for sequence in sequences:
            counter.update(sequence)

        specials = list(specials or [])
        token_to_id: Dict[str, int] = {}
        id_to_token: List[str] = []

        for token in specials:
            if token not in token_to_id:
                token_to_id[token] = len(id_to_token)
                id_to_token.append(token)

        for token, frequency in sorted(counter.items()):
            if frequency < min_frequency or token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)

        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, tokens: Sequence[str], unknown_token: str = "<unk>") -> List[int]:
        """Convert a token sequence into ids."""

        unknown_id = self.token_to_id.get(unknown_token, 0)
        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    def to_dict(self) -> Dict[str, Any]:
        """Export the vocabulary into a JSON-serializable dictionary."""

        return {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }


class SRLQADataset(Dataset[Dict[str, Any]]):
    """Torch dataset wrapper around preprocessed QA examples."""

    def __init__(
        self,
        examples: Sequence[ExampleDict],
        token_vocab: Vocabulary,
        pos_vocab: Vocabulary,
        label_vocab: Vocabulary,
        config: ProjectConfig,
        split_name: str,
    ) -> None:
        """Encode JSON-like examples into fixed-format tensors."""

        self.examples: List[Dict[str, Any]] = []
        max_sentence_length = config.data.max_sentence_length
        max_question_length = config.data.max_question_length
        pad_label = label_vocab.token_to_id["O"]

        skipped = 0
        for example in examples:
            context_tokens = [
                normalize_token(token, config.data.lowercase_tokens)
                for token in example["context_tokens"]
            ]
            question_tokens = [
                normalize_token(token, config.data.lowercase_tokens)
                for token in example["question_tokens"]
            ]

            if len(context_tokens) > max_sentence_length:
                skipped += 1
                continue
            if example["answer_end"] >= max_sentence_length:
                skipped += 1
                continue

            encoded = {
                "context_ids": token_vocab.encode(context_tokens),
                "question_ids": token_vocab.encode(question_tokens),
                "pos_ids": pos_vocab.encode(example["pos_tags"]),
                "predicate_flags": example["predicate_flags"],
                "label_ids": [
                    label_vocab.token_to_id.get(label, pad_label)
                    for label in example["srl_tags"]
                ],
                "answer_start": example["answer_start"],
                "answer_end": example["answer_end"],
                "raw": example,
            }

            if len(encoded["question_ids"]) > max_question_length:
                encoded["question_ids"] = encoded["question_ids"][:max_question_length]

            self.examples.append(encoded)

        if config.runtime.verbose:
            print(
                f"[data] encoded {split_name} split with {len(self.examples)} items"
                f" (skipped {skipped})"
            )

    def __len__(self) -> int:
        """Return the number of encoded examples."""

        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a single encoded example."""

        return self.examples[index]


def initialize_nltk(config: ProjectConfig) -> None:
    """Register the local project NLTK data directory."""

    nltk_path = str(config.paths.nltk_data_dir)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)


def normalize_token(token: str, lowercase: bool) -> str:
    """Normalize a token for vocabulary building."""

    return token.lower() if lowercase else token


def simple_word_tokenize(text: str) -> List[str]:
    """Tokenize text into words and punctuation using a regex fallback."""

    return re.findall(r"[A-Za-z0-9$%]+(?:[-'][A-Za-z0-9$%]+)*|[^\w\s]", text)


def detokenize(tokens: Sequence[str]) -> str:
    """Detokenize Treebank-style tokens into a readable string."""

    return DETOKENIZER.detokenize(list(tokens)).strip()


def strip_bio_prefix(label: str) -> str:
    """Remove BIO prefixes from an SRL tag."""

    if label == "O":
        return label
    if "-" not in label:
        return label
    return label.split("-", maxsplit=1)[1]


def split_contiguous(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Split sorted token indices into contiguous spans."""

    if not indices:
        return []

    spans: List[Tuple[int, int]] = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append((start, previous))
        start = index
        previous = index
    spans.append((start, previous))
    return spans


def build_visible_token_view(
    tree: nltk.Tree,
) -> Tuple[List[str], List[str], Dict[int, int], Dict[int, Tuple[int, ...]]]:
    """Create a visible-token view of a Treebank tree."""

    tokens: List[str] = []
    pos_tags: List[str] = []
    original_to_visible: Dict[int, int] = {}
    leaf_positions: Dict[int, Tuple[int, ...]] = {}

    for leaf_index, leaf_token in enumerate(tree.leaves()):
        leaf_position = tree.leaf_treeposition(leaf_index)
        leaf_positions[leaf_index] = leaf_position
        pos_label = tree[leaf_position[:-1]].label()
        if pos_label == "-NONE-":
            continue
        original_to_visible[leaf_index] = len(tokens)
        tokens.append(leaf_token)
        pos_tags.append(pos_label)

    return tokens, pos_tags, original_to_visible, leaf_positions


def collect_original_leaf_indices(
    tree: nltk.Tree,
    pointer: Any,
    leaf_positions: Dict[int, Tuple[int, ...]],
) -> List[int]:
    """Resolve a PropBank pointer into original Treebank leaf indices."""

    indices: set[int] = set()
    pieces = flatten_pointer_pieces(pointer)
    for piece in pieces:
        tree_position = piece.treepos(tree)
        for leaf_index, leaf_position in leaf_positions.items():
            if leaf_position[: len(tree_position)] == tree_position:
                indices.add(leaf_index)
    return sorted(indices)


def flatten_pointer_pieces(pointer: Any) -> List[Any]:
    """Flatten nested PropBank pointer structures into tree pointers."""

    pieces = getattr(pointer, "pieces", None)
    if not pieces:
        return [pointer]

    flattened: List[Any] = []
    for piece in pieces:
        flattened.extend(flatten_pointer_pieces(piece))
    return flattened


def visible_indices_for_pointer(
    tree: nltk.Tree,
    pointer: Any,
    original_to_visible: Dict[int, int],
    leaf_positions: Dict[int, Tuple[int, ...]],
) -> List[int]:
    """Resolve a PropBank pointer into visible token indices."""

    original_indices = collect_original_leaf_indices(tree, pointer, leaf_positions)
    return sorted(
        original_to_visible[index]
        for index in original_indices
        if index in original_to_visible
    )


def assign_bio_labels(
    labels: List[str],
    spans: Sequence[Tuple[int, int]],
    role: str,
) -> None:
    """Write BIO SRL tags for one role into a sentence label sequence."""

    for start, end in spans:
        for index in range(start, end + 1):
            candidate = f"B-{role}" if index == start else f"I-{role}"
            if labels[index] == "O":
                labels[index] = candidate


def heuristic_named_entities(tokens: Sequence[str], pos_tags: Sequence[str]) -> List[str]:
    """Derive lightweight named-entity tags without external models."""

    ne_tags: List[str] = []
    for token, pos_tag in zip(tokens, pos_tags, strict=True):
        lower = token.lower()
        if pos_tag in {"NNP", "NNPS"}:
            if lower in MONTH_WORDS or lower in DAY_WORDS:
                ne_tags.append("DATE")
            else:
                ne_tags.append("PROPER")
        elif pos_tag == "CD":
            if "%" in token:
                ne_tags.append("PERCENT")
            elif token.startswith("$"):
                ne_tags.append("MONEY")
            else:
                ne_tags.append("NUMBER")
        elif lower in MONTH_WORDS or lower in DAY_WORDS:
            ne_tags.append("DATE")
        elif token.istitle():
            ne_tags.append("PROPER")
        else:
            ne_tags.append("O")
    return ne_tags


def heuristic_dependency_labels(
    tokens: Sequence[str],
    pos_tags: Sequence[str],
    predicate_flags: Sequence[int],
    srl_tags: Sequence[str],
) -> List[str]:
    """Create coarse dependency-like labels from SRL and POS evidence."""

    labels: List[str] = []
    for token, pos_tag, predicate_flag, srl_tag in zip(
        tokens, pos_tags, predicate_flags, srl_tags, strict=True
    ):
        role = strip_bio_prefix(srl_tag)
        lower = token.lower()

        if predicate_flag:
            labels.append("root")
        elif role == "ARG0":
            labels.append("nsubj")
        elif role == "ARG1":
            labels.append("obj")
        elif role in {"ARG2", "ARG3", "ARG4", "ARG5"}:
            labels.append("iobj")
        elif role == "ARGM-TMP":
            labels.append("obl:tmod")
        elif role in {"ARGM-LOC", "ARGM-DIR", "ARGM-GOL"}:
            labels.append("obl:loc")
        elif role in {"ARGM-MNR", "ARGM-ADV", "ARGM-EXT"}:
            labels.append("advmod")
        elif role in {"ARGM-CAU", "ARGM-PRP", "ARGM-PNC"}:
            labels.append("advcl")
        elif pos_tag in {",", ".", ":", "``", "''"}:
            labels.append("punct")
        elif pos_tag in {"DT", "PDT", "WDT"}:
            labels.append("det")
        elif pos_tag == "CD":
            labels.append("nummod")
        elif pos_tag.startswith("JJ"):
            labels.append("amod")
        elif pos_tag.startswith("RB"):
            labels.append("advmod")
        elif pos_tag in {"IN", "TO"} and lower in LOCATION_PREPOSITIONS:
            labels.append("case")
        elif pos_tag in {"NNP", "NNPS"}:
            labels.append("compound")
        else:
            labels.append("dep")
    return labels


def infer_question_type(role: str, role_description: str) -> str:
    """Infer a coarse question type for a target semantic role."""

    if role in ROLE_TO_QTYPE:
        return ROLE_TO_QTYPE[role]

    description = role_description.lower()
    if "agent" in description or "speaker" in description or "person" in description:
        return "WHO"
    if "time" in description or "temporal" in description:
        return "WHEN"
    if "location" in description or "place" in description or "destination" in description:
        return "WHERE"
    if "manner" in description or "extent" in description:
        return "HOW"
    if "cause" in description or "reason" in description or "purpose" in description:
        return "WHY"
    return "WHAT"


def build_question(
    role: str,
    question_type: str,
    predicate_lemma: str,
    predicate_text: str,
    subject_text: str,
    role_description: str,
) -> str:
    """Generate a natural-language question for one semantic argument."""

    subject = subject_text if subject_text else "the predicate event"
    predicate = predicate_lemma.replace("_", " ")
    predicate_surface = predicate_text if predicate_text else predicate
    description = role_description.lower().strip()

    if role == "ARG0":
        return f"Who {predicate_surface}?"
    if role == "ARG1" and subject_text:
        return f"What did {subject} {predicate}?"
    if role == "ARGM-TMP":
        return f"When did {subject} {predicate}?"
    if role in {"ARGM-LOC", "ARGM-DIR", "ARGM-GOL"}:
        return f"Where did {subject} {predicate}?"
    if role in {"ARGM-MNR", "ARGM-ADV", "ARGM-EXT"}:
        return f"How did {subject} {predicate}?"
    if role in {"ARGM-CAU", "ARGM-PRP", "ARGM-PNC"}:
        return f"Why did {subject} {predicate}?"
    if role == "ARG2" and description:
        return f"What was the {description} in the event where {subject} {predicate}?"
    if question_type == "WHO":
        return f"Who was the {description or role.lower()} in the event where {subject} {predicate}?"
    if question_type == "WHEN":
        return f"When did {subject} {predicate}?"
    if question_type == "WHERE":
        return f"Where did {subject} {predicate}?"
    if question_type == "WHY":
        return f"Why did {subject} {predicate}?"
    if question_type == "HOW":
        return f"How did {subject} {predicate}?"
    if description:
        return f"What was the {description} in the event where {subject} {predicate}?"
    return f"What was the {role.lower()} answer for the predicate {predicate_surface}?"


def roleset_metadata(propbank_corpus: Any, roleset_id: str) -> Dict[str, Any]:
    """Load descriptive metadata for a PropBank roleset."""

    try:
        roleset_xml = propbank_corpus.roleset(roleset_id)
    except ValueError:
        return {
            "roleset_id": roleset_id,
            "roleset_name": roleset_id,
            "roleset_vncls": "",
            "role_descriptions": {},
        }
    role_descriptions: Dict[str, str] = {}
    roles_node = roleset_xml.find("roles")
    if roles_node is not None:
        for role_node in roles_node.findall("role"):
            role_number = role_node.attrib.get("n", "").strip()
            if not role_number:
                continue
            role_descriptions[f"ARG{role_number}"] = role_node.attrib.get("descr", "")

    return {
        "roleset_id": roleset_id,
        "roleset_name": roleset_xml.attrib.get("name", ""),
        "roleset_vncls": roleset_xml.attrib.get("vncls", ""),
        "role_descriptions": role_descriptions,
    }


def inspect_corpus(config: ProjectConfig) -> Dict[str, Any]:
    """Inspect the local PropBank installation and return sample metadata."""

    initialize_nltk(config)
    from nltk.corpus import propbank, treebank

    instances = propbank.instances()
    treebank_fileids = set(treebank.fileids())
    usable_count = sum(1 for instance in instances if instance.fileid in treebank_fileids)
    sample_instance = instances[0]

    return {
        "total_instances": len(instances),
        "usable_instances": usable_count,
        "treebank_file_count": len(treebank_fileids),
        "sample_roleset": sample_instance.roleset,
        "sample_fileid": sample_instance.fileid,
        "sample_sentence_index": sample_instance.sentnum,
        "sample_predicate": str(sample_instance.predicate),
        "sample_tree": sample_instance.tree.pformat(margin=100),
        "sample_arguments": [
            {"pointer": str(pointer), "label": label}
            for pointer, label in sample_instance.arguments
        ],
    }


def build_examples_from_propbank(config: ProjectConfig) -> Tuple[List[ExampleDict], Dict[str, Any]]:
    """Create SRL-QA examples directly from the real PropBank corpus."""

    initialize_nltk(config)
    from nltk.corpus import propbank, treebank

    instances = propbank.instances()
    treebank_fileids = set(treebank.fileids())
    metadata_cache: Dict[str, Dict[str, Any]] = {}

    examples: List[ExampleDict] = []
    unique_predicates: set[str] = set()
    unique_rolesets: set[str] = set()
    unique_instance_ids: set[str] = set()
    sentence_lengths: Dict[str, int] = {}
    answer_lengths: List[int] = []
    role_counter: Counter[str] = Counter()
    qtype_counter: Counter[str] = Counter()
    dropped_noncontiguous = 0

    for index, instance in enumerate(instances):
        if config.data.max_instances is not None and len(unique_instance_ids) >= config.data.max_instances:
            break
        if instance.fileid not in treebank_fileids or instance.tree is None:
            continue

        tree = instance.tree
        tokens, pos_tags, original_to_visible, leaf_positions = build_visible_token_view(tree)
        if not tokens:
            continue

        predicate_indices = visible_indices_for_pointer(
            tree=tree,
            pointer=instance.predicate,
            original_to_visible=original_to_visible,
            leaf_positions=leaf_positions,
        )
        if not predicate_indices:
            continue

        predicate_flags = [
            1 if token_index in predicate_indices else 0
            for token_index in range(len(tokens))
        ]
        srl_tags = ["O"] * len(tokens)

        if instance.roleset not in metadata_cache:
            metadata_cache[instance.roleset] = roleset_metadata(propbank, instance.roleset)
        role_metadata = metadata_cache[instance.roleset]

        argument_entries: List[Dict[str, Any]] = []
        subject_text = ""

        for pointer, label in instance.arguments:
            if not label.startswith("ARG"):
                continue

            visible_indices = visible_indices_for_pointer(
                tree=tree,
                pointer=pointer,
                original_to_visible=original_to_visible,
                leaf_positions=leaf_positions,
            )
            if not visible_indices:
                continue

            spans = split_contiguous(visible_indices)
            assign_bio_labels(srl_tags, spans, label)

            role_description = role_metadata["role_descriptions"].get(label, "")
            segment_texts = [
                detokenize(tokens[start : end + 1])
                for start, end in spans
            ]
            answer_text = " ; ".join(segment_texts)
            is_contiguous = len(spans) == 1

            entry = {
                "role": label,
                "role_description": role_description,
                "text": answer_text,
                "token_indices": visible_indices,
                "spans": spans,
                "is_contiguous": is_contiguous,
            }
            argument_entries.append(entry)
            role_counter[label] += 1

            if label == "ARG0" and not subject_text and answer_text:
                subject_text = answer_text

        if not argument_entries:
            continue

        context = detokenize(tokens)
        predicate_text = detokenize([tokens[position] for position in predicate_indices])
        ne_tags = heuristic_named_entities(tokens, pos_tags)
        dependency_labels = heuristic_dependency_labels(
            tokens=tokens,
            pos_tags=pos_tags,
            predicate_flags=predicate_flags,
            srl_tags=srl_tags,
        )

        instance_id = f"{instance.fileid}:{instance.sentnum}:{instance.predid}:{instance.roleset}"
        unique_instance_ids.add(instance_id)
        sentence_lengths[instance_id] = len(tokens)
        unique_predicates.add(instance.baseform)
        unique_rolesets.add(instance.roleset)

        for argument in argument_entries:
            if not argument["is_contiguous"]:
                dropped_noncontiguous += 1
                continue

            start, end = argument["spans"][0]
            question_type = infer_question_type(
                role=argument["role"],
                role_description=argument["role_description"],
            )
            question = build_question(
                role=argument["role"],
                question_type=question_type,
                predicate_lemma=instance.baseform,
                predicate_text=predicate_text,
                subject_text=subject_text,
                role_description=argument["role_description"],
            )

            question_tokens = simple_word_tokenize(question)
            example = {
                "example_id": f"qa_{len(examples):06d}",
                "instance_id": instance_id,
                "fileid": instance.fileid,
                "sentnum": instance.sentnum,
                "context": context,
                "context_tokens": tokens,
                "question": question,
                "question_tokens": question_tokens,
                "answer_text": argument["text"],
                "answer_tokens": tokens[start : end + 1],
                "answer_start": start,
                "answer_end": end,
                "answer_length": end - start + 1,
                "predicate_lemma": instance.baseform,
                "predicate_text": predicate_text,
                "predicate_indices": predicate_indices,
                "predicate_flags": predicate_flags,
                "roleset_id": role_metadata["roleset_id"],
                "roleset_name": role_metadata["roleset_name"],
                "roleset_vncls": role_metadata["roleset_vncls"],
                "target_role": argument["role"],
                "target_role_description": argument["role_description"],
                "question_type": question_type,
                "pos_tags": pos_tags,
                "ne_tags": ne_tags,
                "dependency_labels": dependency_labels,
                "srl_tags": srl_tags,
                "argument_spans": argument_entries,
            }
            examples.append(example)
            qtype_counter[question_type] += 1
            answer_lengths.append(example["answer_length"])

        if config.runtime.verbose and index > 0 and index % 2000 == 0:
            print(
                f"[data] processed {index} PropBank instances -> {len(examples)} QA examples"
            )

    sentence_length_values = list(sentence_lengths.values())
    stats = {
        "total_propbank_instances": len(instances),
        "usable_propbank_instances": len(unique_instance_ids),
        "qa_pair_count": len(examples),
        "unique_predicates": len(unique_predicates),
        "unique_rolesets": len(unique_rolesets),
        "argument_type_distribution": dict(sorted(role_counter.items())),
        "qa_pairs_per_argument_type": dict(sorted(role_counter.items())),
        "qa_pairs_per_question_type": dict(sorted(qtype_counter.items())),
        "sentence_length_distribution": sentence_length_values,
        "answer_span_length_distribution": answer_lengths,
        "sentence_length_summary": summarize_numeric_values(sentence_length_values),
        "answer_length_summary": summarize_numeric_values(answer_lengths),
        "dropped_noncontiguous_arguments": dropped_noncontiguous,
        "sample_qa_pairs": [
            {
                "context": example["context"],
                "question": example["question"],
                "answer": example["answer_text"],
                "role": example["target_role"],
            }
            for example in examples[:5]
        ],
    }

    return examples, stats


def summarize_numeric_values(values: Sequence[int]) -> Dict[str, float]:
    """Summarize a numeric sequence."""

    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def split_examples(
    examples: Sequence[ExampleDict],
    config: ProjectConfig,
) -> Tuple[List[ExampleDict], List[ExampleDict], List[ExampleDict]]:
    """Split QA examples into deterministic train/validation/test sets."""

    shuffled = list(examples)
    random.Random(config.data.random_seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * config.data.train_ratio)
    validation_end = train_end + int(total * config.data.validation_ratio)

    train_split = shuffled[:train_end]
    validation_split = shuffled[train_end:validation_end]
    test_split = shuffled[validation_end:]
    return train_split, validation_split, test_split


def save_json(path: Path, payload: Any) -> None:
    """Save JSON payloads with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_pointer:
        json.dump(payload, file_pointer, indent=2)


def load_json(path: Path) -> Any:
    """Load a JSON payload from disk."""

    with path.open("r", encoding="utf-8") as file_pointer:
        return json.load(file_pointer)


def compute_statistics_from_examples(
    config: ProjectConfig,
    examples: Sequence[ExampleDict],
) -> Dict[str, Any]:
    """Recompute descriptive statistics from cached examples."""

    corpus_overview = inspect_corpus(config)
    unique_predicates = {example["predicate_lemma"] for example in examples}
    unique_rolesets = {example["roleset_id"] for example in examples}
    role_counter = Counter(example["target_role"] for example in examples)
    qtype_counter = Counter(example["question_type"] for example in examples)
    instance_lengths: Dict[str, int] = {}
    answer_lengths: List[int] = []

    for example in examples:
        instance_lengths[example["instance_id"]] = len(example["context_tokens"])
        answer_lengths.append(example["answer_length"])

    return {
        "total_propbank_instances": corpus_overview["total_instances"],
        "usable_propbank_instances": len(instance_lengths),
        "qa_pair_count": len(examples),
        "unique_predicates": len(unique_predicates),
        "unique_rolesets": len(unique_rolesets),
        "argument_type_distribution": dict(sorted(role_counter.items())),
        "qa_pairs_per_argument_type": dict(sorted(role_counter.items())),
        "qa_pairs_per_question_type": dict(sorted(qtype_counter.items())),
        "sentence_length_distribution": list(instance_lengths.values()),
        "answer_span_length_distribution": answer_lengths,
        "sentence_length_summary": summarize_numeric_values(list(instance_lengths.values())),
        "answer_length_summary": summarize_numeric_values(answer_lengths),
        "sample_qa_pairs": [
            {
                "context": example["context"],
                "question": example["question"],
                "answer": example["answer_text"],
                "role": example["target_role"],
            }
            for example in examples[:5]
        ],
    }


def load_or_build_splits(
    config: ProjectConfig,
) -> Tuple[List[ExampleDict], List[ExampleDict], List[ExampleDict], Dict[str, Any]]:
    """Load cached dataset splits or rebuild them from PropBank."""

    if (
        config.paths.train_json.exists()
        and config.paths.val_json.exists()
        and config.paths.test_json.exists()
        and not config.data.rebuild_cache
    ):
        train_split = load_json(config.paths.train_json)
        validation_split = load_json(config.paths.val_json)
        test_split = load_json(config.paths.test_json)
        merged = train_split + validation_split + test_split
        stats = compute_statistics_from_examples(config, merged)
        stats["split_sizes"] = {
            "train": len(train_split),
            "validation": len(validation_split),
            "test": len(test_split),
        }
        return train_split, validation_split, test_split, stats

    examples, stats = build_examples_from_propbank(config)
    train_split, validation_split, test_split = split_examples(examples, config)

    save_json(config.paths.train_json, train_split)
    save_json(config.paths.val_json, validation_split)
    save_json(config.paths.test_json, test_split)

    stats["split_sizes"] = {
        "train": len(train_split),
        "validation": len(validation_split),
        "test": len(test_split),
    }
    return train_split, validation_split, test_split, stats


def build_feature_vocabs(
    train_examples: Sequence[ExampleDict],
    config: ProjectConfig,
) -> Dict[str, Vocabulary]:
    """Build vocabularies from the training corpus."""

    token_sequences = [
        [
            normalize_token(token, config.data.lowercase_tokens)
            for token in example["context_tokens"]
        ]
        for example in train_examples
    ] + [
        [
            normalize_token(token, config.data.lowercase_tokens)
            for token in example["question_tokens"]
        ]
        for example in train_examples
    ]

    pos_sequences = [example["pos_tags"] for example in train_examples]
    label_sequences = [example["srl_tags"] for example in train_examples]

    token_vocab = Vocabulary.build(
        sequences=token_sequences,
        min_frequency=config.data.min_token_frequency,
        specials=["<pad>", "<unk>"],
    )
    pos_vocab = Vocabulary.build(
        sequences=pos_sequences,
        min_frequency=1,
        specials=["<pad>", "<unk>"],
    )

    labels = sorted({label for sequence in label_sequences for label in sequence})
    if "O" in labels:
        labels.remove("O")
        labels.insert(0, "O")
    label_vocab = Vocabulary(
        token_to_id={label: index for index, label in enumerate(labels)},
        id_to_token=labels,
    )

    return {
        "token_vocab": token_vocab,
        "pos_vocab": pos_vocab,
        "label_vocab": label_vocab,
    }


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length examples into padded torch tensors."""

    batch_size = len(batch)
    max_context_length = max(len(item["context_ids"]) for item in batch)
    max_question_length = max(len(item["question_ids"]) for item in batch)

    context_ids = torch.zeros((batch_size, max_context_length), dtype=torch.long)
    pos_ids = torch.zeros((batch_size, max_context_length), dtype=torch.long)
    predicate_flags = torch.zeros((batch_size, max_context_length), dtype=torch.long)
    label_ids = torch.zeros((batch_size, max_context_length), dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_context_length), dtype=torch.bool)

    question_ids = torch.zeros((batch_size, max_question_length), dtype=torch.long)
    question_mask = torch.zeros((batch_size, max_question_length), dtype=torch.bool)

    answer_starts = torch.zeros(batch_size, dtype=torch.long)
    answer_ends = torch.zeros(batch_size, dtype=torch.long)
    raw_examples: List[ExampleDict] = []

    for batch_index, item in enumerate(batch):
        context_length = len(item["context_ids"])
        question_length = len(item["question_ids"])

        context_ids[batch_index, :context_length] = torch.tensor(
            item["context_ids"],
            dtype=torch.long,
        )
        pos_ids[batch_index, :context_length] = torch.tensor(
            item["pos_ids"],
            dtype=torch.long,
        )
        predicate_flags[batch_index, :context_length] = torch.tensor(
            item["predicate_flags"],
            dtype=torch.long,
        )
        label_ids[batch_index, :context_length] = torch.tensor(
            item["label_ids"],
            dtype=torch.long,
        )
        context_mask[batch_index, :context_length] = True

        question_ids[batch_index, :question_length] = torch.tensor(
            item["question_ids"],
            dtype=torch.long,
        )
        question_mask[batch_index, :question_length] = True

        answer_starts[batch_index] = item["answer_start"]
        answer_ends[batch_index] = item["answer_end"]
        raw_examples.append(item["raw"])

    return {
        "context_ids": context_ids,
        "pos_ids": pos_ids,
        "predicate_flags": predicate_flags,
        "label_ids": label_ids,
        "context_mask": context_mask,
        "question_ids": question_ids,
        "question_mask": question_mask,
        "answer_starts": answer_starts,
        "answer_ends": answer_ends,
        "raw_examples": raw_examples,
    }


def build_dataloaders(
    train_examples: Sequence[ExampleDict],
    validation_examples: Sequence[ExampleDict],
    test_examples: Sequence[ExampleDict],
    config: ProjectConfig,
) -> Tuple[Dict[str, DataLoader], Dict[str, Vocabulary]]:
    """Build torch dataloaders and vocabularies for all splits."""

    vocabularies = build_feature_vocabs(train_examples, config)
    token_vocab = vocabularies["token_vocab"]
    pos_vocab = vocabularies["pos_vocab"]
    label_vocab = vocabularies["label_vocab"]

    datasets = {
        "train": SRLQADataset(
            train_examples,
            token_vocab,
            pos_vocab,
            label_vocab,
            config,
            "train",
        ),
        "validation": SRLQADataset(
            validation_examples,
            token_vocab,
            pos_vocab,
            label_vocab,
            config,
            "validation",
        ),
        "test": SRLQADataset(
            test_examples,
            token_vocab,
            pos_vocab,
            label_vocab,
            config,
            "test",
        ),
    }

    dataloaders = {
        split_name: DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.training.num_workers,
            collate_fn=collate_batch,
        )
        for split_name, dataset in datasets.items()
    }
    return dataloaders, vocabularies


def run_data_statistics(
    config: ProjectConfig,
    train_examples: Sequence[ExampleDict],
    validation_examples: Sequence[ExampleDict],
    test_examples: Sequence[ExampleDict],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Print and persist descriptive dataset statistics."""

    corpus_overview = inspect_corpus(config)
    stats = dict(stats)
    stats["split_sizes"] = {
        "train": len(train_examples),
        "validation": len(validation_examples),
        "test": len(test_examples),
    }
    stats["corpus_overview"] = corpus_overview

    print("[data] PropBank total instances:", corpus_overview["total_instances"])
    print("[data] PropBank usable Treebank-backed instances:", stats["usable_propbank_instances"])
    print("[data] Sample roleset:", corpus_overview["sample_roleset"])
    print("[data] Sample predicate pointer:", corpus_overview["sample_predicate"])
    print("[data] Sample arguments:", corpus_overview["sample_arguments"][:5])
    print(
        "[data] Split sizes -> train:",
        len(train_examples),
        "validation:",
        len(validation_examples),
        "test:",
        len(test_examples),
    )
    print("[data] Sample QA pairs:")
    for sample in stats.get("sample_qa_pairs", [])[:5]:
        print("  context:", sample["context"])
        print("  question:", sample["question"])
        print("  answer:", sample["answer"])
        print("  role:", sample["role"])

    save_json(config.paths.results_dir / "data_statistics.json", stats)
    return stats
