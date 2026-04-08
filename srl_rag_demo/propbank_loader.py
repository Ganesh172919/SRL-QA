"""Load SRL-structured documents from local NLTK PropBank data."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

try:
    from .config import DemoConfig
    from .data_models import SRLArgument, SRLDocument
    from .frame_store import FrameStore, light_lemma
except ImportError:  # pragma: no cover - supports `streamlit run srl_rag_demo/app.py`
    from config import DemoConfig
    from data_models import SRLArgument, SRLDocument
    from frame_store import FrameStore, light_lemma


DETOKENIZER = TreebankWordDetokenizer()


def ensure_nltk_data(config: DemoConfig) -> None:
    nltk_path = str(config.nltk_data_dir)
    if nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_path)


def simple_word_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%]+(?:[-'][A-Za-z0-9$%]+)*|[^\w\s]", text)


def detokenize(tokens: Sequence[str]) -> str:
    return DETOKENIZER.detokenize(list(tokens)).strip()


def split_contiguous(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
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


def flatten_pointer_pieces(pointer: Any) -> list[Any]:
    pieces = getattr(pointer, "pieces", None)
    if not pieces:
        return [pointer]
    flattened: list[Any] = []
    for piece in pieces:
        flattened.extend(flatten_pointer_pieces(piece))
    return flattened


def build_visible_token_view(tree: nltk.Tree) -> tuple[list[str], list[str], dict[int, int], dict[int, tuple[int, ...]]]:
    tokens: list[str] = []
    pos_tags: list[str] = []
    original_to_visible: dict[int, int] = {}
    leaf_positions: dict[int, tuple[int, ...]] = {}
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


def visible_indices_for_pointer(
    tree: nltk.Tree,
    pointer: Any,
    original_to_visible: dict[int, int],
    leaf_positions: dict[int, tuple[int, ...]],
) -> list[int]:
    indices: set[int] = set()
    for piece in flatten_pointer_pieces(pointer):
        try:
            tree_position = piece.treepos(tree)
        except Exception:
            continue
        for leaf_index, leaf_position in leaf_positions.items():
            if leaf_position[: len(tree_position)] == tree_position and leaf_index in original_to_visible:
                indices.add(original_to_visible[leaf_index])
    return sorted(indices)


def roleset_metadata(propbank_corpus: Any, roleset_id: str) -> dict[str, Any]:
    try:
        roleset_xml = propbank_corpus.roleset(roleset_id)
    except Exception:
        return {
            "roleset_id": roleset_id,
            "roleset_name": roleset_id,
            "role_descriptions": {},
        }
    descriptions: dict[str, str] = {}
    roles_node = roleset_xml.find("roles")
    if roles_node is not None:
        for role_node in roles_node.findall("role"):
            number = role_node.attrib.get("n", "").strip()
            if number:
                descriptions[f"ARG{number}"] = role_node.attrib.get("descr", "").strip()
    return {
        "roleset_id": roleset_id,
        "roleset_name": roleset_xml.attrib.get("name", ""),
        "role_descriptions": descriptions,
    }


def inspect_corpus(config: DemoConfig) -> dict[str, Any]:
    ensure_nltk_data(config)
    from nltk.corpus import propbank, treebank

    instances = propbank.instances()
    treebank_fileids = set(treebank.fileids())
    usable = sum(1 for instance in instances if instance.fileid in treebank_fileids)
    sample = instances[0] if instances else None
    return {
        "nltk_data_dir": str(config.nltk_data_dir),
        "total_instances": len(instances),
        "treebank_file_count": len(treebank_fileids),
        "usable_treebank_backed": usable,
        "sample_roleset": getattr(sample, "roleset", ""),
        "sample_fileid": getattr(sample, "fileid", ""),
    }


def _argument_from_span(
    tokens: Sequence[str],
    role: str,
    spans: list[tuple[int, int]],
    description: str,
) -> SRLArgument:
    text = " ; ".join(detokenize(tokens[start : end + 1]) for start, end in spans)
    first_start, first_end = spans[0]
    return SRLArgument(
        role=role,
        text=text,
        description=description,
        start_token=first_start,
        end_token=first_end,
        is_contiguous=len(spans) == 1,
        source="propbank",
    )


def build_propbank_documents(
    config: DemoConfig,
    frame_store: FrameStore,
    limit: int = 300,
) -> tuple[list[SRLDocument], dict[str, Any]]:
    """Build SRL documents directly from NLTK PropBank/Treebank."""

    ensure_nltk_data(config)
    from nltk.corpus import propbank, treebank

    corpus_stats = inspect_corpus(config)
    instances = propbank.instances()
    treebank_fileids = set(treebank.fileids())
    metadata_cache: dict[str, dict[str, Any]] = {}
    documents: list[SRLDocument] = []
    role_counts: Counter[str] = Counter()
    skipped_without_tree = 0
    skipped_without_predicate = 0
    skipped_without_arguments = 0

    for instance in instances:
        if len(documents) >= limit:
            break
        if instance.fileid not in treebank_fileids or instance.tree is None:
            skipped_without_tree += 1
            continue
        tree = instance.tree
        tokens, _, original_to_visible, leaf_positions = build_visible_token_view(tree)
        if not tokens:
            skipped_without_tree += 1
            continue
        predicate_indices = visible_indices_for_pointer(
            tree=tree,
            pointer=instance.predicate,
            original_to_visible=original_to_visible,
            leaf_positions=leaf_positions,
        )
        if not predicate_indices:
            skipped_without_predicate += 1
            continue

        if instance.roleset not in metadata_cache:
            metadata_cache[instance.roleset] = roleset_metadata(propbank, instance.roleset)
        metadata = metadata_cache[instance.roleset]

        arguments: list[SRLArgument] = []
        for pointer, label in instance.arguments:
            if not str(label).startswith("ARG"):
                continue
            visible_indices = visible_indices_for_pointer(tree, pointer, original_to_visible, leaf_positions)
            spans = split_contiguous(visible_indices)
            if not spans:
                continue
            role_description = metadata["role_descriptions"].get(label, "")
            argument = _argument_from_span(tokens, label, spans, role_description)
            arguments.append(argument)
            role_counts[label] += 1

        if not arguments:
            skipped_without_arguments += 1
            continue

        predicate_text = detokenize([tokens[index] for index in predicate_indices])
        roleset_id = metadata["roleset_id"]
        frame_hint = frame_store.hint_for(predicate=predicate_text or instance.baseform, roleset_id=roleset_id)
        documents.append(
            SRLDocument(
                doc_id=f"propbank:{instance.fileid}:{instance.sentnum}:{instance.predid}:{roleset_id}",
                source="propbank",
                context=detokenize(tokens),
                tokens=tokens,
                predicate=predicate_text,
                predicate_lemma=instance.baseform or light_lemma(predicate_text),
                predicate_indices=predicate_indices,
                roleset_id=roleset_id,
                roleset_name=metadata.get("roleset_name", ""),
                frame_hint=frame_hint,
                arguments=arguments,
            )
        )

    stats = {
        **corpus_stats,
        "indexed_documents": len(documents),
        "indexed_arguments": int(sum(role_counts.values())),
        "role_distribution": dict(sorted(role_counts.items())),
        "skipped_without_tree": skipped_without_tree,
        "skipped_without_predicate": skipped_without_predicate,
        "skipped_without_arguments": skipped_without_arguments,
    }
    return documents, stats


def _cache_path(config: DemoConfig, limit: int) -> Path:
    return config.propbank_cache_dir / f"propbank_docs_{int(limit)}.json"


def load_or_build_propbank_documents(
    config: DemoConfig,
    frame_store: FrameStore,
    limit: int = 300,
    use_cache: bool = True,
) -> tuple[list[SRLDocument], dict[str, Any]]:
    cache_path = _cache_path(config, limit)
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            [SRLDocument.from_dict(item) for item in payload.get("documents", [])],
            dict(payload.get("stats", {})),
        )
    documents, stats = build_propbank_documents(config, frame_store, limit=limit)
    if use_cache:
        cache_path.write_text(
            json.dumps(
                {
                    "stats": stats,
                    "documents": [document.to_dict() for document in documents],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return documents, stats
