"""Command line entrypoint for RAISE-SRL-QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import get_config


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cmd_show_config(_: argparse.Namespace) -> None:
    _print_json(get_config().to_dict())


def cmd_search_assets(args: argparse.Namespace) -> None:
    from .data.dataset_library import search_huggingface_assets

    _print_json(search_huggingface_assets(args.query, limit=args.limit))


def cmd_download(args: argparse.Namespace) -> None:
    from .data.dataset_library import (
        download_dataset_snapshot,
        download_model_snapshot,
        preview_dataset,
    )

    config = get_config()
    if args.dataset:
        config.dataset.name = args.dataset
    if args.model:
        config.model.encoder_name = args.model

    payload: dict[str, Any] = {}
    payload["model_snapshot"] = download_model_snapshot(config.model.encoder_name, config)
    payload["dataset_snapshot"] = download_dataset_snapshot(config.dataset.name, config)
    try:
        payload["dataset_preview"] = preview_dataset(config, max_examples=args.max_examples)
    except Exception as exc:  # pragma: no cover - depends on optional datasets package
        payload["dataset_preview_error"] = f"{type(exc).__name__}: {exc}"
    _print_json(payload)


def cmd_preview_data(args: argparse.Namespace) -> None:
    from .data.dataset_library import preview_dataset

    config = get_config()
    if args.dataset:
        config.dataset.name = args.dataset
    config.dataset.split = args.split
    _print_json(preview_dataset(config, max_examples=args.max_examples))


def cmd_build_frame_index(args: argparse.Namespace) -> None:
    from .retrieval.propbank_index import FrameIndex

    config = get_config()
    source = Path(args.frames_dir) if args.frames_dir else config.paths.existing_propbank_frames_dir
    index = FrameIndex.from_directory(source)
    index.save(config.paths.frame_store_path)
    _print_json(
        {
            "source": source,
            "output": config.paths.frame_store_path,
            "frame_count": len(index.frames),
        }
    )


def cmd_eval_file(args: argparse.Namespace) -> None:
    from .evaluation.offline_eval import evaluate_prediction_file

    _print_json(evaluate_prediction_file(Path(args.predictions)))


def cmd_ask(args: argparse.Namespace) -> None:
    from .pipeline import RaiseSrlQaSystem

    system = RaiseSrlQaSystem(get_config(), use_teacher_qa=not args.no_model)
    result = system.answer(
        context=args.context,
        question=args.question,
        predicate=args.predicate or "",
        role=args.role or "ARG1",
        expected_answer=args.expected_answer,
        max_corrections=args.max_corrections,
    )
    _print_json(result)


def cmd_demo(args: argparse.Namespace) -> None:
    import json
    from .evaluation.span_metrics import exact_match, token_f1
    from .pipeline import RaiseSrlQaSystem

    config = get_config()
    examples = json.loads(config.paths.challenge_suite_path.read_text(encoding="utf-8"))
    examples = examples[: args.max_examples]
    system = RaiseSrlQaSystem(config, use_teacher_qa=not args.no_model)
    records = []
    for example in examples:
        result = system.answer(
            context=example["context"],
            question=example["question"],
            role=example.get("target_role", "ARG1"),
            expected_answer=example.get("expected_answer"),
            max_corrections=args.max_corrections,
        )
        records.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected_answer": example["expected_answer"],
                "predicted_answer": result["answer"],
                "exact_match": exact_match(result["answer"], example["expected_answer"]),
                "token_f1": token_f1(result["answer"], example["expected_answer"]),
                "role": result["role"],
                "confidence": result["confidence"],
                "correction_history": result["correction_history"],
            }
        )
    summary = {
        "count": len(records),
        "exact_match": sum(record["exact_match"] for record in records) / max(len(records), 1),
        "token_f1": sum(record["token_f1"] for record in records) / max(len(records), 1),
    }
    _print_json({"summary": summary, "records": records})


def cmd_chat(args: argparse.Namespace) -> None:
    from .pipeline import RaiseSrlQaSystem

    system = RaiseSrlQaSystem(get_config(), use_teacher_qa=not args.no_model)
    context = args.context or input("Context: ").strip()
    while True:
        question = input("Question (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = system.answer(
            context=context,
            question=question,
            max_corrections=args.max_corrections,
        )
        print(f"Answer: {result['answer']}")
        print(f"Role: {result['role']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Reasoning: {result['reasoning']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAISE-SRL-QA runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config", help="Print active config")
    show_config.set_defaults(func=cmd_show_config)

    search_assets = subparsers.add_parser("search-assets", help="Search Hugging Face")
    search_assets.add_argument("--query", default="qasrl")
    search_assets.add_argument("--limit", type=int, default=8)
    search_assets.set_defaults(func=cmd_search_assets)

    download = subparsers.add_parser("download", help="Download model/dataset snapshots")
    download.add_argument("--dataset", default=None)
    download.add_argument("--model", default=None)
    download.add_argument("--max-examples", type=int, default=3)
    download.set_defaults(func=cmd_download)

    preview_data = subparsers.add_parser("preview-data", help="Load dataset through datasets")
    preview_data.add_argument("--dataset", default=None)
    preview_data.add_argument("--split", default="train")
    preview_data.add_argument("--max-examples", type=int, default=5)
    preview_data.set_defaults(func=cmd_preview_data)

    frame_index = subparsers.add_parser("build-frame-index", help="Index PropBank frames")
    frame_index.add_argument("--frames-dir", default=None)
    frame_index.set_defaults(func=cmd_build_frame_index)

    eval_file = subparsers.add_parser("eval-file", help="Evaluate a JSONL/JSON prediction file")
    eval_file.add_argument("predictions")
    eval_file.set_defaults(func=cmd_eval_file)

    ask = subparsers.add_parser("ask", help="Answer one context/question pair")
    ask.add_argument("--context", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--predicate", default="")
    ask.add_argument("--role", default="")
    ask.add_argument("--expected-answer", default=None, help="Optional gold answer for recursive correction demos")
    ask.add_argument("--max-corrections", type=int, default=4)
    ask.add_argument("--no-model", action="store_true", help="Use only deterministic SRL heuristics")
    ask.set_defaults(func=cmd_ask)

    demo = subparsers.add_parser("demo", help="Run challenge-suite question answering")
    demo.add_argument("--max-examples", type=int, default=8)
    demo.add_argument("--max-corrections", type=int, default=4)
    demo.add_argument("--no-model", action="store_true", help="Use only deterministic SRL heuristics")
    demo.set_defaults(func=cmd_demo)

    chat = subparsers.add_parser("chat", help="Ask multiple questions against one context")
    chat.add_argument("--context", default=None)
    chat.add_argument("--max-corrections", type=int, default=4)
    chat.add_argument("--no-model", action="store_true", help="Use only deterministic SRL heuristics")
    chat.set_defaults(func=cmd_chat)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
