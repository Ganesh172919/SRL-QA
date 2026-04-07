"""Master runner for the PropBank SRL-QA project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Dict, Tuple

import torch

from benchmark import attach_benchmark_to_metrics, load_latest_benchmark, run_benchmark
from config import ProjectConfig, get_config
from data_loader import build_dataloaders, load_or_build_splits, run_data_statistics
from evaluator import evaluate_model
from hybrid_qa import HybridQASystem
from pdf_generator import generate_all_pdfs
from qa_inference import ask_question, run_demo, run_interactive_session
from trainer import train_model


def configure_runtime(config: ProjectConfig) -> ProjectConfig:
    """Select an execution device automatically."""

    config.runtime.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def prepare_data(config: ProjectConfig) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load or rebuild the PropBank-derived dataset and dataloaders."""

    train_split, validation_split, test_split, stats = load_or_build_splits(config)
    stats = run_data_statistics(config, train_split, validation_split, test_split, stats)
    dataloaders, vocabularies = build_dataloaders(
        train_split,
        validation_split,
        test_split,
        config,
    )
    return (
        {"train": train_split, "validation": validation_split, "test": test_split},
        dataloaders,
        vocabularies,
        stats,
        {
            "train": len(train_split),
            "validation": len(validation_split),
            "test": len(test_split),
        },
    )


def validate_final_outputs(config: ProjectConfig) -> Dict[str, int]:
    """Validate the required final deliverables and return file sizes."""

    required_files = [
        config.paths.outputs_dir / "survey.pdf",
        config.paths.outputs_dir / "implementation_code.py",
        config.paths.outputs_dir / "analysis.pdf",
        config.paths.outputs_dir / "innovation.pdf",
        config.paths.outputs_dir / "research_paper.pdf",
    ]
    sizes: Dict[str, int] = {}
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required deliverable missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Required deliverable is empty: {path}")
        sizes[path.name] = size
    return sizes


def load_metrics_from_disk(config: ProjectConfig) -> Dict[str, Any]:
    """Load saved evaluation metrics from disk."""

    if not config.paths.metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {config.paths.metrics_path}")
    return json.loads(config.paths.metrics_path.read_text(encoding="utf-8"))


def run_hybrid_interactive_session(config: ProjectConfig) -> None:
    """Interactive shell for the hybrid question-answering engine."""

    system = HybridQASystem(config, use_transformer_qa=True, use_sentence_embeddings=True)
    print("[ask] hybrid interactive session")
    print("[ask] type 'quit' at any prompt to exit")
    while True:
        context = input("\n[ask] context: ").strip()
        if context.lower() in {"quit", "exit"}:
            print("[ask] session ended")
            break
        if not context:
            print("[ask] please enter a non-empty context")
            continue

        question = input("[ask] question: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("[ask] session ended")
            break
        if not question:
            print("[ask] please enter a non-empty question")
            continue

        result = system.answer_question(context, question)
        print("[ask] hybrid answer:", result["hybrid_answer"])
        print("[ask] role:", result["role"])
        print("[ask] confidence:", f"{result['confidence']:.4f}")
        print("[ask] reasoning:", result["reasoning_summary"])


def launch_streamlit_app(config: ProjectConfig, port: int) -> None:
    """Launch the Streamlit website."""

    app_path = config.paths.project_root / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    print("[app] launching Streamlit app")
    print("[app] url: http://localhost:" + str(port))
    subprocess.run(command, check=True, cwd=str(config.paths.project_root))


def print_hybrid_result(result: Dict[str, Any]) -> None:
    """Pretty-print a hybrid prediction to the terminal."""

    print("[ask] context:", result["context"])
    print("[ask] question:", result["question"])
    print("[ask] hybrid answer:", result["hybrid_answer"])
    print("[ask] confidence:", f"{result['confidence']:.4f}")
    print("[ask] predicted role:", result["role"])
    print("[ask] baseline answer:", result["baseline_answer"])
    print("[ask] baseline role:", result["baseline_role"])
    print("[ask] reasoning:", result["reasoning_summary"])


def main() -> None:
    """Run the requested project mode."""

    parser = argparse.ArgumentParser(description="PropQA-Net project runner")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "infer", "ask", "full", "app", "benchmark", "report"],
        default="full",
        help="Execution mode.",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Custom context sentence or paragraph for question answering.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Custom natural-language question for question answering.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive question-answering session.",
    )
    parser.add_argument(
        "--engine",
        choices=["hybrid", "baseline"],
        default="hybrid",
        help="Inference engine to use for ask mode.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=160,
        help="Maximum number of test examples to include in benchmark mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit website.",
    )
    args = parser.parse_args()

    config = configure_runtime(get_config())
    metrics: Dict[str, Any] | None = None
    benchmark_payload: Dict[str, Any] | None = None
    custom_question_requested = args.mode == "ask" or args.interactive or args.context is not None or args.question is not None

    if (args.context is None) != (args.question is None):
        parser.error("--context and --question must be provided together.")
    if args.mode in {"train", "eval", "full", "benchmark", "report", "app"} and (args.interactive or args.context is not None or args.question is not None):
        parser.error("Custom question arguments are only supported with --mode ask or --mode infer.")

    splits = dataloaders = vocabularies = stats = split_sizes = None
    if args.mode in {"train", "eval", "full", "benchmark"}:
        splits, dataloaders, vocabularies, stats, split_sizes = prepare_data(config)

    if args.mode in {"train", "full"}:
        _, training_summary = train_model(
            dataloaders["train"],
            dataloaders["validation"],
            vocabularies,
            config,
        )
        print("[main] training summary")
        print(json.dumps(training_summary, indent=2))

    if args.mode in {"eval", "full"}:
        metrics = evaluate_model(dataloaders["test"], config)
        print("[main] evaluation complete")
        print(json.dumps(metrics["qa_performance"], indent=2))

    if args.mode == "app":
        launch_streamlit_app(config, args.port)
        return

    if args.mode == "benchmark":
        metrics = metrics or load_metrics_from_disk(config)
        benchmark_payload = run_benchmark(config, splits["test"], stats, metrics, max_examples=args.max_examples)
        print("[benchmark] combined summary")
        print(json.dumps({name: payload["combined"] for name, payload in benchmark_payload["tracks"].items()}, indent=2))
        return

    if args.mode == "report":
        stats = json.loads((config.paths.results_dir / "data_statistics.json").read_text(encoding="utf-8"))
        metrics = load_metrics_from_disk(config)
        benchmark_payload = load_latest_benchmark(config)
        if benchmark_payload is not None:
            metrics = attach_benchmark_to_metrics(metrics, benchmark_payload)
        page_counts = generate_all_pdfs(config, stats, metrics)
        print("[report] generated PDFs")
        print(json.dumps(page_counts, indent=2))
        print(json.dumps(validate_final_outputs(config), indent=2))
        return

    if args.mode == "ask" or (args.mode == "infer" and custom_question_requested):
        if args.interactive or (args.context is None and args.question is None):
            if args.engine == "baseline":
                run_interactive_session(config)
            else:
                run_hybrid_interactive_session(config)
        else:
            if args.engine == "baseline":
                result = ask_question(config, args.context, args.question)
                print("[ask] context:", result["context"])
                print("[ask] question:", result["question"])
                print("[ask] predicted answer:", result["predicted_answer"])
                print("[ask] confidence:", f"{result['confidence']:.4f}")
                print("[ask] predicted role:", result["predicted_role"])
            else:
                result = HybridQASystem(config, use_transformer_qa=True, use_sentence_embeddings=True).answer_question(args.context, args.question)
                print_hybrid_result(result)
        return

    if args.mode in {"infer", "full"}:
        demo_results = run_demo(config)
        (config.paths.results_dir / "inference_demo.json").write_text(
            json.dumps(demo_results, indent=2),
            encoding="utf-8",
        )

    if args.mode == "full":
        metrics = metrics or load_metrics_from_disk(config)
        benchmark_payload = run_benchmark(config, splits["test"], stats, metrics, max_examples=args.max_examples)
        metrics = attach_benchmark_to_metrics(metrics, benchmark_payload)
        page_counts = generate_all_pdfs(config, stats, metrics)
        print("[main] generated PDFs")
        for filename, page_count in page_counts.items():
            print(f"[ok] {filename} generated successfully - {page_count} pages")

        output_sizes = validate_final_outputs(config)
        print("[main] final deliverable sizes (bytes)")
        print(json.dumps(output_sizes, indent=2))


if __name__ == "__main__":
    main()
