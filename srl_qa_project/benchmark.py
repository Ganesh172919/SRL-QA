"""Benchmark, ablation, and reporting utilities for the hybrid SRL-QA upgrade."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import ProjectConfig
from evaluator import normalize_text
from hybrid_qa import HybridQASystem, load_challenge_suite
from qa_inference import InferenceEngine, simple_word_tokenize
from trainer import token_level_f1


def benchmark_results_dir(config: ProjectConfig) -> Path:
    """Return the directory that stores benchmark results."""

    path = config.paths.results_dir / "benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_test_examples(test_examples: Sequence[Dict[str, Any]], max_examples: int) -> List[Dict[str, Any]]:
    """Create a stable, question-type-aware benchmark subset."""

    if max_examples <= 0 or max_examples >= len(test_examples):
        return list(test_examples)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for example in test_examples:
        grouped.setdefault(example["question_type"], []).append(example)

    question_types = sorted(grouped)
    per_group = max(1, max_examples // max(len(question_types), 1))
    sampled: List[Dict[str, Any]] = []
    for question_type in question_types:
        sampled.extend(grouped[question_type][:per_group])

    if len(sampled) < max_examples:
        remaining = [example for example in test_examples if example not in sampled]
        sampled.extend(remaining[: max_examples - len(sampled)])
    return sampled[:max_examples]


def _baseline_predict(engine: InferenceEngine, context: str, question: str) -> Dict[str, Any]:
    """Run the classical baseline prediction path."""

    start_time = time.perf_counter()
    prediction = engine.infer(context, question)
    return {
        "answer": prediction.answer_text,
        "role": prediction.predicted_role,
        "confidence": float(prediction.confidence),
        "latency_ms": (time.perf_counter() - start_time) * 1000.0,
        "reasoning_summary": "Classical PropQA-Net answer without hybrid reranking.",
        "evidence_spans": [],
    }


def _evaluate_track(
    name: str,
    predictor: Any,
    examples: Sequence[Dict[str, Any]],
    source_key: str,
) -> List[Dict[str, Any]]:
    """Evaluate one prediction track over a sequence of examples."""

    records: List[Dict[str, Any]] = []
    for index, example in enumerate(examples):
        if source_key == "baseline":
            output = _baseline_predict(predictor, example["context"], example["question"])
            predicted_answer = output["answer"]
            predicted_role = output["role"]
        else:
            output = predictor.answer_question(example["context"], example["question"])
            predicted_answer = output["hybrid_answer"]
            predicted_role = output["role"]

        expected_answer = example.get("expected_answer", example.get("answer_text", ""))
        target_role = example.get("target_role", example.get("role", ""))
        predicted_tokens = simple_word_tokenize(predicted_answer)
        gold_tokens = simple_word_tokenize(expected_answer)
        records.append(
            {
                "example_id": example.get("id", example.get("example_id", f"{name}_{index:04d}")),
                "track": name,
                "context": example["context"],
                "question": example["question"],
                "question_type": example.get("question_type", "UNKNOWN"),
                "target_role": target_role,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "predicted_role": predicted_role,
                "confidence": float(output.get("confidence", 0.0)),
                "latency_ms": float(output.get("latency_ms", 0.0)),
                "exact_match": float(normalize_text(predicted_answer) == normalize_text(expected_answer)),
                "token_f1": token_level_f1(predicted_tokens, gold_tokens),
                "role_match": float(predicted_role == target_role),
                "reasoning_summary": output.get("reasoning_summary", ""),
                "evidence_spans": output.get("evidence_spans", []),
            }
        )
    return records


def _aggregate_records(records: Sequence[Dict[str, Any]], shared_metrics: Dict[str, Any], load_time_sec: float) -> Dict[str, Any]:
    """Aggregate benchmark records into summary metrics."""

    exact_match = statistics.mean(record["exact_match"] for record in records) if records else 0.0
    token_f1 = statistics.mean(record["token_f1"] for record in records) if records else 0.0
    role_accuracy = statistics.mean(record["role_match"] for record in records) if records else 0.0
    latency_ms = statistics.mean(record["latency_ms"] for record in records) if records else 0.0
    confidence = statistics.mean(record["confidence"] for record in records) if records else 0.0

    per_question_type: Dict[str, Dict[str, float]] = {}
    for question_type in sorted({record["question_type"] for record in records}):
        subset = [record for record in records if record["question_type"] == question_type]
        per_question_type[question_type] = {
            "exact_match": statistics.mean(item["exact_match"] for item in subset),
            "token_f1": statistics.mean(item["token_f1"] for item in subset),
            "count": float(len(subset)),
        }

    per_role: Dict[str, Dict[str, float]] = {}
    for role in sorted({record["target_role"] for record in records}):
        subset = [record for record in records if record["target_role"] == role]
        per_role[role] = {
            "role_accuracy": statistics.mean(item["role_match"] for item in subset),
            "token_f1": statistics.mean(item["token_f1"] for item in subset),
            "count": float(len(subset)),
        }

    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
        "role_accuracy": role_accuracy,
        "mean_latency_ms": latency_ms,
        "mean_confidence": confidence,
        "load_time_sec": load_time_sec,
        "count": len(records),
        "shared_srl_micro_f1": shared_metrics["srl_performance"]["micro_f1"],
        "shared_srl_macro_f1": shared_metrics["srl_performance"]["macro_f1"],
        "per_question_type": per_question_type,
        "per_role": per_role,
        "samples": list(records[:8]),
    }


def attach_benchmark_to_metrics(metrics: Dict[str, Any], benchmark_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge benchmark outputs into the evaluation metrics payload."""

    merged = dict(metrics)
    merged["hybrid_benchmark"] = benchmark_payload
    return merged


def _plot_ablation_summary(benchmark_payload: Dict[str, Any], output_path: Path) -> None:
    """Plot track-level EM and F1 across challenge and test subsets."""

    track_names = list(benchmark_payload["tracks"].keys())
    challenge_f1 = [benchmark_payload["tracks"][name]["challenge"]["token_f1"] for name in track_names]
    test_f1 = [benchmark_payload["tracks"][name]["test_subset"]["token_f1"] for name in track_names]
    positions = np.arange(len(track_names))

    plt.figure(figsize=(10, 5))
    plt.bar(positions - 0.2, challenge_f1, width=0.4, label="Challenge F1", color="#284B63")
    plt.bar(positions + 0.2, test_f1, width=0.4, label="Test-subset F1", color="#D9BF77")
    plt.xticks(positions, track_names, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Token F1")
    plt.title("Ablation Summary Across Hybrid Tracks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_latency_tradeoff(benchmark_payload: Dict[str, Any], output_path: Path) -> None:
    """Plot average latency against token-level F1."""

    plt.figure(figsize=(8, 5))
    for track_name, track_payload in benchmark_payload["tracks"].items():
        latency = track_payload["combined"]["mean_latency_ms"]
        f1 = track_payload["combined"]["token_f1"]
        plt.scatter(latency, f1, s=120, label=track_name)
        plt.text(latency + 0.2, f1, track_name, fontsize=9)
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Token F1")
    plt.title("Latency vs Accuracy Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_question_type_heatmap(benchmark_payload: Dict[str, Any], output_path: Path) -> None:
    """Plot F1 by question type across tracks."""

    question_types = sorted(
        {
            qtype
            for track_payload in benchmark_payload["tracks"].values()
            for qtype in track_payload["challenge"]["per_question_type"]
        }
    )
    track_names = list(benchmark_payload["tracks"].keys())
    matrix = np.array(
        [
            [
                benchmark_payload["tracks"][track_name]["challenge"]["per_question_type"].get(qtype, {}).get("token_f1", 0.0)
                for qtype in question_types
            ]
            for track_name in track_names
        ]
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=question_types, yticklabels=track_names)
    plt.title("Challenge-set F1 by Question Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_role_heatmap(benchmark_payload: Dict[str, Any], output_path: Path) -> None:
    """Plot role accuracy by target role across tracks."""

    roles = sorted(
        {
            role
            for track_payload in benchmark_payload["tracks"].values()
            for role in track_payload["challenge"]["per_role"]
        }
    )
    track_names = list(benchmark_payload["tracks"].keys())
    matrix = np.array(
        [
            [
                benchmark_payload["tracks"][track_name]["challenge"]["per_role"].get(role, {}).get("role_accuracy", 0.0)
                for role in roles
            ]
            for track_name in track_names
        ]
    )

    plt.figure(figsize=(10, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="rocket_r", xticklabels=roles, yticklabels=track_names)
    plt.title("Challenge-set Role Accuracy by Target Role")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_confidence_histogram(benchmark_payload: Dict[str, Any], output_path: Path) -> None:
    """Plot correct vs incorrect confidence values for the full hybrid track."""

    full_hybrid = benchmark_payload["tracks"]["full_hybrid"]["records"]["challenge"] + benchmark_payload["tracks"]["full_hybrid"]["records"]["test_subset"]
    correct = [record["confidence"] for record in full_hybrid if record["exact_match"] == 1.0]
    incorrect = [record["confidence"] for record in full_hybrid if record["exact_match"] == 0.0]

    plt.figure(figsize=(8, 5))
    plt.hist(correct, bins=10, alpha=0.65, label="Correct", color="#5C946E")
    plt.hist(incorrect, bins=10, alpha=0.65, label="Incorrect", color="#A44A3F")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Calibration Snapshot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_dataset_balance(data_stats: Dict[str, Any], output_path: Path) -> None:
    """Plot benchmark-relevant dataset distributions."""

    question_items = list(data_stats["qa_pairs_per_question_type"].items())
    role_items = sorted(data_stats["qa_pairs_per_argument_type"].items(), key=lambda item: item[1], reverse=True)[:10]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar([item[0] for item in question_items], [item[1] for item in question_items], color="#3C6E71")
    axes[0].set_title("Dataset Balance by Question Type")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar([item[0] for item in role_items], [item[1] for item in role_items], color="#D9BF77")
    axes[1].set_title("Top Argument Types in Dataset")
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_table(records: Sequence[Dict[str, Any]], title: str, output_path: Path, columns: Sequence[str]) -> None:
    """Render a small table as a PNG for reports and the website."""

    rows = [list(columns)]
    for record in records:
        rows.append([str(record.get(column, ""))[:48] for column in columns])

    figure, axis = plt.subplots(figsize=(14, max(3, len(rows) * 0.45)))
    axis.axis("off")
    table = axis.table(cellText=rows[1:], colLabels=rows[0], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    axis.set_title(title, fontsize=12, fontweight="bold", pad=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _plot_research_architecture(output_path: Path) -> None:
    """Create a complete architecture diagram for the upgraded system."""

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.axis("off")
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)

    boxes = [
        (0.5, 5.6, 2.2, 1.0, "Question + Context"),
        (3.2, 5.6, 2.2, 1.0, "PropQA-Net Baseline"),
        (5.9, 5.6, 2.2, 1.0, "Role Heuristics"),
        (8.6, 5.6, 2.2, 1.0, "Transformer QA"),
        (11.3, 5.6, 2.2, 1.0, "Semantic Reranker"),
        (2.0, 2.1, 2.8, 1.0, "Evidence Spans"),
        (5.5, 2.1, 2.8, 1.0, "Reasoning Summary"),
        (9.0, 2.1, 3.2, 1.0, "Streamlit Research App"),
    ]
    for x, y, width, height, label in boxes:
        axis.add_patch(plt.Rectangle((x, y), width, height, facecolor="#D9E2F3", edgecolor="#284B63", linewidth=1.5))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.7, 6.1), (3.2, 6.1)),
        ((5.4, 6.1), (5.9, 6.1)),
        ((8.1, 6.1), (8.6, 6.1)),
        ((10.8, 6.1), (11.3, 6.1)),
        ((12.4, 5.6), (10.7, 3.1)),
        ((7.0, 5.6), (6.9, 3.1)),
        ((4.3, 5.6), (3.4, 3.1)),
        ((6.9, 3.1), (9.0, 2.6)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4, color="#284B63"))

    axis.set_title("Research-Grade Hybrid SRL-QA Architecture", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def generate_benchmark_artifacts(config: ProjectConfig, data_stats: Dict[str, Any], benchmark_payload: Dict[str, Any]) -> Dict[str, str]:
    """Generate the new benchmark and reporting plots."""

    plots_dir = config.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifact_map = {
        "ablation_summary": plots_dir / "ablation_summary.png",
        "latency_accuracy_tradeoff": plots_dir / "latency_accuracy_tradeoff.png",
        "question_type_heatmap": plots_dir / "question_type_heatmap.png",
        "role_heatmap": plots_dir / "role_heatmap.png",
        "confidence_histogram": plots_dir / "confidence_histogram.png",
        "dataset_balance": plots_dir / "dataset_balance.png",
        "challenge_table": plots_dir / "challenge_table.png",
        "error_gallery": plots_dir / "error_gallery.png",
        "research_architecture": plots_dir / "research_architecture.png",
    }
    _plot_ablation_summary(benchmark_payload, artifact_map["ablation_summary"])
    _plot_latency_tradeoff(benchmark_payload, artifact_map["latency_accuracy_tradeoff"])
    _plot_question_type_heatmap(benchmark_payload, artifact_map["question_type_heatmap"])
    _plot_role_heatmap(benchmark_payload, artifact_map["role_heatmap"])
    _plot_confidence_histogram(benchmark_payload, artifact_map["confidence_histogram"])
    _plot_dataset_balance(data_stats, artifact_map["dataset_balance"])
    _plot_table(
        benchmark_payload["tracks"]["full_hybrid"]["records"]["challenge"][:10],
        "Challenge-set Predictions (Full Hybrid)",
        artifact_map["challenge_table"],
        columns=["question", "expected_answer", "predicted_answer", "predicted_role", "exact_match"],
    )
    errors = [record for record in benchmark_payload["tracks"]["full_hybrid"]["records"]["combined"] if record["exact_match"] == 0.0][:8]
    _plot_table(
        errors,
        "Error Gallery (Full Hybrid)",
        artifact_map["error_gallery"],
        columns=["question", "expected_answer", "predicted_answer", "target_role", "predicted_role"],
    )
    _plot_research_architecture(artifact_map["research_architecture"])
    return {name: str(path) for name, path in artifact_map.items()}


def run_benchmark(
    config: ProjectConfig,
    test_examples: Sequence[Dict[str, Any]],
    data_stats: Dict[str, Any],
    base_metrics: Dict[str, Any],
    max_examples: int = 160,
) -> Dict[str, Any]:
    """Run the four-track benchmark suite and save the results."""

    subset = sample_test_examples(test_examples, max_examples)
    challenge_examples = load_challenge_suite(config.paths.project_root)

    baseline_start = time.perf_counter()
    baseline_engine = InferenceEngine(config)
    baseline_load_time = time.perf_counter() - baseline_start

    heuristic_start = time.perf_counter()
    heuristic_system = HybridQASystem(config, use_transformer_qa=False, use_sentence_embeddings=False)
    heuristic_load_time = time.perf_counter() - heuristic_start

    transformer_start = time.perf_counter()
    transformer_system = HybridQASystem(config, use_transformer_qa=True, use_sentence_embeddings=False)
    transformer_system.external_models.ensure_qa_pipeline()
    transformer_load_time = time.perf_counter() - transformer_start

    hybrid_start = time.perf_counter()
    hybrid_system = HybridQASystem(config, use_transformer_qa=True, use_sentence_embeddings=True)
    hybrid_system.external_models.ensure_qa_pipeline()
    hybrid_system.external_models.ensure_embedder()
    hybrid_load_time = time.perf_counter() - hybrid_start

    tracks = {
        "classical_baseline": {
            "predictor": baseline_engine,
            "load_time_sec": baseline_load_time,
            "source_key": "baseline",
        },
        "heuristic_reranker": {
            "predictor": heuristic_system,
            "load_time_sec": heuristic_load_time,
            "source_key": "hybrid",
        },
        "transformer_qa_assist": {
            "predictor": transformer_system,
            "load_time_sec": transformer_load_time,
            "source_key": "hybrid",
        },
        "full_hybrid": {
            "predictor": hybrid_system,
            "load_time_sec": hybrid_load_time,
            "source_key": "hybrid",
        },
    }

    payload = {
        "metadata": {
            "test_subset_size": len(subset),
            "full_test_size": len(test_examples),
            "challenge_size": len(challenge_examples),
            "authoritative_repo_metrics": {
                "qa_exact_match": base_metrics["qa_performance"]["exact_match"],
                "qa_token_f1": base_metrics["qa_performance"]["token_f1"],
                "srl_micro_f1": base_metrics["srl_performance"]["micro_f1"],
                "srl_macro_f1": base_metrics["srl_performance"]["macro_f1"],
            },
        },
        "tracks": {},
    }

    for track_name, track_info in tracks.items():
        challenge_records = _evaluate_track(track_name, track_info["predictor"], challenge_examples, track_info["source_key"])
        test_records = _evaluate_track(track_name, track_info["predictor"], subset, track_info["source_key"])
        combined_records = challenge_records + test_records
        payload["tracks"][track_name] = {
            "challenge": _aggregate_records(challenge_records, base_metrics, track_info["load_time_sec"]),
            "test_subset": _aggregate_records(test_records, base_metrics, track_info["load_time_sec"]),
            "combined": _aggregate_records(combined_records, base_metrics, track_info["load_time_sec"]),
            "records": {
                "challenge": challenge_records,
                "test_subset": test_records,
                "combined": combined_records,
            },
        }

    payload["artifacts"] = generate_benchmark_artifacts(config, data_stats, payload)
    output_path = benchmark_results_dir(config) / "benchmark_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[benchmark] results written to {output_path}")
    return payload


def load_latest_benchmark(config: ProjectConfig) -> Dict[str, Any] | None:
    """Load the latest saved benchmark results if available."""

    output_path = benchmark_results_dir(config) / "benchmark_results.json"
    if not output_path.exists():
        return None
    return json.loads(output_path.read_text(encoding="utf-8"))
