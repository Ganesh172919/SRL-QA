from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qasrl_cpu.data import prepare_grouped_dataset
from qasrl_cpu.inference import predict_dataset
from qasrl_cpu.instashap import InstaShapExplainer
from qasrl_cpu.metrics import compute_dataset_metrics
from qasrl_cpu.modeling import load_trained_model
from qasrl_cpu.roles import normalize_role_mapping


def build_gold_token_set(record: dict) -> set[str]:
    tokens = set(record["predicate"].lower().split())
    for answers in normalize_role_mapping(record["roles"]).values():
        for answer in answers:
            tokens.update(answer.lower().split())
    return tokens


def compute_xai_suite(records: list[dict], predictions: list[str], model, tokenizer, limit: int) -> dict:
    explainer = InstaShapExplainer(model, tokenizer)
    plausibility_scores = []
    faithfulness_scores = []
    example_reports = []
    for record, prediction in list(zip(records, predictions))[:limit]:
        target_text = prediction or record["target_text"]
        explanation = explainer.explain(record["input_text"], target_text)
        gold_tokens = build_gold_token_set(record)
        plausibility = explainer.compute_plausibility(explanation, gold_tokens)
        faithfulness = max(0.0, min(1.0, explanation.confidence_drop))
        plausibility_scores.append(plausibility)
        faithfulness_scores.append(faithfulness)
        example_reports.append(
            {
                "id": record["id"],
                "sentence": record["sentence"],
                "predicate": record["predicate"],
                "plausibility": plausibility,
                "faithfulness": faithfulness,
                "top_tokens": sorted(
                    zip(explanation.tokens, explanation.scores),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:8],
            }
        )
    return {
        "plausibility": round(sum(plausibility_scores) / len(plausibility_scores), 4),
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "xai_score": round(
            (sum(plausibility_scores) / len(plausibility_scores) + sum(faithfulness_scores) / len(faithfulness_scores))
            / 2,
            4,
        ),
        "examples": example_reports,
    }


def write_markdown_report(report_path: Path, evaluation_report: dict) -> None:
    baseline = evaluation_report.get("baseline_metrics")
    tuned = evaluation_report["fine_tuned_metrics"]
    xai = evaluation_report["xai_metrics"]
    lines = [
        "# QA-SRL Fine-Tuning Evaluation",
        "",
        "## Main Metrics",
        "",
        "| Model | Token F1 | Exact Match | Role Coverage | ROUGE-L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if baseline:
        lines.append(
            f"| Zero-shot {evaluation_report['baseline_model_name']} | {baseline['token_f1']:.4f} | "
            f"{baseline['exact_match']:.4f} | {baseline['role_coverage']:.4f} | {baseline['rouge_l']:.4f} |"
        )
    lines.append(
        f"| Fine-tuned model | {tuned['token_f1']:.4f} | {tuned['exact_match']:.4f} | "
        f"{tuned['role_coverage']:.4f} | {tuned['rouge_l']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## XAI Metrics",
            "",
            f"- Plausibility: {xai['plausibility']:.4f}",
            f"- Faithfulness: {xai['faithfulness']:.4f}",
            f"- Combined XAI score: {xai['xai_score']:.4f}",
            "",
            "## Domain Token F1",
            "",
        ]
    )
    for domain, value in tuned["domain_token_f1"].items():
        lines.append(f"- {domain}: {value:.4f}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(args) -> dict:
    dataset = prepare_grouped_dataset(
        data_dir=args.data_dir,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    records = list(dataset[args.split])
    model, tokenizer, metadata = load_trained_model(args.model_dir)
    predictions = predict_dataset(
        model,
        tokenizer,
        records,
        description="Evaluating fine-tuned model",
        max_new_tokens=args.max_target_length,
        num_beams=args.num_beams,
    )
    fine_tuned_metrics = compute_dataset_metrics(records, predictions)
    xai_metrics = compute_xai_suite(records, predictions, model, tokenizer, limit=min(args.xai_limit, len(records)))

    explainer = InstaShapExplainer(model, tokenizer)
    explanation = explainer.explain(records[0]["input_text"], predictions[0] or records[0]["target_text"])
    fig = explainer.plot(explanation)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    chart_path = results_dir / "instashap_example.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    evaluation_report = {
        "model_metadata": metadata,
        "split": args.split,
        "fine_tuned_metrics": fine_tuned_metrics,
        "xai_metrics": xai_metrics,
        "instashap_example_path": str(chart_path),
    }

    baseline_model_name = args.baseline_model_name or metadata.get("base_model_name")
    if baseline_model_name:
        baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
        baseline_model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_name)
        baseline_predictions = predict_dataset(
            baseline_model,
            baseline_tokenizer,
            records,
            description="Evaluating zero-shot baseline",
            max_new_tokens=args.max_target_length,
            num_beams=args.num_beams,
            use_fallback=False,
        )
        evaluation_report["baseline_model_name"] = baseline_model_name
        evaluation_report["baseline_metrics"] = compute_dataset_metrics(records, baseline_predictions)

    json_path = results_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
    write_markdown_report(results_dir / "evaluation_summary.md", evaluation_report)
    return evaluation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned QA-SRL model and compute InstaShap metrics.")
    parser.add_argument("--model-dir", default=str(ROOT / "artifacts" / "flan_t5_small_lora"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--baseline-model-name", default="")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--xai-limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_evaluation(args)
    print(json.dumps(report["fine_tuned_metrics"], indent=2))


if __name__ == "__main__":
    main()
