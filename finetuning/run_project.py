from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import build_parser as build_train_parser
from train import train_model
from evaluate import build_parser as build_eval_parser
from evaluate import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full QA-SRL fine-tuning project.")
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--xai-limit", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    train_args = build_train_parser().parse_args([])
    train_args.train_limit = args.train_limit
    train_args.validation_limit = args.validation_limit
    train_args.test_limit = args.test_limit
    train_args.epochs = args.epochs
    summary = train_model(train_args)

    eval_args = build_eval_parser().parse_args([])
    eval_args.train_limit = args.train_limit
    eval_args.validation_limit = args.validation_limit
    eval_args.test_limit = args.test_limit
    eval_args.xai_limit = args.xai_limit
    report = run_evaluation(eval_args)

    merged = {
        "training_summary_path": str(Path(train_args.results_dir) / "training_summary.json"),
        "evaluation_report_path": str(Path(eval_args.results_dir) / "evaluation_report.json"),
        "fine_tuned_metrics": report["fine_tuned_metrics"],
        "xai_metrics": report["xai_metrics"],
        "zero_shot_baseline": summary["zero_shot_baseline"],
    }
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
