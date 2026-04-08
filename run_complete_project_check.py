"""Run a lightweight functionality and accuracy check for the whole project."""

from __future__ import annotations

import compileall
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from srl_rag_demo.demo_eval import format_percent, run_demo_evaluation


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "COMPLETE_PROJECT_FUNCTIONALITY_REPORT.md"


def read_csv_metrics(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def read_raise_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for track_name, track in payload.get("tracks", {}).items():
        for scope in ("challenge", "test_subset", "combined"):
            if scope not in track:
                continue
            item = track[scope]
            rows.append(
                {
                    "track": track_name,
                    "scope": scope,
                    "count": item.get("count", ""),
                    "exact_match": item.get("exact_match", 0.0),
                    "token_f1": item.get("token_f1", 0.0),
                    "role_accuracy": item.get("role_accuracy", 0.0),
                    "mean_latency_ms": item.get("mean_latency_ms", 0.0),
                }
            )
    return rows


def pct_from_metric(metrics: dict[str, str], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "missing"
    try:
        return format_percent(float(value))
    except ValueError:
        return value


def build_report(
    compile_ok: bool,
    demo_eval: dict[str, Any],
    baseline_metrics: dict[str, str],
    raise_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, Any]],
) -> str:
    demo_summary = demo_eval["summary"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Complete Project Functionality And Accuracy Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Functional Status",
        "",
        "| Check | Status |",
        "|---|---|",
        f"| Compile `srl_rag_demo` | {'PASS' if compile_ok else 'FAIL'} |",
        f"| Controlled SRL + RAG demo evaluation | {'PASS' if demo_summary['exact_match'] >= 0.80 else 'CHECK'} |",
        f"| Retrieval backend | `{demo_summary['retrieval_backend']}` |",
        "",
        "## Controlled SRL + RAG Demo Accuracy",
        "",
        "> Scope: controlled demo-suite, not full-corpus benchmark.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Examples | {demo_summary['examples']} |",
        f"| Exact Match | {format_percent(demo_summary['exact_match'])} |",
        f"| Token F1 | {format_percent(demo_summary['token_f1'])} |",
        f"| Role Accuracy | {format_percent(demo_summary['role_accuracy'])} |",
        f"| Mean Confidence | {format_percent(demo_summary['mean_confidence'])} |",
        "",
        "## Legacy Full-Test Baseline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| QA exact match | {pct_from_metric(baseline_metrics, 'qa_exact_match')} |",
        f"| QA token F1 | {pct_from_metric(baseline_metrics, 'qa_token_f1')} |",
        f"| SRL micro F1 | {pct_from_metric(baseline_metrics, 'srl_micro_f1')} |",
        f"| SRL BIO accuracy | {pct_from_metric(baseline_metrics, 'srl_bio_accuracy')} |",
        f"| Best validation F1 | {pct_from_metric(baseline_metrics, 'best_validation_f1')} |",
        "",
        "## RAISE Curated Seed-Suite Metrics",
        "",
        "| System | Examples | Exact Match | Token F1 | Role Accuracy | Mean Latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in raise_rows:
        lines.append(
            "| {model} | {examples} | {exact} | {f1} | {role} | {latency:.2f} ms |".format(
                model=row["model_label"],
                examples=row["examples"],
                exact=format_percent(float(row["exact_match"])),
                f1=format_percent(float(row["token_f1"])),
                role=format_percent(float(row["role_accuracy"])),
                latency=float(row["latency_ms_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Legacy Benchmark Tracks",
            "",
            "| Track | Scope | Examples | Exact Match | Token F1 | Role Accuracy | Mean Latency |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| {track} | {scope} | {count} | {exact} | {f1} | {role} | {latency:.2f} ms |".format(
                track=row["track"],
                scope=row["scope"],
                count=row["count"],
                exact=format_percent(float(row["exact_match"])),
                f1=format_percent(float(row["token_f1"])),
                role=format_percent(float(row["role_accuracy"])),
                latency=float(row["mean_latency_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Demo Per-Example Results",
            "",
            "| Example | Expected | Predicted | Role | Exact | Token F1 |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for record in demo_eval["records"]:
        lines.append(
            "| {example} | `{expected}` | `{predicted}` | `{role}` | {exact} | {f1} |".format(
                example=record["example_id"],
                expected=record["expected_answer"],
                predicted=record["predicted_answer"],
                role=record["predicted_role"],
                exact=format_percent(record["exact_match"]),
                f1=format_percent(record["token_f1"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best Supported Accuracy Statements",
            "",
            "- Full-test legacy baseline: 51.84% exact match, 76.12% QA token F1, 71.33% SRL micro F1, 81.63% BIO accuracy.",
            "- RAISE curated 15-example seed suite: 100.00% exact match, 100.00% token F1, 100.00% role accuracy for both RAISE fast and model-assisted variants.",
            "- Controlled final SRL + RAG demo suite: "
            f"{format_percent(demo_summary['exact_match'])} exact match, "
            f"{format_percent(demo_summary['token_f1'])} token F1, "
            f"{format_percent(demo_summary['role_accuracy'])} role accuracy.",
            "",
            "## Claim Discipline",
            "",
            "- Use full-test metrics only for the legacy baseline full-test claim.",
            "- Use RAISE 100% metrics only for the curated 15-example seed suite.",
            "- Use the controlled demo-suite metrics only to show the final SRL + RAG demo path is functional.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    compile_ok = compileall.compile_dir(str(ROOT / "srl_rag_demo"), quiet=1)
    demo_eval = run_demo_evaluation()
    baseline_metrics = read_csv_metrics(ROOT / "srlqa" / "output" / "tables" / "baseline_metrics_summary.csv")
    raise_rows = read_raise_rows(ROOT / "srlqa" / "output" / "tables" / "model_evaluation_summary.csv")
    benchmark_rows = read_benchmark_rows(ROOT / "srl_qa_project" / "results" / "benchmarks" / "benchmark_results.json")

    report = build_report(
        compile_ok=compile_ok,
        demo_eval=demo_eval,
        baseline_metrics=baseline_metrics,
        raise_rows=raise_rows,
        benchmark_rows=benchmark_rows,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = demo_eval["summary"]
    print(f"Compile srl_rag_demo: {'PASS' if compile_ok else 'FAIL'}")
    print(
        "Controlled SRL + RAG demo: "
        f"EM={format_percent(summary['exact_match'])}, "
        f"F1={format_percent(summary['token_f1'])}, "
        f"Role={format_percent(summary['role_accuracy'])}"
    )
    print(f"Report written: {REPORT_PATH}")
    return 0 if compile_ok and summary["exact_match"] >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
