from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from presentation.builder import RESULTS_DIR, build_presentation_context


PALETTE = {
    "navy": "#12324a",
    "teal": "#0f766e",
    "gold": "#c68a2d",
    "slate": "#4b5563",
    "ink": "#1f2937",
    "green": "#2e8b57",
    "red": "#b04747",
    "cream": "#f7f3eb",
    "sky": "#d9edf7",
    "violet": "#6d28d9",
}


def _figure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=2, facecolor=face, edgecolor=edge)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, color=PALETTE["ink"], wrap=True)


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, linewidth=2, color=PALETTE["slate"]))


def _plot_research_timeline(ctx, path: Path) -> None:
    points = [
        (2015, "He et al."),
        (2018, "FitzGerald et al."),
        (2020, "Roit et al."),
        (2020.4, "Klein et al."),
        (2025, "InstaSHAP"),
        (2026, "Cross-lingual"),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.hlines(0, 2014.8, 2026.4, color=PALETTE["navy"], linewidth=3)
    for i, (year, label) in enumerate(points):
        color = [PALETTE["navy"], PALETTE["teal"], PALETTE["gold"], PALETTE["slate"], PALETTE["violet"], PALETTE["green"]][i]
        ax.scatter(year, 0, s=220, color=color, zorder=4)
        ax.text(year, 0.2 if i % 2 == 0 else -0.23, f"{int(year)}\n{label}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(2014.7, 2026.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(range(2015, 2027))
    ax.set_title("QA-SRL and Explanation-Aware Milestones (2015-2026)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Year")
    fig.tight_layout()
    _save(fig, path)


def _draw_system_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_box(ax, 0.04, 0.58, 0.2, 0.22, "Input\nSentence + Predicate", PALETTE["cream"], PALETTE["navy"])
    _draw_box(ax, 0.29, 0.58, 0.2, 0.22, "Prompt Builder\nStructured schema", "#e8f5f3", PALETTE["teal"])
    _draw_box(ax, 0.54, 0.58, 0.2, 0.22, "Fine-tuned Model\nflan-t5-small + LoRA", "#eef4fb", PALETTE["navy"])
    _draw_box(ax, 0.79, 0.58, 0.17, 0.22, "Role Output\nQA rendering", "#fff5e5", PALETTE["gold"])
    _draw_box(ax, 0.33, 0.2, 0.2, 0.22, "Cleanup + Snapping", "#f8eef0", PALETTE["red"])
    _draw_box(ax, 0.58, 0.2, 0.2, 0.22, "Fallback + XAI", "#eef0f4", PALETTE["slate"])
    _draw_arrow(ax, 0.24, 0.69, 0.29, 0.69)
    _draw_arrow(ax, 0.49, 0.69, 0.54, 0.69)
    _draw_arrow(ax, 0.74, 0.69, 0.79, 0.69)
    _draw_arrow(ax, 0.64, 0.58, 0.43, 0.42)
    _draw_arrow(ax, 0.53, 0.31, 0.58, 0.31)
    ax.text(0.5, 0.92, "QA-SRL System Overview", ha="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _draw_cleanup_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        "Raw generation",
        "Parse role lines",
        "Refine role mapping",
        "Answer span snapping",
        "Fallback recovery",
    ]
    x_positions = [0.03, 0.22, 0.41, 0.60, 0.79]
    for i, (x, step) in enumerate(zip(x_positions, steps)):
        _draw_box(ax, x, 0.36, 0.16, 0.28, step, [PALETTE["cream"], "#e8f5f3", "#eef4fb", "#fff5e5", "#f8eef0"][i], [PALETTE["gold"], PALETTE["teal"], PALETTE["navy"], PALETTE["gold"], PALETTE["red"]][i])
        if i < len(x_positions) - 1:
            _draw_arrow(ax, x + 0.16, 0.5, x_positions[i + 1], 0.5)
    ax.text(0.5, 0.83, "Inference Cleanup and Robustness Path", ha="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _draw_evaluation_stack(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_box(ax, 0.07, 0.58, 0.26, 0.23, "Task Metrics\nToken F1, Exact Match\nRole Coverage, ROUGE-L", "#eef4fb", PALETTE["navy"])
    _draw_box(ax, 0.37, 0.58, 0.26, 0.23, "Per-example Ledger\nError patterns\nSuccess/failure slices", "#e8f5f3", PALETTE["teal"])
    _draw_box(ax, 0.67, 0.58, 0.26, 0.23, "XAI Metrics\nPlausibility\nFaithfulness", "#fff5e5", PALETTE["gold"])
    _draw_box(ax, 0.22, 0.2, 0.26, 0.22, "Saved Reports\nJSON + Markdown", PALETTE["cream"], PALETTE["slate"])
    _draw_box(ax, 0.52, 0.2, 0.26, 0.22, "InstaShap Plot\nReusable in slides", "#f8eef0", PALETTE["red"])
    _draw_arrow(ax, 0.20, 0.58, 0.31, 0.42)
    _draw_arrow(ax, 0.50, 0.58, 0.35, 0.42)
    _draw_arrow(ax, 0.80, 0.58, 0.65, 0.42)
    ax.text(0.5, 0.92, "Evaluation + Explainability Stack", ha="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _plot_local_results(ctx, path: Path) -> None:
    report = ctx.evaluation_report
    baseline = report["baseline_metrics"]
    tuned = report["fine_tuned_metrics"]
    labels = ["Token F1", "Exact match", "Role coverage", "ROUGE-L"]
    base_vals = [baseline["token_f1"], baseline["exact_match"], baseline["role_coverage"], baseline["rouge_l"]]
    tuned_vals = [tuned["token_f1"], tuned["exact_match"], tuned["role_coverage"], tuned["rouge_l"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    width = 0.35
    ax.bar([i - width / 2 for i in x], base_vals, width=width, color=PALETTE["gold"], label="Local zero-shot")
    ax.bar([i + width / 2 for i in x], tuned_vals, width=width, color=PALETTE["navy"], label="Local fine-tuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Local Zero-Shot vs Fine-Tuned Metrics", fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)


def _plot_domain_analysis(ctx, path: Path) -> None:
    domain_scores = ctx.evaluation_report["fine_tuned_metrics"]["domain_token_f1"]
    labels = list(domain_scores.keys())
    values = [domain_scores[key] for key in labels]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(labels, values, color=[PALETTE["teal"], PALETTE["navy"], PALETTE["gold"]][: len(labels)])
    ax.set_ylim(0, 0.6)
    ax.set_title("Domain-Wise Token F1", fontsize=15, fontweight="bold")
    ax.set_ylabel("Token F1")
    for label, value in zip(labels, values):
        ax.text(label, value + 0.015, f"{value:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _plot_training_curve(ctx, path: Path) -> None:
    history = ctx.training_summary["training_history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]
    selection_f1 = [row["selection_token_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].plot(epochs, train_loss, marker="o", linewidth=2.5, color=PALETTE["navy"], label="Train loss")
    axes[0].plot(epochs, validation_loss, marker="o", linewidth=2.5, color=PALETTE["gold"], label="Validation loss")
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, selection_f1, marker="o", linewidth=2.5, color=PALETTE["teal"], label="Validation token F1")
    axes[1].set_title("Validation Learning Trend", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)

    fig.suptitle("Local QA-SRL Training Dynamics", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _plot_prompt_ablation(path: Path) -> None:
    payload = _safe_json(RESULTS_DIR / "prompt_ablation.json")
    candidates = payload.get("profiles") if isinstance(payload.get("profiles"), list) else payload.get("results", [])
    labels, values = [], []
    for row in candidates[:6]:
        if not isinstance(row, dict):
            continue
        labels.append(str(row.get("name", row.get("prompt", f"P{len(labels)+1}"))))
        values.append(float(row.get("token_f1", 0.0)))
    if not labels:
        labels, values = ["N/A"], [0.0]
    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    ax.bar(labels, values, color=PALETTE["teal"])
    ax.set_ylim(0.0, max(1.0, max(values) + 0.1))
    ax.set_ylabel("Token F1")
    ax.set_title("Prompt Ablation Results", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, path)


def _plot_gemini_vs_local(ctx, path: Path) -> None:
    payload = _safe_json(RESULTS_DIR / "benchmark_comparison.json")
    rows = []
    for key in ("models", "results", "comparisons", "rows"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            rows = [row for row in values if isinstance(row, dict)]
            break

    labels, values = [], []
    for row in rows[:6]:
        labels.append(str(row.get("model", row.get("name", f"M{len(labels)+1}"))))
        values.append(float(row.get("token_f1", row.get("f1", 0.0))))

    if not labels:
        labels = ["Local fine-tuned"]
        values = [float(ctx.evaluation_report["fine_tuned_metrics"]["token_f1"])]

    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    color_cycle = [PALETTE["navy"], PALETTE["gold"], PALETTE["teal"], PALETTE["slate"], PALETTE["green"], PALETTE["violet"]]
    ax.bar(labels, values, color=color_cycle[: len(labels)])
    ax.set_ylabel("Token F1")
    ax.set_ylim(0.0, max(1.0, max(values) + 0.1))
    ax.set_title("Local vs Gemini Benchmark Comparison", fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, path)


def _draw_pipeline_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_box(ax, 0.04, 0.62, 0.18, 0.18, "QA-SRL Data", PALETTE["cream"], PALETTE["navy"])
    _draw_box(ax, 0.26, 0.62, 0.18, 0.18, "Grouped Inputs", "#e8f5f3", PALETTE["teal"])
    _draw_box(ax, 0.48, 0.62, 0.18, 0.18, "LoRA Training", "#eef4fb", PALETTE["navy"])
    _draw_box(ax, 0.70, 0.62, 0.18, 0.18, "Evaluation", "#fff5e5", PALETTE["gold"])
    _draw_box(ax, 0.26, 0.30, 0.18, 0.18, "Inference", "#f8eef0", PALETTE["red"])
    _draw_box(ax, 0.48, 0.30, 0.18, 0.18, "Cleanup/Fallback", "#eef0f4", PALETTE["slate"])
    _draw_box(ax, 0.70, 0.30, 0.18, 0.18, "Streamlit + Deck", "#f1fbf5", PALETTE["green"])
    _draw_arrow(ax, 0.22, 0.71, 0.26, 0.71)
    _draw_arrow(ax, 0.44, 0.71, 0.48, 0.71)
    _draw_arrow(ax, 0.66, 0.71, 0.70, 0.71)
    _draw_arrow(ax, 0.56, 0.62, 0.35, 0.48)
    _draw_arrow(ax, 0.44, 0.39, 0.48, 0.39)
    _draw_arrow(ax, 0.66, 0.39, 0.70, 0.39)
    ax.text(0.5, 0.92, "Pipeline Architecture", ha="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _draw_innovation_comparison(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.28, 0.92, "Field gap", ha="center", fontsize=18, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.72, 0.92, "Local response", ha="center", fontsize=18, fontweight="bold", color=PALETTE["teal"])
    rows = [
        ("Large compute assumptions", "LoRA on flan-t5-small for CPU"),
        ("Weak explanation integration", "Built-in InstaShap-style outputs"),
        ("Fragile small-model outputs", "Cleanup, snapping, fallback"),
        ("Scattered deliverables", "Assets, docs, deck generated together"),
    ]
    y_positions = [0.74, 0.56, 0.38, 0.20]
    for (left, right), y in zip(rows, y_positions):
        _draw_box(ax, 0.06, y, 0.34, 0.12, left, PALETTE["cream"], PALETTE["navy"])
        _draw_box(ax, 0.60, y, 0.34, 0.12, right, "#e8f5f3", PALETTE["teal"])
        _draw_arrow(ax, 0.40, y + 0.06, 0.60, y + 0.06)
    fig.tight_layout()
    _save(fig, path)


def generate_assets() -> dict[str, str]:
    _figure_style()
    ctx = build_presentation_context()

    _plot_research_timeline(ctx, ctx.figure_paths["research_timeline"])
    _draw_system_overview(ctx.figure_paths["system_overview"])
    _draw_cleanup_flow(ctx.figure_paths["cleanup_flow"])
    _draw_evaluation_stack(ctx.figure_paths["evaluation_stack"])
    _plot_local_results(ctx, ctx.figure_paths["local_results"])
    _plot_domain_analysis(ctx, ctx.figure_paths["domain_analysis"])
    _plot_training_curve(ctx, ctx.figure_paths["training_curve"])
    _plot_prompt_ablation(ctx.figure_paths["prompt_ablation_chart"])
    _plot_gemini_vs_local(ctx, ctx.figure_paths["gemini_vs_local"])
    _draw_pipeline_architecture(ctx.figure_paths["pipeline_architecture"])
    _draw_innovation_comparison(ctx.figure_paths["innovation_comparison"])

    source_instashap = RESULTS_DIR / "instashap_example.png"
    if source_instashap.exists():
        shutil.copy2(source_instashap, ctx.figure_paths["instashap_example"])

    asset_index = {figure_id: str(path.relative_to(ctx.root)) for figure_id, path in ctx.figure_paths.items()}
    (ctx.outputs["assets_dir"] / "asset_index.json").write_text(json.dumps(asset_index, indent=2), encoding="utf-8")
    return asset_index


def main() -> None:
    generated = generate_assets()
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
