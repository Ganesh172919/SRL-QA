from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from presentation.builder import build_presentation_context


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
}


def _figure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_training_curve(ctx, path: Path) -> None:
    history = ctx.training_summary["training_history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]
    selection_f1 = [row["selection_token_f1"] for row in history]
    selection_cov = [row["selection_role_coverage"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].plot(epochs, train_loss, marker="o", linewidth=2.5, color=PALETTE["navy"], label="Train loss")
    axes[0].plot(epochs, validation_loss, marker="o", linewidth=2.5, color=PALETTE["gold"], label="Validation loss")
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, selection_f1, marker="o", linewidth=2.5, color=PALETTE["teal"], label="Selection token F1")
    axes[1].plot(epochs, selection_cov, marker="o", linewidth=2.5, color=PALETTE["slate"], label="Selection coverage")
    axes[1].set_title("Validation Selection Metrics", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)

    fig.suptitle("QA-SRL Fine-Tuning Training Curve", fontsize=16, fontweight="bold", color=PALETTE["ink"])
    fig.tight_layout()
    _save(fig, path)


def _plot_metric_comparison(ctx, path: Path) -> None:
    report = ctx.evaluation_report
    baseline = report["baseline_metrics"]
    tuned = report["fine_tuned_metrics"]
    labels = ["Token F1", "Exact match", "Role coverage", "ROUGE-L"]
    baseline_values = [baseline["token_f1"], baseline["exact_match"], baseline["role_coverage"], baseline["rouge_l"]]
    tuned_values = [tuned["token_f1"], tuned["exact_match"], tuned["role_coverage"], tuned["rouge_l"]]
    positions = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar([position - width / 2 for position in positions], baseline_values, width=width, color=PALETTE["gold"], label="Zero-shot")
    ax.bar([position + width / 2 for position in positions], tuned_values, width=width, color=PALETTE["navy"], label="Fine-tuned")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Verified Local Metrics: Zero-Shot vs Fine-Tuned", fontsize=15, fontweight="bold")
    ax.legend(frameon=False)
    for position, value in zip([position - width / 2 for position in positions], baseline_values):
        ax.text(position, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    for position, value in zip([position + width / 2 for position in positions], tuned_values):
        ax.text(position, value + 0.02, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def _plot_domain_performance(ctx, path: Path) -> None:
    domain_scores = ctx.evaluation_report["fine_tuned_metrics"]["domain_token_f1"]
    labels = list(domain_scores.keys())
    values = [domain_scores[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    colors = [PALETTE["teal"], PALETTE["navy"], PALETTE["gold"]]
    ax.bar(labels, values, color=colors[: len(labels)], width=0.55)
    ax.set_ylim(0.0, 0.5)
    ax.set_ylabel("Token F1")
    ax.set_title("Domain-Wise Token F1 on the Verified Test Slice", fontsize=15, fontweight="bold")
    for label, value in zip(labels, values):
        ax.text(label, value + 0.015, f"{value:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _plot_token_f1_histogram(ctx, path: Path) -> None:
    values = [row["token_f1"] for row in ctx.example_rows]
    median_value = ctx.aggregate["median_token_f1"]
    mean_value = ctx.evaluation_report["fine_tuned_metrics"]["token_f1"]
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.hist(values, bins=12, color=PALETTE["navy"], alpha=0.82, edgecolor="white")
    ax.axvline(mean_value, color=PALETTE["gold"], linewidth=2.5, label=f"Mean {mean_value:.4f}")
    ax.axvline(median_value, color=PALETTE["teal"], linewidth=2.5, linestyle="--", label=f"Median {median_value:.4f}")
    ax.set_title("Token F1 Distribution Across 100 Evaluated Examples", fontsize=15, fontweight="bold")
    ax.set_xlabel("Token F1")
    ax.set_ylabel("Example count")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, path)


def _plot_role_frequency(ctx, path: Path) -> None:
    top_roles = ctx.role_frequencies[:10]
    labels = [role for role, _ in top_roles]
    values = [count for _, count in top_roles]
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.bar(labels, values, color=PALETTE["teal"])
    ax.set_title("Role Frequency in `train_grouped_v4.jsonl`", fontsize=15, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Role")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.012, f"{value}", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=2, facecolor=face, edgecolor=edge)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=PALETTE["ink"], wrap=True)


def _draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, linewidth=2, color=PALETTE["slate"]))


def _draw_pipeline_architecture(ctx, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _draw_box(ax, 0.03, 0.61, 0.17, 0.18, "QA-SRL Bank 2.1\narchive + cache", PALETTE["cream"], PALETTE["navy"])
    _draw_box(ax, 0.24, 0.61, 0.18, 0.18, "Grouped predicate\nexamples\n`data.py`", "#e8f5f3", PALETTE["teal"])
    _draw_box(ax, 0.46, 0.61, 0.18, 0.18, "LoRA fine-tuning\n`train.py`", "#eef4fb", PALETTE["navy"])
    _draw_box(ax, 0.68, 0.61, 0.18, 0.18, "Evaluation + XAI\n`evaluate.py`", "#fff5e5", PALETTE["gold"])

    _draw_box(ax, 0.24, 0.26, 0.18, 0.18, "Role parsing +\nrefinement\n`roles.py`", "#f1fbf5", PALETTE["green"])
    _draw_box(ax, 0.46, 0.26, 0.18, 0.18, "Inference +\nfallback\n`inference.py`", "#f8eef0", PALETTE["red"])
    _draw_box(ax, 0.68, 0.26, 0.18, 0.18, "Streamlit UI +\npresentation outputs", "#eef0f4", PALETTE["slate"])

    _draw_arrow(ax, 0.20, 0.70, 0.24, 0.70)
    _draw_arrow(ax, 0.42, 0.70, 0.46, 0.70)
    _draw_arrow(ax, 0.64, 0.70, 0.68, 0.70)
    _draw_arrow(ax, 0.55, 0.61, 0.55, 0.44)
    _draw_arrow(ax, 0.33, 0.61, 0.33, 0.44)
    _draw_arrow(ax, 0.64, 0.35, 0.68, 0.35)
    _draw_arrow(ax, 0.42, 0.35, 0.46, 0.35)

    ax.text(0.5, 0.92, "Local QA-SRL Pipeline in `finetuning/`", ha="center", va="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.5, 0.08, "The architecture emphasizes local feasibility, structured outputs, explanation, and deployability.", ha="center", fontsize=11, color=PALETTE["slate"])
    _save(fig, path)


def _draw_prompt_evolution(ctx, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("Long prompts", "Too much instruction baggage for a small model"),
        ("Compact format", "Predicate + sentence + labels"),
        ("Role serialization", "Stable `AGENT:` / `THEME:` output lines"),
        ("Span snapping", "Pull generations back onto the sentence"),
        ("Fallback recovery", "Keep the deployed system usable"),
    ]
    x_positions = [0.02, 0.22, 0.42, 0.62, 0.82]
    colors = [PALETTE["cream"], "#e8f5f3", "#eef4fb", "#fff5e5", "#f8eef0"]
    edges = [PALETTE["gold"], PALETTE["teal"], PALETTE["navy"], PALETTE["gold"], PALETTE["red"]]
    for idx, ((title, note), x_pos) in enumerate(zip(steps, x_positions)):
        _draw_box(ax, x_pos, 0.36, 0.15, 0.26, f"{title}\n\n{note}", colors[idx], edges[idx])
        if idx < len(x_positions) - 1:
            _draw_arrow(ax, x_pos + 0.15, 0.49, x_positions[idx + 1], 0.49)
    ax.text(0.5, 0.83, "Prompt Tuning Progression For The Local T5 Pipeline", ha="center", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.5, 0.17, "Prompt tuning evolved into a broader robustness stack that includes parsing discipline, span cleanup, and fallback logic.", ha="center", fontsize=11, color=PALETTE["slate"])
    _save(fig, path)


def _draw_innovation_comparison(ctx, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.28, 0.92, "Field gap", ha="center", fontsize=18, fontweight="bold", color=PALETTE["navy"])
    ax.text(0.72, 0.92, "Local response in `finetuning/`", ha="center", fontsize=18, fontweight="bold", color=PALETTE["teal"])

    rows = [
        ("Large compute assumptions", "LoRA over `flan-t5-small` on CPU"),
        ("Weak integration of explanation", "InstaShap-style metrics + plot + UI display"),
        ("Fragile small-model outputs", "Span snapping, role refinement, fallback mapping"),
        ("Scattered demo artifacts", "Artifacts, metrics, app, docs, and deck in one folder"),
    ]
    y_positions = [0.74, 0.56, 0.38, 0.20]
    for (gap, response), y_pos in zip(rows, y_positions):
        _draw_box(ax, 0.06, y_pos, 0.34, 0.12, gap, PALETTE["cream"], PALETTE["navy"])
        _draw_box(ax, 0.60, y_pos, 0.34, 0.12, response, "#e8f5f3", PALETTE["teal"])
        _draw_arrow(ax, 0.40, y_pos + 0.06, 0.60, y_pos + 0.06)
    ax.text(0.5, 0.06, "The novelty is practical integration under constraints, not an unsupported claim of replacing unconstrained research systems.", ha="center", fontsize=11, color=PALETTE["slate"])
    _save(fig, path)


def generate_assets() -> dict[str, str]:
    _figure_style()
    ctx = build_presentation_context()
    _plot_training_curve(ctx, ctx.figure_paths["training_curve"])
    _plot_metric_comparison(ctx, ctx.figure_paths["metric_comparison"])
    _plot_domain_performance(ctx, ctx.figure_paths["domain_performance"])
    _plot_token_f1_histogram(ctx, ctx.figure_paths["token_f1_histogram"])
    _plot_role_frequency(ctx, ctx.figure_paths["role_frequency"])
    _draw_pipeline_architecture(ctx, ctx.figure_paths["pipeline_architecture"])
    _draw_prompt_evolution(ctx, ctx.figure_paths["prompt_evolution"])
    _draw_innovation_comparison(ctx, ctx.figure_paths["innovation_comparison"])
    shutil.copy2(ctx.root / "results" / "instashap_example.png", ctx.figure_paths["instashap_example"])

    asset_index = {
        figure_id: str(path.relative_to(ctx.root))
        for figure_id, path in ctx.figure_paths.items()
    }
    (ctx.outputs["assets_dir"] / "asset_index.json").write_text(json.dumps(asset_index, indent=2), encoding="utf-8")
    return asset_index


def main() -> None:
    generated = generate_assets()
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
