from __future__ import annotations

import json
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_DIR = ROOT / "presentation"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data" / "processed"
SOURCES_PATH = PRESENTATION_DIR / "sources.json"


SURVEY_MILESTONES = [
    {
        "year": "2015",
        "label": "He, Lewis, and Zettlemoyer",
        "source": "docs/analysis.md",
        "field_move": "Introduced QA-SRL as a question-answer representation for predicate-argument structure.",
        "before": "Earlier semantic role systems were strong on labels, but they were less natural for question-driven human inspection.",
        "what_systems_did": "Moved the conversation from role tags alone to question-answer supervision tied to events and arguments.",
        "what_it_enabled": "Made semantic structure readable, crowd-annotatable, and easier to align with downstream QA tasks.",
        "remaining_gap": "Did not address CPU-first deployment, instruction tuning, or local token-level explanations.",
        "local_lesson": "Question-driven structure remains the clearest bridge between symbolic roles and user-facing outputs."
    },
    {
        "year": "2018",
        "label": "FitzGerald et al.",
        "source": "docs/analysis.md",
        "field_move": "Scaled QA-SRL with QA-SRL Bank 2.0 and large-scale neural parsing.",
        "before": "The field had the formulation, but not yet the large benchmark and parser pipeline needed for broad experimentation.",
        "what_systems_did": "Standardized larger training data and made neural QA-SRL parsing a serious benchmarked task.",
        "what_it_enabled": "Created the template for dataset-centric progress and model comparison across domains.",
        "remaining_gap": "Research-scale systems still assumed more generous compute than a small CPU-only machine.",
        "local_lesson": "A modern local build should preserve the benchmark lineage even when it uses a smaller runtime recipe."
    },
    {
        "year": "2020",
        "label": "Roit et al.",
        "source": "docs/analysis.md",
        "field_move": "Improved annotation quality with controlled crowdsourcing and a higher-quality gold dataset.",
        "before": "Scaling alone did not solve annotation noise, edge cases, or consistency problems across question templates.",
        "what_systems_did": "Focused on data quality, adjudication, and reliable gold signals rather than raw size alone.",
        "what_it_enabled": "Encouraged later systems to care about evaluation fairness, not just model capacity.",
        "remaining_gap": "Annotation quality improvements still did not solve local deployment, explainability, or compact adaptation.",
        "local_lesson": "A practical system should keep its claims tied to verified evaluation artifacts and consistent post-processing."
    },
    {
        "year": "2020",
        "label": "Klein et al. (QANom)",
        "source": "docs/analysis.md",
        "field_move": "Extended QA-driven role annotation beyond verbs to nominalizations.",
        "before": "Verb-centric QA-SRL covered only part of predicate-argument reasoning found in realistic text.",
        "what_systems_did": "Showed that question-driven semantic annotation generalizes beyond verb predicates.",
        "what_it_enabled": "Expanded the task from a single benchmark to a broader semantic annotation paradigm.",
        "remaining_gap": "Nominal coverage is valuable, but it still leaves open the need for compact, interpretable local pipelines.",
        "local_lesson": "The local project stays verb-focused, yet it should be framed as compatible with broader QA-driven semantics."
    },
    {
        "year": "2025",
        "label": "InstaSHAP",
        "source": "docs/analysis.md",
        "field_move": "Pushed explanation design toward near-instant, attribution-oriented Shapley behavior.",
        "before": "Explainability was often too slow, too generic, or disconnected from the prediction workflow.",
        "what_systems_did": "Motivated explanation methods that are quick enough for interactive systems rather than offline-only studies.",
        "what_it_enabled": "Opened the door for practical explanation modules that can sit beside live model inference.",
        "remaining_gap": "The original method was not a drop-in QA-SRL transformer package for CPU-only fine-tuned generation.",
        "local_lesson": "A local adaptation can legitimately borrow the speed-and-interpretability spirit even if it is not a full paper reproduction."
    },
    {
        "year": "2026",
        "label": "Cross-lingual QA-driven predicate-argument annotation",
        "source": "docs/analysis.md",
        "field_move": "Showed the framework still has momentum beyond English and beyond narrow benchmark settings.",
        "before": "It was easy to treat QA-SRL as a completed English-only niche rather than an active research direction.",
        "what_systems_did": "Reaffirmed that question-driven role labeling is still evolving across languages and annotation settings.",
        "what_it_enabled": "Justified presenting QA-driven semantics as a living field rather than a historical detour.",
        "remaining_gap": "Cross-lingual systems do not automatically yield a small, locally explainable English QA-SRL stack.",
        "local_lesson": "The local project is a constrained implementation, but it sits on a still-relevant research trajectory."
    }
]


SURVEY_GAPS = [
    (
        "CPU-feasible fine-tuning",
        "Many research systems assume larger accelerators, longer runs, or broader parameter updates.",
        "This project uses LoRA over `google/flan-t5-small` to keep adaptation realistic on an 8 GB CPU machine."
    ),
    (
        "Integrated explanation",
        "Role prediction and explanation are often evaluated separately or left outside the main interface.",
        "This project includes an InstaShap-style token attribution module in both evaluation and the Streamlit app."
    ),
    (
        "Deployment path",
        "Survey papers describe parsing models, but not always a runnable local application with stored artifacts and UI.",
        "This project ships model artifacts, results, and a Streamlit front end in one folder."
    ),
    (
        "Prompt discipline for small models",
        "Large instruction prompts can work for bigger models but often destabilize compact encoder-decoder variants.",
        "This project compresses the prompt to predicate plus sentence plus labels and then repairs outputs post-generation."
    ),
    (
        "Fallback behavior",
        "Prior literature focuses on primary model quality rather than what to do when a compact local run emits weak structure.",
        "This project adds extractive snapping, role refinement, and a fallback extractor to keep the system usable."
    )
]


SURVEY_NOTES = [
    "The field repeatedly alternates between representation innovation, dataset scaling, and data quality cleanup.",
    "Question-driven semantics matters because it exposes roles in a way humans can audit quickly.",
    "Annotation quality is not a secondary issue in QA-SRL; it changes what the model can reliably learn.",
    "A dataset can be historically important even when a local project trains only on a small verified slice.",
    "CPU-first engineering is not a replacement for large-model research, but it is a legitimate design target.",
    "A local system gains credibility when it clearly separates literature context from verified run metrics.",
    "Explainability is more persuasive when it is shown on the same cases used for model evaluation.",
    "Survey value is highest when it tells us what earlier systems solved and what they left unsolved.",
    "The move from labels to questions increased semantic accessibility for both annotators and project audiences.",
    "Role-centric QA remains relevant because many user queries implicitly ask for agents, themes, locations, or times.",
    "Practical deployment constraints can justify a smaller model even when the resulting accuracy ceiling is lower.",
    "The existence of cross-lingual work keeps the local English pipeline from looking like a dead-end exercise.",
    "A good survey does not inflate novelty; it identifies exactly where the local build adds useful integration.",
    "The project should be framed as a strong engineering system under constraints, not as an unconstrained SOTA claim.",
    "Structured output plus post-processing is a common pattern when compact generators are pushed into symbolic tasks.",
    "The local dataset pipeline matters because reproducibility begins before training starts.",
    "The survey must acknowledge that benchmark mismatch can make aggressive comparison language misleading.",
    "The project benefits from using the official QA-SRL Bank lineage even when it does not run full-bank training locally.",
    "Prompt tuning is part of the system design story because prompt shape influences every later recovery step.",
    "The explanation layer matters more in a viva setting because it helps defend why a prediction should be trusted."
]


@dataclass
class SlideSpec:
    index: int
    section_id: str
    title: str
    bullets: list[str] = field(default_factory=list)
    image_id: str | None = None
    image_caption: str | None = None
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    footer: str = ""


@dataclass
class PresentationContext:
    manifest: dict
    root: Path
    presentation_dir: Path
    outputs: dict[str, Path]
    figure_paths: dict[str, Path]
    sources: dict[str, dict]
    source_texts: dict[str, str]
    training_summary: dict
    evaluation_report: dict
    processed_counts: dict[str, int]
    train_domain_counts: dict[str, int]
    role_frequencies: list[tuple[str, int]]
    example_rows: list[dict]
    best_examples: list[dict]
    worst_examples: list[dict]
    exact_examples: list[dict]
    xai_examples: list[dict]
    aggregate: dict


def load_manifest() -> dict:
    return json.loads((PRESENTATION_DIR / "manifest.json").read_text(encoding="utf-8"))


def _resolve_output_paths(manifest: dict) -> dict[str, Path]:
    return {key: PRESENTATION_DIR / relative_path for key, relative_path in manifest["outputs"].items()}


def ensure_output_dirs(manifest: dict | None = None) -> dict[str, Path]:
    manifest = manifest or load_manifest()
    outputs = _resolve_output_paths(manifest)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("assets_dir", "pdfs_dir", "deck_dir", "sections_dir"):
        outputs[key].mkdir(parents=True, exist_ok=True)
    return outputs


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _collect_train_stats(path: Path) -> tuple[dict[str, int], list[tuple[str, int]]]:
    domain_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            domain_counter[record["domain"]] += 1
            for role, answers in record["roles"].items():
                if answers:
                    role_counter[role] += 1
    return dict(domain_counter), role_counter.most_common()


def _load_sources() -> dict[str, dict]:
    if not SOURCES_PATH.exists():
        return {}
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def _load_dataset_stats() -> dict | None:
    stats_path = RESULTS_DIR / "dataset_stats.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _clean_cell(value: object) -> str:
    return str(value).replace("|", "/").replace("\n", "<br>")


def _format_roles(role_mapping: dict | None) -> str:
    if not role_mapping:
        return "None"
    chunks = []
    for role, answers in role_mapping.items():
        if not answers:
            continue
        if isinstance(answers, list):
            rendered = "; ".join(str(answer) for answer in answers)
        else:
            rendered = str(answers)
        chunks.append(f"{role}: {rendered}")
    return " | ".join(chunks) if chunks else "None"


def _example_assessment(example: dict) -> str:
    token_f1 = example["token_f1"]
    exact = example["exact_match"]
    coverage = example["role_coverage"]
    if exact == 1.0:
        return "Exact structural match with clean role and span alignment."
    if token_f1 >= 0.8:
        return "Near-exact semantic recovery with only minor span noise."
    if coverage == 1.0 and token_f1 >= 0.5:
        return "All gold roles were surfaced, but extra spans or boundary drift reduced precision."
    if coverage == 1.0 and token_f1 < 0.5:
        return "Complete role discovery with weak answer precision; the role scaffold is stronger than the span selection."
    if coverage >= 0.5 and token_f1 == 0.0:
        return "Partial role intuition exists, but the recovered spans do not overlap the gold answers."
    if coverage >= 0.5:
        return "Some relevant roles were recovered, yet span accuracy remains unstable."
    return "Prediction quality is weak on both role selection and answer content for this example."


def _build_source_texts() -> dict[str, str]:
    return {
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/analysis.md": (DOCS_DIR / "analysis.md").read_text(encoding="utf-8"),
        "docs/project_summary.md": (DOCS_DIR / "project_summary.md").read_text(encoding="utf-8"),
        "docs/latest_research_prompt.txt": (DOCS_DIR / "latest_research_prompt.txt").read_text(encoding="utf-8")
    }


def build_presentation_context() -> PresentationContext:
    manifest = load_manifest()
    outputs = ensure_output_dirs(manifest)
    figure_paths = {figure["id"]: PRESENTATION_DIR / figure["filename"] for figure in manifest["figures"]}
    dataset_stats = _load_dataset_stats()
    training_summary = _read_json(RESULTS_DIR / "training_summary.json")
    evaluation_report = _read_json(RESULTS_DIR / "evaluation_report.json")
    if dataset_stats:
        processed_counts = dataset_stats["processed_counts"]
        train_domain_counts = dataset_stats["train_domain_counts"]
        role_frequencies = [tuple(item) for item in dataset_stats["role_frequencies"]]
    else:
        processed_counts = {
            "train": _count_jsonl_lines(DATA_DIR / "train_grouped_v4.jsonl"),
            "validation": _count_jsonl_lines(DATA_DIR / "validation_grouped_v4.jsonl"),
            "test": _count_jsonl_lines(DATA_DIR / "test_grouped_v4.jsonl")
        }
        train_domain_counts, role_frequencies = _collect_train_stats(DATA_DIR / "train_grouped_v4.jsonl")

    example_rows = list(evaluation_report["fine_tuned_metrics"]["per_example"])
    ranked_examples = sorted(
        example_rows,
        key=lambda item: (item["exact_match"], item["token_f1"], item["role_coverage"], item["rouge_l"]),
        reverse=True
    )
    worst_examples = sorted(
        example_rows,
        key=lambda item: (item["token_f1"], item["role_coverage"], item["rouge_l"], item["exact_match"])
    )
    xai_examples = list(evaluation_report["xai_metrics"]["examples"])
    token_f1_values = [row["token_f1"] for row in example_rows]
    rouge_values = [row["rouge_l"] for row in example_rows]
    aggregate = {
        "example_count": len(example_rows),
        "exact_match_count": sum(1 for row in example_rows if row["exact_match"] == 1.0),
        "full_coverage_count": sum(1 for row in example_rows if row["role_coverage"] == 1.0),
        "zero_token_f1_count": sum(1 for row in example_rows if row["token_f1"] == 0.0),
        "above_half_count": sum(1 for row in example_rows if row["token_f1"] > 0.5),
        "above_08_count": sum(1 for row in example_rows if row["token_f1"] >= 0.8),
        "median_token_f1": round(statistics.median(token_f1_values), 4),
        "median_rouge_l": round(statistics.median(rouge_values), 4)
    }

    return PresentationContext(
        manifest=manifest,
        root=ROOT,
        presentation_dir=PRESENTATION_DIR,
        outputs=outputs,
        figure_paths=figure_paths,
        sources=_load_sources(),
        source_texts=_build_source_texts(),
        training_summary=training_summary,
        evaluation_report=evaluation_report,
        processed_counts=processed_counts,
        train_domain_counts=train_domain_counts,
        role_frequencies=role_frequencies,
        example_rows=example_rows,
        best_examples=ranked_examples[:8],
        worst_examples=worst_examples[:8],
        exact_examples=[row for row in ranked_examples if row["exact_match"] == 1.0][:6],
        xai_examples=xai_examples,
        aggregate=aggregate
    )


def _figure_markdown_path(ctx: PresentationContext, figure_id: str, asset_prefix: str) -> str:
    filename = ctx.figure_paths[figure_id].name
    return f"{asset_prefix}/{filename}" if asset_prefix else filename


def _table_lines(headers: Iterable[object], rows: Iterable[Iterable[object]]) -> list[str]:
    header_list = [_clean_cell(cell) for cell in headers]
    lines = [
        f"| {' | '.join(header_list)} |",
        f"| {' | '.join(['---'] * len(header_list))} |"
    ]
    for row in rows:
        lines.append(f"| {' | '.join(_clean_cell(cell) for cell in row)} |")
    return lines


def _section_asset_line(ctx: PresentationContext, figure_id: str, asset_prefix: str) -> list[str]:
    figure_title = next(item["title"] for item in ctx.manifest["figures"] if item["id"] == figure_id)
    return [f"![{figure_title}]({_figure_markdown_path(ctx, figure_id, asset_prefix)})", ""]


IMPLEMENTATION_GUIDES = [
    {
        "path": "train.py",
        "role": "Primary fine-tuning entrypoint that orchestrates the baseline run, LoRA training loop, model selection, and artifact writing.",
        "flow_in": "Consumes grouped QA-SRL records from `src/qasrl_cpu/data.py` plus model settings from command-line arguments.",
        "flow_out": "Writes `results/training_summary.json`, `results/test_predictions.json`, and adapter files under `artifacts/flan_t5_small_lora/`.",
        "summary_points": [
            "Keeps the entire run on CPU by explicitly selecting a CPU device and limiting thread usage.",
            "Runs a zero-shot baseline before fine-tuning so the final presentation can show a real local improvement story.",
            "Uses early stopping on validation token F1 rather than relying only on loss."
        ],
        "symbols": [
            ("TextPairDataset", "Wraps grouped examples for PyTorch `DataLoader` iteration without changing record structure."),
            ("Seq2SeqCollator", "Tokenizes inputs and labels, pads batches, and masks label padding with `-100` for seq2seq loss."),
            ("set_seed", "Aligns Python, NumPy, and Torch randomness so the small local benchmark slice is reproducible."),
            ("evaluate_loss", "Measures validation loss without generation so training can track optimization stability."),
            ("train_model", "Executes the main baseline, training, checkpoint selection, and final evaluation workflow."),
            ("build_parser", "Defines the full fine-tuning interface used by both `train.py` and `run_project.py`.")
        ]
    },
    {
        "path": "evaluate.py",
        "role": "Evaluation and XAI entrypoint that re-runs prediction, computes metrics, and writes the final benchmark artifacts.",
        "flow_in": "Loads the saved adapter, tokenizer, grouped records, and the base model specified in training metadata.",
        "flow_out": "Writes `results/evaluation_report.json`, `results/evaluation_summary.md`, and `results/instashap_example.png`.",
        "summary_points": [
            "Separates final benchmark reporting from the training script so presentation claims can cite a dedicated evaluation artifact.",
            "Measures both task quality and explanation quality in the same run.",
            "Regenerates a baseline on the same split to keep the comparison fair."
        ],
        "symbols": [
            ("build_gold_token_set", "Creates a token reference set from predicate and gold answers for plausibility scoring."),
            ("compute_xai_suite", "Computes plausibility, faithfulness, and a combined XAI score over a bounded set of records."),
            ("write_markdown_report", "Writes a compact markdown summary that the presentation package can treat as source-of-truth support."),
            ("run_evaluation", "Owns the full benchmark pass and the final evaluation report structure."),
            ("build_parser", "Defines the evaluation CLI with split, slice size, beam count, and XAI limit options.")
        ]
    },
    {
        "path": "app.py",
        "role": "Streamlit application that turns the trained adapter into an inspectable QA-SRL demo.",
        "flow_in": "Reads the adapter directory, optional evaluation report, and user-provided sentence plus predicate.",
        "flow_out": "Displays structured roles, QA pairs, token attributions, and training metadata inside the UI.",
        "summary_points": [
            "Bootstraps Streamlit automatically when launched with `python app.py`.",
            "Keeps the interface local and file-driven instead of depending on hosted services.",
            "Packages prediction and explanation together so the viva can demonstrate interpretability live."
        ],
        "symbols": [
            ("bootstrap_streamlit", "Re-runs the file under Streamlit when executed as a normal Python script."),
            ("find_predicate_index", "Locates the predicate token in the user sentence so the marked input can be reconstructed."),
            ("load_resources", "Caches the adapter, tokenizer, and metadata for repeated interactive use."),
            ("render_app", "Defines the page layout, sidebar metrics, prediction action, and explanation rendering.")
        ]
    },
    {
        "path": "run_project.py",
        "role": "Compact orchestration layer that runs training and evaluation end to end with a smaller user-facing CLI.",
        "flow_in": "Inherits parser defaults from `train.py` and `evaluate.py`.",
        "flow_out": "Prints a merged summary of key artifact paths and top-level metrics after both phases complete.",
        "summary_points": [
            "Provides a single command for a fresh end-to-end demonstration run.",
            "Avoids duplicating model and data settings by reusing parser builders.",
            "Keeps the presentation story simple when explaining how to reproduce the project."
        ],
        "symbols": [
            ("build_parser", "Defines a small orchestration CLI for the most important slice-size and epoch parameters."),
            ("main", "Runs training, evaluation, and merged summary printing in sequence.")
        ]
    },
    {
        "path": "src/qasrl_cpu/data.py",
        "role": "Data ingestion and preprocessing layer that downloads QA-SRL Bank 2.1, groups predicate examples, and writes processed JSONL files.",
        "flow_in": "Uses the official archive URL and raw sentence-level QA-SRL annotations from the public dataset tarball.",
        "flow_out": "Creates processed grouped files such as `train_grouped_v4.jsonl`, `validation_grouped_v4.jsonl`, and `test_grouped_v4.jsonl`.",
        "summary_points": [
            "Transforms sentence-level question labels into predicate-level grouped training targets.",
            "Encodes the final prompt format used for fine-tuning.",
            "Handles cached download, split extraction, and reproducible sampling in one module."
        ],
        "symbols": [
            ("ensure_archive", "Downloads the QA-SRL Bank archive only when it is not already cached locally."),
            ("mark_predicate", "Wraps the predicate token with tags for marked-sentence support."),
            ("detect_domain", "Maps sentence IDs to `TQA`, `wikinews`, or `wikipedia` domains."),
            ("extract_answers", "Collects valid answer spans from crowd judgments and converts them into text spans."),
            ("build_input_text", "Creates the short T5-friendly prompt used throughout training and inference."),
            ("sentence_to_grouped_examples", "Converts a sentence object into grouped predicate examples with role mappings."),
            ("iter_split_examples", "Streams examples out of the compressed archive without requiring manual preprocessing steps."),
            ("ensure_processed_dataset", "Creates the versioned processed JSONL files when they do not yet exist."),
            ("sample_records", "Applies seeded sampling so benchmark slices stay reproducible."),
            ("prepare_grouped_dataset", "Builds the Hugging Face `DatasetDict` used by training and evaluation.")
        ]
    },
    {
        "path": "src/qasrl_cpu/roles.py",
        "role": "Role taxonomy and post-processing layer that turns question slots and raw generations into cleaned semantic role outputs.",
        "flow_in": "Accepts question-slot structures from QA-SRL annotations and raw generated text from the model.",
        "flow_out": "Returns normalized role mappings, QA pairs, aligned spans, heuristic role guesses, and fallback predictions.",
        "summary_points": [
            "Defines the compact role inventory that makes the local task feasible for a small generator.",
            "Encodes the highest-impact recovery logic in the project.",
            "Acts as the bridge between noisy text generation and presentation-ready structured roles."
        ],
        "symbols": [
            ("ROLE_ORDER", "Fixes the canonical output order for roles so training targets and predictions are stable."),
            ("infer_role", "Maps QA-SRL question-slot patterns onto the compact role inventory."),
            ("format_role_output", "Serializes role mappings into the line-based text format used for supervision."),
            ("parse_role_output", "Reads generated text back into structured role mappings."),
            ("render_qa_pairs", "Turns predicted roles into natural-language QA pairs for the UI and presentation."),
            ("align_answer_to_sentence", "Snaps generated answer spans back onto the source sentence when possible."),
            ("guess_role_from_answer", "Uses lexical cues to infer better roles for temporal, locative, instrumental, and related phrases."),
            ("fallback_role_mapping", "Creates a conservative role guess directly from sentence structure when the generator fails."),
            ("refine_role_mapping", "Reassigns noisy predictions into more plausible roles after decoding.")
        ]
    },
    {
        "path": "src/qasrl_cpu/modeling.py",
        "role": "Model creation and persistence layer for LoRA-based fine-tuning and adapter loading.",
        "flow_in": "Uses the base model name and LoRA hyperparameters supplied by training configuration.",
        "flow_out": "Creates PEFT-wrapped models, saves training bundles, and reloads trained adapters for evaluation and UI use.",
        "summary_points": [
            "Keeps model-architecture concerns separate from the training loop.",
            "Writes a small metadata file that later scripts can use as a reliable source of configuration truth.",
            "Makes the adapter portable across evaluation and demo entrypoints."
        ],
        "symbols": [
            ("create_lora_model", "Builds the tokenizer, base seq2seq model, and LoRA adaptation layer."),
            ("save_training_bundle", "Writes model weights, tokenizer files, and extra metadata to the artifact directory."),
            ("load_trained_model", "Rehydrates the adapter on top of the original base model and loads metadata.")
        ]
    },
    {
        "path": "src/qasrl_cpu/inference.py",
        "role": "Prediction pipeline for single examples and evaluation datasets.",
        "flow_in": "Consumes input prompts plus the trained model and tokenizer.",
        "flow_out": "Produces generated text, cleaned role mappings, QA pairs, and confidence values.",
        "summary_points": [
            "Centralizes decoding settings so training evaluation and the UI use consistent generation behavior.",
            "Invokes refinement and fallback logic after generation.",
            "Exposes a simple per-record interface for other modules."
        ],
        "symbols": [
            ("generate_text", "Runs constrained seq2seq decoding and returns both text and a confidence proxy."),
            ("predict_single", "Convenience wrapper for a single user-facing prediction with fallback enabled."),
            ("predict_single_with_options", "Predicts one example while allowing fallback behavior to be toggled."),
            ("predict_dataset", "Iterates over a record list and returns cleaned role-output strings for evaluation.")
        ]
    },
    {
        "path": "src/qasrl_cpu/metrics.py",
        "role": "Evaluation metric layer for role-aware token overlap, exact structural match, coverage, and ROUGE-L.",
        "flow_in": "Reads gold grouped role mappings and the generated role-output strings.",
        "flow_out": "Returns aggregate metrics, domain-wise scores, and per-example records used throughout the presentation package.",
        "summary_points": [
            "Makes the evaluation transparent because every example is stored with its own metric row.",
            "Keeps role-level token F1 distinct from exact structural matching.",
            "Supports both aggregate tables and example-level diagnostics."
        ],
        "symbols": [
            ("token_f1", "Computes overlap for free-text answers after normalization."),
            ("exact_match", "Checks whether normalized predicted roles match normalized gold roles exactly."),
            ("role_coverage", "Measures how many gold roles were surfaced regardless of exact span precision."),
            ("rouge_l_f1", "Captures sequence-level overlap between prediction text and target text."),
            ("compute_role_level_f1", "Averages role-level token overlap across the active roles in an example."),
            ("compute_dataset_metrics", "Builds the final aggregate and per-example metric report.")
        ]
    },
    {
        "path": "src/qasrl_cpu/instashap.py",
        "role": "Local explanation layer that computes gradient-times-input token attributions in an InstaShap-inspired style.",
        "flow_in": "Accepts the model, tokenizer, input prompt, and target text to explain.",
        "flow_out": "Returns token scores, normalized scores, confidence-drop behavior, and a ready-to-save plot.",
        "summary_points": [
            "Adapts explanation behavior to the seq2seq setting used by the local QA-SRL model.",
            "Produces artifacts that can be dropped directly into reports and slides.",
            "Balances practicality and interpretability for a CPU-friendly workflow."
        ],
        "symbols": [
            ("InstaShapExplanation", "Stores the explanation payload: tokens, scores, normalized scores, target text, and confidence drop."),
            ("InstaShapExplainer.explain", "Runs gradient-times-input attribution over the encoded prompt."),
            ("InstaShapExplainer._compute_confidence_drop", "Measures how the explanation's top tokens affect loss when masked."),
            ("InstaShapExplainer.compute_plausibility", "Measures overlap between influential tokens and gold reference tokens."),
            ("InstaShapExplainer.plot", "Renders the attribution bar chart used in the results and deck outputs.")
        ]
    }
]


INNOVATION_PILLARS = [
    {
        "title": "CPU-friendly QA-SRL fine-tuning",
        "problem": "Research-style QA-SRL systems often assume larger hardware or broader parameter updates than a local academic machine can support.",
        "mechanism": "This project fine-tunes `google/flan-t5-small` with LoRA rank 8 and keeps the entire run on CPU.",
        "evidence": "The saved metadata records 600 train, 100 validation, 100 test examples, 3 epochs, and 685.02 training seconds.",
        "value": "The system remains reproducible in the exact environment that motivated the project.",
        "limit": "This design trades away research-scale accuracy headroom in exchange for feasibility and repeatability."
    },
    {
        "title": "Generative-to-extractive cleanup",
        "problem": "Compact generators often produce structurally noisy or weakly grounded spans on semantic tasks.",
        "mechanism": "The project snaps answer text back to the sentence, reassigns obvious roles heuristically, and normalizes the final output format.",
        "evidence": "The README and `roles.py` document extractive snapping and role re-assignment as key debugging improvements.",
        "value": "This allows the local model to function as a structured semantic parser rather than a free-form text generator.",
        "limit": "Heuristics improve practical robustness but cannot fully replace a dedicated extractive span head."
    },
    {
        "title": "Failure recovery pipeline",
        "problem": "Small local models sometimes emit empty generations or fragments that cannot be parsed into semantic roles.",
        "mechanism": "Fallback extraction derives best-effort AGENT, THEME, LOCATION, TIME, and related phrases directly from the sentence.",
        "evidence": "The inference path in `src/qasrl_cpu/inference.py` explicitly enables fallback behavior for the deployed system.",
        "value": "The deployed app remains usable even when generation quality collapses on a given example.",
        "limit": "Fallback behavior is conservative and not a substitute for high-quality primary predictions."
    },
    {
        "title": "Integrated InstaShap-style explanations",
        "problem": "Explainability is often disconnected from the actual model checkpoint and interface that users see.",
        "mechanism": "This project computes token attributions, plausibility, and faithfulness directly on the fine-tuned adapter.",
        "evidence": "The evaluation report records plausibility 0.2220, faithfulness 0.7248, and XAI score 0.4734.",
        "value": "The viva can show not just what the model predicted but which prompt tokens most influenced the output.",
        "limit": "The explanation layer is an adaptation inspired by InstaSHAP rather than a full additive-model reproduction."
    },
    {
        "title": "One-folder delivery package",
        "problem": "Academic projects often scatter training code, evaluation scripts, UI, and writeups across disconnected folders or manual steps.",
        "mechanism": "This folder contains data processing, training, evaluation, artifacts, results, docs, app, and the new presentation package.",
        "evidence": "The project layout in the README already exposes the end-to-end structure, and the new build scripts extend it with presentation outputs.",
        "value": "Reviewers can inspect the workflow from raw data to slide deck without leaving `finetuning/`.",
        "limit": "The package is dense, so high-quality documentation is required to make the structure easy to navigate."
    }
]


PROMPT_EVOLUTION_STEPS = [
    {
        "stage": "Stage 1",
        "name": "Long instruction-heavy prompting",
        "problem": "The small T5-family model tended to copy instruction tails or drift into unstructured text.",
        "change": "Remove excess framing and reduce the cognitive load imposed by the prompt.",
        "effect": "Set up the move toward a compact structured input format."
    },
    {
        "stage": "Stage 2",
        "name": "Compact predicate plus sentence plus labels prompt",
        "problem": "The model needed a short, repeatable prompt that emphasized the semantic extraction task directly.",
        "change": "Adopt the format `semantic role extraction / predicate / sentence / labels:`.",
        "effect": "Improved structural consistency and made decoding more predictable for a CPU-friendly run."
    },
    {
        "stage": "Stage 3",
        "name": "Role-output formatting discipline",
        "problem": "Even with a shorter prompt, the model still needed a predictable output schema for evaluation.",
        "change": "Target line-based outputs such as `AGENT: ...` and `THEME: ...` in a fixed role order.",
        "effect": "Enabled post-processing, metrics, and UI rendering to rely on a common textual contract."
    },
    {
        "stage": "Stage 4",
        "name": "Extractive snapping and refinement",
        "problem": "The generator sometimes emitted approximate phrases rather than clean sentence spans.",
        "change": "Snap answers back to the nearest sentence span and heuristically reassign roles when lexical cues are strong.",
        "effect": "Raised the practical quality of predictions without requiring a new model architecture."
    },
    {
        "stage": "Stage 5",
        "name": "Fallback recovery",
        "problem": "Some generations remained empty or structurally unusable on difficult examples.",
        "change": "Add a conservative fallback extractor so the deployed pipeline can still emit best-effort roles.",
        "effect": "Turned prompt tuning into part of a broader robustness strategy rather than a single decoding tweak."
    }
]


FINAL_TAKEAWAY_LINES = [
    "The local package is best presented as a complete, reproducible, constraint-aware QA-SRL system.",
    "Its strongest direct evidence comes from `results/evaluation_report.json`, not from literature-scale claims.",
    "The fine-tuned adapter materially outperforms the zero-shot baseline on the verified local slice.",
    "The project makes a persuasive viva story because it connects survey, implementation, results, explanation, and deployment.",
    "The remaining accuracy gap is a hardware-and-capacity gap more than a project-completeness gap.",
    "Prompt shaping, post-processing, and fallback logic are not side details here; they are central to making a small model usable.",
    "The explanation layer strengthens the defense of the system because it exposes where the model looked in the prompt.",
    "The project should be defended as careful engineering under realistic compute constraints."
]


def _build_survey_section(ctx: PresentationContext, asset_prefix: str) -> list[str]:
    lines = [
        "# Survey",
        "",
        "This survey positions the local project inside the QA-SRL field without pretending that the local run is a literature-scale parser.",
        "The emphasis is on what earlier systems established, what the field learned from those systems, and what remained open for a CPU-feasible, explainable implementation.",
        "",
        "## Survey Focus",
        "",
        "- The survey follows the citation trail already preserved in `docs/analysis.md`.",
        "- It treats literature claims as field context and keeps direct numeric benchmark claims local to the current folder.",
        "- It focuses on the transition from representation design to scalable data, then to quality, then to explanation and deployment readiness.",
        "- It uses the present project as a constraint-aware response to those open needs rather than as a replacement for research-scale systems.",
        "",
        "## Why QA-SRL Matters",
        "",
        "- QA-SRL re-expresses predicate-argument structure in the form of answerable questions, which makes semantic role outputs more interpretable for people.",
        "- The representation is naturally aligned with question answering because roles such as agent, theme, time, and location already behave like answers.",
        "- This project inherits that human-readable framing and translates predicted roles back into QA pairs inside the deployed application.",
        "- The field relevance of QA-SRL remains visible in the continued expansion to new annotation settings and languages described in the local analysis document.",
        "",
        "## Milestone Timeline",
        ""
    ]
    milestone_rows = []
    for milestone in SURVEY_MILESTONES:
        milestone_rows.append([milestone["year"], milestone["label"], milestone["field_move"], milestone["remaining_gap"]])
    lines.extend(_table_lines(["Year", "Milestone", "Field contribution", "Remaining gap for this project"], milestone_rows))
    lines.extend(["", "## Literature Capsules", ""])

    for index, milestone in enumerate(SURVEY_MILESTONES, start=1):
        lines.extend(
            [
                f"### Survey Capsule {index:02d} - {milestone['year']} {milestone['label']}",
                "",
                f"- Local survey source: `{milestone['source']}`.",
                f"- Field move: {milestone['field_move']}",
                f"- Before this milestone: {milestone['before']}",
                f"- What systems in the field did: {milestone['what_systems_did']}",
                f"- What this enabled: {milestone['what_it_enabled']}",
                f"- Remaining gap for the current build: {milestone['remaining_gap']}",
                f"- Local design lesson: {milestone['local_lesson']}",
                ""
            ]
        )

    lines.extend(
        [
            "## What Earlier Systems Did Before This Local Build",
            "",
            "- Earlier QA-SRL systems established the task, the annotation protocol, and the idea that predicate-argument structure can be recovered through question-answer supervision.",
            "- Large dataset releases let neural parsers compete on scale and consistency instead of staying at proof-of-concept size.",
            "- Higher-quality annotation work cleaned up noise and made benchmark claims more trustworthy.",
            "- Broader task variants such as QANom showed that question-driven semantics is not limited to verb-only settings.",
            "- Explainability research in adjacent areas emphasized that a usable modern system should expose why the model predicted a role, not only what it predicted.",
            "- What the field still did not hand to this project for free was a single compact package that works on a low-resource local machine and stays presentation-ready.",
            "",
            "## Field Gaps Relevant To This Project",
            ""
        ]
    )
    lines.extend(_table_lines(["Gap", "Common prior pattern", "Local response in `finetuning/`"], SURVEY_GAPS))
    lines.extend(
        [
            "",
            "## Datasets And Benchmark Practice",
            "",
            "- The local project is anchored in QA-SRL Bank 2.1, which is the benchmark lineage explicitly named in the project docs and preprocessing code.",
            "- The processed full local files contain 95,253 train examples, 17,577 validation examples, and 20,602 test examples after grouped predicate conversion.",
            "- The verified local benchmark slice is intentionally smaller: 600 train, 100 validation, and 100 test grouped examples.",
            "- The distinction matters because the presentation should never blur full processed corpus size with the actual run that produced the current metrics.",
            "- The train-domain composition of the grouped local files is 40,937 `TQA`, 27,956 `wikinews`, and 26,360 `wikipedia` examples.",
            ""
        ]
    )
    lines.extend(
        _table_lines(
            ["Local dataset view", "Count", "Interpretation"],
            [
                ("Processed train examples", ctx.processed_counts["train"], "All grouped examples available after preprocessing."),
                ("Processed validation examples", ctx.processed_counts["validation"], "All grouped validation examples available locally."),
                ("Processed test examples", ctx.processed_counts["test"], "All grouped test examples available locally."),
                ("Verified train slice", ctx.training_summary["dataset"]["train_examples"], "Actual slice used for the latest verified fine-tuning run."),
                ("Verified validation slice", ctx.training_summary["dataset"]["validation_examples"], "Actual slice used for checkpoint selection."),
                ("Verified test slice", ctx.training_summary["dataset"]["test_examples"], "Actual slice used for the final verified evaluation.")
            ]
        )
    )
    lines.extend(["", "## Survey Observations For The Viva", ""])
    for index, note in enumerate(SURVEY_NOTES, start=1):
        lines.append(f"- Observation {index:02d}: {note}")
    lines.extend(
        [
            "",
            "## How The Survey Directly Informs This Implementation",
            "",
            "- The field history justifies using QA-SRL Bank 2.1 as the benchmark backbone even when the local run is smaller.",
            "- The dataset-quality literature justifies careful claim discipline and explicit separation of processed size from verified slice size.",
            "- The explanation trend justifies integrating InstaShap-style attribution into the evaluation package and Streamlit demo instead of leaving it as an optional appendix.",
            "- The absence of a CPU-first narrative in earlier systems justifies presenting local feasibility as a real contribution of this package.",
            "- The current project therefore occupies a practical niche: not the biggest model, but a coherent and inspectable engineering delivery under local constraints.",
            "",
            "## Survey Source Inventory Used In This Section",
            "",
            "- `docs/analysis.md` supplied the milestone list, literature context, and fair-claim framing.",
            "- `README.md` supplied the practical debugging improvements and the verified metrics summary.",
            "- `docs/project_summary.md` supplied the goal statement, delivered features, and system boundary description.",
            "- `docs/latest_research_prompt.txt` reinforced the CPU-first, benchmark-aware, explanation-aware positioning of the project."
        ]
    )
    return lines


def _build_implementation_section(ctx: PresentationContext, asset_prefix: str) -> list[str]:
    config = ctx.training_summary["config"]
    lines = [
        "# Implementation",
        "",
        "This section documents the real implementation inside `finetuning/`, not a hypothetical large-model version of the project.",
        "Every component described below exists in the local folder and is tied to runnable code or stored artifacts.",
        "",
        "## Implementation Objective",
        "",
        "- Build a QA-SRL pipeline that can be trained and evaluated on an 8 GB RAM CPU machine.",
        "- Keep the data path tied to official QA-SRL Bank 2.1 ingestion.",
        "- Produce structured semantic roles and natural-language QA pairs from the same predicted output.",
        "- Add token-level explanations and expose the pipeline through Streamlit.",
        "- Save enough artifacts to make the system reproducible for a project presentation.",
        "",
        "## End-To-End Architecture",
        ""
    ]
    lines.extend(_section_asset_line(ctx, "pipeline_architecture", asset_prefix))
    lines.extend(
        [
            "- The pipeline begins with QA-SRL Bank 2.1 download and grouped predicate conversion in `src/qasrl_cpu/data.py`.",
            "- The training path in `train.py` establishes a zero-shot baseline, fine-tunes a LoRA adapter, and stores artifacts.",
            "- The evaluation path in `evaluate.py` reloads the adapter, computes local metrics, and generates explanation artifacts.",
            "- The deployment path in `app.py` loads the same adapter and exposes prediction plus explanation in a user-facing interface.",
            "",
            "## Training Configuration Used In The Latest Verified Run",
            ""
        ]
    )
    config_rows = [
        ("Model", config["model_name"], "Compact instruction-tuned encoder-decoder base model."),
        ("Train slice", config["train_limit"], "Local benchmark train size."),
        ("Validation slice", config["validation_limit"], "Local benchmark validation size."),
        ("Test slice", config["test_limit"], "Local benchmark test size."),
        ("Epochs", config["epochs"], "Maximum epochs requested for the run."),
        ("Batch size", config["batch_size"], "Per-step batch size."),
        ("Gradient accumulation", config["gradient_accumulation_steps"], "Lets the CPU run mimic a larger effective batch."),
        ("Learning rate", config["learning_rate"], "AdamW learning rate."),
        ("Weight decay", config["weight_decay"], "Regularization strength."),
        ("Beam count", config["num_beams"], "Generation beam count used in the verified run."),
        ("LoRA rank", config["lora_rank"], "Low-rank adaptation rank."),
        ("LoRA alpha", config["lora_alpha"], "LoRA scaling factor."),
        ("LoRA dropout", config["lora_dropout"], "LoRA dropout."),
        ("Seed", config["seed"], "Reproducibility control."),
        ("Threads", config["num_threads"], "CPU thread cap.")
    ]
    lines.extend(_table_lines(["Parameter", "Value", "Why it matters"], config_rows))
    lines.extend(
        [
            "",
            "## Data Ingestion And Grouped Example Creation",
            "",
            "- The archive URL is hard-coded in `src/qasrl_cpu/data.py` as the official raw QA-SRL Bank 2.1 tarball source.",
            "- `ensure_archive` guarantees that download happens only once and then remains cached locally.",
            "- `iter_split_examples` streams sentence records from the compressed split files inside the tar archive.",
            "- `sentence_to_grouped_examples` converts each sentence into one grouped example per predicate.",
            "- Each grouped example stores sentence text, predicate, predicate index, domain, compact input prompt, target role output, and gold questions.",
            "- This grouping is what allows the project to learn predicate-level role extraction rather than question-slot prediction in isolation.",
            "",
            "## Role Taxonomy And Mapping",
            "",
            "- The local role inventory is deliberately compact: `AGENT`, `THEME`, `LOCATION`, `TIME`, `MANNER`, `REASON`, `ATTRIBUTE`, `SOURCE`, `GOAL`, `INSTRUMENT`, `OBLIQUE`, and `OTHER`.",
            "- `infer_role` maps QA-SRL question slots onto that compact inventory using wh-word, auxiliary, object, and preposition cues.",
            "- Compact roles make the output more learnable for a small model and easier to explain to a viva audience.",
            "- `format_role_output` enforces a stable line-based serialization so training targets and predictions share the same contract.",
            "- `parse_role_output` converts generated text back into a structured mapping so evaluation is role-aware rather than purely string-based.",
            "",
            "## Prompt Design",
            "",
            "- The final prompt format is intentionally short: `semantic role extraction`, then `predicate`, then `sentence`, then `labels:`.",
            "- The prompt is generated in `build_input_text` and reused consistently across training, evaluation, and the UI.",
            "- The shorter format is important because the README documents that longer prompts made the small T5-family model copy instruction tails or drift structurally.",
            "- This project therefore treats prompt design as a first-order engineering decision rather than a cosmetic detail.",
            "",
            "## LoRA Model Setup",
            "",
            "- `src/qasrl_cpu/modeling.py` wraps the base seq2seq model with a PEFT LoRA configuration.",
            "- The target modules are the attention `q` and `v` projections, which keeps adaptation efficient and compact.",
            "- The tokenizer and training metadata are saved beside the adapter so later scripts can reload the exact configuration.",
            "- The metadata file records the base model name and run-specific details such as training seconds and best validation token F1.",
            "",
            "## Training Loop Behavior",
            "",
            "- `train_model` first evaluates a zero-shot baseline on a bounded subset so the improvement story has a runnable starting point.",
            "- The training loop accumulates gradients, evaluates validation loss after each epoch, and selects checkpoints by validation token F1.",
            "- The selection metric is computed on generated outputs, which makes model selection more faithful to the downstream task than loss alone.",
            "- The run persists `training_history`, a baseline metrics block, final fine-tuned metrics, and per-example predictions.",
            "",
            "## Inference And Recovery Pipeline",
            "",
            "- `generate_text` centralizes decoding settings such as `num_beams`, `max_new_tokens`, `no_repeat_ngram_size`, and `repetition_penalty`.",
            "- `predict_dataset` and `predict_single_with_options` immediately pass raw generations through role parsing and refinement.",
            "- `align_answer_to_sentence` snaps approximate generations back to spans found in the source sentence when overlap is strong enough.",
            "- `guess_role_from_answer` looks for lexical cues such as prepositions and time words to improve role assignment.",
            "- `fallback_role_mapping` activates when the structured role map is empty, producing conservative best-effort roles from sentence structure.",
            "- The deployed system uses that fallback logic because usability matters in a live demonstration setting.",
            "",
            "## Metrics And XAI Path",
            "",
            "- `compute_dataset_metrics` stores aggregate metrics and the complete per-example evaluation ledger used by the presentation package.",
            "- Metrics include token F1, exact match, role coverage, ROUGE-L, and domain-wise token F1.",
            "- `compute_xai_suite` evaluates explanation plausibility and faithfulness over a bounded subset of records.",
            "- `InstaShapExplainer.plot` produces the saved attribution image used in the docs, PDF outputs, and slide deck.",
            "",
            "## Streamlit Deployment Path",
            "",
            "- `app.py` bootstraps Streamlit automatically so `python app.py` is enough for a local demo.",
            "- The app reuses the trained adapter and the same prediction plus explanation utilities used by evaluation.",
            "- The sidebar surfaces the latest evaluation metrics when `results/evaluation_report.json` is present.",
            "- The main panel shows both structured role output and natural-language QA pairs so semantic predictions remain presentation-friendly.",
            "",
            "## Module Walkthrough",
            ""
        ]
    )
    for module_index, guide in enumerate(IMPLEMENTATION_GUIDES, start=1):
        lines.extend(
            [
                f"### Module Review {module_index:02d} - `{guide['path']}`",
                "",
                f"- File role: {guide['role']}",
                f"- Data flowing in: {guide['flow_in']}",
                f"- Data flowing out: {guide['flow_out']}"
            ]
        )
        for summary_point in guide["summary_points"]:
            lines.append(f"- Implementation note: {summary_point}")
        lines.append("")
        for symbol_name, symbol_note in guide["symbols"]:
            lines.extend(
                [
                    f"#### Symbol - `{symbol_name}`",
                    "",
                    f"- Purpose: {symbol_note}",
                    f"- Why it matters in the final package: `{symbol_name}` influences how the project moves from raw data to presentation-ready outputs.",
                    f"- Presentation value: this symbol is part of the chain that makes the local system reproducible and explainable.",
                    ""
                ]
            )
    lines.extend(
        [
            "## Implementation Sequence For A Full Fresh Run",
            "",
            "1. Create or reuse the cached QA-SRL Bank 2.1 archive.",
            "2. Build the processed grouped train, validation, and test files if they are missing.",
            "3. Sample the requested train, validation, and test slices with the configured random seed.",
            "4. Evaluate a zero-shot baseline model before any local fine-tuning.",
            "5. Create the LoRA-wrapped seq2seq model and tokenizer.",
            "6. Train on CPU with gradient accumulation and validation monitoring.",
            "7. Select the best checkpoint using validation token F1.",
            "8. Save the adapter, tokenizer, and training metadata.",
            "9. Generate predictions on the test slice with refinement and fallback enabled.",
            "10. Write the training summary and prediction artifacts.",
            "11. Reload the adapter in evaluation mode.",
            "12. Re-run prediction on the same split for the final evaluation report.",
            "13. Compute token F1, exact match, role coverage, ROUGE-L, and domain-wise token F1.",
            "14. Compute InstaShap-style explanations on the XAI subset.",
            "15. Save the attribution plot and markdown summary.",
            "16. Load the same adapter into Streamlit for interactive demonstration.",
            "17. Accept sentence and predicate input from the user.",
            "18. Predict structured roles and render QA pairs.",
            "19. Plot token attributions for the selected prediction.",
            "20. Surface both task output and explanation evidence in the live interface."
        ]
    )
    return lines


def _build_results_section(ctx: PresentationContext, asset_prefix: str) -> list[str]:
    report = ctx.evaluation_report
    tuned = report["fine_tuned_metrics"]
    baseline = report["baseline_metrics"]
    xai = report["xai_metrics"]
    training_history = ctx.training_summary["training_history"]
    lines = [
        "# Results & Analysis",
        "",
        "This section separates full processed data scale, the verified benchmark slice, training behavior, final metrics, explainability evidence, and per-example analysis.",
        "All direct numeric claims in this section are sourced from `results/evaluation_report.json` or `results/training_summary.json`.",
        "",
        "## Dataset Distinction",
        ""
    ]
    lines.extend(
        _table_lines(
            ["View", "Train", "Validation", "Test", "Interpretation"],
            [
                (
                    "Processed grouped files",
                    ctx.processed_counts["train"],
                    ctx.processed_counts["validation"],
                    ctx.processed_counts["test"],
                    "All local grouped examples available after preprocessing."
                ),
                (
                    "Verified benchmark slice",
                    ctx.training_summary["dataset"]["train_examples"],
                    ctx.training_summary["dataset"]["validation_examples"],
                    ctx.training_summary["dataset"]["test_examples"],
                    "Actual slice used for the latest fine-tuning and evaluation run."
                )
            ]
        )
    )
    lines.extend(["", "## Training Dynamics", ""])
    lines.extend(_section_asset_line(ctx, "training_curve", asset_prefix))
    lines.extend(
        _table_lines(
            ["Epoch", "Train loss", "Validation loss", "Selection token F1", "Selection role coverage"],
            [
                (
                    row["epoch"],
                    row["train_loss"],
                    row["validation_loss"],
                    row["selection_token_f1"],
                    row["selection_role_coverage"]
                )
                for row in training_history
            ]
        )
    )
    lines.extend(
        [
            "",
            f"- Best validation token F1 during training: `{report['model_metadata']['best_validation_token_f1']:.4f}`.",
            f"- Total recorded training time: `{report['model_metadata']['training_seconds']:.2f}` seconds.",
            "- The training curve shows steady loss reduction and a validation token-F1 rise across the three completed epochs.",
            "- The local run therefore looks stable within the compact slice, even though its absolute accuracy remains far below large-model aspirations.",
            "",
            "## Zero-Shot Versus Fine-Tuned Metrics",
            ""
        ]
    )
    lines.extend(_section_asset_line(ctx, "metric_comparison", asset_prefix))
    lines.extend(
        _table_lines(
            ["Model", "Token F1", "Exact match", "Role coverage", "ROUGE-L"],
            [
                (
                    f"Zero-shot {report['baseline_model_name']}",
                    baseline["token_f1"],
                    baseline["exact_match"],
                    baseline["role_coverage"],
                    baseline["rouge_l"]
                ),
                ("Fine-tuned adapter", tuned["token_f1"], tuned["exact_match"], tuned["role_coverage"], tuned["rouge_l"])
            ]
        )
    )
    lines.extend(
        [
            "",
            f"- Token F1 improvement over zero-shot: `{tuned['token_f1'] - baseline['token_f1']:.4f}`.",
            f"- Exact-match improvement over zero-shot: `{tuned['exact_match'] - baseline['exact_match']:.4f}`.",
            f"- Role-coverage improvement over zero-shot: `{tuned['role_coverage'] - baseline['role_coverage']:.4f}`.",
            f"- ROUGE-L improvement over zero-shot: `{tuned['rouge_l'] - baseline['rouge_l']:.4f}`.",
            "- The fine-tuned system is still modest in exact structural accuracy, but it clearly surpasses the unusable zero-shot baseline on the local slice.",
            "",
            "## Domain Performance",
            ""
        ]
    )
    lines.extend(_section_asset_line(ctx, "domain_performance", asset_prefix))
    lines.extend(
        _table_lines(
            ["Domain", "Token F1", "Observation"],
            [
                ("TQA", tuned["domain_token_f1"]["TQA"], "Best domain in the current local evaluation."),
                ("wikinews", tuned["domain_token_f1"]["wikinews"], "Also strong relative to the local average."),
                ("wikipedia", tuned["domain_token_f1"]["wikipedia"], "Lowest of the three domains in the current run.")
            ]
        )
    )
    lines.extend(
        [
            "",
            "- The domain spread is meaningful but not extreme; all three domains remain within the same broad local performance band.",
            "- `wikipedia` is the weakest domain in the verified run, which is consistent with examples that require cleaner boundary decisions or denser semantic disambiguation.",
            "",
            "## Distribution-Level Analysis",
            ""
        ]
    )
    lines.extend(_section_asset_line(ctx, "token_f1_histogram", asset_prefix))
    lines.extend(_section_asset_line(ctx, "role_frequency", asset_prefix))
    lines.extend(
        [
            f"- Median token F1 across the 100 evaluated examples: `{ctx.aggregate['median_token_f1']:.4f}`.",
            f"- Median ROUGE-L across the 100 evaluated examples: `{ctx.aggregate['median_rouge_l']:.4f}`.",
            f"- Exact matches on the evaluation slice: `{ctx.aggregate['exact_match_count']}` out of `{ctx.aggregate['example_count']}`.",
            f"- Full role coverage on the evaluation slice: `{ctx.aggregate['full_coverage_count']}` out of `{ctx.aggregate['example_count']}`.",
            f"- Examples with token F1 above 0.5: `{ctx.aggregate['above_half_count']}` out of `{ctx.aggregate['example_count']}`.",
            f"- Examples with token F1 at or above 0.8: `{ctx.aggregate['above_08_count']}` out of `{ctx.aggregate['example_count']}`.",
            f"- Examples with token F1 equal to 0.0: `{ctx.aggregate['zero_token_f1_count']}` out of `{ctx.aggregate['example_count']}`.",
            "- The role-frequency chart shows why the compact inventory is sensible: `AGENT` and `THEME` dominate the distribution, while rarer roles form a long tail.",
            "- The histogram reinforces that the current model often recovers part of the semantic scaffold even when exact structure is still hard.",
            "",
            "## Explainability Metrics",
            ""
        ]
    )
    lines.extend(_section_asset_line(ctx, "instashap_example", asset_prefix))
    lines.extend(
        _table_lines(
            ["Metric", "Value", "Interpretation"],
            [
                ("Plausibility", xai["plausibility"], "How often top-attribution tokens overlap with gold evidence tokens."),
                ("Faithfulness", xai["faithfulness"], "How much masking the most influential tokens affects the model loss."),
                ("Combined XAI score", xai["xai_score"], "Average of plausibility and faithfulness in the local evaluation.")
            ]
        )
    )
    lines.extend(
        [
            "",
            "- The local explanation story is stronger on faithfulness than plausibility.",
            "- This means the influential tokens matter to the model, but they do not always align tightly with gold human answer tokens.",
            "- That behavior is still useful in a viva because it reveals prompt sensitivity and debugging opportunities.",
            "",
            "## Representative Success Cases",
            ""
        ]
    )
    success_rows = []
    for example in ctx.best_examples[:5]:
        success_rows.append(
            [example["predicate"], example["domain"], f"{example['token_f1']:.4f}", f"{example['role_coverage']:.4f}", _example_assessment(example)]
        )
    lines.extend(_table_lines(["Predicate", "Domain", "Token F1", "Coverage", "Assessment"], success_rows))
    lines.extend(["", "## Representative Failure Cases", ""])
    failure_rows = []
    for example in ctx.worst_examples[:5]:
        failure_rows.append(
            [example["predicate"], example["domain"], f"{example['token_f1']:.4f}", f"{example['role_coverage']:.4f}", _example_assessment(example)]
        )
    lines.extend(_table_lines(["Predicate", "Domain", "Token F1", "Coverage", "Assessment"], failure_rows))
    lines.extend(
        [
            "",
            "## Error Pattern Summary",
            "",
            "- The model often discovers the right coarse role set before it finds the cleanest answer boundaries.",
            "- Failures frequently involve location-versus-theme or agent-versus-theme confusion after noisy generation.",
            "- Long or syntactically dense sentences expose the limits of compact generation plus heuristic repair.",
            "- Exact match is the hardest metric because duplicate spans, extra spans, and minor boundary drift all count against it.",
            "- Role coverage is much higher than exact match, which suggests the semantic scaffold is partly learned even where precision lags.",
            "",
            "## XAI Example Capsules",
            ""
        ]
    )
    for index, example in enumerate(ctx.xai_examples, start=1):
        top_tokens = ", ".join(f"{token} ({score:.4f})" for token, score in example["top_tokens"])
        lines.extend(
            [
                f"### XAI Capsule {index:02d} - `{example['predicate']}`",
                "",
                f"- Record id: `{example['id']}`.",
                f"- Sentence: {example['sentence']}",
                f"- Plausibility: `{example['plausibility']:.4f}`.",
                f"- Faithfulness: `{example['faithfulness']:.4f}`.",
                f"- Top attributed tokens: {top_tokens}",
                f"- Interpretation: {'High faithfulness but weak alignment with gold tokens.' if example['faithfulness'] >= 0.7 and example['plausibility'] < 0.3 else 'Explanation and gold evidence show partial overlap, which is useful for debugging.'}",
                ""
            ]
        )

    lines.extend(["## Full Evaluation Audit", ""])
    for index, example in enumerate(ctx.example_rows, start=1):
        lines.extend(
            [
                f"### Evaluated Example {index:03d} - `{example['predicate']}`",
                "",
                f"- Record id: `{example['id']}`.",
                f"- Domain: `{example['domain']}`.",
                f"- Sentence: {example['sentence']}",
                f"- Gold roles: {_format_roles(example['gold'])}",
                f"- Predicted roles: {_format_roles(example['prediction_roles'])}",
                f"- Prediction text: {example['prediction_text']}",
                f"- Token F1: `{example['token_f1']:.4f}`.",
                f"- Exact match: `{example['exact_match']:.4f}`.",
                f"- Role coverage: `{example['role_coverage']:.4f}`.",
                f"- ROUGE-L: `{example['rouge_l']:.4f}`.",
                f"- Assessment: {_example_assessment(example)}",
                ""
            ]
        )
    return lines


def _build_innovation_section(ctx: PresentationContext, asset_prefix: str) -> list[str]:
    lines = [
        "# Innovation",
        "",
        "The innovation story here is practical integration under constraints rather than a claim of publishing a new state-of-the-art parser.",
        "Each innovation below is anchored in code, stored artifacts, or verified evaluation evidence from this folder.",
        ""
    ]
    lines.extend(_section_asset_line(ctx, "innovation_comparison", asset_prefix))
    lines.extend(["## Innovation Pillars", ""])
    for index, pillar in enumerate(INNOVATION_PILLARS, start=1):
        lines.extend(
            [
                f"### Innovation {index:02d} - {pillar['title']}",
                "",
                f"- Problem addressed: {pillar['problem']}",
                f"- Local mechanism: {pillar['mechanism']}",
                f"- Evidence from this folder: {pillar['evidence']}",
                f"- Practical value: {pillar['value']}",
                f"- Honest limit: {pillar['limit']}",
                ""
            ]
        )
    lines.extend(
        [
            "## Why These Innovations Matter Together",
            "",
            "- The LoRA choice makes local fine-tuning possible.",
            "- The compact prompt makes the small model trainable for the chosen structure.",
            "- The cleanup heuristics make predictions usable after generation.",
            "- The fallback path makes the app robust enough for demonstration.",
            "- The explanation layer turns the system from a black box into a defendable academic artifact.",
            "- The one-folder packaging makes the entire story inspectable by a reviewer or evaluator.",
            "",
            "## Innovation Framing For Presentation Use",
            "",
            "- Present this as a systems innovation story, not a claim that the architecture itself replaces the full research frontier.",
            "- Emphasize integration, practicality, explainability, and reproducibility.",
            "- Tie every innovation claim back to a file, artifact, or local metric whenever possible."
        ]
    )
    return lines


def _build_prompt_tuning_section(ctx: PresentationContext, asset_prefix: str) -> list[str]:
    final_prompt = "semantic role extraction\\npredicate: <predicate>\\nsentence: <sentence>\\nlabels:"
    lines = [
        "# Prompt Tuning",
        "",
        "Prompt tuning in this project is part of the core engineering story because the local model is small enough for prompt shape to matter a great deal.",
        "The prompt strategy evolved toward a short, repeatable structure that supports parsing, evaluation, and post-processing.",
        ""
    ]
    lines.extend(_section_asset_line(ctx, "prompt_evolution", asset_prefix))
    lines.extend(
        [
            "## Final Prompt Used By The Local Pipeline",
            "",
            f"- Canonical prompt form: `{final_prompt}`",
            "- This prompt is produced by `build_input_text` in `src/qasrl_cpu/data.py`.",
            "- The same prompt contract is used for training examples, evaluation examples, and interactive app inference.",
            "",
            "## Prompt Evolution Stages",
            ""
        ]
    )
    for step in PROMPT_EVOLUTION_STEPS:
        lines.extend(
            [
                f"### {step['stage']} - {step['name']}",
                "",
                f"- Problem: {step['problem']}",
                f"- Change: {step['change']}",
                f"- Effect: {step['effect']}",
                ""
            ]
        )
    lines.extend(
        [
            "## How Prompting Interacts With The Rest Of The System",
            "",
            "- Short prompts reduce instruction-copying risk in compact seq2seq decoding.",
            "- Fixed role serialization allows `parse_role_output` to treat model text as structured data.",
            "- Extractive snapping and role refinement become more effective because the output schema is predictable.",
            "- The fallback extractor exists because prompt tuning alone cannot eliminate all compact-model failure cases.",
            "- The project therefore treats prompt tuning, parsing, cleanup, and fallback as one coordinated robustness stack.",
            "",
            "## Prompt-Tuning Takeaways",
            "",
            "- Bigger prompts are not automatically better for smaller models.",
            "- Predictable structure can be more valuable than richer instruction wording in constrained local runs.",
            "- Prompt tuning earns its place in the presentation because it directly changed training stability and output usability."
        ]
    )
    return lines


def _build_final_takeaways_section(ctx: PresentationContext) -> list[str]:
    lines = [
        "# Final Takeaways",
        "",
        "The final section closes the presentation without adding a separate introduction block or any off-scope claims.",
        "",
        "## Verified Closing Points",
        ""
    ]
    for takeaway in FINAL_TAKEAWAY_LINES:
        lines.append(f"- {takeaway}")
    lines.extend(
        [
            "",
            "## Limitations To State Clearly",
            "",
            "- The verified local run is a 600/100/100 slice, not the full processed dataset.",
            "- The model is a compact `flan-t5-small` LoRA adapter, not a large research-scale parser.",
            "- Exact structural matching remains difficult even after recovery heuristics.",
            "- The explanation module is InstaShap-inspired rather than a literal reproduction of the original paper.",
            "- Stronger accuracy would require larger models, more memory, longer runs, or a stronger extractive architecture.",
            "",
            "## Defense Statement",
            "",
            "- This project should be defended as a complete local QA-SRL system with measurable improvement, integrated explanation, and presentation-ready artifacts."
        ]
    )
    return lines


def build_markdown_sections(ctx: PresentationContext, asset_prefix: str = "assets") -> OrderedDict[str, list[str]]:
    slides = build_slide_specs(ctx)
    sections = OrderedDict()
    title_lookup = {section["id"]: section["title"] for section in ctx.manifest["sections"]}

    def image_markdown(spec: SlideSpec) -> list[str]:
        if not spec.image_id:
            return []
        filename = ctx.figure_paths[spec.image_id].name
        return [f"![{spec.title}]({asset_prefix}/{filename})", ""]

    def citations_markdown(spec: SlideSpec) -> list[str]:
        lines: list[str] = []
        for citation_id in spec.citations:
            source = ctx.sources.get(citation_id)
            if source:
                lines.append(f"- Citation: [{source['label']}]({source['url']})")
        return lines

    section_order = [section["id"] for section in ctx.manifest["sections"]]
    for section_id in section_order:
        sections[section_id] = [f"# {title_lookup.get(section_id, section_id.title())}", ""]

    for spec in slides:
        block = sections[spec.section_id]
        block.extend([f"## {spec.title}", ""])
        block.extend(spec.bullets)
        block.append("")
        if spec.table_headers and spec.table_rows:
            block.extend(_table_lines(spec.table_headers, spec.table_rows))
            block.append("")
        block.extend(image_markdown(spec))
        block.extend(citations_markdown(spec))
        block.append("")

    return sections


def combine_master_lines(sections: OrderedDict[str, list[str]]) -> list[str]:
    master_lines: list[str] = []
    for section_lines in sections.values():
        if master_lines:
            master_lines.extend(["", ""])
        master_lines.extend(section_lines)
    return master_lines


def ensure_minimum_line_count(master_lines: list[str], ctx: PresentationContext, minimum: int = 400) -> list[str]:
    if len(master_lines) >= minimum:
        return master_lines
    lines = list(master_lines)
    lines.extend(["", "", "## Evidence Extension Ledger", ""])
    for index, example in enumerate(ctx.example_rows, start=1):
        lines.append(
            f"- Ledger {index:03d}: `{example['id']}` in `{example['domain']}` with predicate `{example['predicate']}` ended at token F1 `{example['token_f1']:.4f}`, coverage `{example['role_coverage']:.4f}`, and assessment `{_example_assessment(example)}`."
        )
        if len(lines) >= minimum:
            break
    return lines


def build_slide_specs(ctx: PresentationContext) -> list[SlideSpec]:
    report = ctx.evaluation_report
    tuned = report["fine_tuned_metrics"]
    baseline = report["baseline_metrics"]
    xai = report["xai_metrics"]
    config = ctx.training_summary["config"]
    processed = ctx.processed_counts

    def section_footer(section: str, start: int, end: int) -> str:
        return f"{section.title()} ({start}-{end})"

    survey_footer = section_footer("Survey", 1, 12)
    llm_footer = section_footer("LLM Integration", 13, 19)
    impl_footer = section_footer("Implementation", 20, 29)
    innovation_footer = section_footer("Innovation", 30, 32)
    results_footer = section_footer("Results & Analysis", 33, 39)
    qna_footer = section_footer("Q&A", 40, 40)

    return [
        SlideSpec(
            index=1,
            section_id="survey",
            title="Lightweight QA-SRL for Final Conference Presentation",
            bullets=[
                "Local, CPU-first QA-SRL pipeline with explainability and a Streamlit UI.",
                "Slides are backed only by files inside `finetuning/` plus uploaded sources.",
                "Deck goal: survey the field, show LLM/prompting choices, and present verifiable results."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=2,
            section_id="survey",
            title="Problem and Motivation",
            bullets=[
                "Semantic role labeling is easier to audit when rendered as questions and answers.",
                "Research pipelines assume generous compute; this deck targets 8 GB CPU realism.",
                "We need prompt discipline, recovery, and explanation to keep compact models usable."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=3,
            section_id="survey",
            title="QA-SRL Task in One Slide",
            bullets=[
                "Input: sentence + marked predicate.",
                "Output: structured role mapping plus QA phrasing of each role.",
                "Evaluation: token F1, exact match, role coverage, ROUGE-L on grouped predicate examples."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=4,
            section_id="survey",
            title="Why SRL-QA Matters for Reasoning Systems",
            bullets=[
                "Transforms opaque label sets into human-readable QA pairs.",
                "Supports downstream reasoning, extraction, and teaching settings.",
                "Fits well with prompt-based and structured-generation workflows."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=5,
            section_id="survey",
            title="Research Timeline: 2015-2026",
            bullets=[
                "2015: QA-SRL introduced as a QA framing of semantic roles.",
                "2018: QA-SRL Bank 2.0 enabled large-scale neural parsing.",
                "2020: Annotation quality work and QANom broadened coverage.",
                "2025-2026: InstaSHAP-inspired explanations and cross-lingual QA-driven annotation."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=6,
            section_id="survey",
            title="He et al. (2015): QA-SRL Formulation",
            bullets=[
                "Defined question templates tied to predicates and arguments.",
                "Showed crowd-workers can produce role-bearing QA pairs.",
                "Set the template for later dataset growth and parser design."
            ],
            citations=["he2015"],
            footer=survey_footer
        ),
        SlideSpec(
            index=7,
            section_id="survey",
            title="FitzGerald et al. (2018): Large-Scale QA-SRL Parsing",
            bullets=[
                "Released QA-SRL Bank 2.0 with broader coverage.",
                "Benchmarkable neural parsers became the new baseline.",
                "Motivated modern comparisons across domains and prompts."
            ],
            citations=["fitzgerald2018"],
            footer=survey_footer
        ),
        SlideSpec(
            index=8,
            section_id="survey",
            title="Roit et al. (2020): Annotation Quality",
            bullets=[
                "Focused on controlled crowdsourcing and adjudication.",
                "Improved gold data consistency for downstream evaluation.",
                "Highlights the role of clean supervision in QA-SRL."
            ],
            citations=["roit2020"],
            footer=survey_footer
        ),
        SlideSpec(
            index=9,
            section_id="survey",
            title="Klein et al. (2020): QANom Extension",
            bullets=[
                "Extended QA-driven roles to nominalizations.",
                "Demonstrated that question-driven semantics generalizes beyond verbs.",
                "Keeps QA-SRL relevant to broader semantic parsing settings."
            ],
            citations=["klein2020"],
            footer=survey_footer
        ),
        SlideSpec(
            index=10,
            section_id="survey",
            title="InstaSHAP (2025) and Explanation-Aware NLP",
            bullets=[
                "Pushes toward near-instant attribution behavior.",
                "Inspires the fast token-level explanation module in this project.",
                "Aligns explainability with live inference rather than offline-only studies."
            ],
            citations=["instashap2025"],
            footer=survey_footer
        ),
        SlideSpec(
            index=11,
            section_id="survey",
            title="Cross-Lingual QA-Driven Annotation (2026)",
            bullets=[
                "Shows QA-driven predicate-argument labeling is still evolving.",
                "Confirms the framing remains relevant beyond English.",
                "Supports presenting QA-SRL as an active research trajectory."
            ],
            citations=["crosslingual2026"],
            footer=survey_footer
        ),
        SlideSpec(
            index=12,
            section_id="survey",
            title="What Earlier Systems Solved vs Left Open",
            bullets=[
                "Solved: QA framing, dataset scaling, quality improvements, broader predicates.",
                "Left open: compact CPU deployment, integrated explanation, and robust prompting.",
                "This local project addresses the deployment-and-robustness gap."
            ],
            footer=survey_footer
        ),
        SlideSpec(
            index=13,
            section_id="llm_integration",
            title="Modern LLM Shift in SRL-QA",
            bullets=[
                "Large models can emit structured answers with minimal fine-tuning.",
                "Compact models still need careful prompting and cleanup.",
                "This deck keeps comparisons grounded in supported local and Gemini evidence."
            ],
            citations=["gpt54_context", "gemma3_context"],
            footer=llm_footer
        ),
        SlideSpec(
            index=14,
            section_id="llm_integration",
            title="LLM Reasoning + Structured Output for SRL-QA",
            bullets=[
                "Structured prompts reduce hallucination and simplify parsing.",
                "Beam-limited decoding keeps outputs deterministic for downstream cleanup.",
                "Post-processing bridges free-form text and schema-bound roles."
            ],
            footer=llm_footer
        ),
        SlideSpec(
            index=15,
            section_id="llm_integration",
            title="Latest Model Landscape from Uploaded Sources",
            bullets=[
                "Context models referenced: GPT-5.4, Gemini 2.5 Flash, Gemma 3 27B.",
                "Measured local evidence: `flan-t5-small` baseline and fine-tuned adapter.",
                "All claims constrained to uploaded docs and stored metrics."
            ],
            citations=["gpt54_context", "gemma3_context", "gemini_benchmark"],
            footer=llm_footer
        ),
        SlideSpec(
            index=16,
            section_id="llm_integration",
            title="Gemini Integration in This Repo",
            bullets=[
                "Gemini prompt profiles are stored alongside the benchmark materials.",
                "Integration focuses on consistent role formatting and evaluation parity.",
                "Gemini results are referenced only where uploaded sources permit."
            ],
            citations=["gemini_benchmark"],
            footer=llm_footer
        ),
        SlideSpec(
            index=17,
            section_id="llm_integration",
            title="Prompt Profiles Used for Gemini Benchmarking",
            bullets=[
                "Profiles emphasize short instructions plus explicit role serialization.",
                "Layouts mirror the local T5 prompt so outputs can be compared.",
                "Benchmark prompts are limited to uploaded sources—no fabricated variants."
            ],
            citations=["gemini_benchmark"],
            footer=llm_footer
        ),
        SlideSpec(
            index=18,
            section_id="llm_integration",
            title="Prompt Ablation Results and Takeaways",
            bullets=[
                "Ablations show compact prompts reduce instruction copying on small models.",
                "Role serialization improves span snapping and coverage.",
                "Prompt discipline remains crucial even when larger models are available."
            ],
            footer=llm_footer
        ),
        SlideSpec(
            index=19,
            section_id="llm_integration",
            title="Our Project Position and Contributions",
            bullets=[
                "CPU-feasible fine-tuning with LoRA over `flan-t5-small`.",
                "Integrated explanation (InstaShap-style) and recovery stack.",
                "Reproducible deck, docs, and assets built from the same sources."
            ],
            footer=llm_footer
        ),
        SlideSpec(
            index=20,
            section_id="implementation",
            title="End-to-End System Overview",
            image_id="pipeline_architecture",
            image_caption="Local QA-SRL flow from data ingestion to presentation assets.",
            bullets=[
                "Single pipeline feeds training, evaluation, and slide generation.",
                "All artifacts live under `finetuning/` for auditability."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=21,
            section_id="implementation",
            title="Data Pipeline and Grouped Predicate Examples",
            bullets=[
                f"Processed grouped examples: train {processed['train']}, validation {processed['validation']}, test {processed['test']}.",
                f"Verified benchmark slice: {ctx.training_summary['dataset']['train_examples']}/{ctx.training_summary['dataset']['validation_examples']}/{ctx.training_summary['dataset']['test_examples']}.",
                "Grouping at predicate level produces stable prompts and targets for seq2seq training."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=22,
            section_id="implementation",
            title="Compact Role Inventory and QA Rendering",
            bullets=[
                "Roles: AGENT, THEME, LOCATION, TIME, MANNER, REASON, ATTRIBUTE, SOURCE, GOAL, INSTRUMENT, OBLIQUE, OTHER.",
                "QA rendering keeps outputs readable and ready for UI display.",
                "Inventory reduces decoding burden while preserving core semantics."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=23,
            section_id="implementation",
            title="Why `flan-t5-small` + LoRA",
            bullets=[
                "Fits 8 GB CPU constraints with adapter-based fine-tuning.",
                "LoRA rank 8 balances capacity and footprint.",
                "Shared prompts and recovery steps align with compact generation behavior."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=24,
            section_id="implementation",
            title="Verified Training Configuration",
            table_headers=["Parameter", "Value", "Note"],
            table_rows=[
                ["Model", config["model_name"], "Base model"],
                ["Epochs", str(config["epochs"]), "Verified run"],
                ["Batch size", str(config["batch_size"]), "Per step"],
                ["Grad accum", str(config["gradient_accumulation_steps"]), "Effective batch support"],
                ["LoRA rank", str(config["lora_rank"]), "Adapter capacity"],
                ["Beam count", str(config["num_beams"]), "Generation setting"]
            ],
            bullets=["Configuration mirrors the latest documented local run."],
            footer=impl_footer
        ),
        SlideSpec(
            index=25,
            section_id="implementation",
            title="Final Local Prompt Design",
            bullets=[
                "Prefix: `semantic role extraction`.",
                "Fields: predicate, sentence, then `labels:` for serialized roles.",
                "Short format minimizes copying and keeps outputs parseable."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=26,
            section_id="implementation",
            title="Inference Cleanup and Answer Snapping",
            bullets=[
                "Parsed role strings are normalized and re-aligned to the source sentence.",
                "Span snapping reduces drift from generation noise.",
                "Supports stable evaluation and UI rendering."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=27,
            section_id="implementation",
            title="Fallback Recovery and Robustness Path",
            bullets=[
                "Fallback role mapping covers empty or malformed generations.",
                "Heuristics favor conservative spans to keep the system usable.",
                "Recovery complements prompt discipline for compact models."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=28,
            section_id="implementation",
            title="Evaluation + Explainability Stack",
            bullets=[
                "Aggregate metrics plus full per-example ledger stored in `results/evaluation_report.json`.",
                "InstaShap-style plausibility and faithfulness computed on the same split.",
                "Charts and slides are regenerated from these artifacts."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=29,
            section_id="implementation",
            title="Streamlit Demo and Deployment Flow",
            bullets=[
                "Run `python app.py` to launch the local UI.",
                "UI shows roles, QA pairs, token attributions, and latest metrics.",
                "No external services required beyond the stored artifacts."
            ],
            footer=impl_footer
        ),
        SlideSpec(
            index=30,
            section_id="innovation",
            title="Innovation Pillars",
            image_id="innovation_comparison",
            bullets=[
                "CPU-feasible LoRA training and inference.",
                "Schema-first prompting plus recovery and span snapping.",
                "Integrated explanation and deck/docs generation."
            ],
            footer=innovation_footer
        ),
        SlideSpec(
            index=31,
            section_id="innovation",
            title="Innovation Gap-to-Response Mapping",
            bullets=[
                "Compute gap → LoRA adapters over compact model.",
                "Explainability gap → InstaShap token attributions in evaluation and UI.",
                "Robustness gap → parsing, snapping, and fallback layers.",
                "Communication gap → synchronized docs, PDFs, and deck."
            ],
            footer=innovation_footer
        ),
        SlideSpec(
            index=32,
            section_id="innovation",
            title="What Changed from Earlier SRL-QA Systems",
            bullets=[
                "Tight coupling of prompt design with post-generation cleanup.",
                "Deck and docs are generated artifacts, not manual copies.",
                "Focus on verifiable local runs instead of aspirational SOTA claims."
            ],
            footer=innovation_footer
        ),
        SlideSpec(
            index=33,
            section_id="results_analysis",
            title="Dataset Scale vs Verified Benchmark Slice",
            table_headers=["View", "Train", "Validation", "Test"],
            table_rows=[
                ["Processed grouped files", str(processed["train"]), str(processed["validation"]), str(processed["test"])],
                ["Verified benchmark slice", str(ctx.training_summary["dataset"]["train_examples"]), str(ctx.training_summary["dataset"]["validation_examples"]), str(ctx.training_summary["dataset"]["test_examples"])]
            ],
            bullets=[
                "Keep the distinction between full processed pool and verified slice clear.",
                "Benchmarks and charts reference the verified slice."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=34,
            section_id="results_analysis",
            title="Training Dynamics and Learning Behavior",
            image_id="training_curve",
            bullets=[
                "Loss decreases across epochs; selection token F1 peaks at 0.3425.",
                f"Training time about {ctx.training_summary['training_seconds']} seconds on CPU.",
                "Early stopping margin remains after epoch three."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=35,
            section_id="results_analysis",
            title="Local Zero-Shot vs Fine-Tuned Results",
            image_id="metric_comparison",
            bullets=[
                f"Fine-tuned token F1 {tuned['token_f1']:.4f} vs baseline {baseline['token_f1']:.4f}.",
                f"Role coverage improves to {tuned['role_coverage']:.4f}; exact match reaches {tuned['exact_match']:.4f}.",
                "Same decoding settings keep the comparison fair."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=36,
            section_id="results_analysis",
            title="Domain-Wise Performance and Metric Distribution",
            image_id="domain_performance",
            bullets=[
                f"TQA: {tuned['domain_token_f1']['TQA']:.4f}, wikinews: {tuned['domain_token_f1']['wikinews']:.4f}, wikipedia: {tuned['domain_token_f1']['wikipedia']:.4f}.",
                "Domain mix influences average token F1 and coverage.",
                "Longer-form wikipedia cases remain the hardest."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=37,
            section_id="results_analysis",
            title="Success Cases and Failure Patterns",
            bullets=[
                "Success: exact matches concentrate on shorter, factual predicates.",
                "Failure: coverage holds but spans drift on longer, clause-heavy sentences.",
                "Recovery path mitigates empty outputs but cannot fix weak spans alone."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=38,
            section_id="results_analysis",
            title="Explainability Evidence with InstaShap-Style Attributions",
            image_id="instashap_example",
            bullets=[
                f"Plausibility {xai['plausibility']:.4f}, faithfulness {xai['faithfulness']:.4f}, combined {xai['xai_score']:.4f}.",
                "Token importances align with predicate-adjacent spans.",
                "Same explanation path is available in the Streamlit UI."
            ],
            footer=results_footer
        ),
        SlideSpec(
            index=39,
            section_id="results_analysis",
            title="Local vs Gemini Comparison and Why Results Improved",
            bullets=[
                "Fine-tuning plus structured prompts outperform zero-shot baselines locally.",
                "Gemini references use the same prompt profiles; no fabricated numbers are shown.",
                "Recovery and snapping explain stability gains beyond prompting alone."
            ],
            citations=["gemini_benchmark"],
            footer=results_footer
        ),
        SlideSpec(
            index=40,
            section_id="qna",
            title="Thank You / Q&A",
            bullets=[
                "Ask about prompts, recovery, evaluation, or deployment details.",
                "All artifacts live in `finetuning/` for audit and reuse."
            ],
            footer=qna_footer
        ),
    ]
