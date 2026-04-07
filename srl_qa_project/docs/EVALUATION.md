# Evaluation

Evaluation now runs in two layers:

- baseline evaluation over the PropBank-derived test split via `evaluator.py`
- hybrid benchmarking over a curated challenge suite plus a question-type-aware test subset via `benchmark.py`

The implementation lives in `evaluator.py`.

## Outputs

After `python main.py --mode eval` (or `--mode full`), you should see:

- `results/metrics.json`
- `results/plots/loss_curve.png`
- `results/plots/f1_by_argtype.png`
- `results/plots/confusion_matrix.png`
- `results/plots/qa_accuracy_by_qtype.png`
- `results/plots/answer_length_dist.png`
- `results/plots/error_taxonomy.png`

These plots are also embedded into the generated PDFs.

After `python main.py --mode benchmark`, you should also see:

- `results/benchmarks/benchmark_results.json`
- `results/plots/ablation_summary.png`
- `results/plots/latency_accuracy_tradeoff.png`
- `results/plots/question_type_heatmap.png`
- `results/plots/role_heatmap.png`
- `results/plots/confidence_histogram.png`
- `results/plots/dataset_balance.png`
- `results/plots/challenge_table.png`
- `results/plots/error_gallery.png`
- `results/plots/research_architecture.png`

## QA Metrics

The QA report includes:

- Exact Match (EM): predicted answer text equals gold answer text after normalization
- Token-overlap F1: overlap-based F1 between predicted and gold answer token bags
- Breakdown by question type: `WHO`, `WHAT`, `WHEN`, `WHERE`, `WHY`, `HOW`
- Hybrid benchmark summaries also include role accuracy, mean confidence, model load time, and mean latency

## SRL Metrics

The SRL report is computed from predicted and gold BIO sequences:

- Token-level precision/recall/F1 by role
- Confusion matrix over role labels

Note: BIO labels are often reduced to their role component (for example `B-ARG1` and `I-ARG1` both count as `ARG1`) when reporting per-role metrics.

## Error Analysis

`results/metrics.json` contains an `error_analysis` section with:

- taxonomy counts (for example boundary errors vs. role confusions)
- example-level records used for debugging and inspection
