# Question Answering Using Semantic Roles (PropQA-Net)

This repository implements **PropQA-Net**, a classical (non-transformer) question answering system *anchored in semantic roles*. It is trained on real **PropBank** annotations (via **NLTK**) and learns two tasks jointly:

- **SRL tagging** (BIO tags over a sentence: `B-ARG0`, `I-ARG1`, `O`, ...)
- **Extractive QA** (predict an answer span inside the sentence)

At a high level, the pipeline:

1. Loads PropBank instances from the local `nltk_data/` folder.
2. Keeps only instances that can be aligned to the included Penn Treebank sample (so token spans can be reconstructed deterministically).
3. Generates SRL labels and natural-language questions from PropBank arguments.
4. Splits examples deterministically into train/validation/test (seed `42`) and caches them under `data/`.
5. Trains PropQA-Net with a combined SRL + QA loss.
6. Evaluates and writes metrics + plots to `results/`.
7. Runs a 10-example inference demo.
8. Generates academic PDF deliverables and a concatenated implementation bundle under `outputs/`.

Deeper docs live under `docs/`.

## Quickstart

Requirements:

- Python 3.10+

From inside `srl_qa_project/`:

```bash
python -m pip install -r requirements.txt
python main.py --mode full
```

Supported modes:

```bash
python main.py --mode train
python main.py --mode eval
python main.py --mode infer
python main.py --mode ask
python main.py --mode benchmark
python main.py --mode report
python main.py --mode app
python main.py --mode full
```

Ask your own questions with the trained model:

```bash
python main.py --mode ask --context "The chef cooked a delicious meal in the kitchen yesterday." --question "Who cooked?"
python main.py --mode ask --engine baseline --context "The chef cooked a delicious meal in the kitchen yesterday." --question "Who cooked?"
python main.py --mode ask --interactive
python main.py --mode app
```

## Repository Layout

```text
srl_qa_project/
|-- main.py                 # CLI runner (train/eval/infer/ask/full)
|-- config.py               # All paths + hyperparameters in dataclasses
|-- data_loader.py          # PropBank -> (context, question, answer) examples + DataLoaders
|-- model.py                # PropQA-Net (BiLSTM encoders + SRL head + span heads)
|-- trainer.py              # Training loop + early stopping + checkpointing
|-- evaluator.py            # Metrics + plots + error analysis
|-- qa_inference.py         # Runtime inference wrapper, custom question asking, and 10-example demo
|-- hybrid_qa.py            # Hybrid inference layer with role-aware reranking
|-- benchmark.py            # Hybrid benchmark runner and report plot generation
|-- app.py                  # Streamlit research website
|-- pdf_generator.py        # Programmatic PDFs + export of outputs/implementation_code.py
|-- requirements.txt
|-- nltk_data/              # Included NLTK corpora assets (PropBank + Treebank subset)
|-- data/                   # Cached processed splits (train/val/test JSON)
|-- checkpoints/            # Saved model checkpoint (best_model.pt)
|-- results/                # metrics.json, data_statistics.json, plots/
|-- outputs/                # PDFs + implementation_code.py (generated)
`-- docs/                   # Extra documentation
```

## Research Upgrade

The codebase now includes a research-grade hybrid layer on top of the original checkpoint:

- `hybrid_qa.py`: role-aware hybrid inference with optional local transformer QA and semantic matching
- `benchmark.py`: four-track benchmark runner, challenge-set evaluation, and new analysis plots
- `app.py`: Streamlit website with QA, architecture, experiments, documentation, and downloads
- `data/challenge_suite.json`: curated semantic-role challenge suite for demonstrations and evaluation

The original PropQA-Net metrics remain the authoritative local baseline. Literature-level numbers are treated as cited references, not as locally reproduced claims.

## What Gets Generated

After `python main.py --mode full`, the main artifacts are:

- `checkpoints/best_model.pt`: best checkpoint by validation F1
- `results/data_statistics.json`: dataset coverage and descriptive stats
- `results/metrics.json`: SRL + QA metrics, error analysis, and a small prediction sample
- `results/benchmarks/benchmark_results.json`: baseline vs hybrid benchmark outputs
- `results/plots/*.png`: required plots (loss curve, confusion matrix, etc.)
- `results/plots/ablation_summary.png`
- `results/plots/latency_accuracy_tradeoff.png`
- `results/plots/question_type_heatmap.png`
- `results/plots/role_heatmap.png`
- `results/plots/confidence_histogram.png`
- `results/plots/research_architecture.png`
- `outputs/survey.pdf`
- `outputs/analysis.pdf`
- `outputs/innovation.pdf`
- `outputs/research_paper.pdf`
- `outputs/implementation_code.py`: concatenation of runnable modules (generated)

## Data Source Notes (PropBank Coverage)

NLTK exposes a large PropBank index, but exact answer-span reconstruction is only possible when the PropBank instance can be aligned to a Treebank parse that exists locally. This repository includes an NLTK-style `nltk_data/` directory with PropBank plus a Treebank subset, so the loader:

- reports the total PropBank instances visible to NLTK
- filters down to the “Treebank-backed” subset that can be reconstructed deterministically

This keeps the project grounded in real PropBank annotations while remaining runnable offline.

## Configuration Knobs

All configuration is centralized in `config.py`. Common knobs:

- `ProjectConfig.data.max_instances`: limit the number of PropBank instances to speed up runs
- `ProjectConfig.data.rebuild_cache`: force rebuilding `data/*.json` splits
- `ProjectConfig.training.max_epochs`, `batch_size`, `learning_rate`: training speed/quality tradeoffs

## Docs Index

- `docs/OVERVIEW.md`: end-to-end architecture and “where to start”
- `docs/DATA.md`: dataset generation, cached split format, JSON schema
- `docs/MODEL.md`: PropQA-Net architecture, losses, decoding
- `docs/EVALUATION.md`: metrics, plots, and error taxonomy
- `docs/PDF_DELIVERABLES.md`: how PDFs and `implementation_code.py` are produced
- `docs/TROUBLESHOOTING.md`: common environment and runtime issues
