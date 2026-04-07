# Overview

PropQA-Net is an SRL-anchored extractive QA baseline built on *real PropBank annotations*. The upgraded repository now pairs that reproducible baseline with a hybrid inference layer, benchmark runner, and Streamlit research website while keeping the original checkpoint and metrics as the authoritative local baseline.

## Where To Start

If you're new to the codebase, read files in this order:

1. `main.py`: orchestrates the pipeline (`--mode train|eval|infer|ask|benchmark|report|app|full`)
2. `config.py`: project paths and all hyperparameters
3. `data_loader.py`: PropBank + Treebank alignment, example generation, and caching
4. `model.py`: PropQA-Net architecture and decoding
5. `trainer.py`: training loop + early stopping + checkpointing
6. `evaluator.py`: metrics, plots, and error analysis
7. `qa_inference.py`: baseline raw-text inference wrapper and the classical demo
8. `hybrid_qa.py`: role-aware hybrid inference with optional local transformer QA and sentence embeddings
9. `benchmark.py`: challenge-set evaluation, ablations, and extra report plots
10. `app.py`: local Streamlit research website
11. `pdf_generator.py`: programmatic PDF deliverables and `outputs/implementation_code.py`

## Pipeline Diagram (Conceptual)

```text
NLTK PropBank + Treebank subset (nltk_data/)
            |
            v
PropBank instance alignment + span reconstruction  (data_loader.py)
            |
            v
Example JSON schema (context, question, answer span, SRL BIO tags)
            |
            +--> cache to data/train.json|val.json|test.json
            |
            v
Torch DataLoaders + vocabularies (data_loader.py)
            |
            v
PropQA-Net training (trainer.py + model.py)  -> checkpoints/best_model.pt
            |
            v
Evaluation + plots (evaluator.py)            -> results/metrics.json, results/plots/*.png
            |
            +--> Baseline inference demo / custom question asking (qa_inference.py) -> results/inference_demo.json
            |
            +--> Hybrid QA + challenge benchmarks (hybrid_qa.py, benchmark.py) -> results/benchmarks/benchmark_results.json
            |
            +--> Streamlit website (app.py) -> local research dashboard
            |
            v
PDF deliverables (pdf_generator.py)          -> outputs/*.pdf + outputs/implementation_code.py
```

## Key Design Decisions

- Offline, reproducible data: the loader registers `nltk_data/` into NLTK's search path so the project can run without downloading corpora.
- Exact spans: only Treebank-backed PropBank instances are used so answer spans and BIO tags can be expressed in the same tokenization.
- Multi-task model: SRL BIO tagging and extractive QA share the same context encoder so the QA head is “grounded” in roles.
- Deterministic splits: cached JSON splits are generated with a fixed random seed to make results reproducible.

## Detailed Documentation

This directory now includes comprehensive reference documents:

- `DETAILED_ANALYSIS.md`: In-depth analysis of dataset, model, training, evaluation, errors, hybrid system, benchmarks, ablation, latency, confidence calibration, and limitations
- `DETAILED_INNOVATION.md`: Detailed description of all 10 innovations, novelty assessment, impact, comparison with state of the art, and future directions
- `DETAILED_SURVEY.md`: Comprehensive literature survey covering SRL foundations, classical and neural approaches, PropBank resources, QA evolution, transformer-based QA, QA-SRL, LLMs and structured semantics, benchmarks, explainability, reproducibility, and gap analysis
- `DETAILED_ARCHITECTURE.md`: Complete system architecture with high-level diagrams, data pipeline, model architecture, training, evaluation, inference, hybrid QA, benchmark, Streamlit app, configuration, data flow diagrams, component interactions, deployment, and extensibility points

## Outputs Cheat Sheet

- `data/*.json`: cached examples; delete or set `ProjectConfig.data.rebuild_cache=True` to regenerate
- `checkpoints/best_model.pt`: best checkpoint selected by validation F1
- `results/data_statistics.json`: corpus coverage + descriptive stats
- `results/metrics.json`: SRL + QA metrics, error taxonomy, sample predictions
- `results/benchmarks/benchmark_results.json`: four-track benchmark results and challenge-set records
- `results/plots/*.png`: plots used both for inspection and for the PDFs
- `outputs/*.pdf`: final PDFs
- `outputs/implementation_code.py`: generated concatenation of runnable modules
