# Survey, Innovation, Implementation, And Analysis Connection

## Purpose Of This Document

This document connects the whole project into one story:

```text
Survey -> Innovation -> Implementation -> Analysis -> Demo
```

It also collects accuracy values across the project. The values below are taken from local project artifacts and are labeled by evaluation scope. Do not mix a small seed-suite score with a full test-set score without naming the scope.

## 1. Survey Connection

The survey part of the project motivates why Semantic Role Labeling (SRL) is useful for Question Answering (QA).

The key survey ideas are:

- **Semantic Role Labeling** identifies the meaning of arguments around a predicate.
- **PropBank** provides real predicate-argument annotations such as `ARG0`, `ARG1`, `ARGM-LOC`, and `ARGM-TMP`.
- **Extractive QA** can be improved by mapping questions to semantic roles.
- **RAG systems** can retrieve better evidence when documents contain structured semantic triples, not only raw text.
- **Explainable QA** becomes easier when answers are connected to visible reasoning paths.

In simple terms:

```text
The survey explains why role-aware QA should be more explainable than plain text QA.
```

Example:

```text
Question: Where was the package delivered?
Expected role: ARGM-LOC
Answer: to the office
```

This is easier to explain than a black-box answer because the answer is tied to a semantic role.

## 2. Innovation Connection

The project innovation is not only to answer questions, but to answer them with SRL-grounded evidence.

The main innovation chain is:

```text
PropBank SRL data
    -> predicate-role structures
    -> SRL-enhanced retrieval
    -> role-aware answer selection
    -> semantic graph explanation
```

Key innovations in the workspace:

| Innovation | Where It Appears | Purpose |
|---|---|---|
| PropBank frame retrieval | `srlqa/retrieval/` and `srl_rag_demo/frame_store.py` | Uses PropBank frames to connect predicates with role definitions. |
| Constrained span decoding | `srlqa/decoding/` | Keeps answer spans extractive and role-compatible. |
| Evidence verifier | `srlqa/verification/` | Scores candidate spans using extractability and frame-role compatibility. |
| Self-correction loop | `srlqa/verification/self_correction.py` | Lets the system retry candidate spans in evaluation/demo mode. |
| Hybrid reranking | `srl_qa_project/hybrid_qa.py` | Improves the legacy baseline using role-aware heuristics and optional local models. |
| SRL + RAG demo | `srl_rag_demo/` | Retrieves SRL-structured documents and visualizes reasoning paths. |
| Explainable graph QA | `srl_rag_demo/graphing.py` | Builds graph nodes for question, document, predicate, role, frame, candidate, and answer. |

## 3. Implementation Connection

The project has four main implementation branches.

### A. Legacy PropQA-Net: `srl_qa_project`

This is the original SRL + QA baseline.

It implements:

- NLTK PropBank loading
- Treebank-backed span reconstruction
- QA pair generation from PropBank roles
- Classical SRL + QA model
- Evaluation and benchmark scripts
- Streamlit research app

Best use:

```text
Use this branch to explain the original dataset pipeline and baseline model.
```

### B. RAISE-SRL-QA Scaffold: `srlqa`

This is the newer experimental framework.

It implements:

- PropBank frame indexing
- Deterministic SRL-QA
- Optional model-backed QA
- Model comparison runner
- Evaluation tables and plots
- Streamlit app

Best use:

```text
Use this branch to explain the innovation modules: retrieval, verifier, correction, and model comparison.
```

### C. Standalone SRL + RAG Demo: `srl_rag_demo`

This is the final demo app.

It implements:

- Local PropBank loader
- SRL document builder
- User pasted/uploaded document ingestion
- Hybrid retrieval with embeddings plus TF-IDF fallback
- Role-aware answer selection
- NetworkX + Plotly graph explanation
- Streamlit interface

Best use:

```text
Use this branch for the live demo because it connects SRL, RAG, QA, and explainability in one app.
```

### D. LoRA / QLoRA Artifacts

These are the experiment artifact folders:

```text
propbank_srlqa_artifacts/
propbank_srlqa_2b_artifacts/
```

They contain:

- research summaries
- prediction CSVs
- adapter checkpoints
- plots
- notebook outputs

Best use:

```text
Use these as experimental evidence and future-modeling references, not as the main live demo path.
```

## 4. Analysis Connection

The analysis part answers:

```text
Did SRL structure improve QA behavior?
```

The strongest local evidence is:

- The legacy baseline performs reasonably on the full test set.
- Hybrid and RAISE-style role-aware systems improve on curated role challenge suites.
- The standalone SRL + RAG demo answers the main demo example correctly and produces a reasoning graph.
- LoRA experiments show early FAST_DEV progress but should not be overclaimed as final full-corpus results.

## 5. Recommended Accuracy Values To Report

These are the safest values to report because they are directly supported by local artifacts.

### Full Baseline Test Metrics

Source:

```text
srl_qa_project/results/metrics.json
srlqa/output/tables/baseline_metrics_summary.csv
```

| System | Scope | Exact Match / Accuracy | Token F1 | SRL Micro F1 | SRL BIO Accuracy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Legacy PropQA-Net baseline | Baseline test set | 51.84% | 76.12% | 71.33% | 81.63% | Best full-test-set value to report. |

Additional baseline detail:

| Metric | Value |
|---|---:|
| QA exact match | 51.84% |
| QA token F1 | 76.12% |
| SRL micro precision | 73.20% |
| SRL micro recall | 69.55% |
| SRL micro F1 | 71.33% |
| SRL macro F1 | 16.19% |
| SRL BIO accuracy | 81.63% |
| Best validation F1 during training | 77.27% |

### Curated SRL-QA Seed Suite Metrics

Source:

```text
srlqa/output/tables/model_evaluation_summary.csv
srlqa/output/tables/ablation_study.csv
srlqa/plots/RAISE_FAST_SEED_RESULTS.csv
```

Scope: `challenge_suite_v2`, 15 examples.

| System | Accuracy / Exact Match | Token F1 | Role Accuracy | Mean Confidence | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Legacy PropQA-Net baseline | 20.00% | 55.22% | 33.33% | 47.77% | 207.54 ms |
| Legacy Hybrid | 66.67% | 82.30% | 93.33% | 65.31% | 495.87 ms |
| RAISE-SRL-QA Fast | 100.00% | 100.00% | 100.00% | 95.60% | 7.08 ms |
| RAISE-SRL-QA Model | 100.00% | 100.00% | 100.00% | 95.71% | 1294.63 ms |

Important wording for presentation:

```text
On the small curated role-challenge seed suite, the RAISE fast pipeline reaches 100% exact match and 100% role accuracy. This is a controlled demo-suite result, not a full-corpus claim.
```

### All-Model Smoke Example

Source:

```text
srlqa/plots/EXACT_VALUES.json
```

Scope: one demo example, `Where was the package delivered?`

| System | Exact Match | Token F1 | Role Accuracy | Answer |
|---|---:|---:|---:|---|
| RAISE-SRL-QA Fast | 100.00% | 100.00% | 100.00% | `to the office` |
| RAISE-SRL-QA Model | 100.00% | 100.00% | 100.00% | `to the office` |
| Legacy Hybrid | 100.00% | 100.00% | 100.00% | `to the office` |
| Legacy PropQA-Net Baseline | 0.00% | 54.55% | 0.00% | `delivered the package to the office at noon` |

Mean across these four systems on this one example:

| Metric | Value |
|---|---:|
| Mean exact match | 75.00% |
| Mean token F1 | 88.64% |
| Mean role accuracy | 75.00% |

### Legacy Benchmark Tracks

Source:

```text
srl_qa_project/results/benchmarks/benchmark_results.json
```

Scope: challenge set of 20 examples and combined 60-example benchmark.

| Track | Scope | Exact Match | Token F1 | Role Accuracy |
|---|---:|---:|---:|---:|
| Classical baseline | Challenge, 20 examples | 10.00% | 47.01% | 20.00% |
| Heuristic reranker | Challenge, 20 examples | 60.00% | 77.98% | 100.00% |
| Transformer QA assist | Challenge, 20 examples | 60.00% | 77.98% | 100.00% |
| Full hybrid | Challenge, 20 examples | 60.00% | 77.98% | 100.00% |
| Classical baseline | Combined, 60 examples | 8.33% | 36.24% | 28.33% |
| Heuristic reranker | Combined, 60 examples | 33.33% | 53.83% | 73.33% |
| Transformer QA assist | Combined, 60 examples | 33.33% | 53.83% | 73.33% |
| Full hybrid | Combined, 60 examples | 33.33% | 53.83% | 73.33% |

Best interpretation:

```text
Role-aware reranking strongly improves role accuracy and token F1 on challenge-style examples compared with the classical baseline.
```

### LoRA / QLoRA FAST_DEV Metrics

Source:

```text
propbank_srlqa_artifacts/research_summary.txt
propbank_srlqa_artifacts/test_predictions.csv
propbank_srlqa_2b_artifacts/research_summary_2b.txt
propbank_srlqa_2b_artifacts/generative_test_predictions.csv
```

| Experiment | Scope | Exact Match | Token F1 | Notes |
|---|---:|---:|---:|---|
| DistilBERT LoRA extractive QA | FAST_DEV_RUN, 8 test examples | 12.50% | 41.51% | Improved over baseline validation token F1 of 31.09%. |
| Tiny-GPT2 / Gemma 2B QLoRA scaffold | FAST_DEV_RUN, 8 test examples | 37.50% in summary; 0.00% in CSV aggregation | 40.97% in summary; 0.00% in CSV aggregation | Treat as unstable dev artifact because summary and prediction CSV disagree. |

Safe presentation wording:

```text
The LoRA experiments are early FAST_DEV validation runs. They show implementation feasibility, but the final reported accuracy should rely on the legacy full-test metrics and the RAISE curated challenge-suite metrics.
```

## 6. Accuracy Values For A Strong Presentation Slide

Use this compact table for a clean presentation:

| Project Branch | Best Supported Accuracy Statement |
|---|---|
| Legacy PropQA-Net full test | 51.84% exact match, 76.12% QA token F1, 71.33% SRL micro F1 |
| Legacy Hybrid challenge suite | 60.00% exact match, 77.98% token F1, 100.00% role accuracy on 20 challenge examples |
| RAISE-SRL-QA seed suite | 100.00% exact match, 100.00% token F1, 100.00% role accuracy on 15 curated seed examples |
| SRL + RAG demo | Correctly answers the live courier example with `to the office` and builds a reasoning graph |
| DistilBERT LoRA FAST_DEV | 12.50% exact match, 41.51% token F1 on an 8-example dev test |

Recommended sentence:

```text
On the full PropBank-derived baseline test set, the legacy model achieves 76.12% QA token F1 and 71.33% SRL micro F1. On controlled SRL role challenge suites, role-aware hybrid and RAISE pipelines improve answer selection substantially, reaching up to 100% exact match on the curated 15-example seed suite. The new SRL + RAG Streamlit demo connects those ideas with retrieved SRL evidence and explainable semantic graphs.
```

## 7. How Survey, Innovation, Implementation, And Analysis Fit Together

| Section | Project Role | Evidence |
|---|---|---|
| Survey | Explains why SRL and PropBank matter for QA. | Project docs and literature framing. |
| Innovation | Adds retrieval, verification, correction, and graph explanation. | `srlqa/` and `srl_rag_demo/` modules. |
| Implementation | Provides runnable loaders, QA systems, RAG app, and Streamlit demos. | `srl_qa_project/`, `srlqa/`, `srl_rag_demo/`. |
| Analysis | Measures baseline accuracy, challenge-suite improvements, and dev experiment behavior. | JSON/CSV metrics and generated plots. |

## 8. Final Project Story

The project starts with a survey insight:

```text
Question answering needs semantic structure to become more explainable.
```

It turns that into an innovation:

```text
Use SRL roles and PropBank frames inside retrieval and answer selection.
```

It implements the idea in three layers:

```text
Legacy baseline -> RAISE SRL-QA framework -> standalone SRL + RAG demo
```

It analyzes the result with measured values:

```text
Baseline full-test token F1: 76.12%
Baseline SRL micro F1: 71.33%
Legacy hybrid challenge token F1: 77.98%
RAISE seed-suite token F1: 100.00%
```

It demonstrates the final output in Streamlit:

```text
Retrieve SRL evidence -> select answer -> show semantic graph reasoning
```

## 9. Claim Discipline

Use the following claim discipline in reports and presentations:

- Report the legacy PropQA-Net metrics as full-test metrics.
- Report the RAISE 100% result only as a curated 15-example seed-suite result.
- Report LoRA and QLoRA values only as FAST_DEV implementation experiments.
- Do not claim the Gemma 2B target result as achieved unless a full, consistent evaluation run is produced.
- Use the SRL + RAG demo as a functional explainability demo, not as a full benchmark replacement.
