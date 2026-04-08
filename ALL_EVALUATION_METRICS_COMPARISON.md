# All Evaluation Metrics And Implementation Comparison

## Purpose

This file collects the evaluation metrics across the whole project and compares all major implementations in one place.

It covers:

- `srl_qa_project`
- `srlqa`
- `srl_rag_demo`
- `propbank_srlqa_artifacts`
- `propbank_srlqa_2b_artifacts`

Use this file when you need one clear table for reports, presentations, demos, or viva explanations.

## 1. Evaluation Sources

| Source | Project Area | What It Measures |
|---|---|---|
| `srl_qa_project/results/metrics.json` | Legacy PropQA-Net | Full baseline QA and SRL metrics. |
| `srl_qa_project/results/benchmarks/benchmark_results.json` | Legacy benchmark tracks | Classical baseline, heuristic reranker, transformer assist, and full hybrid tracks. |
| `srlqa/output/tables/baseline_metrics_summary.csv` | RAISE reporting | Linked baseline summary, corpus counts, and training metadata. |
| `srlqa/output/tables/model_evaluation_summary.csv` | RAISE model comparison | Legacy baseline, legacy hybrid, RAISE fast, and RAISE model comparison. |
| `srlqa/output/tables/ablation_study.csv` | RAISE ablation | Incremental improvement from baseline to hybrid and RAISE variants. |
| `srlqa/plots/EXACT_VALUES.json` | All-model smoke example | One-example smoke comparison for the courier example. |
| `srl_rag_demo/smoke_test.py` | SRL + RAG demo | PropBank loading, retrieval, answer selection, and graph construction smoke test. |
| `propbank_srlqa_artifacts/research_summary.txt` | DistilBERT LoRA | FAST_DEV extractive QA fine-tuning metrics. |
| `propbank_srlqa_2b_artifacts/research_summary_2b.txt` | Tiny-GPT2 / Gemma 2B QLoRA scaffold | FAST_DEV generative QA experiment metrics. |

## 2. Metric Definitions

| Metric | Meaning | Used In |
|---|---|---|
| Accuracy | Fraction of examples answered correctly. In this project it is usually equivalent to exact match for QA. | `srlqa` model comparison. |
| Exact Match | Predicted answer exactly equals the gold answer. | All QA systems. |
| Token Precision | Fraction of predicted answer tokens that overlap with the gold answer. | `srlqa` comparison and generative summaries. |
| Token Recall | Fraction of gold answer tokens found in the predicted answer. | `srlqa` comparison and generative summaries. |
| Token F1 | Harmonic mean of token precision and token recall. | Main QA quality metric. |
| BLEU | Token-overlap generation metric. | Secondary `srlqa` model comparison metric. |
| Role Accuracy | Whether the predicted semantic role matches the target role. | SRL-QA systems and challenge suites. |
| SRL Micro Precision | Overall precision across all SRL labels before averaging. | Legacy full-test SRL evaluation. |
| SRL Micro Recall | Overall recall across all SRL labels before averaging. | Legacy full-test SRL evaluation. |
| SRL Micro F1 | Overall SRL F1 across all labels. | Main SRL quality metric. |
| SRL Macro Precision | Average precision across roles. | Rare-role analysis. |
| SRL Macro Recall | Average recall across roles. | Rare-role analysis. |
| SRL Macro F1 | Average F1 across roles. | Rare-role analysis. |
| BIO Accuracy | Token-level sequence-labeling accuracy for `B`, `I`, and `O` tags. | Legacy SRL evaluation. |
| Confidence | Model or heuristic confidence score. | Ranking and explanation, not a direct correctness metric. |
| Latency | Time required to produce an answer. | Live demo and model comparison. |
| Median Latency | Middle latency value. | RAISE model comparison. |
| P95 Latency | 95th percentile latency. | RAISE model comparison. |
| Span Accuracy | Whether the predicted extractive span matches the target span. | LoRA extractive QA artifact. |
| Ok Count | Successful examples without runtime errors. | RAISE model comparison. |
| Error Count | Failed examples with runtime errors. | RAISE model comparison. |
| Graph Nodes | Number of nodes in the explanation graph. | SRL + RAG demo smoke test. |
| Graph Edges | Number of edges in the explanation graph. | SRL + RAG demo smoke test. |
| Retrieval Backend | Retrieval method used in the demo. | SRL + RAG demo smoke test. |

## 3. Full Legacy Baseline Metrics

Source:

```text
srl_qa_project/results/metrics.json
srlqa/output/tables/baseline_metrics_summary.csv
```

Scope:

```text
Legacy PropQA-Net baseline test set.
```

| Metric | Value |
|---|---:|
| QA exact match | 51.84% |
| QA token F1 | 76.12% |
| SRL micro precision | 73.20% |
| SRL micro recall | 69.55% |
| SRL micro F1 | 71.33% |
| SRL macro precision | 21.11% |
| SRL macro recall | 15.92% |
| SRL macro F1 | 16.19% |
| SRL BIO accuracy | 81.63% |
| Best validation F1 | 77.27% |
| Best epoch | 6 |
| Parameter count | 1,784,352 |

Corpus and data metrics:

| Metric | Value |
|---|---:|
| Total PropBank instances | 112,917 |
| Usable PropBank instances in baseline statistics | 9,073 |
| QA pair count | 23,007 |
| Unique predicates | 1,340 |
| Unique rolesets | 1,670 |

Best statement:

```text
On the full PropBank-derived baseline test set, the legacy PropQA-Net system reaches 51.84% exact match, 76.12% QA token F1, 71.33% SRL micro F1, and 81.63% BIO accuracy.
```

## 4. Main Implementation Comparison Table

This is the main table to use for comparing all implementations.

| Implementation | Source Project | Evaluation Scope | Examples | Exact Match / Accuracy | Token F1 | Role Accuracy | SRL Micro F1 | BIO Accuracy | Mean Confidence | Mean Latency | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Legacy PropQA-Net full baseline | `srl_qa_project` | Full baseline test | full test | 51.84% | 76.12% | not separately reported | 71.33% | 81.63% | not reported | not reported | Best full-test baseline metric row. |
| Classical baseline | `srl_qa_project` | Challenge benchmark | 20 | 10.00% | 47.01% | 20.00% | 71.33% shared | 81.63% shared | 45.70% | 2.89 ms | Baseline on curated challenge examples. |
| Heuristic reranker | `srl_qa_project` | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 71.33% shared | 81.63% shared | 64.50% | 3.26 ms | Role-aware reranking improvement. |
| Transformer QA assist | `srl_qa_project` | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 71.33% shared | 81.63% shared | 64.50% | 285.43 ms | Adds local transformer assist with higher latency. |
| Full hybrid | `srl_qa_project` | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 71.33% shared | 81.63% shared | 64.50% | 369.37 ms | Most complete legacy hybrid track. |
| Legacy PropQA-Net baseline | `srlqa` | 15-example curated seed suite | 15 | 20.00% | 55.22% | 33.33% | linked baseline | linked baseline | 47.77% | 207.54 ms | RAISE comparison row for legacy baseline. |
| Legacy Hybrid | `srlqa` | 15-example curated seed suite | 15 | 66.67% | 82.30% | 93.33% | linked baseline | linked baseline | 65.31% | 495.87 ms | Legacy hybrid in RAISE comparison. |
| RAISE-SRL-QA Fast | `srlqa` | 15-example curated seed suite | 15 | 100.00% | 100.00% | 100.00% | linked baseline | linked baseline | 95.60% | 7.08 ms | Fast deterministic role-aware pipeline. |
| RAISE-SRL-QA Model | `srlqa` | 15-example curated seed suite | 15 | 100.00% | 100.00% | 100.00% | linked baseline | linked baseline | 95.71% | 1294.63 ms | Model-assisted RAISE pipeline. |
| SRL + RAG Streamlit demo | `srl_rag_demo` | Smoke test | 1 demo case | 1/1 correct | 100.00% on smoke case | ARGM-LOC correct | not a full SRL benchmark | not a full SRL benchmark | answer confidence available in app | not benchmarked | Correctly answers `to the office` and builds a graph. |
| DistilBERT LoRA extractive QA | `propbank_srlqa_artifacts` | FAST_DEV test | 8 | 12.50% | 41.51% | not reported | not reported | not reported | not reported | not reported | Early feasibility run. |
| Tiny-GPT2 / Gemma 2B QLoRA scaffold | `propbank_srlqa_2b_artifacts` | FAST_DEV test | 8 | 37.50% in summary | 40.97% in summary | not reported | not reported | not reported | not reported | not reported | Treat as unstable dev artifact because generated CSV and summary should be reconciled before final claim. |

Important note:

```text
Do not compare the 100% RAISE seed-suite result directly with the full baseline test result. The RAISE score is from a small curated 15-example challenge suite, while the full baseline result is from the larger PropBank-derived test evaluation.
```

## 5. Legacy Benchmark Track Comparison

Source:

```text
srl_qa_project/results/benchmarks/benchmark_results.json
```

### Challenge Split

| Track | Examples | Exact Match | Token F1 | Role Accuracy | Mean Confidence | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Classical baseline | 20 | 10.00% | 47.01% | 20.00% | 45.70% | 2.89 ms |
| Heuristic reranker | 20 | 60.00% | 77.98% | 100.00% | 64.50% | 3.26 ms |
| Transformer QA assist | 20 | 60.00% | 77.98% | 100.00% | 64.50% | 285.43 ms |
| Full hybrid | 20 | 60.00% | 77.98% | 100.00% | 64.50% | 369.37 ms |

### Test-Subset Split

| Track | Examples | Exact Match | Token F1 | Role Accuracy | Mean Confidence | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Classical baseline | 40 | 7.50% | 30.86% | 32.50% | 45.29% | 4.49 ms |
| Heuristic reranker | 40 | 20.00% | 41.76% | 60.00% | 56.17% | 5.84 ms |
| Transformer QA assist | 40 | 20.00% | 41.76% | 60.00% | 56.17% | 290.72 ms |
| Full hybrid | 40 | 20.00% | 41.76% | 60.00% | 56.17% | 380.99 ms |

### Combined Split

| Track | Examples | Exact Match | Token F1 | Role Accuracy | Mean Confidence | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Classical baseline | 60 | 8.33% | 36.24% | 28.33% | 45.43% | 3.96 ms |
| Heuristic reranker | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 4.98 ms |
| Transformer QA assist | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 288.96 ms |
| Full hybrid | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 377.11 ms |

Interpretation:

```text
The role-aware reranker and hybrid systems improve exact match, token F1, and role accuracy over the classical baseline. Transformer-assisted variants have similar quality on these splits but higher latency.
```

## 6. RAISE Model Comparison Metrics

Source:

```text
srlqa/output/tables/model_evaluation_summary.csv
```

Scope:

```text
challenge_suite_v2, 15 curated seed examples.
```

| Model | Ok / Error | Accuracy | Exact Match | Token Precision | Token Recall | Token F1 | BLEU | Role Accuracy | Confidence | Mean Latency | Median Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy PropQA-Net Baseline | 15 / 0 | 20.00% | 20.00% | 53.81% | 86.89% | 55.22% | 40.46% | 33.33% | 47.77% | 207.54 ms | 160.79 ms | 397.39 ms |
| Legacy Hybrid | 15 / 0 | 66.67% | 66.67% | 86.90% | 88.00% | 82.30% | 77.34% | 93.33% | 65.31% | 495.87 ms | 402.34 ms | 1005.50 ms |
| RAISE-SRL-QA Fast | 15 / 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 95.60% | 7.08 ms | 2.09 ms | 26.78 ms |
| RAISE-SRL-QA Model | 15 / 0 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 95.71% | 1294.63 ms | 362.54 ms | 5053.02 ms |

Interpretation:

```text
The RAISE fast system is the strongest live-demo choice in this table because it has perfect curated-suite performance with very low latency. The model-assisted version also scores perfectly on this suite but is slower.
```

## 7. RAISE Ablation Metrics

Source:

```text
srlqa/output/tables/ablation_study.csv
```

| Variant | Change | Scope | Accuracy | Token F1 | Role Accuracy | Delta F1 vs Previous | Delta F1 vs Legacy Baseline |
|---|---|---|---:|---:|---:|---:|---:|
| Legacy baseline | Legacy checkpoint baseline | challenge_suite_v2 seed suite | 20.00% | 55.22% | 33.33% | not applicable | 0.00 |
| Legacy hybrid | Legacy hybrid additions | challenge_suite_v2 seed suite | 66.67% | 82.30% | 93.33% | +27.07 points | +27.07 points |
| RAISE-SRL-QA Fast | RAISE retrieval + verifier + deterministic rules | challenge_suite_v2 seed suite | 100.00% | 100.00% | 100.00% | +17.70 points | +44.78 points |
| RAISE-SRL-QA Model | RAISE plus transformer QA candidates | challenge_suite_v2 seed suite | 100.00% | 100.00% | 100.00% | +0.00 points | +44.78 points |

Interpretation:

```text
The ablation table shows that adding role-aware hybrid behavior and then RAISE-style retrieval and verification gives the main quality gain on the curated seed suite.
```

## 8. All-Model Smoke Example

Source:

```text
srlqa/plots/EXACT_VALUES.json
```

Scope:

```text
One courier example used for smoke testing and demonstration.
```

Example:

```text
Context: The courier delivered the package to the office at noon.
Question: Where was the package delivered?
Expected answer: to the office
Expected role: ARGM-LOC
```

| Model | Answer | Exact Match | Token F1 | Role Accuracy | Role | Confidence | Latency |
|---|---|---:|---:|---:|---|---:|---:|
| RAISE-SRL-QA Fast | `to the office` | 100.00% | 100.00% | 100.00% | `ARGM-LOC` | 99.55% | 54.91 ms |
| RAISE-SRL-QA Model | `to the office` | 100.00% | 100.00% | 100.00% | `ARGM-LOC` | 99.55% | 21470.03 ms |
| Legacy Hybrid | `to the office` | 100.00% | 100.00% | 100.00% | `ARGM-LOC` | 65.79% | 3009.40 ms |
| Legacy PropQA-Net Baseline | `delivered the package to the office at noon` | 0.00% | 54.55% | 0.00% | `ARG1` | 41.98% | 114.48 ms |

Smoke-test mean:

| Metric | Value |
|---|---:|
| Mean exact match | 75.00% |
| Mean token F1 | 88.64% |
| Mean role accuracy | 75.00% |

Interpretation:

```text
This one-example smoke test demonstrates why role selection matters: the baseline extracts too large a span with the wrong role, while role-aware systems select the location span.
```

## 9. SRL + RAG Demo Metrics

Source:

```text
srl_rag_demo/smoke_test.py
```

Scope:

```text
Local functional smoke test for PropBank loading, retrieval, QA, and graph construction.
```

| Metric | Value |
|---|---:|
| PropBank instances loaded through NLTK | 112,917 |
| Treebank-backed usable count in demo loader | 9,353 |
| Built PropBank demo documents | 40 |
| Retrieval backend in smoke test | TF-IDF |
| Retrieval hits | 1 |
| Smoke question | `Where was the package delivered?` |
| Smoke answer | `to the office` |
| Smoke role | `ARGM-LOC` |
| Graph nodes | 12 |
| Graph edges | 15 |

Interpretation:

```text
The SRL + RAG demo is verified as a functional local demo. It should be presented as an explainability demo, not as a full benchmark replacement.
```

## 10. LoRA And QLoRA FAST_DEV Metrics

Sources:

```text
propbank_srlqa_artifacts/research_summary.txt
propbank_srlqa_2b_artifacts/research_summary_2b.txt
```

### DistilBERT LoRA Extractive QA

| Metric | Value |
|---|---:|
| Runtime mode | FAST_DEV_RUN |
| Model | `distilbert-base-cased-distilled-squad` |
| Tuning method | LoRA |
| Train / validation / test examples | 57 / 7 / 8 |
| Baseline validation token F1 | 31.09% |
| Fine-tuned test token F1 | 41.51% |
| Observed F1 change | +10.42 points |
| Test exact match | 12.50% |
| Test span accuracy | 12.50% |

### Tiny-GPT2 / Gemma 2B QLoRA Scaffold

| Metric | Value |
|---|---:|
| Runtime mode | FAST_DEV_RUN |
| Model used in execution | `sshleifer/tiny-gpt2` |
| Full Colab target model | `google/gemma-2-2b-it` |
| Tuning method | LoRA |
| Train / validation / test examples | 64 / 8 / 8 |
| Baseline validation exact match | 6.25% |
| Baseline validation token F1 | 14.87% |
| Fine-tuned test exact match in summary | 37.50% |
| Fine-tuned test precision in summary | 42.13% |
| Fine-tuned test recall in summary | 39.86% |
| Fine-tuned test token F1 in summary | 40.97% |
| 80% exact-match target met | False |
| 80% token-F1 target met | False |

Important caveat:

```text
The QLoRA scaffold is an unstable FAST_DEV artifact. The summary values should not be used as a final Gemma 2B claim unless the generated prediction CSV and a full evaluation run are reconciled.
```

## 11. Metric Coverage By Implementation

| Implementation | Exact Match | Token Precision | Token Recall | Token F1 | BLEU | Role Accuracy | SRL Micro F1 | SRL Macro F1 | BIO Accuracy | Confidence | Latency | Graph Metrics |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Legacy PropQA-Net full baseline | yes | no | no | yes | no | per prediction / benchmark | yes | yes | yes | prediction-level | no | no |
| Legacy benchmark tracks | yes | no | no | yes | no | yes | shared baseline | shared baseline | shared baseline | yes | yes | no |
| RAISE model comparison | yes | yes | yes | yes | yes | yes | linked baseline | linked baseline | linked baseline | yes | yes | no |
| RAISE ablation study | accuracy only | no | no | yes | no | yes | no | no | no | no | no | no |
| All-model smoke example | yes | no | no | yes | no | yes | no | no | no | yes | yes | no |
| SRL + RAG demo smoke test | functional correctness | no | no | smoke answer correct | no | yes | no | no | no | app confidence | not benchmarked | yes |
| DistilBERT LoRA FAST_DEV | yes | no | no | yes | no | no | no | no | no | no | no | no |
| Tiny-GPT2 / Gemma 2B QLoRA FAST_DEV | yes | yes | yes | yes | no | no | no | no | no | no | no | no |

## 12. Which Metric To Use For Each Claim

| Claim Type | Use This Metric | Best Supported Value |
|---|---|---|
| Full baseline answer quality | QA token F1 | 76.12% |
| Full baseline strict correctness | QA exact match | 51.84% |
| Full baseline SRL quality | SRL micro F1 | 71.33% |
| Full baseline token tagging | BIO accuracy | 81.63% |
| Rare-role SRL difficulty | SRL macro F1 | 16.19% |
| Challenge-suite role selection | Role accuracy | 100.00% for hybrid challenge tracks, 100.00% for RAISE seed suite |
| RAISE curated-suite answer quality | Exact match and token F1 | 100.00% on 15 curated examples |
| Live demo correctness | Smoke answer and graph metrics | `to the office`, `ARGM-LOC`, 12 graph nodes, 15 graph edges |
| LoRA feasibility | FAST_DEV token F1 | 41.51% for DistilBERT LoRA |
| QLoRA scaffold progress | FAST_DEV summary token F1 | 40.97%, with caveat |

## 13. Recommended Final Comparison Statement

Use this in a report or presentation:

```text
The legacy PropQA-Net baseline achieves 51.84% exact match, 76.12% QA token F1, 71.33% SRL micro F1, and 81.63% BIO accuracy on the full PropBank-derived baseline evaluation. On challenge-style role questions, the role-aware heuristic and hybrid tracks improve over the classical baseline, reaching 60.00% exact match, 77.98% token F1, and 100.00% role accuracy on the 20-example challenge split. In the newer RAISE-SRL-QA framework, the fast and model-assisted pipelines reach 100.00% exact match, 100.00% token F1, and 100.00% role accuracy on a controlled 15-example curated seed suite. The SRL + RAG Streamlit demo verifies the final explainability workflow by retrieving evidence, selecting the answer "to the office", assigning the role ARGM-LOC, and building a 12-node, 15-edge reasoning graph.
```

## 14. Claim Discipline

Use these rules:

- Use the legacy PropQA-Net full baseline metrics for full-test claims.
- Use the benchmark track metrics for challenge, test-subset, and combined split claims.
- Use the RAISE 100% metrics only for the 15-example curated seed-suite claim.
- Use SRL + RAG demo metrics only for functional demo and explainability claims.
- Use LoRA and QLoRA metrics only as FAST_DEV experiment evidence.
- Do not claim the QLoRA scaffold as a final Gemma 2B result without a full consistent evaluation run.

