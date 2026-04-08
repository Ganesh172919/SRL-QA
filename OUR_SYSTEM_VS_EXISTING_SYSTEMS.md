# Our System Vs Existing Systems

## Purpose

This document compares our final SRL + RAG explainable QA system with the existing systems in the project and with common external QA system categories.

The comparison focuses on:

- implementation design,
- metrics,
- explainability,
- retrieval strategy,
- demo readiness,
- limitations and fair claims.

## 1. What Is "Our System"?

In this comparison, "our system" mainly means the final standalone demo:

```text
srl_rag_demo/
```

It combines:

```text
PropBank SRL data
    -> SRL structured documents
    -> hybrid retrieval
    -> role-aware QA
    -> explainable semantic graph
    -> Streamlit demo
```

Important note:

```text
The strongest measured QA benchmark numbers are in srl_qa_project and srlqa.
The final srl_rag_demo system is the integrated explainability demo that connects SRL, RAG, QA, and graph reasoning.
```

So the comparison separates:

- full-test baseline metrics,
- challenge-suite metrics,
- curated seed-suite metrics,
- SRL + RAG smoke-test demo metrics.

## 2. High-Level Implementation Comparison

| System | Main Folder | Implementation Type | Retrieval | SRL Usage | Explainability | Demo Ready |
|---|---|---|---|---|---|---|
| Legacy PropQA-Net baseline | `srl_qa_project/` | Baseline SRL-QA model | No RAG retrieval | Uses PropBank-derived SRL QA examples | Limited, mostly prediction output | Yes, but legacy style |
| Classical baseline benchmark | `srl_qa_project/` | Classical baseline track | No RAG retrieval | Uses shared SRL model outputs | Limited | For benchmark comparison |
| Heuristic reranker | `srl_qa_project/` | Role-aware heuristic reranking | No document RAG | Uses question-role mapping | Better role-level explanation | For benchmark comparison |
| Transformer QA assist | `srl_qa_project/` | Hybrid local model assistance | No document RAG | Uses SRL plus model candidate | Some explanation through role match | Slower |
| Full hybrid | `srl_qa_project/` | Hybrid heuristic + model system | No document RAG | Uses SRL role matching | Better than baseline | Slower |
| RAISE-SRL-QA Fast | `srlqa/` | Deterministic role-aware pipeline | Frame-aware retrieval | Strong SRL and frame usage | Strong reasoning summaries | Good for fast comparison |
| RAISE-SRL-QA Model | `srlqa/` | Model-assisted RAISE pipeline | Frame-aware retrieval | Strong SRL and frame usage | Strong reasoning summaries | Slower |
| Our SRL + RAG demo | `srl_rag_demo/` | Integrated SRL + RAG + graph app | Hybrid embedding / TF-IDF retrieval | Builds SRL documents and role-aware QA | Strong graph explanation | Best final demo |
| DistilBERT LoRA artifact | `propbank_srlqa_artifacts/` | FAST_DEV extractive QA fine-tuning | No RAG retrieval | Predicate-aware QA prompts | Limited | Research artifact |
| Tiny-GPT2 / Gemma QLoRA scaffold | `propbank_srlqa_2b_artifacts/` | FAST_DEV generative QA scaffold | No RAG retrieval | Predicate-aware prompts | Limited | Research artifact |

## 3. Main Metrics Comparison Across Project Implementations

| System | Scope | Examples | Exact Match / Accuracy | Token F1 | Role Accuracy | Mean Latency | Main Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Legacy PropQA-Net full baseline | Full baseline test | full test | 51.84% | 76.12% | not separately reported | not reported | Best full-test baseline metric row. |
| Classical baseline | Challenge benchmark | 20 | 10.00% | 47.01% | 20.00% | 2.89 ms | Fast but weak on role-specific questions. |
| Heuristic reranker | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 3.26 ms | Strong improvement from role-aware reranking. |
| Transformer QA assist | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 285.43 ms | Same quality on this split, slower due to model assistance. |
| Full hybrid | Challenge benchmark | 20 | 60.00% | 77.98% | 100.00% | 369.37 ms | Complete legacy hybrid, higher latency. |
| Legacy PropQA-Net baseline | RAISE curated seed suite | 15 | 20.00% | 55.22% | 33.33% | 207.54 ms | Baseline comparison inside RAISE evaluation. |
| Legacy Hybrid | RAISE curated seed suite | 15 | 66.67% | 82.30% | 93.33% | 495.87 ms | Hybrid improves answer and role accuracy. |
| RAISE-SRL-QA Fast | RAISE curated seed suite | 15 | 100.00% | 100.00% | 100.00% | 7.08 ms | Best curated-suite speed and accuracy. |
| RAISE-SRL-QA Model | RAISE curated seed suite | 15 | 100.00% | 100.00% | 100.00% | 1294.63 ms | Same curated-suite accuracy, much slower. |
| Our SRL + RAG demo | Smoke test | 1 demo case | 1/1 correct | 100.00% on smoke case | `ARGM-LOC` correct | not benchmarked | Best integrated explainability demo. |
| DistilBERT LoRA | FAST_DEV test | 8 | 12.50% | 41.51% | not reported | not reported | Early fine-tuning feasibility result. |
| Tiny-GPT2 / Gemma QLoRA scaffold | FAST_DEV test | 8 | 37.50% in summary | 40.97% in summary | not reported | not reported | Unstable dev artifact, not a final Gemma claim. |

Important interpretation:

```text
Our final srl_rag_demo is the best integrated demo because it combines SRL, RAG, QA, and graph explanation. The strongest local benchmark numbers are from the RAISE curated seed suite and the legacy full baseline metrics.
```

## 4. Full Baseline Metric Comparison

Source:

```text
srl_qa_project/results/metrics.json
srlqa/output/tables/baseline_metrics_summary.csv
```

| Metric | Legacy PropQA-Net Full Baseline |
|---|---:|
| QA exact match | 51.84% |
| QA token F1 | 76.12% |
| SRL micro precision | 73.20% |
| SRL micro recall | 69.55% |
| SRL micro F1 | 71.33% |
| SRL macro F1 | 16.19% |
| SRL BIO accuracy | 81.63% |
| Best validation F1 | 77.27% |
| Parameters | 1,784,352 |
| Total PropBank instances | 112,917 |
| QA pair count | 23,007 |

How our system connects:

```text
The final SRL + RAG demo reuses the same project direction, PropBank SRL grounding, and local NLTK data. It adds retrieval and graph explanation on top of the SRL-QA idea.
```

## 5. Challenge Benchmark Comparison

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

### Combined Split

| Track | Examples | Exact Match | Token F1 | Role Accuracy | Mean Confidence | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Classical baseline | 60 | 8.33% | 36.24% | 28.33% | 45.43% | 3.96 ms |
| Heuristic reranker | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 4.98 ms |
| Transformer QA assist | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 288.96 ms |
| Full hybrid | 60 | 33.33% | 53.83% | 73.33% | 58.95% | 377.11 ms |

Implementation lesson:

```text
Adding role-aware reranking gives a large improvement over the classical baseline. Adding transformer assistance did not improve these benchmark scores further, but it increased latency.
```

## 6. RAISE System Comparison

Source:

```text
srlqa/output/tables/model_evaluation_summary.csv
```

Scope:

```text
challenge_suite_v2, 15 curated seed examples.
```

| System | Accuracy | Exact Match | Token Precision | Token Recall | Token F1 | BLEU | Role Accuracy | Confidence | Mean Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy PropQA-Net Baseline | 20.00% | 20.00% | 53.81% | 86.89% | 55.22% | 40.46% | 33.33% | 47.77% | 207.54 ms | 397.39 ms |
| Legacy Hybrid | 66.67% | 66.67% | 86.90% | 88.00% | 82.30% | 77.34% | 93.33% | 65.31% | 495.87 ms | 1005.50 ms |
| RAISE-SRL-QA Fast | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 95.60% | 7.08 ms | 26.78 ms |
| RAISE-SRL-QA Model | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 95.71% | 1294.63 ms | 5053.02 ms |

Implementation lesson:

```text
RAISE-SRL-QA Fast is the best speed-quality tradeoff on the curated seed suite. The model-assisted version has the same score on this small suite but is much slower.
```

## 7. Our Final SRL + RAG Demo Comparison

Source:

```text
srl_rag_demo/smoke_test.py
```

| Feature | Our SRL + RAG Demo |
|---|---|
| Loads PropBank through NLTK | Yes |
| Uses local PropBank / Treebank assets | Yes |
| Reuses PropBank frame store | Yes |
| Supports pasted or uploaded user documents | Yes |
| Uses RAG retrieval | Yes |
| Uses embeddings when available | Yes |
| Uses TF-IDF fallback | Yes |
| Uses role-aware answer selection | Yes |
| Optional transformer QA | Yes |
| Builds semantic graph explanation | Yes |
| Streamlit app | Yes |
| External API key required | No |

Smoke-test results:

| Metric | Value |
|---|---:|
| PropBank instances loaded | 112,917 |
| Treebank-backed usable count in demo loader | 9,353 |
| Built PropBank demo documents | 40 |
| Retrieval backend in smoke test | TF-IDF |
| Smoke question | `Where was the package delivered?` |
| Smoke answer | `to the office` |
| Smoke role | `ARGM-LOC` |
| Graph nodes | 12 |
| Graph edges | 15 |

Why it matters:

```text
The final demo is stronger than a plain metric table because it shows the answer, retrieved evidence, SRL structure, and reasoning graph in one interactive app.
```

## 8. Comparison With Common External QA System Categories

These are conceptual comparisons. They should not be treated as metric comparisons unless all systems are run on the same dataset and benchmark.

| Existing System Category | Typical Implementation | Limitation Compared With Our System | Our Improvement |
|---|---|---|---|
| Keyword retrieval QA | Retrieves documents using keyword overlap | Does not understand semantic roles | Our demo includes SRL triples and role-aware answer selection. |
| Plain extractive QA | Selects answer span from context | May over-select spans or ignore role meaning | Our system uses expected semantic roles like `ARGM-LOC` for where questions. |
| Standard RAG | Retrieves raw text and answers from it | Retrieval evidence may not be semantically structured | Our RAG documents include predicates, roles, spans, and frame hints. |
| Neural QA only | Uses model confidence to pick answer | Can be hard to explain | Our system shows graph reasoning with predicate and role nodes. |
| SRL-only pipeline | Extracts semantic roles | May not include retrieval or user-facing QA | Our system connects SRL to retrieval, QA, and Streamlit visualization. |
| Generative QA | Generates free-form answers | May hallucinate or produce non-extractive answers | Our default answer is extractive and tied to evidence spans. |

Fairness note:

```text
External systems are not assigned numeric metrics here because they were not evaluated inside this project on the same PropBank-derived benchmark. The fair numeric comparison is between the local systems listed above.
```

## 9. Implementation Advantage Of Our System

| Requirement | Existing Baselines | Our SRL + RAG Demo |
|---|---|---|
| Uses PropBank data | Yes | Yes |
| Loads data through NLTK | Yes | Yes |
| Builds QA examples | Yes | Uses SRL docs and QA candidates |
| Uses RAG | No in legacy baseline | Yes |
| Supports user documents | Limited | Yes, pasted/uploaded text |
| Has retrieval fallback | Not central | Yes, TF-IDF fallback |
| Has role-aware QA | Partial in hybrid systems | Yes |
| Has PropBank frame hints | Yes in `srlqa` | Yes, reused from `srlqa` |
| Has graph explanation | No or limited | Yes |
| Has Streamlit final demo | Some legacy apps | Yes, focused final demo |
| Needs external API | No | No |
| Best use | Benchmark / model comparison | Live explainable QA demo |

## 10. Metric Advantage Summary

| Comparison | Result |
|---|---|
| Legacy full baseline | Strongest full-test reference: 76.12% QA token F1 and 71.33% SRL micro F1. |
| Classical vs heuristic challenge benchmark | Token F1 improves from 47.01% to 77.98%; role accuracy improves from 20.00% to 100.00%. |
| Legacy baseline vs RAISE fast seed suite | Token F1 improves from 55.22% to 100.00%; role accuracy improves from 33.33% to 100.00% on the curated 15-example suite. |
| RAISE fast vs RAISE model | Same curated-suite accuracy, but RAISE fast is much lower latency. |
| Our SRL + RAG demo vs previous systems | Adds the most complete user-facing workflow: retrieval, QA, evidence display, and graph explanation. |

## 11. Which System Should Be Presented As Best?

Use different "best" labels depending on the claim:

| Claim | Best System To Mention | Reason |
|---|---|---|
| Best full-test baseline metric | Legacy PropQA-Net full baseline | It has the full baseline test metrics. |
| Best curated seed-suite score | RAISE-SRL-QA Fast / Model | Both achieve 100% on 15 curated examples. |
| Best speed-quality tradeoff on seed suite | RAISE-SRL-QA Fast | 100% curated score with 7.08 ms mean latency. |
| Best final demo | Our `srl_rag_demo` | It integrates SRL, RAG, QA, graph explanation, and Streamlit. |
| Best future research direction | Our SRL + RAG demo plus graph scoring | It can grow into multi-hop explainable semantic reasoning. |

## 12. Presentation-Ready Comparison Speech

Use this for a professor explanation:

```text
Our final system improves over the earlier baselines mainly in implementation scope and explainability. The legacy PropQA-Net baseline gives the strongest full-test reference, with 51.84% exact match, 76.12% QA token F1, and 71.33% SRL micro F1. The role-aware heuristic and hybrid benchmark tracks show that semantic roles improve challenge-style QA, increasing token F1 from 47.01% to 77.98% and role accuracy from 20.00% to 100.00% on the 20-example challenge split. The RAISE-SRL-QA fast pipeline reaches 100.00% exact match and role accuracy on a controlled 15-example curated seed suite. Our final SRL + RAG Streamlit demo connects these ideas into one usable system: it loads PropBank through NLTK, retrieves SRL-structured evidence, selects answers using semantic roles, and visualizes the reasoning path as a graph.
```

## 13. Claim Discipline

Use these rules:

- Do not compare the RAISE 100% seed-suite result directly against the full-test baseline as if they are the same benchmark.
- Use the legacy full baseline for full-test claims.
- Use the RAISE comparison for curated seed-suite claims.
- Use the benchmark track comparison for challenge, test-subset, and combined split claims.
- Use the final SRL + RAG demo as the integrated explainability demo claim.
- Use LoRA and QLoRA only as FAST_DEV experiment artifacts.

## 14. Final Summary

```text
Existing baseline systems show the model and metric foundation.
RAISE systems show role-aware accuracy improvements on curated examples.
Our final SRL + RAG demo shows the complete explainable QA workflow.
```

The strongest overall project story is:

```text
Baseline SRL-QA proves the task.
Role-aware systems improve answer selection.
SRL + RAG graph demo makes the reasoning visible.
```

