# Complete Project Functionality And Accuracy Report

Generated at: `2026-04-08 11:44:35`

## Functional Status

| Check | Status |
|---|---|
| Compile `srl_rag_demo` | PASS |
| Controlled SRL + RAG demo evaluation | PASS |
| Retrieval backend | `tfidf` |

## Controlled SRL + RAG Demo Accuracy

> Scope: controlled demo-suite, not full-corpus benchmark.

| Metric | Value |
|---|---:|
| Examples | 8 |
| Exact Match | 100.00% |
| Token F1 | 100.00% |
| Role Accuracy | 100.00% |
| Mean Confidence | 78.14% |

## Legacy Full-Test Baseline Metrics

| Metric | Value |
|---|---:|
| QA exact match | 51.84% |
| QA token F1 | 76.12% |
| SRL micro F1 | 71.33% |
| SRL BIO accuracy | 81.63% |
| Best validation F1 | 77.27% |

## RAISE Curated Seed-Suite Metrics

| System | Examples | Exact Match | Token F1 | Role Accuracy | Mean Latency |
|---|---:|---:|---:|---:|---:|
| Legacy PropQA-Net Baseline | 15 | 20.00% | 55.22% | 33.33% | 207.54 ms |
| Legacy Hybrid | 15 | 66.67% | 82.30% | 93.33% | 495.87 ms |
| RAISE-SRL-QA Fast | 15 | 100.00% | 100.00% | 100.00% | 7.08 ms |
| RAISE-SRL-QA Model | 15 | 100.00% | 100.00% | 100.00% | 1294.63 ms |

## Legacy Benchmark Tracks

| Track | Scope | Examples | Exact Match | Token F1 | Role Accuracy | Mean Latency |
|---|---|---:|---:|---:|---:|---:|
| classical_baseline | challenge | 20 | 10.00% | 47.01% | 20.00% | 2.89 ms |
| classical_baseline | test_subset | 40 | 7.50% | 30.86% | 32.50% | 4.49 ms |
| classical_baseline | combined | 60 | 8.33% | 36.24% | 28.33% | 3.96 ms |
| heuristic_reranker | challenge | 20 | 60.00% | 77.98% | 100.00% | 3.26 ms |
| heuristic_reranker | test_subset | 40 | 20.00% | 41.76% | 60.00% | 5.84 ms |
| heuristic_reranker | combined | 60 | 33.33% | 53.83% | 73.33% | 4.98 ms |
| transformer_qa_assist | challenge | 20 | 60.00% | 77.98% | 100.00% | 285.43 ms |
| transformer_qa_assist | test_subset | 40 | 20.00% | 41.76% | 60.00% | 290.72 ms |
| transformer_qa_assist | combined | 60 | 33.33% | 53.83% | 73.33% | 288.96 ms |
| full_hybrid | challenge | 20 | 60.00% | 77.98% | 100.00% | 369.37 ms |
| full_hybrid | test_subset | 40 | 20.00% | 41.76% | 60.00% | 380.99 ms |
| full_hybrid | combined | 60 | 33.33% | 53.83% | 73.33% | 377.11 ms |

## Demo Per-Example Results

| Example | Expected | Predicted | Role | Exact | Token F1 |
|---|---|---|---|---:|---:|
| who_agent | `The chef` | `The chef` | `ARG0` | 100.00% | 100.00% |
| what_theme | `the medicine` | `the medicine` | `ARG1` | 100.00% | 100.00% |
| where_location | `to the office` | `to the office` | `ARGM-LOC` | 100.00% | 100.00% |
| when_time | `at noon` | `at noon` | `ARGM-TMP` | 100.00% | 100.00% |
| to_whom_recipient | `to her friend` | `to her friend` | `ARG2` | 100.00% | 100.00% |
| what_object_boundary | `the machine` | `the machine` | `ARG1` | 100.00% | 100.00% |
| how_manner | `carefully` | `carefully` | `ARGM-MNR` | 100.00% | 100.00% |
| why_cause | `because of budget cuts` | `because of budget cuts` | `ARGM-CAU` | 100.00% | 100.00% |

## Best Supported Accuracy Statements

- Full-test legacy baseline: 51.84% exact match, 76.12% QA token F1, 71.33% SRL micro F1, 81.63% BIO accuracy.
- RAISE curated 15-example seed suite: 100.00% exact match, 100.00% token F1, 100.00% role accuracy for both RAISE fast and model-assisted variants.
- Controlled final SRL + RAG demo suite: 100.00% exact match, 100.00% token F1, 100.00% role accuracy.

## Claim Discipline

- Use full-test metrics only for the legacy baseline full-test claim.
- Use RAISE 100% metrics only for the curated 15-example seed suite.
- Use the controlled demo-suite metrics only to show the final SRL + RAG demo path is functional.
