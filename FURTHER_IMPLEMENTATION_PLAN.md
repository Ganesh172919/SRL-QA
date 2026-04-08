# Further Implementation Plan

## Purpose

This document converts the future innovation ideas into a practical implementation plan.

It explains:

- what to build next,
- which project folders should be updated,
- which modules should be added,
- how to test each feature,
- how to evaluate the improvements,
- how to present the completed work.

The goal is to extend the current project from:

```text
SRL + RAG demo with one reasoning path
```

to:

```text
SRL + RAG system with stronger user-document SRL, multi-hop graph reasoning, hybrid verification, and a unified evaluation benchmark.
```

## 1. Current Implementation Baseline

The current workspace already contains:

| Folder | Current Role |
|---|---|
| `srl_qa_project/` | Legacy PropQA-Net baseline and benchmark results. |
| `srlqa/` | RAISE-SRL-QA framework with frame retrieval, verifier, correction, comparison tables, and plots. |
| `srl_rag_demo/` | Standalone Streamlit SRL + RAG demo with PropBank loading, retrieval, role-aware QA, and graph explanation. |
| `propbank_srlqa_artifacts/` | DistilBERT LoRA FAST_DEV artifacts. |
| `propbank_srlqa_2b_artifacts/` | Tiny-GPT2 / Gemma 2B QLoRA scaffold artifacts. |

The best next implementation path should mainly extend:

```text
srl_rag_demo/
srlqa/
```

and leave the legacy project mostly unchanged except for optional benchmark comparison references.

## 2. Implementation Principles

Use these rules while implementing:

- Keep the final demo local and CPU-friendly.
- Keep API keys optional and avoid external service dependency.
- Prefer deterministic SRL/RAG logic for the live demo.
- Add neural models only as optional modules.
- Keep all new generated data and indexes inside the relevant project folder.
- Do not mix full-test metrics with seed-suite metrics.
- Add smoke tests for every major feature.
- Keep explanation output readable for a professor or evaluator.

## 3. Proposed Implementation Phases

| Phase | Feature | Main Folder | Priority |
|---|---|---|---|
| Phase 1 | Stronger user-document SRL pipeline | `srl_rag_demo/` | High |
| Phase 2 | Graph-based evidence scoring | `srl_rag_demo/` | High |
| Phase 3 | Multi-hop SRL graph reasoning | `srl_rag_demo/` and `srlqa/` | High |
| Phase 4 | Hybrid neural-symbolic verifier | `srlqa/` and `srl_rag_demo/` | Medium |
| Phase 5 | Unified evaluation benchmark | `srlqa/` and new benchmark folder | High |
| Phase 6 | Better Streamlit comparison UI | `srl_rag_demo/` | Medium |
| Phase 7 | No-answer and contradiction handling | `srl_rag_demo/` | Medium |
| Phase 8 | Full fine-tuning pipeline | artifact/training folder | Lower for live demo, high for research |

## 4. Phase 1: Stronger User-Document SRL Pipeline

### Goal

Improve how pasted or uploaded documents are converted into SRL-like structured documents.

### Current Behavior

The existing demo can ingest user text, but arbitrary user documents do not have gold PropBank annotations. The app currently uses a simple fallback method.

### Implementation Tasks

Add or improve these files:

| File | Change |
|---|---|
| `srl_rag_demo/user_docs.py` | Add better sentence splitting, predicate detection, and role candidate extraction. |
| `srl_rag_demo/data_models.py` | Add optional fields for parser source, parse confidence, and predicate detection method. |
| `srl_rag_demo/app.py` | Show whether a document came from PropBank gold SRL or user-document heuristic SRL. |
| `srl_rag_demo/smoke_test.py` | Add smoke test for uploaded/pasted document parsing. |

### Suggested Implementation Steps

1. Add a sentence splitter using `nltk.sent_tokenize` when available and a regex fallback otherwise.
2. Add a simple predicate detector based on verb-like tokens.
3. Add candidate argument span extraction using preposition and noun phrase heuristics.
4. Add role guesses:
   - `Where`-like prepositional phrases -> `ARGM-LOC`
   - time expressions -> `ARGM-TMP`
   - manner adverbs -> `ARGM-MNR`
   - before-predicate noun phrase -> likely `ARG0`
   - after-predicate noun phrase -> likely `ARG1`
5. Store a `parser_source` field such as `propbank_gold`, `user_heuristic`, or `model_srl`.

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Pasted courier sentence | Creates predicate `delivered`. |
| Pasted courier sentence | Creates role candidate `to the office`. |
| Pasted courier sentence | Maps `to the office` to `ARGM-LOC`. |
| App evidence view | Shows user document source and role guesses. |
| Smoke test | Passes without internet or API keys. |

## 5. Phase 2: Graph-Based Evidence Scoring

### Goal

Use semantic graph features to improve answer selection and explanation quality.

### Implementation Tasks

Add or improve these files:

| File | Change |
|---|---|
| `srl_rag_demo/graphing.py` | Add graph path scoring and edge metadata. |
| `srl_rag_demo/qa.py` | Use graph path score during answer candidate ranking. |
| `srl_rag_demo/data_models.py` | Add `graph_score`, `role_score`, `frame_score`, and `explainability_score` fields. |
| `srl_rag_demo/app.py` | Display score breakdown for the final answer. |

### Suggested Score Formula

```text
final_score =
    0.35 * retrieval_score
    + 0.25 * role_match_score
    + 0.20 * frame_compatibility_score
    + 0.10 * extractability_score
    + 0.10 * graph_path_score
```

### Score Components

| Score | Meaning |
|---|---|
| `retrieval_score` | How strongly the document matches the question. |
| `role_match_score` | Whether the candidate role matches the question type. |
| `frame_compatibility_score` | Whether the role is supported by the predicate frame. |
| `extractability_score` | Whether the answer span is directly present in the source text. |
| `graph_path_score` | Whether the answer is connected through a clean graph path. |

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Courier example | `to the office` remains top answer. |
| Overlong baseline span | Penalized by extractability and role score. |
| App output | Shows score breakdown. |
| Graph JSON | Contains edge scores and node types. |

## 6. Phase 3: Multi-Hop SRL Graph Reasoning

### Goal

Support questions that need more than one event or predicate.

### Example

```text
Sentence 1: The courier picked up the package from the warehouse.
Sentence 2: The courier delivered the package to the office.
Question: Where did the package go after it was picked up?
Answer: to the office
```

### Implementation Tasks

Add or improve:

| File | Change |
|---|---|
| `srl_rag_demo/graphing.py` | Add multi-document graph construction. |
| `srl_rag_demo/qa.py` | Add shared-argument chain detection. |
| `srl_rag_demo/retrieval.py` | Retrieve multiple supporting documents instead of only top single evidence. |
| `srl_rag_demo/smoke_test.py` | Add two-sentence multi-hop smoke test. |
| `srlqa/verification/` | Optionally add reusable multi-hop verification utilities. |

### Multi-Hop Logic

```text
Find event A from question clue
Find shared argument between event A and event B
Find event B containing target role
Select answer from target role in event B
Return graph path across both events
```

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Two-event courier example | Finds both `picked up` and `delivered`. |
| Shared argument | Links both events through `the package`. |
| Final answer | Selects `to the office`. |
| Graph view | Shows both events and the shared argument. |

## 7. Phase 4: Hybrid Neural-Symbolic Verifier

### Goal

Create a verifier that combines neural QA confidence with symbolic SRL checks.

### Implementation Tasks

Add or improve:

| File | Change |
|---|---|
| `srlqa/verification/hybrid_verifier.py` | Add reusable neural-symbolic verification logic. |
| `srl_rag_demo/qa.py` | Call verifier before final answer selection. |
| `srl_rag_demo/app.py` | Display verifier decisions and rejection reasons. |
| `srlqa/output/tables/` | Add verifier comparison outputs after evaluation. |

### Verifier Checks

| Check | Purpose |
|---|---|
| Extractability | Answer must appear in evidence text. |
| Role match | Predicted role should match question type. |
| Frame compatibility | Predicate frame should support role. |
| Retrieval support | Evidence document should be highly ranked. |
| Neural score | Optional transformer QA confidence. |
| Graph connectivity | Candidate must be connected to the reasoning path. |

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Correct courier answer | Accepted by verifier. |
| Wrong role candidate | Penalized or rejected. |
| Non-extractive answer | Rejected or flagged. |
| App UI | Shows accepted/rejected candidate reasons. |

## 8. Phase 5: Unified Evaluation Benchmark

### Goal

Create one consistent benchmark suite for all future comparisons.

### Proposed New Folder

```text
evaluation_suite/
```

Suggested files:

| File | Purpose |
|---|---|
| `evaluation_suite/datasets/challenge_roles.jsonl` | Hard role questions. |
| `evaluation_suite/datasets/multi_hop_srl.jsonl` | Multi-hop SRL questions. |
| `evaluation_suite/datasets/user_doc_rag.jsonl` | User-document style RAG questions. |
| `evaluation_suite/run_evaluation.py` | Run all systems on benchmark splits. |
| `evaluation_suite/metrics.py` | Exact match, token F1, role accuracy, retrieval metrics, graph metrics. |
| `evaluation_suite/report.py` | Generate markdown and CSV reports. |
| `evaluation_suite/README.md` | Explain how to run benchmark. |

### Benchmark Splits

| Split | Purpose |
|---|---|
| `challenge_roles` | Tests where, when, why, how, who, and what role questions. |
| `multi_hop_srl` | Tests reasoning across multiple predicates. |
| `user_doc_rag` | Tests arbitrary user-document retrieval and QA. |
| `no_answer` | Tests whether the system can abstain. |
| `contradiction` | Tests whether the system detects conflicting evidence. |

### Metrics To Implement

| Metric | Formula / Meaning |
|---|---|
| Exact Match | Exact normalized answer match. |
| Token Precision | Overlap tokens divided by predicted tokens. |
| Token Recall | Overlap tokens divided by gold tokens. |
| Token F1 | Harmonic mean of token precision and recall. |
| Role Accuracy | Predicted role equals gold role. |
| Recall@K | Correct evidence appears in top K retrieved documents. |
| MRR | Reciprocal rank of correct evidence. |
| nDCG | Ranked retrieval quality. |
| Graph Path Accuracy | Correct evidence path appears in graph. |
| No-Answer Accuracy | Correctly abstains when no answer exists. |
| Contradiction Accuracy | Detects conflicting answers. |
| Mean Latency | Average answer time. |
| P95 Latency | 95th percentile answer time. |

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Benchmark runner | Runs without external API keys. |
| Output CSV | Contains one row per system per split. |
| Output markdown | Contains summary comparison tables. |
| Metrics | Include QA, role, retrieval, graph, and latency metrics. |

## 9. Phase 6: Better Streamlit Comparison UI

### Goal

Make the demo stronger for professor presentation.

### Implementation Tasks

Improve:

| File | Change |
|---|---|
| `srl_rag_demo/app.py` | Add comparison tab. |
| `srl_rag_demo/qa.py` | Return candidate list and score breakdown. |
| `srl_rag_demo/graphing.py` | Add color-coded graph nodes by role and source. |
| `srl_rag_demo/retrieval.py` | Add retrieval result diagnostics. |

### New UI Tabs

| Tab | Purpose |
|---|---|
| QA | Final answer and confidence. |
| Evidence | Retrieved documents and SRL triples. |
| Graph | Visual reasoning path. |
| Compare | Compare deterministic QA, optional transformer QA, and graph-scored QA. |
| Metrics | Show smoke-test and benchmark summary. |
| Debug | Show candidate spans, scores, and rejection reasons. |

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Compare tab | Shows at least two answer strategies. |
| Score breakdown | Visible for top answer. |
| Graph | Color-codes question, document, predicate, role, candidate, and answer. |
| Export | Download graph JSON and answer report. |

## 10. Phase 7: No-Answer And Contradiction Handling

### Goal

Make the QA system safer and more realistic.

### Implementation Tasks

Add:

| File | Change |
|---|---|
| `srl_rag_demo/qa.py` | Add no-answer threshold and contradiction candidate detection. |
| `srl_rag_demo/graphing.py` | Show conflicting candidate answer nodes. |
| `srl_rag_demo/app.py` | Display warnings for unsupported or conflicting answers. |
| `evaluation_suite/datasets/no_answer.jsonl` | Add no-answer test examples. |
| `evaluation_suite/datasets/contradiction.jsonl` | Add contradiction examples. |

### Contradiction Example

```text
Document 1: The courier delivered the package to the office.
Document 2: The courier delivered the package to the warehouse.
Question: Where was the package delivered?
```

Expected behavior:

```text
Show both candidate answers and flag contradiction.
```

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| Missing evidence question | Returns low-confidence or no-answer warning. |
| Conflicting documents | Shows multiple candidate answers. |
| Graph view | Highlights conflict paths. |
| Evaluation | Reports no-answer and contradiction accuracy. |

## 11. Phase 8: Full Fine-Tuning Implementation

### Goal

Move FAST_DEV model artifacts toward full-scale experiments.

### Proposed Folder

```text
training_experiments/
```

Suggested files:

| File | Purpose |
|---|---|
| `training_experiments/prepare_propbank_qa.py` | Convert PropBank examples for model training. |
| `training_experiments/train_extractive_qa.py` | Train BERT/RoBERTa/DeBERTa style extractive QA. |
| `training_experiments/train_lora.py` | Train parameter-efficient LoRA model. |
| `training_experiments/train_qlora_generative.py` | Train QLoRA generative model if GPU is available. |
| `training_experiments/evaluate_models.py` | Evaluate exact match, token F1, and role accuracy. |
| `training_experiments/configs/` | Store model and dataset configs. |

### Training Controls

Use:

- fixed random seeds,
- train/validation/test splits,
- early stopping,
- full metrics CSV,
- model card style summary,
- clear FAST_DEV vs FULL_RUN flags.

### Acceptance Criteria

| Check | Expected Result |
|---|---|
| FAST_DEV mode | Runs quickly on small sample. |
| FULL_RUN mode | Uses full prepared dataset when compute is available. |
| Output metrics | Exact match, token F1, role accuracy, latency. |
| Output docs | Generates a research summary markdown. |

## 12. Suggested File Change Plan

| Priority | File / Folder | Type | Purpose |
|---|---|---|---|
| 1 | `srl_rag_demo/user_docs.py` | Modify | Stronger user document SRL. |
| 2 | `srl_rag_demo/data_models.py` | Modify | Add scoring and parser metadata fields. |
| 3 | `srl_rag_demo/qa.py` | Modify | Graph scoring, verifier hooks, no-answer handling. |
| 4 | `srl_rag_demo/graphing.py` | Modify | Multi-hop graph and scored edges. |
| 5 | `srl_rag_demo/retrieval.py` | Modify | Multi-document evidence support and retrieval diagnostics. |
| 6 | `srl_rag_demo/app.py` | Modify | Add compare, metrics, and debug tabs. |
| 7 | `srl_rag_demo/smoke_test.py` | Modify | Add tests for new behavior. |
| 8 | `srlqa/verification/hybrid_verifier.py` | Add | Reusable verifier. |
| 9 | `evaluation_suite/` | Add | Unified benchmark runner and datasets. |
| 10 | `training_experiments/` | Add | Full model training plan and scripts. |

## 13. Minimal Viable Implementation

If time is limited, implement only this subset:

```text
1. Add graph-based score breakdown.
2. Add stronger user-document SRL heuristics.
3. Add comparison tab to Streamlit.
4. Add one multi-hop smoke example.
5. Add unified evaluation markdown/CSV report generator.
```

This gives the best improvement for a professor demo without requiring heavy GPU training.

## 14. Testing Plan

| Test | Command / Method | Expected Result |
|---|---|---|
| Syntax check | `python -m compileall -q srl_rag_demo` | No syntax errors. |
| Demo smoke test | `python srl_rag_demo\smoke_test.py` | Courier answer remains `to the office`. |
| User document test | Paste courier sentence in app | Extracts `to the office` as location. |
| Multi-hop test | Run new multi-hop smoke case | Finds shared argument and final location. |
| Graph test | Inspect graph JSON | Contains question, document, predicate, role, candidate, answer, and score edges. |
| Evaluation suite test | `python evaluation_suite\run_evaluation.py --fast-dev` | Produces CSV and markdown report. |
| Streamlit startup | `streamlit run srl_rag_demo\app.py` | App starts without external API keys. |

## 15. Implementation Risks And Fixes

| Risk | Impact | Fix |
|---|---|---|
| SRL parser model is too heavy | App becomes slow | Keep model optional and preserve heuristic fallback. |
| Multi-hop logic becomes complex | Hard to debug | Start with shared-argument matching only. |
| Graph gets crowded | Hard to explain | Limit graph to top K documents and top N candidates. |
| Metrics become mixed | Presentation confusion | Label every result by scope. |
| QLoRA needs GPU | Not runnable locally | Keep QLoRA as separate optional training experiment. |
| Uploaded document parsing is noisy | Wrong answer spans | Show parser source and confidence, and allow no-answer warnings. |

## 16. Presentation Milestones

| Milestone | What To Show Professor |
|---|---|
| After Phase 1 | User document converted into SRL-like structure. |
| After Phase 2 | Answer score breakdown with role and graph score. |
| After Phase 3 | Multi-hop graph path across two events. |
| After Phase 4 | Verifier accepts/rejects candidate answers with reasons. |
| After Phase 5 | Unified benchmark comparison table. |
| After Phase 6 | Streamlit comparison tab. |
| After Phase 7 | No-answer or contradiction warning. |

## 17. Recommended Next Work Order

Implement in this order:

```text
1. Extend data models for score metadata.
2. Add stronger user-document parsing.
3. Add graph-based scoring.
4. Add score breakdown to Streamlit.
5. Add multi-hop smoke example.
6. Add unified evaluation suite.
7. Add no-answer and contradiction handling.
8. Add optional full training scripts.
```

This order avoids jumping into heavy model training before the core explainable SRL + RAG system is stronger.

## 18. Final Implementation Summary

The next implementation should focus on making the system:

- more general, by improving SRL for uploaded documents,
- more explainable, by scoring graph paths,
- more powerful, by supporting multi-hop reasoning,
- more reliable, by adding verifier and no-answer handling,
- more research-ready, by adding a unified benchmark suite.

Final one-line implementation goal:

```text
Build the next version as a scored, multi-hop, verifier-backed SRL + RAG system with a stronger Streamlit demo and a unified evaluation suite.
```

