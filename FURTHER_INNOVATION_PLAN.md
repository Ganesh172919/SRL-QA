# Further Innovation Plan

## Purpose

This document proposes future innovations for the project after the current SRL + RAG explainable QA demo.

The current project already includes:

- PropBank loading through NLTK,
- SRL-based QA generation,
- legacy PropQA-Net baseline evaluation,
- RAISE-SRL-QA role-aware retrieval and verification,
- standalone Streamlit SRL + RAG demo,
- semantic graph explanations,
- evaluation and documentation.

The next step is to make the system stronger, more general, more measurable, and more research-ready.

## 1. Current System Summary

Current pipeline:

```text
PropBank + Treebank
    -> SRL structured documents
    -> role-aware retrieval
    -> extractive QA
    -> semantic graph explanation
    -> Streamlit demo
```

Current strengths:

- Uses real PropBank SRL annotations.
- Gives role-aware answers instead of only raw text answers.
- Supports local RAG with TF-IDF fallback.
- Visualizes reasoning using semantic graphs.
- Has full baseline metrics and curated challenge-suite comparisons.

Current limitations:

- User-uploaded documents use simple SRL-like parsing.
- RAISE `100.00%` results are from a small curated seed suite.
- LoRA and QLoRA outputs are FAST_DEV artifacts.
- The graph explains one reasoning path, but not multiple competing paths.
- Retrieval evaluation is functional but not yet a full benchmark.

## 2. Main Future Innovation Theme

The main future innovation can be stated as:

```text
Move from SRL-assisted QA to a full semantic reasoning system that retrieves, verifies, compares, and explains multiple evidence paths.
```

In simple words:

```text
The current system answers with one SRL path.
The future system should reason over many SRL paths and explain why one path is best.
```

## 3. Proposed Innovation Roadmap

| Phase | Innovation | Goal |
|---|---|---|
| Phase 1 | Stronger SRL for user documents | Make uploaded documents more accurate. |
| Phase 2 | Multi-hop SRL reasoning | Answer questions requiring multiple events. |
| Phase 3 | Graph-based evidence ranking | Rank answers using semantic graph paths. |
| Phase 4 | Neural + symbolic hybrid verifier | Combine model confidence with SRL role rules. |
| Phase 5 | Full benchmark suite | Evaluate RAG, QA, SRL, and explainability together. |
| Phase 6 | Better visualization | Show competing paths, confidence, and role definitions. |
| Phase 7 | Full model fine-tuning | Replace FAST_DEV experiments with stable full runs. |

## 4. Innovation 1: Stronger SRL For User Documents

### Problem

The current demo can accept pasted or uploaded documents, but those documents do not come with gold PropBank annotations. The app builds simple SRL-like structures for them.

### Proposed Innovation

Add a real SRL parser for arbitrary user documents.

Possible approaches:

- use an AllenNLP-style SRL model,
- use a Hugging Face token-classification SRL model,
- train or fine-tune a lightweight SRL model on PropBank,
- add predicate detection before role extraction.

### Expected Benefit

The demo will become more useful on new documents, not only PropBank examples.

Future pipeline:

```text
Uploaded document
    -> sentence splitter
    -> predicate detector
    -> SRL parser
    -> frame mapper
    -> retriever index
    -> QA answer
    -> graph explanation
```

### Evaluation

| Metric | Target |
|---|---:|
| Predicate detection accuracy | Improve over simple heuristic baseline |
| Role extraction F1 | Compare against PropBank-style held-out examples |
| QA token F1 on uploaded document benchmark | Improve over TF-IDF-only baseline |

## 5. Innovation 2: Multi-Hop SRL Reasoning

### Problem

The current system mainly answers questions from one retrieved evidence document or one semantic event.

### Proposed Innovation

Add multi-hop reasoning across multiple predicates and documents.

Example:

```text
Sentence 1: The courier picked up the package from the warehouse.
Sentence 2: The courier delivered the package to the office.

Question: Where did the package go after it was picked up?
Answer: to the office
```

This requires linking:

```text
picked up -> package -> delivered -> to the office
```

### Expected Benefit

The system can answer more complex reasoning questions.

Future graph:

```text
Question
    -> Event 1 predicate
    -> Shared argument
    -> Event 2 predicate
    -> Target role
    -> Answer
```

### Evaluation

| Metric | Target |
|---|---:|
| Multi-hop exact match | Compare with single-hop baseline |
| Multi-hop token F1 | Improve answer overlap |
| Evidence path accuracy | Check whether graph path contains correct events |
| Role-chain accuracy | Check whether connected roles are correct |

## 6. Innovation 3: Graph-Based Evidence Ranking

### Problem

Current retrieval ranks documents mostly using text similarity and SRL-enhanced text.

### Proposed Innovation

Use graph features to rank evidence and answer candidates.

Graph features can include:

- path length from question role to candidate answer,
- predicate-role compatibility,
- frame-role match,
- argument overlap across documents,
- confidence of each edge,
- whether the candidate is extractable from the source text.

### Expected Benefit

The answer will be selected based on both retrieval similarity and semantic graph quality.

Future scoring:

```text
final_score =
    retrieval_score
    + role_match_score
    + frame_compatibility_score
    + graph_path_score
    + extractability_score
```

### Evaluation

| Metric | Target |
|---|---:|
| Answer exact match | Improve over retrieval-only ranking |
| Token F1 | Improve over retrieval-only ranking |
| Role accuracy | Maintain or improve role correctness |
| Explanation path correctness | New metric for graph quality |

## 7. Innovation 4: Neural + Symbolic Hybrid Verifier

### Problem

Pure neural QA can over-select spans or choose a semantically wrong role. Pure heuristic systems can be brittle.

### Proposed Innovation

Build a hybrid verifier that combines:

- neural QA score,
- SRL role match,
- PropBank frame compatibility,
- answer span extractability,
- question-type compatibility,
- graph path strength.

### Expected Benefit

This creates a stronger answer verification layer.

Verifier flow:

```text
Candidate answer
    -> neural QA confidence
    -> SRL role check
    -> frame compatibility check
    -> graph path check
    -> final acceptance or rejection
```

### Evaluation

| Metric | Target |
|---|---:|
| False positive reduction | Fewer wrong high-confidence answers |
| Exact match | Improve over model-only QA |
| Role accuracy | Improve over model-only QA |
| Calibration error | Confidence should better match correctness |

## 8. Innovation 5: Explainability Score

### Problem

The current system shows a graph, but it does not produce a single numeric explainability score.

### Proposed Innovation

Create an explainability score based on:

- whether the answer span is extractive,
- whether the answer role matches the question type,
- whether a PropBank frame supports the role,
- whether the answer is connected in the graph,
- whether the evidence document was highly ranked.

Possible formula:

```text
explainability_score =
    0.25 * extractability
    + 0.25 * role_match
    + 0.20 * frame_match
    + 0.20 * graph_connectivity
    + 0.10 * retrieval_rank_score
```

### Expected Benefit

The demo can show not only the answer confidence, but also how explainable the answer is.

### Evaluation

| Metric | Target |
|---|---:|
| Explanation score correlation with correctness | Positive correlation |
| Human explanation rating | Improve over answer-only QA |
| Low-explainability warning accuracy | Identify uncertain reasoning paths |

## 9. Innovation 6: Full Evaluation Benchmark Suite

### Problem

Current results come from different scopes:

- full baseline test,
- benchmark challenge split,
- curated RAISE seed suite,
- one smoke example,
- FAST_DEV model experiments.

These are useful, but a future research version should use one unified benchmark suite.

### Proposed Innovation

Create a unified benchmark for:

- single-hop SRL-QA,
- multi-hop SRL-QA,
- user-document RAG,
- frame-role compatibility,
- graph explanation correctness,
- latency and robustness.

### Proposed Benchmark Splits

| Split | Purpose |
|---|---|
| `propbank_full` | Full PropBank-derived QA evaluation. |
| `challenge_roles` | Hard role questions such as where, when, why, how. |
| `multi_hop_srl` | Questions requiring two or more predicates. |
| `user_docs` | Uploaded document style examples. |
| `explainability_eval` | Human or rule-based graph explanation evaluation. |

### Evaluation Metrics

| Metric Category | Metrics |
|---|---|
| QA quality | Exact Match, Token Precision, Token Recall, Token F1 |
| SRL quality | Role Accuracy, SRL Micro F1, SRL Macro F1, BIO Accuracy |
| Retrieval quality | Recall@K, MRR, nDCG |
| Graph quality | Path correctness, edge correctness, explanation score |
| Robustness | No-answer accuracy, contradiction detection |
| Efficiency | Mean latency, median latency, P95 latency |

## 10. Innovation 7: Contradiction And No-Answer Handling

### Problem

The current system is designed to find the best answer from retrieved evidence. It should also know when evidence is missing or contradictory.

### Proposed Innovation

Add:

- no-answer detection,
- contradiction detection between retrieved documents,
- low-confidence warning,
- graph conflict visualization.

Example:

```text
Document 1: The package was delivered to the office.
Document 2: The package was delivered to the warehouse.

Question: Where was the package delivered?
```

The system should show both candidate locations and mark the conflict.

### Evaluation

| Metric | Target |
|---|---:|
| No-answer accuracy | Correctly abstain when answer is missing |
| Contradiction detection accuracy | Detect conflicting evidence |
| False answer reduction | Reduce unsupported answers |

## 11. Innovation 8: Better Streamlit Demo Experience

### Problem

The current Streamlit demo is functional, but future versions can be more presentation-ready.

### Proposed Innovation

Add:

- side-by-side answer comparison across systems,
- graph path confidence labels,
- collapsible PropBank frame definitions,
- downloadable evaluation report,
- custom document benchmark upload,
- one-click demo examples,
- color-coded role labels.

### Expected Benefit

The demo becomes easier to explain to professors, reviewers, and users.

Future demo tabs:

| Tab | Purpose |
|---|---|
| QA | Ask question and show final answer. |
| Evidence | Show retrieved documents and SRL triples. |
| Graph | Visualize reasoning path. |
| Compare | Compare baseline, hybrid, RAISE, and RAG outputs. |
| Metrics | Show evaluation summary. |
| Debug | Show candidate spans and scores. |

## 12. Innovation 9: Full Model Training Plan

### Problem

The LoRA and QLoRA folders are FAST_DEV artifacts, not final full-scale training results.

### Proposed Innovation

Run full training experiments with:

- BERT,
- RoBERTa,
- DeBERTa,
- DistilBERT,
- Gemma 2B QLoRA,
- possibly a small instruction-tuned local model.

### Training Improvements

Use:

- predicate markers in the context,
- role-aware prompts,
- frame description prompts,
- full PropBank-derived QA corpus,
- multiple random seeds,
- train/validation/test reporting,
- early stopping,
- calibration evaluation.

### Evaluation

| Metric | Target |
|---|---:|
| Exact Match | Improve over current FAST_DEV runs |
| Token F1 | Improve over current FAST_DEV runs |
| Role Accuracy | Match or improve role-aware deterministic systems |
| Latency | Keep demo usable |
| Calibration | Confidence should reflect correctness |

## 13. Innovation 10: Research Paper Style Contribution

The future paper-style contribution can be framed as:

```text
An SRL-grounded RAG architecture for explainable question answering that combines PropBank frame semantics, role-aware answer selection, graph-based evidence reasoning, and hybrid neural-symbolic verification.
```

Possible research questions:

| Research Question | Expected Experiment |
|---|---|
| Does SRL-structured retrieval improve QA? | Compare raw-text retrieval vs SRL-enhanced retrieval. |
| Does role-aware answer selection improve accuracy? | Compare baseline span selection vs role-aware selection. |
| Does graph verification reduce unsupported answers? | Compare QA with and without graph verifier. |
| Are explanations useful to humans? | Human rating study for graph explanations. |
| Can multi-hop SRL graphs improve complex QA? | Compare single-hop vs multi-hop benchmark. |

## 14. Proposed Implementation Timeline

| Stage | Work | Output |
|---|---|---|
| Week 1 | Add stronger SRL parser for user documents | Better user-document ingestion. |
| Week 2 | Add graph-based evidence scoring | Improved answer ranking. |
| Week 3 | Add no-answer and contradiction handling | Safer QA behavior. |
| Week 4 | Add comparison tab in Streamlit | Better demo presentation. |
| Week 5 | Build unified evaluation benchmark | Consistent metric reporting. |
| Week 6 | Run full benchmark and update docs | Research-ready evaluation. |
| Week 7 | Run larger fine-tuning experiment | Stronger modeling result. |
| Week 8 | Prepare final paper/report/presentation | Complete final submission. |

## 15. Priority Recommendations

If time is limited, prioritize these:

| Priority | Innovation | Why |
|---|---|---|
| 1 | Graph-based evidence scoring | Directly improves the existing SRL + RAG idea. |
| 2 | Stronger SRL parser for user documents | Makes the demo more general. |
| 3 | Unified benchmark suite | Makes metrics easier to defend. |
| 4 | Comparison tab in Streamlit | Makes professor demo stronger. |
| 5 | No-answer handling | Makes the system safer and more realistic. |

## 16. Presentation-Ready Future Work Speech

Use this in a viva or presentation:

```text
In future work, I plan to extend the project from single-path SRL-based QA to multi-path semantic reasoning. The first improvement is to add a stronger SRL parser for arbitrary uploaded documents, so the system is not limited to PropBank-style examples. The second improvement is graph-based evidence ranking, where the answer is selected not only by retrieval similarity but also by role match, frame compatibility, graph path strength, and extractability. The third improvement is a unified benchmark suite that evaluates QA accuracy, SRL role accuracy, retrieval quality, graph explanation correctness, and latency under one consistent setup. Finally, I would extend the model training work beyond FAST_DEV experiments and run full-scale BERT, RoBERTa, DeBERTa, and Gemma-style experiments with multiple seeds.
```

## 17. Final Innovation Summary

The future innovation direction is:

```text
Current system:
SRL + RAG + one reasoning graph path

Future system:
SRL + RAG + multi-hop semantic graph reasoning + hybrid verifier + unified explainability benchmark
```

Final one-line statement:

```text
The next innovation is to turn the current explainable SRL + RAG demo into a complete semantic reasoning system that can parse new documents, compare multiple evidence paths, detect uncertainty, and evaluate explanation quality.
```

