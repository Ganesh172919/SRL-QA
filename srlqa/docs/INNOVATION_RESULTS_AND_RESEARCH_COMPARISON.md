# Innovation Results And Research Comparison

Date: 2026-04-07

This document summarizes the RAISE-SRL-QA innovation, local results, and how the
project compares with current research. The comparison is intentionally careful:
local challenge-suite results are not official public benchmark results.

## 1. Implemented Innovation

RAISE-SRL-QA stands for:

Retrieval-Augmented, Iteratively Self-correcting, Explainable Semantic Role
Labeling Question Answering.

Implemented in `srlqa/`:

| Innovation | Implemented file(s) | Status |
|---|---|---|
| Library dataset loading | `srlqa/data/dataset_library.py` | Done |
| QA-SRL MRC conversion | `srlqa/data/convert_to_mrc.py` | Done |
| DeBERTa-compatible MRC model | `srlqa/models/mrc_srl_qa.py` | Scaffold done and forward-tested |
| PropBank frame retrieval | `srlqa/retrieval/` | Done, 4,659 frame records indexed |
| Constrained span decoding | `srlqa/decoding/` | Done |
| Verifier | `srlqa/verification/span_verifier.py` | Done |
| Recursive correction | `srlqa/pipeline.py` | Done |
| Hard negatives | `srlqa/training/hard_negative_mining.py` | Utility done |
| Teacher distillation hooks | `srlqa/distillation/` | Utility done |
| Nominal QA extension | `srlqa/nominal/` | Initial templates done |
| Proto-role diagnostics | `srlqa/proto_roles/` | Initial utilities done |
| Calibration and leaderboard | `srlqa/evaluation/`, `srlqa/reports/` | Done |
| All-model runner | `srlqa/run_all_models.py` | Done |
| RAISE Streamlit app | `srlqa/raise_streamlit_app.py` and `srlqa/app.py` | Done |
| Existing Streamlit integration | `srl_qa_project/app.py` | Done, added `All Model QA` page |

## 2. Model Families Now Available

| Model key | Meaning |
|---|---|
| `legacy_baseline` | Original PropQA-Net checkpoint from `srl_qa_project` |
| `legacy_hybrid` | Existing hybrid system from `srl_qa_project/hybrid_qa.py` |
| `raise_srlqa_fast` | New deterministic RAISE mode without model loading |
| `raise_srlqa_model` | New RAISE mode with DeBERTa/SQuAD-style QA candidates and recursive correction |
| `all` | Runs every model family and returns side-by-side outputs |

## 3. Local Results

### Existing baseline

From existing saved artifacts:

| Metric | Value |
|---|---:|
| QA exact match | 0.5184 |
| QA token F1 | 0.7612 |
| SRL micro F1 | 0.7133 |
| SRL macro F1 | 0.1619 |
| QA pairs | 23,007 |
| Usable PropBank instances | 9,353 |

### New seed challenge suite

File: `srlqa/data/challenge_suite_v2.json`

The 15 examples test:

- agent/patient flips
- location-time boundaries
- recipient `ARG2`
- manner boundaries
- causal spans
- nominal event location

Command run:

```powershell
python -m srlqa.main demo --max-examples 15
```

Observed result:

| System | Count | Exact match | Token F1 | Notes |
|---|---:|---:|---:|---|
| RAISE-SRL-QA model-backed | 15 | 1.000 | 1.000 | Uses QA model candidates plus SRL correction |
| RAISE-SRL-QA fast | 15 | 1.000 | 1.000 | Uses deterministic SRL candidates plus verifier |

Example correction:

| Question | Model candidate problem | Corrected answer |
|---|---|---|
| `Where was the package delivered?` | QA model proposed near-misses such as `office` | `to the office` |
| `When was the package delivered?` | QA model proposed `noon.` | `at noon` |
| `How did the engineer repair the machine?` | QA model proposed instrument phrase | `carefully` |
| `What was administered?` | QA model proposed `medicine` | `the medicine` |

Important: this is a local seed challenge-suite result. It should be presented
as evidence that the correction logic works on targeted cases, not as a public
benchmark score.

## 4. Comparison With Research

| Work | What it contributes | Relation to this project |
|---|---|---|
| Large-Scale QA-SRL Parsing, ACL 2018 | QA-SRL Bank 2.0 with over 250,000 QA pairs and a QA-SRL parser; reported 77.6% span-level accuracy under human evaluation | Provides the main QA-SRL framing: semantic roles as questions and answer spans |
| PropBank Comes of Age, 2022 | Shows PropBank grew beyond verbal predicates into more domains, genres, languages, and predicate types | Justifies using PropBank frames as explicit semantic knowledge |
| LLMs Can Also Do Well, 2025 | Uses retrieval-augmented generation and self-correction for SRL; reports SOTA across SRL benchmarks | Directly motivates RAISE's retrieval plus correction architecture |
| QA-Noun, 2025 | Adds noun-centered QA semantics and complements verbal QA-SRL | Motivates the `nominal/` extension in RAISE |
| Effective QA-driven Annotation Across Languages, 2026 | Uses QA-SRL as a transferable interface and projects annotations across languages | Motivates future multilingual data expansion |

## 5. Honest Comparison Statement

Use this in the presentation:

> Compared with recent SRL research, our project implements the same direction:
> retrieval, self-correction, QA-based semantics, and explainability. However,
> our reported scores are local. We do not claim official SOTA. Our contribution
> is a locally runnable, explainable SRL-QA system that integrates these research
> ideas and demonstrates strong targeted challenge-suite behavior.

## 6. Why Our Architecture Is Strong

Pure QA model:

- good at span extraction
- weak at role explanation
- can miss prepositions or include extra modifiers

Pure rule-based system:

- fast and explainable
- weak on unfamiliar syntax

Pure LLM prompting:

- flexible
- can hallucinate, change format, or invent spans

RAISE hybrid:

- uses model candidates
- uses PropBank frame knowledge
- keeps answers extractive
- applies role-aware boundary constraints
- recursively corrects wrong candidates in evaluation mode
- exposes evidence and reasoning

## 7. Research Sources

- [Large-Scale QA-SRL Parsing](https://aclanthology.org/P18-1191/)
- [PropBank Comes of Age](https://aclanthology.org/2022.starsem-1.24/)
- [LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models](https://arxiv.org/abs/2506.05385)
- [Effective QA-driven Annotation of Predicate-Argument Relations Across Languages](https://arxiv.org/abs/2602.22865)
- [QA-Noun: Representing Nominal Semantics via Natural Language Question-Answer Pairs](https://aclanthology.org/2025.ijcnlp-long.147/)

## 8. Final Claim Boundary

Allowed:

> RAISE-SRL-QA achieved 1.0 exact match and 1.0 token F1 on the local
> 15-example seed challenge suite.

Allowed:

> The original local baseline has 0.7612 QA token F1.

Not allowed:

> This project beats public SOTA SRL systems.

Why not:

Public SOTA comparison needs the same dataset, split, preprocessing, and
official scorer. This project has not yet run that official validation.
