# Last-Minute Revision Sheet

Presentation date: 2026-04-08

## Elevator Pitch

This is a Hybrid Semantic Role Labeling Question Answering project. It answers
natural language questions from a context and also explains the semantic role of
the answer with respect to a predicate.

Example:

- Context: `The courier delivered the package to the office at noon.`
- Question: `Where was the package delivered?`
- Answer: `to the office`
- Predicate: `delivered`
- Role: `ARGM-LOC`

## Key Terms

| Term | Meaning |
|---|---|
| SRL | Semantic Role Labeling: finding who did what to whom, when, where, why, and how |
| Predicate | The event word, usually a verb such as `delivered` |
| Argument | A participant or modifier attached to the predicate |
| PropBank | A frame resource that defines roles for predicates |
| QA-SRL | SRL represented through natural-language questions and answer spans |
| BIO tags | Token labels like `B-ARG0`, `I-ARG0`, `O` |
| Exact match | Prediction exactly equals gold answer after normalization |
| Token F1 | Token overlap between predicted and gold answer |

## Existing Project

Folder: `srl_qa_project/`

| File | Use |
|---|---|
| `data_loader.py` | Creates PropBank-derived QA examples |
| `model.py` | Defines `PropQANet` |
| `trainer.py` | Trains and saves checkpoint |
| `evaluator.py` | Computes metrics |
| `qa_inference.py` | Runs baseline model |
| `hybrid_qa.py` | Runs hybrid model |
| `app.py` | Streamlit dashboard |

Baseline metrics:

| Metric | Value |
|---|---:|
| QA exact match | 0.5184 |
| QA token F1 | 0.7612 |
| SRL micro F1 | 0.7133 |
| SRL macro F1 | 0.1619 |
| QA pairs | 23,007 |
| Usable PropBank instances | 9,353 |

## New Innovation

Folder: `srlqa/`

RAISE-SRL-QA means:

Retrieval-Augmented, Iteratively Self-correcting, Explainable SRL-QA.

New components:

- PropBank frame retrieval
- DeBERTa-compatible MRC model scaffold
- deterministic SRL span candidates
- verifier that only chooses spans from the context
- recursive correction
- all-model runner
- standalone RAISE Streamlit app

## Commands

All-model terminal runner:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py
```

Single RAISE question:

```powershell
python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

RAISE Streamlit:

```powershell
streamlit run raise_streamlit_app.py
```

Existing Streamlit:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
streamlit run app.py
```

## Results To Say Carefully

Correct statement:

> The original baseline has 0.7612 local QA token F1. The new RAISE pipeline
> scored 1.0 exact match and 1.0 token F1 on the 15-example seed challenge suite.
> This is a local challenge-suite result, not an official public benchmark SOTA
> claim.

Do not say:

> We beat all existing systems globally.

Say instead:

> We built a research-level, explainable, locally runnable SRL-QA artifact and
> demonstrated strong results on targeted local challenge cases.

## Viva Answers

Q: Why combine QA and SRL?  
A: QA gives natural questions and answer spans; SRL gives semantic structure and
explainability.

Q: Why PropBank?  
A: It provides predicate frames and role definitions, useful for verifying
whether an answer role is semantically compatible.

Q: Why recursive correction?  
A: A QA model may predict a near-miss like `office`; correction can choose the
better SRL span `to the office`.

Q: What is the main limitation?  
A: The 100% result is on a small seed challenge suite. Larger frozen evaluation
and official scorers are needed for strong benchmark claims.

Q: What is the best innovation?  
A: The best innovation is the combination of retrieval, exact span constraints,
and verifier-based self-correction while keeping answers extractive.
