# Project Index: Hybrid SRL-QA and RAISE-SRL-QA

Prepared for presentation: 2026-04-08

This file is the starting point for understanding and demonstrating the whole
project. Read this first when you want a simple map of what exists, what changed,
which numbers are implemented, and how to run the demo.

## 1. Project In One Paragraph

This is a Semantic Role Labeling Question Answering project. A normal QA system
answers a question by returning a text span. This project also explains the
semantic role of that answer with respect to an event or predicate.

Example:

| Item | Value |
|---|---|
| Context | `The courier delivered the package to the office at noon.` |
| Question | `Where was the package delivered?` |
| Answer | `to the office` |
| Predicate | `delivered` |
| Semantic role | `ARGM-LOC`, meaning location modifier |

The main idea is: answer the user question, but also say what role the answer
plays in the event.

## 2. What To Read Before Presentation

Read in this order:

| Priority | File | Purpose |
|---:|---|---|
| 1 | `PROJECT_INDEX.md` | Fast map of the complete project |
| 2 | `srlqa/docs/SIMPLE_ANALYSIS_AND_INNOVATION.md` | Simple explanation of old system, new system, key terms, and demo |
| 3 | `COMPLETE_FUNCTIONAL_PROJECT_GUIDE.md` | Complete functional status, architecture, accuracy, and demo guide |
| 4 | `WHAT_NEXT.md` | What to do next after the current functional implementation |
| 5 | `srlqa/docs/LAST_MINUTE_REVISION.md` | One-page revision sheet before the presentation |
| 6 | `srlqa/docs/PROJECT_PRESENTATION_MASTER_GUIDE.md` | Detailed explanation and likely viva questions |
| 7 | `srlqa/docs/INNOVATION_RESULTS_AND_RESEARCH_COMPARISON.md` | Research comparison and implemented-result boundaries |

Use the PPT after reading the first two files:

| PPT | Purpose |
|---|---|
| `FINAL_PROJECT_PRESENTATION_40_SLIDES.pptx` | Original deck, kept unchanged |
| `FINAL_PROJECT_PRESENTATION_RAISE_UPDATED.pptx` | Updated deck with new RAISE-SRL-QA explanation and demo slides |

## 3. Folder Map

| Folder or file | Meaning |
|---|---|
| `srl_qa_project/` | Existing project with baseline PropQA-Net, hybrid QA, training, evaluation, and Streamlit dashboard |
| `srlqa/` | New RAISE-SRL-QA package with retrieval, correction, all-model runner, model hub, and standalone Streamlit app |
| `srl_qa_project/results/metrics.json` | Saved baseline QA and SRL metrics |
| `srl_qa_project/results/data_statistics.json` | Dataset statistics for the existing project |
| `srlqa/retrieval/frame_store.json` | Built PropBank frame index with 4,659 frame records |
| `srlqa/run_all_models.py` | Terminal app where the user selects one model or all models |
| `srlqa/raise_streamlit_app.py` | Separate RAISE-SRL-QA Streamlit app |
| `srl_qa_project/app.py` | Existing Streamlit app, updated with an All Model QA page |

## 4. Previous State Of The Project

The previous implementation lived in `srl_qa_project/`.

It had:

- A locally trained baseline model called `PropQANet`.
- PropBank-derived QA pairs.
- SRL BIO tagging and QA span prediction.
- A hybrid inference layer with rules and reranking.
- Evaluation metrics and plots.
- A Streamlit dashboard.

The saved baseline results were:

| Metric | Implemented value |
|---|---:|
| QA exact match | `0.5184` |
| QA token F1 | `0.7612` |
| SRL micro F1 | `0.7133` |
| SRL macro F1 | `0.1619` |
| QA pairs | `23,007` |
| Usable PropBank instances | `9,073` |

Simple explanation:

> The old project already worked as a local SRL-QA baseline and hybrid demo, but
> span boundaries and rare roles were still weak. For example, a baseline model
> could return a longer phrase than needed or assign the wrong semantic role.

## 5. New RAISE-SRL-QA State

The new implementation lives in `srlqa/`.

RAISE-SRL-QA means:

> Retrieval-Augmented, Iteratively Self-correcting, Explainable Semantic Role
> Labeling Question Answering.

It adds:

- PropBank frame retrieval.
- DeBERTa-compatible MRC model scaffolding.
- Deterministic SRL span candidates.
- Constrained span decoding rules.
- A verifier and recursive correction loop.
- A common model hub for all available project models.
- Terminal and Streamlit interfaces for asking questions.

Implemented local demo result:

| Local demo suite | Implemented value |
|---|---:|
| Seed challenge examples | `15` |
| Exact match on seed challenge suite | `1.0` |
| Token F1 on seed challenge suite | `1.0` |

Important:

> This is a small local seed-suite demonstration result. It is not an official
> public benchmark result and should not be described as global SOTA.

## 6. Demo Commands

Run all available models from terminal:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py
```

Run the fastest scripted demo:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model raise_srlqa_fast --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Ask one RAISE-SRL-QA question:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

Run interactive RAISE chat:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main chat --context "The nurse administered the medicine to the patient after dinner."
```

Run separate RAISE Streamlit app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
streamlit run raise_streamlit_app.py
```

Run existing project Streamlit app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
streamlit run app.py
```

## 7. Demo Script To Speak

Use this example:

```text
Context: The courier delivered the package to the office at noon.
Question: Where was the package delivered?
```

Expected strong answer:

```text
Answer: to the office
Role: ARGM-LOC
Predicate: delivered
```

How to explain it:

> The word "delivered" is the event. The phrase "to the office" tells the
> location or destination of that event, so the role is ARGM-LOC. A pure baseline
> can return a longer span like "delivered the package to the office at noon",
> but the RAISE pipeline uses role-aware span rules and correction to select the
> exact location phrase.

## 8. What To Say About Accuracy

Say:

> The existing baseline has 0.7612 local QA token F1. The new RAISE pipeline
> achieved 1.0 exact match and 1.0 token F1 on the 15-example seed challenge
> suite used for targeted demo checks. This is a local seed-suite result, not an
> official public benchmark claim.

Do not say:

> We beat all existing systems globally.

Say instead:

> We built a locally runnable, explainable SRL-QA system and demonstrated that
> retrieval, role-aware decoding, and recursive correction improve targeted demo
> behavior.

## 9. Final Presentation Checklist

- Open `FINAL_PROJECT_PRESENTATION_RAISE_UPDATED.pptx`.
- Keep one terminal ready in `srlqa/`.
- Use the courier delivery example for a clean demo.
- If asked about 95% F1, say it is a future target, not an official result.
- If asked what is new, say: all-model runner, separate RAISE app, PropBank
  frame retrieval, recursive correction, and implemented-result documentation.
