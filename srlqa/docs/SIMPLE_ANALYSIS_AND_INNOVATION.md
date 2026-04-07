# Simple Analysis And Innovation Guide

Prepared for presentation: 2026-04-08

This document explains the complete project in simple language. It focuses on
what was already implemented, what was newly added, how the architecture works,
what key terms mean, and which result values are safe to present.

## 1. Project Goal

The project solves Semantic Role Labeling Question Answering, or SRL-QA.

A normal question answering system returns an answer span from the context.
This project returns:

- the answer span,
- the predicate or event,
- the semantic role of the answer,
- a confidence score,
- evidence-backed reasoning when available.

Example:

```text
Context:  The courier delivered the package to the office at noon.
Question: Where was the package delivered?
Answer:   to the office
Role:     ARGM-LOC
Event:    delivered
```

In one sentence:

> This project answers questions by understanding who did what to whom, when,
> where, why, and how.

## 2. Key Terms

| Term | Simple meaning | Example |
|---|---|---|
| SRL | Semantic Role Labeling: finding event roles in a sentence | `The nurse` is the agent of `administered` |
| QA-SRL | Representing semantic roles as questions and answers | `Who received the medicine?` -> `the patient` |
| Predicate | The event word or relation anchor | `delivered`, `gave`, `hired` |
| Argument | A participant or modifier attached to a predicate | `the package`, `to the office`, `at noon` |
| PropBank | A resource of predicate frames and role definitions | `deliver.01` can define sender, thing sent, destination |
| ARG0 | Usually the doer, agent, or initiator | `The courier` |
| ARG1 | Usually the main thing affected or transferred | `the package` |
| ARG2 | Often recipient, destination, beneficiary, or secondary participant | `the patient` in a medicine example |
| ARGM-LOC | Location modifier | `to the office` |
| ARGM-TMP | Time modifier | `at noon` |
| ARGM-MNR | Manner modifier | `with care` |
| ARGM-CAU | Cause modifier | `because of rain` |
| Exact match | Predicted answer exactly equals gold answer after normalization | `to the office` equals `to the office` |
| Token F1 | Token overlap between prediction and gold answer | `the office` partially overlaps `to the office` |
| Retrieval | Looking up helpful external knowledge | retrieving PropBank frame roles for `deliver` |
| Constrained decoding | Applying rules so spans are valid and precise | do not include time phrase in a WHERE answer |
| Verifier | A second-pass checker for candidate answers | checks if `to the office` really answers WHERE |
| Recursive correction | Rejecting a weak candidate and choosing the next best evidence span | replace `office` with `to the office` |

## 3. Previous Implementation State

The earlier project is in:

```text
srl_qa_project/
```

It contained a complete local SRL-QA baseline:

| Part | File | What it did |
|---|---|---|
| Data loading | `data_loader.py` | Loaded PropBank data and created QA pairs |
| Baseline model | `model.py` | Defined `PropQANet` |
| Training | `trainer.py` | Trained and saved the baseline checkpoint |
| Evaluation | `evaluator.py` | Computed QA and SRL scores |
| Baseline inference | `qa_inference.py` | Ran the trained baseline on a user question |
| Hybrid inference | `hybrid_qa.py` | Added rules, candidate reranking, and optional transformer QA |
| Streamlit UI | `app.py` | Displayed project results and interactive features |
| CLI | `main.py` | Ran train, evaluate, ask, benchmark, and app modes |

### 3.1 PropQA-Net Baseline

The baseline model is called `PropQANet`.

It uses:

- word embeddings,
- POS tag embeddings,
- predicate indicator features,
- BiLSTM sentence encoder,
- BiLSTM question encoder,
- SRL BIO tag classifier,
- QA start and end span heads.

It tries to learn two tasks together:

| Task | Meaning |
|---|---|
| SRL tagging | Label each token with a role tag such as `B-ARG0`, `I-ARG0`, or `O` |
| QA span prediction | Predict answer start and end token positions |

Strength:

> It is locally trained, reproducible, and tied to PropBank-style semantic
> roles.

Limitation:

> It is weaker than modern transformer MRC models for exact answer boundaries
> and rare role behavior.

### 3.2 Legacy Hybrid System

The existing hybrid system improved practical behavior by adding:

- question intent parsing,
- role-specific candidate spans,
- optional transformer QA,
- optional sentence-transformer embeddings,
- reranking,
- reasoning traces.

Simple explanation:

> The hybrid system did not replace the baseline. It wrapped the baseline with
> rules and reranking to make answers more useful in demos.

## 4. Implemented Baseline Values

These values come from the saved local project results.

| Metric | Implemented value |
|---|---:|
| QA exact match | `0.5184` |
| QA token F1 | `0.7612` |
| SRL micro F1 | `0.7133` |
| SRL macro F1 | `0.1619` |
| QA pairs | `23,007` |
| Usable PropBank instances | `9,073` |
| Unique predicates | `1,340` |
| Unique rolesets | `1,670` |

What these numbers mean:

- `0.5184` exact match means about half of local QA predictions exactly matched
  the expected span.
- `0.7612` token F1 means the predicted spans often had strong token overlap
  even when exact boundaries were imperfect.
- `0.7133` SRL micro F1 means frequent role labels performed reasonably.
- `0.1619` SRL macro F1 means rare roles were much weaker, because macro F1
  gives equal weight to rare and common roles.

## 5. New RAISE-SRL-QA Implementation

The new implementation is in:

```text
srlqa/
```

RAISE-SRL-QA means:

> Retrieval-Augmented, Iteratively Self-correcting, Explainable Semantic Role
> Labeling Question Answering.

The new package adds:

| Part | What it does |
|---|---|
| Dataset loading through library | Uses Hugging Face `datasets` for QA-SRL-style data access |
| Model through library | Uses Transformers-compatible DeBERTa model loading |
| PropBank frame retrieval | Builds a local frame index from PropBank XML frames |
| Constrained decoding | Keeps answer spans compatible with the question type and role |
| Verifier | Checks candidate spans instead of inventing free-form answers |
| Recursive correction | Selects a better candidate if the first answer is weak or rejected |
| Model hub | Exposes old and new models through one runner |
| Streamlit apps | Supports both existing dashboard and separate RAISE app |

Implemented frame index:

| Item | Implemented value |
|---|---:|
| PropBank frame records | `4,659` |

## 6. RAISE Architecture

Simple flow:

```text
User context + question
        |
        v
Question intent parser
        |
        v
Predicate detector
        |
        v
PropBank frame retriever
        |
        v
Candidate span generator
        |
        v
Constrained decoder
        |
        v
Verifier and recursive correction
        |
        v
Final answer + role + confidence + evidence
```

How to explain this in presentation:

> The system first understands the question type. If the question asks "where",
> it expects a location role such as `ARGM-LOC`. Then it finds the predicate,
> retrieves PropBank frame knowledge, proposes answer spans, filters invalid
> spans, and uses a verifier/correction step to choose the best evidence-backed
> answer.

## 7. Real Demo Walkthrough

Use this example in the presentation:

```text
Context:  The courier delivered the package to the office at noon.
Question: Where was the package delivered?
```

Expected output:

```text
Answer:     to the office
Role:       ARGM-LOC
Predicate:  delivered
Confidence: about 0.9955 in the local demo
```

Why this is correct:

- `delivered` is the predicate or event.
- `the package` is the thing delivered.
- `to the office` is the location or destination.
- `at noon` is time, not location.
- Therefore the WHERE answer should be `to the office`, not the longer phrase
  `to the office at noon`.

All-model sample comparison from local smoke test:

| Model | Answer | Role | Confidence |
|---|---|---|---:|
| `raise_srlqa_fast` | `to the office` | `ARGM-LOC` | `0.9955` |
| `raise_srlqa_model` | `to the office` | `ARGM-LOC` | `0.9955` |
| `legacy_hybrid` | `to the office` | `ARGM-LOC` | `0.6579` |
| `legacy_baseline` | `delivered the package to the office at noon` | `ARG1` | `0.4198` |

Presentation explanation:

> The baseline selected a longer and less semantically precise span. The hybrid
> and RAISE paths selected the exact location phrase. RAISE gives the highest
> confidence because the candidate satisfies the WHERE intent, span rule, and
> role compatibility checks.

## 8. Local Seed Challenge Result

The local seed challenge suite contains 15 targeted examples.

Implemented local result:

| Metric | Implemented value |
|---|---:|
| Number of seed examples | `15` |
| Exact match | `1.0` |
| Token F1 | `1.0` |

Correct way to say this:

> RAISE-SRL-QA scored 1.0 exact match and 1.0 token F1 on our 15-example local
> seed challenge suite. This demonstrates targeted boundary and role behavior.

What not to say:

> This proves the system has 100% accuracy on all SRL-QA tasks.

Why:

> The seed suite is small and local. Official claims require larger frozen test
> splits and official scorers.

## 9. Demo Commands

Run all available models:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py
```

Ask one question:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

Run interactive chat:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m srlqa.main chat --context "The nurse administered the medicine to the patient after dinner."
```

Run RAISE Streamlit:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
streamlit run raise_streamlit_app.py
```

Run existing Streamlit app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
streamlit run app.py
```

## 10. Innovation Summary

The innovation is not simply "use a bigger model".

The innovation is the combination of:

| Innovation | Why it matters |
|---|---|
| Retrieval | Uses PropBank frame knowledge to make semantic roles explicit |
| Constrained decoding | Reduces invalid or overlong spans |
| Verifier | Checks answer candidates against the context |
| Recursive correction | Moves from a weaker candidate to a better evidence-backed span |
| All-model comparison | Lets the user compare baseline, hybrid, and RAISE outputs |
| Streamlit demo | Makes the project understandable and interactive |

## 11. Honest Research Comparison

This project is aligned with recent research directions:

| Research direction | Relation to project |
|---|---|
| QA-SRL parsing | The project uses natural-language questions as an SRL interface |
| PropBank frames | The project uses predicate-role knowledge for explainability |
| Retrieval-augmented SRL | RAISE retrieves PropBank frames for role compatibility |
| Self-correction | RAISE corrects among extracted candidate spans instead of hallucinating |
| QA-Noun and broader semantics | Kept as future extension, not claimed as implemented benchmark result |

Safe claim:

> Our project is a research-level local artifact that combines supervised SRL-QA,
> retrieval, constrained decoding, and correction for explainable outputs.

Unsafe claim:

> Our project beats all latest public SRL systems.

## 12. Likely Questions And Short Answers

**Q: What is the core problem?**

The problem is to answer user questions while also identifying the event role of
the answer.

**Q: What is the difference between QA and SRL-QA?**

QA returns an answer span. SRL-QA also explains the semantic role, such as agent,
patient, location, time, cause, or manner.

**Q: Why did you add RAISE?**

The old baseline was useful but could make span-boundary and rare-role errors.
RAISE adds retrieval, constrained decoding, verification, and correction to make
answers more precise and explainable.

**Q: Why not claim 95% F1?**

Because 95% official F1 requires a frozen benchmark and official scorer. It is a
roadmap target, not a finished official result.

**Q: What should be demonstrated live?**

Run the all-model runner or RAISE Streamlit app, ask the courier delivery
question, and explain why `to the office` is the correct location role.

## 13. Final Speaking Summary

Say this at the end:

> The previous project built a working PropBank-based SRL-QA baseline and hybrid
> system. The new RAISE-SRL-QA layer makes the project more explainable and demo
> ready by adding PropBank frame retrieval, constrained span selection,
> recursive correction, all-model comparison, and a separate Streamlit app. The
> implemented baseline is 0.7612 QA token F1, and the new RAISE demo reaches
> 1.0 token F1 on a small 15-example local seed challenge suite. We do not claim
> public SOTA; we present this as a clear, locally runnable, research-oriented
> SRL-QA system.
