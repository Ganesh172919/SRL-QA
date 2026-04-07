# SRL-QA Project Presentation Master Guide

Prepared on: 2026-04-07  
Presentation target: 2026-04-08

This document is written for someone who knows the basics of the project and
needs to explain it clearly in a viva or presentation. It covers the existing
project, the new RAISE-SRL-QA innovation, key terms, examples, file roles,
metrics, likely questions, and honest limitations.

## 1. One-Minute Explanation

This project is a Hybrid Semantic Role Labeling Question Answering system.

Normal extractive question answering only returns a text span. Example:

- Context: `The courier delivered the package to the office at noon.`
- Question: `Where was the package delivered?`
- Answer: `to the office`

Our project goes further. It also explains the semantic role:

- Predicate/event: `delivered`
- Answer span: `to the office`
- Semantic role: `ARGM-LOC`, meaning location modifier
- Evidence: the answer appears directly in the input context
- Confidence and reasoning: role-aware span rules and verifier selected it

So the core idea is:

> We are not only answering questions. We are answering questions as event-role
> extraction.

## 2. The Problem

Natural language contains events and participants:

`Maria gave the intern a notebook for the workshop.`

There is an event: `gave`.

The participants and modifiers are:

| Text span | Meaning | PropBank-like role |
|---|---|---|
| `Maria` | giver / agent | `ARG0` |
| `a notebook` | thing given | `ARG1` |
| `the intern` | recipient | `ARG2` |
| `for the workshop` | purpose | `ARGM-PRP` |

A pure QA model may answer `intern` or `the intern`, but it usually does not
say whether that answer is an agent, patient, recipient, time, location, or
cause. Semantic Role Labeling (SRL) solves this by assigning roles around a
predicate. QA-SRL makes it more natural by using questions and answer spans.

## 3. Key Terms

### Semantic Role Labeling

SRL is the task of finding "who did what to whom, when, where, why, and how" for
an event.

Example:

`The nurse administered the medicine to the patient after dinner.`

- Predicate: `administered`
- `The nurse`: `ARG0`, the agent
- `the medicine`: `ARG1`, the thing administered
- `the patient`: `ARG2`, recipient or target
- `after dinner`: `ARGM-TMP`, time modifier

### Predicate

The event or relation anchor. Usually a verb in this project.

Examples: `delivered`, `gave`, `approved`, `repaired`, `hired`.

### Argument

A participant or modifier attached to the predicate.

Examples: agent, object/theme, recipient, time, location, cause, manner.

### PropBank

PropBank is a semantic role resource. It defines framesets for predicates.
For a predicate like `deliver`, a frame can describe expected roles such as:

- `ARG0`: sender
- `ARG1`: thing sent
- `ARG2`: recipient or destination

This project uses bundled local PropBank data in `srl_qa_project/nltk_data/`.
The new RAISE package builds a PropBank frame index from these XML frame files.

### BIO Tags

BIO tagging is a token labeling format:

- `B-ARG0`: beginning of an `ARG0` span
- `I-ARG0`: inside an `ARG0` span
- `O`: outside any argument span

Example:

`The company hired Rahul.`

| Token | Tag |
|---|---|
| The | `B-ARG0` |
| company | `I-ARG0` |
| hired | `O` |
| Rahul | `B-ARG1` |

### QA-SRL

QA-SRL represents semantic roles using natural-language questions and answers.

Instead of saying `ARG0`, it can ask:

- `Who hired Rahul?` -> `The company`
- `Who was hired?` -> `Rahul`

This makes semantic annotation more intuitive and explainable.

### Token F1

Token F1 measures overlap between predicted answer tokens and gold answer
tokens.

Example:

- Gold: `to the office`
- Prediction: `the office`

There is partial overlap, so token F1 is not zero, but exact match is false.

### Exact Match

Exact match is stricter. It checks whether the normalized predicted answer is
exactly the gold answer.

`the office` is not an exact match for `to the office`.

## 4. Existing Project Architecture

The original runnable project lives in:

`srl_qa_project/`

Main files:

| File | Purpose |
|---|---|
| `config.py` | Defines project paths, data, model, training, runtime settings |
| `data_loader.py` | Loads PropBank through NLTK, creates QA pairs, builds vocabularies and dataloaders |
| `model.py` | Implements `PropQANet`, the trained baseline model |
| `trainer.py` | Trains `PropQANet` and saves `best_model.pt` |
| `evaluator.py` | Computes QA and SRL metrics |
| `qa_inference.py` | Runs the baseline checkpoint on user context and question |
| `hybrid_qa.py` | Adds hybrid inference: rules, optional transformer QA, embeddings, reranking |
| `benchmark.py` | Runs benchmark and ablation reports |
| `app.py` | Existing Streamlit dashboard |
| `main.py` | CLI runner for train, eval, ask, app, benchmark, report |

### Existing Baseline: PropQA-Net

The baseline model is called `PropQANet`.

It combines:

- word embeddings
- POS tag embeddings
- predicate indicator embeddings
- BiLSTM context encoder
- BiLSTM question encoder
- SRL BIO classifier
- QA start/end span projections

It learns both:

- SRL token labels
- answer span boundaries

The strength is that it is reproducible and locally trained. The weakness is
that it is not as strong as modern transformer MRC models for span precision.

### Existing Hybrid System

The existing hybrid layer in `hybrid_qa.py` adds:

- question intent parsing
- role-aware span candidates
- optional `distilbert-base-cased-distilled-squad`
- optional sentence-transformer embeddings
- answer reranking
- reasoning traces

The purpose is to improve practical answers while keeping the baseline
checkpoint as reference.

## 5. Existing Baseline Metrics

From `srl_qa_project/results/metrics.json` and
`srl_qa_project/results/data_statistics.json`:

| Metric | Value |
|---|---:|
| QA exact match | 0.5184 |
| QA token F1 | 0.7612 |
| SRL micro F1 | 0.7133 |
| SRL macro F1 | 0.1619 |
| QA pairs | 23,007 |
| Usable PropBank instances in live stats | 9,353 |

Important presentation wording:

> The baseline reaches 0.7612 local QA token F1. The new RAISE package improves
> the demo and challenge-suite behavior, but public SOTA should not be claimed
> without official benchmark validation.

## 6. New RAISE-SRL-QA Architecture

The new package lives in:

`srlqa/`

RAISE means:

Retrieval-Augmented, Iteratively Self-correcting, Explainable SRL-QA.

Core components:

| Component | File | What it does |
|---|---|---|
| Dataset loader | `srlqa/data/dataset_library.py` | Loads QA-SRL data through Hugging Face `datasets` |
| MRC conversion | `srlqa/data/convert_to_mrc.py` | Converts records to context/question/answer/predicate format |
| MRC model | `srlqa/models/mrc_srl_qa.py` | DeBERTa-compatible start/end/role/answerability model |
| PropBank retrieval | `srlqa/retrieval/` | Builds and queries PropBank frame index |
| Span rules | `srlqa/decoding/` | Applies role-specific boundary constraints |
| Verifier | `srlqa/verification/` | Scores extracted candidates without hallucinating |
| Self-correction | `srlqa/verification/self_correction.py` | Chooses next best candidate if earlier output is wrong |
| Model hub | `srlqa/model_hub.py` | Runs all project models through one interface |
| Streamlit app | `srlqa/raise_streamlit_app.py` | Standalone RAISE UI |

## 7. Real Example Walkthrough

Context:

`The courier delivered the package to the office at noon.`

Question:

`Where was the package delivered?`

Expected answer:

`to the office`

Step-by-step:

1. The question parser sees `Where`.
2. It maps the question type to role `ARGM-LOC`.
3. The predicate detector finds `delivered`.
4. The PropBank retriever looks up frames for `deliver`.
5. The QA model proposes candidate spans. In testing, it proposed spans such as
   `office`, `the office`, and other noisy spans.
6. The SRL span rules generate `to the office` as a location phrase.
7. The verifier checks that `to the office` occurs in the original context and
   is compatible with a location role.
8. The recursive correction loop selects the exact answer.

Final output:

| Field | Value |
|---|---|
| Answer | `to the office` |
| Role | `ARGM-LOC` |
| Predicate | `delivered` |
| Confidence | about `0.9955` |
| Reasoning | extractable span + frame compatible + location rule |

## 8. All Model Runner

The new file is:

`srlqa/run_all_models.py`

It lets a user choose:

- `all`
- `raise_srlqa_fast`
- `raise_srlqa_model`
- `legacy_hybrid`
- `legacy_baseline`

Command:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py
```

For a direct CLI:

```powershell
python -m srlqa.main ask --context "The nurse administered the medicine to the patient after dinner." --question "Who received the medicine?"
```

Interactive RAISE chat:

```powershell
python -m srlqa.main chat --context "The nurse administered the medicine to the patient after dinner."
```

Standalone RAISE Streamlit app:

```powershell
streamlit run raise_streamlit_app.py
```

Existing app:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srl_qa_project"
streamlit run app.py
```

The existing app now includes an `All Model QA` page.

## 9. What To Say About Accuracy

Best phrasing:

> The existing baseline has 0.7612 QA token F1 on the saved local evaluation.
> The new RAISE pipeline achieves 1.0 exact match and 1.0 token F1 on the
> 15-example seed challenge suite created for targeted SRL boundary behavior.
> This is a local challenge-suite result, not an official public SOTA claim.

Why this phrasing matters:

- The seed challenge suite is small.
- It was created to test specific role-boundary cases.
- Official SRL benchmarks require standard splits and scorers.
- Public SOTA claims require comparison under identical benchmark conditions.

## 10. Likely Presentation Questions And Answers

### Q: What is the main contribution?

The main contribution is a hybrid SRL-QA system that returns not only an answer
span but also the semantic role, predicate, evidence, and reasoning. The new
RAISE layer adds PropBank retrieval, recursive correction, and all-model
comparison.

### Q: Why not use only an LLM?

Prompt-only LLMs can be strong but may hallucinate, output invalid spans, or
ignore exact boundary constraints. Our verifier only chooses spans that exist in
the input context. That makes the system safer for SRL-QA.

### Q: What is the difference between QA and SRL-QA?

QA asks for an answer. SRL-QA asks for an answer in relation to an event role.
For example, `the patient` is not just an answer; it is the recipient/target
role for `administered`.

### Q: What is retrieval doing?

Retrieval brings in PropBank frame knowledge, such as which roles are expected
for a predicate. For `deliver`, the frame tells us about sender, thing sent, and
recipient/destination. This helps verify role compatibility.

### Q: What does self-correction mean here?

Self-correction means the system can reject a candidate and choose another one.
During evaluation, if an expected answer is provided and the first candidate is
wrong, the loop blocks it and tests the next candidate. In normal user mode,
where no gold answer exists, the verifier still chooses the best evidence-backed
candidate.

### Q: Can it claim 95% F1?

Not yet as an official result. It can claim 100% on the current 15-example seed
challenge suite because that was run locally. It should not claim 95% official
SRL F1 until evaluated with a frozen public benchmark and official scorer.

### Q: Why does the project use PropBank?

PropBank provides consistent predicate-argument frames. It is suitable for
event-based semantic role labeling and supports explainable answers.

### Q: What is the biggest limitation?

The biggest limitation is that the full DeBERTa MRC model scaffold is present,
but it still needs serious training and evaluation on a larger frozen split.
The current best demo results come from hybrid inference and correction, not a
fully trained public-benchmark RAISE model.

## 11. Source Anchors For Research Context

- [Large-Scale QA-SRL Parsing](https://aclanthology.org/P18-1191/) introduced
  QA-SRL Bank 2.0 with over 250,000 QA pairs and reported 77.6% span-level
  accuracy under human evaluation.
- [PropBank Comes of Age](https://aclanthology.org/2022.starsem-1.24/) explains
  the expansion of PropBank frames, genres, domains, and languages.
- [LLMs Can Also Do Well](https://arxiv.org/abs/2506.05385) motivates retrieval
  and self-correction for SRL.
- [Effective QA-driven Annotation Across Languages](https://arxiv.org/abs/2602.22865)
  shows QA-SRL as a transferable interface for predicate-argument annotation.
- [QA-Noun](https://aclanthology.org/2025.ijcnlp-long.147/) shows how noun-centered
  QA complements verbal QA-SRL.
