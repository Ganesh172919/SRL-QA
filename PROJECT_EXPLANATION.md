# Project Explanation: Semantic Role Labeling + Question Answering + RAG

## Overview

This project explores question answering systems that use Semantic Role Labeling (SRL) to make answers more structured and explainable. Instead of treating a sentence or document as plain text only, the system identifies events, predicates, and semantic roles such as:

- `ARG0`: who performs the action
- `ARG1`: what is affected
- `ARG2`: receiver, beneficiary, or secondary participant
- `ARGM-LOC`: location
- `ARGM-TMP`: time
- `ARGM-MNR`: manner
- `ARGM-CAU`: cause or reason

The core idea is that SRL can improve question answering by turning documents into predicate-argument structures. These structures can then support retrieval, answer extraction, and visual reasoning paths.

## Main Goal

The goal is to build and demonstrate an SRL-grounded QA system with explainability:

1. Load real PropBank SRL data through NLTK.
2. Build QA examples from predicate-argument annotations.
3. Use SRL structures to retrieve better evidence for questions.
4. Answer questions using role-aware span selection.
5. Visualize reasoning using semantic graphs.
6. Provide a Streamlit demo for presentation.

## Data Source

The project uses PropBank through NLTK. PropBank provides predicate-argument annotations over Treebank sentences. For example, in a sentence like:

```text
The courier delivered the package to the office at noon.
```

An SRL structure can look like:

```text
delivered -> ARG0 -> The courier
delivered -> ARG1 -> the package
delivered -> ARGM-LOC -> to the office
delivered -> ARGM-TMP -> at noon
```

The local workspace contains NLTK data under:

```text
srl_qa_project/nltk_data
```

The current local corpus check found:

- `112,917` PropBank instances visible through NLTK
- `9,353` Treebank-backed instances usable for local token-span reconstruction

## Workspace Structure

### `srl_qa_project`

This is the original legacy PropQA-Net project. It includes:

- NLTK PropBank loading
- PropBank-to-QA preprocessing
- A classical SRL + QA model
- A saved checkpoint
- Evaluation results, plots, and generated PDFs
- A Streamlit research app

It is useful as the main source of the real NLTK PropBank data and as a baseline QA system.

### `srlqa`

This is the newer RAISE-SRL-QA scaffold. It includes:

- A cleaner package structure
- PropBank frame retrieval
- Deterministic SRL-QA logic
- Optional model-backed QA using local Transformers models
- A model hub that can compare multiple model families
- A Streamlit app

It also contains a PropBank frame store at:

```text
srlqa/retrieval/frame_store.json
```

The frame store contains `4,659` PropBank frame records and is reused by the new SRL + RAG demo.

### `srl_rag_demo`

This is the standalone SRL + RAG explainable QA demo. It was created as a separate top-level folder so it does not modify the existing projects.

It includes:

- Local PropBank loader
- SRL document representation
- Pasted/uploaded document ingestion
- Hybrid retrieval using embeddings when available and TF-IDF fallback
- Role-aware answer selection
- NetworkX + Plotly semantic graph visualization
- Streamlit app for live demo

Main command:

```powershell
streamlit run srl_rag_demo\app.py
```

## System Architecture

The project has three layers:

1. Data and SRL layer
2. QA and retrieval layer
3. Explainability and demo layer

### 1. Data And SRL Layer

The system loads PropBank through NLTK and aligns PropBank annotations with local Treebank parses. It extracts:

- Sentence text
- Tokens
- Predicate
- Roleset ID
- Role descriptions
- Argument spans
- PropBank frame hints

These become structured SRL documents.

### 2. QA And Retrieval Layer

The RAG demo indexes both PropBank examples and user-provided text. Retrieval text includes:

- Raw context
- Predicate
- Roleset
- SRL triples
- Role descriptions
- PropBank frame hints

Retrieval supports:

- Sentence-transformer embeddings when available
- TF-IDF fallback for reliable local CPU demos

For answering, the system infers the question type and expected role. For example:

```text
Where was the package delivered?
```

maps to:

```text
Question type: WHERE
Expected role: ARGM-LOC
```

The system then ranks candidate spans from retrieved SRL arguments and heuristic spans.

### 3. Explainability And Demo Layer

The Streamlit app displays:

- Final answer
- Confidence
- Predicate
- Semantic role
- Source document
- Retrieved evidence
- Candidate spans
- PropBank frame hints
- Reasoning graph

The graph contains nodes such as:

- Question
- Retrieved document
- Predicate
- Semantic role
- PropBank frame
- Answer candidate
- Final answer

This makes the QA process easier to explain during a demo.

## Example Demo

Input context:

```text
The courier delivered the package to the office at noon.
```

Question:

```text
Where was the package delivered?
```

Expected answer:

```text
to the office
```

Reasoning:

```text
Question asks WHERE -> expected role is ARGM-LOC -> retrieved document contains delivered -> ARGM-LOC -> to the office -> select "to the office"
```

## How To Run

Use the main runbook:

```text
RUN_ALL_EXPERIMENTS_DEMO.md
```

Fastest demo path:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
python srl_rag_demo\smoke_test.py
streamlit run srl_rag_demo\app.py
```

Optional RAISE comparison:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python run_all_models.py --model all --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

## Key Strengths

- Uses real PropBank annotations through NLTK.
- Converts QA into an SRL-aware task rather than plain text matching only.
- Supports local retrieval without requiring API keys.
- Provides TF-IDF fallback when embedding models are unavailable.
- Includes graph-based explanations for reasoning paths.
- Separates the new demo from earlier project folders.

## Current Limitations

- The standalone RAG demo uses deterministic role-aware span selection, not a newly trained neural model.
- User-uploaded text uses lightweight SRL heuristics rather than full neural SRL parsing.
- The legacy project has existing generated PDFs, but its `pdf_generator.py` source file is not present in the current checkout.
- Full model-backed demos may depend on local model weights or internet access for Hugging Face downloads.

## Final Summary

This project demonstrates how SRL can make question answering more structured, explainable, and retrieval-friendly. The legacy project proves the PropBank data pipeline and baseline model. The newer `srlqa` package provides RAISE-style retrieval and correction ideas. The new `srl_rag_demo` folder brings those ideas together into a complete Streamlit demo with local RAG, SRL evidence, and graph-based reasoning.
