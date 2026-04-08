# Understanding The Project

## What This Project Is About

This project is about building Question Answering (QA) systems that understand **who did what, to whom, where, when, how, and why**.

The project uses **Semantic Role Labeling (SRL)** to structure text before answering questions. SRL converts a sentence into an event-based structure.

Example sentence:

```text
The courier delivered the package to the office at noon.
```

SRL view:

```text
Predicate: delivered
ARG0: The courier
ARG1: the package
ARGM-LOC: to the office
ARGM-TMP: at noon
```

Now a question like:

```text
Where was the package delivered?
```

can be answered by looking for the location role:

```text
ARGM-LOC -> to the office
```

## The Big Idea

Normal QA often retrieves text chunks and tries to extract an answer. This project adds SRL structure before answering.

Instead of only retrieving this:

```text
The courier delivered the package to the office at noon.
```

the system can retrieve this richer representation:

```text
delivered -> ARG0 -> The courier
delivered -> ARG1 -> the package
delivered -> ARGM-LOC -> to the office
delivered -> ARGM-TMP -> at noon
```

That makes the QA system easier to explain because every answer can be connected to a semantic role.

## Important Terms

### SRL

Semantic Role Labeling. It identifies the predicate and its arguments in a sentence.

### Predicate

The main event or action.

Example:

```text
delivered
```

### Argument

A participant or detail connected to the predicate.

Examples:

```text
ARG0: the doer
ARG1: the thing affected
ARGM-LOC: location
ARGM-TMP: time
```

### PropBank

PropBank is the dataset that provides real predicate-argument annotations. This project loads PropBank through NLTK.

### RAG

Retrieval-Augmented Generation or Retrieval-Augmented QA. In this project, RAG means retrieving useful SRL-structured evidence before answering.

### Explainable QA

QA where the answer is not just shown alone. The system also shows why the answer was selected, what evidence was retrieved, and how the answer connects to SRL roles.

## Folder Guide

### `srl_qa_project`

This is the older baseline project.

It contains:

- NLTK PropBank data loading
- PropBank to QA example generation
- A trained PropQA-Net checkpoint
- Evaluation outputs
- Plots and generated reports
- A Streamlit research app

Think of this folder as:

```text
Original SRL + QA baseline project
```

It is important because it contains the local NLTK PropBank data used by the newer demo.

Key data path:

```text
srl_qa_project/nltk_data
```

### `srlqa`

This is the newer RAISE-SRL-QA scaffold.

It contains:

- A cleaner Python package
- PropBank frame retrieval
- Deterministic SRL-QA logic
- Optional model-backed QA
- Model comparison runner
- Streamlit demo

Think of this folder as:

```text
Newer experimental SRL-QA framework
```

Important file:

```text
srlqa/retrieval/frame_store.json
```

This contains PropBank frame information used to explain predicates and roles.

### `srl_rag_demo`

This is the new standalone demo folder.

It contains:

- PropBank loader
- SRL document builder
- RAG retrieval logic
- Role-aware QA logic
- Explainable graph builder
- Streamlit app
- Smoke test

Think of this folder as:

```text
Final demo app for SRL + RAG + explainable QA
```

Main app:

```text
srl_rag_demo/app.py
```

Run it with:

```powershell
streamlit run srl_rag_demo\app.py
```

## How The System Works

The new demo follows this pipeline:

```text
PropBank / user text
        |
        v
SRL document creation
        |
        v
Hybrid retrieval
        |
        v
Role-aware answer selection
        |
        v
Semantic graph explanation
        |
        v
Streamlit demo
```

## Step 1: Load PropBank

The project loads PropBank through NLTK from:

```text
srl_qa_project/nltk_data
```

It reads PropBank instances and aligns them with Treebank sentences so answer spans can be reconstructed.

Current local data summary:

```text
PropBank instances: 112,917
Treebank-backed usable instances: 9,353
```

## Step 2: Build SRL Documents

Each useful sentence becomes an SRL document.

An SRL document stores:

- source
- context sentence
- predicate
- predicate lemma
- PropBank roleset
- semantic arguments
- frame hint
- retrieval text

Example:

```text
context: The courier delivered the package to the office at noon.
predicate: delivered
ARG0: The courier
ARG1: the package
ARGM-LOC: to the office
ARGM-TMP: at noon
```

## Step 3: Add User Documents

The Streamlit app also lets the user paste or upload text.

For pasted/uploaded text, the demo uses lightweight SRL heuristics. This means it tries to detect:

- subject before predicate
- object after predicate
- location phrases
- time phrases
- cause phrases
- manner adverbs

This is not a full neural SRL parser, but it keeps the demo local and reliable.

## Step 4: Retrieve Evidence

The RAG system retrieves documents using:

- sentence-transformer embeddings when available
- TF-IDF fallback when embeddings are unavailable

The retrieval text includes both the raw sentence and the SRL structure.

Example retrieval text:

```text
The courier delivered the package to the office at noon.
predicate: delivered
delivered -> ARG0 -> The courier
delivered -> ARG1 -> the package
delivered -> ARGM-LOC -> to the office
delivered -> ARGM-TMP -> at noon
```

## Step 5: Answer The Question

The question is mapped to an expected SRL role.

Examples:

```text
Who...?    -> ARG0 or ARG2
What...?   -> ARG1
Where...?  -> ARGM-LOC
When...?   -> ARGM-TMP
How...?    -> ARGM-MNR
Why...?    -> ARGM-CAU
```

Then the system chooses the best candidate span from the retrieved documents.

For:

```text
Where was the package delivered?
```

it looks for:

```text
ARGM-LOC
```

and returns:

```text
to the office
```

## Step 6: Build The Reasoning Graph

The demo creates a semantic graph using NetworkX and Plotly.

The graph can include:

- question node
- retrieved document nodes
- predicate nodes
- role nodes
- frame nodes
- answer candidate nodes
- final answer node

This is useful for explaining why the answer was selected.

## Main Demo Command

Run:

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
streamlit run srl_rag_demo\app.py
```

Then use:

```text
Question: Where was the package delivered?
Text: The courier delivered the package to the office at noon.
```

Expected answer:

```text
to the office
```

## Smoke Test Command

Run:

```powershell
python srl_rag_demo\smoke_test.py
```

This checks:

- PropBank loading
- SRL document creation
- retrieval
- QA answer selection
- reasoning graph creation

## Other Helpful Files

### `PROJECT_EXPLANATION.md`

A formal project explanation for presentation.

### `RUN_ALL_EXPERIMENTS_DEMO.md`

A command runbook for running demos and experiments.

### `srl_rag_demo/EXISTING_WORKSPACE_ANALYSIS.md`

A focused analysis of the existing folders and how the new demo reuses them.

## What To Say In A Presentation

Short version:

```text
This project uses Semantic Role Labeling to make question answering more structured and explainable. It loads real PropBank data through NLTK, converts sentences into predicate-argument structures, retrieves evidence using SRL-enhanced RAG, selects answers based on semantic roles, and visualizes the reasoning path as a graph in Streamlit.
```

## Limitations

- The new demo is local and deterministic, not a fully trained neural SRL parser.
- Pasted/uploaded text uses heuristics for SRL-style extraction.
- Model-backed features may need local Hugging Face weights or internet access.
- The legacy baseline and the new RAG demo serve different purposes: baseline modeling versus explainable demo.

## Final Mental Model

Remember the project like this:

```text
PropBank gives the role structure.
SRL turns text into event-role triples.
RAG retrieves the best structured evidence.
QA selects the role-matching answer span.
The graph explains the reasoning path.
Streamlit makes it demo-ready.
```
