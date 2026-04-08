# Key Terms And Project Work

## Purpose

This file explains the important terms used in the project, connects each term to the actual implementation, and summarizes what has been completed.

The project can be understood as:

```text
Semantic Role Labeling -> PropBank data -> Role-aware QA -> RAG retrieval -> Explainable graph demo
```

## 1. Semantic Role Labeling (SRL)

Semantic Role Labeling is the task of identifying who did what, to whom, where, when, and how in a sentence.

Example:

```text
Sentence: The courier delivered the package to the office at noon.
Predicate: delivered
ARG0: The courier
ARG1: the package
ARGM-LOC: to the office
ARGM-TMP: at noon
```

Project connection:

- `srl_qa_project` uses SRL-style labels from PropBank to build question-answer examples.
- `srlqa` uses SRL roles for constrained answer selection and verification.
- `srl_rag_demo` uses SRL structures to retrieve evidence and explain answers.

What we have done:

- Loaded PropBank SRL data through NLTK.
- Converted predicate-role annotations into structured QA evidence.
- Used SRL roles to make answer selection more explainable.

## 2. Predicate

A predicate is usually the main verb or event in a sentence.

Example:

```text
delivered
```

In the sentence:

```text
The courier delivered the package to the office at noon.
```

the predicate is `delivered`.

Project connection:

- Predicates are used to locate the event being discussed.
- The SRL + RAG demo creates graph nodes for predicates.
- PropBank frames describe how each predicate is expected to behave.

What we have done:

- Extracted predicates from PropBank instances.
- Connected predicates with roles such as `ARG0`, `ARG1`, and `ARGM-LOC`.
- Used predicates as part of the retrieval and graph explanation pipeline.

## 3. Argument

An argument is a participant or detail connected to a predicate.

Example:

```text
Predicate: delivered
ARG0: The courier
ARG1: the package
ARGM-LOC: to the office
ARGM-TMP: at noon
```

Project connection:

- Arguments become answer candidates for QA.
- Arguments are stored as structured fields in SRL documents.
- Arguments appear as graph nodes in the explainable QA demo.

What we have done:

- Built answer candidates from PropBank arguments.
- Used argument roles to decide which span best answers a question.
- Displayed argument evidence in the Streamlit demo.

## 4. Core Roles: ARG0, ARG1, ARG2

Core roles describe the main participants in an event.

Common interpretation:

| Role | Common Meaning | Example |
|---|---|---|
| `ARG0` | Agent or doer | `The courier` |
| `ARG1` | Patient, theme, or thing affected | `the package` |
| `ARG2` | Recipient, destination, or secondary participant | depends on predicate |

Important note:

```text
ARG0 and ARG1 are predicate-specific. PropBank frames define their exact meaning.
```

Project connection:

- `srlqa/retrieval/frame_store.json` stores PropBank frame information.
- The app uses frame definitions when available to explain role meaning.
- Role-aware QA uses these labels to select better answer spans.

What we have done:

- Reused PropBank role definitions from the local frame store.
- Added role descriptions into retrieval text.
- Used roles to improve answer explainability.

## 5. Modifier Roles: ARGM

`ARGM` roles describe extra information such as location, time, manner, cause, or direction.

Common examples:

| Role | Meaning | Example |
|---|---|---|
| `ARGM-LOC` | Location | `to the office` |
| `ARGM-TMP` | Time | `at noon` |
| `ARGM-MNR` | Manner | `carefully` |
| `ARGM-CAU` | Cause | `because of rain` |
| `ARGM-DIR` | Direction | `toward the station` |

Project connection:

- Many natural questions map directly to modifier roles.
- "Where" questions often map to `ARGM-LOC`.
- "When" questions often map to `ARGM-TMP`.
- "How" questions often map to `ARGM-MNR`.

What we have done:

- Implemented role-aware question matching.
- Demonstrated location question answering with the courier example.
- Used modifier roles in the explainable graph output.

## 6. PropBank

PropBank is a linguistic dataset that annotates predicates and their arguments.

It provides structured examples like:

```text
predicate -> role -> span
```

Project connection:

- PropBank is the main dataset source for the project.
- NLTK is used to load PropBank instances.
- PropBank frames are used to describe predicate-specific roles.

What we have done:

- Loaded local PropBank data through NLTK.
- Confirmed the local NLTK corpus has `112,917` PropBank instances.
- Built Treebank-backed SRL documents from PropBank examples.
- Reused the PropBank frame store from `srlqa`.

## 7. NLTK

NLTK is a Python NLP library used for accessing linguistic datasets such as PropBank and Treebank.

Project connection:

- `srl_qa_project/nltk_data` contains local NLTK data.
- `srl_rag_demo` sets the NLTK data path to reuse this local corpus.
- NLTK PropBank and Treebank are used to reconstruct sentence-level examples.

What we have done:

- Created a PropBank loader in `srl_rag_demo`.
- Loaded PropBank instances without requiring an external API.
- Kept the demo local and CPU-friendly.

## 8. Treebank

Treebank contains parsed sentences that help reconstruct the text around PropBank annotations.

Project connection:

- PropBank annotations reference Treebank sentences.
- Treebank-backed examples are easier to convert into readable QA evidence.
- The demo reports Treebank-backed usable document counts.

What we have done:

- Used Treebank-backed PropBank instances for structured SRL documents.
- Confirmed the demo can build a capped set of usable PropBank SRL documents.

## 9. Question Answering (QA)

Question Answering is the task of returning an answer for a given question and context.

Example:

```text
Question: Where was the package delivered?
Context: The courier delivered the package to the office at noon.
Answer: to the office
```

Project connection:

- `srl_qa_project` builds and evaluates a baseline SRL-QA pipeline.
- `srlqa` adds role-aware and verification-based QA.
- `srl_rag_demo` combines retrieved evidence with role-aware QA.

What we have done:

- Created QA examples from PropBank roles.
- Compared baseline and role-aware QA behavior.
- Built a Streamlit interface for asking questions over retrieved SRL evidence.

## 10. Extractive QA

Extractive QA means the answer is selected as a span from the given context instead of being freely generated.

Example:

```text
Context: The courier delivered the package to the office at noon.
Extracted answer: to the office
```

Project connection:

- The project focuses on extractive spans because they are easier to verify.
- SRL arguments naturally provide extractive answer spans.
- The optional transformer QA in the demo is also extractive.

What we have done:

- Used SRL argument spans as answer candidates.
- Added role-aware span selection.
- Kept the live demo explainable by linking answers back to evidence spans.

## 11. RAG

RAG means Retrieval-Augmented Generation or Retrieval-Augmented QA.

In this project, RAG means:

```text
retrieve relevant SRL evidence first, then answer using that evidence
```

Project connection:

- `srl_rag_demo` implements the final SRL + RAG demo.
- Retrieved documents include both raw text and SRL triples.
- Retrieval is used before answer selection.

What we have done:

- Built a local retriever using TF-IDF fallback.
- Added optional sentence-transformer embeddings.
- Indexed PropBank examples and user-provided documents.
- Returned top retrieved SRL evidence for the question.

## 12. Hybrid Retrieval

Hybrid retrieval combines more than one retrieval strategy.

In this project:

```text
Embedding retrieval + TF-IDF fallback
```

Project connection:

- If sentence-transformer embeddings are available, the demo can use them.
- If embeddings fail or are unavailable, the demo still works using TF-IDF.
- This makes the app reliable for CPU-only local demos.

What we have done:

- Implemented a hybrid retriever in `srl_rag_demo/retrieval.py`.
- Added environment settings to avoid TensorFlow conflicts.
- Verified retrieval works with TF-IDF fallback.

## 13. TF-IDF

TF-IDF is a classical text retrieval method that scores words by importance.

Project connection:

- TF-IDF is the safe fallback retrieval backend.
- It allows the demo to work without downloading large models.
- It is useful for fast local testing.

What we have done:

- Implemented TF-IDF retrieval with `sklearn`.
- Verified the smoke test returns non-empty retrieval results.

## 14. Sentence Transformers

Sentence Transformers create dense vector embeddings for text similarity.

Project connection:

- The demo can use sentence-transformer embeddings when installed and available.
- Embeddings help retrieve semantically similar evidence.
- The app still remains functional without them.

What we have done:

- Added optional embedding retrieval support.
- Used TF-IDF fallback to keep the demo robust.

## 15. Semantic Graph

A semantic graph represents the reasoning path as connected nodes and edges.

Example graph path:

```text
Question -> Retrieved Document -> Predicate -> Role -> Answer
```

Project connection:

- `srl_rag_demo/graphing.py` builds graph explanations.
- NetworkX creates the graph structure.
- Plotly displays the graph in Streamlit.

What we have done:

- Created graph nodes for question, documents, predicates, roles, frames, candidates, and final answer.
- Added graph edges showing retrieval score, role match, and answer selection.
- Added downloadable graph JSON in the demo.

## 16. Explainable QA

Explainable QA means the system not only gives an answer but also shows why the answer was chosen.

Project connection:

- SRL roles explain what each answer span means.
- PropBank frames explain predicate-specific role behavior.
- Graphs visualize the reasoning path.

What we have done:

- Built a Streamlit graph view for answer explanation.
- Connected answers to retrieved evidence and SRL roles.
- Made the final demo easier to present and understand.

## 17. Frame Store

A frame store contains predicate frame information, including role definitions.

Project connection:

- `srlqa/retrieval/frame_store.json` contains the PropBank frame store.
- `srl_rag_demo` reuses this frame store.
- Frame information improves the explanation of roles.

What we have done:

- Reused the existing frame store rather than recreating it.
- Added frame hints to retrieval documents.
- Connected frames to the reasoning graph.

## 18. Streamlit

Streamlit is a Python framework for building interactive demo apps.

Project connection:

- The final demo app is in `srl_rag_demo/app.py`.
- The app supports question input, document input, retrieval settings, evidence display, and graph explanation.

What we have done:

- Built a complete Streamlit demo for SRL + RAG QA.
- Added tabs for QA, corpus/index status, retrieved evidence, and graph reasoning.
- Verified the Streamlit app starts locally.

Run command:

```powershell
streamlit run srl_rag_demo\app.py
```

## 19. Accuracy, Exact Match, And Token F1

Accuracy and Exact Match measure whether the predicted answer exactly matches the expected answer.

Token F1 gives partial credit when the predicted answer overlaps with the expected answer.

Example:

```text
Expected: to the office
Predicted: delivered the package to the office at noon
Exact Match: 0%
Token F1: partial credit
```

Project connection:

- `srl_qa_project` reports full baseline metrics.
- `srlqa` reports model comparison metrics.
- The new demo is mainly for explainability and live demonstration.

What we have done:

- Reported legacy full-test metrics.
- Reported controlled challenge-suite metrics.
- Clearly separated full-test, curated-suite, smoke-test, and FAST_DEV results.

## 20. Project Folder Connection

| Folder / File | Meaning | What We Have Done |
|---|---|---|
| `srl_qa_project/` | Legacy PropQA-Net project | Built and evaluated the original SRL-QA baseline. |
| `srlqa/` | Newer RAISE-SRL-QA framework | Added role-aware retrieval, verification, correction, and comparison utilities. |
| `srl_rag_demo/` | Final standalone SRL + RAG demo | Built a local Streamlit app for SRL retrieval, QA, and graph explanations. |
| `propbank_srlqa_artifacts/` | LoRA experiment artifacts | Kept early FAST_DEV training outputs and summaries. |
| `propbank_srlqa_2b_artifacts/` | QLoRA / generative experiment artifacts | Kept experimental generative outputs for future reference. |
| `RUN_ALL_EXPERIMENTS_DEMO.md` | Experiment runbook | Added commands for running demos and experiments. |
| `PROJECT_EXPLANATION.md` | Formal project explanation | Added a complete project overview. |
| `PROJECT_UNDERSTANDING_GUIDE.md` | Beginner guide | Added simpler conceptual explanation. |
| `SURVEY_INNOVATION_IMPLEMENTATION_ANALYSIS.md` | Survey-to-analysis connection | Added presentation-safe metrics and project story. |
| `KEY_TERMS_PROJECT_WORK.md` | This glossary | Connects key terms to project implementation and completed work. |

## 21. What We Have Completed

The completed work can be summarized as:

| Work Area | Completed Work |
|---|---|
| Dataset loading | Loaded PropBank through local NLTK data. |
| SRL processing | Converted PropBank predicate-role annotations into structured evidence. |
| Baseline QA | Preserved and documented the legacy PropQA-Net baseline. |
| Improved QA | Used role-aware answer selection, verification, and hybrid reranking. |
| RAG demo | Built a standalone SRL + RAG Streamlit app. |
| Explainability | Built semantic graph reasoning paths using NetworkX and Plotly. |
| Evaluation | Collected accuracy, exact match, token F1, role accuracy, and SRL metrics. |
| Documentation | Added project explanation, understanding guide, experiment runbook, and survey-analysis connection docs. |

## 22. Final Project Explanation In One Paragraph

This project builds an explainable question answering system using Semantic Role Labeling. It starts from PropBank data loaded through NLTK, turns predicate-argument annotations into structured QA evidence, compares baseline and role-aware approaches, and finishes with a Streamlit SRL + RAG demo. The final demo retrieves SRL-structured documents, selects an answer using semantic roles, and visualizes the reasoning path as a graph so users can see how the answer connects to the question, retrieved document, predicate, role, and evidence span.

