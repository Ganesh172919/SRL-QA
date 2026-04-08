# SRL + RAG Explainable QA Demo

This is a standalone local demo for semantic-role-structured retrieval augmented QA.
It reuses:

- `../srl_qa_project/nltk_data` for NLTK PropBank and Treebank.
- `../srlqa/retrieval/frame_store.json` for PropBank frame hints.

It does not train a model and does not require API keys.

## Run

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
python -m pip install -r srl_rag_demo\requirements.txt
streamlit run srl_rag_demo\app.py
```

## Smoke Test

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
python srl_rag_demo\smoke_test.py
```

The smoke test loads PropBank through NLTK, builds a small SRL corpus, indexes a
sample pasted document, answers a role-aware question, and builds the graph JSON.

## Complete Project Health Check

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project"
python run_complete_project_check.py
```

This compiles the demo package, runs the controlled SRL + RAG demo evaluation,
reads the existing baseline and RAISE metric tables, and writes
`COMPLETE_PROJECT_FUNCTIONALITY_REPORT.md`.

Current controlled demo-suite result:

| Metric | Value |
|---|---:|
| Exact Match | 100.00% |
| Token F1 | 100.00% |
| Role Accuracy | 100.00% |

This is a controlled demo-suite value, not a full-corpus benchmark claim. Use
the legacy baseline metrics for full-test claims.

## Design

- PropBank documents are converted into SRL records with predicate, roleset,
  role arguments, context text, and frame hints.
- User pasted/uploaded documents are converted into lightweight SRL records with
  deterministic heuristics.
- Retrieval uses sentence-transformer embeddings when available and TF-IDF as a
  guaranteed local fallback.
- QA selects extractive spans from retrieved SRL arguments and heuristic spans.
- The reasoning graph uses NetworkX and Plotly to show question, retrieved
  documents, predicates, roles, frames, candidates, and selected answer.
