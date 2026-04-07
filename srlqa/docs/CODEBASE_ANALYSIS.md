# Existing Codebase Analysis

Date: 2026-04-07

Workspace root: `C:\Users\RAVIPRAKASH\Downloads\NLP Project`

## Root Project

- `.gitignore`: Git ignore rules.
- `README.md`: Points users to the runnable `srl_qa_project/` baseline and
  its `python main.py --mode full` pipeline.
- `FINAL_PROJECT_PRESENTATION.md`, `FINAL_PROJECT_PRESENTATION_40_SLIDES.pptx`,
  `build_final_project_ppt.py`, and `ppt_assets/`: presentation generation and
  rendered assets. These are deliverable artifacts and are unrelated to runtime
  SRL-QA code.

## Existing Runtime Code: `srl_qa_project/`

- `config.py`: Central dataclass configuration for paths, data splits, the
  BiLSTM-style baseline model, training, and runtime device.
- `data_loader.py`: Loads PropBank through local NLTK assets, aligns instances
  with the Penn Treebank sample, generates BIO tags and QA examples, builds
  train/validation/test splits, computes statistics, creates vocabularies, and
  returns PyTorch dataloaders.
- `model.py`: Implements `PropQANet`, a word/POS/predicate-flag BiLSTM with SRL
  BIO classification plus QA start/end span scoring.
- `trainer.py`: Trains `PropQANet`, computes validation EM/token-F1, serializes
  vocabularies, and writes the best checkpoint.
- `evaluator.py`: Loads the checkpoint, produces prediction records, computes
  QA exact match/token-F1, SRL micro/macro metrics, and plots.
- `qa_inference.py`: Wraps checkpoint inference for raw context/question pairs
  with simple tokenization, POS heuristics, and predicate guessing.
- `hybrid_qa.py`: Adds the current hybrid inference layer with question intent
  parsing, rule-based span candidates, optional Transformers QA pipeline,
  optional sentence-transformer embeddings, and role-aware reranking.
- `benchmark.py`: Evaluates baseline and hybrid tracks over challenge/test
  subsets, aggregates metrics, and generates ablation/latency/heatmap plots.
- `app.py`: Streamlit UI for the current baseline/hybrid system.
- `main.py`: Master runner for train/eval/infer/ask/app/benchmark/report/full
  modes.

Observed issue: `main.py` calls `generate_all_pdfs(...)` in `report` and `full`
mode, but the visible top-level imports do not import that symbol and the source
file for `pdf_generator.py` is not present in `srl_qa_project/` even though a
pycache entry exists. I did not modify the old project, but the new `srlqa`
folder avoids depending on that report path.

## Existing Data And Artifacts

- `srl_qa_project/data/val.json` and `srl_qa_project/data/test.json`: Current
  cached splits. `train.json` is referenced by config but was not present in the
  live data folder during inspection.
- `srl_qa_project/data/challenge_suite.json`: Small manually curated challenge
  set.
- `srl_qa_project/checkpoints/best_model.pt`: Saved baseline checkpoint.
- `srl_qa_project/nltk_data/corpora/propbank`: Local PropBank corpus and XML
  frame files used by the baseline.
- `srl_qa_project/nltk_data/corpora/treebank`: Local Penn Treebank sample used
  for sentence alignment.
- `srl_qa_project/results/metrics.json`: Saved baseline metrics. PowerShell's
  JSON converter fails on this file because duplicate mixed-case role keys are
  present, so robust evaluation tooling should use Python JSON loading or clean
  role normalization.
- `srl_qa_project/results/data_statistics.json`: Saved data statistics. It
  reports `qa_pair_count = 23007` and live `usable_instances = 9353`.
- `srl_qa_project/results/plots`: Plot artifacts from the existing pipeline.
- `srl_qa_project/outputs`: Final PDFs and implementation export.

## Baseline State

The saved metrics show:

| Metric | Value |
|---|---:|
| QA exact match | 0.5184004636 |
| QA token F1 | 0.7611748435 |
| SRL micro F1 | 0.7133258791 |
| SRL macro F1 | 0.1618963367 |
| QA pairs | 23,007 |
| Usable PropBank instances in live stats | 9,353 |

The original architecture is a solid local baseline, but its main limitation is
span precision. The new `srlqa/` package therefore focuses on a stronger MRC
reader, PropBank frame retrieval, constrained decoding, verifier-based
self-correction, hard negatives, distillation, nominal semantics, proto-role
diagnostics, calibration, and leaderboard reporting.
