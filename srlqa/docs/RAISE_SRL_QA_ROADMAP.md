# RAISE-SRL-QA Implementation Roadmap

This package implements the first engineering pass of the innovation plan.

## Implemented Scaffold

- Library dataset loading through Hugging Face `datasets`.
- Library model loading through Hugging Face `transformers`.
- Online asset search through `huggingface_hub.HfApi`.
- PropBank frame parsing from the existing local NLTK PropBank XML files.
- MRC conversion for QA-SRL-style records.
- DeBERTa-compatible multi-task MRC SRL-QA model heads.
- Hard-negative utilities.
- Constrained span decoder.
- Evidence-only verifier and self-correction loop.
- Teacher-student distillation filtering hooks.
- QA-Noun-style nominal templates.
- Proto-role diagnostic hooks.
- Calibration, span metrics, and leaderboard generation.

## Next Milestones

1. Install `srlqa/requirements.txt`.
2. Run `python -m srlqa.main download --dataset marcov/qa_srl_promptsource --model microsoft/deberta-v3-base`.
3. Run `python -m srlqa.main preview-data --max-examples 10` and inspect the normalized schema.
4. Build the PropBank frame store with `python -m srlqa.main build-frame-index`.
5. Train the first MRC baseline on a small sample.
6. Add frame hints to the MRC examples.
7. Evaluate with `srlqa/evaluation/offline_eval.py`.
8. Expand `challenge_suite_v2.json` from the current seed set toward 300 frozen examples.

## Feasibility Note

This scaffold does not claim 95% token-F1. It creates the architecture needed to
measureably pursue that target with stronger supervised training, frame
retrieval, constrained decoding, verifier correction, hard negatives, and
calibrated reporting.
