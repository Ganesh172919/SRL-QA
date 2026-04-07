# RAISE-SRL-QA

This folder is a clean, separate implementation scaffold for the requested
RAISE-SRL-QA innovation:

Retrieval-Augmented, Iteratively Self-correcting, Explainable Semantic Role
Labeling Question Answering.

It does not modify the existing `srl_qa_project/` baseline. The new code uses
library-based downloads for both the dataset and model:

- Default dataset: `marcov/qa_srl_promptsource` through Hugging Face `datasets`
- Alternative datasets found online: `luheng/qa_srl`, `biu-nlp/qa_srl2018`,
  `biu-nlp/qa_srl2020`
- Default encoder: `microsoft/deberta-v3-base` through Hugging Face
  `transformers`
- Optional QA teacher/verifier checkpoint: `deepset/deberta-v3-base-squad2`

## Quick Start

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\srlqa"
python -m pip install -r requirements.txt
python -m srlqa.main show-config
python -m srlqa.main search-assets --query qasrl
python -m srlqa.main download --model microsoft/deberta-v3-base --dataset marcov/qa_srl_promptsource
python -m srlqa.main preview-data --max-examples 5
python -m srlqa.main build-frame-index
```

`preview-data` requires the `datasets` package because it calls
`datasets.load_dataset(...)`. The default dataset is `marcov/qa_srl_promptsource`
because the currently installed `datasets` release refuses script-based dataset
repos such as `luheng/qa_srl`. `download` can still fetch repository snapshots
through `huggingface_hub`.

## Ask Questions

Single question with the PyTorch QA model plus SRL correction:

```powershell
python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

For a scored correction demo where the gold answer is known:

```powershell
python -m srlqa.main ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?" --expected-answer "to the office"
```

Run the seed challenge suite:

```powershell
python -m srlqa.main demo --max-examples 15
```

Ask multiple questions against one context:

```powershell
python -m srlqa.main chat --context "The nurse administered the medicine to the patient after dinner."
```

Compare every model family from one terminal runner:

```powershell
python run_all_models.py
```

Launch the standalone RAISE-SRL-QA Streamlit app:

```powershell
streamlit run raise_streamlit_app.py
```

Use `--no-model` for a fast deterministic SRL mode. The model-backed mode uses
`deepset/deberta-v3-base-squad2` for extractive QA candidates and then applies
PropBank retrieval, SRL span rules, and verifier-based correction.

## What Was Added

- `srlqa/data`: Hugging Face dataset loading and QA-SRL-to-MRC conversion.
- `srlqa/models`: DeBERTa-compatible multi-task MRC SRL-QA model wrapper.
- `srlqa/retrieval`: PropBank frame indexing and retrieval.
- `srlqa/decoding`: constrained span decoding and role priors.
- `srlqa/verification`: evidence-based verifier and self-correction loop.
- `srlqa/training`: collator, losses, hard negatives, and training entrypoint.
- `srlqa/evaluation`: token F1, exact match, calibration, and offline eval.
- `srlqa/ensemble`: calibrated confidence and weighted voting.
- `srlqa/reports`: leaderboard reporting.
- `docs/CODEBASE_ANALYSIS.md`: analysis of the existing project.
- `docs/EVALUATION_PROTOCOL.md`: frozen benchmark and claim discipline.
- `docs/PROJECT_PRESENTATION_MASTER_GUIDE.md`: full explanation for presentation.
- `docs/LAST_MINUTE_REVISION.md`: short final revision sheet.
- `docs/INNOVATION_RESULTS_AND_RESEARCH_COMPARISON.md`: local results and research comparison.

## Current Baseline Reference

From the existing project artifacts:

- QA exact match: `0.5184`
- QA token F1: `0.7612`
- SRL micro F1: `0.7133`
- SRL macro F1: `0.1619`
- QA pairs: `23,007`
- Usable PropBank instances in the live statistics file: `9,353`

The target remains a research goal, not a claim. This package makes the 95%
local token-F1 roadmap implementable and measurable, but it should not be
reported as achieved until a frozen benchmark script produces that score.
