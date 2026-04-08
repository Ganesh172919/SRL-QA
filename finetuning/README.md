# QA-SRL Fine-Tuning with InstaShap on CPU

This folder is a fresh, self-contained QA-SRL project built for an 8 GB RAM CPU machine.

## What it does

- Downloads QA-SRL Bank 2.1 from the official public source using Python code.
- Converts predicate-level annotations into a compact role-generation format.
- Fine-tunes `google/flan-t5-small` with LoRA so the training run stays CPU-friendly.
- Evaluates the model with token F1, exact match, role coverage, ROUGE-L, and domain-wise F1.
- Computes InstaShap-style token attributions using gradient-times-input.
- Serves the trained model in a Streamlit interface that shows predicted QA pairs and explanations.

## Project layout

- `train.py`: fine-tuning and zero-shot baseline run
- `evaluate.py`: benchmark metrics and XAI evaluation
- `run_project.py`: end-to-end runner
- `app.py`: Streamlit UI
- `src/qasrl_cpu/`: reusable data, model, inference, metrics, and XAI modules
- `results/`: generated metrics, predictions, reports, and plots
- `artifacts/`: saved LoRA adapter and tokenizer files
- `docs/`: analysis, summary, and research prompt files

## Quick start

```powershell
cd "C:\Users\RAVIPRAKASH\Downloads\NLP Project\finetuning"
python train.py --train-limit 600 --validation-limit 100 --test-limit 100 --epochs 3 --batch-size 2 --gradient-accumulation-steps 2 --num-beams 1
python evaluate.py --train-limit 600 --validation-limit 100 --test-limit 100 --xai-limit 15 --num-beams 1
python app.py
```

`python app.py` now auto-launches Streamlit for convenience, so it is equivalent to running `streamlit run app.py`.

## Presentation package

The folder now includes a self-contained presentation build pipeline under `presentation/`.

Build commands:

```powershell
python build_presentation_assets.py
python build_presentation_docs.py
python build_presentation_deck.py
```

Main generated outputs:

- `presentation/MASTER_PROJECT_PRESENTATION.md`
- `presentation/pdfs/MASTER_PROJECT_PRESENTATION.pdf`
- `presentation/deck/QA_SRL_Finetuning_Academic_Viva_30_Slides.pptx`
- `presentation/deck/QA_SRL_Finetuning_Academic_Viva_30_Slides.pdf`

The master markdown starts with the Survey section, contains more than 2000 lines, and is backed only by material inside `finetuning/`.

## Latest verified run

The latest completed run in this folder used:

- Base model: `google/flan-t5-small`
- Fine-tuning: LoRA rank `8`
- Benchmark slice: `600` train / `100` validation / `100` test examples from the official QA-SRL Bank 2.1 split
- Training time: about `11.4` minutes on CPU

Latest evaluated metrics from `results/evaluation_summary.md`:

| Model | Token F1 | Exact Match | Role Coverage | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot `google/flan-t5-small` | 0.0000 | 0.0000 | 0.0000 | 0.3140 |
| Fine-tuned adapter | 0.3257 | 0.0400 | 0.8075 | 0.5191 |

XAI metrics from the same run:

- Plausibility: `0.2220`
- Faithfulness: `0.7248`
- Combined XAI score: `0.4734`

Domain token F1:

- `TQA`: `0.3382`
- `wikinews`: `0.3432`
- `wikipedia`: `0.2592`

Use `results/evaluation_report.json` and `results/evaluation_summary.md` as the final benchmark source of truth. `results/training_summary.json` remains the training-time artifact.

## What was improved during debugging

- Replaced a failing dataset URL with a stable raw source.
- Fixed the coarse role mapping so training labels were no longer dominated by false `RECIPIENT` assignments.
- Simplified the prompt into a T5-friendly `predicate + sentence + labels` format.
- Added extractive answer snapping so generated spans are pulled back onto the source sentence.
- Added heuristic role re-assignment for obvious temporal, locative, instrumental, and oblique phrases.
- Added a conservative fallback extractor for empty or unstructured generations so the fine-tuned system can still emit best-effort roles.
- Updated evaluation so the zero-shot baseline is measured on the same 100-example test slice as the fine-tuned model.
- Kept the zero-shot baseline raw during evaluation, while allowing the deployed fine-tuned system to use the recovery heuristics that make the UI and final pipeline practical.

## Important note on accuracy

This implementation is deliberately designed around the real machine constraint here: 8 GB RAM with CPU-only training. That means it uses a smaller instruction-tuned model and a compact structured-output formulation instead of a full Flan-T5-XL or 7B-class setup. The code is functional end to end, but any claims about state-of-the-art 2025-2026 accuracy should be treated separately from the actual local run metrics produced in `results/`.

The result is a working and meaningfully improved QA-SRL system for this hardware class, but it is not a true SOTA QA-SRL parser. Reaching the `>85%` target from large-model research-style setups would require larger models, more training data, longer runs, and stronger decoding or extraction modules than are realistic on this 8 GB CPU-only machine.
