# Project Summary

## Goal

Build a new QA-SRL project in a separate `finetuning` folder that is fully functional on an 8 GB RAM CPU machine, uses a fine-tunable foundation model, adds InstaShap-style explainability, exposes a Streamlit interface, and writes out evaluation artifacts and analysis documents.

## What was delivered

- A new isolated project under `finetuning/`
- Official QA-SRL Bank 2.1 download and preprocessing code
- LoRA fine-tuning pipeline for `google/flan-t5-small`
- Evaluation pipeline with:
  - token F1
  - exact match
  - role coverage
  - ROUGE-L
  - domain-wise token F1
  - XAI plausibility and faithfulness
- InstaShap-style explainer for token-level attributions
- Streamlit app connected to the saved adapter
- Result artifacts in `results/`
- Model artifacts in `artifacts/flan_t5_small_lora/`
- Updated docs and analysis files

## Final verified metrics

From the latest `results/evaluation_summary.md`:

- Zero-shot `google/flan-t5-small`: token F1 `0.0000`, exact match `0.0000`, role coverage `0.0000`, ROUGE-L `0.3140`
- Fine-tuned adapter: token F1 `0.3257`, exact match `0.0400`, role coverage `0.8075`, ROUGE-L `0.5191`
- XAI plausibility `0.2220`
- XAI faithfulness `0.7248`
- Combined XAI score `0.4734`

## Functional example

Verified local inference after training:

- Sentence: `The company successfully launched its new product last Tuesday.`
- Predicate: `launched`
- Output:
  - `AGENT: The company`
  - `THEME: its new product`
  - `TIME: last Tuesday`

## Main limitations

- This is a CPU-feasible engineering system, not a full research-scale SOTA parser.
- The benchmark run uses a 600/100/100 slice for practicality.
- The model is much smaller than the LLMs normally used in top-end 2025-2026 experiments.
- The explainability component is an InstaShap-style adaptation for transformer token attribution rather than a full additive-model reproduction of the 2025 paper.

## Most important conclusion

The project is now complete, runnable, and documented. Accuracy was materially improved over the earlier broken or zero-shot state, errors were fixed, the model is connected to a working UI, and the evaluation metrics are written into the project files. The remaining gap is not a software-completeness gap; it is a hardware-and-model-capacity gap.
