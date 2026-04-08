# Presentation Package

This folder contains the generated presentation package for the CPU-first QA-SRL fine-tuning project.

## Build commands

- `python build_presentation_assets.py`
- `python build_presentation_docs.py`
- `python build_presentation_deck.py`

## Main outputs

- `MASTER_PROJECT_PRESENTATION.md`
- `pdfs/MASTER_PROJECT_PRESENTATION.pdf`
- `pdfs/01_survey.pdf`
- `pdfs/02_implementation.pdf`
- `pdfs/03_results_analysis.pdf`
- `pdfs/04_innovation.pdf`
- `pdfs/05_prompt_tuning.pdf`
- `pdfs/06_final_takeaways.pdf`
- `deck/QA_SRL_Finetuning_Academic_Viva_30_Slides.pptx`
- `deck/QA_SRL_Finetuning_Academic_Viva_30_Slides.pdf`

## Supporting content

- `manifest.json` keeps the shared section order, output names, figure IDs, claim policy, and slide allocation.
- `assets/` stores generated plots, diagrams, and the copied InstaShap example image.
- `sections/` stores the section markdown files used to build the section PDFs.
- `pdfs/pdf_index.json` and `deck/slide_manifest.json` record the generated output inventory.

## Source discipline

- This package uses only material inside `finetuning/`.
- Direct numeric claims are sourced from `results/evaluation_report.json` and `results/training_summary.json`.
- The master markdown starts with the Survey section and contains no standalone introduction block.
