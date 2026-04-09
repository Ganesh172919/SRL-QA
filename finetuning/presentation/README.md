# Presentation Package (Legacy Path)

This folder now mirrors the final manifest/source inventory, while generated outputs are written to `presentation_final/`.

## Build commands

- `python build_final_presentation_assets.py`
- `python build_presentation_docs.py`
- `python build_presentation_deck.py`

## Main outputs

- `../presentation_final/MASTER_FINAL_PRESENTATION.md`
- `../presentation_final/pdfs/MASTER_FINAL_PRESENTATION.pdf`
- `../presentation_final/deck/QA_SRL_Final_Conference_40_Slides.pptx`
- `../presentation_final/deck/QA_SRL_Final_Conference_40_Slides.pdf`

## Supporting content

- `manifest.json` keeps the shared section order, output names, figure IDs, claim policy, and slide allocation.
- `assets/` stores generated plots, diagrams, and the copied InstaShap example image.
- `sections/` stores the section markdown files used to build the section PDFs.
- `pdfs/pdf_index.json` and `deck/slide_manifest.json` record the generated output inventory.

## Source discipline

- This package uses only material inside `finetuning/`.
- Direct numeric claims are sourced from `results/evaluation_report.json` and `results/training_summary.json`.
- The master markdown starts with the Survey section and contains no standalone introduction block.
