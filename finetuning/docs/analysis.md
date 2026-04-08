# QA-SRL Project Analysis

## Scope

This project builds a CPU-feasible QA-SRL pipeline that:

- loads QA-SRL Bank 2.1 through Python code,
- fine-tunes a foundation model that fits on an 8 GB RAM CPU machine,
- predicts structured semantic roles for a sentence and predicate,
- renders those roles as natural-language QA pairs,
- computes InstaShap-style token attributions,
- and exposes the trained model through Streamlit.

## Actual benchmark configuration

The final verified run used:

- `google/flan-t5-small` as the base model
- LoRA rank `8`
- `600` train / `100` validation / `100` test grouped predicate examples
- `3` epochs
- CPU-only execution

This is a deliberately reduced benchmark slice from the official QA-SRL Bank 2.1 split. It is representative and reproducible on this machine, but it is not the full-bank training recipe used in research-scale systems.

## Actual results

The latest evaluation in [`results/evaluation_summary.md`](../results/evaluation_summary.md) reports:

| Model | Token F1 | Exact Match | Role Coverage | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot `google/flan-t5-small` | 0.0000 | 0.0000 | 0.0000 | 0.3140 |
| Fine-tuned adapter | 0.3257 | 0.0400 | 0.8075 | 0.5191 |

XAI metrics:

- Plausibility: `0.2220`
- Faithfulness: `0.7248`
- Combined XAI score: `0.4734`

Per-domain token F1:

- `TQA`: `0.3382`
- `wikinews`: `0.3432`
- `wikipedia`: `0.2592`

Training highlights:

- Best validation token F1 during training: `0.3425`
- Training time: about `685` seconds

For clarity: the authoritative final benchmark is stored in `results/evaluation_report.json`, not `results/training_summary.json`. The training summary captures the state of metrics at the end of the training script, while the evaluation report reflects the latest postprocessing and benchmark pass.

## What improved accuracy

The final checkpoint outperformed the earlier quick runs because of six practical changes:

1. A corrected role taxonomy.
   The initial mapping over-predicted one label and polluted the supervision signal. Replacing it with a cleaner `AGENT / THEME / LOCATION / TIME / MANNER / REASON / ...` inventory improved learning stability.

2. A shorter prompt.
   Small T5-family models copied the tail of a long instruction prompt. Switching to a compact `semantic role extraction / predicate / sentence / labels:` prompt improved structural consistency.

3. Extractive post-processing.
   QA-SRL answers should be sentence spans. The final pipeline snaps generated answers back to the closest source span before scoring them.

4. Lightweight role reassignment heuristics.
   Phrases like `in the same area`, `with the sugar reward`, and `last Tuesday` are reassigned to more plausible roles after decoding.

5. A fallback extractor for empty generations.
   Some small-model outputs repeat the source sentence or emit a fragment with no role labels. The final pipeline converts those failures into conservative best-effort role guesses so the deployed system remains functional.

6. A fairer evaluator.
   The zero-shot baseline is now scored on the same 100-example test slice as the fine-tuned model, so the comparison in the docs is apples-to-apples.

## Why this is not true SOTA

The user asked for a state-of-the-art 2025-2026 QA-SRL system with F1 above 85. On this machine, that is not realistic.

Reasons:

- `flan-t5-small` is a compact model chosen for CPU feasibility, not top-end QA-SRL accuracy.
- The run uses a reduced official benchmark slice rather than full-bank, long-horizon fine-tuning.
- The system predicts a compact structured role inventory instead of full question-slot generation.
- InstaSHAP in this project is an NLP-oriented, gradient-based adaptation for token attribution, not a full reimplementation of the original additive-model training regime.

The delivered system is therefore best described as:

- complete,
- functional,
- reproducible on 8 GB CPU hardware,
- meaningfully better than its zero-shot baseline,
- stronger than the earlier intermediate local runs after the recovery heuristics were added,
- but still far from the full research-scale target.

## Literature context

Important milestones relevant to this project:

- He, Lewis, and Zettlemoyer (EMNLP 2015) introduced QA-SRL as a question-answer representation for predicate-argument structure.
- FitzGerald et al. (ACL 2018) introduced QA-SRL Bank 2.0 and the first large-scale neural QA-SRL parser.
- Roit et al. (ACL 2020) improved annotation quality with a controlled crowdsourcing protocol and released a higher-quality gold dataset.
- Klein et al. (COLING 2020) extended the idea to nominalizations with QANom.
- Recent 2026 work on cross-lingual QA-driven predicate-argument annotation shows the framework is still actively evolving beyond English and beyond verb-only setups.
- InstaSHAP (ICLR 2025) motivates fast, attribution-oriented explanation design by emphasizing near-instant Shapley-style explanation behavior.

## Where this project is novel in practice

This implementation is not a novel published SOTA method, but it does combine pieces that are useful in a single engineering package:

- QA-SRL Bank 2.1 ingestion with no manual data download step
- CPU-feasible LoRA fine-tuning
- extractive cleanup for generative role outputs
- InstaShap-style token attribution for QA-SRL predictions
- a Streamlit UI connected to the trained adapter
- generated docs and benchmark artifacts inside the project folder

## Recommended next steps

If the goal is stronger accuracy, the next upgrades should be:

1. Move from `flan-t5-small` to `flan-t5-base` or a quantized encoder-decoder model on more memory.
2. Train on the full grouped QA-SRL Bank split, not the 600-example slice.
3. Add constrained decoding so only valid role labels can be emitted.
4. Add an explicit extractive span head or reranker instead of relying on generation alone.
5. Extend the evaluator to full question-slot QA-SRL metrics, not only the compact role projection used here.
6. Add a stronger explanation benchmark with human plausibility labels instead of token-overlap proxies.

## Sources

- [He et al. 2015: Question-Answer Driven Semantic Role Labeling](https://aclanthology.org/D15-1076/)
- [FitzGerald et al. 2018: Large-Scale QA-SRL Parsing](https://arxiv.org/abs/1805.05377)
- [Roit et al. 2020: Controlled Crowdsourcing for High-Quality QA-SRL Annotation](https://aclanthology.org/2020.acl-main.626/)
- [Klein et al. 2020: QANom](https://aclanthology.org/2020.coling-main.274/)
- [QA-SRL publications page](https://qasrl.org/publications/)
- [Effective QA-driven Annotation of Predicate-Argument Relations Across Languages (2026 preprint)](https://arxiv.org/abs/2602.22865)
- [InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly (ICLR 2025 / arXiv)](https://arxiv.org/abs/2502.14177)
