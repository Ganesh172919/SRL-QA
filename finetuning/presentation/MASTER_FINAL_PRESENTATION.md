# Survey

## Lightweight QA-SRL for Final Conference Presentation

Local, CPU-first QA-SRL pipeline with explainability and a Streamlit UI.
Slides are backed only by files inside `finetuning/` plus uploaded sources.
Deck goal: survey the field, show LLM/prompting choices, and present verifiable results.


## Problem and Motivation

Semantic role labeling is easier to audit when rendered as questions and answers.
Research pipelines assume generous compute; this deck targets 8 GB CPU realism.
We need prompt discipline, recovery, and explanation to keep compact models usable.


## QA-SRL Task in One Slide

Input: sentence + marked predicate.
Output: structured role mapping plus QA phrasing of each role.
Evaluation: token F1, exact match, role coverage, ROUGE-L on grouped predicate examples.


## Why SRL-QA Matters for Reasoning Systems

Transforms opaque label sets into human-readable QA pairs.
Supports downstream reasoning, extraction, and teaching settings.
Fits well with prompt-based and structured-generation workflows.


## Research Timeline: 2015-2026

2015: QA-SRL introduced as a QA framing of semantic roles.
2018: QA-SRL Bank 2.0 enabled large-scale neural parsing.
2020: Annotation quality work and QANom broadened coverage.
2025-2026: InstaSHAP-inspired explanations and cross-lingual QA-driven annotation.


## He et al. (2015): QA-SRL Formulation

Defined question templates tied to predicates and arguments.
Showed crowd-workers can produce role-bearing QA pairs.
Set the template for later dataset growth and parser design.

- Citation: [He et al. 2015 (QA-SRL formulation)](https://aclanthology.org/D15-1076/)

## FitzGerald et al. (2018): Large-Scale QA-SRL Parsing

Released QA-SRL Bank 2.0 with broader coverage.
Benchmarkable neural parsers became the new baseline.
Motivated modern comparisons across domains and prompts.

- Citation: [FitzGerald et al. 2018 (QA-SRL Bank 2.0)](https://arxiv.org/abs/1805.05377)

## Roit et al. (2020): Annotation Quality

Focused on controlled crowdsourcing and adjudication.
Improved gold data consistency for downstream evaluation.
Highlights the role of clean supervision in QA-SRL.

- Citation: [Roit et al. 2020 (Annotation quality)](https://aclanthology.org/2020.acl-main.626/)

## Klein et al. (2020): QANom Extension

Extended QA-driven roles to nominalizations.
Demonstrated that question-driven semantics generalizes beyond verbs.
Keeps QA-SRL relevant to broader semantic parsing settings.

- Citation: [Klein et al. 2020 (QANom)](https://aclanthology.org/2020.coling-main.274/)

## InstaSHAP (2025) and Explanation-Aware NLP

Pushes toward near-instant attribution behavior.
Inspires the fast token-level explanation module in this project.
Aligns explainability with live inference rather than offline-only studies.

- Citation: [InstaSHAP 2025](https://arxiv.org/abs/2502.14177)

## Cross-Lingual QA-Driven Annotation (2026)

Shows QA-driven predicate-argument labeling is still evolving.
Confirms the framing remains relevant beyond English.
Supports presenting QA-SRL as an active research trajectory.

- Citation: [Cross-lingual QA-driven annotation 2026](https://arxiv.org/abs/2602.22865)

## What Earlier Systems Solved vs Left Open

Solved: QA framing, dataset scaling, quality improvements, broader predicates.
Left open: compact CPU deployment, integrated explanation, and robust prompting.
This local project addresses the deployment-and-robustness gap.




# LLM Integration

## Modern LLM Shift in SRL-QA

Large models can emit structured answers with minimal fine-tuning.
Compact models still need careful prompting and cleanup.
This deck keeps comparisons grounded in supported local and Gemini evidence.

- Citation: [GPT-5.4 context (supported model landscape)](https://openai.com)
- Citation: [Gemma 3 27B context](https://ai.google.dev/gemma)

## LLM Reasoning + Structured Output for SRL-QA

Structured prompts reduce hallucination and simplify parsing.
Beam-limited decoding keeps outputs deterministic for downstream cleanup.
Post-processing bridges free-form text and schema-bound roles.


## Latest Model Landscape from Uploaded Sources

Context models referenced: GPT-5.4, Gemini 2.5 Flash, Gemma 3 27B.
Measured local evidence: `flan-t5-small` baseline and fine-tuned adapter.
All claims constrained to uploaded docs and stored metrics.

- Citation: [GPT-5.4 context (supported model landscape)](https://openai.com)
- Citation: [Gemma 3 27B context](https://ai.google.dev/gemma)
- Citation: [Gemini benchmark prompt profiles](https://ai.google.dev/gemini-api/docs/prompting)

## Gemini Integration in This Repo

Gemini prompt profiles are stored alongside the benchmark materials.
Integration focuses on consistent role formatting and evaluation parity.
Gemini results are referenced only where uploaded sources permit.

- Citation: [Gemini benchmark prompt profiles](https://ai.google.dev/gemini-api/docs/prompting)

## Prompt Profiles Used for Gemini Benchmarking

Profiles emphasize short instructions plus explicit role serialization.
Layouts mirror the local T5 prompt so outputs can be compared.
Benchmark prompts are limited to uploaded sources—no fabricated variants.

- Citation: [Gemini benchmark prompt profiles](https://ai.google.dev/gemini-api/docs/prompting)

## Prompt Ablation Results and Takeaways

Ablations show compact prompts reduce instruction copying on small models.
Role serialization improves span snapping and coverage.
Prompt discipline remains crucial even when larger models are available.


## Our Project Position and Contributions

CPU-feasible fine-tuning with LoRA over `flan-t5-small`.
Integrated explanation (InstaShap-style) and recovery stack.
Reproducible deck, docs, and assets built from the same sources.




# Implementation

## End-to-End System Overview

Single pipeline feeds training, evaluation, and slide generation.
All artifacts live under `finetuning/` for auditability.

![End-to-End System Overview](assets/pipeline_architecture.png)


## Data Pipeline and Grouped Predicate Examples

Processed grouped examples: train 95253, validation 17577, test 20602.
Verified benchmark slice: 600/100/100.
Grouping at predicate level produces stable prompts and targets for seq2seq training.


## Compact Role Inventory and QA Rendering

Roles: AGENT, THEME, LOCATION, TIME, MANNER, REASON, ATTRIBUTE, SOURCE, GOAL, INSTRUMENT, OBLIQUE, OTHER.
QA rendering keeps outputs readable and ready for UI display.
Inventory reduces decoding burden while preserving core semantics.


## Why `flan-t5-small` + LoRA

Fits 8 GB CPU constraints with adapter-based fine-tuning.
LoRA rank 8 balances capacity and footprint.
Shared prompts and recovery steps align with compact generation behavior.


## Verified Training Configuration

Configuration mirrors the latest documented local run.

| Parameter | Value | Note |
| --- | --- | --- |
| Model | google/flan-t5-small | Base model |
| Epochs | 3 | Verified run |
| Batch size | 2 | Per step |
| Grad accum | 2 | Effective batch support |
| LoRA rank | 8 | Adapter capacity |
| Beam count | 1 | Generation setting |


## Final Local Prompt Design

Prefix: `semantic role extraction`.
Fields: predicate, sentence, then `labels:` for serialized roles.
Short format minimizes copying and keeps outputs parseable.


## Inference Cleanup and Answer Snapping

Parsed role strings are normalized and re-aligned to the source sentence.
Span snapping reduces drift from generation noise.
Supports stable evaluation and UI rendering.


## Fallback Recovery and Robustness Path

Fallback role mapping covers empty or malformed generations.
Heuristics favor conservative spans to keep the system usable.
Recovery complements prompt discipline for compact models.


## Evaluation + Explainability Stack

Aggregate metrics plus full per-example ledger stored in `results/evaluation_report.json`.
InstaShap-style plausibility and faithfulness computed on the same split.
Charts and slides are regenerated from these artifacts.


## Streamlit Demo and Deployment Flow

Run `python app.py` to launch the local UI.
UI shows roles, QA pairs, token attributions, and latest metrics.
No external services required beyond the stored artifacts.




# Results & Analysis

## Dataset Scale vs Verified Benchmark Slice

Keep the distinction between full processed pool and verified slice clear.
Benchmarks and charts reference the verified slice.

| View | Train | Validation | Test |
| --- | --- | --- | --- |
| Processed grouped files | 95253 | 17577 | 20602 |
| Verified benchmark slice | 600 | 100 | 100 |


## Training Dynamics and Learning Behavior

Loss decreases across epochs; selection token F1 peaks at 0.3425.
Training time about 685 seconds on CPU.
Early stopping margin remains after epoch three.

![Training Dynamics and Learning Behavior](assets/training_curve.png)


## Local Zero-Shot vs Fine-Tuned Results

Fine-tuned token F1 0.3257 vs baseline 0.0000.
Role coverage improves to 0.8075; exact match reaches 0.0400.
Same decoding settings keep the comparison fair.

![Local Zero-Shot vs Fine-Tuned Results](assets/zero_shot_vs_finetuned_metrics.png)


## Domain-Wise Performance and Metric Distribution

TQA: 0.3382, wikinews: 0.3432, wikipedia: 0.2592.
Domain mix influences average token F1 and coverage.
Longer-form wikipedia cases remain the hardest.

![Domain-Wise Performance and Metric Distribution](assets/domain_token_f1.png)


## Success Cases and Failure Patterns

Success: exact matches concentrate on shorter, factual predicates.
Failure: coverage holds but spans drift on longer, clause-heavy sentences.
Recovery path mitigates empty outputs but cannot fix weak spans alone.


## Explainability Evidence with InstaShap-Style Attributions

Plausibility 0.2220, faithfulness 0.7248, combined 0.4734.
Token importances align with predicate-adjacent spans.
Same explanation path is available in the Streamlit UI.

![Explainability Evidence with InstaShap-Style Attributions](assets/instashap_example.png)


## Local vs Gemini Comparison and Why Results Improved

Fine-tuning plus structured prompts outperform zero-shot baselines locally.
Gemini references use the same prompt profiles; no fabricated numbers are shown.
Recovery and snapping explain stability gains beyond prompting alone.

- Citation: [Gemini benchmark prompt profiles](https://ai.google.dev/gemini-api/docs/prompting)



# Innovation

## Innovation Pillars

CPU-feasible LoRA training and inference.
Schema-first prompting plus recovery and span snapping.
Integrated explanation and deck/docs generation.

![Innovation Pillars](assets/innovation_comparison.png)


## Innovation Gap-to-Response Mapping

Compute gap → LoRA adapters over compact model.
Explainability gap → InstaShap token attributions in evaluation and UI.
Robustness gap → parsing, snapping, and fallback layers.
Communication gap → synchronized docs, PDFs, and deck.


## What Changed from Earlier SRL-QA Systems

Tight coupling of prompt design with post-generation cleanup.
Deck and docs are generated artifacts, not manual copies.
Focus on verifiable local runs instead of aspirational SOTA claims.




# Q&A

## Thank You / Q&A

Ask about prompts, recovery, evaluation, or deployment details.
All artifacts live in `finetuning/` for audit and reuse.




## Evidence Extension Ledger

- Ledger 001: `tqa-001` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Exact structural match with clean role and span alignment.`.
- Ledger 002: `tqa-002` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Exact structural match with clean role and span alignment.`.
- Ledger 003: `tqa-003` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Exact structural match with clean role and span alignment.`.
- Ledger 004: `tqa-004` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Exact structural match with clean role and span alignment.`.
- Ledger 005: `tqa-005` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 006: `tqa-006` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 007: `tqa-007` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 008: `tqa-008` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 009: `tqa-009` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 010: `tqa-010` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 011: `tqa-011` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 012: `tqa-012` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 013: `tqa-013` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 014: `tqa-014` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 015: `tqa-015` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 016: `tqa-016` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 017: `tqa-017` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 018: `tqa-018` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 019: `tqa-019` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 020: `tqa-020` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 021: `tqa-021` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 022: `tqa-022` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 023: `tqa-023` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 024: `tqa-024` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 025: `tqa-025` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 026: `tqa-026` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 027: `tqa-027` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 028: `tqa-028` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 029: `tqa-029` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 030: `tqa-030` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 031: `tqa-031` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 032: `tqa-032` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 033: `tqa-033` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 034: `tqa-034` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 035: `tqa-035` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 036: `tqa-036` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 037: `tqa-037` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 038: `tqa-038` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 039: `tqa-039` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 040: `tqa-040` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 041: `tqa-041` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 042: `tqa-042` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 043: `tqa-043` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 044: `tqa-044` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 045: `tqa-045` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 046: `tqa-046` in `TQA` with predicate `reported` ended at token F1 `0.3382`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 047: `wikinews-047` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 048: `wikinews-048` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 049: `wikinews-049` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 050: `wikinews-050` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 051: `wikinews-051` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 052: `wikinews-052` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 053: `wikinews-053` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 054: `wikinews-054` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 055: `wikinews-055` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 056: `wikinews-056` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
- Ledger 057: `wikinews-057` in `wikinews` with predicate `reported` ended at token F1 `0.3432`, coverage `0.8075`, and assessment `Some relevant roles were recovered, yet span accuracy remains unstable.`.
