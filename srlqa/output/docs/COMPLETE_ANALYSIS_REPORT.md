# Complete End-to-End Analysis Report: RAISE-SRL-QA

Generated on: 2026-04-07T22:27:01

## Abstract

This report analyzes the local `srlqa` project using only files available on disk. The pipeline traversed the project, loaded Hugging Face Arrow datasets, parsed local JSON artifacts, inspected model files, ran available model families on the seed challenge suite, and regenerated tables, plots, documentation, and a PPT under `output/`.

The run found 240 source/inventory files and loaded 123099 rows across dataset artifacts. On `challenge_suite_v2`, the best local measured variant was `RAISE-SRL-QA Fast` with accuracy/exact match `1.0000` and token F1 `1.0000`. This is a seed-suite result, not an official benchmark claim.

## 1. Introduction

Semantic Role Labeling Question Answering asks natural-language questions about predicate-argument structure and returns extractive answer spans. The local project implements RAISE-SRL-QA: Retrieval-Augmented, Iteratively Self-correcting, Explainable Semantic Role Labeling Question Answering.

## 2. Literature Survey

The codebase combines QA-SRL style question-answer annotations, PropBank-style semantic role inventories, and transformer machine reading comprehension. No external leaderboard values are reported because the requirement is to avoid assumed numbers; every numeric table below comes from local files or local model execution.

## 3. Project Understanding

Inventory summary:

| category | files |
| --- | --- |
| plot_or_image | 95 |
| script | 65 |
| dataset | 23 |
| model | 22 |
| other | 21 |
| documentation | 5 |
| result_or_config | 4 |
| presentation | 3 |
| log | 2 |

Identified artifact types include dataset files under `.hf_cache/datasets` and `data/`, model files under `.hf_cache/models`, Python scripts under `srlqa/` and root scripts, documentation in Markdown, plots/images, result/config JSON files, and existing PPT decks.

Pipeline flow: input context and question -> question type and role inference -> predicate inference -> PropBank retrieval -> heuristic and optional transformer QA candidates -> constrained span selection -> verifier scoring -> self-correction -> final answer span and role.

## 4. Dataset EDA

Dataset overview:

| dataset_name | split | rows | columns | file | size_bytes | builder | version |
| --- | --- | --- | --- | --- | --- | --- | --- |
| challenge_suite_v2 | seed | 15 | 7 | data\challenge_suite_v2.json | 4638 | json |  |
| propbank_srl_seq2seq | train | 37057 | 2 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... | 6625464 | csv | 0.0.0 |
| propbank_srl_seq2seq | validation | 9264 | 2 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... | 1666840 | csv | 0.0.0 |
| qa_srl_promptsource | test | 15407 | 10 | .hf_cache\datasets\marcov___qa_srl_promptsource\default\0.0.0\46fcd4605989a61447... | 12214976 | parquet | 0.0.0 |
| qa_srl_promptsource | train | 44898 | 10 | .hf_cache\datasets\marcov___qa_srl_promptsource\default\0.0.0\46fcd4605989a61447... | 35255688 | parquet | 0.0.0 |
| qa_srl_promptsource | validation | 15281 | 10 | .hf_cache\datasets\marcov___qa_srl_promptsource\default\0.0.0\46fcd4605989a61447... | 12115776 | parquet | 0.0.0 |
| task1519_qa_srl_question_generation | test | 35 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... | 76888 | parquet | 0.0.0 |
| task1519_qa_srl_question_generation | train | 274 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... | 598880 | parquet | 0.0.0 |
| task1519_qa_srl_question_generation | valid | 34 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... | 75120 | parquet | 0.0.0 |
| task1520_qa_srl_answer_generation | test | 84 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1520_qa_srl_answer_generation\default\0.0... | 161208 | parquet | 0.0.0 |
| task1520_qa_srl_answer_generation | train | 667 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1520_qa_srl_answer_generation\default\0.0... | 1278968 | parquet | 0.0.0 |
| task1520_qa_srl_answer_generation | valid | 83 | 3 | .hf_cache\datasets\Lots-of-LoRAs___task1520_qa_srl_answer_generation\default\0.0... | 160240 | parquet | 0.0.0 |

Top missing-value fields:

| dataset_name | split | column | feature | observed_type | missing_count | missing_rate | file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| propbank_srl_seq2seq | train | prompt | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... |
| propbank_srl_seq2seq | train | response | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... |
| propbank_srl_seq2seq | validation | prompt | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... |
| propbank_srl_seq2seq | validation | response | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\cu-kairos___propbank_srl_seq2seq\default\0.0.0\3ac8d9eb632afd... |
| task1519_qa_srl_question_generation | test | input | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | test | output | List(Value('string')) | list | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | test | id | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | train | input | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | train | output | List(Value('string')) | list | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | train | id | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | valid | input | Value('string') | str | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |
| task1519_qa_srl_question_generation | valid | output | List(Value('string')) | list | 0 | 0.0000 | .hf_cache\datasets\Lots-of-LoRAs___task1519_qa_srl_question_generation\default\0... |

Question type distribution:

| question_type | count |
| --- | --- |
| UNKNOWN | 46321 |
| WHAT | 38159 |
| WHO | 18125 |
| WHEN | 7715 |
| WHERE | 6108 |
| HOW | 3281 |
| WHY | 2668 |
| HOW MUCH | 720 |
| TO-WHOM | 2 |

Semantic role distribution is computed from artifacts with explicit or parsed role labels: PropBank seq2seq responses, the frame store, the local challenge suite, and linked baseline role statistics. QA-SRL promptsource rows do not expose gold role labels in the cached schema, so inferred roles are saved separately.

| role | count |
| --- | --- |
| ARG1 | 32502 |
| ARG0 | 23990 |
| ARGM | 13779 |
| ARG2 | 10631 |
| ARG3 | 1859 |
| ARGM-TMP | 1597 |
| ARGM-MOD | 873 |
| ARGM-ADV | 859 |
| ARG4 | 636 |
| ARGM-MNR | 636 |
| ARGM-LOC | 612 |
| ARGM-DIS | 453 |
| ARGM-NEG | 318 |
| ARG2-TO | 236 |
| ARGM-PNC | 215 |

## 5. Methodology

The analysis uses `datasets.Dataset.from_file` for Arrow files, deterministic token normalization for exact match and token F1, a local smoothed sentence-BLEU implementation for text overlap, `safetensors` metadata for parameter counts, and `python-pptx` for the presentation.

## 6. Experiments

The seed-suite evaluation used 15 examples from `data/challenge_suite_v2.json`.

| model_key | model_label | examples | ok_count | error_count | accuracy | exact_match | token_precision | token_recall | token_f1 | bleu | role_accuracy | confidence_mean | latency_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_baseline | Legacy PropQA-Net Baseline | 15 | 15 | 0 | 0.2000 | 0.2000 | 0.5381 | 0.8689 | 0.5522 | 0.4046 | 0.3333 | 0.4777 | 207.5450 |
| legacy_hybrid | Legacy Hybrid | 15 | 15 | 0 | 0.6667 | 0.6667 | 0.8690 | 0.8800 | 0.8230 | 0.7734 | 0.9333 | 0.6531 | 495.8688 |
| raise_srlqa_fast | RAISE-SRL-QA Fast | 15 | 15 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9560 | 7.0815 |
| raise_srlqa_model | RAISE-SRL-QA Model | 15 | 15 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9571 | 1294.6330 |

Linked baseline metrics are a separate scope because they come from `srl_qa_project/results/metrics.json`, not from the seed suite:

| metric | value | source | scope |
| --- | --- | --- | --- |
| qa_exact_match | 0.5184 | linked baseline metrics.json | baseline test |
| qa_token_f1 | 0.7612 | linked baseline metrics.json | baseline test |
| srl_micro_precision | 0.7320 | linked baseline metrics.json | baseline test |
| srl_micro_recall | 0.6955 | linked baseline metrics.json | baseline test |
| srl_micro_f1 | 0.7133 | linked baseline metrics.json | baseline test |
| srl_macro_f1 | 0.1619 | linked baseline metrics.json | baseline test |
| srl_bio_accuracy | 0.8163 | linked baseline metrics.json | baseline test |
| best_epoch | 6.0000 | linked baseline metrics.json | baseline training |
| best_validation_f1 | 0.7727 | linked baseline metrics.json | baseline training |
| parameter_count | 1784352.0000 | linked baseline metrics.json | baseline training |
| total_propbank_instances | 112917.0000 | linked baseline data_statistics.json | baseline corpus |
| usable_propbank_instances | 9073.0000 | linked baseline data_statistics.json | baseline corpus |
| qa_pair_count | 23007.0000 | linked baseline data_statistics.json | baseline corpus |
| unique_predicates | 1340.0000 | linked baseline data_statistics.json | baseline corpus |
| unique_rolesets | 1670.0000 | linked baseline data_statistics.json | baseline corpus |

## 7. Results

Generated figure references include:

- `output/plots/dataset_rows_by_split.png`
- `output/plots/token_length_histogram.png`
- `output/plots/question_type_distribution.png`
- `output/plots/semantic_role_distribution.png`
- `output/plots/model_accuracy_comparison.png`
- `output/plots/model_f1_comparison.png`
- `output/plots/role_confusion_matrix.png`
- `output/plots/latency_comparison.png`

Linked baseline QA EM: `0.5184`. Linked baseline QA token F1: `0.7612`. Linked baseline SRL micro F1: `0.7133`. Linked baseline SRL macro F1: `0.1619`.

## 8. Error Analysis

Seed-suite error rows are saved to `output/tables/error_cases.csv`. Linked baseline taxonomy:

| error_category | count |
| --- | --- |
| correct | 1789 |
| span boundary error | 866 |
| other | 109 |
| wrong role | 685 |
| predicate miss | 2 |

Highest linked baseline role error rates:

| role | errors | total | rate |
| --- | --- | --- | --- |
| ARG2-for | 8 | 8 | 1.0000 |
| ARG1-about | 6 | 6 | 1.0000 |
| ARG2-PNC | 4 | 4 | 1.0000 |
| ARG3-at | 4 | 4 | 1.0000 |
| ARG1-of | 3 | 3 | 1.0000 |
| ARG1-by | 2 | 2 | 1.0000 |
| ARG2-against | 2 | 2 | 1.0000 |
| ARG2-between | 2 | 2 | 1.0000 |
| ARG3-on | 2 | 2 | 1.0000 |
| ARG4-at | 2 | 2 | 1.0000 |
| ARG5-DIR | 2 | 2 | 1.0000 |
| ARG0-in | 1 | 1 | 1.0000 |

## 9. Ablation and Innovation Analysis

No controlled ablation log file was found for every individual innovation. The table below is therefore an available-variant comparison on the same local seed suite.

| variant | innovation_or_model_change | dataset_scope | accuracy | token_f1 | role_accuracy | delta_f1_vs_previous_available_variant | delta_f1_vs_legacy_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_baseline | Legacy checkpoint baseline | challenge_suite_v2 seed suite | 0.2000 | 0.5522 | 0.3333 |  | 0.0000 |
| legacy_hybrid | Legacy hybrid additions | challenge_suite_v2 seed suite | 0.6667 | 0.8230 | 0.9333 | 0.2707 | 0.2707 |
| raise_srlqa_fast | RAISE retrieval + verifier + deterministic rules | challenge_suite_v2 seed suite | 1.0000 | 1.0000 | 1.0000 | 0.1770 | 0.4478 |
| raise_srlqa_model | RAISE plus transformer QA candidates | challenge_suite_v2 seed suite | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.4478 |

Innovation evidence:

| innovation | evidence_file | file_exists | file_size_bytes | seed_suite_fast_f1_context | seed_suite_model_f1_context | controlled_ablation_log_found |
| --- | --- | --- | --- | --- | --- | --- |
| PropBank frame retrieval | srlqa/retrieval/propbank_index.py | True | 4018 | 1.0000 | 1.0000 | False |
| Constrained span decoding | srlqa/decoding/span_rules.py | True | 2508 | 1.0000 | 1.0000 | False |
| Role priors | srlqa/decoding/role_priors.py | True | 517 | 1.0000 | 1.0000 | False |
| Evidence span verifier | srlqa/verification/span_verifier.py | True | 2231 | 1.0000 | 1.0000 | False |
| Self-correction loop | srlqa/verification/self_correction.py | True | 1053 | 1.0000 | 1.0000 | False |
| Hard negative mining | srlqa/training/hard_negative_mining.py | True | 1875 | 1.0000 | 1.0000 | False |
| Distillation scaffolding | srlqa/distillation/teacher_runner.py | True | 932 | 1.0000 | 1.0000 | False |
| Calibrated ensemble | srlqa/ensemble/weighted_voter.py | True | 1218 | 1.0000 | 1.0000 | False |
| Nominal event templates | srlqa/nominal/qa_noun_templates.py | True | 540 | 1.0000 | 1.0000 | False |
| Proto-role features | srlqa/proto_roles/proto_role_features.py | True | 639 | 1.0000 | 1.0000 | False |

## 10. Complexity Analysis

Complexity values are computed from local artifacts: cached model file sizes, safetensors parameter counts, measured seed-suite latency, and Python source line/function/class counts.

| artifact | kind | size_bytes | size_mb | parameter_count | tensor_count | model_key | latency_ms_mean | latency_ms_median | latency_ms_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\refs\main | file | 40.0000 | 0.0000 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | json | 23.0000 | 0.0000 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | json | 992.0000 | 0.0009 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | safetensors | 735360940.0000 | 701.2948 | 183833602.0000 | 201.0000 |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | json | 173.0000 | 0.0002 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | model | 2464616.0000 | 2.3504 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | json | 8648791.0000 | 8.2481 |  |  |  |  |  |  |
| .hf_cache\models\models--deepset--deberta-v3-base-squad2\snapshots\eea39c60cc305... | json | 379.0000 | 0.0004 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\refs\main | file | 40.0000 | 0.0000 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\refs\refs\pr\14 | file | 40.0000 | 0.0000 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\snapshots\8ccc9b6f36199bec69... | json | 579.0000 | 0.0006 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\snapshots\8ccc9b6f36199bec69... | bin | 371146213.0000 | 353.9526 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\snapshots\8ccc9b6f36199bec69... | model | 2464616.0000 | 2.3504 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\snapshots\8ccc9b6f36199bec69... | json | 52.0000 | 0.0000 |  |  |  |  |  |  |
| .hf_cache\models\models--microsoft--deberta-v3-base\snapshots\de19fe7db5162df5f3... | safetensors | 371101258.0000 | 353.9097 | 185537893.0000 | 210.0000 |  |  |  |  |
| measured runtime variant | runtime |  |  |  |  | legacy_baseline | 207.5450 | 160.7948 | 397.3875 |
| measured runtime variant | runtime |  |  |  |  | legacy_hybrid | 495.8688 | 402.3419 | 1005.4955 |
| measured runtime variant | runtime |  |  |  |  | raise_srlqa_fast | 7.0815 | 2.0871 | 26.7820 |
| measured runtime variant | runtime |  |  |  |  | raise_srlqa_model | 1294.6330 | 362.5436 | 5053.0153 |
|  |  |  |  |  |  |  |  |  |  |

## 11. Comparison with Modern QA Systems

The workspace contains cached DeBERTa artifacts (`microsoft/deberta-v3-base` and `deepset/deberta-v3-base-squad2`). External transformer or LLM benchmark scores are not invented here. Future comparison should run those systems through the same `output/code/evaluation_pipeline.py` metrics.

## 12. Conclusion

The project is a reproducible SRL-QA scaffold with retrieval, verification, self-correction, and model-hub components. The generated outputs support local seed-suite comparison and linked-baseline analysis while preserving claim boundaries.

## 13. Future Work

- Add controlled ablations for retrieval, span rules, verification, self-correction, and transformer candidates.
- Freeze a larger held-out benchmark and save prediction JSONL for every model.
- Add calibration plots and reliability analysis.
- Evaluate modern transformer and LLM baselines with the same local metrics.
