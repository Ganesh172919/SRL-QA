# Complete Project Guide

## 1. Purpose of This Guide

This document is a repo-wide handbook for the `srl_qa_project` codebase.

It is written to explain the project end to end.

It focuses on how the repository is organized.

It explains what every major file and directory does.

It connects the source code to the cached data, metrics, plots, checkpoints, and generated PDFs.

It uses examples drawn from the actual repository state.

It includes ASCII diagrams so the project flow is easy to visualize.

It is intentionally detailed.

The audience for this guide includes:

- a student submitting the project,
- a teammate trying to understand the implementation,
- a reviewer checking whether the repository is complete,
- a future maintainer extending the code,
- and a reader who wants to connect SRL theory to the implementation.

This guide covers all major authored files in the repository.

It also covers the important generated artifacts and the packaged corpus assets.

For the bundled `nltk_data/` directory, the guide explains the file families and representative files rather than pretending each vendored corpus file needs a separate prose essay.

That is the practical interpretation used here for “use all files in the repo.”

The guide is grounded in the actual current repository contents.

Those contents include:

- source modules,
- local documentation files,
- cached dataset splits,
- a trained checkpoint,
- evaluation outputs,
- generated plots,
- PDF deliverables,
- and packaged PropBank and Treebank resources.

## 2. Project in One Sentence

This project builds a classical, fully reproducible semantic-role-based extractive question answering system called PropQA-Net using local PropBank and Treebank resources.

## 3. Project in One Paragraph

The project starts from real PropBank predicate-argument annotations distributed through NLTK.

It aligns those annotations with locally available Penn Treebank parse trees.

It reconstructs sentence tokens and answer spans.

It converts semantic arguments into question-answer training examples.

It trains a joint SRL and QA model.

It evaluates that model using both role metrics and answer metrics.

It runs an inference demo.

It supports custom question asking.

It then generates several academic PDF deliverables and a single-file implementation bundle.

## 4. High-Level Repository Layout

```text
srl_qa_project/
|
|-- config.py
|-- data_loader.py
|-- model.py
|-- trainer.py
|-- evaluator.py
|-- qa_inference.py
|-- pdf_generator.py
|-- main.py
|-- README.md
|-- requirements.txt
|
|-- checkpoints/
|   `-- best_model.pt
|
|-- data/
|   |-- train.json
|   |-- val.json
|   `-- test.json
|
|-- docs/
|   |-- OVERVIEW.md
|   |-- DATA.md
|   |-- MODEL.md
|   |-- EVALUATION.md
|   |-- PDF_DELIVERABLES.md
|   |-- TROUBLESHOOTING.md
|   `-- COMPLETE_PROJECT_GUIDE.md
|
|-- nltk_data/
|   `-- corpora/
|       |-- propbank/
|       `-- treebank/
|
|-- outputs/
|   |-- survey.pdf
|   |-- analysis.pdf
|   |-- innovation.pdf
|   |-- research_paper.pdf
|   `-- implementation_code.py
|
`-- results/
    |-- data_statistics.json
    |-- metrics.json
    |-- inference_demo.json
    `-- plots/
        |-- loss_curve.png
        |-- confusion_matrix.png
        |-- qa_accuracy_by_qtype.png
        |-- error_taxonomy.png
        |-- ...
```

## 5. Repository Roles by Directory

### 5.1 Root Python modules

These files implement the runnable pipeline.

They are the core of the repository.

### 5.2 `data/`

This directory stores cached JSON examples after preprocessing.

It allows later runs to skip expensive corpus reconstruction.

### 5.3 `checkpoints/`

This directory stores the trained model checkpoint.

The checkpoint is later reused for evaluation, demo inference, and custom question asking.

### 5.4 `results/`

This directory stores derived analytic outputs.

These include descriptive statistics, performance metrics, prediction samples, and plots.

### 5.5 `outputs/`

This directory stores the required final deliverables.

These are the academic PDFs plus the concatenated implementation bundle.

### 5.6 `docs/`

This directory stores human-oriented project explanations.

The current document is the most exhaustive one.

### 5.7 `nltk_data/`

This directory stores local corpus assets.

The project uses this directory so it can run offline and reproducibly.

## 6. What Problem the Project Solves

Traditional extractive QA systems answer questions from text.

Many such systems do not explicitly model event structure.

Semantic role labeling, or SRL, does model event structure.

SRL asks questions like:

- who did the action,
- what was affected,
- where it happened,
- when it happened,
- why it happened,
- and how it happened.

This project combines these two ideas.

It turns PropBank semantic role annotations into QA training examples.

Then it trains a model to answer those questions by predicting spans from the sentence.

The result is a QA system that is grounded in explicit predicate-argument structure.

## 7. Why PropBank Matters Here

PropBank provides labeled predicate-argument structures.

Those labels are tied to predicates such as `agree.01`, `find.01`, or `make.03`.

Each predicate instance in a sentence may have:

- a predicate pointer,
- numbered arguments like `ARG0`, `ARG1`, `ARG2`,
- adjuncts like `ARGM-TMP`, `ARGM-LOC`, `ARGM-MNR`,
- and roleset metadata stored in XML frame files.

This project uses PropBank as the semantic source of truth.

It does not invent roles from scratch.

It reads the roles and maps them to natural-language questions.

## 8. Why Treebank Matters Here

PropBank annotations alone do not automatically provide simple token spans usable for extractive QA.

The project needs exact sentence tokens.

It also needs precise token boundaries for answer spans.

The Penn Treebank provides parse trees and terminal positions.

Those parse trees make it possible to reconstruct visible sentence tokens and align PropBank pointers to token indices.

Without this alignment step, the QA supervision would be much less reliable.

## 9. The Core Design Decision

The most important design decision in this repository is:

Use only PropBank instances that can be aligned to locally available Treebank sentences.

That decision reduces total available data.

But it dramatically improves reproducibility and span correctness.

The numbers in the current repository make this tradeoff clear:

- total PropBank instances visible through NLTK: `112,917`,
- usable Treebank-backed instances: `9,073`,
- resulting QA pairs: `23,007`.

So the project chooses accuracy and determinism over raw corpus size.

## 10. End-to-End Pipeline at a Glance

```text
            +----------------------+
            |  main.py CLI runner  |
            +----------+-----------+
                       |
                       v
          +---------------------------+
          | configure_runtime(config) |
          +-------------+-------------+
                        |
                        v
            +------------------------+
            | prepare_data(config)   |
            +-----------+------------+
                        |
                        v
     +------------------------------------------+
     | load_or_build_splits(config)             |
     | - read cached JSON OR rebuild from NLTK  |
     +-------------------+----------------------+
                         |
                         v
     +------------------------------------------+
     | build_dataloaders(...)                   |
     | - build vocabs                           |
     | - encode examples                        |
     | - create PyTorch DataLoaders             |
     +-------------------+----------------------+
                         |
                         v
       +----------------+-----------------------------+
       |                                              |
       v                                              v
+-------------+                                +--------------+
| train_model |                                | evaluate_model|
+------+------+                                +------+-------+
       |                                              |
       v                                              v
best_model.pt                                  metrics.json
       |                                              |
       +----------------------+-----------------------+
                              |
                              v
                      +---------------+
                      | qa_inference  |
                      +-------+-------+
                              |
                              v
                     inference_demo.json
                              |
                              v
                    +-------------------+
                    | pdf_generator.py  |
                    +--------+----------+
                             |
                             v
      survey.pdf / analysis.pdf / innovation.pdf / research_paper.pdf
                             |
                             v
                    implementation_code.py
```

## 11. Detailed Execution Modes

The CLI in `main.py` supports five modes.

### 11.1 `train`

This mode prepares data and trains the model.

It saves the best checkpoint.

### 11.2 `eval`

This mode prepares data and evaluates the saved checkpoint on the test set.

It produces `metrics.json` and plots.

### 11.3 `infer`

This mode runs the fixed 10-example inference demo.

It writes `results/inference_demo.json`.

### 11.4 `ask`

This mode runs the new custom question-asking feature.

It can be used in one-shot mode with `--context` and `--question`.

It can also be used in terminal-interactive mode with `--interactive`.

### 11.5 `full`

This mode does everything:

- data preparation,
- training,
- evaluation,
- demo inference,
- PDF generation,
- implementation bundle export,
- output validation.

## 12. Main Commands You Can Run

```bash
python main.py --mode train
python main.py --mode eval
python main.py --mode infer
python main.py --mode ask
python main.py --mode ask --interactive
python main.py --mode ask --context "The chef cooked a delicious meal in the kitchen yesterday." --question "Who cooked?"
python main.py --mode full
```

## 13. Quick Repository Facts from the Current Run

These values come from the actual repository state:

- train examples: `16,104`,
- validation examples: `3,451`,
- test examples: `3,452`,
- usable sentence instances: `9,073`,
- total QA pairs: `23,007`,
- unique predicates: `1,340`,
- unique rolesets: `1,670`.

Current QA metrics from `results/metrics.json`:

- exact match: `0.5184`,
- token F1: `0.7612`,
- mean answer-length deviation: `-1.0090`,
- mean absolute answer-length deviation: `1.8505`.

Current SRL metrics from `results/metrics.json`:

- macro precision: `0.2111`,
- macro recall: `0.1592`,
- macro F1: `0.1619`,
- micro precision: `0.7320`,
- micro recall: `0.6955`,
- micro F1: `0.7133`,
- BIO accuracy: `0.8163`.

Training diagnostics from the same file:

- best epoch: `6`,
- best validation F1: `0.7727`,
- trainable parameter count: `1,784,352`.

## 14. File-by-File Explanation

This section explains the important files individually.

## 15. `README.md`

`README.md` is the top-level entry document for the project.

It does the following:

- introduces PropQA-Net,
- explains the high-level pipeline,
- gives quickstart commands,
- lists repository layout,
- explains generated outputs,
- notes the Treebank-backed subset issue,
- points readers to the `docs/` directory,
- and now includes the custom `ask` mode.

Why this file matters:

- it is the first orientation layer,
- it communicates the project scope quickly,
- and it serves as the shortest successful onboarding path.

What it does not do:

- it does not explain every function,
- it does not contain deep implementation details,
- and it does not replace the more detailed docs.

## 16. `requirements.txt`

This file defines the Python dependencies.

The exact listed packages are:

- `nltk==3.9.2`,
- `numpy==2.2.6`,
- `pandas==2.2.3`,
- `matplotlib==3.10.7`,
- `reportlab==4.2.5`,
- `scikit-learn==1.7.2`,
- `torch==2.10.0+cpu`.

The file also includes an installation note for the CPU-only PyTorch build.

Why each dependency exists:

- `nltk` provides PropBank and Treebank access.
- `numpy` supports numerical statistics and plotting helpers.
- `pandas` is available for tabular work, though the code relies more heavily on Python dicts and lists than on DataFrames.
- `matplotlib` is used for plots and diagram assets.
- `reportlab` is used to build the PDFs.
- `scikit-learn` is used for classification metrics and the confusion matrix.
- `torch` is used for the model, tensors, and training.

Practical implication:

This project is intentionally lightweight by modern QA standards.

It avoids large external transformer stacks.

That makes it easier to reproduce in a controlled classroom or offline environment.

## 17. `config.py`

`config.py` centralizes all paths and hyperparameters.

It is one of the most important architectural simplifications in the repository.

Instead of scattering constants across files, the project gathers them in dataclasses.

The key dataclasses are:

- `PathConfig`,
- `DataConfig`,
- `ModelConfig`,
- `TrainingConfig`,
- `RuntimeConfig`,
- `ProjectConfig`.

### 17.1 `PathConfig`

`PathConfig` computes all major filesystem paths from the project root.

Important derived paths include:

- `data_dir`,
- `checkpoints_dir`,
- `results_dir`,
- `plots_dir`,
- `outputs_dir`,
- `nltk_data_dir`,
- `propbank_dir`,
- `treebank_dir`,
- `train_json`,
- `val_json`,
- `test_json`,
- `metrics_path`,
- `checkpoint_path`.

This means the rest of the code can refer to `config.paths.train_json` or `config.paths.checkpoint_path` instead of hand-building strings repeatedly.

### 17.2 `PathConfig.ensure_directories`

This method creates the required directories if they do not exist.

That is important because later stages expect writable directories.

It also makes the pipeline less fragile.

### 17.3 `DataConfig`

`DataConfig` contains preprocessing settings.

Its values include:

- random seed `42`,
- train ratio `0.70`,
- validation ratio `0.15`,
- test ratio `0.15`,
- lowercase token normalization,
- minimum token frequency,
- max sentence length `128`,
- max question length `32`,
- optional `max_instances`,
- optional `rebuild_cache`.

These fields influence both corpus generation and model input encoding.

### 17.4 `ModelConfig`

`ModelConfig` defines architecture hyperparameters.

Important values:

- word embedding dimension `100`,
- POS embedding dimension `32`,
- predicate embedding dimension `8`,
- hidden size `128`,
- question hidden size `128`,
- dropout `0.30`,
- multi-task interpolation `alpha = 0.50`.

### 17.5 `TrainingConfig`

`TrainingConfig` defines training choices.

Important values:

- batch size `64`,
- learning rate `1e-3`,
- weight decay `1e-5`,
- max epochs `6`,
- patience `5`,
- gradient clip norm `5.0`,
- data loader workers `0`.

### 17.6 `RuntimeConfig`

`RuntimeConfig` currently stores:

- device,
- verbose mode.

This keeps runtime concerns separate from training and data concerns.

### 17.7 `ProjectConfig`

`ProjectConfig` nests all sub-configs together.

It provides `to_dict()` so config values can be serialized into checkpoints and logs.

### 17.8 `get_config()`

`get_config()` is the public entry point for the rest of the codebase.

It instantiates the full project config and ensures the directories exist.

Why this module matters:

- it provides a single source of truth,
- it makes experimentation easier,
- it reduces duplicated path logic,
- and it makes checkpoint metadata cleaner.

## 18. `data_loader.py`

`data_loader.py` is the largest and most structurally important preprocessing module.

It converts raw corpus resources into training-ready SRL-QA examples.

It also builds vocabularies and PyTorch datasets.

If `model.py` is the heart of the network, `data_loader.py` is the heart of the data pipeline.

### 18.1 Main responsibilities

This module does all of the following:

- registers local NLTK paths,
- reconstructs visible sentence tokens,
- resolves PropBank pointers to token indices,
- assigns BIO labels,
- generates role-based questions,
- builds example dictionaries,
- computes descriptive statistics,
- caches dataset splits,
- builds vocabularies,
- creates dataset objects,
- pads batches,
- returns DataLoaders.

### 18.2 Important constants

The module defines several helper lexicons:

- `MONTH_WORDS`,
- `DAY_WORDS`,
- `LOCATION_PREPOSITIONS`,
- `ROLE_TO_QTYPE`.

These are small but important.

They support heuristic named-entity labels, question-type mapping, and dependency-like labels.

### 18.3 `Vocabulary`

The `Vocabulary` dataclass stores:

- `token_to_id`,
- `id_to_token`.

Its methods include:

- `build`,
- `encode`,
- `to_dict`.

This is a deliberately simple vocabulary implementation.

It is enough for a classical token-based pipeline.

It avoids external tokenizer dependencies.

### 18.4 `SRLQADataset`

This class converts JSON-like examples into encoded examples suitable for PyTorch.

It:

- lowercases tokens if configured,
- truncates overlong questions,
- skips sentences or answers exceeding the configured maximum sentence length,
- stores encoded IDs,
- keeps the raw example for downstream evaluation and debugging.

This dual representation is very useful.

The tensors support model training.

The raw example supports interpretability later.

### 18.5 `initialize_nltk`

This function prepends the local `nltk_data/` path to `nltk.data.path`.

That one line is central to offline reproducibility.

Without it, the pipeline might search the user’s global NLTK installation instead of the packaged local corpora.

### 18.6 Tokenization helpers

The module provides:

- `normalize_token`,
- `simple_word_tokenize`,
- `detokenize`.

The tokenizer is regex based.

The detokenizer uses `TreebankWordDetokenizer`.

That pairing keeps preprocessing consistent enough for a classical baseline.

### 18.7 Span and label helpers

The module includes:

- `strip_bio_prefix`,
- `split_contiguous`,
- `assign_bio_labels`.

These are foundational utilities.

They allow the code to move between:

- raw token indices,
- contiguous spans,
- and BIO tag sequences.

### 18.8 Treebank visibility reconstruction

A subtle but critical function is `build_visible_token_view`.

Treebank parse trees may contain invisible or empty constituents.

The function:

- walks tree leaves,
- records leaf positions,
- drops `-NONE-` nodes,
- builds a mapping from original leaf indices to visible token indices,
- and collects aligned POS tags.

This is the bridge from tree-internal structure to real sentence tokens.

### 18.9 PropBank pointer resolution

The functions:

- `collect_original_leaf_indices`,
- `flatten_pointer_pieces`,
- `visible_indices_for_pointer`,

resolve PropBank pointers into token indices.

This is where the project earns its “real PropBank grounded” claim.

It does not use invented answer spans.

It reconstructs them from annotated pointers and parse trees.

### 18.10 Heuristic named entities

`heuristic_named_entities` creates lightweight NE-like tags without using a separate NER model.

These tags are not currently core model inputs.

But they are stored in the example payload.

That makes them available for:

- analysis,
- debugging,
- future ablations,
- and richer documentation.

### 18.11 Heuristic dependency labels

`heuristic_dependency_labels` creates coarse dependency-like labels.

It uses:

- predicate flags,
- SRL labels,
- POS tags,
- token surface forms.

For example:

- predicate token becomes `root`,
- `ARG0` often becomes `nsubj`,
- `ARG1` often becomes `obj`,
- temporal adjuncts become `obl:tmod`,
- locative adjuncts become `obl:loc`.

Again, these are not central training inputs right now.

But they enrich the cached examples for later analysis and extension.

### 18.12 Question typing

`infer_question_type` maps a role and role description to a coarse question class.

Supported types include:

- `WHO`,
- `WHAT`,
- `WHEN`,
- `WHERE`,
- `WHY`,
- `HOW`.

This mapping is partly hard-coded and partly description-based.

That hybrid approach gives sensible defaults even for less common roles.

### 18.13 Question generation

`build_question` converts semantic roles into natural-language questions.

Examples of the logic:

- `ARG0` often becomes `Who ...?`
- `ARG1` often becomes `What did SUBJECT PREDICATE?`
- `ARGM-TMP` becomes `When did SUBJECT PREDICATE?`
- `ARGM-LOC` becomes `Where did SUBJECT PREDICATE?`
- `ARGM-MNR` becomes `How did SUBJECT PREDICATE?`
- `ARGM-CAU` becomes `Why did SUBJECT PREDICATE?`

This function is important because it defines the project’s transformation from semantic annotation to QA supervision.

### 18.14 Roleset metadata

`roleset_metadata` reads roleset XML metadata from PropBank.

It returns:

- `roleset_id`,
- `roleset_name`,
- `roleset_vncls`,
- `role_descriptions`.

This metadata enriches the generated QA examples.

It also supports question wording.

### 18.15 `inspect_corpus`

This function is a lightweight introspection tool.

It:

- counts PropBank instances,
- counts usable Treebank-backed instances,
- extracts a sample roleset,
- extracts a sample predicate pointer,
- returns sample arguments.

This is used in `run_data_statistics` so printed summaries are grounded in actual corpus metadata.

### 18.16 `build_examples_from_propbank`

This is the central data-construction function.

Its high-level steps are:

1. initialize NLTK,
2. import `propbank` and `treebank`,
3. iterate over PropBank instances,
4. skip any whose `fileid` is not locally available in Treebank,
5. reconstruct tokens and POS tags,
6. find predicate indices,
7. initialize predicate flags and empty SRL tags,
8. gather argument entries,
9. assign BIO labels,
10. choose a subject text from `ARG0` when possible,
11. create context text and heuristic features,
12. generate one QA example per contiguous argument,
13. collect corpus statistics while iterating.

This function is where the raw corpus becomes model-ready supervision.

### 18.17 Contiguity policy

The code drops non-contiguous arguments from final QA example creation.

This is an important modeling simplification.

It keeps extractive answer spans clean.

But it also means some true semantic arguments are excluded from QA supervision.

That is a defensible baseline choice.

It is also explicitly tracked through `dropped_noncontiguous_arguments`.

### 18.18 Example schema

Each built example stores many fields.

Important fields include:

- `example_id`,
- `instance_id`,
- `fileid`,
- `sentnum`,
- `context`,
- `context_tokens`,
- `question`,
- `question_tokens`,
- `answer_text`,
- `answer_tokens`,
- `answer_start`,
- `answer_end`,
- `answer_length`,
- `predicate_lemma`,
- `predicate_text`,
- `predicate_indices`,
- `predicate_flags`,
- `roleset_id`,
- `roleset_name`,
- `roleset_vncls`,
- `target_role`,
- `target_role_description`,
- `question_type`,
- `pos_tags`,
- `ne_tags`,
- `dependency_labels`,
- `srl_tags`,
- `argument_spans`.

That schema is intentionally rich.

It supports both modeling and reporting.

### 18.19 Statistics helpers

The module computes:

- sentence-length distributions,
- answer-length distributions,
- argument-type counts,
- question-type counts,
- split sizes,
- sample QA pairs.

These are eventually written to `results/data_statistics.json`.

### 18.20 Deterministic splitting

`split_examples` shuffles examples with `random.Random(config.data.random_seed)`.

That means split generation is deterministic for a given seed.

This is a key reproducibility feature.

### 18.21 JSON caching

The functions:

- `save_json`,
- `load_json`,
- `load_or_build_splits`,

support caching.

If `train.json`, `val.json`, and `test.json` already exist and `rebuild_cache` is `False`, the project reuses them.

That makes repeated development much faster.

### 18.22 Vocabulary building

`build_feature_vocabs` builds three vocabularies:

- token vocabulary,
- POS vocabulary,
- label vocabulary.

The token vocabulary is built from both context tokens and question tokens in the training split.

The POS vocabulary is built from training POS tags.

The label vocabulary is built from observed SRL BIO tags.

The label `O` is forced to the front.

That is useful because `O` acts like a natural background class.

### 18.23 `collate_batch`

This function pads variable-length examples into tensors.

It creates:

- `context_ids`,
- `pos_ids`,
- `predicate_flags`,
- `label_ids`,
- `context_mask`,
- `question_ids`,
- `question_mask`,
- `answer_starts`,
- `answer_ends`,
- `raw_examples`.

This separation between padded tensors and raw examples is one of the cleaner design choices in the repository.

### 18.24 `build_dataloaders`

This function creates:

- a training dataset and loader,
- a validation dataset and loader,
- a test dataset and loader,
- plus the vocabularies.

The train loader is shuffled.

The others are not.

### 18.25 `run_data_statistics`

This function:

- augments stats with split sizes and corpus overview,
- prints representative corpus information,
- prints sample QA pairs,
- saves `results/data_statistics.json`.

This means data understanding is part of the pipeline, not an afterthought.

## 19. Example from the Cached Data

A real training example from the repository has:

- `example_id`: `qa_010207`,
- `instance_id`: `wsj_0091.mrg:6:rel:agree.01`,
- target role: `ARGM-TMP`,
- question type: `WHEN`,
- question: `When did Western Union agree?`,
- answer text: `When Bell established that the Berliner patent caveat was registered 10 days before Edison's application`.

This example is a good illustration of the dataset’s character.

The answer can be fairly long.

The question is generated from role structure.

The semantics come from PropBank.

The surface span comes from Treebank alignment.

## 20. `model.py`

`model.py` contains the PropQA-Net architecture.

The code is compact.

But it packs several important modeling ideas into a relatively small network.

### 20.1 What the model predicts

The model jointly predicts:

- SRL BIO labels over the sentence,
- the start index of the answer span,
- the end index of the answer span.

### 20.2 Why multi-task learning is used

The intuition is simple:

If the model learns where semantic arguments live in the sentence, that should help it answer role-targeted questions.

Likewise, if it learns question-conditioned answer extraction, that should pressure the contextual representations to become useful for semantic slot selection.

### 20.3 ASCII architecture

```text
Context token ids ----> Word Embedding --------------------\
POS tag ids ----------> POS Embedding ----------------------+--> Concatenate --> BiLSTM --> Context states
Predicate flags ------> Predicate Embedding ---------------/
                                                                  |                     |
                                                                  |                     |
                                                                  v                     v
                                                           SRL BIO classifier      Interaction builder
                                                                                         ^
                                                                                         |
Question ids ------> Shared Word Embedding --> Question BiLSTM --> Mean pooling --> Question vector
                                                                                         |
                                                                                         v
                                                                     Start scorer / End scorer
                                                                                         |
                                                                                         v
                                                                   Candidate span selection
                                                                                         |
                                                                                         v
                                                                        Final answer span + role
```

### 20.4 Helper functions

The module defines:

- `strip_bio_prefix`,
- `masked_mean_pooling`,
- `decode_bio_spans`,
- `majority_role`.

These functions support decoding and scoring.

### 20.5 `PredictionResult`

This small dataclass stores:

- `start`,
- `end`,
- `role`,
- `confidence`,
- `decoded_labels`.

This structure is returned by prediction-time decoding.

### 20.6 Embeddings

The model uses three context-side embeddings:

- word embedding,
- POS embedding,
- predicate-flag embedding.

The question side uses the same word embedding table as the context.

This shared lexical space is a good classical design choice.

It encourages context and question tokens to live in the same embedding geometry.

### 20.7 Encoders

The model uses:

- one BiLSTM for the context,
- one BiLSTM for the question.

The context encoder consumes concatenated word, POS, and predicate embeddings.

The question encoder consumes only word embeddings.

### 20.8 `_encode_lstm`

This helper wraps packed-sequence processing:

- compute lengths from the mask,
- pack padded sequences,
- run the LSTM,
- pad outputs back to the full batch shape.

This lets the network handle variable lengths correctly without wasting as much computation on pad tokens.

### 20.9 `encode_context`

This function:

- embeds the three context features,
- concatenates them,
- runs the context BiLSTM,
- applies dropout.

### 20.10 `encode_question`

This function:

- embeds question tokens,
- runs the question BiLSTM,
- mean-pools the contextual outputs using the question mask.

The output is a fixed-size question vector.

### 20.11 Interaction features

Once the projected question vector is available, the model builds an interaction tensor from:

- context states,
- question vector repeated across tokens,
- elementwise product,
- absolute difference.

This is a classical matching strategy.

It gives the start and end heads richer token-question interaction features than a simple concatenation alone would provide.

### 20.12 Heads

The model has three task heads:

- an SRL classifier,
- a start projection,
- an end projection.

The SRL classifier is a linear layer over context states.

The start and end heads are linear projections over the interaction tensor.

### 20.13 Losses

During training, the model computes:

- token-level cross entropy for SRL,
- cross entropy for answer start,
- cross entropy for answer end.

The final loss is:

`alpha * srl_loss + (1 - alpha) * qa_loss`

where `qa_loss` is the mean of start and end losses.

### 20.14 Masking

The model masks invalid context positions before start and end softmaxing.

This ensures pad positions cannot be selected as answers.

### 20.15 Prediction logic

`predict()` is more interesting than a naive “argmax start, argmax end” decoder.

It:

1. runs a forward pass without supervision,
2. decodes the SRL label sequence,
3. converts BIO labels into candidate spans,
4. gets start and end probabilities,
5. chooses a boundary-favored fallback span,
6. scores candidate SRL spans by combining:
   - cosine similarity between span vector and question vector,
   - boundary probability,
7. returns the best candidate span if any exist,
8. otherwise returns the fallback span.

This hybrid decoder is one of the most distinctive parts of the project.

It is not just QA span prediction.

It is question-conditioned span selection grounded in SRL span candidates.

### 20.16 Scoring formula

Candidate span score is:

- `0.60 * cosine_similarity_normalized`,
- plus `0.40 * boundary_score`.

That weighting indicates the project values semantic alignment to the question slightly more than raw span-boundary confidence.

### 20.17 Fallback behavior

If no candidate BIO spans decode cleanly, the model:

- chooses the best start and end boundary span,
- assigns the majority non-`O` role from the selected label window.

This keeps the system robust under imperfect SRL predictions.

### 20.18 `model_summary`

This helper returns:

- model name,
- trainable parameter count,
- hidden sizes.

The current checkpoint reports `1,784,352` trainable parameters.

That is compact relative to modern transformer QA systems.

## 21. `trainer.py`

`trainer.py` handles optimization, checkpointing, and validation monitoring.

### 21.1 `set_random_seed`

This function seeds:

- Python random,
- NumPy,
- PyTorch CPU RNG,
- PyTorch CUDA RNG if available.

That improves reproducibility.

### 21.2 `move_batch_to_device`

This helper moves tensor fields in a batch onto the runtime device while leaving Python objects such as `raw_examples` untouched.

That makes downstream code cleaner.

### 21.3 `token_level_f1`

This function computes overlap-based token F1 for predicted vs gold answer tokens.

It lowercases tokens and counts multiplicities.

This is standard extractive-QA style bag overlap rather than order-sensitive exact structural matching.

### 21.4 Validation evaluation

`evaluate_validation_split`:

- runs the model on the validation loader,
- computes average validation loss,
- computes exact match,
- computes token F1.

This is used during training to decide whether a checkpoint is best.

### 21.5 `train_model`

This is the core training loop.

Its steps are:

1. seed RNGs,
2. build the model,
3. construct the Adam optimizer,
4. iterate over epochs,
5. train on all batches,
6. compute validation loss and QA metrics,
7. append the epoch record to history,
8. save checkpoint on improved validation F1,
9. stop early if patience is exhausted,
10. reload the best checkpoint before returning.

### 21.6 Checkpoint contents

The saved checkpoint includes:

- `model_state`,
- `config`,
- `history`,
- `best_epoch`,
- `best_validation_f1`,
- `vocabularies`,
- `model_summary`.

This is a thoughtfully complete checkpoint.

It stores not only weights but also enough metadata to reconstruct the model and its training context.

### 21.7 Why validation F1 is used

The code uses validation token F1 as the main early-stopping and checkpoint criterion.

That makes sense because the end task is extractive QA.

The SRL head is important.

But the submission’s observable answer quality is likely best captured by QA F1.

### 21.8 Current training trace summary

From `metrics.json`, training history shows steady improvement over six epochs.

Validation F1 grows from roughly `0.5360` in epoch 1 to `0.7727` by epoch 6.

That indicates the baseline trains stably on the cached dataset.

## 22. `evaluator.py`

`evaluator.py` converts predictions into metrics, analyses, and plots.

This file is the bridge between model behavior and human-readable evidence.

### 22.1 Main responsibilities

It:

- reloads the trained model,
- runs predictions on the test set,
- creates per-example records,
- computes SRL metrics,
- computes QA metrics,
- classifies errors,
- builds error summaries,
- generates plots,
- saves `metrics.json`.

### 22.2 `load_trained_model`

This function:

- loads the checkpoint,
- rebuilds the model shape using serialized vocab sizes,
- loads model weights,
- sets eval mode.

That is the shared entrypoint used by evaluation and inference.

### 22.3 `prediction_records`

This function generates one record per test example.

Each record includes:

- example ID,
- context,
- question,
- question type,
- gold text,
- predicted text,
- gold role,
- predicted role,
- gold BIO sequence,
- predicted BIO sequence,
- confidence,
- exact match,
- token F1,
- answer length difference,
- sentence length.

This is a very good design choice.

Rather than computing only aggregates, the project preserves example-level evidence.

That supports detailed error analysis and paper writing.

### 22.4 SRL metrics

`role_metrics_from_records` computes:

- per-role precision,
- per-role recall,
- per-role F1,
- per-role support,
- macro precision,
- macro recall,
- macro F1,
- micro precision,
- micro recall,
- micro F1,
- BIO accuracy,
- confusion matrix,
- confusion matrix labels.

The function strips BIO prefixes when computing role-level metrics.

That means `B-ARG1` and `I-ARG1` both contribute to `ARG1`.

This is appropriate for role-centric reporting.

### 22.5 QA metrics

`qa_metrics_from_records` computes:

- overall exact match,
- overall token F1,
- mean answer-length deviation,
- mean absolute answer-length deviation,
- per-question-type breakdown.

The per-question-type breakdown is especially useful because the project generates specific classes of questions.

### 22.6 Current QA breakdown by question type

From the repository’s current metrics:

- `WHO`: EM `0.6182`, F1 `0.7788`, count `867`,
- `WHAT`: EM `0.5049`, F1 `0.8024`, count `1937`,
- `WHEN`: EM `0.5650`, F1 `0.6871`, count `246`,
- `WHERE`: EM `0.3818`, F1 `0.6190`, count `110`,
- `HOW`: EM `0.3590`, F1 `0.5301`, count `234`,
- `WHY`: EM `0.1754`, F1 `0.6344`, count `57`.

This tells a meaningful story.

The model is relatively stronger on `WHO` and `WHAT`.

It struggles more on `WHERE`, `HOW`, and especially `WHY`.

That is plausible because causal and manner questions can require more subtle phrase boundaries and deeper semantics.

### 22.7 Error taxonomy

`classify_error` assigns each prediction to one of:

- `correct`,
- `predicate miss`,
- `wrong role`,
- `span boundary error`,
- `other`.

### 22.8 Current error counts

From `results/metrics.json`, current taxonomy counts are:

- `correct`: `1789`,
- `span boundary error`: `866`,
- `wrong role`: `685`,
- `other`: `109`,
- `predicate miss`: `2`.

This is very informative.

The model rarely completely misses the predicate structure.

Most failures are partial semantic or boundary mistakes.

### 22.9 Plot generation

The file defines plot functions for:

- loss curves,
- role F1 by argument type,
- role confusion matrix,
- QA accuracy by question type,
- answer-length distribution,
- error taxonomy.

These plots make the results suitable for a project report instead of only a terminal log.

### 22.10 `evaluate_model`

This function orchestrates full evaluation.

It:

- loads the checkpoint,
- creates prediction records,
- computes SRL and QA metrics,
- performs error analysis,
- writes plots,
- saves `metrics.json`,
- returns the full metrics payload.

## 23. `qa_inference.py`

`qa_inference.py` is the runtime inference wrapper.

It serves two related purposes:

- demo inference on fixed examples,
- custom user question answering on raw text.

### 23.1 Why this file exists

Training and evaluation use dataset examples built from structured corpora.

But a useful QA system also needs runtime behavior on user-provided text.

This module provides that bridge.

### 23.2 Raw-text inference challenge

At inference time, the system does not have:

- Treebank parse trees,
- gold POS tags,
- gold predicate index,
- gold BIO labels.

So it must approximate those inputs heuristically.

### 23.3 Runtime heuristics used

The module implements:

- regex tokenization,
- approximate lemmatization,
- heuristic POS tagging,
- predicate index inference from question overlap and verb morphology.

These heuristics are intentionally lightweight.

They keep the raw-text inference path simple and dependency-free.

### 23.4 `InferenceEngine`

This class loads the saved checkpoint and stores:

- the runtime device,
- the model,
- the checkpoint,
- serialized vocabularies,
- the label ID mapping.

### 23.5 `infer(context, question)`

This method:

1. tokenizes the context,
2. tokenizes the question,
3. assigns heuristic POS tags,
4. infers a predicate index,
5. builds predicate flags,
6. encodes tokens with stored vocabularies,
7. constructs a one-example batch,
8. runs the model’s `predict()` decoder,
9. converts the predicted span back to text,
10. returns answer text, confidence, and predicted role.

### 23.6 Empty answer handling

The current code includes a small defensive improvement:

If the decoded span is empty after joining tokens, it returns `(no answer found)`.

That keeps the UI cleaner for edge cases.

### 23.7 New custom question-asking feature

The current repository now supports:

- `ask_question(config, context, question)`,
- `ask_question_with_engine(engine, context, question)`,
- `run_interactive_session(config)`.

This is the new feature added on top of the original demo-only inference path.

### 23.8 Why the two ask functions exist

`ask_question` is convenient for one-off use.

`ask_question_with_engine` is efficient for repeated use because it reuses an already loaded model.

That matters in interactive sessions.

Without it, every new question would reload the checkpoint.

### 23.9 Fixed demo examples

The module still keeps the original 10-example demo set.

Those examples cover:

- agent questions,
- temporal questions,
- causal questions,
- patient/object questions,
- location questions,
- manner questions.

That makes the demo a useful smoke test.

### 23.10 Current demo artifact

The file `results/inference_demo.json` stores 10 inference results.

The first example in the current repository is:

- context: `The chef cooked a delicious meal in the kitchen yesterday.`,
- question: `Who cooked?`,
- predicted answer: `The chef`,
- predicted role: `ARG0`,
- match: `CORRECT`.

### 23.11 Custom ask examples

One-shot CLI example:

```bash
python main.py --mode ask --context "The courier delivered the package to the office at noon." --question "Where was the package delivered?"
```

Interactive example:

```bash
python main.py --mode ask --interactive
```

Example interaction:

```text
[ask] context: The chef cooked a delicious meal in the kitchen yesterday.
[ask] question: Who cooked?
[ask] predicted answer: The chef
[ask] confidence: 0.6827
[ask] predicted role: ARG0
```

## 24. `pdf_generator.py`

`pdf_generator.py` is the report-production layer.

It turns project outputs into academic-style deliverables.

This module is very large because it mixes:

- document templates,
- shared styles,
- tables,
- paragraphs,
- figure generation,
- bibliography content,
- PDF assembly,
- implementation bundle export.

### 24.1 Why this file is so large

The project requirements clearly include more than just a runnable model.

They also include deliverables such as:

- a survey,
- an analysis report,
- an innovation proposal,
- a research paper,
- and a single-file code bundle.

So a lot of narrative content and formatting logic lives here.

### 24.2 Template classes

The file defines:

- `AcademicDocTemplate`,
- `TwoColumnDocTemplate`.

The first is used for single-column documents.

The second is used for the research paper.

Both support headers, footers, and table-of-contents registration.

### 24.3 Style system

`build_styles()` defines shared ReportLab paragraph styles.

These include:

- body,
- title,
- subtitle,
- heading1,
- heading2,
- caption,
- TOC heading.

This is a good example of DRY document generation.

### 24.4 Reusable composition helpers

The file includes helper functions like:

- `heading`,
- `add_cover_page`,
- `add_toc`,
- `add_paragraphs`,
- `add_image_with_caption`,
- `add_table`.

These helpers keep the later PDF functions readable.

### 24.5 Diagram generation

Several functions create figures with Matplotlib:

- `create_horizontal_flow_diagram`,
- `create_propbank_example_figure`,
- `create_frame_graph_diagram`,
- `create_hybrid_architecture_diagram`,
- `create_propagent_diagram`,
- `create_benchmark_diagram`,
- `create_diagram_assets`.

This means the project does not merely consume plots from evaluation.

It also programmatically creates conceptual diagrams for the written reports.

### 24.6 Shared bibliography

The function `reference_entries()` returns a bibliography list.

This allows the generated PDFs to include literature support without duplicating citation text across functions.

### 24.7 Metrics tables

The function `metrics_tables(...)` constructs report-ready tables from:

- dataset statistics,
- overall metrics,
- question-type metrics,
- training diagnostics.

This connects numeric evaluation directly to the reporting layer.

### 24.8 `generate_survey_pdf`

This function builds `outputs/survey.pdf`.

The survey covers:

- abstract,
- introduction,
- semantic role labeling background,
- PropBank corpus structure and statistics,
- SRL system history,
- QA overview,
- SRL-based QA related work,
- detailed literature capsules,
- datasets,
- research gaps,
- conclusion,
- extended discussion,
- final reflection,
- references.

This document is literature-oriented.

It uses the current local run statistics to ground the abstract and data discussion.

### 24.9 `generate_analysis_pdf`

This function builds `outputs/analysis.pdf`.

Its sections include:

- experimental setup,
- data analysis,
- SRL performance results,
- QA performance results,
- training diagnostics,
- error analysis,
- strengths vs limitations,
- discussion,
- conclusion of analysis.

This is the most directly empirical PDF.

It uses metrics and plots heavily.

### 24.10 `generate_innovation_pdf`

This function builds `outputs/innovation.pdf`.

Its sections include:

- abstract,
- why innovate,
- architecture enhancements,
- automated workflow,
- application use cases,
- hybrid SRL-QA + LLM proposal,
- benchmark proposal,
- conclusion and roadmap,
- implementation sketches,
- use-case walkthroughs,
- risks and mitigations.

This document pushes beyond the current baseline.

It is a design-thinking artifact.

### 24.11 `generate_research_paper_pdf`

This function builds `outputs/research_paper.pdf`.

It uses the two-column document template.

Its sections include:

- abstract,
- introduction,
- related work,
- background and data,
- PropQA-Net,
- experiments,
- error analysis,
- innovation summary,
- conclusion,
- reproducibility notes,
- practical implications,
- limitations and future work.

This is the most paper-like deliverable.

### 24.12 `export_implementation_bundle`

This function creates `outputs/implementation_code.py`.

It concatenates:

- `config.py`,
- `data_loader.py`,
- `model.py`,
- `trainer.py`,
- `evaluator.py`,
- `qa_inference.py`,
- `pdf_generator.py`,
- `main.py`.

It removes repeated `from __future__ import annotations` lines so the bundle remains valid as a single file.

This is an elegant and practical implementation detail.

### 24.13 `generate_all_pdfs`

This function orchestrates all final deliverables.

It:

- creates diagram assets,
- generates all PDFs,
- exports the implementation bundle,
- returns page counts.

## 25. `main.py`

`main.py` is the central CLI runner.

It wires the modules together.

### 25.1 What `main.py` does well

It keeps orchestration separate from implementation.

It does not bury training logic inside the trainer.

It does not bury evaluation logic inside the evaluator.

It coordinates them.

### 25.2 Main functions

Important functions are:

- `configure_runtime`,
- `prepare_data`,
- `validate_final_outputs`,
- `main`.

### 25.3 `configure_runtime`

This function auto-selects:

- `cuda` if available,
- otherwise `cpu`.

### 25.4 `prepare_data`

This function wraps:

- split loading/building,
- data statistics generation,
- DataLoader building.

It returns both examples and loaders plus vocabularies and stats.

### 25.5 `validate_final_outputs`

This function checks that all required deliverables exist and are non-empty:

- `survey.pdf`,
- `implementation_code.py`,
- `analysis.pdf`,
- `innovation.pdf`,
- `research_paper.pdf`.

That is a very useful final guardrail.

### 25.6 CLI arguments

The current CLI supports:

- `--mode`,
- `--context`,
- `--question`,
- `--interactive`.

### 25.7 Ask-mode validation

The code now validates:

- `--context` and `--question` must appear together,
- custom question arguments are only supported in `ask` or `infer`-with-custom mode.

That prevents ambiguous behavior.

### 25.8 Execution branching

The CLI logic now branches into:

- train flow,
- eval flow,
- demo inference flow,
- custom ask flow,
- full flow.

This makes the program more usable without changing the project’s original architecture.

## 26. Existing Project Docs in `docs/`

The repository already included several focused docs.

This guide extends them rather than replacing them.

### 26.1 `docs/OVERVIEW.md`

Purpose:

- high-level introduction,
- reading order,
- conceptual pipeline diagram,
- design decisions,
- output cheat sheet.

### 26.2 `docs/DATA.md`

Purpose:

- explain PropBank and Treebank alignment,
- describe cached example schema,
- note where split JSON files live,
- summarize statistics outputs.

### 26.3 `docs/MODEL.md`

Purpose:

- explain inputs,
- explain architecture,
- explain losses,
- explain decoding,
- state limitations.

### 26.4 `docs/EVALUATION.md`

Purpose:

- describe evaluation outputs,
- define QA metrics,
- define SRL metrics,
- describe error analysis.

### 26.5 `docs/PDF_DELIVERABLES.md`

Purpose:

- explain generated deliverables,
- explain how PDFs are built,
- explain the generated `implementation_code.py` bundle.

### 26.6 `docs/TROUBLESHOOTING.md`

Purpose:

- handle missing corpora,
- handle torch install issues,
- handle matplotlib backend issues,
- explain slow runs,
- explain cache regeneration.

### 26.7 Role of this current guide

This guide unifies those smaller documents.

It also adds repo-wide file inventory, current metrics, artifact discussion, and practical walk-throughs.

## 27. `data/` Directory

The `data/` directory stores serialized cached splits:

- `train.json`,
- `val.json`,
- `test.json`.

### 27.1 Why caching matters

Building the dataset from PropBank is not free.

It requires:

- corpus access,
- parse-tree traversal,
- pointer resolution,
- span conversion,
- question generation,
- statistics collection.

Caching allows later runs to skip that cost.

### 27.2 Current file sizes

From the current repository state:

- `train.json`: `76,342,040` bytes,
- `val.json`: `16,427,962` bytes,
- `test.json`: `16,291,325` bytes.

### 27.3 Current split sizes

From the JSON content:

- `train.json`: `16,104` examples,
- `val.json`: `3,451` examples,
- `test.json`: `3,452` examples.

### 27.4 Representative field interpretation

If an example contains:

- `predicate_indices`,
- `predicate_flags`,
- `srl_tags`,
- `answer_start`,
- `answer_end`,

then the model can learn both:

- where the event is,
- and which span answers the question.

### 27.5 Why the JSON is rich

A minimal QA dataset would need only:

- context,
- question,
- answer start,
- answer end.

This project stores much more.

That makes the dataset inspectable and extensible.

## 28. `checkpoints/` Directory

The checkpoint directory currently contains:

- `best_model.pt`.

### 28.1 Current file size

The checkpoint size is `7,444,277` bytes.

### 28.2 What the checkpoint represents

It is the model snapshot that achieved the best validation QA F1 during training.

### 28.3 Why only one checkpoint is kept

For a project repository, one best checkpoint is usually enough.

It avoids unnecessary storage growth.

It keeps downstream evaluation deterministic.

### 28.4 What depends on this file

The following code paths require `best_model.pt`:

- evaluation,
- fixed inference demo,
- custom ask mode,
- any reproducibility check that reloads the model.

## 29. `results/` Directory

The `results/` directory is where the project’s evidence lives.

### 29.1 `results/data_statistics.json`

This file stores descriptive dataset information.

Current size:

- `278,519` bytes.

Important top-level keys include:

- `total_propbank_instances`,
- `usable_propbank_instances`,
- `qa_pair_count`,
- `unique_predicates`,
- `unique_rolesets`,
- `argument_type_distribution`,
- `qa_pairs_per_argument_type`,
- `qa_pairs_per_question_type`,
- `sentence_length_distribution`,
- `answer_span_length_distribution`,
- `sentence_length_summary`,
- `answer_length_summary`,
- `sample_qa_pairs`,
- `split_sizes`,
- `corpus_overview`.

This file is useful for:

- sanity checks,
- report writing,
- diagnosing class imbalance,
- understanding answer-length behavior.

### 29.2 `results/metrics.json`

This file stores model performance and error analysis.

Current size:

- `203,890` bytes.

Important top-level keys include:

- `srl_performance`,
- `qa_performance`,
- `training_diagnostics`,
- `error_analysis`,
- `prediction_sample`.

This file is the best single machine-readable summary of model quality.

### 29.3 `results/inference_demo.json`

This file stores the mandatory 10-example demo.

Current size:

- `3,590` bytes.

Important fields per item include:

- context,
- question,
- predicted answer,
- confidence,
- predicted role,
- expected answer,
- match.

### 29.4 `results/plots/`

This directory stores visualizations generated by evaluation and by the PDF generator.

The current plot files are:

- `answer_length_dist.png`,
- `benchmark.png`,
- `confusion_matrix.png`,
- `error_taxonomy.png`,
- `f1_by_argtype.png`,
- `frame_graph.png`,
- `frame_memory.png`,
- `hybridpropqa.png`,
- `loss_curve.png`,
- `multi_predicate.png`,
- `propagent.png`,
- `propbank_example.png`,
- `propqa_architecture.png`,
- `qa_accuracy_by_qtype.png`,
- `srl_pipeline.png`.

Some are evaluation plots.

Some are conceptual diagrams.

### 29.5 What each plot roughly represents

`loss_curve.png`:

- training dynamics over epochs.

`confusion_matrix.png`:

- SRL role confusions.

`qa_accuracy_by_qtype.png`:

- question-type performance comparison.

`answer_length_dist.png`:

- gold answer length behavior.

`error_taxonomy.png`:

- relative frequency of error categories.

`f1_by_argtype.png`:

- role-level performance breakdown.

`propbank_example.png`:

- a small visual schematic of predicate-argument annotation.

`srl_pipeline.png`:

- conceptual SRL-QA processing flow.

`propqa_architecture.png`:

- conceptual model architecture.

`frame_graph.png`:

- multi-sentence frame chaining idea.

`frame_memory.png`:

- frame-aware predicate disambiguation concept.

`multi_predicate.png`:

- answer composition across predicates.

`propagent.png`:

- multi-agent SRL-QA workflow concept.

`hybridpropqa.png`:

- SRL + LLM hybrid concept.

`benchmark.png`:

- benchmark lifecycle concept.

## 30. `outputs/` Directory

The `outputs/` directory contains the final deliverables.

Current contents and sizes are:

- `analysis.pdf`: `1,969,399` bytes,
- `implementation_code.py`: `206,409` bytes,
- `innovation.pdf`: `521,758` bytes,
- `research_paper.pdf`: `1,419,257` bytes,
- `survey.pdf`: `244,089` bytes.

### 30.1 Why this directory exists separately from `results/`

`results/` holds analytic evidence.

`outputs/` holds final submission-style artifacts.

That is a clean separation.

### 30.2 `implementation_code.py`

The generated implementation bundle currently contains `4,399` lines.

Its opening banner makes clear that it is generated and should not be edited directly.

### 30.3 What to edit if PDFs need changes

Do not edit the PDFs directly.

Edit `pdf_generator.py`.

Then rerun:

```bash
python main.py --mode full
```

### 30.4 What to edit if the implementation bundle needs changes

Do not edit `outputs/implementation_code.py` directly.

Edit the real source modules.

Then regenerate the bundle.

## 31. `nltk_data/` Directory

This directory is crucial for reproducibility.

It is not a decorative addition.

It is how the repository remains runnable without downloading corpora at runtime.

### 31.1 High-level structure

Important packaged corpus resources include:

- `nltk_data/corpora/propbank/`,
- `nltk_data/corpora/treebank/`,
- zipped variants of those corpora,
- frame XML files,
- sample Treebank WSJ files.

### 31.2 PropBank packaged files

Important PropBank files and roles:

- `README`: explains PropBank format,
- `NOTES.txt`: release notes,
- `prop.txt`: main annotation data,
- `verbs.txt`: verb inventory,
- `vloc.txt`: verb locations in Treebank,
- `frames/*.xml`: lexical and roleset guidelines.

### 31.3 What `propbank/README` tells us

The README explains that PropBank is an additional annotation layer on the Penn Treebank.

It explains:

- filename,
- sentence index,
- terminal index,
- tagger,
- frameset,
- inflection,
- proposition labels.

It also explains secondary labels like:

- `TMP`,
- `LOC`,
- `MNR`,
- `CAU`,
- `PNC`,
- `ADV`.

This is directly relevant to how `data_loader.py` interprets labels and question types.

### 31.4 What `NOTES.txt` tells us

`NOTES.txt` reports:

- total propositions in PropBank I: `112,917`,
- total framed verbs: `3,323`,
- total framesets: `4,659`.

Those values align nicely with the repository’s current runtime corpus totals.

### 31.5 What `verbs.txt` and `vloc.txt` represent

`verbs.txt` is a lexical inventory.

`vloc.txt` maps verbs to positions in Treebank files.

That mapping helps show how PropBank annotations are anchored to the source WSJ corpus.

### 31.6 Representative frame file: `frames/agree.xml`

The sampled `agree.xml` frame file defines:

- `agree.01`,
- role `ARG0` as agreer,
- role `ARG1` as proposition,
- role `ARG2` as other entity agreeing.

It also includes examples.

This is exactly the kind of metadata `roleset_metadata()` uses to enrich questions and examples.

### 31.7 Treebank packaged files

Representative Treebank files include:

- `treebank/README`,
- `combined/README`,
- `combined/wsj_0001.mrg`,
- many more `wsj_*.mrg` files,
- tagged files under `tagged/`.

### 31.8 What `treebank/README` tells us

It says the packaged sample is about 5 percent of Penn Treebank.

It contains about 1,650 WSJ sentences from files `wsj_0001` through `wsj_0099`.

It explains the different corpus views:

- raw,
- tagged,
- parsed,
- combined.

This matters because the project relies on tree structures and POS tags from this local sample.

### 31.9 Representative Treebank parse file

The sampled `combined/wsj_0001.mrg` file shows standard bracketed parse trees.

The project uses structures like that to:

- recover visible tokens,
- recover POS tags,
- align PropBank pointer paths,
- reconstruct answer spans.

### 31.10 Why the local sample limits coverage

PropBank coverage is much larger than the shipped Treebank subset.

So only a subset of PropBank instances can be used.

That limitation is not a bug.

It is an explicit tradeoff tied to local reproducibility.

## 32. Practical Flow: From Corpora to QA Example

Here is the most important conceptual transformation in the entire repository.

```text
Treebank parse tree
    |
    v
Visible tokens + POS tags
    |
    v
PropBank predicate pointer -> predicate token indices
    |
    v
PropBank argument pointers -> argument token spans
    |
    v
BIO SRL labels over all tokens
    |
    v
Question type inference from role
    |
    v
Natural-language question generation
    |
    v
Extractive QA example with answer_start and answer_end
```

## 33. Example Walkthrough: Training Example Construction

Suppose the sentence contains a predicate like `agree`.

The pipeline does roughly this:

1. load the Treebank sentence,
2. get visible leaf tokens,
3. map the predicate pointer to token indices,
4. gather all argument pointers,
5. convert argument indices to spans,
6. assign BIO labels over the whole sentence,
7. identify `ARG0` as likely subject text,
8. read frame metadata from `agree.xml`,
9. decide that `ARGM-TMP` maps to a `WHEN` question,
10. generate something like `When did Western Union agree?`,
11. store the gold answer span indices,
12. append the example to the dataset.

This is the key “semantic annotation to QA supervision” move.

## 34. Practical Flow: Training

```text
Cached examples
    |
    v
Vocabulary building
    |
    v
SRLQADataset encoding
    |
    v
PyTorch DataLoader batches
    |
    v
PropQA-Net forward pass
    |
    +--> SRL loss
    |
    +--> QA start loss
    |
    +--> QA end loss
    |
    v
Weighted total loss
    |
    v
Backpropagation
    |
    v
Validation EM/F1
    |
    v
Checkpoint if best
```

## 35. Practical Flow: Evaluation

```text
best_model.pt
    |
    v
Rebuild model and load weights
    |
    v
Run test loader
    |
    v
Create prediction records
    |
    +--> SRL metrics
    +--> QA metrics
    +--> Error taxonomy
    +--> Plots
    |
    v
Save metrics.json
```

## 36. Practical Flow: Custom Asking

```text
User types context and question
    |
    v
Regex tokenization
    |
    v
Heuristic POS tags
    |
    v
Predicate index inference
    |
    v
Vocabulary encoding
    |
    v
Single-example batch
    |
    v
Model prediction
    |
    v
Decoded answer text + role + confidence
```

## 37. Example Walkthrough: `Who cooked?`

Consider:

- context: `The chef cooked a delicious meal in the kitchen yesterday.`
- question: `Who cooked?`

The inference path roughly does this:

1. tokenize context into words and punctuation,
2. tokenize question,
3. infer that `cooked` is the likely predicate,
4. mark the predicate position with a `1` in predicate flags,
5. assign heuristic POS tags,
6. encode tokens using the stored vocabularies,
7. run the network,
8. decode BIO spans,
9. compare candidate spans to the question vector,
10. select the best answer span.

In the current repository, the result is:

- predicted answer: `The chef`,
- predicted role: `ARG0`,
- confidence: about `0.68`.

That is exactly the kind of behavior the project is designed to support.

## 38. Example Walkthrough: `Where was the package delivered?`

Example:

- context: `The courier delivered the package to the office at noon.`
- question: `Where was the package delivered?`

Expected semantics:

- predicate: `delivered`,
- target role: locative adjunct,
- likely label: `ARGM-LOC`,
- gold answer: `to the office`.

This illustrates how spatial questions are treated as role-targeted extraction rather than open-ended generation.

## 39. Example Walkthrough: Why Questions

Example:

- context: `The company announced layoffs because of budget cuts.`
- question: `Why were layoffs announced?`

Expected semantics:

- target role: `ARGM-CAU`,
- answer span: `because of budget cuts`.

The current metrics show why questions are among the harder categories.

They are often low-frequency and semantically subtle.

## 40. Why the Model Is Stronger on `WHO` and `WHAT`

The current results suggest better performance on `WHO` and `WHAT`.

Likely reasons:

- these roles are more frequent,
- `ARG0` and `ARG1` are central arguments,
- they often align with prominent NP spans,
- they are easier to detect from lexical and positional cues.

## 41. Why the Model Is Weaker on `WHY`

Likely reasons include:

- fewer training examples,
- causal phrases can be syntactically diverse,
- causal spans may include clausal material,
- semantic boundary decisions are harder.

This is visible in the current `WHY` exact match of `0.1754`.

## 42. Strengths of the Current Architecture

Key strengths:

- fully reproducible local pipeline,
- grounded in real PropBank annotations,
- explicit role-aware QA formulation,
- transparent preprocessing,
- manageable model size,
- rich evaluation artifacts,
- report generation built into the repo,
- custom asking supported at runtime.

## 43. Limitations of the Current Architecture

Important limitations:

- sentence-local only,
- extractive only,
- heuristic raw-text inference,
- no transformer contextual embeddings,
- drops non-contiguous arguments,
- limited to local Treebank-backed subset,
- no calibrated abstention,
- no document-level reasoning.

## 44. Why the Project Is Still Valuable Despite Those Limits

Because it is structured and inspectable.

Because it demonstrates a full classical pipeline.

Because it connects linguistic annotation to QA behavior clearly.

Because it is a good baseline for future extensions.

Because it generates evidence-rich outputs, not just a single score.

## 45. Current Data Distribution Notes

From `data_statistics.json`:

- `WHAT` questions dominate the dataset.
- `WHO` is also common.
- `WHY` is rare.
- mean sentence length is about `28.60`.
- mean answer length is about `5.28`.
- max sentence length observed in the usable set is `249`.
- max answer length observed is `62`.

These statistics explain several modeling behaviors.

Longer answers raise boundary difficulty.

Rare question types raise generalization difficulty.

## 46. What the Model Actually Learns

It learns a shared representation that supports:

- tagging semantic-role structure over tokens,
- matching question intent to candidate spans,
- and selecting a final extractive answer.

It does not learn to generate novel text.

It does not learn long-form reasoning.

It learns structured extraction.

## 47. Why Explicit Predicate Flags Matter

The predicate flag embedding is small.

But conceptually it is very important.

It tells the model which event the question is about.

That reduces ambiguity in sentences with multiple actions.

Without a predicate anchor, role prediction becomes much noisier.

## 48. Role of the Question Encoder

The question encoder exists so the model does not only locate semantic arguments in general.

It locates the argument relevant to the specific question.

That distinction matters when multiple arguments are present in the same sentence.

## 49. Role of the SRL Decoder in QA

The model does not trust only start-end scores.

It also tries to decode semantically coherent argument spans from BIO tags.

That is the project’s main conceptual novelty relative to a plain extractive QA baseline.

## 50. Why the Repository Is Good for Teaching

This repository is good for teaching because:

- each stage is visible,
- the model is not too large to understand,
- the dataset schema is readable,
- the evaluation artifacts are rich,
- and the reporting layer is integrated.

## 51. Suggested Reading Order for New Contributors

Recommended order:

1. `README.md`
2. `docs/OVERVIEW.md`
3. `config.py`
4. `main.py`
5. `data_loader.py`
6. `model.py`
7. `trainer.py`
8. `evaluator.py`
9. `qa_inference.py`
10. `pdf_generator.py`
11. `results/data_statistics.json`
12. `results/metrics.json`

## 52. Suggested Reading Order for Reviewers

If someone is reviewing the repository quickly:

1. `README.md`
2. `main.py`
3. `model.py`
4. `results/metrics.json`
5. `outputs/research_paper.pdf`

## 53. Suggested Reading Order for Extenders

If someone wants to extend the model:

1. `config.py`
2. `data_loader.py`
3. `model.py`
4. `trainer.py`
5. `evaluator.py`
6. `qa_inference.py`

## 54. Common Extension Directions

The current repository itself suggests several future directions:

- frame-aware predicate disambiguation,
- multi-predicate reasoning,
- sentence-to-document expansion,
- calibrated abstention,
- hybrid SRL + LLM workflows,
- richer raw-text preprocessing,
- better handling of non-contiguous arguments.

## 55. How to Add a New Feature Safely

A safe extension workflow would be:

1. decide whether the change belongs to data, model, training, evaluation, inference, or reporting,
2. update `config.py` if the feature needs new parameters,
3. update cached preprocessing only if needed,
4. retrain if the model inputs or targets change,
5. rerun evaluation,
6. inspect `metrics.json`,
7. regenerate PDFs if the deliverables should reflect the change.

## 56. Reproducibility Notes

This repository already encodes several good reproducibility practices:

- local corpora,
- deterministic split seed,
- serialized vocabularies,
- saved training history,
- metrics written to disk,
- output validation.

These design choices are worth preserving.

## 57. Performance Interpretation Notes

The exact match score near `0.52` and token F1 near `0.76` tell an important story.

The model often gets at least part of the answer span right.

But exact boundary matching remains difficult.

That is common in extractive QA.

It is especially common when answers are multi-token or clausal.

## 58. Why Boundary Errors Dominate

Boundary errors dominate because:

- the right semantic area may be found,
- but span start and end selection can still overshoot or undershoot,
- longer spans make this worse,
- adjuncts can have fuzzy clausal boundaries.

The repository’s taxonomy confirms this.

## 59. How Reports and Code Reinforce Each Other

One of the strengths of this repo is that:

- source code creates outputs,
- outputs feed reports,
- reports are reproducible from the same pipeline.

This creates a nice closed loop:

```text
Code -> Data -> Model -> Metrics -> Plots -> PDFs
```

That is a strong submission pattern for academic project work.

## 60. Final Summary of the Whole Project

PropQA-Net is a classical SRL-grounded extractive QA system.

It turns local PropBank annotations into QA examples.

It uses Treebank alignment to make spans exact and reproducible.

It trains a compact BiLSTM multi-task model.

It evaluates both semantics and answer extraction.

It exposes demo and custom inference.

It generates polished final deliverables.

It is a strong educational and baseline research repository because its structure is explicit from corpus to paper.
