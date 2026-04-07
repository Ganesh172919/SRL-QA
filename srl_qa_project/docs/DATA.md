# Data

This project constructs QA examples from PropBank argument annotations, but it must also reconstruct *exact answer spans* as token indices. To make that possible, the loader aligns PropBank instances with the local Penn Treebank parse trees shipped in `nltk_data/`.

## Source Assets

The repository contains an NLTK-style `nltk_data/` directory:

- `nltk_data/corpora/propbank`: PropBank frames + annotation index
- `nltk_data/corpora/treebank`: a Treebank subset used to reconstruct sentence tokens

At runtime, `data_loader.initialize_nltk(...)` prepends this folder to `nltk.data.path`, so `nltk.corpus.propbank` and `nltk.corpus.treebank` resolve locally.

## Example Generation (High Level)

For each PropBank instance:

1. Locate the referenced Treebank sentence so the sentence tokens exist locally.
2. Choose a predicate anchor token index.
3. Create SRL BIO tags for the context tokens.
4. For each target role (for example `ARG0`, `ARG1`, `ARGM-TMP`, ...), generate a natural-language question (for example `Who ...?`, `When ...?`) and compute the gold answer span as `(start_token_index, end_token_index)`.
5. Save the result as a JSON-friendly dictionary.

The cached splits live in:

- `data/train.json`
- `data/val.json`
- `data/test.json`

They are loaded on subsequent runs unless `ProjectConfig.data.rebuild_cache` is set to `True`.

## Cached Example Schema

Each item in `data/*.json` is a single extractive QA example. The schema is intentionally redundant so debugging and reporting can be done without reloading NLTK.

Fields you will see in practice include:

- `example_id`: stable string identifier used for reporting
- `instance_id`: PropBank instance identifier (file, sentence number, roleset)
- `context`: detokenized sentence text
- `context_tokens`: list of sentence tokens
- `pos_tags`: POS tags aligned to `context_tokens`
- `predicate_index`: integer token index of the predicate anchor
- `roleset`: PropBank roleset id such as `agree.01`
- `srl_tags`: BIO SRL tags for every context token
- `target_role`: role this question is asking about (`ARG0`, `ARG1`, `ARGM-LOC`, ...)
- `question_type`: coarse question class (`WHO`, `WHAT`, `WHEN`, `WHERE`, `WHY`, `HOW`)
- `question`: natural-language question
- `question_tokens`: tokenized question
- `answer_text`: detokenized gold answer text
- `answer_tokens`: gold answer tokens
- `answer_start`, `answer_end`: inclusive token boundaries in `context_tokens`

If you want to inspect a few examples quickly, open `data/train.json` and search for `question` and `answer_text`.

## Statistics

The pipeline writes a descriptive summary to `results/data_statistics.json`, including:

- PropBank total instances visible to NLTK
- usable Treebank-backed instances
- split sizes
- sample QA pairs
