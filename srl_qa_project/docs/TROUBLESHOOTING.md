# Troubleshooting

## NLTK Corpora Not Found

Symptoms:

- `LookupError: Resource ... not found`

What to check:

- Confirm `srl_qa_project/nltk_data/` exists.
- Confirm you run from inside `srl_qa_project/` (or that your current working directory is correct).

The loader explicitly registers the local `nltk_data/` directory at runtime, so you should not need to download corpora for this project.

## Torch Install Issues

Symptoms:

- pip cannot resolve `torch==...+cpu`
- import errors when importing torch

What to check:

- Try installing torch separately first, then install the remaining requirements.
- Ensure you are using a Python version supported by your chosen torch build.

## Matplotlib Backend Errors

Symptoms:

- failures when saving figures
- errors about interactive backends

What to try:

- Run from a normal terminal session (not embedded in some restricted environments).
- Ensure `results/plots/` is writable.

## Slow Runs

What to try:

- Reduce `ProjectConfig.data.max_instances` to limit the number of PropBank instances processed.
- Keep cached splits under `data/` and avoid rebuilding unless necessary.

## Regenerating Cached Data

If you change preprocessing behavior and want to regenerate `data/*.json`, set:

- `ProjectConfig.data.rebuild_cache = True`

Then run `python main.py --mode train` (or `--mode full`).

