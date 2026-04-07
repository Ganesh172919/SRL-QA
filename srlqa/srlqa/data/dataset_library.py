"""Online dataset and model discovery/loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from .convert_to_mrc import normalize_records


def _safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def search_huggingface_assets(query: str, limit: int = 8) -> dict[str, list[str]]:
    """Search Hugging Face for datasets and models.

    This is intentionally a lightweight online search through the official
    Hugging Face Hub client, which keeps the selected dataset/model explicit
    and reproducible.
    """

    from huggingface_hub import HfApi

    api = HfApi()
    datasets = [item.id for item in api.list_datasets(search=query, limit=limit)]
    models = [item.id for item in api.list_models(search=query, limit=limit)]
    return {"query": [query], "datasets": datasets, "models": models}


def load_library_dataset(
    config: ProjectConfig,
    split: str | None = None,
    max_examples: int | None = None,
) -> Any:
    """Load the configured QA-SRL dataset through `datasets.load_dataset`."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for library dataset loading. "
            "Install it with `python -m pip install -r requirements.txt`."
        ) from exc

    split_name = split or config.dataset.split
    kwargs: dict[str, Any] = {
        "split": split_name,
        "cache_dir": str(config.paths.hf_cache_dir / "datasets"),
    }
    if config.dataset.trust_remote_code:
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(config.dataset.name, **kwargs)
    limit = max_examples if max_examples is not None else config.dataset.max_examples
    if limit is not None and hasattr(dataset, "select"):
        limit = min(int(limit), len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def preview_dataset(config: ProjectConfig, max_examples: int = 5) -> dict[str, Any]:
    """Load and normalize a tiny dataset sample for inspection."""

    dataset = load_library_dataset(config, max_examples=max_examples)
    records = [dict(dataset[index]) for index in range(min(max_examples, len(dataset)))]
    normalized = normalize_records(records, source=config.dataset.name)
    return {
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "raw_columns": list(dataset.column_names) if hasattr(dataset, "column_names") else [],
        "sample_count": len(records),
        "normalized_samples": normalized,
    }


def download_model_snapshot(model_name: str, config: ProjectConfig) -> dict[str, str]:
    """Download a model repository snapshot through `huggingface_hub`."""

    from huggingface_hub import snapshot_download

    target = config.paths.hf_cache_dir / "models" / _safe_dir_name(model_name)
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return {"model": model_name, "path": str(Path(path).resolve())}


def download_dataset_snapshot(dataset_name: str, config: ProjectConfig) -> dict[str, str]:
    """Download a dataset repository snapshot through `huggingface_hub`."""

    from huggingface_hub import snapshot_download

    target = config.paths.hf_cache_dir / "datasets" / _safe_dir_name(dataset_name)
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return {"dataset": dataset_name, "path": str(Path(path).resolve())}
