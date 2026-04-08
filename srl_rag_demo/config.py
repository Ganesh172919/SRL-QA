"""Configuration for the standalone SRL + RAG demo."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Keep Transformers on the PyTorch path. This avoids the Keras 3 / TensorFlow
# import issue observed in the current local environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


@dataclass(slots=True)
class DemoConfig:
    """Filesystem and model defaults for the demo."""

    demo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    workspace_root: Path = field(init=False)
    legacy_root: Path = field(init=False)
    srlqa_root: Path = field(init=False)
    nltk_data_dir: Path = field(init=False)
    propbank_frames_dir: Path = field(init=False)
    frame_store_path: Path = field(init=False)
    cache_dir: Path = field(init=False)
    propbank_cache_dir: Path = field(init=False)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    teacher_qa_model_name: str = "deepset/deberta-v3-base-squad2"
    default_propbank_limit: int = 300
    default_top_k: int = 5

    def __post_init__(self) -> None:
        self.workspace_root = self.demo_root.parent
        self.legacy_root = self.workspace_root / "srl_qa_project"
        self.srlqa_root = self.workspace_root / "srlqa"
        self.nltk_data_dir = self.legacy_root / "nltk_data"
        self.propbank_frames_dir = self.nltk_data_dir / "corpora" / "propbank" / "frames"
        self.frame_store_path = self.srlqa_root / "retrieval" / "frame_store.json"
        self.cache_dir = self.demo_root / ".cache"
        self.propbank_cache_dir = self.cache_dir / "propbank"

    def ensure_cache_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.propbank_cache_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> DemoConfig:
    config = DemoConfig()
    config.ensure_cache_dirs()
    return config
