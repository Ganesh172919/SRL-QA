"""Project-wide configuration for the SRL-QA pipeline.

This module centralizes every path and hyperparameter used across the
project so the rest of the codebase can stay focused on data processing,
modeling, evaluation, and reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class PathConfig:
    """Filesystem layout for the project.

    Attributes:
        project_root: Root directory of the runnable project package.
        data_dir: Directory that stores serialized train/validation/test data.
        checkpoints_dir: Directory that stores model checkpoints.
        results_dir: Directory that stores metrics and plots.
        plots_dir: Directory that stores evaluation plots.
        outputs_dir: Directory that stores the final required deliverables.
        nltk_data_dir: Local NLTK data cache used by the loader.
        propbank_dir: Local PropBank corpus directory.
        treebank_dir: Local Treebank corpus directory.
        metrics_path: JSON file with evaluation metrics.
        checkpoint_path: Best checkpoint path.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    data_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    nltk_data_dir: Path = field(init=False)
    propbank_dir: Path = field(init=False)
    treebank_dir: Path = field(init=False)
    train_json: Path = field(init=False)
    val_json: Path = field(init=False)
    test_json: Path = field(init=False)
    metrics_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived paths after initialization."""

        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.results_dir = self.project_root / "results"
        self.plots_dir = self.results_dir / "plots"
        self.outputs_dir = self.project_root / "outputs"
        self.nltk_data_dir = self.project_root / "nltk_data"
        self.propbank_dir = self.nltk_data_dir / "corpora" / "propbank"
        self.treebank_dir = self.nltk_data_dir / "corpora" / "treebank"
        self.train_json = self.data_dir / "train.json"
        self.val_json = self.data_dir / "val.json"
        self.test_json = self.data_dir / "test.json"
        self.metrics_path = self.results_dir / "metrics.json"
        self.checkpoint_path = self.checkpoints_dir / "best_model.pt"

    def ensure_directories(self) -> None:
        """Create the project directories when they do not exist."""

        for path in (
            self.data_dir,
            self.checkpoints_dir,
            self.results_dir,
            self.plots_dir,
            self.outputs_dir,
            self.nltk_data_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def as_serializable_dict(self) -> Dict[str, str]:
        """Convert paths into a JSON-serializable dictionary.

        Returns:
            A dictionary with stringified filesystem paths.
        """

        return {
            field_name: str(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
            if isinstance(getattr(self, field_name), Path)
        }


@dataclass(slots=True)
class DataConfig:
    """Data preprocessing configuration."""

    random_seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    lowercase_tokens: bool = True
    min_token_frequency: int = 1
    max_sentence_length: int = 128
    max_question_length: int = 32
    max_instances: int | None = None
    rebuild_cache: bool = False


@dataclass(slots=True)
class ModelConfig:
    """Model architecture hyperparameters."""

    word_embedding_dim: int = 100
    pos_embedding_dim: int = 32
    predicate_embedding_dim: int = 8
    hidden_size: int = 128
    question_hidden_size: int = 128
    dropout: float = 0.30
    alpha: float = 0.50


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 6
    patience: int = 5
    grad_clip_norm: float = 5.0
    num_workers: int = 0


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime options."""

    device: str = "cpu"
    verbose: bool = True


@dataclass(slots=True)
class ProjectConfig:
    """Top-level configuration container."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert nested dataclasses into serializable dictionaries.

        Returns:
            A nested configuration dictionary.
        """

        payload = asdict(self)
        payload["paths"] = self.paths.as_serializable_dict()
        return payload


def get_config() -> ProjectConfig:
    """Create the default project configuration.

    Returns:
        A populated project configuration instance.
    """

    config = ProjectConfig()
    config.paths.ensure_directories()
    return config
