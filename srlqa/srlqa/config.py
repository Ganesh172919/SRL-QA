"""Configuration for the RAISE-SRL-QA scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROLE_LABELS = [
    "O",
    "ARG0",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARG4",
    "ARG5",
    "ARGA",
    "ARGM-ADV",
    "ARGM-CAU",
    "ARGM-DIR",
    "ARGM-EXT",
    "ARGM-LOC",
    "ARGM-MNR",
    "ARGM-PNC",
    "ARGM-PRP",
    "ARGM-TMP",
]


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return package_root().parent


@dataclass(slots=True)
class PathConfig:
    """Filesystem layout for the new isolated package."""

    project_root: Path = field(default_factory=package_root)
    workspace_root: Path = field(default_factory=workspace_root)
    existing_project_root: Path = field(init=False)
    data_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    hf_cache_dir: Path = field(init=False)
    frame_store_path: Path = field(init=False)
    challenge_suite_path: Path = field(init=False)
    eval_manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.existing_project_root = self.workspace_root / "srl_qa_project"
        self.data_dir = self.project_root / "data"
        self.docs_dir = self.project_root / "docs"
        self.results_dir = self.project_root / "results"
        self.model_dir = self.project_root / "models"
        self.hf_cache_dir = self.project_root / ".hf_cache"
        self.frame_store_path = self.project_root / "retrieval" / "frame_store.json"
        self.challenge_suite_path = self.data_dir / "challenge_suite_v2.json"
        self.eval_manifest_path = self.results_dir / "eval_manifest.json"

    @property
    def existing_propbank_frames_dir(self) -> Path:
        return (
            self.existing_project_root
            / "nltk_data"
            / "corpora"
            / "propbank"
            / "frames"
        )

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.docs_dir,
            self.results_dir,
            self.model_dir,
            self.hf_cache_dir,
            self.frame_store_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DatasetConfig:
    """Online dataset configuration."""

    name: str = "marcov/qa_srl_promptsource"
    alternatives: tuple[str, ...] = (
        "luheng/qa_srl",
        "biu-nlp/qa_srl2018",
        "biu-nlp/qa_srl2020",
        "Lots-of-LoRAs/task1520_qa_srl_answer_generation",
    )
    split: str = "train"
    trust_remote_code: bool = False
    max_examples: int | None = 512


@dataclass(slots=True)
class ModelConfig:
    """Online model configuration."""

    encoder_name: str = "microsoft/deberta-v3-base"
    teacher_qa_name: str = "deepset/deberta-v3-base-squad2"
    optional_qasrl_model: str = "biu-nlp/contextualizer_qasrl"
    role_labels: tuple[str, ...] = tuple(ROLE_LABELS)
    max_length: int = 384
    doc_stride: int = 96
    dropout: float = 0.15


@dataclass(slots=True)
class TrainingConfig:
    """Default training knobs for a first MRC experiment."""

    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_ratio: float = 0.06
    hard_negative_ratio: float = 0.35
    verifier_threshold: float = 0.65


@dataclass(slots=True)
class DecodingConfig:
    """Constrained decoder defaults."""

    top_k: int = 20
    max_span_length: int = 18
    no_answer_threshold: float = 0.25
    length_penalty: float = 0.03


@dataclass(slots=True)
class ProjectConfig:
    """Top-level configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload["paths"].items():
            if isinstance(value, Path):
                payload["paths"][key] = str(value)
        return payload


def get_config() -> ProjectConfig:
    config = ProjectConfig()
    config.paths.ensure()
    return config
