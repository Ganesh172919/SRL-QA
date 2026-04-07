"""CLI helper for building the frame index."""

from __future__ import annotations

from pathlib import Path

from ..config import get_config
from .propbank_index import FrameIndex


def build(frames_dir: Path | None = None) -> Path:
    config = get_config()
    source = frames_dir or config.paths.existing_propbank_frames_dir
    index = FrameIndex.from_directory(source)
    index.save(config.paths.frame_store_path)
    return config.paths.frame_store_path
