"""Generate a local leaderboard report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


LEADERBOARD_COLUMNS = ["system", "exact_match", "token_f1", "role_accuracy", "ece", "notes"]


def leaderboard_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    header = "| " + " | ".join(LEADERBOARD_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in LEADERBOARD_COLUMNS) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(str(row.get(column, "")) for column in LEADERBOARD_COLUMNS)
            + " |"
        )
    return "\n".join([header, divider, *body])


def write_leaderboard(rows: Sequence[Mapping[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    else:
        output_path.write_text(leaderboard_markdown(rows), encoding="utf-8")
    return output_path
