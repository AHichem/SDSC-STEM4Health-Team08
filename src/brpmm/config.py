"""Paths and constants shared across the project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Process = Literal["A", "B"]
PROCESSES: tuple[Process, ...] = ("A", "B")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

load_dotenv(REPO_ROOT / ".env")

_DATA_DIR_CANDIDATES = (
    Path("/home/renku/work/input-data/Project 2 - BRPMM"),
    REPO_ROOT / "challenge_docs" / "input-data" / "Project 2 - BRPMM",
)

LABEL_ACCEPTED = "accepted"
LABEL_REJECTED = "rejected"
BATCH_COL = "batch_id"


def data_dir() -> Path:
    """Return the challenge data folder (``BRPMM_DATA_DIR`` overrides the defaults)."""
    if env := os.getenv("BRPMM_DATA_DIR"):
        path = Path(env)
        if not path.is_dir():
            raise FileNotFoundError(f"BRPMM_DATA_DIR={env!r} does not exist")
        return path
    for path in _DATA_DIR_CANDIDATES:
        if path.is_dir():
            return path
    tried = "\n  ".join(str(p) for p in _DATA_DIR_CANDIDATES)
    raise FileNotFoundError(f"Challenge data not found. Set BRPMM_DATA_DIR. Tried:\n  {tried}")
