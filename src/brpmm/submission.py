"""Build, validate and submit leaderboard predictions.

Adapted from ``Leaderboard_template_project2.ipynb`` provided by the organisers.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from brpmm.config import BATCH_COL, LABEL_ACCEPTED, LABEL_REJECTED, OUTPUTS_DIR, data_dir

BASE_URL = (
    "https://leaderboard-project2-runai-innovation-thibaut.inference.compute.datascience.ch/api/v1"
)
SUBMISSION_COLUMNS = [BATCH_COL, "project", "label"]


def load_template(root: Path | None = None) -> pd.DataFrame:
    return pd.read_csv((root or data_dir()) / "BRPMM_template.csv", dtype={BATCH_COL: str})


def build_submission(predictions: dict[str, pd.Series], template: pd.DataFrame) -> pd.DataFrame:
    """Fill the template with per-process predictions.

    ``predictions`` maps ``"A"``/``"B"`` to a Series indexed by batch_id with
    ``"accepted"``/``"rejected"`` values.
    """
    labels = pd.concat(predictions.values())
    sub = template[[BATCH_COL, "project"]].copy()
    sub["label"] = sub[BATCH_COL].map(labels)
    validate_submission(sub, template)
    return sub


def validate_submission(sub: pd.DataFrame, template: pd.DataFrame) -> None:
    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Columns must be {SUBMISSION_COLUMNS}, got {list(sub.columns)}")
    missing = set(template[BATCH_COL]) - set(sub[BATCH_COL])
    if missing:
        raise ValueError(f"{len(missing)} batches missing, e.g. {sorted(missing)[:5]}")
    if sub[BATCH_COL].duplicated().any():
        raise ValueError("Duplicate batch_id in submission")
    bad = ~sub["label"].isin([LABEL_ACCEPTED, LABEL_REJECTED])
    if bad.any():
        examples = sub.loc[bad, BATCH_COL].head().tolist()
        raise ValueError(f"{bad.sum()} invalid/missing labels, e.g. {examples}")


def save_submission(sub: pd.DataFrame, name: str) -> Path:
    """Write a timestamped copy to outputs/ so every submission is traceable."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.csv"
    sub.to_csv(path, index=False)
    return path


def get_leaderboard() -> pd.DataFrame:
    leaderboard = requests.get(f"{BASE_URL}/leaderboard", timeout=30).json()
    return pd.json_normalize(leaderboard["entries"])


def submit(sub: pd.DataFrame, user: str | None = None, token: str | None = None) -> dict:
    """Submit predictions; credentials default to BRPMM_TEAM / BRPMM_TOKEN from .env."""
    user = user or os.getenv("BRPMM_TEAM")
    if not user:
        raise RuntimeError("No team name: pass user=... or set BRPMM_TEAM in .env")
    token = token if token is not None else os.getenv("BRPMM_TOKEN", "")
    payload = {"predictions": sub.to_dict(orient="records")}
    response = requests.post(f"{BASE_URL}/submit", json=payload, auth=(user, token), timeout=60)
    if response.status_code // 100 != 2:
        raise RuntimeError(f"Unexpected error (HTTP {response.status_code}): {response.text}")
    return response.json()["metrics"]
