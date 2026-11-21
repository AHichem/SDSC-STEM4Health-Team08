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


# ---- Evaluation and model settings (used by features / methods / evaluate / report) ----
SEED = 42  # one seed for CV splits and the random forest, so reruns give identical numbers
CV_FOLDS = 5  # 5 folds: exactly 1 of B's 5 rejected batches per held-out fold; A keeps 8
CV_REPEATS = 5  # repeating with new shuffles averages out the luck of which batches share a fold
CV_INNER_FOLDS = 3  # inner CV inside the training part, only to choose the accept/reject cut-off
NOISE_F1 = 0.03  # with ~160 batches, F1 differences below this are probably noise

# batch features
EDGE_FRAC = 0.05  # start / end level = mean of the first / last 5 % of the batch
# a batch's own baseline = mean of its first 10 %; shape features are measured against it
BASELINE_FRAC = 0.10
N_SEGMENTS = 10  # stages of the batch (tenths of its duration) for "how far has it moved by then"
# window means for steps / max deviation: smooth noise, keep real steps (B: 50 rows each)
N_WINDOWS = 100

# methods
# MSPC: keep components explaining 90 % of normal variation, the rest goes to SPE
PCA_VARIANCE = 0.90
LOGREG_C = 0.05  # strong L2 regularisation: far more features than batches
RF_N_TREES = 300  # enough trees for stable probabilities on ~130 training batches
RF_MIN_SAMPLES_LEAF = 3  # no rule built on a single batch
TOP_REASONS = 3  # reasons reported per batch
# PCA + per-feature limits: total false-alarm risk shared over all per-feature limits (Bonferroni)
BONFERRONI_ALPHA = 0.05

# SVM (iteration of 21 Nov on process A, median + std per sensor, nested CV F1 0.914)
# soft margin: C 0.3-0.5 all score 0.92-0.93; smaller C = smoother boundary, less overfit
SVM_C = 0.3
SVM_GAMMA_MULT = 2.0  # rbf width = 2 / n_features; x1 and x2 both good, x4 overfits
OCSVM_NU = 0.05  # one-class SVM: ~5 % of the accepted training batches allowed outside the boundary
OCSVM_GAMMA_MULT = 0.5  # wider kernel than the classifier: only accepted batches to learn from
ENSEMBLE_KNN_K = 5  # kNN on PCA components inside the ensemble: k 5 scored best of 5, 11
