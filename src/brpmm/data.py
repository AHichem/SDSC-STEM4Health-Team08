"""Load the raw challenge data.

Each CSV is in *long* format: one row per time step, rows of a batch are
contiguous and in time order, and there is no explicit timestamp. We add a
per-batch time index ``t`` (0, 1, 2, ...) so time-aware methods have it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from brpmm.config import BATCH_COL, LABEL_ACCEPTED, LABEL_REJECTED, Process, data_dir

TIME_COL = "t"

# Columns that never vary in process B (simulator set-points); they carry no signal.
CONSTANT_COLUMNS: dict[Process, list[str]] = {
    "A": [],
    "B": [
        "hot_cold_switch",
        "feed_substate_concentration",
        "cooling_water_temperature",
        "heating_water_temperature",
    ],
}


def _process_dir(process: Process, root: Path | None = None) -> Path:
    return (root or data_dir()) / f"process {process}"


def _read_timeseries(path: Path, drop_constant: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={BATCH_COL: str})
    df = df.drop(columns=[c for c in drop_constant if c in df.columns])
    df.insert(1, TIME_COL, df.groupby(BATCH_COL, sort=False).cumcount())
    return df


def load_calibration(
    process: Process, root: Path | None = None, drop_constant: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(timeseries, labels)`` for the labelled calibration set.

    ``labels`` is indexed by batch_id with values ``"accepted"`` / ``"rejected"``.
    """
    pdir = _process_dir(process, root) / "calibration data"
    drop = CONSTANT_COLUMNS[process] if drop_constant else []
    ts = _read_timeseries(pdir / "calibration_data.csv", drop)
    labels = pd.read_csv(pdir / "calibration_labels.csv", dtype={BATCH_COL: str})
    labels = labels.set_index(BATCH_COL)["label"]

    unexpected = set(labels.unique()) - {LABEL_ACCEPTED, LABEL_REJECTED}
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")
    if set(ts[BATCH_COL]) != set(labels.index):
        raise ValueError("Calibration batches and labels do not match")
    return ts, labels


def load_test(
    process: Process, root: Path | None = None, drop_constant: bool = True
) -> pd.DataFrame:
    """Return the unlabelled test time series."""
    path = _process_dir(process, root) / "test data" / "test_data.csv"
    return _read_timeseries(path, CONSTANT_COLUMNS[process] if drop_constant else [])


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """All measured variables (everything except batch id and time index)."""
    return [c for c in df.columns if c not in (BATCH_COL, TIME_COL)]


def to_binary(labels: pd.Series) -> pd.Series:
    """Encode labels as 1 = rejected (the positive / alarm class), 0 = accepted."""
    return (labels == LABEL_REJECTED).astype(int)


def from_binary(y: pd.Series) -> pd.Series:
    """Inverse of :func:`to_binary`."""
    return y.map({1: LABEL_REJECTED, 0: LABEL_ACCEPTED})
