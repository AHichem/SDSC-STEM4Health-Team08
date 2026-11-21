"""Batch features: one row per batch, describing where it runs (level) and how it behaves (shape).

Inputs: long time series from brpmm.data (batch_id, t, one column per sensor).
Outputs: DataFrame indexed by batch_id, one column per feature, named "<sensor>__<feature>".
Depends on: config (feature settings), data (sensor_columns).

Why level + shape, and not sample-level rules: most variation in process A is between batches
(batch-to-batch spread 5-11x the within-batch spread for most sensors, data quality audit), so a
sample value is only normal or abnormal relative to its own batch. Process B shows it directly:
a faulty batch drops aeration from 7.4 to 5.2, while other accepted batches run at 4.8-5.2.
So every batch is described by
  - level: where it runs (median, start, end),
  - shape: how it moves relative to itself (deviation from its own start per stage of the batch,
    biggest step, biggest deviation, trend, spread),
  - relations: whether sensors still move together inside the batch (within-batch correlation).
Features are in sensor units; the methods scale them using the training batches only.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from brpmm.config import BASELINE_FRAC, BATCH_COL, EDGE_FRAC, N_SEGMENTS, N_WINDOWS
from brpmm.data import sensor_columns


def batch_features(ts: pd.DataFrame) -> pd.DataFrame:
    """One row of level + shape + relation features per batch, in the order batches appear."""
    sensors = sensor_columns(ts)
    rows = {}
    for batch_id, g in ts.groupby(BATCH_COL, sort=False):
        rows[batch_id] = _one_batch(g[sensors].to_numpy(dtype=float), sensors)
    feats = pd.DataFrame.from_dict(rows, orient="index")
    feats.index.name = BATCH_COL
    return feats


def _one_batch(X: np.ndarray, sensors: list[str]) -> dict[str, float]:
    n = len(X)
    frac = np.arange(n) / n  # position in the batch, 0..1: compares batches of different length
    out: dict[str, float] = {"batch__n_rows": float(n)}  # A: very short runs are mostly rejected

    # ---- level: where the batch runs ----
    edge = max(1, round(EDGE_FRAC * n))
    start, end = X[:edge].mean(axis=0), X[-edge:].mean(axis=0)
    median = np.median(X, axis=0)

    # ---- shape: the batch compared with itself ----
    baseline = X[: max(1, round(BASELINE_FRAC * n))].mean(axis=0)  # the batch's own early level
    # window means smooth sample noise so a step is a real step and not a single noisy sample;
    # array_split keeps the number of windows fixed whatever the batch length
    windows = np.array([w.mean(axis=0) for w in np.array_split(X, min(N_WINDOWS, n))])
    dev = windows - baseline
    max_dev = np.abs(dev).max(axis=0)
    max_step = (
        np.abs(np.diff(windows, axis=0)).max(axis=0) if len(windows) > 1 else np.zeros(X.shape[1])
    )
    std = X.std(axis=0)
    fc = frac - frac.mean()
    slope = (fc @ (X - X.mean(axis=0))) / (fc @ fc) if n > 1 else np.zeros(X.shape[1])
    seg = np.minimum((frac * N_SEGMENTS).astype(int), N_SEGMENTS - 1)

    for j, s in enumerate(sensors):
        out[f"{s}__median"] = median[j]
        out[f"{s}__start"] = start[j]
        out[f"{s}__end"] = end[j]
        out[f"{s}__std"] = std[j]
        out[f"{s}__slope"] = slope[j]
        out[f"{s}__max_dev_from_start"] = max_dev[j]
        out[f"{s}__max_step"] = max_step[j]
        for k in range(N_SEGMENTS):
            vals = X[seg == k, j]
            # stage k minus the batch's own start: "how far had this sensor moved by then"
            out[f"{s}__seg{k}_vs_start"] = vals.mean() - baseline[j] if len(vals) else np.nan

    # ---- relations: do sensors still move together inside this batch? ----
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False) if n > 2 else np.full((len(sensors),) * 2, np.nan)
    for i, j in combinations(range(len(sensors)), 2):
        # NaN when a sensor is constant inside the batch; methods impute it from training batches
        out[f"corr__{sensors[i]}__{sensors[j]}"] = corr[i, j]
    return out


def feature_sensor(name: str) -> str:
    """Sensor of a feature, for explanations: 'aeration_rate__max_step' -> 'aeration_rate'."""
    if name.startswith("corr__"):
        return "corr: " + " & ".join(name.split("__")[1:])
    return name.split("__")[0]
