"""Signal preprocessing for batch time series: segmentation and data-quality checks.

Process A batches have a start-up and a shut-down transient around a steady
middle section (measured in the process A EDA: slow sensors sit ~10 robust sd
from their steady level at step 0 and settle after ~150 steps; the end drifts
over the last ~150 steps). Statistics are computed per segment so transients
don't masquerade as process noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brpmm.config import BATCH_COL, Process
from brpmm.data import TIME_COL, sensor_columns

SEGMENT_COL = "segment"
START, STEADY, END = "start", "steady", "end"

# Gross outliers (sensor dropout, stoppage) are judged in the steady segment
# against a centred rolling median, so slow drift is not flagged. A 151-step
# window tolerates dropouts up to ~75 consecutive rows; A_239 has 62. Each sensor
# is scaled by its *typical 99th-percentile* residual (sensors differ: temperature_4
# routinely reaches 13-59x its median noise), and a row is gross when it exceeds
# GROSS_OUTLIER_Z times that. A_239's dropout sits at 200-370x median noise.
ROLLING_WINDOW = 151
TAIL_QUANTILE = 0.99
GROSS_OUTLIER_Z = 10.0
MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class SegmentParams:
    trim_start: int  # steps treated as start-up transient
    trim_end: int  # steps treated as shut-down transient
    min_steady: int  # below this many remaining steps, fall back to quarters


SEGMENT_PARAMS: dict[Process, SegmentParams] = {
    "A": SegmentParams(trim_start=150, trim_end=150, min_steady=100),
}


def segment_batches(ts: pd.DataFrame, params: SegmentParams) -> pd.Series:
    """Label every row ``start`` / ``steady`` / ``end``.

    Batches too short for the fixed trims (e.g. A_232, 196 steps) use the
    first quarter / middle half / last quarter instead.
    """
    n = ts.groupby(BATCH_COL, sort=False)[TIME_COL].transform("size")
    t = ts[TIME_COL]
    short = n < params.trim_start + params.trim_end + params.min_steady
    start_len = np.where(short, n // 4, params.trim_start)
    end_len = np.where(short, n // 4, params.trim_end)
    seg = np.where(t < start_len, START, np.where(t >= n - end_len, END, STEADY))
    return pd.Series(seg, index=ts.index, name=SEGMENT_COL)


def short_batches(ts: pd.DataFrame, params: SegmentParams) -> pd.Series:
    """Boolean per batch: True when the quarter fallback was used."""
    n = ts.groupby(BATCH_COL, sort=False).size()
    return n < params.trim_start + params.trim_end + params.min_steady


def _rolling_residual(ts: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    med = ts.groupby(BATCH_COL, sort=False)[sensors].transform(
        lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=1).median()
    )
    return ts[sensors] - med


def outlier_scale(ts: pd.DataFrame, params: SegmentParams) -> pd.Series:
    """Per sensor: median over batches of the TAIL_QUANTILE |rolling residual| in
    the steady segment. Fit on calibration, reuse on test."""
    sensors = sensor_columns(ts)
    steady = segment_batches(ts, params) == STEADY
    res = _rolling_residual(ts, sensors).abs()[steady]
    return res.groupby(ts.loc[steady, BATCH_COL], sort=False).quantile(TAIL_QUANTILE).median()


def gross_outlier_mask(
    ts: pd.DataFrame, params: SegmentParams, scale: pd.Series | None = None
) -> pd.Series:
    """True for steady-segment rows where any sensor is > GROSS_OUTLIER_Z x its
    tail scale off the rolling median. Start/end transients are never flagged."""
    sensors = sensor_columns(ts)
    scale = outlier_scale(ts, params) if scale is None else scale
    z = _rolling_residual(ts, sensors).abs() / scale[sensors].replace(0, np.nan)
    steady = segment_batches(ts, params) == STEADY
    return ((z > GROSS_OUTLIER_Z).any(axis=1) & steady).rename("gross_outlier")


def _max_identical_run(s: pd.Series) -> int:
    return int((s != s.shift()).cumsum().value_counts().max())


def quality_report(
    ts: pd.DataFrame, params: SegmentParams, scale: pd.Series | None = None
) -> pd.DataFrame:
    """One row per batch with data-quality indicators (no labels used)."""
    sensors = sensor_columns(ts)
    gross = gross_outlier_mask(ts, params, scale)
    g = ts.groupby(BATCH_COL, sort=False)
    return pd.DataFrame(
        {
            "n_steps": g.size(),
            "short_batch": short_batches(ts, params),
            "n_missing": ts[sensors].isna().any(axis=1).groupby(ts[BATCH_COL], sort=False).sum(),
            "n_gross_outlier_rows": gross.groupby(ts[BATCH_COL], sort=False).sum(),
            "max_identical_run": g[sensors].agg(_max_identical_run).max(axis=1),
        }
    )
