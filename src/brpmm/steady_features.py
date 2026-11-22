"""Batch-level features: one row per batch, independent of batch length.

Feature names are ``<sensor>__<stat>`` so results can be rolled up per sensor.

Per sensor (steady segment unless noted):
    median        operating level
    mad           robust spread (scaled to sd)
    std           spread incl. occasional excursions
    slope         drift, change per 1000 steps
    noise         short-term noise, sd(first difference) / sqrt(2)
    ac1           lag-1 autocorrelation (how smooth the signal is)
    start_offset  start-up segment mean minus steady median
    end_offset    shut-down segment mean minus steady median
Per batch:
    n_steps, short_batch
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brpmm.config import BATCH_COL
from brpmm.data import TIME_COL, sensor_columns
from brpmm.preprocessing import (
    END,
    MAD_TO_SD,
    START,
    STEADY,
    SegmentParams,
    segment_batches,
    short_batches,
)

SEP = "__"


def sensor_of(feature: str) -> str:
    """``"temperature_1__std"`` -> ``"temperature_1"``; batch-level features map to themselves."""
    return feature.split(SEP)[0]


def _suffix(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    return df.add_suffix(f"{SEP}{stat}")


def batch_features(
    ts: pd.DataFrame, params: SegmentParams, exclude: pd.Series | None = None
) -> pd.DataFrame:
    """``exclude``: boolean row mask (e.g. gross outliers) left out of all statistics;
    the per-batch count is kept as ``n_excluded_rows`` because upsets carry signal."""
    sensors = sensor_columns(ts)
    seg = segment_batches(ts, params)
    n_steps = ts.groupby(BATCH_COL, sort=False).size()
    short = short_batches(ts, params).astype(int)
    n_excluded = (
        exclude.groupby(ts[BATCH_COL], sort=False).sum()
        if exclude is not None
        else pd.Series(0, index=n_steps.index)
    )
    if exclude is not None:
        ts, seg = ts[~exclude], seg[~exclude]
    steady = ts[seg == STEADY]
    b = steady[BATCH_COL]
    g = steady.groupby(b, sort=False)[sensors]

    median = g.median()
    mad = (steady[sensors] - g.transform("median")).abs().groupby(b, sort=False).median()
    std = g.std()

    t = steady[TIME_COL] - steady.groupby(b, sort=False)[TIME_COL].transform("mean")
    x = steady[sensors] - g.transform("mean")
    tt = (t**2).groupby(b, sort=False).sum()
    slope = x.mul(t, axis=0).groupby(b, sort=False).sum().div(tt, axis=0) * 1000

    diff = g.diff()
    noise = diff.groupby(b, sort=False).std() / np.sqrt(2)
    # Pearson correlation over (x_t, x_{t-1}) pairs within each batch
    lag = g.shift()
    cur = steady[sensors].where(lag.notna())
    xc = cur - cur.groupby(b, sort=False).transform("mean")
    lc = lag - lag.groupby(b, sort=False).transform("mean")
    ac1 = (xc * lc).groupby(b, sort=False).sum() / np.sqrt(
        (xc**2).groupby(b, sort=False).sum() * (lc**2).groupby(b, sort=False).sum()
    )

    start = ts[seg == START].groupby(BATCH_COL, sort=False)[sensors].mean() - median
    end = ts[seg == END].groupby(BATCH_COL, sort=False)[sensors].mean() - median

    feats = pd.concat(
        [
            _suffix(median, "median"),
            _suffix(mad * MAD_TO_SD, "mad"),
            _suffix(std, "std"),
            _suffix(slope, "slope"),
            _suffix(noise, "noise"),
            _suffix(ac1, "ac1"),
            _suffix(start, "start_offset"),
            _suffix(end, "end_offset"),
        ],
        axis=1,
    )
    lead = pd.DataFrame(
        {"n_steps": n_steps, "short_batch": short, "n_excluded_rows": n_excluded.astype(int)}
    )
    feats = pd.concat([lead.reindex(feats.index), feats], axis=1)  # one concat, no fragmentation
    feats.index.name = BATCH_COL
    return feats


SCALE_STATS = ("mad", "std", "noise")


def to_model_space(feats: pd.DataFrame) -> pd.DataFrame:
    """Make features closer to Gaussian before PCA / linear models.

    log for spread statistics (strictly positive, right-skewed), log1p for counts,
    Fisher z for autocorrelations (bounded near 1). ``short_batch`` is dropped: it
    is a data-quality flag carried in the metadata, and ``n_steps`` keeps the information.
    """
    out = feats.drop(columns=["short_batch"], errors="ignore").copy()
    for col in out.columns:
        stat = col.split(SEP)[-1]
        if col == "n_excluded_rows":
            out[col] = np.log1p(out[col])
        elif stat in SCALE_STATS:
            out[col] = np.log(out[col].clip(lower=1e-9))
        elif stat == "ac1":
            out[col] = np.arctanh(out[col].clip(-0.999, 0.999))
    return out
