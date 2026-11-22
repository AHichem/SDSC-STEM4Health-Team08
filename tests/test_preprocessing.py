import numpy as np
import pandas as pd

from brpmm.preprocessing import (
    END,
    START,
    STEADY,
    SegmentParams,
    gross_outlier_mask,
    outlier_scale,
    quality_report,
    segment_batches,
)

PARAMS = SegmentParams(trim_start=10, trim_end=10, min_steady=20)


def make_ts(lengths: dict[str, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame(
            {"batch_id": b, "t": np.arange(n), "x": rng.normal(0, 1, n), "y": rng.normal(5, 1, n)}
        )
        for b, n in lengths.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_segments_use_fixed_trims():
    ts = make_ts({"B1": 100})
    seg = segment_batches(ts, PARAMS)
    assert (seg[:10] == START).all()
    assert (seg[10:90] == STEADY).all()
    assert (seg[90:] == END).all()


def test_short_batch_falls_back_to_quarters():
    ts = make_ts({"S": 32})  # 32 < 10 + 10 + 20
    seg = segment_batches(ts, PARAMS).value_counts()
    assert seg[START] == 8
    assert seg[END] == 8
    assert seg[STEADY] == 16
    assert quality_report(ts, PARAMS).loc["S", "short_batch"]


def test_dropout_is_flagged_but_noise_is_not():
    # scale is learned on clean calibration batches and reused, as in the pipeline
    scale = outlier_scale(make_ts({f"C{i}": 400 for i in range(10)}, seed=1), PARAMS)
    ts = make_ts({"B1": 400, "B2": 400})
    hit = ts.index[(ts.batch_id == "B1") & ts.t.between(200, 219)]
    ts.loc[hit, "x"] = -500.0  # 20-row dropout
    mask = gross_outlier_mask(ts, PARAMS, scale)
    assert mask[hit].all()
    assert not mask[ts.batch_id == "B2"].any()
    report = quality_report(ts, PARAMS, scale)
    assert report.loc["B1", "n_gross_outlier_rows"] == 20
    assert report.loc["B2", "n_gross_outlier_rows"] == 0


def test_transients_are_never_flagged():
    ts = make_ts({"B1": 400, "B2": 400})
    ts.loc[(ts.batch_id == "B1") & (ts.t < 5), "x"] = 1000.0
    assert not gross_outlier_mask(ts, PARAMS).any()
