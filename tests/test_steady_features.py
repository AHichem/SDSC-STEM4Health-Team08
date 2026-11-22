import numpy as np
import pandas as pd
import pytest

from brpmm.preprocessing import SegmentParams
from brpmm.steady_features import batch_features, sensor_of, to_model_space

PARAMS = SegmentParams(trim_start=10, trim_end=10, min_steady=20)


def batch(name: str, x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"batch_id": name, "t": np.arange(len(x)), "x": x})


def test_level_and_drift_are_recovered():
    t = np.arange(1020)
    ts = batch("B", 3.0 + 0.002 * t)  # +2 per 1000 steps
    f = batch_features(ts, PARAMS).loc["B"]
    assert f["x__slope"] == pytest.approx(2.0)
    assert f["x__median"] == pytest.approx(3.0 + 0.002 * np.median(t[10:-10]))


def test_white_noise_vs_smooth_signal_autocorrelation():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=2000)
    smooth = np.convolve(noise, np.ones(50) / 50, mode="same")
    ts = pd.concat([batch("noise", noise), batch("smooth", smooth)], ignore_index=True)
    f = batch_features(ts, PARAMS)
    assert abs(f.loc["noise", "x__ac1"]) < 0.1
    assert f.loc["smooth", "x__ac1"] > 0.9


def test_features_do_not_depend_on_batch_length():
    rng = np.random.default_rng(1)
    ts = pd.concat(
        [batch("short", rng.normal(5, 1, 500)), batch("long", rng.normal(5, 1, 3000))],
        ignore_index=True,
    )
    f = batch_features(ts, PARAMS)
    assert f.loc["short", "x__median"] == pytest.approx(f.loc["long", "x__median"], abs=0.15)
    assert f.loc["short", "x__std"] == pytest.approx(f.loc["long", "x__std"], abs=0.1)


def test_excluded_rows_are_left_out_and_counted():
    x = np.ones(200)
    x[100:110] = 1e6
    ts = batch("B", x)
    exclude = pd.Series(ts.t.between(100, 109), index=ts.index)
    f = batch_features(ts, PARAMS, exclude=exclude).loc["B"]
    assert f["x__std"] == pytest.approx(0.0)
    assert f["n_excluded_rows"] == 10


def test_model_space_transforms():
    feats = pd.DataFrame(
        {
            "short_batch": [0],
            "n_excluded_rows": [0],
            "x__std": [np.e],
            "x__ac1": [0.0],
            "x__median": [7.0],
        }
    )
    out = to_model_space(feats)
    assert "short_batch" not in out
    assert out.loc[0, "x__std"] == pytest.approx(1.0)
    assert out.loc[0, "x__median"] == 7.0
    assert sensor_of("temperature_1__std") == "temperature_1"
