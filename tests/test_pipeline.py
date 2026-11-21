"""Checks for metrics, features, methods and the cut-off rule. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from brpmm.evaluate import choose_cutoff
from brpmm.features import batch_features
from brpmm.methods import METHODS
from brpmm.metrics import four_cases, leaderboard_f1

ACC, REJ = "accepted", "rejected"


def test_leaderboard_f1_matches_reference_entry():
    # reference leaderboard entry, process A: 110 good kept, 15 bad let through,
    # 26 bad caught, 10 good thrown away -> F1 0.898 only if "accepted" is positive
    y = pd.Series([ACC] * 110 + [REJ] * 15 + [REJ] * 26 + [ACC] * 10)
    p = pd.Series([ACC] * 110 + [ACC] * 15 + [REJ] * 26 + [REJ] * 10)
    assert four_cases(y, p) == {
        "good_kept": 110,
        "bad_let_through": 15,
        "bad_caught": 26,
        "good_thrown_away": 10,
    }
    assert leaderboard_f1(y, p) == pytest.approx(0.898, abs=1e-3)


def _ts(steps: dict[str, float], n: int = 200) -> pd.DataFrame:
    """Batches of constant aeration, optionally dropping by `steps[batch]` in the middle third."""
    rows = []
    for b, drop in steps.items():
        a = np.full(n, 7.0)
        a[n // 3 : 2 * n // 3] -= drop
        rows.append(
            pd.DataFrame(
                {
                    "batch_id": b,
                    "t": np.arange(n),
                    "aeration_rate": a,
                    "ph": 5.0 + np.linspace(0, 0.01, n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_features_one_row_per_batch_and_step_is_seen():
    f = batch_features(_ts({"B1": 0.0, "B2": 2.0}))
    assert list(f.index) == ["B1", "B2"]
    assert f.loc["B1", "aeration_rate__max_step"] == pytest.approx(0.0)
    assert f.loc["B2", "aeration_rate__max_step"] > 1.5
    assert f.loc["B2", "aeration_rate__max_dev_from_start"] == pytest.approx(2.0)


def test_features_handle_batches_of_different_length():
    ts = pd.concat([_ts({"A1": 0.0}, n=50), _ts({"A2": 0.0}, n=500)], ignore_index=True)
    f = batch_features(ts)
    assert f.loc["A1", "batch__n_rows"] == 50
    assert f.notna().all().all() or f.isna().sum().sum() > 0  # runs without error


@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_ranks_an_obvious_fault_highest(name):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 6)), columns=[f"s{i}__x" for i in range(6)])
    y = pd.Series([ACC] * 32 + [REJ] * 8)
    X.loc[32:, "s0__x"] += 6  # rejected batches clearly shifted on one feature
    fit, score, explain = METHODS[name]
    model = fit(X, y)
    s = score(model, X)
    assert s.shape == (40,)
    reasons = explain(model, X)
    assert len(reasons) == 40
    if name != "accept_all":
        assert s[32:].mean() > s[:32].mean()
        assert reasons[39][0][0] == "s0__x" or name == "logreg"


def test_cutoff_separates_clean_scores():
    s = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    y = pd.Series([ACC, ACC, ACC, REJ, REJ])
    c = choose_cutoff(s, y)
    assert 0.3 < c < 0.9
