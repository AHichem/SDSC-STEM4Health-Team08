import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from brpmm.models import member_ranks, ocsvm, rank_average, resampled_cube, rocket, sample_level


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    ids = [f"A_{i:03d}" for i in range(40)]
    y = np.array([1 if i % 4 == 0 else 0 for i in range(40)])
    rows = {
        b: rng.normal(1.5 * yy, 1, (30, 3)).astype(np.float32) for b, yy in zip(ids, y, strict=True)
    }
    X = pd.DataFrame({"f": [r.mean() for r in rows.values()]}, index=ids)
    return X, y, rows


def test_sample_level_learns_from_samples(toy):
    X, y, rows = toy
    fp = sample_level(rows, lambda: ExtraTreesClassifier(50, random_state=0))
    s, _ = fp(X.iloc[:30], y[:30], X.iloc[30:])
    assert s[y[30:] == 1].mean() > s[y[30:] == 0].mean()


def test_rank_average_is_mean_of_member_ranks(toy):
    X, y, _ = toy
    members = {
        "a": lambda _Xt, _yt, Xe: (Xe["f"].to_numpy(), None),
        "b": lambda _Xt, _yt, Xe: (-Xe["f"].to_numpy(), None),
    }
    ranks = member_ranks(members, X, y, X)
    s, _ = rank_average(members)(X, y, X)
    np.testing.assert_allclose(s, ranks.mean(axis=1))
    assert np.allclose(s, s[0])  # opposite members cancel exactly


def test_ocsvm_ignores_rejected_training_batches(toy):
    X, y, _ = toy
    s1, _ = ocsvm()(X, y, X)
    X2 = X.copy()
    X2.loc[y == 1, "f"] += 100  # changing rejected training batches must not change the model
    s2, _ = ocsvm()(X2, y, X)
    np.testing.assert_allclose(s1, s2)


def test_resampled_cube_shape_and_endpoints():
    st = pd.DataFrame({"batch_id": ["B"] * 5, "t": range(5), "x": [0.0, 1, 2, 3, 4]})
    cube = resampled_cube(st, ["B"], length=9)
    assert cube.shape == (1, 1, 9)
    assert cube[0, 0, 0] == 0.0 and cube[0, 0, -1] == 4.0


def test_minirocket_member_runs():
    pytest.importorskip("aeon")
    rng = np.random.default_rng(1)
    ids = [f"b{i}" for i in range(24)]
    y = np.array([i % 3 == 0 for i in range(24)], int)
    cube = rng.normal(size=(24, 2, 64)) + y[:, None, None]
    pos = {b: i for i, b in enumerate(ids)}
    X = pd.DataFrame(index=ids, data={"dummy": 0.0})
    s, _ = rocket(lambda idx: cube[[pos[b] for b in idx]])(X.iloc[:18], y[:18], X.iloc[18:])
    assert s.shape == (6,)


def test_median_std_features_and_team_members():
    from brpmm.models import median_std_features, team_members

    rng = np.random.default_rng(3)
    ts = pd.DataFrame(
        {
            "batch_id": np.repeat([f"b{i}" for i in range(30)], 20),
            "t": np.tile(range(20), 30),
            "x": rng.normal(size=600),
            "z": rng.normal(size=600),
        }
    )
    f = median_std_features(ts)
    assert list(f.columns) == ["x_median", "x_std", "z_median", "z_std"]
    y = np.array([i % 3 == 0 for i in range(30)], int)
    f.loc[y == 1, "x_median"] += 3  # rejected batches differ
    for name, fp in team_members(list(f.columns)).items():
        s, _ = fp(f.iloc[:24], y[:24], f.iloc[24:])
        assert s.shape == (6,), name


def test_zscore_ensemble_scales_members_from_training_only(toy):
    from brpmm.models import zscore_ensemble

    X, y, _ = toy
    big = {
        "a": lambda _Xt, _yt, Xe: (1000 * Xe["f"].to_numpy(), None),
        "b": lambda _Xt, _yt, Xe: (Xe["f"].to_numpy(), None),
    }
    s, _ = zscore_ensemble(big)(X.iloc[:30], y[:30], X.iloc[30:])
    s_b, _ = zscore_ensemble({"b": big["b"]})(X.iloc[:30], y[:30], X.iloc[30:])
    np.testing.assert_allclose(s, s_b)  # a member's raw scale does not change its weight
    # the standardisation comes from training batches only: other scored batches don't matter
    s_part, _ = zscore_ensemble(big)(X.iloc[:30], y[:30], X.iloc[30:33])
    np.testing.assert_allclose(s[:3], s_part)
