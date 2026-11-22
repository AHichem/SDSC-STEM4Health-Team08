import numpy as np
import pandas as pd
import pytest

from brpmm.mspc import PCAMonitor, scaled_chi2_limit

LOADINGS = np.random.default_rng(42).normal(size=(2, 6))  # one fixed "process"


def correlated(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    X = latent @ LOADINGS + 0.1 * rng.normal(size=(n, 6))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])


@pytest.fixture
def monitor() -> PCAMonitor:
    return PCAMonitor(n_components=2, alpha=0.95).fit(correlated(400, 0))


def test_false_alarm_rate_close_to_design(monitor):
    rate = monitor.score(correlated(2000, 1))["alarm"].mean()
    assert 0.02 < rate < 0.15  # ~5% per statistic, two statistics


def test_broken_correlation_raises_spe(monitor):
    X = correlated(50, 2)
    X["f0"] += 3.0  # breaks the latent structure without moving far in T²
    assert monitor.score(X)["SPE_ratio"].median() > 1


def test_contributions_sum_to_statistics(monitor):
    X = correlated(10, 3)
    s = monitor.score(X)
    np.testing.assert_allclose(monitor.spe_contributions(X).sum(axis=1), s["SPE"], rtol=1e-8)
    np.testing.assert_allclose(monitor.t2_contributions(X).sum(axis=1), s["T2"], rtol=1e-8)


def test_box_limit_matches_chi2_for_chi2_data():
    v = np.random.default_rng(0).chisquare(4, 200_000)
    assert scaled_chi2_limit(v, 0.95) == pytest.approx(9.49, rel=0.02)


def test_invalid_limit_method():
    with pytest.raises(ValueError):
        PCAMonitor(limit_method="magic")
