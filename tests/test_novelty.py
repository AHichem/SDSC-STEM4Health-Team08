import numpy as np
import pandas as pd

from brpmm.novelty import KNNMonitor


def two_modes(n: int, seed: int) -> pd.DataFrame:
    """Normal data in two separate clusters, like process A's operating modes."""
    rng = np.random.default_rng(seed)
    centre = np.where(rng.random(n) < 0.5, -5.0, 5.0)[:, None]
    return pd.DataFrame(centre + rng.normal(size=(n, 3)), columns=list("abc"))


def test_both_modes_are_normal_but_the_gap_is_not():
    monitor = KNNMonitor(k=10, alpha=0.95).fit(two_modes(300, 0))
    normal = monitor.score(two_modes(1000, 1))
    assert normal["alarm"].mean() < 0.15
    between = monitor.score(pd.DataFrame(np.zeros((20, 3)), columns=list("abc")))
    assert between["alarm"].all()  # the midpoint of the two modes was never observed
