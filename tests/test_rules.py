import numpy as np
import pandas as pd

from brpmm.rules import SetpointDeviationRule, longest_run


def batch(
    name: str, level: float, dip: tuple[int, int] | None = None, n: int = 2000
) -> pd.DataFrame:
    x = np.full(n, level) * (1 + np.random.default_rng(0).normal(0, 0.001, n))
    if dip:
        x[dip[0] : dip[1]] *= 0.7
    return pd.DataFrame({"batch_id": name, "t": np.arange(n), "aeration_rate": x})


def test_longest_run():
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == 3
    assert longest_run(np.zeros(5, bool)) == 0


def test_rule_uses_each_batch_own_level():
    ts = pd.concat(
        [
            batch("low_but_steady", 4.8),
            batch("high_with_fault", 8.0, (500, 1250)),
            batch("short_blip", 8.0, (900, 950)),
        ],
        ignore_index=True,
    )
    out = SetpointDeviationRule().evaluate(ts)
    assert not out.loc["low_but_steady", "rejected"]  # a low level alone is acceptable
    assert out.loc["high_with_fault", "rejected"]
    assert out.loc["high_with_fault", "longest_run"] == 750
    assert out.loc["high_with_fault", "first_step_beyond"] == 500
    assert not out.loc["short_blip", "rejected"]  # 50 samples < min_run
