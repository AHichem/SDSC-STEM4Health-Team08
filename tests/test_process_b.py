import numpy as np
import pandas as pd

from brpmm.process_b import ProcessBModel


def batch(name, aeration, volume=100.0, dip=False, n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full(n, aeration) * (1 + rng.normal(0, 0.001, n))
    if dip:
        a[100:300] *= 0.7
    return pd.DataFrame(
        {
            "batch_id": name,
            "t": np.arange(n),
            "aeration_rate": a,
            "fermentation_volume": volume + rng.normal(0, 0.1, n),
            "substrate_concentration": 10 + rng.normal(0, 0.1, n),
            "biomass_concentration": 1 + rng.normal(0, 0.01, n),
        }
    )


def test_rule_decides_and_monitor_only_flags_for_review():
    cal = pd.concat(
        [batch(f"C{i}", 6 + i * 0.2, seed=i) for i in range(12)]
        + [batch("R0", 7.0, dip=True, seed=99)],
        ignore_index=True,
    )
    labels = pd.Series(
        ["accepted"] * 12 + ["rejected"], index=[*(f"C{i}" for i in range(12)), "R0"]
    )
    model = ProcessBModel().fit(cal, labels)
    test = pd.concat(
        [
            batch("fault", 7.0, dip=True, seed=5),
            batch("new_condition", 9.5, volume=120.0, seed=6),
            batch("normal", 6.5, seed=7),
        ],
        ignore_index=True,
    )
    out = model.predict(test)
    assert out.loc["fault", "label"] == "rejected"
    assert "consecutive samples" in out.loc["fault", "reason"]
    assert out.loc["new_condition", "label"] == "accepted"  # new operating point is not a fault
    assert "aeration_rate" in out.loc["new_condition", "start_outside_calibration"]
    assert out.loc["normal", "label"] == "accepted"
    # a monitor alarm on an accepted batch is a review flag; a rejected batch is never "review"
    assert out.loc["new_condition", "monitor_alarm"]
    assert out.loc["new_condition", "review"]
    assert not out.loc["fault", "review"]
    assert not out.loc["normal", "review"]
