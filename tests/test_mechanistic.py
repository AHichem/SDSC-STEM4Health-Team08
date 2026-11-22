import numpy as np
import pandas as pd
import pytest

from brpmm.mechanistic import MECHANISTIC, VOLATILE, mechanistic_features, volatility_features


def batch(name: str, n: int = 200, seed: int = 0, spike: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {
        s: 10 + rng.normal(0, 0.1, n)
        for s in (
            "flow_rate_1",
            "flow_rate_2",
            "pressure_1",
            "pressure_2",
            "pressure_3",
            "pressure_4",
            "temperature_1",
            "temperature_2",
            "temperature_3",
            "temperature_4",
            "density",
        )
    }
    d["flow_rate_1"] = 20 + rng.normal(0, 0.1, n)  # flow / pressure = 2
    if spike:
        d["temperature_1"][50:70] += 5  # a 10 % excursion
    return pd.DataFrame({"batch_id": name, "t": np.arange(n), **d})


def test_mechanistic_features_are_physical_combinations():
    f = mechanistic_features(batch("B1"))
    assert list(f.columns) == [f"{m}__{k}" for k in ("median", "std") for m in MECHANISTIC]
    assert f.loc["B1", "flow_per_pressure__median"] == pytest.approx(2.0, abs=0.02)
    assert f.loc["B1", "dp_3_2__median"] == pytest.approx(0.0, abs=0.05)


def test_volatility_measures_noise_and_excursions():
    ts = pd.concat([batch("calm"), batch("spiky", spike=True)], ignore_index=True)
    f = volatility_features(ts)
    assert set(f.columns) == {f"{s}__{k}" for s in VOLATILE for k in ("noise", "excursion")}
    assert f.loc["calm", "temperature_1__excursion"] < 0.02
    assert f.loc["spiky", "temperature_1__excursion"] == pytest.approx(0.10, abs=0.01)
