"""Process A: mechanistic and volatility batch features, to extend the submitted ensemble.

Mechanistic signals are physically motivated combinations of sensors, computed per sample
and summarised per batch with the median and standard deviation, the same summaries the
submitted ensemble uses (so ``brpmm.methods`` can select them by their ``__median`` /
``__std`` suffix):

    flow_per_pressure   flow_rate_1 / pressure_1   resistance of the line (they correlate 0.93)
    flow_split          flow_rate_2 / flow_rate_1  how the flow divides
    dp_3_2, dp_2_4      pressure_3 - pressure_2, pressure_2 - pressure_4   pressure drops
    dt_2_1, dt_3_4      temperature_2 - temperature_1, temperature_3 - temperature_4
    mass_flow           density * flow_rate_1

Volatility features for the sensors that drive rejections in the error analysis
(temperature_1, flow_rate_1, pressure_1, temperature_4, density):

    <sensor>__noise     sd of the first difference (short-term volatility)
    <sensor>__excursion share of samples beyond the batch median +/- 3 robust sd

No labels are used; every feature is computed from one batch alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brpmm.config import BATCH_COL

MECHANISTIC = {
    "flow_per_pressure": lambda d: d["flow_rate_1"] / d["pressure_1"],
    "flow_split": lambda d: d["flow_rate_2"] / d["flow_rate_1"],
    "dp_3_2": lambda d: d["pressure_3"] - d["pressure_2"],
    "dp_2_4": lambda d: d["pressure_2"] - d["pressure_4"],
    "dt_2_1": lambda d: d["temperature_2"] - d["temperature_1"],
    "dt_3_4": lambda d: d["temperature_3"] - d["temperature_4"],
    "mass_flow": lambda d: d["density"] * d["flow_rate_1"],
}
VOLATILE = ("temperature_1", "flow_rate_1", "pressure_1", "temperature_4", "density")
MAD_TO_SD = 1.4826


def mechanistic_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Median and sd per batch of each mechanistic signal: ``<signal>__median/__std``."""
    signals = pd.DataFrame({name: f(ts) for name, f in MECHANISTIC.items()}, index=ts.index)
    signals = signals.replace([np.inf, -np.inf], np.nan)
    g = signals.groupby(ts[BATCH_COL], sort=False)
    out = pd.concat([g.median().add_suffix("__median"), g.std().add_suffix("__std")], axis=1)
    out.index.name = BATCH_COL
    return out


def _volatility(x: np.ndarray) -> tuple[float, float]:
    noise = float(np.std(np.diff(x))) if len(x) > 2 else np.nan
    med = np.median(x)
    band = 3 * MAD_TO_SD * np.median(np.abs(x - med))
    excursion = float(np.mean(np.abs(x - med) > band)) if band > 0 else 0.0
    return noise, excursion


def volatility_features(ts: pd.DataFrame, sensors=VOLATILE) -> pd.DataFrame:
    """``<sensor>__noise`` and ``<sensor>__excursion`` per batch for the volatile sensors."""
    rows = {}
    for batch, g in ts.groupby(BATCH_COL, sort=False):
        row = {}
        for s in sensors:
            row[f"{s}__noise"], row[f"{s}__excursion"] = _volatility(g[s].to_numpy(float))
        rows[batch] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = BATCH_COL
    return out
