"""Process A: feature engineering and a time-series member inside the submitted ensemble.

    python scripts/improve_process_a.py                 # ~10 min -> outputs/process_A/improve/
    python scripts/improve_process_a.py --confirm A B   # re-test named variants on new splits

The submitted ensemble (brpmm.methods.ensemble: RBF SVM + kNN on PCA + one-class SVM,
scores standardised on training out-of-fold scores, cut-off chosen in inner CV) is kept
exactly as it is; only its input columns change, or a member is added:

  base            the 22 median/std features it was submitted with
  + mechanistic   median/std of flow/pressure, flow split, pressure drops, temperature
                  differences and mass flow (brpmm.mechanistic)
  + volatility    short-term noise and excursion share of the reject-driving sensors
  + ts member     MiniRocket + ridge on the resampled trajectories of those sensors only

Scored on the fresh splits (StratifiedKFold(5, shuffle, s), s = 50..53; base = 0.941);
--confirm re-scores on splits s = 60..63, never used before, to check a gain holds.
"""

from __future__ import annotations

import argparse
import json
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

import brpmm.methods as maria
from brpmm.config import LABEL_REJECTED, OUTPUTS_DIR
from brpmm.data import load_calibration, sensor_columns, to_binary
from brpmm.evaluation import FRESH_SEEDS, leaderboard_summary, submitted_ensemble_decisions
from brpmm.features import batch_features
from brpmm.mechanistic import MECHANISTIC, VOLATILE, mechanistic_features, volatility_features
from brpmm.models import resampled_cube, steady_rows

warnings.filterwarnings("ignore")
CONFIRM_SEEDS = (60, 61, 62, 63)
TOP_OF_LEADERBOARD = 0.939  # second place, F1(A)
OUT = OUTPUTS_DIR / "process_A" / "improve"


@contextmanager
def ensemble_columns(columns: list[str]):
    """The submitted ensemble reads its inputs through brpmm.methods._median_std_columns;
    point it at ``columns`` for the duration and always restore it."""
    original = maria._median_std_columns
    maria._median_std_columns = lambda X: [c for c in columns if c in X.columns]
    try:
        yield
    finally:
        maria._median_std_columns = original


@contextmanager
def extra_member(name: str, fit, score):
    maria._MEMBERS[name] = (fit, score)
    try:
        yield
    finally:
        del maria._MEMBERS[name]


def rocket_member(cube: np.ndarray, pos: dict):
    """MiniRocket + ridge on the given trajectories, as a (fit, score) pair for _MEMBERS."""

    def fit(X: pd.DataFrame, y: pd.Series) -> dict:
        from aeon.transformations.collection.convolution_based import MiniRocket

        A = cube[[pos[b] for b in X.index]]
        mu, sd = A.mean(axis=(0, 2), keepdims=True), A.std(axis=(0, 2), keepdims=True) + 1e-9
        tf = MiniRocket(random_state=0)
        F = tf.fit_transform((A - mu) / sd)
        sc = StandardScaler().fit(F)
        clf = RidgeClassifierCV(alphas=np.logspace(-1, 4, 12), class_weight="balanced")
        clf.fit(sc.transform(F), (pd.Series(y).to_numpy() == LABEL_REJECTED).astype(int))
        return {"mu": mu, "sd": sd, "tf": tf, "sc": sc, "clf": clf}

    def score(model: dict, X: pd.DataFrame) -> np.ndarray:
        A = (cube[[pos[b] for b in X.index]] - model["mu"]) / model["sd"]
        return model["clf"].decision_function(model["sc"].transform(model["tf"].transform(A)))

    return fit, score


def frame():
    ts, lab = load_calibration("A")
    X = batch_features(ts).join(mechanistic_features(ts)).join(volatility_features(ts))
    X = X.loc[lab.index]
    sensors = sensor_columns(ts)
    base = [f"{s}__{k}" for s in sensors for k in ("median", "std")]
    mech = [f"{m}__{k}" for m in MECHANISTIC for k in ("median", "std")]
    vol = [f"{s}__{k}" for s in VOLATILE for k in ("noise", "excursion")]
    st = steady_rows(ts)
    cube = resampled_cube(st[["batch_id", "t", *VOLATILE]], list(lab.index))
    pos = {b: i for i, b in enumerate(lab.index)}
    return X, lab, base, mech, vol, rocket_member(cube, pos)


def evaluate(variants: dict, X, lab, seeds) -> dict:
    y = to_binary(lab).to_numpy()
    out = {}
    for name, (columns, member) in variants.items():
        with ensemble_columns(columns):
            if member is None:
                dec = submitted_ensemble_decisions(X, lab, seeds)
            else:
                with extra_member("ts_volatile", *member):
                    dec = submitted_ensemble_decisions(X, lab, seeds)
        out[name] = leaderboard_summary(dec, y, TOP_OF_LEADERBOARD)
        print(f"  {name:44s} {out[name]}", flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--confirm", nargs="*", default=None, help="variant names to re-test")
    args = ap.parse_args(argv)
    X, lab, base, mech, vol, ts_member = frame()
    variants = {
        "base (submitted)": (base, None),
        "base + mechanistic": (base + mech, None),
        "base + volatility": (base + vol, None),
        "base + mechanistic + volatility": (base + mech + vol, None),
        "base + ts member (volatile sensors)": (base, ts_member),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if args.confirm is None:
        result = {"seeds": list(FRESH_SEEDS), **evaluate(variants, X, lab, FRESH_SEEDS)}
        path = OUT / "variants.json"
    else:
        keep = {k: v for k, v in variants.items() if k == "base (submitted)" or k in args.confirm}
        result = {"seeds": list(CONFIRM_SEEDS), **evaluate(keep, X, lab, CONFIRM_SEEDS)}
        path = OUT / "confirm.json"
    path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
