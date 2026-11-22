"""Process A alternative ensembles, used as evidence against the submitted model.

The leaderboard model is Maria's ``brpmm.methods.ensemble`` (``python -m brpmm.predict``).
This module holds the alternatives it was compared with, on the same splits, by
scripts/select_final_a.py and scripts/tune_process_a.py:

* teammate members re-implemented on 22 median+std features (svm22, pcaknn22,
  ocsvm22), combined by within-set rank averaging: F1 0.935 vs 0.941 submitted
* extra views (sample-level extra trees on raw samples; MiniRocket + ridge on the
  resampled trajectory), added with Maria's combination rule (``zscore_ensemble``):
  0.932-0.937, i.e. no gain
* batch-level extra trees and a one-class SVM on the 90 steady-state features

All members are deterministic (fixed seeds) and follow the FitPredict contract of
``brpmm.evaluation``: ``fit_predict(X_train, y_train, X_test)`` with batch_id as
the index of X, returning (score, decision). Every tunable value is in ``SETTINGS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from brpmm.config import (
    BATCH_COL,
    ENSEMBLE_KNN_K,
    OCSVM_GAMMA_MULT,
    OCSVM_NU,
    PCA_VARIANCE,
    SVM_C,
    SVM_GAMMA_MULT,
)
from brpmm.data import sensor_columns
from brpmm.evaluation import supervised
from brpmm.preprocessing import (
    SEGMENT_PARAMS,
    STEADY,
    gross_outlier_mask,
    outlier_scale,
    segment_batches,
)

SEED = 0

SETTINGS: dict[str, dict] = {
    # steady-state / sample / trajectory views
    "sample_et": {
        "every": 12,
        "aggregate": "q75",
        "n_estimators": 300,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
    "minirocket": {"length": 512, "transform": "mini", "ridge_log10_alphas": [-1, 4, 12]},
    "batch_et": {"n_estimators": 500, "min_samples_leaf": 2},
    "ocsvm": {"nu": 0.05, "limit_quantile": 0.95},
    # the submitted ensemble's settings (brpmm.config; gamma = mult / n_features)
    "svm22": {"C": SVM_C, "gamma_mult": SVM_GAMMA_MULT},
    "pcaknn22": {"pca_variance": PCA_VARIANCE, "k": ENSEMBLE_KNN_K},
    "ocsvm22": {"nu": OCSVM_NU, "gamma_mult": OCSVM_GAMMA_MULT},
}
ROCKET_LENGTH = SETTINGS["minirocket"]["length"]  # steady segment resampled to this many points


# --------------------------------------------------------------------------- data views
def steady_rows(ts: pd.DataFrame, scale: pd.Series | None = None) -> pd.DataFrame:
    """Steady-segment rows without gross outliers. Pass the calibration ``scale`` for test."""
    p = SEGMENT_PARAMS["A"]
    scale = outlier_scale(ts, p) if scale is None else scale
    keep = (segment_batches(ts, p) == STEADY) & ~gross_outlier_mask(ts, p, scale)
    return ts[keep]


def sample_table(
    st: pd.DataFrame, every: int = SETTINGS["sample_et"]["every"]
) -> dict[str, np.ndarray]:
    """Every ``every``-th steady sample of each batch, raw sensor values."""
    sensors = sensor_columns(st)
    keep = st[st.groupby(BATCH_COL).cumcount() % every == 0]
    return {b: d[sensors].to_numpy(np.float32) for b, d in keep.groupby(BATCH_COL)}


def resampled_cube(st: pd.DataFrame, ids, length: int = ROCKET_LENGTH) -> np.ndarray:
    """(batches, sensors, length): each steady segment linearly interpolated to ``length``."""
    sensors = sensor_columns(st)
    by = {b: d[sensors].to_numpy() for b, d in st.groupby(BATCH_COL)}
    grid = np.linspace(0, 1, length)
    cube = np.empty((len(ids), len(sensors), length))
    for i, b in enumerate(ids):
        v = by[b]
        x = np.linspace(0, 1, len(v))
        cube[i] = np.stack([np.interp(grid, x, v[:, j]) for j in range(len(sensors))])
    return cube


# --------------------------------------------------------------------------- members
AGGREGATORS = {
    "mean": np.mean,
    "q75": lambda p: np.quantile(p, 0.75),
    "q90": lambda p: np.quantile(p, 0.9),
}


def sample_level(rows: dict[str, np.ndarray], make, agg: str = "q75"):
    """Train on samples (each carries its batch's label), decide per batch."""

    def fit_predict(X_tr, y_tr, X_te):
        Xs = np.vstack([rows[b] for b in X_tr.index])
        ys = np.concatenate(
            [np.full(len(rows[b]), yy) for b, yy in zip(X_tr.index, y_tr, strict=True)]
        )
        model = make().fit(Xs, ys)
        s = np.array([AGGREGATORS[agg](model.predict_proba(rows[b])[:, 1]) for b in X_te.index])
        return s, (s >= 0.5).astype(int)

    return fit_predict


def rocket(cube_of, transform: str = "mini", log10_alphas=(-1, 4, 12)):
    """MiniRocket / MultiRocket features of the resampled batch, ridge classifier on top.

    Channels are standardised with training-batch statistics (not per batch), so
    level differences between batches are kept.
    """

    def fit_predict(X_tr, y_tr, X_te):
        from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

        A, B = cube_of(X_tr.index), cube_of(X_te.index)
        mu = A.mean(axis=(0, 2), keepdims=True)
        sd = A.std(axis=(0, 2), keepdims=True) + 1e-9
        A, B = (A - mu) / sd, (B - mu) / sd
        tf = (MiniRocket if transform == "mini" else MultiRocket)(random_state=SEED)
        Fa, Fb = tf.fit_transform(A), tf.transform(B)
        sc = StandardScaler().fit(Fa)
        clf = RidgeClassifierCV(alphas=np.logspace(*log10_alphas), class_weight="balanced")
        clf.fit(sc.transform(Fa), y_tr)
        s = clf.decision_function(sc.transform(Fb))
        return s, (s >= 0).astype(int)

    return fit_predict


def ocsvm(nu: float = 0.05, alpha: float = 0.95):
    """One-class SVM fitted on accepted batches only."""

    def fit_predict(X_tr, y_tr, X_te):
        sc = StandardScaler().fit(X_tr[y_tr == 0])
        m = OneClassSVM(nu=nu, gamma="scale").fit(sc.transform(X_tr[y_tr == 0]))
        s = -m.score_samples(sc.transform(X_te))
        lim = np.quantile(-m.score_samples(sc.transform(X_tr[y_tr == 0])), alpha)
        return s, (s > lim).astype(int)

    return fit_predict


def batch_extra_trees(n_estimators: int = 500, min_samples_leaf: int = 2):
    return supervised(
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    )


def sample_extra_trees(n_estimators: int = 300, min_samples_leaf: int = 20, max_features="sqrt"):
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


# --------------------------------------------------------------------------- ensemble
def member_ranks(members: dict, X_tr, y_tr, X_te) -> pd.DataFrame:
    """Each member's score as a percentile rank within X_te (columns = member names)."""
    return pd.DataFrame(
        {
            n: pd.Series(fp(X_tr, y_tr, X_te)[0]).rank(pct=True).to_numpy()
            for n, fp in members.items()
        },
        index=X_te.index,
    )


def rank_average(members: dict):
    """Equal-weight average of within-set percentile ranks; no fitted weights.

    Ranks are taken within the scored set, so the decision threshold assumes the
    scored set has a reject share similar to calibration (true for the leaderboard
    test set; for single batches in production use the accepted-only monitor).
    """

    def fit_predict(X_tr, y_tr, X_te):
        s = member_ranks(members, X_tr, y_tr, X_te).mean(axis=1).to_numpy()
        return s, (s >= 0.75).astype(int)

    return fit_predict


def ensemble_members(rows: dict[str, np.ndarray], cube_of, settings: dict = SETTINGS) -> dict:
    se, mr = settings["sample_et"], settings["minirocket"]
    tree_kw = {k: se[k] for k in ("n_estimators", "min_samples_leaf", "max_features")}
    return {
        "sample_et": sample_level(rows, lambda: sample_extra_trees(**tree_kw), se["aggregate"]),
        "minirocket": rocket(cube_of, mr["transform"], tuple(mr["ridge_log10_alphas"])),
        "batch_et": batch_extra_trees(**settings["batch_et"]),
        "ocsvm": ocsvm(settings["ocsvm"]["nu"], settings["ocsvm"]["limit_quantile"]),
    }


# --------------------------------------------------------------------------- teammate's members
# Re-implemented from docs/model_results.md (A17-A19). Inputs: median and std of each
# sensor over the whole batch (22 features).
def median_std_features(ts: pd.DataFrame) -> pd.DataFrame:
    sensors = sensor_columns(ts)
    f = ts.groupby(BATCH_COL)[sensors].agg(["median", "std"])
    f.columns = [f"{s}_{a}" for s, a in f.columns]
    return f


def on_columns(fp, cols):
    """Run a member on a column subset of a shared feature frame."""
    return lambda X_tr, y_tr, X_te: fp(X_tr[cols], y_tr, X_te[cols])


def team_members(cols22: list[str], settings: dict = SETTINGS) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    g = 1.0 / len(cols22)
    sv, pk, oc = settings["svm22"], settings["pcaknn22"], settings["ocsvm22"]

    def svm22(X_tr, y_tr, X_te):
        m = make_pipeline(
            StandardScaler(), SVC(C=sv["C"], gamma=sv["gamma_mult"] * g, class_weight="balanced")
        )
        s = m.fit(X_tr, y_tr).decision_function(X_te)
        return s, (s > 0).astype(int)

    def pcaknn22(X_tr, y_tr, X_te):
        m = make_pipeline(
            StandardScaler(),
            PCA(pk["pca_variance"]),
            KNeighborsClassifier(pk["k"], weights="distance"),
        )
        s = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        return s, (s >= 0.5).astype(int)

    def ocsvm22(X_tr, y_tr, X_te):
        sc = StandardScaler().fit(X_tr[y_tr == 0])
        m = OneClassSVM(nu=oc["nu"], gamma=oc["gamma_mult"] * g).fit(sc.transform(X_tr[y_tr == 0]))
        s = -m.decision_function(sc.transform(X_te))
        return s, (s > 0).astype(int)

    return {
        n: on_columns(f, cols22)
        for n, f in (("svm22", svm22), ("pcaknn22", pcaknn22), ("ocsvm22", ocsvm22))
    }


def zscore_ensemble(members: dict, inner_splits: int = 3, seed: int = SEED):
    """Maria's combination rule (brpmm.methods.fit_ensemble) for any FitPredict members.

    Each member's score is standardised with the mean and sd of its out-of-fold scores
    on the training batches (inner CV), then the standardised scores are averaged. The
    scale comes from training data only, so a single batch can be scored on its own.
    """
    from sklearn.model_selection import StratifiedKFold

    def fit_predict(X_tr, y_tr, X_te):
        folds = list(
            StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(X_tr, y_tr)
        )
        z = []
        for fp in members.values():
            oof = np.empty(len(X_tr))
            for tr, te in folds:
                oof[te] = fp(X_tr.iloc[tr], y_tr[tr], X_tr.iloc[te])[0]
            s = fp(X_tr, y_tr, X_te)[0]
            z.append((s - oof.mean()) / (oof.std() + 1e-12))
        s = np.mean(z, axis=0)
        return s, (s > 0).astype(int)

    return fit_predict
