"""The methods we compare. Each one is three plain functions:

    fit(X, y) -> model          X: batch features (DataFrame), y: labels "accepted"/"rejected"
    score(model, X) -> array    one number per batch, higher = more likely rejected
    explain(model, X) -> list   per batch, the top reasons as (feature, value) pairs

The accept/reject cut-off is NOT part of a method: evaluate.py chooses it the same way for all
of them, so the comparison is fair.

Depends on: config (SEED, method settings), features (feature names only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM

from brpmm.config import (
    BONFERRONI_ALPHA,
    CV_INNER_FOLDS,
    ENSEMBLE_KNN_K,
    LABEL_ACCEPTED,
    LABEL_REJECTED,
    LOGREG_C,
    OCSVM_GAMMA_MULT,
    OCSVM_NU,
    PCA_VARIANCE,
    RF_MIN_SAMPLES_LEAF,
    RF_N_TREES,
    SEED,
    SVM_C,
    SVM_GAMMA_MULT,
    TOP_REASONS,
)

Model = dict[str, Any]
Reasons = list[list[tuple[str, float]]]


# ---- shared helpers ---------------------------------------------------------------------------


def _impute(X: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Fill NaN (e.g. correlation of a sensor constant in that batch) with training medians."""
    return X[medians.index].fillna(medians).fillna(0.0).to_numpy(dtype=float)


def _robust_scale(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust spread (1.4826 * MAD ~ standard deviation for normal data).

    Median/MAD instead of mean/std so the odd batches we want to find don't widen the normal range.
    The floor stops a feature that is constant in the training batches from producing infinite
    z-scores on the tiniest numerical difference.
    """
    med = np.median(A, axis=0)
    scale = 1.4826 * np.median(np.abs(A - med), axis=0)
    floor = 1e-6 + 1e-3 * np.abs(med)
    return med, np.maximum(scale, floor)


def _top(values: np.ndarray, names: list[str], k: int = TOP_REASONS) -> Reasons:
    """Top-k features by |value| for every row."""
    idx = np.argsort(-np.abs(values), axis=1)[:, :k]
    return [[(names[j], float(values[i, j])) for j in row] for i, row in enumerate(idx)]


def _is_rejected(y: pd.Series) -> np.ndarray:
    return (pd.Series(y).to_numpy() == LABEL_REJECTED).astype(int)


# ---- 1. accept everything: the floor every method must beat ------------------------------------


def fit_accept_all(X: pd.DataFrame, y: pd.Series) -> Model:
    return {}


def score_accept_all(model: Model, X: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(X))


def explain_accept_all(model: Model, X: pd.DataFrame) -> Reasons:
    return [[] for _ in range(len(X))]


# ---- 2. per-feature limits (one-class): normal range learned from accepted batches only -------


def fit_feature_limits(X: pd.DataFrame, y: pd.Series) -> Model:
    medians = X.median()
    A = _impute(X[pd.Series(y).to_numpy() == LABEL_ACCEPTED], medians)
    center, scale = _robust_scale(A)
    return {"medians": medians, "center": center, "scale": scale, "names": list(medians.index)}


def _z(model: Model, X: pd.DataFrame) -> np.ndarray:
    return (_impute(X, model["medians"]) - model["center"]) / model["scale"]


def score_feature_limits(model: Model, X: pd.DataFrame) -> np.ndarray:
    # the worst feature decides: one clearly abnormal aspect is enough to reject a batch,
    # which matches A (each rejected batch fails on a different sensor) and is easy to explain
    return np.abs(_z(model, X)).max(axis=1)


def explain_feature_limits(model: Model, X: pd.DataFrame) -> Reasons:
    return _top(_z(model, X), model["names"])  # values = robust z (how many "normal spreads" away)


# ---- 3. PCA / MSPC (one-class): Hotelling T2 + SPE on accepted batches only --------------------


def fit_pca_mspc(X: pd.DataFrame, y: pd.Series) -> Model:
    medians = X.median()
    A = _impute(X[pd.Series(y).to_numpy() == LABEL_ACCEPTED], medians)
    mean = A.mean(axis=0)
    std = np.maximum(A.std(axis=0), 1e-6 + 1e-3 * np.abs(mean))
    Z = (A - mean) / std
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    var = s**2 / max(len(Z) - 1, 1)
    # keep components until PCA_VARIANCE of the normal variation is explained (usual MSPC choice);
    # the rest is left to SPE, which flags batches that break the usual relations between features
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), PCA_VARIANCE) + 1)
    model = {
        "medians": medians,
        "mean": mean,
        "std": std,
        "P": Vt[:k].T,
        "eig": var[:k],
        "names": list(medians.index),
    }
    t2, spe, _ = _pca_stats(model, X[pd.Series(y).to_numpy() == LABEL_ACCEPTED])
    # scale each statistic by its 95th percentile on the training accepted batches so the two can
    # be combined; the final cut-off is still chosen in evaluate.py
    model["lim_t2"] = max(np.quantile(t2, 0.95), 1e-12)
    model["lim_spe"] = max(np.quantile(spe, 0.95), 1e-12)
    return model


def _pca_stats(model: Model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = (_impute(X, model["medians"]) - model["mean"]) / model["std"]
    T = Z @ model["P"]
    resid = Z - T @ model["P"].T
    return (T**2 / model["eig"]).sum(axis=1), (resid**2).sum(axis=1), resid


def score_pca_mspc(model: Model, X: pd.DataFrame) -> np.ndarray:
    t2, spe, _ = _pca_stats(model, X)
    return np.maximum(t2 / model["lim_t2"], spe / model["lim_spe"])


def explain_pca_mspc(model: Model, X: pd.DataFrame) -> Reasons:
    """Contribution plot: each feature's share of T2 and SPE, each scaled by its limit.

    T2 share = z_j * sum_a(t_a * p_ja / lambda_a) (too far along the normal pattern);
    SPE share = residual_j ** 2 (breaks the normal pattern). Same units as the score.
    """
    _, _, resid = _pca_stats(model, X)
    Z = (_impute(X, model["medians"]) - model["mean"]) / model["std"]
    t2_share = Z * (((Z @ model["P"]) / model["eig"]) @ model["P"].T)
    contrib = t2_share / model["lim_t2"] + resid**2 / model["lim_spe"]
    return _top(contrib, model["names"])


# ---- 4. logistic regression (supervised, strongly regularised) ---------------------------------


def fit_logreg(X: pd.DataFrame, y: pd.Series) -> Model:
    medians = X.median()
    M = _impute(X, medians)
    mean = M.mean(axis=0)
    std = np.maximum(M.std(axis=0), 1e-6 + 1e-3 * np.abs(mean))
    # many more features than batches -> strong L2 regularisation (small C) to avoid memorising;
    # balanced class weights so the 25 % (A) / 12.5 % (B) rejected batches count as much as the rest
    clf = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    clf.fit((M - mean) / std, _is_rejected(y))
    return {"medians": medians, "mean": mean, "std": std, "clf": clf, "names": list(medians.index)}


def _logreg_inputs(model: Model, X: pd.DataFrame) -> np.ndarray:
    return (_impute(X, model["medians"]) - model["mean"]) / model["std"]


def score_logreg(model: Model, X: pd.DataFrame) -> np.ndarray:
    return model["clf"].predict_proba(_logreg_inputs(model, X))[:, 1]


def explain_logreg(model: Model, X: pd.DataFrame) -> Reasons:
    contrib = (
        _logreg_inputs(model, X) * model["clf"].coef_[0]
    )  # how much each feature pushes to "rejected"
    return _top(contrib, model["names"])


# ---- 5. random forest (supervised) --------------------------------------------------------------


def fit_random_forest(X: pd.DataFrame, y: pd.Series) -> Model:
    medians = X.median()
    M = _impute(X, medians)
    clf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,  # leaves of >= 3 batches: no rules built on one batch
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,  # one job: identical results on every rerun
    )
    clf.fit(M, _is_rejected(y))
    A = M[pd.Series(y).to_numpy() == LABEL_ACCEPTED]
    center, scale = _robust_scale(A)
    return {
        "medians": medians,
        "clf": clf,
        "center": center,
        "scale": scale,
        "names": list(medians.index),
    }


def score_random_forest(model: Model, X: pd.DataFrame) -> np.ndarray:
    return model["clf"].predict_proba(_impute(X, model["medians"]))[:, 1]


def explain_random_forest(model: Model, X: pd.DataFrame) -> Reasons:
    # A forest has no simple per-batch formula. As a readable proxy: among the 20 features the
    # forest relies on most, the ones where this batch is furthest from the accepted batches.
    imp = model["clf"].feature_importances_
    keep = np.argsort(-imp)[:20]
    z = (_impute(X, model["medians"]) - model["center"]) / model["scale"]
    return _top(z[:, keep], [model["names"][j] for j in keep])


# ---- 6. PCA + per-feature limits (teammate's recipe, notebooks ProcessA_PCA_HA) ---------------
# PCA catches broken relations in the correlated groups (flows; pressure_2 + pressure_4); a limit
# per feature catches a deviation on one independent sensor (density, temperatures) that the global
# T2 dilutes. A batch is flagged if T2, SPE or any single feature is outside its limit.


def fit_pca_limits(X: pd.DataFrame, y: pd.Series, columns: list[str] | None = None) -> Model:
    cols = columns if columns is not None else list(X.columns)
    Xc = X[cols]
    acc = pd.Series(y).to_numpy() == LABEL_ACCEPTED
    pca = fit_pca_mspc(Xc, y)
    Z = (_impute(Xc[acc], pca["medians"]) - pca["mean"]) / pca["std"]
    # per-feature limit = Bonferroni-corrected percentile of |z| on the accepted batches
    q = 1 - BONFERRONI_ALPHA / len(cols)
    lim = np.maximum(np.quantile(np.abs(Z), q, axis=0), 1e-6)
    return {"cols": cols, "pca": pca, "z_lim": lim}


def _pca_limits_parts(model: Model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    Xc = X[model["cols"]]
    pca = model["pca"]
    Z = (_impute(Xc, pca["medians"]) - pca["mean"]) / pca["std"]
    return score_pca_mspc(pca, Xc), np.abs(Z) / model["z_lim"]


def score_pca_limits(model: Model, X: pd.DataFrame) -> np.ndarray:
    pca_score, zr = _pca_limits_parts(model, X)
    return np.maximum(pca_score, zr.max(axis=1))  # > 1 on any part = outside its limit


def explain_pca_limits(model: Model, X: pd.DataFrame) -> Reasons:
    _, zr = _pca_limits_parts(model, X)
    return _top(zr, model["cols"])  # values = how many times its own limit


def _median_std_columns(X: pd.DataFrame) -> list[str]:
    cols = [c for c in X.columns if c.endswith("__median") or c.endswith("__std")]
    return cols or list(X.columns)  # fall back to all features if the naming is different


def fit_pca_limits_simple(X: pd.DataFrame, y: pd.Series) -> Model:
    return fit_pca_limits(X, y, _median_std_columns(X))  # teammate's features: median + std


def fit_pca_limits_full(X: pd.DataFrame, y: pd.Series) -> Model:
    return fit_pca_limits(X, y)  # all level + shape features


# ---- 7. SVM (rbf kernel) on median + std: supervised, and one-class --------------------------
# Iteration of 21 Nov on process A: the best methods found, on the teammate's 22 features (median +
# std per sensor). Supervised SVM: F1 0.914 under nested CV (0.931 at the fixed settings below).
# One-class SVM (accepted batches only): 0.897 on A, 0.997 on B.
# An SVM has no per-feature formula; reasons are the median/std features where the batch is
# furthest from the accepted training batches (robust z), the same proxy as the random forest.


def _svm_inputs(model: Model, X: pd.DataFrame) -> np.ndarray:
    M = _impute(X[model["cols"]], model["medians"])
    return (M - model["mean"]) / model["std"]


def _svm_base(X: pd.DataFrame, rows: np.ndarray) -> Model:
    cols = _median_std_columns(X)
    medians = X[cols].median()
    M = _impute(X[cols][rows], medians)
    mean = M.mean(axis=0)
    std = np.maximum(M.std(axis=0), 1e-6 + 1e-3 * np.abs(mean))
    return {"cols": cols, "medians": medians, "mean": mean, "std": std, "names": cols}


def _accepted_center_scale(model: Model, X: pd.DataFrame, acc: np.ndarray):
    """Robust centre/spread of the accepted batches, for the 'furthest from normal' reasons."""
    return _robust_scale(_impute(X[model["cols"]], model["medians"])[acc])


def fit_svm(X: pd.DataFrame, y: pd.Series) -> Model:
    model = _svm_base(X, np.ones(len(X), dtype=bool))
    clf = SVC(C=SVM_C, gamma=SVM_GAMMA_MULT / len(model["cols"]), class_weight="balanced")
    clf.fit(_svm_inputs(model, X), _is_rejected(y))
    acc = pd.Series(y).to_numpy() == LABEL_ACCEPTED
    center, scale = _accepted_center_scale(model, X, acc)
    return {**model, "clf": clf, "center": center, "scale": scale}


def score_svm(model: Model, X: pd.DataFrame) -> np.ndarray:
    return model["clf"].decision_function(_svm_inputs(model, X))  # distance to the boundary


def fit_ocsvm(X: pd.DataFrame, y: pd.Series) -> Model:
    acc = pd.Series(y).to_numpy() == LABEL_ACCEPTED
    model = _svm_base(X, acc)  # scaling learned from accepted batches only
    clf = OneClassSVM(nu=OCSVM_NU, gamma=OCSVM_GAMMA_MULT / len(model["cols"]))
    clf.fit(_svm_inputs(model, X[acc]))
    center, scale = _accepted_center_scale(model, X, acc)
    return {**model, "clf": clf, "center": center, "scale": scale}


def score_ocsvm(model: Model, X: pd.DataFrame) -> np.ndarray:
    return -model["clf"].decision_function(_svm_inputs(model, X))  # higher = further outside normal


def explain_svm(model: Model, X: pd.DataFrame) -> Reasons:
    z = (_impute(X[model["cols"]], model["medians"]) - model["center"]) / model["scale"]
    return _top(z, model["names"])


# ---- 8. ensemble: SVM + kNN on PCA components + one-class SVM ----------------------------------
# Iteration of 21 Nov, process A: best leaderboard F1 found (0.930 honest, nested CV; 0.943 at these
# fixed settings). The three see the data differently: a supervised boundary (SVM), "which known
# batches does it resemble" (kNN), "how far outside normal is it" (one-class SVM). Each member's
# score is standardised with its out-of-fold scores on the training batches (inner CV), so the
# three scales are comparable; the ensemble score is their mean.


def _pca_knn_fit(X: pd.DataFrame, y: pd.Series) -> Model:
    base = _svm_base(X, np.ones(len(X), dtype=bool))
    Z = _svm_inputs(base, X)
    _, s, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    var = s**2
    k = int(np.searchsorted(np.cumsum(var) / var.sum(), PCA_VARIANCE) + 1)
    P, mu = Vt[:k].T, Z.mean(axis=0)
    clf = KNeighborsClassifier(ENSEMBLE_KNN_K, weights="distance").fit(
        (Z - mu) @ P, _is_rejected(y)
    )
    return {**base, "P": P, "mu": mu, "clf": clf}


def _pca_knn_score(model: Model, X: pd.DataFrame) -> np.ndarray:
    Z = _svm_inputs(model, X)
    return model["clf"].predict_proba((Z - model["mu"]) @ model["P"])[:, 1]


_MEMBERS = {
    "svm": (lambda X, y: fit_svm(X, y), lambda m, X: score_svm(m, X)),
    "pca_knn": (_pca_knn_fit, _pca_knn_score),
    "ocsvm": (lambda X, y: fit_ocsvm(X, y), lambda m, X: score_ocsvm(m, X)),
}


def fit_ensemble(X: pd.DataFrame, y: pd.Series) -> Model:
    y = pd.Series(y).reset_index(drop=True)
    oof = np.zeros((len(X), len(_MEMBERS)))
    folds = StratifiedKFold(CV_INNER_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in folds.split(X, y):
        for j, (fit, score) in enumerate(_MEMBERS.values()):
            oof[te, j] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
    members = {name: fit(X, y) for name, (fit, _) in _MEMBERS.items()}
    return {
        "members": members,
        "mu": oof.mean(axis=0),
        "sd": oof.std(axis=0) + 1e-12,
    }


def score_ensemble(model: Model, X: pd.DataFrame) -> np.ndarray:
    S = np.column_stack([score(model["members"][n], X) for n, (_, score) in _MEMBERS.items()])
    return ((S - model["mu"]) / model["sd"]).mean(axis=1)


def explain_ensemble(model: Model, X: pd.DataFrame) -> Reasons:
    # proxy: median/std features furthest from the accepted batches
    return explain_svm(model["members"]["svm"], X)


# ---- registry: the order is the order in the report ---------------------------------------------

METHODS = {
    "accept_all": (fit_accept_all, score_accept_all, explain_accept_all),
    "feature_limits": (fit_feature_limits, score_feature_limits, explain_feature_limits),
    "pca_mspc": (fit_pca_mspc, score_pca_mspc, explain_pca_mspc),
    "logreg": (fit_logreg, score_logreg, explain_logreg),
    "random_forest": (fit_random_forest, score_random_forest, explain_random_forest),
    "pca_limits_simple": (fit_pca_limits_simple, score_pca_limits, explain_pca_limits),
    "pca_limits_full": (fit_pca_limits_full, score_pca_limits, explain_pca_limits),
    "svm": (fit_svm, score_svm, explain_svm),
    "ocsvm": (fit_ocsvm, score_ocsvm, explain_svm),
    "ensemble": (fit_ensemble, score_ensemble, explain_ensemble),
}

# how each method reaches a decision, for the report (interpretability is a judging criterion)
INTERPRETABILITY = {
    "accept_all": "trivial",
    "feature_limits": "high: names the feature and how far outside the normal range it is",
    "pca_mspc": "medium-high: contribution plot shows which features break the normal pattern",
    "logreg": "medium-high: each feature's push towards 'rejected' is one weight x value",
    "random_forest": "medium: global importances; per-batch reasons are a proxy",
    "pca_limits_simple": "high: names the feature (median/std) and how far past its limit",
    "pca_limits_full": "high: names the feature and how far past its limit",
    "svm": "medium: boundary has no per-feature formula; reasons are a proxy",
    "ocsvm": "medium: learns only from accepted batches; reasons are a proxy",
    "ensemble": "low-medium: three models averaged; reasons are a proxy",
}
