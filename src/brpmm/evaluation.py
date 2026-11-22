"""Batch-level validation harness.

Every model is wrapped as a ``FitPredict`` callable::

    fit_predict(X_train, y_train, X_test) -> (score, decision)

``score`` ranks batches (higher = more likely rejected) and ``decision`` is
the model's own accept/reject call, with any threshold set on the training
fold only. Splits are always by batch, stratified by label, and repeated,
because a single 5-fold split on 161 batches is noisy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from brpmm.mspc import PCAMonitor

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], tuple[np.ndarray, np.ndarray]]


def supervised(estimator) -> FitPredict:
    """Any sklearn/imblearn classifier with predict_proba; decision at p >= 0.5."""

    def fit_predict(X_tr, y_tr, X_te):
        model = clone(estimator).fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        return p, (p >= 0.5).astype(int)

    return fit_predict


def one_class(monitor_cls=PCAMonitor, **monitor_kwargs) -> FitPredict:
    """A monitor (PCAMonitor, KNNMonitor, ...) trained on the accepted batches of
    the training fold only. Rejected labels are used for evaluation, never fitting."""

    def fit_predict(X_tr, y_tr, X_te):
        monitor = monitor_cls(**monitor_kwargs).fit(X_tr[y_tr == 0])
        s = monitor.score(X_te)
        return s["score"].to_numpy(), s["alarm"].to_numpy().astype(int)

    return fit_predict


def binary_metrics(y: np.ndarray, score: np.ndarray, decision: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "balanced_accuracy": balanced_accuracy_score(y, decision),
        "mcc": matthews_corrcoef(y, decision),
        "precision_rejected": precision_score(y, decision, pos_label=1, zero_division=0),
        "recall_rejected": recall_score(y, decision, pos_label=1, zero_division=0),
        "specificity": recall_score(y, decision, pos_label=0, zero_division=0),
        "f1_rejected": f1_score(y, decision, pos_label=1, zero_division=0),
        # The leaderboard's F1 behaves as F1 of the *accepted* class (see README, Scoring)
        "f1_accepted": f1_score(y, decision, pos_label=0, zero_division=0),
    }


@dataclass
class CVResult:
    name: str
    metrics: pd.DataFrame  # one row per repeat
    oof_scores: pd.DataFrame  # repeats x batches
    oof_decisions: pd.DataFrame  # repeats x batches, 1 = rejected

    def summary(self) -> pd.Series:
        m = self.metrics
        return pd.concat([m.mean().add_suffix("_mean"), m.std().add_suffix("_sd")]).rename(
            self.name
        )


def _oof(fit_predict, X, y, cv) -> tuple[np.ndarray, np.ndarray]:
    score = np.empty(len(y))
    decision = np.empty(len(y), dtype=int)
    for tr, te in cv.split(X, y):
        score[te], decision[te] = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
    return score, decision


def repeated_cv(
    fit_predict: FitPredict,
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> CVResult:
    rows, scores, decisions = [], {}, {}
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        score, decision = _oof(fit_predict, X, y, cv)
        rows.append(binary_metrics(y, score, decision))
        scores[r], decisions[r] = score, decision
    return CVResult(
        name,
        pd.DataFrame(rows),
        pd.DataFrame(scores, index=X.index).T,
        pd.DataFrame(decisions, index=X.index).T,
    )


def permutation_test(
    fit_predict: FitPredict,
    X: pd.DataFrame,
    y: np.ndarray,
    n_permutations: int = 100,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float, np.ndarray, float]:
    """Is the cross-validated ROC-AUC better than chance? Returns (observed, null, p)."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    observed = roc_auc_score(y, _oof(fit_predict, X, y, cv)[0])
    rng = np.random.default_rng(seed)
    null = np.array(
        [
            roc_auc_score(yp, _oof(fit_predict, X, yp, cv)[0])
            for yp in (rng.permutation(y) for _ in range(n_permutations))
        ]
    )
    p = (1 + np.sum(null >= observed)) / (1 + n_permutations)
    return observed, null, p


def learning_curve(
    fit_predict: FitPredict,
    X: pd.DataFrame,
    y: np.ndarray,
    fractions: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0),
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Test-fold ROC-AUC when training on a stratified fraction of each training fold."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for frac in fractions:
            score = np.empty(len(y))
            for tr, te in cv.split(X, y):
                keep = np.concatenate(
                    [
                        rng.choice(idx, max(2, int(round(frac * len(idx)))), replace=False)
                        for idx in (tr[y[tr] == 0], tr[y[tr] == 1])
                    ]
                )
                score[te] = fit_predict(X.iloc[keep], y[keep], X.iloc[te])[0]
            n_train = int(round(frac * len(y) * (n_splits - 1) / n_splits))
            rows.append(
                {
                    "repeat": r,
                    "fraction": frac,
                    "n_train": n_train,
                    "roc_auc": roc_auc_score(y, score),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- thresholds
def best_threshold(y: np.ndarray, score: np.ndarray, metric: str = "f1_accepted") -> float:
    """Score threshold (decision = score >= t) that maximises ``metric`` on (y, score).

    Ties are broken towards the middle of the best plateau, so the threshold
    does not sit on the edge of one lucky batch.
    """
    fn = {
        "f1": lambda yy, d: f1_score(yy, d, zero_division=0),
        "f1_accepted": lambda yy, d: f1_score(yy, d, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "mcc": matthews_corrcoef,
    }[metric]
    candidates = np.unique(score)
    values = np.array([fn(y, (score >= t).astype(int)) for t in candidates])
    best = np.flatnonzero(values == values.max())
    return float(candidates[best[len(best) // 2]])


def tuned_threshold(
    fit_predict: FitPredict, metric: str = "f1_accepted", inner_splits: int = 5, seed: int = 0
) -> FitPredict:
    """Wrap a model so its decision threshold is tuned by an *inner* CV on the
    training fold only (nested CV). The outer test fold never influences it."""

    def wrapped(X_tr, y_tr, X_te):
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
        oof = _oof(fit_predict, X_tr, y_tr, inner)[0]
        threshold = best_threshold(y_tr, oof, metric)
        score = fit_predict(X_tr, y_tr, X_te)[0]
        return score, (score >= threshold).astype(int)

    return wrapped


def bootstrap_metrics(
    y: np.ndarray,
    decisions: np.ndarray,
    n_test: int,
    n_boot: int = 2000,
    extra: tuple[int, int, int, int] = (0, 0, 0, 0),
    seed: int = 0,
) -> pd.DataFrame:
    """Spread of F1 / precision / recall on a test set of ``n_test`` batches.

    ``decisions`` is (repeats x batches) of out-of-fold decisions; each bootstrap
    draws a repeat, then ``n_test`` batches with replacement. ``extra`` adds fixed
    (tp, fp, fn, tn) counts, e.g. another process scored in the same leaderboard.
    """
    rng = np.random.default_rng(seed)
    decisions = np.atleast_2d(decisions)
    rows = []
    for _ in range(n_boot):
        d = decisions[rng.integers(len(decisions))]
        idx = rng.integers(len(y), size=n_test)
        yy, dd = y[idx], d[idx]
        tp = int(((yy == 1) & (dd == 1)).sum()) + extra[0]
        fp = int(((yy == 0) & (dd == 1)).sum()) + extra[1]
        fn = int(((yy == 1) & (dd == 0)).sum()) + extra[2]
        tn = int(((yy == 0) & (dd == 0)).sum()) + extra[3]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        bal = 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0))
        rows.append({"f1": f1, "precision": precision, "recall": recall, "balanced_accuracy": bal})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- importance
Fit = Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], np.ndarray]]


def grouped_permutation_importance(
    fit: Fit,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    n_repeats: int = 3,
    n_shuffles: int = 5,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop in out-of-fold ROC-AUC when all columns of a group (e.g. one sensor's
    features) are shuffled together across the test batches.

    ``fit(X_train, y_train)`` returns a scoring function. Returns one row per
    (group, repeat) with ``auc_drop`` averaged over shuffles.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repeats):
        base = np.empty(len(y))
        perm = {g: np.empty((n_shuffles, len(y))) for g in groups}
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y):
            score = fit(X.iloc[tr], y[tr])
            X_te = X.iloc[te]
            base[te] = score(X_te)
            for g, cols in groups.items():
                for s in range(n_shuffles):
                    Xp = X_te.copy()
                    Xp[cols] = X_te[cols].to_numpy()[rng.permutation(len(te))]
                    perm[g][s, te] = score(Xp)
        auc0 = roc_auc_score(y, base)
        for g in groups:
            drops = [auc0 - roc_auc_score(y, perm[g][s]) for s in range(n_shuffles)]
            rows.append({"group": g, "repeat": r, "auc": auc0, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- submitted model
# Shared by scripts/select_final_a.py and scripts/tune_process_a.py: the submitted process A
# ensemble (brpmm.methods.ensemble) run through its own harness (brpmm.evaluate) on given splits.
FRESH_SEEDS = (50, 51, 52, 53)  # never used to choose a model; the submitted A model scores 0.941


@contextmanager
def harness_seed(seed: int) -> Iterator[None]:
    """brpmm.evaluate reads a module-level SEED for its splits; set it and always restore it."""
    import brpmm.evaluate as harness

    old = harness.SEED
    harness.SEED = seed
    try:
        yield
    finally:
        harness.SEED = old


def submitted_ensemble_decisions(
    X: pd.DataFrame, labels: pd.Series, seeds=FRESH_SEEDS
) -> np.ndarray:
    """Out-of-fold decisions (repeats x batches, 1 = rejected) of brpmm.methods.ensemble, one
    repeat of StratifiedKFold(5, shuffle, seed) per seed, cut-off chosen in its inner CV."""
    import brpmm.evaluate as harness
    from brpmm.config import LABEL_REJECTED
    from brpmm.methods import METHODS

    dec = []
    for seed in seeds:
        with harness_seed(seed):
            _, oof = harness.cross_validate(
                X, labels, methods={"ensemble": METHODS["ensemble"]}, repeats=1
            )
        pred = oof.set_index("batch_id").loc[X.index, "pred"]
        dec.append((pred == LABEL_REJECTED).astype(int).to_numpy())
    return np.array(dec)


def leaderboard_summary(dec: np.ndarray, y: np.ndarray, top_f1: float) -> dict:
    """F1 (accepted positive) per repeat from out-of-fold decisions (repeats x batches,
    1 = rejected; y: 1 = rejected), the mean cases, and the chance that one 161-batch test
    draw beats ``top_f1``."""
    tp = ((dec == 0) & (y == 0)).sum(1)
    fp = ((dec == 0) & (y == 1)).sum(1)
    fn = ((dec == 1) & (y == 0)).sum(1)
    f1 = 2 * tp / (2 * tp + fp + fn)
    boot = bootstrap_metrics(1 - y, 1 - dec, n_test=161, n_boot=4000)
    return {
        "f1_accepted": round(float(f1.mean()), 4),
        "f1_sd": round(float(f1.std()), 4),
        "bad_caught_of_40": round(float(((dec == 1) & (y == 1)).sum(1).mean()), 1),
        "good_thrown_away_of_121": round(float(((dec == 1) & (y == 0)).sum(1).mean()), 1),
        "one_test_draw_median_f1": round(float(boot.f1.median()), 4),
        "p_beat_top": round(float((boot.f1 > top_f1).mean()), 3),
    }
