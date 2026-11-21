"""Cross-validation harness: every method on exactly the same splits, scored the same way.

Inputs: calibration data of one process (via brpmm.data), the methods in brpmm.methods.
Outputs: per-fold scores, out-of-fold predictions with reasons, settings used (returned as a dict;
         brpmm.report turns it into the findings report).
Depends on: config, data, features, metrics, methods.

Run:  BRPMM_DATA_DIR=... python -m brpmm.evaluate --process A B

How a method is tested (identical for all):
1. Repeated stratified 5-fold CV on the calibration batches (each fold keeps the accepted/rejected
   ratio). A model is never scored on a batch it was trained on.
2. Inside the training part only, an inner 3-fold CV gives out-of-fold scores, and the cut-off that
   maximises leaderboard F1 on them is chosen. The held-out fold never influences its own cut-off.
3. The method is refitted on the whole training part, scores the held-out fold, the cut-off turns
   scores into accepted/rejected.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from brpmm.config import (
    CV_FOLDS,
    CV_INNER_FOLDS,
    CV_REPEATS,
    LABEL_ACCEPTED,
    LABEL_REJECTED,
    PROCESSES,
    SEED,
)
from brpmm.data import load_calibration
from brpmm.features import batch_features
from brpmm.methods import METHODS
from brpmm.metrics import leaderboard_f1, summarize


def choose_cutoff(scores: np.ndarray, y: pd.Series) -> float:
    """Cut-off (reject if score > cut-off) that maximises leaderboard F1 on these scores.

    Candidates sit halfway between consecutive distinct scores, plus "reject nothing".
    Among equally good cut-offs the middle one is taken, so the choice doesn't hug one batch.
    """
    u = np.unique(scores)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1]]])
    f1 = np.array([leaderboard_f1(y, _labels(scores, c)) for c in cands])
    best = np.flatnonzero(f1 == f1.max())
    return float(cands[best[len(best) // 2]])


def _labels(scores: np.ndarray, cutoff: float) -> np.ndarray:
    return np.where(scores > cutoff, LABEL_REJECTED, LABEL_ACCEPTED)


def _inner_cutoff(fit, score, X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    oof = np.empty(len(X))
    inner = StratifiedKFold(CV_INNER_FOLDS, shuffle=True, random_state=seed)
    for tr, te in inner.split(X, y):
        oof[te] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
    return choose_cutoff(oof, y)


def load_features(process: str) -> tuple[pd.DataFrame, pd.Series]:
    """Batch features and labels of the calibration batches of one process."""
    ts, labels = load_calibration(process)
    X = batch_features(ts)
    return X, labels.reindex(X.index)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    methods: dict | None = None,
    repeats: int = CV_REPEATS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every method on the same repeated stratified folds.

    Returns (per-fold scores, out-of-fold rows).
    """
    methods = methods if methods is not None else METHODS
    fold_rows, oof_rows = [], []
    for r in range(repeats):
        outer = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + r)
        for k, (tr, te) in enumerate(outer.split(X, y)):
            Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
            for name, (fit, score, explain) in methods.items():
                # seed differs per split so inner folds are not the same batches every time,
                # but is fixed so a rerun gives identical results
                cut = _inner_cutoff(fit, score, Xtr, ytr, seed=SEED + 1000 + 10 * r + k)
                model = fit(Xtr, ytr)
                s = score(model, Xte)
                pred = _labels(s, cut)
                fold_rows.append(
                    {"method": name, "repeat": r, "fold": k, "cutoff": cut, **summarize(yte, pred)}
                )
                for b, si, pi, ri in zip(Xte.index, s, pred, explain(model, Xte), strict=True):
                    oof_rows.append(
                        {
                            "method": name,
                            "repeat": r,
                            "batch_id": b,
                            "label": y[b],
                            "score": float(si),
                            "pred": pi,
                            "reasons": ri,
                        }
                    )
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def run_process(process: str, repeats: int = CV_REPEATS) -> dict:
    """Cross-validate every method on one process, in the shape brpmm.report expects."""
    X, y = load_features(process)
    folds, oof = cross_validate(X, y, repeats=repeats)
    return {
        "process": process,
        "n_batches": len(X),
        "n_rejected": int((y == LABEL_REJECTED).sum()),
        "n_features": X.shape[1],
        "folds": folds,
        "oof": oof,
    }


def summary_table(res: dict) -> pd.DataFrame:
    """Per method: F1 over the whole calibration set per repeat (pooled out-of-fold), mean and std,
    the four cases averaged over repeats, and in how many folds it beat 'accept everything'."""
    oof, folds = res["oof"], res["folds"]
    rows = []
    base = folds[folds.method == "accept_all"].set_index(["repeat", "fold"])["f1"]
    for name, g in oof.groupby("method", sort=False):
        per_rep = [summarize(gr["label"], gr["pred"]) for _, gr in g.groupby("repeat")]
        f1s = np.array([p["f1"] for p in per_rep])
        mf = folds[folds.method == name].set_index(["repeat", "fold"])["f1"]
        rows.append(
            {
                "method": name,
                "f1_mean": f1s.mean(),
                "f1_std": f1s.std(),
                "good_kept": np.mean([p["good_kept"] for p in per_rep]),
                "bad_caught": np.mean([p["bad_caught"] for p in per_rep]),
                "bad_let_through": np.mean([p["bad_let_through"] for p in per_rep]),
                "good_thrown_away": np.mean([p["good_thrown_away"] for p in per_rep]),
                "folds_beating_accept_all": int((mf > base.reindex(mf.index)).sum()),
                "n_folds": len(mf),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def main() -> None:
    # ---- PLUMBING: arguments, timing, report ----
    from brpmm.report import build_report

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--process", nargs="+", default=list(PROCESSES), choices=list(PROCESSES))
    p.add_argument("--repeats", type=int, default=CV_REPEATS)
    a = p.parse_args()
    results = {}
    for proc in a.process:
        t0 = time.time()
        results[proc] = run_process(proc, a.repeats)
        print(f"process {proc}: {time.time() - t0:.0f} s")
        print(summary_table(results[proc]).round(3).to_string())
    print(f"report: {build_report(results)}")


if __name__ == "__main__":
    main()
