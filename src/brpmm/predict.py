"""Final models: train on all calibration batches, predict the test batches, write a submission.

Process A: `ensemble` (SVM + kNN on PCA components + one-class SVM), honest CV F1 0.930.
Process B: `pca_mspc` trained on accepted batches only, CV F1 1.000.
See docs/model_results.md (ids A24/A25 and B02) for how these were chosen.

Inputs:  calibration and test data of both processes (data_dir()), the template.
Outputs: outputs/<timestamp>_<name>.csv            submission (batch_id, project, label)
         outputs/<timestamp>_<name>_details.csv    score, cut-off, traffic light, top 3 reasons
Nothing is sent to the leaderboard: submitting is a separate, deliberate step (submission.submit).

Run: python -m brpmm.predict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from brpmm.config import CV_FOLDS, CV_REPEATS, LABEL_ACCEPTED, LABEL_REJECTED, SEED
from brpmm.data import load_test
from brpmm.evaluate import choose_cutoff, load_features
from brpmm.features import batch_features
from brpmm.methods import _MEMBERS, METHODS
from brpmm.submission import build_submission, load_template, save_submission

FINAL_METHOD = {"A": "ensemble", "B": "pca_mspc"}


def cv_cutoff(fit, score, X: pd.DataFrame, y: pd.Series) -> float:
    """Cut-off for the model trained on all batches: median of the cut-offs chosen on the
    out-of-fold scores of each CV repeat. Same splits as the evaluation (SEED + repeat), and the
    test data plays no part. The median, not the mean, so one odd repeat can't move it."""
    cuts = []
    for r in range(CV_REPEATS):
        oof = np.empty(len(X))
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + r)
        for tr, te in folds.split(X, y):
            oof[te] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
        cuts.append(choose_cutoff(oof, y))
    return float(np.median(cuts))


def traffic_light(X: pd.DataFrame, y: pd.Series, Xt: pd.DataFrame) -> pd.Series:
    """Process A only. Each ensemble member votes with its own cut-off; green = all three accept,
    red = all three reject, amber = they disagree. Out-of-fold on calibration the amber group was
    22 % of batches but held 45 % of the errors: those are the batches to send to a person.
    This does not change the label; the label is the ensemble decision."""
    votes = []
    for fit, score in _MEMBERS.values():
        cut = cv_cutoff(fit, score, X, y)
        votes.append(score(fit(X, y), Xt) > cut)
    n = np.sum(votes, axis=0)
    light = np.select([n == 0, n == len(votes)], ["green", "red"], "amber")
    return pd.Series(light, index=Xt.index)


def predict_process(process: str) -> pd.DataFrame:
    fit, score, explain = METHODS[FINAL_METHOD[process]]
    X, y = load_features(process)
    Xt = batch_features(load_test(process)).reindex(columns=X.columns)  # same columns, same order
    cut = cv_cutoff(fit, score, X, y)
    model = fit(X, y)
    s = score(model, Xt)
    out = pd.DataFrame(
        {
            "project": process,
            "label": np.where(s > cut, LABEL_REJECTED, LABEL_ACCEPTED),
            "score": np.round(s, 4),
            "cutoff": round(cut, 4),
        },
        index=Xt.index,
    )
    out["light"] = traffic_light(X, y, Xt) if process == "A" else ""
    out["top_reasons"] = ["; ".join(f"{f} ({v:.2f})" for f, v in r) for r in explain(model, Xt)]
    return out


def main(name: str = "A-ensemble_B-pca_NOT-SUBMITTED") -> None:
    details = pd.concat([predict_process(p) for p in ("A", "B")])
    details.index.name = "batch_id"
    sub = build_submission(
        {p: d["label"] for p, d in details.groupby("project")}, load_template()
    )  # also validates against the template
    path = save_submission(sub, name)
    details.reset_index().to_csv(path.with_name(path.stem + "_details.csv"), index=False)

    # ---- PLUMBING: short summary on screen ----
    print(f"saved {path}")
    print(details.groupby("project")["label"].value_counts().to_string())
    print(details.loc[details.project == "A", "light"].value_counts().to_string())


if __name__ == "__main__":
    main()
