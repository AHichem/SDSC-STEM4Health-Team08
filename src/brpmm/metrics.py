"""Scores shared by every method, so all of them are measured the same way.

Inputs: true and predicted labels as strings ("accepted" / "rejected").
Outputs: plain dicts of numbers.
Depends on: config (label names).

Why strings and not 0/1: data.to_binary encodes rejected = 1, while the leaderboard F1 treats
"accepted" as the positive class. Working on the label names directly removes any doubt about
which class is "positive". Deduced from the reference leaderboard entry (A: accuracy 0.845,
precision 0.88, recall 0.917 -> about 120 positives out of 161 = the accepted batches);
test_metrics checks that the reference counts give its F1 of 0.898.
"""

from __future__ import annotations

import pandas as pd

from brpmm.config import LABEL_ACCEPTED, LABEL_REJECTED


def four_cases(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Count the four outcomes, named by what the model said and what the label says."""
    t = pd.Series(y_true).to_numpy()
    p = pd.Series(y_pred).to_numpy()
    return {
        "good_kept": int(((p == LABEL_ACCEPTED) & (t == LABEL_ACCEPTED)).sum()),
        "bad_let_through": int(((p == LABEL_ACCEPTED) & (t == LABEL_REJECTED)).sum()),
        "bad_caught": int(((p == LABEL_REJECTED) & (t == LABEL_REJECTED)).sum()),
        "good_thrown_away": int(((p == LABEL_REJECTED) & (t == LABEL_ACCEPTED)).sum()),
    }


def f1_from_cases(cases: dict[str, int]) -> float:
    """Leaderboard F1: "accepted" is the positive class.

    precision = good kept / everything we accepted; recall = good kept / all good batches.
    Returns 0.0 (not NaN) when nothing is accepted and nothing is good, so an empty fold can't
    silently drop out of an average.
    """
    tp = cases["good_kept"]
    fp = cases["bad_let_through"]
    fn = cases["good_thrown_away"]
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def leaderboard_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """F1 exactly as the leaderboard scores it. Use this, never sklearn's f1_score directly."""
    return f1_from_cases(four_cases(y_true, y_pred))


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """F1 plus the four cases."""
    cases = four_cases(y_true, y_pred)
    return {"f1": f1_from_cases(cases), **cases}
