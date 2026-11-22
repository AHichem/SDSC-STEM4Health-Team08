"""Semi-supervised self-training on unlabelled batches (used for process A).

The organisers allowed training on the unlabelled test batches. The model is unchanged; only
its training data grows:

  1. fit on the labelled batches; cut-off from CV on them (brpmm.predict.cv_cutoff)
  2. score the unlabelled batches; those beyond the cut-off by ``margin`` get that label
     (pseudo-labels); kind "accepted" keeps only the confidently accepted ones
  3. refit on labelled + pseudo-labelled batches, new CV cut-off, score the unlabelled batches

With ``bags`` the single fit in steps 1 and 3 is replaced by subagging
(brpmm.evaluation.subagged): cut-off on out-of-bag scores, mean score of the bagged models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from brpmm.config import LABEL_ACCEPTED, LABEL_REJECTED
from brpmm.evaluation import subagged
from brpmm.predict import cv_cutoff


@dataclass(frozen=True)
class SelfTrainingResult:
    scores: pd.Series  # final score per unlabelled batch (higher = more likely rejected)
    cutoff: float  # final cut-off: rejected if score > cutoff
    pseudo: pd.Series  # pseudo-labels given in step 2 (subset of the unlabelled batches)
    first_cutoff: float  # cut-off of step 1

    @property
    def labels(self) -> pd.Series:
        return pd.Series(
            np.where(self.scores > self.cutoff, LABEL_REJECTED, LABEL_ACCEPTED),
            index=self.scores.index,
        )


def _fit_cut_score(fit, score, X, y, Xt, bags: dict | None) -> tuple[float, np.ndarray]:
    if bags is not None:
        return subagged(fit, score, X, y, Xt, **bags)
    return cv_cutoff(fit, score, X, y), score(fit(X, y), Xt)


def self_train(
    fit,
    score,
    X: pd.DataFrame,
    y: pd.Series,
    Xt: pd.DataFrame,
    kind: str = "accepted",
    margin: float = 1.0,
    bags: dict | None = None,
) -> SelfTrainingResult:
    """``kind`` "accepted": add only confidently accepted batches; "both": also confidently
    rejected ones. ``bags``: keyword arguments for subagging (n_bags, fraction, seed), or None."""
    if kind not in ("accepted", "both"):
        raise ValueError(f"kind must be 'accepted' or 'both', not {kind!r}")
    cut, s = _fit_cut_score(fit, score, X, y, Xt, bags)
    sure_rej = s > cut + margin if kind == "both" else np.zeros(len(s), bool)
    sure_acc = s < cut - margin
    pick = sure_rej | sure_acc
    pseudo = pd.Series(np.where(sure_rej, LABEL_REJECTED, LABEL_ACCEPTED), index=Xt.index)[pick]
    Xa, ya = pd.concat([X, Xt[pick]]), pd.concat([y, pseudo])
    cut_a, s_a = _fit_cut_score(fit, score, Xa, ya, Xt, bags)
    return SelfTrainingResult(pd.Series(s_a, index=Xt.index), float(cut_a), pseudo, float(cut))
