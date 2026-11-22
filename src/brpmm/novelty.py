"""Distance-based novelty detection, trained on accepted batches only.

``KNNMonitor`` scores a batch by its mean distance to the k nearest accepted
batches (standardised features). Unlike a single PCA model it makes no
assumption that "normal" is one elliptical cloud, so it copes with the two
operating modes found in process A. Output schema matches ``PCAMonitor.score``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class KNNMonitor:
    def __init__(self, k: int = 10, alpha: float = 0.95):
        self.k = k
        self.alpha = alpha

    def fit(self, X: pd.DataFrame) -> KNNMonitor:
        """Fit on normal (accepted) batches only."""
        self.feature_names_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        # leave-self-out distances give an honest limit for unseen normal batches
        self.train_dist_ = self.nn_.kneighbors(Z)[0][:, 1:].mean(axis=1)
        self.limit_ = float(np.quantile(self.train_dist_, self.alpha))
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.scaler_.transform(X[self.feature_names_])
        dist = self.nn_.kneighbors(Z, n_neighbors=self.k)[0].mean(axis=1)
        out = pd.DataFrame({"knn_distance": dist}, index=X.index)
        out["score"] = out["knn_distance"] / self.limit_
        out["alarm"] = out["score"] > 1
        return out
