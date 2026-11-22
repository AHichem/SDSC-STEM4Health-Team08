"""PCA-based multivariate statistical process monitoring (MSPC).

Fit on accepted batches only. Two statistics per batch:

* **T²** (Hotelling): how far the batch sits from the centre *inside* the
  normal correlation pattern, i.e. unusually large but familiar variation.
* **SPE** (squared prediction error, a.k.a. Q): how much of the batch the
  normal pattern cannot reconstruct, i.e. broken relationships between variables.

Limits are set on **out-of-fold** statistics of the accepted batches.
In-sample statistics are optimistic for small sample sizes, so the limits would
be too tight otherwise. The default ``limit_method="empirical"`` takes the
alpha-quantile and is robust to a few extreme normal batches. ``"box"``
(scaled chi-square, the textbook choice) matches mean and variance, so a single
extreme batch can inflate it by orders of magnitude: in process A, A_239 did
this before its dropout rows were excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def scaled_chi2_limit(values: np.ndarray, alpha: float) -> float:
    """Box approximation: values ~ g * chi2(h), moments matched to the sample."""
    mean, var = float(np.mean(values)), float(np.var(values, ddof=1))
    g, h = var / (2 * mean), 2 * mean**2 / var
    return float(g * stats.chi2.ppf(alpha, h))


class PCAMonitor:
    def __init__(
        self,
        n_components: int | float = 0.8,
        alpha: float = 0.95,
        cv_limits: bool = True,
        limit_method: str = "empirical",
        n_splits: int = 5,
        random_state: int = 0,
    ):
        if limit_method not in ("empirical", "box"):
            raise ValueError("limit_method must be 'empirical' or 'box'")
        self.n_components = n_components
        self.alpha = alpha
        self.cv_limits = cv_limits
        self.limit_method = limit_method
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _fit_core(X: np.ndarray, n_components: int | float) -> tuple[StandardScaler, PCA]:
        scaler = StandardScaler().fit(X)
        pca = PCA(n_components=n_components, svd_solver="full").fit(scaler.transform(X))
        return scaler, pca

    @staticmethod
    def _stats(scaler: StandardScaler, pca: PCA, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = scaler.transform(X)
        T = pca.transform(Z)
        t2 = (T**2 / pca.explained_variance_).sum(axis=1)
        spe = ((Z - pca.inverse_transform(T)) ** 2).sum(axis=1)
        return t2, spe

    def fit(self, X: pd.DataFrame) -> PCAMonitor:
        """Fit on normal (accepted) batches only."""
        self.feature_names_ = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        self.scaler_, self.pca_ = self._fit_core(Xv, self.n_components)
        self.n_components_ = self.pca_.n_components_
        if self.cv_limits:
            t2 = np.empty(len(Xv))
            spe = np.empty(len(Xv))
            kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
            for tr, te in kf.split(Xv):
                scaler, pca = self._fit_core(Xv[tr], self.n_components_)
                t2[te], spe[te] = self._stats(scaler, pca, Xv[te])
        else:
            t2, spe = self._stats(self.scaler_, self.pca_, Xv)
        limit = (
            (lambda v: float(np.quantile(v, self.alpha)))
            if self.limit_method == "empirical"
            else (lambda v: scaled_chi2_limit(v, self.alpha))
        )
        self.t2_limit_, self.spe_limit_ = limit(t2), limit(spe)
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """T², SPE, their ratios to the limits, and the alarm (either ratio > 1)."""
        t2, spe = self._stats(self.scaler_, self.pca_, X[self.feature_names_].to_numpy(float))
        out = pd.DataFrame({"T2": t2, "SPE": spe}, index=X.index)
        out["T2_ratio"] = out["T2"] / self.t2_limit_
        out["SPE_ratio"] = out["SPE"] / self.spe_limit_
        out["score"] = out[["T2_ratio", "SPE_ratio"]].max(axis=1)
        out["alarm"] = out["score"] > 1
        return out

    def spe_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature squared residual; sums to SPE."""
        Z = self.scaler_.transform(X[self.feature_names_].to_numpy(float))
        R = Z - self.pca_.inverse_transform(self.pca_.transform(Z))
        return pd.DataFrame(R**2, index=X.index, columns=self.feature_names_)

    def t2_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to T² (sums to T²)."""
        Z = self.scaler_.transform(X[self.feature_names_].to_numpy(float))
        T = self.pca_.transform(Z)
        weighted = (T / self.pca_.explained_variance_) @ self.pca_.components_
        return pd.DataFrame(weighted * Z, index=X.index, columns=self.feature_names_)
