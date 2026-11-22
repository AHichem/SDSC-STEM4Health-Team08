import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from brpmm.evaluation import (
    best_threshold,
    bootstrap_metrics,
    grouped_permutation_importance,
    one_class,
    permutation_test,
    repeated_cv,
    supervised,
    tuned_threshold,
)
from brpmm.mspc import PCAMonitor


def toy(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 4 == 0).astype(int)  # 25% positives, like process A
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    X["a"] += 2.5 * y
    return X, y


def test_repeated_cv_shapes_and_signal():
    X, y = toy()
    res = repeated_cv(supervised(LogisticRegression()), X, y, "lr", n_repeats=3)
    assert len(res.metrics) == 3
    assert res.oof_scores.shape == (3, len(y))
    assert res.summary()["roc_auc_mean"] > 0.85


def test_one_class_trains_on_accepted_only():
    X, y = toy()
    score, _ = one_class(n_components=2)(X, y, X)
    reference = PCAMonitor(n_components=2).fit(X[y == 0]).score(X)["score"].to_numpy()
    np.testing.assert_allclose(score, reference)
    res = repeated_cv(one_class(n_components=2), X, y, "oc", n_repeats=2)
    assert res.summary()["roc_auc_mean"] > 0.8


def test_permutation_null_is_near_chance():
    X, y = toy()
    obs, null, p = permutation_test(supervised(LogisticRegression()), X, y, n_permutations=20)
    assert obs > 0.85
    assert abs(null.mean() - 0.5) < 0.1
    assert p <= 1 / 21 + 1e-9


def test_best_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_threshold(y, score)
    assert 0.3 < t <= 0.8


def test_tuned_threshold_is_nested_and_learns():
    X, y = toy()
    res = repeated_cv(tuned_threshold(supervised(LogisticRegression())), X, y, "t", n_repeats=2)
    assert res.summary()["f1_rejected_mean"] > 0.6


def test_bootstrap_metrics_perfect_and_extra_counts():
    y = np.array([0, 0, 1, 1])
    perfect = bootstrap_metrics(y, y[None, :], n_test=50, n_boot=200)
    assert perfect["f1"].min() == 1.0
    # adding 5 missed rejects from "another process" must lower recall below 1
    worse = bootstrap_metrics(y, y[None, :], n_test=50, n_boot=200, extra=(0, 0, 5, 0))
    assert worse["recall"].max() < 1.0


def test_grouped_importance_finds_the_signal_column():
    X, y = toy()

    def fit(X_tr, y_tr):
        model = LogisticRegression().fit(X_tr, y_tr)
        return lambda X_te: model.predict_proba(X_te)[:, 1]

    imp = grouped_permutation_importance(
        fit, X, y, {"a": ["a"], "rest": list("bcde")}, n_repeats=2, n_shuffles=3
    )
    drops = imp.groupby("group").auc_drop.mean()
    assert drops["a"] > 0.2 > drops["rest"]


def test_f1_accepted_prefers_fewer_false_alarms():
    # 8 accepted, 2 rejected; flagging one extra accepted batch costs F1 on the accepted class
    y = np.array([0] * 8 + [1] * 2)
    score = np.array([0.1] * 7 + [0.6] + [0.7, 0.9])
    assert best_threshold(y, score, "f1_accepted") > 0.6
