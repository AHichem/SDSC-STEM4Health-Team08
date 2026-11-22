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


def test_tuned_threshold_learns_a_working_cutoff():
    X, y = toy()
    res = repeated_cv(tuned_threshold(supervised(LogisticRegression())), X, y, "t", n_repeats=2)
    assert res.summary()["f1_accepted_mean"] > 0.85


def test_tuned_threshold_uses_training_data_only():
    # the cut-off must not depend on the scored batches: scoring a subset of the test rows
    # gives the same decisions for those rows as scoring all of them
    X, y = toy()
    fp = tuned_threshold(supervised(LogisticRegression()))
    _, all_dec = fp(X.iloc[:90], y[:90], X.iloc[90:])
    _, part_dec = fp(X.iloc[:90], y[:90], X.iloc[90:100])
    np.testing.assert_array_equal(all_dec[:10], part_dec)


def test_harness_seed_is_restored_even_on_error():
    import brpmm.evaluate as harness
    from brpmm.evaluation import harness_seed

    before = harness.SEED
    try:
        with harness_seed(before + 7):
            assert before + 7 == harness.SEED
            raise RuntimeError
    except RuntimeError:
        pass
    assert before == harness.SEED


def test_leaderboard_summary_counts_and_f1():
    from brpmm.evaluation import leaderboard_summary

    y = np.array([0] * 8 + [1] * 2)
    dec = np.array([[0] * 7 + [1] + [1, 0]])  # one false alarm, one reject caught, one missed
    out = leaderboard_summary(dec, y, top_f1=0.9)
    assert out["bad_caught_of_40"] == 1.0 and out["good_thrown_away_of_121"] == 1.0
    assert out["f1_accepted"] == round(2 * 7 / (2 * 7 + 1 + 1), 4)


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


def test_subagged_uses_out_of_bag_scores_and_is_reproducible():
    from brpmm.config import LABEL_ACCEPTED, LABEL_REJECTED
    from brpmm.evaluation import subagged

    X, y01 = toy()
    y = pd.Series(np.where(y01 == 1, LABEL_REJECTED, LABEL_ACCEPTED), index=X.index)
    seen = []

    def fit(Xf, yf):
        seen.append(set(Xf.index))
        return LogisticRegression().fit(Xf, yf == LABEL_REJECTED)

    def score(model, Xs):
        return model.decision_function(Xs)

    cut, s = subagged(fit, score, X, y, X.iloc[:10], n_bags=8, fraction=0.8, seed=1)
    assert len(seen) == 8 and all(len(b) == int(0.8 * 90) + int(0.8 * 30) for b in seen)
    np.testing.assert_allclose(subagged(fit, score, X, y, X.iloc[:10], 8, 0.8, 1)[1], s)
    pred = s > cut
    assert pred[y01[:10] == 1].mean() > pred[y01[:10] == 0].mean()  # signal survives the cut


def test_self_training_adds_only_confident_accepted_batches():
    from brpmm.config import LABEL_ACCEPTED, LABEL_REJECTED
    from brpmm.self_training import self_train

    X, y01 = toy(160)
    y = pd.Series(np.where(y01 == 1, LABEL_REJECTED, LABEL_ACCEPTED), index=X.index)
    Xl, yl, Xu = X.iloc[:120], y.iloc[:120], X.iloc[120:]

    def fit(Xf, yf):
        return LogisticRegression().fit(Xf, yf == LABEL_REJECTED)

    def score(model, Xs):
        return model.decision_function(Xs)

    res = self_train(fit, score, Xl, yl, Xu, kind="accepted", margin=1.0)
    assert (res.pseudo == LABEL_ACCEPTED).all() and 0 < len(res.pseudo) < len(Xu)
    assert res.labels.index.equals(Xu.index)
    caught = (res.labels == LABEL_REJECTED)[y01[120:] == 1].mean()
    assert caught > (res.labels == LABEL_REJECTED)[y01[120:] == 0].mean()
