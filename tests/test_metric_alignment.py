"""Both code paths must score exactly what the leaderboard scores (F1, accepted positive)."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from brpmm.config import LABEL_ACCEPTED, LABEL_REJECTED
from brpmm.evaluation import binary_metrics
from brpmm.metrics import leaderboard_f1


def test_leaderboard_f1_equals_f1_accepted_on_random_predictions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 161)  # 1 = rejected, as brpmm.data.to_binary encodes
        d = rng.integers(0, 2, 161)
        as_labels = lambda v: pd.Series(np.where(v == 1, LABEL_REJECTED, LABEL_ACCEPTED))  # noqa: E731
        ours = binary_metrics(y, d.astype(float), d)["f1_accepted"]
        assert np.isclose(leaderboard_f1(as_labels(y), as_labels(d)), ours)
        assert np.isclose(ours, f1_score(y, d, pos_label=0))


def test_reference_leaderboard_entry():
    # TEMPLATE row: accuracy 0.845, precision 0.88, recall 0.917 -> 110 / 15 / 10 / 26
    y = np.array([0] * 110 + [1] * 15 + [0] * 10 + [1] * 26)
    d = np.array([0] * 110 + [0] * 15 + [1] * 10 + [1] * 26)
    assert round(binary_metrics(y, d.astype(float), d)["f1_accepted"], 3) == 0.898
