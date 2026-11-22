"""Process A final selection: the submitted ensemble vs a re-implementation of its members.

    python scripts/select_final_a.py   # ~2 min -> outputs/process_A/final_selection.json

Same fresh splits (StratifiedKFold(5, shuffle, s), s = 50..53, never used to choose
anything) and the same inner 3-fold threshold protocol for both. Both use the same three
model types and settings (RBF SVM, kNN on PCA components, one-class SVM on 22 median+std
features); the second re-implements them and combines them differently:
  submitted  brpmm.methods.ensemble: member scores standardised with out-of-fold scores on
             the training batches, mean, cut-off chosen in inner CV (brpmm.evaluate)
  rank       brpmm.models.team_members + rank_average: mean of within-set percentile ranks
Also reports the chance that one 161-batch test draw beats the current top F1(A).
"""

from __future__ import annotations

import json
import warnings

from brpmm.config import OUTPUTS_DIR
from brpmm.data import load_calibration, to_binary
from brpmm.evaluate import load_features
from brpmm.evaluation import (
    FRESH_SEEDS,
    leaderboard_summary,
    repeated_cv,
    submitted_ensemble_decisions,
    tuned_threshold,
)
from brpmm.models import median_std_features, rank_average, team_members

warnings.filterwarnings("ignore")
TOP_OF_LEADERBOARD = 0.939  # Team 13, F1(A)


def main() -> None:
    ts, lab = load_calibration("A")
    X, labels = load_features("A")
    if not X.index.equals(lab.index):  # both decision arrays below are in this batch order
        raise ValueError("feature rows and labels are in a different batch order")
    y = to_binary(lab).to_numpy()

    submitted = leaderboard_summary(submitted_ensemble_decisions(X, labels), y, TOP_OF_LEADERBOARD)
    f22 = median_std_features(ts).loc[lab.index]
    fp = tuned_threshold(rank_average(team_members(list(f22.columns))), inner_splits=3, seed=7)
    res = repeated_cv(fp, f22, y, "rank", n_repeats=len(FRESH_SEEDS), seed=FRESH_SEEDS[0])
    rank = leaderboard_summary(res.oof_decisions.to_numpy(), y, TOP_OF_LEADERBOARD)

    record = {
        "protocol": f"StratifiedKFold(5, shuffle, seed) for seed in {list(FRESH_SEEDS)}; "
        "inner 3-fold threshold",
        "top_of_leaderboard_f1_A": TOP_OF_LEADERBOARD,
        "maria_ensemble_native": submitted,
        "team_rank_average_ours": rank,
        "selected": "maria_ensemble_native"
        if submitted["f1_accepted"] >= rank["f1_accepted"]
        else "team_rank_average_ours",
    }
    path = OUTPUTS_DIR / "process_A" / "final_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
