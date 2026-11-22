"""Leaderboard files with semi-supervised self-training on the unlabelled test batches.

    python scripts/predict_self_training.py            # all variants, ~6 min
    python scripts/predict_self_training.py --only self_training_subagging_m1.0

Allowed by the organisers (J&J): using the unlabelled test data for training. Nothing is
submitted. Process A uses the submitted pipeline unchanged (brpmm.methods.ensemble and
brpmm.predict.cv_cutoff); only its training data grows:

  1. fit on the 161 calibration batches, cut-off from CV on them (as submitted)
  2. score the 161 test batches; those beyond the cut-off by ``margin`` get that label
     (pseudo-labels); "accepted" variants keep only the confidently accepted ones
  3. refit on calibration + pseudo-labelled test, new CV cut-off, predict all test batches
Process B: the submitted pca_mspc model (brpmm.predict), unchanged.

The "subagging" variant replaces the single fit in steps 1 and 3 by the average of N_BAGS
ensembles, each trained on a stratified 80 % subsample; the cut-off is chosen on the scores
each ensemble gives the batches it left out (out-of-bag), so cut-off and test scores come
from the same kind of model.

Validated in simulation on calibration data (docs/model_results.md A50-A53): F1 0.941 ->
0.944 (both, margin 0.5) and 0.943 (accepted only, margin 1.0, fewest false alarms);
self-training + subagging 0.950 vs 0.943 (seeds 50-53) and 0.950 vs 0.948 (seeds 60-63)
(A58-A63). On the real test it flags only 26 batches where ~40 rejects are expected (the
calibration share, 40 of 161), so with an unlabelled pool as large as the training set it
over-corrects; not submitted.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from brpmm.config import LABEL_REJECTED, OUTPUTS_DIR
from brpmm.data import load_test
from brpmm.evaluate import load_features
from brpmm.features import batch_features
from brpmm.methods import METHODS
from brpmm.predict import predict_process
from brpmm.self_training import self_train
from brpmm.submission import build_submission, load_template, save_submission

VARIANTS = {  # name: (pseudo-labels added, margin in ensemble-score units, subagging)
    "self_training_both_m0.5": ("both", 0.5, False),
    "self_training_accepted_m1.0": ("accepted", 1.0, False),
    "self_training_subagging_m1.0": ("accepted", 1.0, True),
}
N_BAGS, BAG_FRACTION, BAG_SEED = 50, 0.8, 0


def self_train_a(kind: str, margin: float, bags: bool) -> tuple[pd.DataFrame, dict]:
    fit, score, _ = METHODS["ensemble"]
    X, y = load_features("A")
    Xt = batch_features(load_test("A")).reindex(columns=X.columns)
    bag_args = {"n_bags": N_BAGS, "fraction": BAG_FRACTION, "seed": BAG_SEED} if bags else None
    res = self_train(fit, score, X, y, Xt, kind, margin, bag_args)
    out = pd.DataFrame(
        {
            "project": "A",
            "label": res.labels,
            "score": res.scores.round(4),
            "cutoff": round(res.cutoff, 4),
            "pseudo_label": res.pseudo.reindex(Xt.index).fillna(""),
        },
        index=Xt.index,
    )
    info = {
        "first_pass_cutoff": round(res.first_cutoff, 4),
        "pseudo_labelled": len(res.pseudo),
        "pseudo_rejected": int((res.pseudo == LABEL_REJECTED).sum()),
        "final_cutoff": round(res.cutoff, 4),
        "test_rejected": int((out.label == LABEL_REJECTED).sum()),
        # sanity check without labels: the calibration share of rejects applied to the test set
        "expected_rejects": round(len(Xt) * float((y == LABEL_REJECTED).mean())),
    }
    return out, info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only", nargs="*", choices=list(VARIANTS), help="variants to build (default: all)"
    )
    args = ap.parse_args(argv)
    b = predict_process("B")[["project", "label"]]
    for name in args.only or VARIANTS:
        kind, margin, bags = VARIANTS[name]
        a, info = self_train_a(kind, margin, bags)
        sub = build_submission({"A": a["label"], "B": b["label"]}, load_template())
        path = save_submission(sub, name)
        details = pd.concat([a, b])
        details.index.name = "batch_id"
        details.to_csv(path.with_name(path.stem + "_details.csv"))
        path.with_name(path.stem + "_run.json").write_text(
            json.dumps(
                {
                    "variant": name,
                    "pseudo_labels": kind,
                    "margin": margin,
                    "subagging": {"bags": N_BAGS, "fraction": BAG_FRACTION, "seed": BAG_SEED}
                    if bags
                    else None,
                    **info,
                    "submitted": False,
                },
                indent=2,
            )
        )
        print(f"{name}: {info} -> {path.relative_to(OUTPUTS_DIR.parent)}", flush=True)


if __name__ == "__main__":
    main()
