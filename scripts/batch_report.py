"""Per-batch decision report: label, calibrated confidence, borderline flag and the evidence.

    python scripts/batch_report.py   # ~3 min -> outputs/<time>_batch_report.csv and .md

Labels are those of the submitted file (process A: self-training, accepted only, margin 1.0;
process B: the submitted model, identical to the aeration rule). Nothing here changes a label;
it says how sure each label is, why, and which batches a person should look at.

Process A, per test batch:
  p_rejected   P(rejected) from Platt scaling of the margin (score - cut-off), fitted on the
               out-of-fold margins of the calibration batches (5 repeats of 5-fold CV)
  light        member agreement (brpmm.predict.traffic_light): green / amber / red
  similar      the 5 most similar calibration batches (standardised 22 features) and how many
               of them were rejected: a precedent a person can open and compare
  drivers      the features furthest from normal (brpmm.methods explain)
  monitor      the accepted-only solution (scripts/monitor.py): outside normal or not, top sensor
  note         possible sensor fault (gross-outlier rows, brpmm.preprocessing) or a new pattern
               (rejected, but at most 1 of the 5 most similar past batches was rejected)
  review       yes if 0.2 < p_rejected < 0.8, the members disagree (amber), or there is a note
Process B: the rule's evidence (largest deviation of aeration from its own level, how long,
from which step), a confidence from the distance to the rule's limits, and the monitor's
review flag for batches unlike any accepted batch.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from brpmm.config import (
    BATCH_COL,
    CV_FOLDS,
    CV_REPEATS,
    LABEL_REJECTED,
    OUTPUTS_DIR,
    SEED,
)
from brpmm.data import load_calibration, load_test
from brpmm.evaluate import choose_cutoff, load_features
from brpmm.features import batch_features
from brpmm.methods import METHODS, _median_std_columns
from brpmm.predict import predict_process
from brpmm.preprocessing import SEGMENT_PARAMS, outlier_scale, quality_report
from brpmm.process_b import ProcessBModel
from brpmm.self_training import self_train

REVIEW_BAND = (0.2, 0.8)
N_SIMILAR = 5


def oof_margins(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold margins (score - that repeat's cut-off) of the submitted ensemble."""
    fit, score, _ = METHODS["ensemble"]
    margins, ys = [], []
    for r in range(CV_REPEATS):
        oof = np.empty(len(X))
        for tr, te in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + r).split(X, y):
            oof[te] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
        margins.append(oof - choose_cutoff(oof, y))
        ys.append((y == LABEL_REJECTED).to_numpy())
    return np.concatenate(margins), np.concatenate(ys)


def reliability(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    bins = pd.cut(p, [0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0], include_lowest=True)
    return (
        pd.DataFrame({"bin": bins, "p": p, "y": y})
        .groupby("bin", observed=True)
        .agg(batches=("y", "size"), mean_p=("p", "mean"), share_rejected=("y", "mean"))
        .round(2)
    )


def latest_monitor() -> pd.DataFrame | None:
    files = sorted(OUTPUTS_DIR.glob("*_monitor.csv"))
    return pd.read_csv(files[-1], index_col=BATCH_COL) if files else None


def report_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y = load_features("A")
    Xt = batch_features(load_test("A")).reindex(columns=X.columns)
    fit, score, _ = METHODS["ensemble"]
    a = self_train(fit, score, X, y, Xt, kind="accepted", margin=1.0)  # the submitted file
    base = predict_process("A")

    m, yr = oof_margins(X, y)
    platt = LogisticRegression(C=1e6).fit(m[:, None], yr)
    p_oof = platt.predict_proba(m[:, None])[:, 1]
    p = platt.predict_proba((a.scores - a.cutoff).to_numpy()[:, None])[:, 1]

    cols = _median_std_columns(X)
    sc = StandardScaler().fit(X[cols])
    nn = NearestNeighbors(n_neighbors=N_SIMILAR).fit(sc.transform(X[cols]))
    _, idx = nn.kneighbors(sc.transform(Xt[cols]))
    rejected = (y == LABEL_REJECTED).to_numpy()

    out = pd.DataFrame(index=Xt.index)
    out["project"] = "A"
    out["label"] = a.labels
    out["p_rejected"] = p.round(2)
    out["confidence"] = np.where(a.labels == LABEL_REJECTED, p, 1 - p).round(2)
    out["light"] = base["light"]
    out["similar_rejected"] = [f"{rejected[i].sum()}/{N_SIMILAR}" for i in idx]
    out["similar_batches"] = [", ".join(X.index[i]) for i in idx]
    out["drivers"] = base["top_reasons"]
    mon = latest_monitor()
    if mon is not None:
        mon = mon[mon["project"] == "A"].reindex(Xt.index)
        out["monitor"] = np.where(
            mon["alarm"], "outside normal (" + mon["top_sensor"] + ")", "normal"
        )
    # Beyond the label: a sensor fault is not a process fault, and a reject unlike every past
    # reject may be a new failure mode. Both go to a person.
    ts_test = load_test("A")
    scale = outlier_scale(load_calibration("A")[0], SEGMENT_PARAMS["A"])
    gross = quality_report(ts_test, SEGMENT_PARAMS["A"], scale)["n_gross_outlier_rows"]
    n_similar_rejected = np.array([rejected[i].sum() for i in idx])
    novel = (out["label"] == LABEL_REJECTED).to_numpy() & (n_similar_rejected <= 1)
    out["note"] = [
        "; ".join(
            filter(
                None,
                [
                    f"possible sensor fault: {n} gross-outlier rows" if n else "",
                    "new pattern: no close past reject" if nv else "",
                ],
            )
        )
        for n, nv in zip(gross.reindex(out.index).fillna(0).astype(int), novel, strict=True)
    ]
    out["review"] = np.where(
        ((p > REVIEW_BAND[0]) & (p < REVIEW_BAND[1]))
        | (out["light"] == "amber")
        | (out["note"] != ""),
        "yes",
        "",
    )
    out["reason"] = [
        f"{r.label}: P(rejected) {r.p_rejected:.2f}; models {r.light}; "
        f"{r.similar_rejected} most similar past batches rejected; drivers {r.drivers.split(';')[0]}"
        for r in out.itertuples()
    ]
    return out, reliability(p_oof, yr)


def report_b() -> pd.DataFrame:
    ts, lab = load_calibration("B")
    model = ProcessBModel().fit(ts, lab)
    b = model.predict(load_test("B"))
    submitted = predict_process("B")["label"]
    assert (submitted.reindex(b.index) == b["label"]).all(), "B: rule and submitted model differ"
    tol, run = model.rule.rel_tolerance, model.rule.min_run
    # distance to the rule's limits: deviation vs 10 %, run length vs 100 samples
    strength = np.where(
        b["label"] == LABEL_REJECTED,
        np.minimum(b["max_rel_deviation"] / tol, b["longest_run"] / run),
        tol / b["max_rel_deviation"].clip(lower=1e-6),
    )
    out = pd.DataFrame(index=b.index)
    out["project"] = "B"
    out["label"] = b["label"]
    out["confidence"] = np.where(strength >= 2, "high", "moderate")
    out["evidence"] = [
        f"max aeration deviation {d:.2%} (limit {tol:.0%}); longest run {int(n)} samples (limit {run})"
        for d, n in zip(b["max_rel_deviation"], b["longest_run"], strict=True)
    ]
    out["review"] = np.where(b["review"], "yes", "")
    out["reason"] = b["reason"]
    return out


def main() -> None:
    a, rel = report_a()
    b = report_b()
    stem = OUTPUTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_batch_report"
    pd.concat([a, b]).rename_axis(BATCH_COL).to_csv(stem.with_suffix(".csv"))
    rel.to_csv(stem.with_name(stem.name + "_reliability.csv"))
    rev_a, rev_b = a[a["review"] == "yes"], b[b["review"] == "yes"]
    lines = [
        "# Batch decision report (test set)",
        "",
        f"Process A: {int((a.label == LABEL_REJECTED).sum())} rejected of {len(a)}; "
        f"{len(rev_a)} sent to review (P(rejected) between {REVIEW_BAND[0]} and "
        f"{REVIEW_BAND[1]}, the models disagree, a possible sensor fault or a new pattern); "
        f"{int((a.review == '').sum())} decided without review.",
        f"Process B: {int((b.label == LABEL_REJECTED).sum())} rejected of {len(b)}; "
        f"{len(rev_b)} sent to review (unlike any accepted batch, though the rule accepts them).",
        "",
        "## Is P(rejected) honest? Calibration batches, out of fold",
        "",
        rel.to_markdown(),
        "",
        "## Process A: batches for review",
        "",
        rev_a.sort_values("p_rejected", ascending=False)[
            ["label", "p_rejected", "light", "similar_rejected", "note", "drivers"]
        ].to_markdown(),
        "",
        "## Process B: rejected and review",
        "",
        b[(b.label == LABEL_REJECTED) | (b.review == "yes")][
            ["label", "confidence", "reason"]
        ].to_markdown(),
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
