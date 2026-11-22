"""Process A: datasets behind the interactive analysis page.

    python scripts/eda_process_a.py        # first: features, model table, learning curve
    python scripts/dashboard_process_a.py  # then: this (~8 min)

Writes one JSON file per dataset to outputs/process_A/dashboard/, each as
{"rows": [...], "meta": [...]} with this script's own source in ``meta`` so a
reader can check how every number was made.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from brpmm.config import BATCH_COL, OUTPUTS_DIR, PROCESSED_DIR
from brpmm.data import load_calibration, sensor_columns, to_binary
from brpmm.evaluation import (
    bootstrap_metrics,
    grouped_permutation_importance,
    one_class,
    repeated_cv,
    supervised,
    tuned_threshold,
)
from brpmm.mspc import PCAMonitor
from brpmm.novelty import KNNMonitor
from brpmm.preprocessing import (
    SEGMENT_PARAMS,
    STEADY,
    SegmentParams,
    gross_outlier_mask,
    outlier_scale,
    segment_batches,
)
from brpmm.steady_features import SEP, batch_features, sensor_of, to_model_space

sys.path.insert(0, str(Path(__file__).parent))
from eda_process_a import mode_split  # noqa: E402

SEED = 0
REPEATS = 10
EDA_DIR = OUTPUTS_DIR / "process_A" / "eda"
OUT = OUTPUTS_DIR / "process_A" / "dashboard"
SOURCE = Path(__file__).read_text()
RF = dict(
    n_estimators=300,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=SEED,
    class_weight="balanced_subsample",
)
# Leaderboard: F1 per process with accepted as the positive class, score = mean of A and B.
# Every team scores F1(B) = 1.0, so the expected score is (F1(A) + 1) / 2.
F1_B = 1.0


def clean(v):
    if isinstance(v, float | np.floating):
        return None if math.isnan(v) or math.isinf(v) else round(float(v), 5)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def write(name: str, df: pd.DataFrame, inputs: str) -> None:
    rows = [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    meta = [
        {"label": "dashboard_process_a.py", "value": SOURCE, "type": "code", "lang": "python"},
        {"label": "Input", "value": inputs},
    ]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}))
    print(f"  {name}: {len(rows)} rows")


# --------------------------------------------------------------------------- one-class wrappers
def sklearn_one_class(make, alpha=0.95):
    """Fit a scikit-learn novelty model on accepted batches; higher score = more abnormal."""

    def fit_predict(X_tr, y_tr, X_te):
        sc = StandardScaler().fit(X_tr[y_tr == 0])
        model = make().fit(sc.transform(X_tr[y_tr == 0]))
        train = -model.score_samples(sc.transform(X_tr[y_tr == 0]))
        s = -model.score_samples(sc.transform(X_te))
        return s, (s > np.quantile(train, alpha)).astype(int)

    return fit_predict


def autoencoder(alpha=0.95):
    """Small MLP that reconstructs accepted batches; reconstruction error is the score."""

    def fit_predict(X_tr, y_tr, X_te):
        sc = StandardScaler().fit(X_tr[y_tr == 0])
        A = sc.transform(X_tr[y_tr == 0])
        net = MLPRegressor(
            hidden_layer_sizes=(16,), alpha=1e-2, max_iter=3000, random_state=SEED
        ).fit(A, A)
        err = lambda Z: ((net.predict(Z) - Z) ** 2).mean(axis=1)  # noqa: E731
        s = err(sc.transform(X_te))
        return s, (s > np.quantile(err(A), alpha)).astype(int)

    return fit_predict


def ensemble():
    """Average of the kNN and PCA monitor scores (each is 1.0 at its own 95% limit)."""
    knn, pca = one_class(KNNMonitor, k=10), one_class()

    def fit_predict(X_tr, y_tr, X_te):
        s = (knn(X_tr, y_tr, X_te)[0] + pca(X_tr, y_tr, X_te)[0]) / 2
        return s, (s > 1).astype(int)

    return fit_predict


CORE = {
    "kNN distance": one_class(KNNMonitor, k=10),
    "PCA monitor": one_class(),
    "kNN + PCA average": ensemble(),
    "Random forest": supervised(RandomForestClassifier(**RF)),
}


# --------------------------------------------------------------------------- datasets
def batches(ts, F, raw, lab, y, scale, params):
    """Per batch: out-of-fold scores, share of samples outside the normal envelope."""
    st = ts[(segment_batches(ts, params) == STEADY) & ~gross_outlier_mask(ts, params, scale)]
    sensors = sensor_columns(ts)
    out = pd.DataFrame(index=F.index)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(F, y):
        acc = F.index[tr][y[tr] == 0]
        knn = KNNMonitor(k=10).fit(F.loc[acc]).score(F.iloc[te])
        pca = PCAMonitor().fit(F.loc[acc]).score(F.iloc[te])
        out.loc[F.index[te], "knn_score"] = knn["score"]
        out.loc[F.index[te], "pca_score"] = pca["score"]
        out.loc[F.index[te], "t2_ratio"] = pca["T2_ratio"]
        out.loc[F.index[te], "spe_ratio"] = pca["SPE_ratio"]
        ref = st[st[BATCH_COL].isin(acc)][sensors]
        lo, hi = ref.quantile(0.001), ref.quantile(0.999)
        sub = st[st[BATCH_COL].isin(F.index[te])]
        outside = ((sub[sensors] < lo) | (sub[sensors] > hi)).any(axis=1)
        frac = outside.groupby(sub[BATCH_COL]).mean()
        out.loc[frac.index, "share_outside"] = frac
    out["label"] = lab
    out["mode"] = mode_split(F)(F)
    out["n_steps"] = raw["n_steps"]
    out["n_excluded_rows"] = raw["n_excluded_rows"]
    out["any_sample_outside"] = out["share_outside"] > 0
    write(
        "batches",
        out.reset_index(),
        "data/processed/process_A_features_calibration.parquet, "
        "data/raw/process A/calibration data/calibration_data.csv",
    )


def sensor_corr(ts, raw, params):
    sensors = sensor_columns(ts)
    st = ts[segment_batches(ts, params) == STEADY]
    within = (st[sensors] - st.groupby(BATCH_COL)[sensors].transform("mean")).corr()
    between = raw[[f"{s}{SEP}median" for s in sensors]].corr()
    between.index = between.columns = sensors
    rows = [
        {"kind": kind, "a": a, "b": b, "r": m.loc[a, b]}
        for kind, m in (("within", within), ("between", between))
        for a in sensors
        for b in sensors
    ]
    write(
        "sensor_corr",
        pd.DataFrame(rows),
        "data/raw/process A/calibration data/calibration_data.csv",
    )


def feature_auc():
    uni = pd.read_csv(EDA_DIR / "tables" / "univariate_auc.csv", index_col=0)
    uni["stat"] = [f.split(SEP)[1] if SEP in f else "batch" for f in uni.index]
    uni["sensor"] = [sensor_of(f) if SEP in f else "batch" for f in uni.index]
    uni["significant"] = uni["q_value_bh"] < 0.05
    write(
        "feature_auc",
        uni.reset_index(names="feature")[
            ["feature", "sensor", "stat", "auc", "ci_low", "ci_high", "q_value_bh", "significant"]
        ],
        "outputs/process_A/eda/tables/univariate_auc.csv",
    )


def sensor_impact(F, y):
    groups = {s: [c for c in F.columns if c.startswith(s + SEP)] for s in sensor_columns_from(F)}
    groups["batch length & dropouts"] = ["n_steps", "n_excluded_rows"]

    def fit_knn(X, yy):
        m = KNNMonitor(k=10).fit(X[yy == 0])
        return lambda Z: m.score(Z)["score"].to_numpy()

    def fit_rf(X, yy):
        m = RandomForestClassifier(**RF).fit(X, yy)
        return lambda Z: m.predict_proba(Z)[:, 1]

    imp = {
        name: grouped_permutation_importance(
            fit, F, y, groups, n_repeats=3, n_shuffles=5, seed=SEED
        )
        .groupby("group")
        .auc_drop.agg(["mean", "std"])
        for name, fit in (("knn", fit_knn), ("rf", fit_rf))
    }
    share = pd.read_csv(EDA_DIR / "tables" / "contribution_share_by_sensor.csv", index_col=0)
    share.loc["batch length & dropouts"] = share.loc[["n_steps", "n_excluded_rows"]].sum()
    uni = pd.read_csv(EDA_DIR / "tables" / "univariate_auc.csv", index_col=0)
    uni["group"] = [sensor_of(f) if SEP in f else "batch length & dropouts" for f in uni.index]
    uni["separation"] = (uni.auc - 0.5).abs()
    best = (
        uni.sort_values("separation")
        .groupby("group")
        .tail(1)
        .reset_index(names="feature")
        .set_index("group")
    )
    impact = pd.DataFrame(
        {
            "knn_auc_drop": imp["knn"]["mean"],
            "knn_auc_drop_sd": imp["knn"]["std"],
            "rf_auc_drop": imp["rf"]["mean"],
            "rf_auc_drop_sd": imp["rf"]["std"],
        }
    )
    impact["pca_share_rejected"] = share["rejected"].reindex(impact.index)
    impact["pca_share_accepted"] = share["accepted"].reindex(impact.index)
    impact["best_feature"] = best["feature"].reindex(impact.index)
    impact["best_feature_auc"] = best["auc"].reindex(impact.index)
    write(
        "sensor_impact",
        impact.reset_index(names="sensor"),
        "data/processed/process_A_features_calibration.parquet, outputs/process_A/eda/tables",
    )


def sensor_columns_from(F):
    return sorted({sensor_of(c) for c in F.columns if SEP in c})


def profiles(ts, lab):
    sensors = sensor_columns(ts)
    n = ts.groupby(BATCH_COL)["t"].transform("size")
    d = ts.assign(pos=np.minimum((ts["t"] / n * 50).astype(int), 49))
    pb = d.groupby([BATCH_COL, "pos"])[sensors].mean().reset_index()
    pb["label"] = pb[BATCH_COL].map(lab)
    rows = []
    for s in sensors:
        acc = pb[pb.label == "accepted"].groupby("pos")[s]
        rej = pb[pb.label == "rejected"].groupby("pos")[s]
        q = pd.DataFrame(
            {
                "acc_p10": acc.quantile(0.1),
                "acc_p50": acc.median(),
                "acc_p90": acc.quantile(0.9),
                "rej_p10": rej.quantile(0.1),
                "rej_p50": rej.median(),
                "rej_p90": rej.quantile(0.9),
            }
        )
        q["sensor"], q["fraction_elapsed"] = s, (q.index + 0.5) / 50
        rows.append(q)
    write(
        "profiles",
        pd.concat(rows, ignore_index=True),
        "data/raw/process A/calibration data/calibration_data.csv",
    )


def samples_needed(ts, lab, y, scale):
    """Decide from only the first N steady-state samples of each batch.

    Batch length and the end-of-batch offset are unknown early, so they are left
    out at every N (including 'all') to keep the comparison fair.
    """
    base = SEGMENT_PARAMS["A"]
    early = SegmentParams(trim_start=base.trim_start, trim_end=0, min_steady=20)
    rows = []
    # Starts at 50: below that the 151-step dropout window is dominated by the start-up
    # transient and flags every steady sample.
    for n in (50, 100, 200, 400, 800, 1600, None):
        cut = ts if n is None else ts[ts["t"] < base.trim_start + n]
        params = base if n is None else early
        raw = batch_features(cut, params, exclude=gross_outlier_mask(cut, params, scale)).loc[
            lab.index
        ]
        F = to_model_space(raw).drop(columns=["n_steps"])
        F = F.loc[:, ~F.columns.str.endswith("end_offset")]
        for model, fp in (
            ("kNN distance", one_class(KNNMonitor, k=10)),
            ("PCA monitor", one_class()),
        ):
            r = repeated_cv(fp, F, y, model, n_repeats=5, seed=SEED).metrics.roc_auc
            rows.append(
                {
                    "samples": "all" if n is None else n,
                    "samples_numeric": int(cut.groupby(BATCH_COL).size().median()) - base.trim_start
                    if n is None
                    else n,
                    "model": model,
                    "roc_auc": r.mean(),
                    "roc_auc_sd": r.std(),
                }
            )
    write(
        "samples_needed",
        pd.DataFrame(rows),
        "data/raw/process A/calibration data/calibration_data.csv",
    )


def models(F, y):
    eda = pd.read_csv(EDA_DIR / "tables" / "model_comparison.csv", index_col=0)
    extra = {
        "kNN + PCA average | accepted only": ensemble(),
        "Isolation forest | accepted only": sklearn_one_class(
            lambda: IsolationForest(n_estimators=300, random_state=SEED)
        ),
        "Local outlier factor | accepted only": sklearn_one_class(
            lambda: LocalOutlierFactor(n_neighbors=10, novelty=True)
        ),
        "One-class SVM | accepted only": sklearn_one_class(
            lambda: OneClassSVM(nu=0.05, gamma="scale")
        ),
        "Gaussian mixture (2) | accepted only": sklearn_one_class(
            lambda: GaussianMixture(2, covariance_type="diag", random_state=SEED)
        ),
        "Autoencoder (MLP) | accepted only": autoencoder(),
    }
    new = pd.DataFrame(
        [
            repeated_cv(fp, F, y, n, n_repeats=REPEATS, seed=SEED).summary()
            for n, fp in extra.items()
        ]
    )
    table = pd.concat([eda, new])
    table = table[table.index != "majority (no model)"]
    comparison = pd.DataFrame(
        {
            "model": [n.split(" | ")[0] for n in table.index],
            "variant": [n.split(" | ")[1] if " | " in n else "" for n in table.index],
            "trained_on": [
                "accepted only" if "accepted only" in n else "accepted + rejected labels"
                for n in table.index
            ],
            "roc_auc": table.roc_auc_mean,
            "roc_auc_sd": table.roc_auc_sd,
            "pr_auc": table.pr_auc_mean,
            "balanced_accuracy": table.balanced_accuracy_mean,
            "recall": table.recall_rejected_mean,
            "specificity": table.specificity_mean,
        }
    ).sort_values("roc_auc", ascending=False)
    comparison["name"] = table.loc[comparison.index].index
    write(
        "models",
        comparison.reset_index(drop=True),
        "data/processed/process_A_features_calibration.parquet, "
        "outputs/process_A/eda/tables/model_comparison.csv",
    )
    return comparison


def accepted_metrics(dec: np.ndarray, y: np.ndarray) -> dict:
    """Leaderboard view (accepted = positive) of one repeat's decisions (1 = rejected)."""
    tp = ((dec == 0) & (y == 0)).sum()
    fp = ((dec == 0) & (y == 1)).sum()
    fn = ((dec == 1) & (y == 0)).sum()
    tn = ((dec == 1) & (y == 1)).sum()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": 0.5 * (recall + tn / max(tn + fp, 1)),
    }


def reliability(F, y):
    """F1 / precision / recall with accepted as positive (as the leaderboard scores): the CV
    spread, the spread to expect on one 161-batch test set, and the resulting score range."""
    rows, curves = [], []
    for model, fp in CORE.items():
        for rule, wrapped in (("own limit", fp), ("F1-tuned (nested)", tuned_threshold(fp))):
            res = repeated_cv(wrapped, F, y, model, n_repeats=REPEATS, seed=SEED)
            dec = res.oof_decisions.to_numpy()
            cv = pd.DataFrame([accepted_metrics(d, y) for d in dec])
            boot = bootstrap_metrics(1 - y, 1 - dec, n_test=len(y), seed=SEED)  # accepted = 1
            for metric in ("f1", "precision", "recall", "balanced_accuracy"):
                score = (boot[metric] + F1_B) / 2 if metric == "f1" else None
                rows.append(
                    {
                        "model": model,
                        "threshold_rule": rule,
                        "metric": metric,
                        "cv_mean": cv[metric].mean(),
                        "cv_sd": cv[metric].std(),
                        "test_a_low": boot[metric].quantile(0.025),
                        "test_a_high": boot[metric].quantile(0.975),
                        "score_low": None if score is None else score.quantile(0.025),
                        "score_high": None if score is None else score.quantile(0.975),
                    }
                )
            if rule == "own limit" and model in ("kNN distance", "Random forest"):
                s = res.oof_scores.to_numpy()
                for share in np.linspace(0.02, 0.6, 30):
                    per_rep = [
                        accepted_metrics((r >= np.quantile(r, 1 - share)).astype(int), y) for r in s
                    ]
                    curves.append(
                        {
                            "model": model,
                            "share_flagged": share,
                            **{
                                k: np.mean([m[k] for m in per_rep])
                                for k in ("precision", "recall", "f1")
                            },
                        }
                    )
    write(
        "metric_reliability",
        pd.DataFrame(rows),
        "data/processed/process_A_features_calibration.parquet; score = (F1(A) + F1(B) = 1.0) / 2",
    )
    write(
        "threshold_curve",
        pd.DataFrame(curves),
        "data/processed/process_A_features_calibration.parquet",
    )


def learning():
    summary = json.loads((EDA_DIR / "summary.json").read_text())
    rows = [
        {"model": m.split(" | ")[0], "training_batches": int(n), "roc_auc": v}
        for m, curve in summary["learning_curve"].items()
        for n, v in curve.items()
    ]
    write("learning_curve", pd.DataFrame(rows), "outputs/process_A/eda/summary.json")


def model_options(models_df):
    auc = models_df.set_index("name").roc_auc
    tested = lambda n: f"ROC-AUC {auc[n]:.2f}" if n in auc else "not tested"  # noqa: E731
    rows = [
        (
            "kNN distance",
            "one-class, distance",
            "no",
            "yes",
            "~40+ accepted batches",
            tested("kNN distance | accepted only"),
            "Recommended core: near-top AUC, flat learning curve, explains alarms by nearest normal batches.",
        ),
        (
            "PCA monitor (T², SPE)",
            "one-class, MSPC",
            "no",
            "yes",
            "~100+ accepted batches",
            tested("PCA monitor | accepted only"),
            "Recommended alongside kNN: pharma-standard charts and per-sensor contributions.",
        ),
        (
            "Random forest / gradient boosting",
            "supervised trees",
            "yes",
            "yes (fixed seed)",
            "labelled rejects",
            tested("random forest | class weights"),
            "Weaker than one-class here; keep only if a nested-CV ensemble shows a gain.",
        ),
        (
            "Logistic regression",
            "supervised linear",
            "yes",
            "yes",
            "labelled rejects",
            tested("logistic | class weights"),
            "Too simple for this signal.",
        ),
        (
            "SMOTE / synthetic rejects",
            "resampling",
            "yes",
            "yes (fixed seed)",
            "labelled rejects",
            tested("random forest | SMOTE"),
            "No AUC gain; synthetic failures are hard to justify in GMP.",
        ),
        (
            "Isolation forest, LOF, one-class SVM, GMM",
            "one-class, other",
            "no",
            "yes (fixed seed)",
            "accepted batches",
            ", ".join(
                f"{k.split(' | ')[0]} {auc[k]:.2f}"
                for k in auc.index
                if any(
                    t in k
                    for t in ("Isolation", "Local outlier", "One-class SVM", "Gaussian mixture")
                )
            ),
            "One-class SVM scored highest of all models; worth tuning in nested CV and adding to the "
            "kNN + PCA ensemble. The others score below kNN.",
        ),
        (
            "Autoencoder (small MLP)",
            "one-class, neural",
            "no",
            "yes (fixed seed)",
            "accepted batches",
            tested("Autoencoder (MLP) | accepted only"),
            "Non-linear PCA; no gain at 121 normal batches.",
        ),
        (
            "Diffusion models",
            "generative, neural",
            "no",
            "no (stochastic sampling)",
            "thousands of normal examples, GPU",
            "not tested",
            "Not worth it here: 121 normal batches, a sensor-limited ceiling near AUC 0.85, "
            "stochastic output and hard GMP validation. A research topic, not a scoring gain.",
        ),
        (
            "LSTM / Transformer classifiers",
            "supervised sequence",
            "yes",
            "yes at inference",
            "many labelled batches",
            "not tested",
            "161 labelled batches is far too few to train one without overfitting.",
        ),
        (
            "Time-series foundation models (e.g. Chronos)",
            "pretrained forecasting",
            "no",
            "depends on sampling",
            "large model",
            "not tested",
            "Could give forecast residuals; heavy and unvalidated for GMP; optional later.",
        ),
        (
            "Self-supervised virtual sensors",
            "one-class, regression",
            "no",
            "yes",
            "accepted samples (hundreds of thousands)",
            "not tested yet",
            "Next candidate: predict each sensor from the others on accepted samples; "
            "residuals flag broken relationships. Matches the brief's self-supervised hint.",
        ),
        (
            "LLMs",
            "language",
            "-",
            "no",
            "-",
            "not applicable",
            "Not for classifying numeric time series; at most to word a report from fixed results.",
        ),
    ]
    cols = [
        "approach",
        "category",
        "uses_reject_labels",
        "deterministic",
        "data_it_needs",
        "evidence_here",
        "verdict",
    ]
    write(
        "model_options",
        pd.DataFrame(rows, columns=cols),
        "Assessment by Claude from the measured results in this folder; 'not tested' rows are "
        "judgements, not measurements.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = SEGMENT_PARAMS["A"]
    ts, lab = load_calibration("A")
    raw = pd.read_parquet(PROCESSED_DIR / "process_A_features_calibration.parquet").drop(
        columns="label"
    )
    raw = raw.loc[lab.index]
    F = to_model_space(raw)
    y = to_binary(lab).to_numpy()
    scale = outlier_scale(ts, params)
    print("[1/9] batches")
    batches(ts, F, raw, lab, y, scale, params)
    print("[2/9] sensor correlation")
    sensor_corr(ts, raw, params)
    print("[3/9] feature AUC")
    feature_auc()
    print("[4/9] sensor impact")
    sensor_impact(F, y)
    print("[5/9] profiles")
    profiles(ts, lab)
    print("[6/9] samples needed")
    samples_needed(ts, lab, y, scale)
    print("[7/9] models")
    mdf = models(F, y)
    print("[8/9] metric reliability")
    reliability(F, y)
    print("[9/9] learning curve + options")
    learning()
    model_options(mdf)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
