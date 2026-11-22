"""Accepted-only monitoring: the brief's second solution (not scored on the leaderboard).

    python scripts/monitor.py   # ~3 min -> outputs/<time>_monitor.csv + _monitor.json

Learns "normal" from accepted calibration batches only and flags whatever deviates.
Rejected calibration batches are used only to report how well that works.

Process A  steady-state batch features (start-up/shut-down trimmed, sensor dropouts
           excluded; brpmm.steady_features). Two monitors fitted on accepted batches:
           PCA (Hotelling T2 + SPE, brpmm.mspc) and kNN distance (brpmm.novelty), each
           scaled to 1.0 at its 95 % limit. Alarm when their average exceeds 1.0. The
           reason names the sensor contributing most to the PCA statistic.
Process B  aeration setpoint rule + accepted-only PCA monitor (brpmm.process_b); a
           monitor alarm on a batch the rule accepts is a review flag.
The JSON record holds the cross-validated performance of exactly these rules. The dropout
detector's noise scale is fitted on all calibration batches (no labels used); the B rule's
tolerance and run length were designed after inspecting the 5 calibration rejects.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from brpmm.config import BATCH_COL, LABEL_ACCEPTED, LABEL_REJECTED, OUTPUTS_DIR
from brpmm.data import load_calibration, load_test, to_binary
from brpmm.evaluation import repeated_cv
from brpmm.mspc import PCAMonitor
from brpmm.novelty import KNNMonitor
from brpmm.preprocessing import SEGMENT_PARAMS, gross_outlier_mask, outlier_scale
from brpmm.process_b import ProcessBModel
from brpmm.steady_features import batch_features, sensor_of, to_model_space

KNN_K = 10


def features_a(ts: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    p = SEGMENT_PARAMS["A"]
    return to_model_space(batch_features(ts, p, exclude=gross_outlier_mask(ts, p, scale)))


class MonitorA:
    """PCA + kNN monitors on accepted batches; score = mean of the two limit ratios."""

    def fit(self, X: pd.DataFrame) -> MonitorA:
        self.pca_ = PCAMonitor().fit(X)
        self.knn_ = KNNMonitor(k=KNN_K).fit(X)
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        pca, knn = self.pca_.score(X), self.knn_.score(X)
        contrib = self.pca_.spe_contributions(X) + self.pca_.t2_contributions(X).abs()
        by_sensor = contrib.T.groupby(contrib.columns.map(sensor_of)).sum().T
        out = pd.DataFrame(
            {
                "pca_ratio": pca["score"],
                "knn_ratio": knn["score"],
                "score": (pca["score"] + knn["score"]) / 2,
                "top_sensor": by_sensor.idxmax(axis=1),
            },
            index=X.index,
        )
        out["alarm"] = out["score"] > 1
        return out


def fit_predict_a(X_tr, y_tr, X_te):
    s = MonitorA().fit(X_tr[y_tr == 0]).score(X_te)
    return s["score"].to_numpy(), s["alarm"].to_numpy().astype(int)


def reason_a(r: pd.Series) -> str:
    if r["alarm"]:
        return (
            f"outside normal: {r['score']:.1f}x the limit (PCA {r['pca_ratio']:.1f}x, "
            f"kNN {r['knn_ratio']:.1f}x), mostly {r['top_sensor']}"
        )
    return f"within normal ({r['score']:.2f}x the limit)"


def validate_b(ts: pd.DataFrame, lab: pd.Series) -> dict:
    dec = pd.Series(index=lab.index, dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(lab, lab):
        model = ProcessBModel().fit(ts[ts[BATCH_COL].isin(lab.index[tr])], lab.iloc[tr])
        p = model.predict(ts[ts[BATCH_COL].isin(lab.index[te])])
        dec[p.index] = p["label"]
    caught = int(((dec == LABEL_REJECTED) & (lab == LABEL_REJECTED)).sum())
    return {
        "f1_accepted": round(float(f1_score(lab, dec, pos_label=LABEL_ACCEPTED)), 4),
        "rejects_caught": f"{caught}/{int((lab == LABEL_REJECTED).sum())}",
        "false_alarms": int(((dec == LABEL_REJECTED) & (lab == LABEL_ACCEPTED)).sum()),
    }


def main() -> None:
    record: dict = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds")}

    print("Process A", flush=True)
    ts, lab = load_calibration("A")
    scale = outlier_scale(ts, SEGMENT_PARAMS["A"])  # calibration scale, reused on test
    X_cal = features_a(ts, scale).loc[lab.index]
    X_test = features_a(load_test("A"), scale)[X_cal.columns]
    y = to_binary(lab).to_numpy()
    cv = repeated_cv(fit_predict_a, X_cal, y, "monitor A", n_repeats=5, seed=42).summary()
    record["process_A_cv"] = {
        k: round(float(cv[f"{k}_mean"]), 4)
        for k in ("roc_auc", "recall_rejected", "specificity", "f1_accepted")
    }
    print(f"  calibration CV (trained on accepted only): {record['process_A_cv']}", flush=True)
    a = MonitorA().fit(X_cal[y == 0]).score(X_test)
    a["reason"] = a.apply(reason_a, axis=1)

    print("Process B", flush=True)
    ts_b, lab_b = load_calibration("B")
    record["process_B_cv"] = validate_b(ts_b, lab_b)
    print(f"  calibration CV: {record['process_B_cv']}", flush=True)
    b = ProcessBModel().fit(ts_b, lab_b).predict(load_test("B"))
    b["alarm"] = b["label"] == LABEL_REJECTED

    out = pd.concat(
        [
            a[["alarm", "score", "top_sensor", "reason"]].assign(project="A"),
            b[["alarm", "monitor_score", "monitor_top_sensor", "review", "reason"]]
            .rename(columns={"monitor_score": "score", "monitor_top_sensor": "top_sensor"})
            .assign(project="B"),
        ]
    )
    out.index.name = BATCH_COL
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / f"{stamp}_monitor.csv"
    out.to_csv(path)
    record["test_alarms"] = {p: sorted(g.index[g.alarm]) for p, g in out.groupby("project")}
    record["test_review_B"] = sorted(b.index[b["review"]])
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    for p, g in out.groupby("project"):
        print(f"  {p}: {int(g.alarm.sum())} of {len(g)} test batches flagged", flush=True)
    print(f"monitor -> {path}")


if __name__ == "__main__":
    main()
