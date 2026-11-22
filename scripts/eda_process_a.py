"""Process A: data understanding and model-selection evidence.

    python scripts/eda_process_a.py                  # full run (~5-10 min)
    python scripts/eda_process_a.py --repeats 3 --permutations 20   # quick

Writes figures, tables, summary.json and REPORT.md to outputs/process_A/eda/,
and the batch feature tables to data/processed/. Every number in the report is
computed here; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brpmm.config import BATCH_COL, OUTPUTS_DIR, PROCESSED_DIR
from brpmm.data import TIME_COL, load_calibration, load_test, sensor_columns, to_binary
from brpmm.evaluation import learning_curve, one_class, permutation_test, repeated_cv, supervised
from brpmm.mspc import PCAMonitor
from brpmm.novelty import KNNMonitor
from brpmm.plotting import (
    AQUA,
    BLUE,
    DIVERGING,
    LABEL_COLORS,
    MUTED,
    ORANGE,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
    save,
)
from brpmm.preprocessing import (
    SEGMENT_PARAMS,
    STEADY,
    gross_outlier_mask,
    outlier_scale,
    quality_report,
    segment_batches,
)
from brpmm.steady_features import batch_features, sensor_of, to_model_space

PROCESS = "A"
SEED = 0
# One-class models carried into the permutation test and learning curve (slots 2-3)
ONE_CLASS_SERIES = {
    "PCA monitor | accepted only": ORANGE,
    "kNN distance | accepted only": AQUA,
}


# --------------------------------------------------------------------------- helpers
def fast_auc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise ROC-AUC (P[rejected value > accepted value]) via ranks."""
    ranks = stats.rankdata(X, axis=0)
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def scatter(ax, x, y, color, label, marker="o", size=36):
    ring = {"edgecolors": SURFACE, "linewidths": 0.8} if marker == "o" else {}
    ax.scatter(x, y, s=size, c=color, marker=marker, label=label, zorder=3, **ring)


def clip_view(ax, pts: np.ndarray, names) -> None:
    """Zoom to the bulk of the points; name the ones left off-scale."""
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    pad = (hi - lo) * 0.15
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    off = ((pts < lo) | (pts > hi)).any(axis=1)
    if off.any():
        ax.text(
            0.01,
            0.01,
            "off-scale: " + ", ".join(np.asarray(names)[off]),
            transform=ax.transAxes,
            fontsize=8,
            color=TEXT_SECONDARY,
        )


def mode_split(F_fit: pd.DataFrame) -> tuple:
    """Unsupervised 2-mode split on PC1 (labels not used). Returns a predict function."""
    sc = StandardScaler().fit(F_fit)
    pc1 = PCA(1).fit(sc.transform(F_fit))
    km = KMeans(2, n_init=10, random_state=SEED).fit(pc1.transform(sc.transform(F_fit)))
    # name modes by batch length so "L" (long) / "S" (short) is stable across runs
    lengths = pd.Series(F_fit["n_steps"].to_numpy()).groupby(km.labels_).median()
    names = {int(lengths.idxmax()): "L", int(lengths.idxmin()): "S"}

    def predict(X: pd.DataFrame) -> np.ndarray:
        return np.array([names[k] for k in km.predict(pc1.transform(sc.transform(X)))])

    return predict


def per_mode_monitor(n_components=0.8):
    """PCA monitor fitted separately on the accepted batches of each operating mode."""

    def fit_predict(X_tr, y_tr, X_te):
        predict = mode_split(X_tr)
        m_tr, m_te = predict(X_tr), predict(X_te)
        score, decision = np.zeros(len(X_te)), np.zeros(len(X_te), dtype=int)
        for mode in np.unique(m_te):
            mon = PCAMonitor(n_components=n_components).fit(X_tr[(m_tr == mode) & (y_tr == 0)])
            s = mon.score(X_te[m_te == mode])
            score[m_te == mode], decision[m_te == mode] = s["score"], s["alarm"]
        return score, decision

    return fit_predict


# --------------------------------------------------------------------------- steps
def data_overview(ts, te, lab, params, out, fig_dir, tab_dir):
    scale = outlier_scale(ts, params)
    q_cal = quality_report(ts, params, scale).join(lab)
    q_test = quality_report(te, params, scale)
    q_cal.to_csv(tab_dir / "quality_calibration.csv")
    q_test.to_csv(tab_dir / "quality_test.csv")
    flagged = q_cal[q_cal.n_gross_outlier_rows > 0]
    lengths = q_cal.n_steps
    bins = pd.cut(
        lengths,
        [0, 2300, 2700, 2800, np.inf],
        right=False,
        labels=["<2300", "2300-2699", "2700-2799", ">=2800"],
    )
    by_len = q_cal.groupby(bins, observed=False).label.agg(
        n="size", rejected=lambda s: int((s == "rejected").sum())
    )
    by_len["reject_rate"] = (by_len.rejected / by_len.n).round(3)
    by_len.to_csv(tab_dir / "length_vs_label.csv")
    out["data"] = {
        "n_batches_calibration": int(len(lab)),
        "n_batches_test": int(te[BATCH_COL].nunique()),
        "n_rows_calibration": int(len(ts)),
        "n_rows_test": int(len(te)),
        "n_sensors": len(sensor_columns(ts)),
        "n_accepted": int((lab == "accepted").sum()),
        "n_rejected": int((lab == "rejected").sum()),
        "reject_rate": round(float((lab == "rejected").mean()), 3),
        "missing_values": int(q_cal.n_missing.sum() + q_test.n_missing.sum()),
        "stuck_sensor_max_run": int(
            max(q_cal.max_identical_run.max(), q_test.max_identical_run.max())
        ),
        "length_min_max_median": [int(lengths.min()), int(lengths.max()), int(lengths.median())],
        "short_batches_calibration": q_cal.index[q_cal.short_batch].tolist(),
        "gross_outlier_batches_calibration": {
            b: {"rows": int(r.n_gross_outlier_rows), "label": r.label}
            for b, r in flagged.iterrows()
        },
        "gross_outlier_batches_test": int((q_test.n_gross_outlier_rows > 0).sum()),
        "length_vs_label": by_len.reset_index().astype({"n_steps": str}).to_dict("records"),
    }

    # fig: batch length by label
    fig, ax = plt.subplots(figsize=(7, 2.6))
    rng = np.random.default_rng(SEED)
    for i, (name, vals) in enumerate(
        [
            ("accepted", lengths[lab == "accepted"]),
            ("rejected", lengths[lab == "rejected"]),
            ("test", q_test.n_steps),
        ]
    ):
        color = LABEL_COLORS.get(name, MUTED)
        scatter(ax, vals, i + rng.uniform(-0.18, 0.18, len(vals)), color, name, size=20)
    ax.set_yticks([0, 1, 2], ["accepted", "rejected", "test (unlabelled)"])
    ax.set_xlabel("batch length (time steps)")
    ax.set_title(
        "Batch length: most batches run ~2,720 steps; short batches are rejected more often"
    )
    ax.grid(axis="y", visible=False)
    save(fig, fig_dir / "01_batch_length.png")
    return scale


def transients(ts, params, out, fig_dir):
    sensors = sensor_columns(ts)
    slow = ["density", "temperature_1", "temperature_2", "temperature_3", "temperature_4"]
    n = ts.groupby(BATCH_COL)[TIME_COL].transform("size")
    u = ts[TIME_COL] / n
    mid = ts[(u > 0.3) & (u < 0.7)].groupby(BATCH_COL)[sensors]
    med = mid.median().reindex(ts[BATCH_COL]).to_numpy()
    sd = mid.std().reindex(ts[BATCH_COL]).to_numpy()
    z = pd.DataFrame(np.abs((ts[sensors].to_numpy() - med) / sd), columns=sensors)[slow].mean(
        axis=1
    )
    ref = float(z[((u > 0.3) & (u < 0.7)).to_numpy()].mean())
    from_end = (n - 1 - ts[TIME_COL]).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    profiles = {}
    # Right panel runs right-to-left (0 = batch end), so labels sit on the far side of the line.
    for ax, pos, title, trim, ha in [
        (axes[0], ts[TIME_COL].to_numpy(), "steps since batch start", params.trim_start, "left"),
        (axes[1], from_end, "steps before batch end", params.trim_end, "right"),
    ]:
        keep = pos <= 600
        prof = z[keep].groupby(pos[keep] // 10 * 10).mean()
        profiles[title] = prof
        ax.plot(prof.index, prof.values, color=BLUE)
        ax.axhline(ref, color=MUTED, lw=1)
        ax.axvline(trim, color=TEXT_SECONDARY, lw=1)
        ax.text(
            trim + 8,
            prof.max() * 0.9,
            f" trim = {trim} steps ",
            color=TEXT_SECONDARY,
            fontsize=8,
            ha=ha,
        )
        ax.text(
            0.98,
            ref,
            "steady reference",
            color=MUTED,
            fontsize=8,
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )
        ax.set_xlabel(title)
    axes[1].invert_xaxis()
    axes[0].set_ylabel("mean |z| of slow sensors\n(vs the batch's own steady level)")
    fig.suptitle(
        "Start-up and shut-down transients: excluded from steady-state statistics",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    save(fig, fig_dir / "02_transients.png")
    start = profiles["steps since batch start"]
    out["transients"] = {
        "steady_reference_abs_z": round(ref, 2),
        "abs_z_at_start": round(float(start.iloc[0]), 2),
        "abs_z_at_trim_start": round(float(start.loc[params.trim_start]), 2),
        "abs_z_at_end": round(float(profiles["steps before batch end"].iloc[0]), 2),
        "trim_start": params.trim_start,
        "trim_end": params.trim_end,
    }


def sensor_profiles(ts, lab, params, fig_dir):
    sensors = sensor_columns(ts)
    n = ts.groupby(BATCH_COL)[TIME_COL].transform("size")
    d = ts.assign(bin=np.minimum((ts[TIME_COL] / n * 50).astype(int), 49))
    per_batch = d.groupby([BATCH_COL, "bin"])[sensors].mean().reset_index()
    per_batch["label"] = per_batch[BATCH_COL].map(lab)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.5))
    for ax, s in zip(axes.flat, sensors, strict=False):
        for name, color in LABEL_COLORS.items():
            g = per_batch[per_batch.label == name].groupby("bin")[s]
            x = g.median().index / 50
            ax.fill_between(x, g.quantile(0.25), g.quantile(0.75), color=color, alpha=0.15, lw=0)
            ax.plot(x, g.median(), color=color, lw=1.5, label=name)
        ax.set_title(s, fontsize=9)
        ax.tick_params(labelsize=7)
    last = axes.flat[-1]
    last.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    last.legend(handles, labels, loc="center", title="median and IQR band\nover normalised time")
    fig.suptitle(
        "Sensor trajectories (x = fraction of batch elapsed): differences are level shifts,"
        " not a time window",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, fig_dir / "03_sensor_profiles.png")


def sensor_correlation(ts, params, out, fig_dir):
    sensors = sensor_columns(ts)
    st = ts[segment_batches(ts, params) == STEADY]
    cc = (st[sensors] - st.groupby(BATCH_COL)[sensors].transform("mean")).corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cc, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(sensors)), sensors, rotation=60, ha="right")
    ax.set_yticks(range(len(sensors)), sensors)
    ax.grid(False)
    r = cc.to_numpy()
    for i in range(len(sensors)):
        for j in range(len(sensors)):
            if i != j and abs(r[i, j]) >= 0.5:
                ax.text(
                    j,
                    i,
                    f"{r[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(r[i, j]) > 0.75 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.8, label="within-batch correlation")
    ax.set_title("Within-batch sensor correlation (|r| >= 0.5 labelled)")
    save(fig, fig_dir / "04_sensor_correlation.png")
    pairs = [(a, b, cc.loc[a, b]) for i, a in enumerate(sensors) for b in sensors[i + 1 :]]
    out["sensor_correlation_top"] = [
        {"pair": f"{a} ~ {b}", "r": round(float(v), 2)}
        for a, b, v in sorted(pairs, key=lambda p: -abs(p[2]))[:6]
    ]


def univariate(F, y, out, fig_dir, tab_dir, n_boot=500):
    X = F.to_numpy()
    auc = fast_auc(X, y)
    rng = np.random.default_rng(SEED)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    boots = np.array(
        [
            fast_auc(
                X[np.r_[rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))]],
                np.r_[np.zeros(len(idx0), int), np.ones(len(idx1), int)],
            )
            for _ in range(n_boot)
        ]
    )
    p = np.array([stats.mannwhitneyu(X[y == 1, j], X[y == 0, j]).pvalue for j in range(X.shape[1])])
    q = stats.false_discovery_control(p)
    uni = pd.DataFrame(
        {
            "auc": auc,
            "ci_low": np.quantile(boots, 0.025, axis=0),
            "ci_high": np.quantile(boots, 0.975, axis=0),
            "p_value": p,
            "q_value_bh": q,
            "sensor": [sensor_of(c) for c in F.columns],
        },
        index=F.columns,
    )
    uni["separation"] = (uni.auc - 0.5).abs()
    uni = uni.sort_values("separation", ascending=False)
    uni.to_csv(tab_dir / "univariate_auc.csv")

    top = uni.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ypos = np.arange(len(top))
    ax.hlines(ypos, top.ci_low, top.ci_high, color=BLUE, lw=1.5)
    sig = top.q_value_bh < 0.05
    ax.scatter(top.auc[sig], ypos[sig], s=40, color=BLUE, zorder=3, label="FDR q < 0.05")
    ax.scatter(
        top.auc[~sig],
        ypos[~sig],
        s=40,
        facecolors=SURFACE,
        edgecolors=BLUE,
        zorder=3,
        label="not significant after FDR",
    )
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.text(0.5, len(top) - 0.3, " no separation", color=MUTED, fontsize=8)
    ax.set_yticks(ypos, top.index, fontsize=8)
    ax.set_xlabel("ROC-AUC of the single feature (>0.5: higher in rejected), 95% bootstrap CI")
    ax.set_title("Top 20 single features: individually weak (best AUC ~0.7)")
    ax.legend(loc="lower right")
    save(fig, fig_dir / "05_univariate_auc.png")

    corr = F.corr().abs().to_numpy()
    iu = np.triu_indices_from(corr, 1)
    out["univariate"] = {
        "n_features": int(F.shape[1]),
        "n_significant_fdr_0.05": int((uni.q_value_bh < 0.05).sum()),
        "best": [
            {
                "feature": f,
                "auc": round(float(r.auc), 3),
                "ci": [round(float(r.ci_low), 3), round(float(r.ci_high), 3)],
                "q": float(f"{r.q_value_bh:.2g}"),
            }
            for f, r in uni.head(8).iterrows()
        ],
        "feature_pairs_abs_r_gt_0.9": int((corr[iu] > 0.9).sum()),
        "n_feature_pairs": int(len(iu[0])),
    }
    return uni


def pca_structure(F, lab, out, fig_dir, tab_dir):
    Z = StandardScaler().fit_transform(F)
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k80 = int(np.searchsorted(cum, 0.8) + 1)
    scores = pca.transform(Z)[:, :2]
    loadings = pd.DataFrame(pca.components_[:3].T, index=F.columns, columns=["PC1", "PC2", "PC3"])
    loadings.reindex(loadings.PC1.abs().sort_values(ascending=False).index).to_csv(
        tab_dir / "pca_loadings.csv"
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, cum * 100, color=BLUE, marker="o", ms=3)
    ax.axhline(80, color=MUTED, lw=1)
    ax.axvline(k80, color=TEXT_SECONDARY, lw=1)
    ax.text(k80 + 1, 40, f"{k80} components\nexplain 80%", color=TEXT_SECONDARY, fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative variance explained (%)")
    ax.set_title("PCA scree: information is spread over many directions")
    save(fig, fig_dir / "06_pca_scree.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color in LABEL_COLORS.items():
        m = (lab == name).to_numpy()
        scatter(ax, scores[m, 0], scores[m, 1], color, f"{name} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}% of variance)")
    ax.set_title("PC1 vs PC2 (unsupervised): two operating modes, classes overlap")
    clip_view(ax, scores, F.index)
    ax.legend(loc="upper right")
    save(fig, fig_dir / "07_pca_scores.png")

    y = to_binary(lab).to_numpy()
    pc_auc = fast_auc(pca.transform(Z)[:, :10], y)
    out["pca"] = {
        "components_for_80pct": k80,
        "components_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
        "explained_pc1_pc2_pc3": [round(float(v), 3) for v in pca.explained_variance_ratio_[:3]],
        "auc_of_pc1_to_pc10": [round(float(a), 2) for a in pc_auc],
        "top_pc1_loadings": loadings.PC1.abs()
        .sort_values(ascending=False)
        .head(6)
        .round(3)
        .to_dict(),
    }


def operating_modes(F, Ft, raw_cal, lab, out, tab_dir):
    predict = mode_split(F)
    mode, mode_test = predict(F), predict(Ft)
    med = raw_cal.filter(like="__median")
    table = pd.DataFrame(
        {"mode": mode, "label": lab.to_numpy(), "n_steps": raw_cal.n_steps}, index=F.index
    )
    summary = table.groupby("mode").agg(
        n=("label", "size"),
        rejected=("label", lambda s: int((s == "rejected").sum())),
        median_steps=("n_steps", "median"),
    )
    summary["reject_rate"] = (summary.rejected / summary.n).round(3)
    levels = med.groupby(mode).median().T
    levels["diff_in_pooled_sd"] = (levels["S"] - levels["L"]) / med.std()
    levels.sort_values("diff_in_pooled_sd", key=abs, ascending=False).to_csv(
        tab_dir / "operating_mode_levels.csv"
    )
    table.to_csv(tab_dir / "operating_modes.csv")
    num = F.index.str[2:].astype(int)
    out["operating_modes"] = {
        "summary": summary.reset_index().to_dict("records"),
        "test_counts": pd.Series(mode_test).value_counts().to_dict(),
        "largest_level_differences_sd": levels.diff_in_pooled_sd.abs()
        .sort_values(ascending=False)
        .head(4)
        .round(2)
        .to_dict(),
        "mode_S_share_by_batch_number_quartile": pd.Series(mode == "S")
        .groupby(pd.qcut(num, 4, labels=["Q1 (earliest ids)", "Q2", "Q3", "Q4"]), observed=True)
        .mean()
        .round(2)
        .to_dict(),
    }


def mspc(F, lab, y, out, fig_dir, tab_dir):
    """Out-of-fold monitor statistics for every calibration batch, plus contributions."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    parts, contribs = [], []
    for tr, te in cv.split(F, y):
        mon = PCAMonitor().fit(F.iloc[tr][y[tr] == 0])
        Xte = F.iloc[te]
        parts.append(mon.score(Xte))
        spe_c = mon.spe_contributions(Xte)
        t2_c = mon.t2_contributions(Xte).abs()
        contribs.append(
            0.5 * spe_c.div(spe_c.sum(axis=1), axis=0) + 0.5 * t2_c.div(t2_c.sum(axis=1), axis=0)
        )
    s = pd.concat(parts).loc[F.index].join(lab)
    c = pd.concat(contribs).loc[F.index]
    s.to_csv(tab_dir / "mspc_out_of_fold_scores.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, color in LABEL_COLORS.items():
        m = s.label == name
        scatter(ax, s.T2_ratio[m], s.SPE_ratio[m], color, name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1, color=TEXT_SECONDARY, lw=1)
    ax.axhline(1, color=TEXT_SECONDARY, lw=1)
    ax.text(1.05, ax.get_ylim()[0] * 1.3, "T² limit (95%)", color=TEXT_SECONDARY, fontsize=8)
    ax.text(ax.get_xlim()[0] * 1.3, 1.08, "SPE limit (95%)", color=TEXT_SECONDARY, fontsize=8)
    ax.set_xlabel("T² / limit  (unusual size within the normal pattern)")
    ax.set_ylabel("SPE / limit  (broken sensor relationships)")
    ax.set_title("Accepted-only monitor, out-of-fold: alarms outside the lower-left box")
    ax.legend(loc="upper left")
    save(fig, fig_dir / "08_mspc_chart.png")

    by_sensor = c.T.groupby(c.columns.map(sensor_of)).sum().T
    share = pd.DataFrame(
        {name: by_sensor[(lab == name).to_numpy()].mean() for name in LABEL_COLORS}
    ).sort_values("rejected")
    share.to_csv(tab_dir / "contribution_share_by_sensor.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ypos = np.arange(len(share))
    h = 0.38
    ax.barh(ypos + h / 2, share.rejected, height=h - 0.03, color=ORANGE, label="rejected")
    ax.barh(ypos - h / 2, share.accepted, height=h - 0.03, color=BLUE, label="accepted")
    ax.set_yticks(ypos, share.index)
    ax.set_xlabel("mean share of the monitor statistic (T² and SPE)")
    ax.set_title("Which sensors drive the monitor statistic")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, fig_dir / "09_contributions.png")

    out["mspc"] = {
        "alarm_rate_accepted_oof": round(float(s.alarm[s.label == "accepted"].mean()), 3),
        "alarm_rate_rejected_oof": round(float(s.alarm[s.label == "rejected"].mean()), 3),
        "rejected_alarm_by_T2_only": int(
            ((s.T2_ratio > 1) & (s.SPE_ratio <= 1) & (s.label == "rejected")).sum()
        ),
        "rejected_alarm_by_SPE_only": int(
            ((s.SPE_ratio > 1) & (s.T2_ratio <= 1) & (s.label == "rejected")).sum()
        ),
        "rejected_alarm_by_both": int(
            ((s.SPE_ratio > 1) & (s.T2_ratio > 1) & (s.label == "rejected")).sum()
        ),
        "top_sensors_rejected": share.rejected.sort_values(ascending=False)
        .head(4)
        .round(3)
        .to_dict(),
        "most_normal_looking_rejected": s[s.label == "rejected"]
        .score.nsmallest(5)
        .round(2)
        .to_dict(),
    }


def model_candidates() -> dict:
    lr = dict(C=0.1, max_iter=5000)
    rf = dict(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=SEED
    )
    hgb = dict(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)
    return {
        "majority (no model)": supervised(DummyClassifier(strategy="prior")),
        "single best feature, logistic": supervised(
            make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif, k=1),
                LogisticRegression(class_weight="balanced", **lr),
            )
        ),
        "logistic | no rebalancing": supervised(
            make_pipeline(StandardScaler(), LogisticRegression(**lr))
        ),
        "logistic | class weights": supervised(
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", **lr))
        ),
        "logistic | SMOTE": supervised(
            ImbPipeline(
                [
                    ("scale", StandardScaler()),
                    ("smote", SMOTE(random_state=SEED)),
                    ("clf", LogisticRegression(**lr)),
                ]
            )
        ),
        "logistic | undersampling": supervised(
            ImbPipeline(
                [
                    ("scale", StandardScaler()),
                    ("under", RandomUnderSampler(random_state=SEED)),
                    ("clf", LogisticRegression(**lr)),
                ]
            )
        ),
        "random forest | no rebalancing": supervised(RandomForestClassifier(**rf)),
        "random forest | class weights": supervised(
            RandomForestClassifier(class_weight="balanced_subsample", **rf)
        ),
        "random forest | SMOTE": supervised(
            ImbPipeline(
                [
                    ("scale", StandardScaler()),
                    ("smote", SMOTE(random_state=SEED)),
                    ("clf", RandomForestClassifier(**rf)),
                ]
            )
        ),
        "gradient boosting | no rebalancing": supervised(HistGradientBoostingClassifier(**hgb)),
        "gradient boosting | class weights": supervised(
            HistGradientBoostingClassifier(class_weight="balanced", **hgb)
        ),
        "PCA monitor | accepted only": one_class(),
        "PCA monitor per mode | accepted only": per_mode_monitor(),
        "kNN distance | accepted only": one_class(KNNMonitor, k=10),
    }


def compare_models(F, y, repeats, out, fig_dir, tab_dir):
    results = {
        name: repeated_cv(fp, F, y, name, n_repeats=repeats, seed=SEED)
        for name, fp in model_candidates().items()
    }
    table = pd.DataFrame([r.summary() for r in results.values()])
    table.to_csv(tab_dir / "model_comparison.csv")

    order = table.roc_auc_mean.sort_values().index
    t = table.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    ypos = np.arange(len(t))
    oc = np.array(["accepted only" in n for n in t.index])
    for ax, metric, title in [
        (axes[0], "roc_auc", "ROC-AUC (ranking quality)"),
        (axes[1], "balanced_accuracy", "Balanced accuracy (at the model's own threshold)"),
    ]:
        mean, sd = t[f"{metric}_mean"], t[f"{metric}_sd"]
        for mask, color, label in [
            (~oc, BLUE, "supervised (uses rejected labels)"),
            (oc, ORANGE, "one-class (accepted batches only)"),
        ]:
            ax.hlines(ypos[mask], (mean - sd)[mask], (mean + sd)[mask], color=color, lw=1.5)
            ax.scatter(mean[mask], ypos[mask], color=color, s=40, zorder=3, label=label)
        ax.axvline(0.5, color=MUTED, lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.4, 1.0)
    axes[0].set_yticks(ypos, t.index, fontsize=8)
    axes[1].legend(loc="lower right")
    fig.suptitle(
        f"Model and imbalance-strategy comparison, {repeats}x repeated 5-fold CV (mean ± sd)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, fig_dir / "10_model_comparison.png")

    keep = ["roc_auc", "pr_auc", "balanced_accuracy", "mcc", "recall_rejected", "specificity"]
    out["models"] = {
        name: {
            m: [
                round(float(table.loc[name, f"{m}_mean"]), 3),
                round(float(table.loc[name, f"{m}_sd"]), 3),
            ]
            for m in keep
        }
        for name in table.index
    }
    return table


def smote_geometry(F, y, out, fig_dir):
    """Where do SMOTE's synthetic 'rejected' batches land relative to real data?"""
    Z = StandardScaler().fit_transform(F)
    Xs, ys = SMOTE(random_state=SEED).fit_resample(Z, y)
    synth = Xs[len(Z) :]
    nn = NearestNeighbors(n_neighbors=5).fit(Z)
    _, idx = nn.kneighbors(synth)
    frac_acc_nn1 = float((y[idx[:, 0]] == 0).mean())
    frac_majority_acc = float(((y[idx] == 0).mean(axis=1) > 0.5).mean())
    _, idx_real = nn.kneighbors(Z[y == 1], n_neighbors=6)
    real_majority_acc = float(((y[idx_real[:, 1:]] == 0).mean(axis=1) > 0.5).mean())

    pca = PCA(2).fit(Z)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    P = pca.transform(Z)
    S = pca.transform(synth)
    scatter(ax, P[y == 0, 0], P[y == 0, 1], BLUE, "accepted (real)")
    scatter(ax, P[y == 1, 0], P[y == 1, 1], ORANGE, "rejected (real)")
    scatter(ax, S[:, 0], S[:, 1], AQUA, "rejected (SMOTE synthetic)", marker="x", size=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"SMOTE adds {len(synth)} synthetic rejects by interpolating between real ones")
    clip_view(ax, P, F.index)
    ax.legend(loc="upper right")
    save(fig, fig_dir / "11_smote_geometry.png")
    out["smote"] = {
        "n_synthetic": int(len(synth)),
        "synthetic_nearest_real_is_accepted": round(frac_acc_nn1, 3),
        "synthetic_with_accepted_majority_of_5nn": round(frac_majority_acc, 3),
        "real_rejected_with_accepted_majority_of_5nn": round(real_majority_acc, 3),
    }


def significance(F, y, best_name, n_perm, out, fig_dir):
    cands = model_candidates()
    colors = {best_name: BLUE, **ONE_CLASS_SERIES}
    tests = {name: cands[name] for name in colors}
    fig, axes = plt.subplots(1, len(tests), figsize=(13, 3.2), sharey=True)
    out["permutation_test"] = {}
    for ax, (name, fp) in zip(np.atleast_1d(axes), tests.items(), strict=False):
        obs, null, p = permutation_test(fp, F, y, n_permutations=n_perm, seed=SEED)
        ax.hist(null, bins=20, color=MUTED, edgecolor=SURFACE)
        ax.axvline(obs, color=colors[name], lw=2)
        ax.text(obs, ax.get_ylim()[1] * 0.9, f" observed {obs:.2f}\n p = {p:.3f}", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("ROC-AUC with shuffled labels")
        out["permutation_test"][name] = {
            "observed_auc": round(float(obs), 3),
            "null_mean": round(float(null.mean()), 3),
            "null_max": round(float(null.max()), 3),
            "p_value": round(float(p), 4),
            "n_permutations": n_perm,
        }
    fig.suptitle(
        "Permutation test: is the signal real?", x=0.01, ha="left", fontweight="bold", fontsize=11
    )
    fig.tight_layout()
    save(fig, fig_dir / "12_permutation_test.png")


def learning(F, y, best_name, out, fig_dir):
    cands = model_candidates()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    out["learning_curve"] = {}
    for name, color in {best_name: BLUE, **ONE_CLASS_SERIES}.items():
        lc = (
            learning_curve(cands[name], F, y, seed=SEED)
            .groupby("n_train")
            .roc_auc.agg(["mean", "std"])
        )
        ax.fill_between(
            lc.index, lc["mean"] - lc["std"], lc["mean"] + lc["std"], color=color, alpha=0.15, lw=0
        )
        ax.plot(lc.index, lc["mean"], color=color, marker="o", ms=4, label=name)
        out["learning_curve"][name] = {int(k): round(float(v), 3) for k, v in lc["mean"].items()}
    ax.set_xlabel("training batches per fold")
    ax.set_ylabel("ROC-AUC (test folds)")
    ax.set_title("Learning curve: would more batches help?")
    ax.legend(loc="lower right")
    save(fig, fig_dir / "13_learning_curve.png")


def shift_and_test(F, Ft, y, out, tab_dir):
    X = pd.concat([F, Ft])
    d = np.r_[np.zeros(len(F)), np.ones(len(Ft))]
    p = cross_val_predict(
        RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
        X,
        d,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba",
    )[:, 1]
    ks = pd.Series({c: stats.ks_2samp(F[c], Ft[c]).pvalue for c in F.columns}).sort_values()
    mon = PCAMonitor().fit(F[y == 0])
    st = mon.score(Ft)
    st.to_csv(tab_dir / "test_monitor_scores.csv")
    out["calibration_vs_test"] = {
        "adversarial_auc": round(float(roc_auc_score(d, p)), 3),
        "features_ks_p_below_0.01": int((ks < 0.01).sum()),
        "most_shifted_features": ks.head(5).map(lambda v: float(f"{v:.2g}")).to_dict(),
        "test_alarm_rate_monitor_95pct": round(float(st.alarm.mean()), 3),
    }


# --------------------------------------------------------------------------- report
def write_report(out: dict, path: Path) -> None:
    d, m = out["data"], out["models"]
    best = out["best_supervised"]
    lines = [
        "# Process A — data understanding and model-selection evidence",
        "",
        "_Generated by `scripts/eda_process_a.py`. Every number below is computed from the data._",
        "",
        "## 1. Data",
        f"- {d['n_batches_calibration']} calibration batches ({d['n_accepted']} accepted, "
        f"{d['n_rejected']} rejected, reject rate {d['reject_rate']:.0%}); "
        f"{d['n_batches_test']} unlabelled test batches; {d['n_sensors']} numeric sensors.",
        f"- {d['n_rows_calibration']:,} calibration rows; one row = one time step; no timestamp.",
        f"- Missing values: {d['missing_values']}. Longest run of identical values (stuck sensor check): "
        f"{d['stuck_sensor_max_run']}.",
        f"- Batch length min/max/median: {d['length_min_max_median']}. Short batches (quarter-split "
        f"fallback): {', '.join(d['short_batches_calibration']) or 'none'}.",
        "- **There is no text in this dataset**, so text preprocessing does not apply. The equivalent "
        "for sensor data is signal preprocessing: segmentation, outlier handling and transforms (below).",
        "",
        "![](figures/01_batch_length.png)",
        "",
        "| length bin | n | rejected | reject rate |",
        "|---|---|---|---|",
        *[
            f"| {r['n_steps']} | {r['n']} | {r['rejected']} | {r['reject_rate']:.0%} |"
            for r in d["length_vs_label"]
        ],
        "",
        "## 2. Preprocessing",
        f"- **Transients**: slow sensors start at mean |z| = {out['transients']['abs_z_at_start']} and "
        f"end at {out['transients']['abs_z_at_end']} vs a steady reference of "
        f"{out['transients']['steady_reference_abs_z']}. The first {out['transients']['trim_start']} "
        f"and last {out['transients']['trim_end']} steps are kept as separate start/end features.",
        "- **Gross outliers** (dropouts) are detected against a rolling median and excluded from "
        "statistics; their count is kept as a feature.",
        *[
            f"  - {b}: {v['rows']} rows ({v['label']})"
            for b, v in d["gross_outlier_batches_calibration"].items()
        ],
        f"  - test batches with gross-outlier rows: {d['gross_outlier_batches_test']}",
        "- **Features**: one row per batch, length-independent; log for spreads, Fisher z for "
        "autocorrelations.",
        "",
        "![](figures/02_transients.png)",
        "![](figures/03_sensor_profiles.png)",
        "![](figures/04_sensor_correlation.png)",
        "",
        "## 3. Single features",
        f"- {out['univariate']['n_significant_fdr_0.05']} of {out['univariate']['n_features']} "
        "features differ between classes after Benjamini-Hochberg FDR correction (q < 0.05).",
        f"- {out['univariate']['feature_pairs_abs_r_gt_0.9']} of {out['univariate']['n_feature_pairs']} "
        "feature pairs are near-duplicates (|r| > 0.9).",
        "",
        "| feature | AUC | 95% CI | q |",
        "|---|---|---|---|",
        *[
            f"| {b['feature']} | {b['auc']} | {b['ci'][0]}–{b['ci'][1]} | {b['q']} |"
            for b in out["univariate"]["best"]
        ],
        "",
        "![](figures/05_univariate_auc.png)",
        "",
        "## 4. PCA",
        f"- {out['pca']['components_for_80pct']} components for 80% of variance, "
        f"{out['pca']['components_for_90pct']} for 90%; PC1–3 explain "
        f"{', '.join(f'{v:.0%}' for v in out['pca']['explained_pc1_pc2_pc3'])}.",
        f"- Single-PC AUC (PC1..PC10): {out['pca']['auc_of_pc1_to_pc10']}. The class difference "
        "is not on the main variance directions.",
        "",
        "![](figures/06_pca_scree.png)",
        "![](figures/07_pca_scores.png)",
        "",
        "### Operating modes",
        "Unsupervised 2-cluster split on PC1 (labels not used); named by batch length.",
        "",
        "| mode | batches | rejected | reject rate | median steps |",
        "|---|---|---|---|---|",
        *[
            f"| {r['mode']} | {r['n']} | {r['rejected']} | {r['reject_rate']:.0%} | {r['median_steps']:.0f} |"
            for r in out["operating_modes"]["summary"]
        ],
        "",
        f"- Test batches per mode: {out['operating_modes']['test_counts']}.",
        f"- Largest sensor-level differences between modes (pooled sd): "
        f"{out['operating_modes']['largest_level_differences_sd']}.",
        f"- Share of mode S by batch-number quartile: "
        f"{out['operating_modes']['mode_S_share_by_batch_number_quartile']} "
        "(both modes occur throughout the campaign).",
        "",
        "## 5. Accepted-only PCA monitor (T² and SPE)",
        f"- Out-of-fold alarm rate: accepted {out['mspc']['alarm_rate_accepted_oof']:.0%} "
        f"(designed ≈5%), rejected {out['mspc']['alarm_rate_rejected_oof']:.0%}.",
        f"- Rejected alarms by T² only / SPE only / both: {out['mspc']['rejected_alarm_by_T2_only']} / "
        f"{out['mspc']['rejected_alarm_by_SPE_only']} / {out['mspc']['rejected_alarm_by_both']}.",
        f"- Top contributing sensors for rejected batches: {out['mspc']['top_sensors_rejected']}.",
        f"- Most normal-looking rejected batches (score < 1 means inside limits): "
        f"{out['mspc']['most_normal_looking_rejected']}.",
        "",
        "![](figures/08_mspc_chart.png)",
        "![](figures/09_contributions.png)",
        "",
        "## 6. Class imbalance and model comparison",
        f"Repeated stratified 5-fold CV by batch ({out['repeats']} repeats). Mean ± sd.",
        "",
        "| model | ROC-AUC | PR-AUC | balanced acc. | MCC | recall rej. | specificity |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {n.replace('|', chr(92) + '|')} | "
            + " | ".join(f"{v[0]:.3f} ± {v[1]:.3f}" for v in r.values())
            + " |"
            for n, r in sorted(m.items(), key=lambda kv: -kv[1]["roc_auc"][0])
        ],
        "",
        f"PR-AUC of a random model equals the reject rate ({d['reject_rate']:.2f}).",
        "",
        "![](figures/10_model_comparison.png)",
        "",
        "### SMOTE",
        f"- SMOTE created {out['smote']['n_synthetic']} synthetic rejects by interpolating "
        "between real ones.",
        f"- The classes overlap: {out['smote']['real_rejected_with_accepted_majority_of_5nn']:.0%} "
        "of *real* rejects have mostly accepted batches as their 5 nearest neighbours "
        f"({out['smote']['synthetic_with_accepted_majority_of_5nn']:.0%} of synthetic ones). "
        "Interpolating inside that overlap adds no new information.",
        *[
            f"- {fam}: ROC-AUC with SMOTE {m[f'{fam} | SMOTE']['roc_auc'][0]:.3f} vs class weights "
            f"{m[f'{fam} | class weights']['roc_auc'][0]:.3f}; recall {m[f'{fam} | SMOTE']['recall_rejected'][0]:.2f}"
            f" vs {m[f'{fam} | class weights']['recall_rejected'][0]:.2f}, specificity "
            f"{m[f'{fam} | SMOTE']['specificity'][0]:.2f} vs {m[f'{fam} | class weights']['specificity'][0]:.2f}."
            for fam in ("logistic", "random forest")
        ],
        "- Relative to class weights, SMOTE does not improve ranking (ROC-AUC). Where it changes "
        "anything (random forest) it trades specificity for recall, which moving the decision "
        "threshold achieves without synthetic data.",
        "",
        "![](figures/11_smote_geometry.png)",
        "",
        "## 7. Is the signal real, and would more data help?",
        *[
            f"- {n}: observed AUC {v['observed_auc']}, null mean {v['null_mean']}, null max "
            f"{v['null_max']}, p = {v['p_value']} ({v['n_permutations']} permutations)."
            for n, v in out["permutation_test"].items()
        ],
        *[f"- Learning curve {n}: {v}" for n, v in out["learning_curve"].items()],
        "",
        "![](figures/12_permutation_test.png)",
        "![](figures/13_learning_curve.png)",
        "",
        "## 8. Calibration vs test",
        f"- Adversarial validation AUC {out['calibration_vs_test']['adversarial_auc']} "
        "(0.5 = indistinguishable).",
        f"- Features with KS p < 0.01: {out['calibration_vs_test']['features_ks_p_below_0.01']}.",
        f"- Accepted-only monitor (95% limits) alarms on "
        f"{out['calibration_vs_test']['test_alarm_rate_monitor_95pct']:.0%} of test batches.",
        "",
        "## 9. Best supervised vs one-class (ROC-AUC)",
        f"- Best supervised, {best}: {m[best]['roc_auc'][0]:.3f} ± {m[best]['roc_auc'][1]:.3f}.",
        *[f"- {n}: {m[n]['roc_auc'][0]:.3f} ± {m[n]['roc_auc'][1]:.3f}." for n in ONE_CLASS_SERIES],
        f"- Picking the best of {len(m) - 1} candidates by CV is mildly optimistic; confirm with "
        "nested CV before submitting.",
        "",
    ]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "process_A" / "eda")
    args = parser.parse_args(argv)

    apply_style()
    fig_dir, tab_dir = args.out / "figures", args.out / "tables"
    for p in (fig_dir, tab_dir, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)
    params = SEGMENT_PARAMS[PROCESS]
    out: dict = {"repeats": args.repeats, "segment_params": params.__dict__}

    print("[1/9] loading + data quality")
    ts, lab = load_calibration(PROCESS)
    te = load_test(PROCESS)
    y = to_binary(lab).to_numpy()
    scale = data_overview(ts, te, lab, params, out, fig_dir, tab_dir)

    print("[2/9] transients, profiles, correlation")
    transients(ts, params, out, fig_dir)
    sensor_profiles(ts, lab, params, fig_dir)
    sensor_correlation(ts, params, out, fig_dir)

    print("[3/9] batch features")
    raw_cal = batch_features(ts, params, exclude=gross_outlier_mask(ts, params, scale)).loc[
        lab.index
    ]
    raw_test = batch_features(te, params, exclude=gross_outlier_mask(te, params, scale))
    raw_cal.join(lab).to_parquet(PROCESSED_DIR / "process_A_features_calibration.parquet")
    raw_test.to_parquet(PROCESSED_DIR / "process_A_features_test.parquet")
    F, Ft = to_model_space(raw_cal), to_model_space(raw_test)

    print("[4/9] univariate")
    univariate(F, y, out, fig_dir, tab_dir)
    print("[5/9] PCA")
    pca_structure(F, lab, out, fig_dir, tab_dir)
    operating_modes(F, Ft, raw_cal, lab, out, tab_dir)
    print("[6/9] accepted-only monitor")
    mspc(F, lab, y, out, fig_dir, tab_dir)
    print("[7/9] model + imbalance comparison")
    table = compare_models(F, y, args.repeats, out, fig_dir, tab_dir)
    smote_geometry(F, y, out, fig_dir)
    supervised_rows = table[~table.index.str.contains("accepted only")].drop(
        index="majority (no model)"
    )
    out["best_supervised"] = best = supervised_rows.roc_auc_mean.idxmax()
    print("[8/9] permutation test + learning curve")
    significance(F, y, best, args.permutations, out, fig_dir)
    learning(F, y, best, out, fig_dir)
    print("[9/9] calibration vs test")
    shift_and_test(F, Ft, y, out, tab_dir)

    (args.out / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    write_report(out, args.out / "REPORT.md")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
