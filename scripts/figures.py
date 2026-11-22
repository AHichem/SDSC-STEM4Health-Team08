"""Figures for the README and the slides, from the data and the recorded results.

    python scripts/figures.py   # -> docs/figures/*.png (run make report and make tune-semi first)

data_b.png        process B: aeration relative to each batch's own level, all calibration batches
data_a.png        process A: which features separate rejects, and how much the classes overlap
families.png      model families on the team's shared CV protocol (docs/model_results.md)
improvements.png  change in F1 of each idea tried on top of the chosen ensemble, same splits
confidence.png    is P(rejected) honest (out of fold), and how the test batches are routed
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from brpmm.config import BATCH_COL, LABEL_REJECTED, OUTPUTS_DIR  # noqa: E402
from brpmm.data import TIME_COL, load_calibration  # noqa: E402
from brpmm.evaluate import load_features  # noqa: E402
from brpmm.methods import _median_std_columns  # noqa: E402
from brpmm.rules import SetpointDeviationRule  # noqa: E402

FIG = OUTPUTS_DIR.parent / "docs" / "figures"
TUNING = OUTPUTS_DIR / "process_A" / "tuning"
ACC, REJ = "#2a78d6", "#eb6834"  # validated categorical slots 1-2 (blue / orange)
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a8a69e"
plt.rcParams.update(
    {
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _title(fig, text: str, sub: str) -> None:
    """Title and subtitle flush with the left edge of everything drawn (tick labels included)."""
    fig.canvas.draw()
    x0 = fig.get_tightbbox(fig.canvas.get_renderer()).x0 / fig.get_figwidth()
    fig.text(x0, 1.06, text, fontsize=14, fontweight="bold", ha="left", va="bottom")
    fig.text(x0, 1.0, sub, fontsize=11, color=INK2, ha="left", va="bottom")


def data_b() -> None:
    ts, lab = load_calibration("B")
    rule = SetpointDeviationRule()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhspan(-100 * rule.rel_tolerance, 100 * rule.rel_tolerance, color="#f1f0ec", zorder=0)
    for rejected in (True, False):  # accepted drawn last, so they stay visible on the zero line
        for b in lab.index[(lab == LABEL_REJECTED) == rejected]:
            d = ts[ts[BATCH_COL] == b].sort_values(TIME_COL)
            x = d[rule.variable].to_numpy()
            rel = 100 * (x / np.median(x[: rule.baseline_steps]) - 1)
            ax.plot(
                d[TIME_COL],
                rel,
                color=REJ if rejected else ACC,
                lw=2 if rejected else 1,
                alpha=0.9 if rejected else 0.6,
                label=None,
            )
    ax.plot([], [], color=ACC, lw=1.5, label="accepted (35)")
    ax.plot([], [], color=REJ, lw=2, label="rejected (5)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("time step")
    ax.set_ylabel("aeration vs own early level (%)")
    ax.set_ylim(-40, 15)
    ax.text(5000, 11.5, "rule limit ±10 %", color=INK2, ha="right", fontsize=10)
    ax.text(
        1300,
        -30,
        "5 rejected: −30 % for 751 steps,\nfrom step 500",
        color=REJ,
        fontsize=11,
        fontweight="bold",
        va="center",
    )
    ax.text(
        2600,
        3.5,
        "35 accepted: within 0.35 % of their own level",
        color=ACC,
        fontsize=11,
        fontweight="bold",
    )
    _title(
        fig,
        "Process B: every reject is the same aeration fault",
        "Each line is one calibration batch, relative to its own first 500 steps",
    )
    fig.savefig(FIG / "data_b.png")
    plt.close(fig)


def data_a() -> None:
    X, y = load_features("A")
    cols = _median_std_columns(X)
    Z = pd.DataFrame(StandardScaler().fit_transform(X[cols]), index=X.index, columns=cols)
    rej = (y == LABEL_REJECTED).to_numpy()
    d = (Z[rej].mean() - Z[~rej].mean()) / Z.std()
    top = d.abs().sort_values(ascending=False).head(8)[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    names = [c.replace("__", " ") for c in top.index]
    a1.barh(names, d[top.index], color=ACC, height=0.6)
    a1.axvline(0, color=MUTED, lw=1)
    a1.set_xlabel("rejected minus accepted (standard deviations)")
    a1.grid(axis="y", visible=False)
    a1.set_title("Strongest signals: variability, not level", loc="left", fontsize=11, color=INK2)
    f1, f2 = d.abs().sort_values(ascending=False).index[:2]
    a2.scatter(
        Z.loc[~rej, f1],
        Z.loc[~rej, f2],
        s=26,
        color=ACC,
        alpha=0.75,
        label="accepted (121)",
        edgecolor="white",
        linewidth=0.6,
    )
    a2.scatter(
        Z.loc[rej, f1],
        Z.loc[rej, f2],
        s=34,
        color=REJ,
        marker="^",
        alpha=0.9,
        label="rejected (40)",
        edgecolor="white",
        linewidth=0.6,
    )
    a2.set_xlabel(f"{f1.replace('__', ' ')} (standardised)")
    a2.set_ylabel(f"{f2.replace('__', ' ')} (standardised)")
    a2.legend(frameon=False, loc="upper right")
    a2.set_title(
        "The classes overlap: no single cut separates them", loc="left", fontsize=11, color=INK2
    )
    _title(
        fig,
        "Process A: weak, batch-wide signal in 11 sensors",
        "161 calibration batches, one point per batch (median and spread of each sensor)",
    )
    fig.savefig(FIG / "data_a.png")
    plt.close(fig)


FAMILIES = [  # docs/model_results.md, team protocol (5 repeats x 5-fold CV), F1 accepted
    ("Accept every batch (baseline)", 0.858, "A01"),
    ("Logistic regression, 243 features", 0.846, "A04"),
    ("PCA monitor (accepted only)", 0.855, "A03"),
    ("Gradient boosting", 0.858, "A28"),
    ("Random forest", 0.860, "A05"),
    ("1D-CNN on trajectories", 0.890, "A22"),
    ("One-class SVM (accepted only)", 0.899, "A18"),
    ("MLP", 0.902, "A21"),
    ("kNN", 0.903, "A16"),
    ("RBF SVM", 0.931, "A19"),
    ("Ensemble: SVM + kNN + one-class SVM", 0.943, "A24"),
]


def families() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = [n for n, _, _ in FAMILIES]
    vals = np.array([v for _, v, _ in FAMILIES])
    colors = [ACC if n.startswith("Ensemble") else MUTED for n in names]
    ax.hlines(names, 0.83, vals, color=GRID, lw=2)
    ax.scatter(vals, names, color=colors, s=70, zorder=3)
    for v, n in zip(vals, names, strict=True):
        ax.text(
            v + 0.002,
            n,
            f"{v:.3f}",
            va="center",
            fontsize=10,
            color=INK if n.startswith("Ensemble") else INK2,
        )
    ax.set_xlim(0.83, 0.96)
    ax.set_xlabel("F1 (accepted class, as scored), cross-validated")
    ax.grid(axis="y", visible=False)
    _title(
        fig,
        "Process A: 30+ models compared on one protocol",
        "Simple robust features + a small ensemble beat deep learning and trees on 161 batches",
    )
    fig.savefig(FIG / "families.png")
    plt.close(fig)


def _f1(part: str, key: str) -> float:
    return json.loads((TUNING / f"{part}.json").read_text())[key]["f1_accepted"]


NOTES = {
    "Subagging, out-of-bag cut-off": "  but −0.002 on new splits: not adopted",
    "Self-training on unlabelled test batches": "  (+0.007 with a test-sized pool)\nhidden test: 0.931 → 0.939, adopted",
}


def improvements() -> None:
    sub, team = "maria ensemble (submitted)", "team (z)"
    rows = [  # (label, value, its own baseline)
        (
            "Nested hyperparameter search (81 settings)",
            _f1("hpo", "tuned_nested"),
            _f1("hpo", "default"),
        ),
        (
            "+ TabPFN 3.5 (foundation model) member",
            _f1("tabpfn", "maria ensemble + TabPFN"),
            _f1("tabpfn", sub),
        ),
        (
            "+ MiniRocket time-series member",
            _f1("combine", "team + minirocket (z)"),
            _f1("combine", team),
        ),
        (
            "+ sample-level trees member",
            _f1("combine", "team + sample_et (z)"),
            _f1("combine", team),
        ),
        (
            "SMOTE synthetic rejects",
            _f1("augment", "team, SVM on SMOTE-enlarged rejects (z)"),
            _f1("augment", team),
        ),
        (
            "Sensor-dropout rows removed",
            _f1("clean", "dropout rows removed"),
            _f1("clean", "raw batch (submitted)"),
        ),
        (
            "Subagging, out-of-bag cut-off",
            _f1("rules", "subagging 25 x 80 %, out-of-bag cut-off"),
            _f1("rules", "median of 5 x 5-fold cut-offs (final-fit rule)"),
        ),
        (
            "Self-training on unlabelled test batches",
            _f1("semi", "self-training, confident accepted only, margin 1.0"),
            _f1("semi", "submitted (labelled batches only)"),
        ),
    ]
    imp = json.loads((OUTPUTS_DIR / "process_A" / "improve" / "variants.json").read_text())
    base = imp["base (submitted)"]["f1_accepted"]
    rows.insert(
        4,
        (
            "+ physics features (flow/pressure, ΔT, Δp)",
            imp["base + mechanistic"]["f1_accepted"],
            base,
        ),
    )
    rows.insert(5, ("+ volatility features", imp["base + volatility"]["f1_accepted"], base))
    labels = [r[0] for r in rows][::-1]
    delta = np.array([r[1] - r[2] for r in rows])[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axvspan(-0.01, 0.01, color="#f1f0ec", zorder=0)
    adopted = ["Self-training" in s for s in labels]
    ax.hlines(labels, 0, delta, color=[ACC if a else MUTED for a in adopted], lw=3)
    ax.scatter(delta, labels, color=[ACC if a else MUTED for a in adopted], s=55, zorder=3)
    for v, s, a in zip(delta, labels, adopted, strict=True):
        ax.text(
            v + (0.0012 if v >= 0 else -0.0012),
            s,
            f"{round(v, 3) + 0:+.3f}" + NOTES.get(s, ""),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            color=ACC if a else INK2,
            fontweight="bold" if a else "normal",
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-0.03, 0.045)
    ax.text(
        0.0098,
        len(labels) - 0.4,
        "±0.01: the gain we required",
        fontsize=9.5,
        color=INK2,
        ha="right",
    )
    ax.set_xlabel("change in F1 vs the ensemble it was added to (same splits, 4 repeats)")
    ax.grid(axis="y", visible=False)
    _title(
        fig,
        "What we tried on top of the ensemble",
        "Nothing beat it by more than noise, except self-training, which then held on the hidden test",
    )
    fig.savefig(FIG / "improvements.png")
    plt.close(fig)


def confidence() -> None:
    stem = sorted(OUTPUTS_DIR.glob("*_batch_report.csv"))[-1]
    rep = pd.read_csv(stem, index_col=BATCH_COL)
    rel = pd.read_csv(stem.with_name(stem.stem + "_reliability.csv"))
    a = rep[rep["project"] == "A"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a1.scatter(rel["mean_p"], rel["share_rejected"], s=rel["batches"] / 2 + 20, color=ACC, zorder=3)
    for _, r in rel.iterrows():
        a1.text(
            r["mean_p"] + (0.03 if r["mean_p"] < 0.6 else -0.03),
            r["share_rejected"]
            + (-0.05 if r["mean_p"] < 0.6 else 0.03 if r["mean_p"] < 0.9 else -0.07),
            f"n={int(r['batches'])}",
            ha="left" if r["mean_p"] < 0.6 else "right",
            fontsize=9,
            color=INK2,
        )
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_xlabel("predicted P(rejected)")
    a1.set_ylabel("share actually rejected")
    a1.set_title(
        "Out of fold on calibration: close to the diagonal", loc="left", fontsize=11, color=INK2
    )
    bins = np.linspace(0, 1, 21)
    rej = a["label"] == LABEL_REJECTED
    a2.axvspan(0.2, 0.8, color="#f1f0ec", zorder=0)
    a2.hist(
        [a.loc[~rej, "p_rejected"], a.loc[rej, "p_rejected"]],
        bins=bins,
        stacked=True,
        color=[ACC, REJ],
        label=["labelled accepted", "labelled rejected"],
        rwidth=0.88,
    )
    n_rev = int((a["review"] == "yes").sum())
    a2.text(
        0.5,
        a2.get_ylim()[1] * 0.55,
        f"review band\n{n_rev} batches go to a person",
        ha="center",
        va="top",
        fontsize=10.5,
        color=INK,
    )
    a2.set_xlabel("P(rejected), 161 test batches")
    a2.set_ylabel("batches")
    a2.legend(frameon=False, loc="upper right")
    a2.grid(axis="x", visible=False)
    a2.set_title(
        f"{len(a) - n_rev} of {len(a)} decided automatically", loc="left", fontsize=11, color=INK2
    )
    _title(
        fig,
        "A confidence you can check, and a route for every batch",
        "Also sent to review: model disagreement, likely sensor fault, or a reject with no close precedent",
    )
    fig.savefig(FIG / "confidence.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for make in (data_b, data_a, families, improvements, confidence):
        make()
        print(f"  {make.__name__}.png", flush=True)


if __name__ == "__main__":
    main()
