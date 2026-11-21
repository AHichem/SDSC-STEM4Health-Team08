"""Findings report: turns the cross-validation results into a Markdown page with figures.

Inputs: the dict per process returned by brpmm.evaluate.run_process.
Outputs: outputs/report_<timestamp>/report.md, figures (PNG) and results.json next to it.
Depends on: config, evaluate (summary_table), features (feature_sensor), methods (INTERPRETABILITY).

The report answers the questions the jury asks: how accurate (leaderboard F1), how robust
(spread over repeats, folds beating the baseline), and why a batch is rejected (which sensors
and which aspect of the batch), with example batches.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brpmm.config import (  # noqa: E402
    CV_FOLDS,
    CV_INNER_FOLDS,
    CV_REPEATS,
    LABEL_ACCEPTED,
    LABEL_REJECTED,
    NOISE_F1,
    OUTPUTS_DIR,
    SEED,
)
from brpmm.evaluate import summary_table  # noqa: E402
from brpmm.features import feature_sensor  # noqa: E402
from brpmm.methods import INTERPRETABILITY, METHODS  # noqa: E402

COLORS = {LABEL_ACCEPTED: "#2a9d8f", LABEL_REJECTED: "#d1495b"}


def build_report(results: dict[str, dict]) -> Path:
    out = OUTPUTS_DIR / f"report_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    tables = {p: summary_table(r) for p, r in results.items()}

    md = [
        f"# Cross-validation report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "**How to read this.** Every method is tested on exactly the same splits of the "
        "calibration "
        f"batches: stratified {CV_FOLDS}-fold cross-validation, repeated {CV_REPEATS} times "
        f"(seed {SEED}). "
        "A batch is only ever scored by a model that did not see it. "
        "The accept/reject cut-off is chosen "
        f"inside the training part ({CV_INNER_FOLDS}-fold inner CV) to maximise leaderboard F1.",
        "",
        "- **F1** = leaderboard F1 (*accepted* is the positive class), "
        "over all calibration batches, "
        f"mean ± std over the {CV_REPEATS} repeats.",
        "- **Four cases** (average per repeat): good kept · **bad let through** (dangerous) · "
        "bad caught · **good thrown away** (costly).",
        f"- Differences smaller than ~{NOISE_F1} F1 are probably noise with this few batches.",
        "",
    ]
    _plot_f1(results, out / "f1_by_method.png")
    md += ["![F1 per fold](f1_by_method.png)", ""]

    record = {
        "settings": {
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "inner_folds": CV_INNER_FOLDS,
            "seed": SEED,
            "methods": list(METHODS),
        },
        "processes": {},
    }
    for proc, res in results.items():
        tab = tables[proc]
        best = _best_method(tab)
        md += _process_section(proc, res, tab, best, out)
        record["processes"][proc] = {
            "n_batches": res["n_batches"],
            "n_rejected": res["n_rejected"],
            "n_features": res["n_features"],
            "best_method": best,
            "summary": tab.round(4).reset_index().to_dict(orient="records"),
        }

    md += [
        "## Caveats",
        "",
        "- Cross-validation on the calibration batches only; "
        "the test batches may contain situations "
        "(e.g. other fault types) that the calibration batches don't.",
        "- The cut-off is tuned for leaderboard F1, which ignores how many bad batches are caught. "
        "For a plant, a lower cut-off (more batches sent to review) may be preferable.",
        "- Random-forest reasons are a proxy (important features where the batch deviates most), "
        "not an exact per-batch decomposition.",
        "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return out / "report.md"


def _best_method(tab: pd.DataFrame) -> str:
    return tab.drop(index="accept_all", errors="ignore")["f1_mean"].idxmax()


def _process_section(proc: str, res: dict, tab: pd.DataFrame, best: str, out: Path) -> list[str]:
    n_bad = res["n_rejected"]
    md = [
        f"## Process {proc}",
        "",
        f"{res['n_batches']} calibration batches, {n_bad} rejected; "
        f"{res['n_features']} batch features "
        "(level, shape relative to the batch's own start, steps, trend, "
        "within-batch correlations).",
        "",
        "| Method | F1 (mean ± std) | Bad caught | Bad let through | Good thrown away "
        "| Folds beating accept-all "
        "| Interpretability |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in tab.iterrows():
        mark = " ⭐" if m == best else ""
        md.append(
            f"| {m}{mark} | {r.f1_mean:.3f} ± {r.f1_std:.3f} | {r.bad_caught:.1f} / {n_bad} | "
            f"{r.bad_let_through:.1f} | {r.good_thrown_away:.1f} | "
            f"{int(r.folds_beating_accept_all)} / {int(r.n_folds)} | "
            f"{INTERPRETABILITY.get(m, '')} |"
        )
    others = tab.drop(index=["accept_all", best], errors="ignore").sort_values(
        "f1_mean", ascending=False
    )
    md.append("")
    md.append(
        f"**Best by F1: `{best}`** ({tab.loc[best, 'f1_mean']:.3f}; "
        f"accept-all {tab.loc['accept_all', 'f1_mean']:.3f})."
    )
    if len(others):
        gap = tab.loc[best, "f1_mean"] - others["f1_mean"].iloc[0]
        verdict = "within noise: choose on interpretability" if gap < NOISE_F1 else "a real gap"
        md.append(
            f"Next: `{others.index[0]}` ({others['f1_mean'].iloc[0]:.3f}), "
            f"gap {gap:.3f} → {verdict}."
        )
    md.append("")

    oof = res["oof"]
    g = oof[(oof.method == best) & (oof.repeat == 0)]
    _plot_scores(g, best, proc, out / f"scores_{proc}.png")
    md += [f"![scores {proc}](scores_{proc}.png)", ""]

    caught = g[(g.label == LABEL_REJECTED) & (g.pred == LABEL_REJECTED)]
    sensors = Counter(feature_sensor(f) for rs in caught["reasons"] for f, _ in rs)
    aspects = Counter(
        f.split("__", 1)[1] if "__" in f else f
        for rs in caught["reasons"]
        for f, _ in rs
        if not f.startswith("corr__")
    )
    _plot_counter(
        sensors,
        f"Process {proc} — sensors behind correct rejections ({best})",
        out / f"sensors_{proc}.png",
    )
    md += [
        f"### Why batches are rejected (`{best}`, repeat 1)",
        "",
        "Sensors and aspects that appear most often among the top reasons of correctly "
        "rejected batches:",
        "",
        f"![sensors {proc}](sensors_{proc}.png)",
        "",
        "- **Sensors:** " + ", ".join(f"{s} ({c})" for s, c in sensors.most_common(8)),
        "- **Aspects:** " + ", ".join(f"{a} ({c})" for a, c in aspects.most_common(8)),
        "",
        "### Example batches",
        "",
    ]
    for title, sel in [
        ("Bad batches caught", caught),
        ("Bad batches let through", g[(g.label == LABEL_REJECTED) & (g.pred == LABEL_ACCEPTED)]),
        ("Good batches thrown away", g[(g.label == LABEL_ACCEPTED) & (g.pred == LABEL_REJECTED)]),
    ]:
        md += [f"**{title}** ({len(sel)}):", ""]
        for _, row in sel.sort_values("score", ascending=False).head(4).iterrows():
            why = "; ".join(f"`{f}` = {v:+.2f}" for f, v in row["reasons"])
            md.append(f"- {row.batch_id} — score {row.score:.3g} — {why}")
        md.append("")
    return md


# ---- PLUMBING: figures ----


def _plot_f1(results: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 3.8), squeeze=False)
    for ax, (proc, res) in zip(axes[0], results.items(), strict=True):
        f = res["folds"]
        names = list(METHODS)
        ax.boxplot([f.loc[f.method == m, "f1"] for m in names], tick_labels=names, showmeans=True)
        ax.axhline(f.loc[f.method == "accept_all", "f1"].mean(), ls="--", c="grey", lw=1)
        ax.set_title(f"Process {proc}: F1 per held-out fold")
        ax.set_ylabel("leaderboard F1")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_scores(g: pd.DataFrame, method: str, proc: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    rng = np.random.default_rng(SEED)
    for i, lab in enumerate([LABEL_ACCEPTED, LABEL_REJECTED]):
        s = g.loc[g.label == lab, "score"].to_numpy()
        ax.scatter(
            s, i + rng.uniform(-0.15, 0.15, len(s)), s=14, alpha=0.7, c=COLORS[lab], label=lab
        )
    if (g.score > 0).all() and g.score.max() / max(g.score.min(), 1e-12) > 100:
        ax.set_xscale("log")
    ax.set_yticks([0, 1], [LABEL_ACCEPTED, LABEL_REJECTED])
    ax.set_xlabel(f"{method} score (higher = more abnormal), held-out batches")
    ax.set_title(f"Process {proc}: how well the scores separate the two classes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_counter(c: Counter, title: str, path: Path) -> None:
    items = c.most_common(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if items:
        ax.barh([k for k, _ in items], [v for _, v in items], color="#3d5a80")
    ax.set_title(title)
    ax.set_xlabel("times among the top reasons")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
