"""Process A: can tuning, more views, cleaner inputs, synthetic rejects or TabPFN improve the
submitted ensemble? Calibration data only.

    python scripts/tune_process_a.py --part hpo       # ~10 min: nested grid search, Maria's code
    python scripts/tune_process_a.py --part combine   # ~15 min: extra members, her combination rule
    python scripts/tune_process_a.py --part clean     # ~2 min: her ensemble on cleaned batches
    python scripts/tune_process_a.py --part augment   # ~2 min: synthetic rejects for the SVM member
    python scripts/tune_process_a.py --part tabpfn    # ~15 min: pip install -e ".[tabpfn]"
    python scripts/tune_process_a.py --part semi      # ~3 min: self-training on unlabelled batches
    python scripts/tune_process_a.py --part semi_half # ~2 min: same, unlabelled pool = labelled size
    python scripts/tune_process_a.py --part rules     # ~15 min: cut-off rules, subagging
    python scripts/tune_process_a.py --part rules_confirm  # same on new splits s = 60..63
    python scripts/tune_process_a.py --part rules_half     # same, 2 folds

Every part uses the fresh splits of scripts/select_final_a.py (StratifiedKFold(5, shuffle, s),
s = 50..53) so every number is comparable with the submitted model's 0.941 there. The test
set and the leaderboard play no part: tuning on leaderboard feedback would fit the test set.

hpo      For every outer training fold, each of the 81 settings is scored by an inner 3-fold
         CV (with its own inner cut-off); the best is refitted and scored on the outer fold.
         "default" runs the same procedure with the submitted settings fixed. A gain counts
         only if the tuned procedure beats default on these outer folds.
combine  Our extra views (MiniRocket, sample-level trees) added to the re-implemented members,
         combined with Maria's rule (scores standardised on training out-of-fold scores).
clean    The submitted ensemble, unchanged, on batches with sensor-dropout rows removed, and
         additionally with the start-up / shut-down transients trimmed.
augment  The supervised SVM member trained on rejects multiplied synthetically inside each
         training fold: SMOTE (interpolation between rejects) or jittered copies (noise 0.1 sd).
tabpfn   TabPFN, a pretrained transformer for small tabular data: alone, as a fourth member of
         the re-implemented ensemble, and (v2+) as a fourth member of the submitted ensemble.
semi     Semi-supervised self-training, simulated: each held-out fold plays the unlabelled test
         set. The submitted ensemble labels the held-out batches it is confident about (score
         beyond its cut-off by a margin), adds them to training with those pseudo-labels, picks
         a new cut-off and scores the fold. Real test data is never used. semi_half uses 2
         folds, so the unlabelled pool is as large as the labelled set (as with the real
         161 test batches); its numbers compare only with its own "submitted" row.
rules    How the cut-off is set, and variance reduction, around the unchanged ensemble: the
         final-fit rule (median of 5 x 5-fold cut-offs, brpmm.predict.cv_cutoff), one cut-off
         on the pooled 5 x 5 out-of-fold scores, Platt-calibrated P(rejected) with the
         F1-optimal rule P(accepted) > F*/2, subagging (25 ensembles on stratified 80 %
         subsamples, cut-off on out-of-bag scores), and self-training (confident accepted,
         margin 1.0) with and without subagging. rules_confirm re-runs it on splits never
         used before; rules_half uses 2 folds like semi_half.
The dropout detector's noise scale is fitted on all calibration batches without labels.
Writes outputs/process_A/tuning/<part>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from contextlib import contextmanager
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import brpmm.evaluate as maria_eval
import brpmm.methods as maria
from brpmm.config import (
    CV_FOLDS,
    CV_REPEATS,
    ENSEMBLE_KNN_K,
    LABEL_ACCEPTED,
    LABEL_REJECTED,
    OCSVM_NU,
    OUTPUTS_DIR,
    SEED,
    SVM_C,
    SVM_GAMMA_MULT,
)
from brpmm.data import load_calibration, to_binary
from brpmm.evaluation import (
    FRESH_SEEDS,
    leaderboard_summary,
    repeated_cv,
    subagged,
    submitted_ensemble_decisions,
    tuned_threshold,
)
from brpmm.features import batch_features as maria_batch_features
from brpmm.metrics import leaderboard_f1
from brpmm.models import (
    SETTINGS,
    ensemble_members,
    median_std_features,
    on_columns,
    resampled_cube,
    sample_table,
    steady_rows,
    team_members,
    zscore_ensemble,
)
from brpmm.preprocessing import (
    SEGMENT_PARAMS,
    STEADY,
    gross_outlier_mask,
    outlier_scale,
    segment_batches,
)
from brpmm.steady_features import batch_features, to_model_space

warnings.filterwarnings("ignore")
TOP_OF_LEADERBOARD = 0.939  # Team 13, F1(A)
OUT = OUTPUTS_DIR / "process_A" / "tuning"
TABPFN_MODEL_VERSION = "v3.5"  # explicit, so the record states which checkpoint ran
DEFAULT = {
    "SVM_C": SVM_C,
    "SVM_GAMMA_MULT": SVM_GAMMA_MULT,
    "ENSEMBLE_KNN_K": ENSEMBLE_KNN_K,
    "OCSVM_NU": OCSVM_NU,
}
GRID = {
    "SVM_C": [0.1, 0.3, 1.0],
    "SVM_GAMMA_MULT": [1.0, 2.0, 4.0],
    "ENSEMBLE_KNN_K": [3, 5, 9],
    "OCSVM_NU": [0.02, 0.05, 0.1],
}


def summary(dec: np.ndarray, y: np.ndarray) -> dict:
    return leaderboard_summary(dec, y, TOP_OF_LEADERBOARD)


# --------------------------------------------------------------------------- Maria's code
@contextmanager
def maria_settings(cfg: dict):
    """Maria's methods read their settings from module globals at call time; set them for
    the duration of one fit and always restore them (single-threaded use only)."""
    old = {k: getattr(maria, k) for k in cfg}
    for k, v in cfg.items():
        setattr(maria, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(maria, k, v)


@contextmanager
def extra_maria_member(name: str, fit, score):
    """Temporarily add a member to Maria's ensemble registry (brpmm.methods._MEMBERS), so
    her own fit_ensemble / score_ensemble (OOF-standardised mean) include it."""
    maria._MEMBERS[name] = (fit, score)
    try:
        yield
    finally:
        del maria._MEMBERS[name]


def fit_with(cfg: dict):
    def fit(X, y):
        with maria_settings(cfg):
            return maria.fit_ensemble(X, y)

    return fit


def cut_and_predict(cfg, X_tr, y_tr, X_te, seed):
    fit = fit_with(cfg)
    cut = maria_eval._inner_cutoff(fit, maria.score_ensemble, X_tr, y_tr, seed=seed)
    return maria_eval._labels(maria.score_ensemble(fit(X_tr, y_tr), X_te), cut)


def inner_f1(cfg, X, y, seed) -> float:
    pred = np.empty(len(X), dtype=object)
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        pred[te] = cut_and_predict(cfg, X.iloc[tr], y.iloc[tr], X.iloc[te], seed + 1)
    return leaderboard_f1(y, pred)


def closeness(cfg) -> float:
    """Tie-break towards the submitted settings (log distance)."""
    return sum(abs(np.log(cfg[k] / DEFAULT[k])) for k in cfg)


# --------------------------------------------------------------------------- shared pieces
def cached(fp):
    """Memoise a member on (train ids, test ids) so candidates sharing it reuse the fit."""
    store = {}

    def wrapped(X_tr, y_tr, X_te):
        key = (tuple(X_tr.index), tuple(X_te.index))
        if key not in store:
            store[key] = fp(X_tr, y_tr, X_te)
        return store[key]

    return wrapped


def team_frame():
    """Calibration batches of A: 22 median+std features and the 90 steady-state features."""
    ts, lab = load_calibration("A")
    params = SEGMENT_PARAMS["A"]
    mask = gross_outlier_mask(ts, params, outlier_scale(ts, params))
    ours90 = to_model_space(batch_features(ts, params, exclude=mask).loc[lab.index])
    f22 = median_std_features(ts).loc[lab.index]
    X = pd.concat([ours90, f22], axis=1)
    return ts, lab, X, to_binary(lab).to_numpy(), list(f22.columns), list(ours90.columns)


def run_candidates(candidates: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    out = {}
    for name, fp in candidates.items():
        t0 = time.time()
        res = repeated_cv(
            tuned_threshold(fp, inner_splits=3, seed=7),
            X,
            y,
            name,
            n_repeats=len(FRESH_SEEDS),
            seed=FRESH_SEEDS[0],
        )
        out[name] = summary(res.oof_decisions.to_numpy(), y)
        print(f"  {name:44s} {out[name]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


# --------------------------------------------------------------------------- parts
def part_hpo() -> dict:
    X, yl = maria_eval.load_features("A")
    y = (yl == LABEL_REJECTED).astype(int).to_numpy()
    configs = [dict(zip(GRID, v, strict=True)) for v in itertools.product(*GRID.values())]
    dec = {"default": [], "tuned": []}
    chosen = []
    for s in FRESH_SEEDS:
        rep = {"default": np.zeros(len(X), int), "tuned": np.zeros(len(X), int)}
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, yl)):
            Xtr, ytr, Xte = X.iloc[tr], yl.iloc[tr], X.iloc[te]
            cut_seed = s + 1000 + k  # the harness's own cut-off seed for one repeat
            scores = [
                (inner_f1(c, Xtr, ytr, cut_seed), -closeness(c), i) for i, c in enumerate(configs)
            ]
            best = configs[max(scores)[2]]
            chosen.append({"seed": s, "fold": k, **best, "inner_f1": round(max(scores)[0], 4)})
            for name, cfg in (("default", DEFAULT), ("tuned", best)):
                rep[name][te] = cut_and_predict(cfg, Xtr, ytr, Xte, cut_seed) == LABEL_REJECTED
            print(f"  seed {s} fold {k}: chose {best}", flush=True)
        for name in dec:
            dec[name].append(rep[name])
    on_all = max((inner_f1(c, X, yl, 999), -closeness(c), i) for i, c in enumerate(configs))
    return {
        "grid": GRID,
        "default": summary(np.array(dec["default"]), y),
        "tuned_nested": summary(np.array(dec["tuned"]), y),
        "chosen_per_fold": chosen,
        "would_choose_on_all_161": configs[on_all[2]],
    }


def part_combine() -> dict:
    ts, lab, X, y, c22, c90 = team_frame()
    st = steady_rows(ts)
    rows = sample_table(st)
    cube = resampled_cube(st, list(lab.index))
    pos = {b: i for i, b in enumerate(lab.index)}
    ours = {
        n: on_columns(fp, c90)
        for n, fp in ensemble_members(rows, lambda ids: cube[[pos[b] for b in ids]]).items()
    }
    team = team_members(c22)
    return run_candidates(
        {
            "team (z)": zscore_ensemble(team),
            "team + minirocket (z)": zscore_ensemble({**team, "minirocket": ours["minirocket"]}),
            "team + sample_et (z)": zscore_ensemble({**team, "sample_et": ours["sample_et"]}),
            "team + minirocket + sample_et (z)": zscore_ensemble(
                {**team, "minirocket": ours["minirocket"], "sample_et": ours["sample_et"]}
            ),
        },
        X,
        y,
    )


def part_clean() -> dict:
    ts, lab = load_calibration("A")
    params = SEGMENT_PARAMS["A"]
    mask = gross_outlier_mask(ts, params, outlier_scale(ts, params))
    steady = segment_batches(ts, params) == STEADY
    y = to_binary(lab).to_numpy()
    out = {}
    for name, view in {
        "raw batch (submitted)": ts,
        "dropout rows removed": ts[~mask],
        "dropout rows removed + transients trimmed": ts[~mask & steady],
    }.items():
        X = maria_batch_features(view).loc[lab.index]
        out[name] = summary(submitted_ensemble_decisions(X, lab), y)
        print(f"  {name:44s} {out[name]}", flush=True)
    return out


def svm_member(cols: list[str], augment: str | None):
    """The team SVM (SETTINGS['svm22']); optionally trained on synthetically enlarged rejects."""
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    sv = SETTINGS["svm22"]

    def fit_predict(X_tr, y_tr, X_te):
        sc = StandardScaler().fit(X_tr[cols])
        A, yy = sc.transform(X_tr[cols]), np.asarray(y_tr)
        if augment == "smote":
            A, yy = SMOTE(random_state=0).fit_resample(A, yy)
        elif augment == "jitter":
            rng = np.random.default_rng(0)
            rej = A[yy == 1]
            n_new = int((yy == 0).sum() - (yy == 1).sum())
            new = rej[rng.integers(len(rej), size=n_new)] + rng.normal(0, 0.1, (n_new, A.shape[1]))
            A, yy = np.vstack([A, new]), np.concatenate([yy, np.ones(n_new, int)])
        clf = SVC(
            C=sv["C"],
            gamma=sv["gamma_mult"] / len(cols),
            class_weight="balanced" if augment is None else None,
        ).fit(A, yy)
        s = clf.decision_function(sc.transform(X_te[cols]))
        return s, (s > 0).astype(int)

    return fit_predict


def part_augment() -> dict:
    _, _, X, y, c22, _ = team_frame()
    team = {n: cached(fp) for n, fp in team_members(c22).items()}

    def with_svm(augment: str) -> dict:
        return {**team, "svm22": cached(svm_member(c22, augment))}

    return run_candidates(
        {
            "team (z)": zscore_ensemble(team),
            "team, SVM on SMOTE-enlarged rejects (z)": zscore_ensemble(with_svm("smote")),
            "team, SVM on jittered reject copies (z)": zscore_ensemble(with_svm("jitter")),
            "SVM alone": team["svm22"],
            "SVM alone, SMOTE": cached(svm_member(c22, "smote")),
        },
        X,
        y,
    )


def tabpfn_v1() -> bool:
    return version("tabpfn").startswith("0.")  # v1 has no tabpfn.__version__


def new_tabpfn():
    """TabPFN v2+ (needs TABPFN_TOKEN and its licence) or v1 (open weights, <= 100 features,
    CPU; runs with torch 2.1, numpy < 2, scikit-learn 1.5). Raw features: it scales itself."""
    import torch
    from tabpfn import TabPFNClassifier

    if tabpfn_v1():
        return TabPFNClassifier(device="cpu", N_ensemble_configurations=32, seed=0)
    return TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=0)


def tabpfn_member(cols: list[str]):
    def fit_predict(X_tr, y_tr, X_te):
        m = new_tabpfn().fit(X_tr[cols].to_numpy(np.float32), np.asarray(y_tr))
        s = m.predict_proba(X_te[cols].to_numpy(np.float32))[:, 1]
        return s, (s >= 0.5).astype(int)

    return fit_predict


def fit_tabpfn_maria(X: pd.DataFrame, y: pd.Series) -> dict:
    """TabPFN on the submitted ensemble's 22 median+std columns."""
    cols = maria._median_std_columns(X)
    clf = new_tabpfn().fit(
        X[cols].to_numpy(np.float32), (pd.Series(y).to_numpy() == LABEL_REJECTED).astype(int)
    )
    return {"clf": clf, "cols": cols}


def score_tabpfn_maria(model: dict, X: pd.DataFrame) -> np.ndarray:
    return model["clf"].predict_proba(X[model["cols"]].to_numpy(np.float32))[:, 1]


def part_tabpfn() -> dict:
    os.environ.setdefault("TABPFN_MODEL_VERSION", TABPFN_MODEL_VERSION)
    record = {
        "tabpfn_package": version("tabpfn"),
        "tabpfn_model": "v1 (open weights)" if tabpfn_v1() else os.environ["TABPFN_MODEL_VERSION"],
    }
    if not tabpfn_v1():  # the submitted ensemble, with and without TabPFN as a 4th member
        Xm, labels = maria_eval.load_features("A")
        y_m = (labels == LABEL_REJECTED).astype(int).to_numpy()
        t0 = time.time()
        record["maria ensemble (submitted)"] = summary(
            submitted_ensemble_decisions(Xm, labels), y_m
        )
        with extra_maria_member("tabpfn", fit_tabpfn_maria, score_tabpfn_maria):
            record["maria ensemble + TabPFN"] = summary(
                submitted_ensemble_decisions(Xm, labels), y_m
            )
        for key in ("maria ensemble (submitted)", "maria ensemble + TabPFN"):
            print(f"  {key:44s} {record[key]}", flush=True)
        print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    _, _, X, y, c22, c90 = team_frame()
    team = {n: cached(fp) for n, fp in team_members(c22).items()}
    tab22 = cached(tabpfn_member(c22))
    candidates = {
        "TabPFN alone (22 median+std)": tab22,
        "team + TabPFN (z)": zscore_ensemble({**team, "tabpfn": tab22}),
    }
    if not tabpfn_v1():  # v1 accepts at most 100 features
        candidates["TabPFN alone (22 + 90 steady features)"] = cached(tabpfn_member(c22 + c90))
    return {**record, **run_candidates(candidates, X, y)}


def part_semi(n_splits: int = 5) -> dict:
    X, labels = maria_eval.load_features("A")
    y = (labels == LABEL_REJECTED).astype(int).to_numpy()
    fit, score = maria.fit_ensemble, maria.score_ensemble
    variants = {  # (which pseudo-labels to add, margin in ensemble-score units around the cut-off)
        "submitted (labelled batches only)": None,
        "self-training, both classes, margin 0.5": ("both", 0.5),
        "self-training, both classes, margin 1.0": ("both", 1.0),
        "self-training, both classes, margin 2.0": ("both", 2.0),
        "self-training, confident accepted only, margin 1.0": ("accepted", 1.0),
    }
    dec = {name: [] for name in variants}
    added = {name: [] for name in variants}
    for s in FRESH_SEEDS:
        rep_dec = {name: np.zeros(len(X), int) for name in variants}
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=s).split(X, labels)
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr, Xte = X.iloc[tr], labels.iloc[tr], X.iloc[te]
            cut_seed = s + 1000 + k  # harness cut-off seed: with 5 folds "submitted" gives 0.941
            cut = maria_eval._inner_cutoff(fit, score, Xtr, ytr, seed=cut_seed)
            s_te = score(fit(Xtr, ytr), Xte)
            for name, variant in variants.items():
                if variant is None:
                    rep_dec[name][te] = s_te > cut
                    continue
                kind, margin = variant
                sure_rej = s_te > cut + margin if kind == "both" else np.zeros(len(te), bool)
                sure_acc = s_te < cut - margin
                pick = sure_rej | sure_acc
                pseudo = pd.Series(
                    np.where(sure_rej, LABEL_REJECTED, maria.LABEL_ACCEPTED), index=Xte.index
                )[pick]
                Xa, ya = pd.concat([Xtr, Xte[pick]]), pd.concat([ytr, pseudo])
                cut_a = maria_eval._inner_cutoff(fit, score, Xa, ya, seed=cut_seed)
                rep_dec[name][te] = score(fit(Xa, ya), Xte) > cut_a
                added[name].append(int(pick.sum()))
        for name in variants:
            dec[name].append(rep_dec[name])
    out = {}
    for name in variants:
        out[name] = summary(np.array(dec[name]), y)
        if added[name]:
            out[name]["pseudo_labelled_per_fold"] = round(float(np.mean(added[name])), 1)
        print(f"  {name:52s} {out[name]}", flush=True)
    return out


def _oof(fit, score, X: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray:
    o = np.empty(len(X))
    for tr, te in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X, y):
        o[te] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
    return o


def _confident_accepted(X, y, Xte, s_te, cut, margin=1.0):
    sure = s_te < cut - margin
    pseudo = pd.Series(LABEL_ACCEPTED, index=Xte.index[sure])
    return pd.concat([X, Xte[sure]]), pd.concat([y, pseudo])


def part_rules(seeds=FRESH_SEEDS, n_splits: int = 5) -> dict:
    X, labels = maria_eval.load_features("A")
    y = (labels == LABEL_REJECTED).astype(int).to_numpy()
    fit, score = maria.fit_ensemble, maria.score_ensemble
    names = [
        "submitted (inner 3-fold cut-off, evaluation harness)",
        "median of 5 x 5-fold cut-offs (final-fit rule)",
        "one cut-off on pooled 5 x 5 out-of-fold scores",
        "Platt P(rejected), accept if P(accepted) > F*/2",
        "subagging 25 x 80 %, out-of-bag cut-off",
        "self-training (accepted, margin 1.0), final-fit rule",
        "self-training (accepted, margin 1.0) + subagging",
    ]
    dec = {name: [] for name in names}
    for s in seeds:
        rep = {name: np.zeros(len(X), int) for name in names}
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=s).split(X, labels)
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr, Xte = X.iloc[tr], labels.iloc[tr], X.iloc[te]
            s_te = score(fit(Xtr, ytr), Xte)
            c0 = maria_eval._inner_cutoff(fit, score, Xtr, ytr, seed=s + 1000 + k)
            oofs = [_oof(fit, score, Xtr, ytr, SEED + r) for r in range(CV_REPEATS)]
            c1 = float(np.median([maria_eval.choose_cutoff(o, ytr) for o in oofs]))
            pooled, ypool = np.concatenate(oofs), pd.concat([ytr] * CV_REPEATS)
            c2 = maria_eval.choose_cutoff(pooled, ypool)
            platt = LogisticRegression(C=1e6).fit(pooled[:, None], ypool == LABEL_REJECTED)
            f_star = leaderboard_f1(ypool, maria_eval._labels(pooled, c2))
            p_acc = 1 - platt.predict_proba(s_te[:, None])[:, 1]
            c4, s4 = subagged(fit, score, Xtr, ytr, Xte, n_bags=25, seed=s * 10 + k)
            Xa, ya = _confident_accepted(Xtr, ytr, Xte, s_te, c1)
            c5 = float(
                np.median(
                    [
                        maria_eval.choose_cutoff(_oof(fit, score, Xa, ya, SEED + r), ya)
                        for r in range(CV_REPEATS)
                    ]
                )
            )
            s5 = score(fit(Xa, ya), Xte)
            Xb, yb = _confident_accepted(Xtr, ytr, Xte, s4, c4)
            c6, s6 = subagged(fit, score, Xb, yb, Xte, n_bags=25, seed=s * 10 + k + 5)
            for name, d in zip(
                names,
                [s_te > c0, s_te > c1, s_te > c2, p_acc <= f_star / 2, s4 > c4, s5 > c5, s6 > c6],
                strict=True,
            ):
                rep[name][te] = d
            print(f"  seed {s} fold {k}", flush=True)
        for name in names:
            dec[name].append(rep[name])
    out = {"seeds": list(seeds), "folds": n_splits}
    for name in names:
        out[name] = summary(np.array(dec[name]), y)
        print(f"  {name:56s} {out[name]}", flush=True)
    return out


PARTS = {
    "hpo": part_hpo,
    "combine": part_combine,
    "clean": part_clean,
    "augment": part_augment,
    "tabpfn": part_tabpfn,
    "semi": part_semi,
    "semi_half": lambda: part_semi(n_splits=2),
    "rules": part_rules,
    "rules_confirm": lambda: part_rules(seeds=(60, 61, 62, 63)),
    "rules_half": lambda: part_rules(n_splits=2),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=list(PARTS), required=True)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "splits": f"StratifiedKFold(5, shuffle, s) for s in {list(FRESH_SEEDS)}",
        **PARTS[args.part](),
    }
    (OUT / f"{args.part}.json").write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "chosen_per_fold"}, indent=2, default=str
        )
    )


if __name__ == "__main__":
    main()
