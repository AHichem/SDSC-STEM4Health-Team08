# BRPMM strategy — Team 03 (Renku group08)

Draft for the team to validate. Edit freely: tick a box when we agree, write your name next to a task you take.

## 1. What we must deliver (deadline Sunday 22 Nov, 15:00 — no pushes after)

- **Leaderboard:** for each test batch, predict `accepted` / `rejected` (file `BRPMM_template.csv`, 201 rows: 161 process A + 40 process B).
- **Score:** mean of F1 on project A and F1 on project B. Positive class is very probably `accepted`
  (deduced from the reference entry's accuracy/precision/recall: ~120 positives out of 161). Confirm with organisers.
- **Final submission:** repository link + Markdown description (required), slide deck or README (optional).
- **Rules that shape the strategy:**
  - Attempts count: same score with fewer attempts ranks higher → **few, deliberate submissions**.
  - Brute-forcing the 200 test labels = disqualification.
  - They read the repo: **a score we can't explain from the code can be disqualified** → every submission comes from committed code.

## 2. The data

| | Process A (real industrial) | Process B (penicillin simulation, PenSim-like) |
|---|---|---|
| Variables | 11: flow_rate_1/2, density, pressure_1–4, temperature_1–4 | 19 (15 after dropping 4 constant set-points in `data.py`) |
| Calibration | 161 batches: 121 accepted / 40 rejected | 40 batches: 35 accepted / **5 rejected** |
| Test | 161 batches | 40 batches |
| Batch length | **varies, 196–2932 rows** (median ~2720) | fixed, 5001 rows |
| Time | no time column: row order = time (`data.py` adds index `t`) | same |

Baselines to beat (if test ratios match calibration): "accept everything" ≈ F1 0.86 on A, 0.93 on B.
Reference entry on the leaderboard: A F1 0.898, B F1 1.0 (score 0.949). **Process A is where the ranking is decided.**

## 3. How we measure (same for every method)

**The four cases** — we name results by what the model said and what the label says:

| Model said | Label says | Meaning |
|---|---|---|
| accepted | accepted | correct |
| accepted | rejected | **bad batch let through** (the dangerous mistake) |
| rejected | rejected | correct — bad batch caught |
| rejected | accepted | **good batch thrown away** (the costly mistake) |

- **Main metric = leaderboard F1**, with `accepted` as the positive class.
  ⚠️ `data.to_binary` encodes `rejected = 1`. scikit-learn's `f1_score` scores class 1 by default, which would be
  F1(rejected), the wrong number. Everyone uses one shared function `metrics.leaderboard_f1` (`pos_label=0`), never `f1_score` directly.
  Check: the reference counts 110 / 15 / 26 / 10 must give 0.898.
- **Also reported:** bad batches caught (e.g. 26 of 41) — not in the leaderboard score, but what a plant cares about.

**Validation = cross-validation on the calibration data.** We have no separate validation set, so we rotate:
- Stratified 5-fold (each fold keeps the accepted/rejected ratio), repeated 5 times with different shuffles, fixed seed.
- A model is **never scored on batches it was trained on**.
- Why 5 folds: process B has 5 rejected batches → exactly 1 per held-out fold; with 6+ folds some folds would have none.
  For A, 5 folds leave 8 rejected per held-out fold and train on 80 % of the data.
- **All methods use exactly the same splits**, defined once in config. Report mean ± std and how many splits one method beats another.
- With 161 batches, differences below ~0.03 F1 are probably noise.
- The test set is only scored by the leaderboard (max 3 times).

## 4. The pipeline: same steps for both processes, a separate model per process

```
load → audit → batch features → method (fit + score) → cut-off → CV report → submission file
```

One model for both processes would not work (different sensors, lengths, failures). We share the **code and the steps**;
each process gets its **own fitted model**, and the winning method may differ (likely supervised for A, one-class for B).

**Code layout** (builds on the existing package):

| File | Status | Role |
|---|---|---|
| `src/brpmm/config.py` | done | paths, constants; add SEED and CV settings here |
| `src/brpmm/data.py` | done | load calibration/test, labels, drop constant B columns |
| `src/brpmm/submission.py` | done | fill/validate template, save, submit |
| `src/brpmm/metrics.py` | **to do** | `leaderboard_f1`, the four cases |
| `src/brpmm/features.py` | **to do** | one row per batch (see below) |
| `src/brpmm/methods.py` | **to do** | each method = `fit(X, y)` + `score(model, X)` (higher = more abnormal) |
| `src/brpmm/evaluate.py` | **to do** | runs every method on the same CV splits, writes the results table |

**Batch features (one row per batch).** Per sensor: mean, std, min, max, quantiles, start/end values, slope, and the
mean in ~10 equal time segments. Plus batch length (A varies). Works for both processes and is easy to explain.

**Audit first.** Missing values, impossible values, stuck sensors, odd batches. Key question for A: are the short
batches (~200 rows) the rejected ones?

## 5. Methods

Every method gives each batch a score; the accept/reject **cut-off is chosen on the training part of each split only**.

**First round (all five on the same splits):**

| # | Method | Type | Why |
|---|---|---|---|
| 1 | Accept everything | baseline | the floor every method must beat |
| 2 | Per-variable limits | one-class | normal range per feature from accepted batches; what plants use; suits A's low correlation; easiest to explain |
| 3 | PCA (MSPC: T² + SPE) | one-class | trained on accepted batches only; the industry method in the challenge slide; no failures needed (B has 5) |
| 4 | Logistic regression (regularised) | supervised | simplest classifier; if it ties with RF, prefer it |
| 5 | Random forest | supervised | handles many features, low correlation, non-linear; feature importance explains rejections. Keep trees shallow (`min_samples_leaf` 3–5) |

**Second round, only if the first round points there:**
- supervised wins on A → gradient boosting
- one-class wins on B → isolation forest / Mahalanobis distance
- time-aware: multiway PCA on full trajectories; self-supervised autoencoder (the brief's SSL angle) only if it beats round 1

## 6. The strategy story (what the jury asks for)

| Slide point | What we show |
|---|---|
| Limited initial calibration set | **Learning curve:** CV score vs number of calibration batches (10, 20, 40, 80, all) |
| Low correlation among variables | Correlation matrix of A; per-variable limits vs PCA vs RF compared |
| Sensitivity to unseen variability | Train on early batches, test on later ones; add noise/shifts to accepted batches and count good batches thrown away |
| Acceptable vs unacceptable variation | The four cases + *which sensor* drove each rejection (contributions, feature importance) |
| Easy recalibration / self-supervised | One-class methods refit on new accepted batches in seconds, no labels needed; how many new batches fix a drift |

## 7. Submission policy

- **Max 3 leaderboard submissions** unless the team agrees otherwise:
  1. first solid model to check the pipeline end to end,
  2. best model after improvements,
  3. final, frozen by Sunday ~13:00.
- Each one: committed code, CV score logged in `SUBMISSIONS.md`, agreed by at least two people.
- Team name for `submit()` (set `BRPMM_TEAM` in `.env`): **to confirm** (Team 03 or group08?).

## 8. Plan by milestones

**Saturday afternoon**
- [ ] `metrics.py` + test against the reference numbers (who: ___)
- [ ] Audit report for A and B (who: ___)
- [ ] `features.py` (who: ___)
- [ ] `evaluate.py` + methods 1–5 (who: ___)
- [ ] Domain questions to mentors (who: ___)

**Saturday evening**
- [ ] First results table, first submission
- [ ] Learning curve + early-vs-late test (strategy evidence)
- [ ] Pitch owner starts slides

**Sunday morning**
- [ ] Second-round methods, only where the table says so
- [ ] Explainability plots (which sensor drove each rejection)
- [ ] Final model, final submission, README / description

**Sunday 13:00–15:00**
- [ ] Code freeze, final Markdown description, slides, rehearsal. **Last push before 15:00.**

## 9. Open questions for mentors / organisers

- Is `accepted` the positive class for F1?
- Team name to use for submissions; any limit on the number of attempts?
- What do process A's variables physically measure? Any known phases inside a batch?
- Why are some A batches so short (aborted runs)?
- May we use the **unlabelled test trajectories** for self-supervised pretraining (no labels involved)? Only if they say yes.

## 10. Decisions to validate (tick when agreed)

- [ ] Same pipeline steps, separate model per process
- [ ] Repeated stratified 5-fold CV on calibration data is the only basis for choosing models
- [ ] Leaderboard F1 with `accepted` as positive, via `metrics.leaderboard_f1` only
- [ ] First round = the five methods above, on the same splits
- [ ] Max 3 leaderboard submissions
- [ ] Roles above
