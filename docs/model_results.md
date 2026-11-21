# BRPMM model results — Team 03 (Maria's runs)

Last updated: 2026-11-21 19:45 (Europe/Madrid). Branch `maria` of `SDSC-STEM4Health-group08`.

**Purpose.** One place to compare every model the team has trained, on the same terms. Each model has a fixed
id, what it was trained on, how it was validated, and the same metrics. A teammate (or another LLM) can add a
row in the same format and compare directly. A machine-readable copy of all rows is at the end (JSON).

## Best so far

| Process | Use for the leaderboard | F1 (accepted positive) | Honest estimate |
|---|---|---|---|
| A | `ensemble` (SVM + kNN on PCA + one-class SVM), id A24 | 0.943 ± 0.007 at fixed settings | **0.930 ± 0.014** (A25, everything chosen inside training) |
| B | `pca_mspc` trained on accepted batches only, id B02 | 1.000 ± 0.000 | 1.000 (no tuning involved) |

For round 2 (use in a plant), the one-class models trained on accepted batches only are the story:
`ocsvm` (A 0.899, B 0.997) and `pca_limits_simple` (A 0.873, B 0.997) name the sensor behind each rejection.

## The task and the data

- Classify each **batch** as `accepted` or `rejected`. Labels exist per batch, not per sample.
- Process A (real industrial): 11 sensors, 161 calibration batches (121 accepted, 40 rejected), batch
  length 196–2932 rows. Process B (penicillin simulation): 19 sensors (4 constant, dropped), 40 calibration
  batches (35 accepted, 5 rejected), 5001 rows each. Test: 161 (A) and 40 (B) unlabelled batches.
- Data: `/home/renku/work/input-data/Project 2 - BRPMM/` on Renku (local copy in `data/`, gitignored).

## Metric — read this before comparing

- **F1 with `accepted` as the positive class**, per process. `precision` = good kept / all batches we
  accepted; `recall` = good kept / all truly good batches. This is what our tests use (team decision, 21 Nov).
  The leaderboard text says "rejected class", but the published leaderboard numbers only fit `accepted`;
  check with the organisers before relying on either.
- Leaderboard score = mean of F1(A) and F1(B). Every team on the board has F1(B) = 1.000, so **A decides**.
- Also reported: accuracy, and the four cases — good kept, **bad let through**, bad caught, **good thrown away**.
- Code: `brpmm.metrics.leaderboard_f1` (works on the label strings, so the positive class can't be mixed up).
  `brpmm.data.to_binary` encodes rejected = 1; never call sklearn's `f1_score` on that without `pos_label=0`.

## Validation protocol — use the same to compare

1. **Repeated stratified 5-fold cross-validation** on the calibration batches: 5 repeats, `StratifiedKFold(5,
   shuffle=True, random_state=42 + repeat)` for repeat 0..4 → 25 held-out folds. Same splits for every model.
2. A batch is **only ever scored by a model that did not see it**.
3. The accept/reject **cut-off is chosen inside the training part** (inner `StratifiedKFold(3, shuffle=True,
   random_state=1042 + 10*repeat + fold)`) to maximise F1. Models marked "P > 0.5" use a fixed threshold instead.
4. Reported F1 = F1 over all calibration batches per repeat (pooled out-of-fold predictions), mean ± std over
   the 5 repeats. Counts (bad caught etc.) = average per repeat. Precision, recall, accuracy are computed from
   those average counts.
5. **validation column:** `cv` = as above; `nested cv` = model settings also chosen inside training (honest);
   `cv (settings chosen after seeing results)` = optimistic, because the settings were picked from many tries;
   `in-sample` = scored on its own training batches (**not comparable**).
6. With ~160 batches, F1 differences below ~0.03 are probably noise.

**Training data vocabulary:** `accepted only` = the model learns "normal" from accepted batches of the training
part; `accepted + rejected` = learns from both classes with labels; `accepted + rejected, labels ignored` =
everything treated as normal (contaminated reference).

**Features:** `22 median+std` = median and standard deviation of each of the 11 sensors over the whole batch
(the teammate's set; 32 for B). `243 / 393 level+shape` = `brpmm.features.batch_features`: median, start, end,
std, slope, deviation from the batch's own start per tenth, biggest step, within-batch sensor correlations.

## Process A — all models (sorted by F1)

| id | model | family | trained on | features | validation | F1 | ± | prec. | recall | acc. | bad caught | good thrown away |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A24 | `ensemble_fixed` | ensemble (supervised + one-class) | accepted + rejected (ocsvm member: accepted only) | 22 median+std | cv (settings chosen after seeing results) | 0.943 | 0.007 | 0.931 | 0.955 | 0.913 | 31.4 / 40 | 5.4 / 121 |
| A29 | `ensemble_plus_tbin_svm` | ensemble | accepted + rejected (ocsvm member: accepted only) | 22 median+std; tbin member 220 time-bin | cv | 0.941 | 0.007 | 0.926 | 0.957 | 0.911 | 30.8 / 40 | 5.2 / 121 |
| A32 | `ensemble_majority_vote` | ensemble (vote) | accepted + rejected (ocsvm member: accepted only) | 22 median+std | cv | 0.937 | 0.003 | 0.915 | 0.960 | 0.903 | 29.2 / 40 | 4.8 / 121 |
| A19 | `svm_fixed` | supervised | accepted + rejected | 22 median+std (scaled) | cv (settings chosen after seeing results) | 0.931 | 0.006 | 0.910 | 0.954 | 0.894 | 28.6 / 40 | 5.6 / 121 |
| A25 | `ensemble_nested` | ensemble (supervised + one-class) | accepted + rejected | 22 median+std | nested cv | 0.930 | 0.014 | 0.919 | 0.942 | 0.894 | 30.0 / 40 | 7.0 / 121 |
| A31 | `ensemble_plus_tbin_svm_plus_hgb` | ensemble | accepted + rejected (ocsvm member: accepted only) | 22 median+std + 220 time-bin | cv | 0.928 | 0.012 | 0.917 | 0.940 | 0.891 | 29.6 / 40 | 7.2 / 121 |
| A27 | `svm_base_plus_tbins` | supervised | accepted + rejected | 22 median+std + 220 time-bin medians | cv | 0.918 | 0.004 | 0.894 | 0.944 | 0.873 | 26.4 / 40 | 6.8 / 121 |
| A30 | `ensemble_plus_hgb` | ensemble | accepted + rejected (ocsvm member: accepted only) | 22 median+std | cv | 0.918 | 0.009 | 0.907 | 0.931 | 0.876 | 28.4 / 40 | 8.4 / 121 |
| A20 | `svm_nested` | supervised | accepted + rejected | 22 median+std (scaled) | nested cv | 0.914 | 0.012 | 0.900 | 0.927 | 0.868 | 27.6 / 40 | 8.8 / 121 |
| A26 | `tbin_svm` | supervised | accepted + rejected | 220 time-bin medians (20 bins x 11 sensors), absolute deviation from accepted median | cv | 0.914 | 0.004 | 0.894 | 0.936 | 0.868 | 26.6 / 40 | 7.8 / 121 |
| A17 | `pca_knn_classifier_k5` | supervised | accepted + rejected | 22 median+std -> PCA 90 % var | cv | 0.907 | 0.007 | 0.870 | 0.949 | 0.855 | 22.8 / 40 | 6.2 / 121 |
| A16 | `knn_classifier_k7` | supervised | accepted + rejected | 22 median+std (scaled) | cv | 0.903 | 0.009 | 0.863 | 0.947 | 0.847 | 21.8 / 40 | 6.4 / 121 |
| A21 | `mlp` | supervised (deep learning) | accepted + rejected | 22 median+std + log(batch length) | cv | 0.902 | 0.015 | 0.908 | 0.896 | 0.853 | 29.0 / 40 | 12.6 / 121 |
| A18 | `ocsvm` | one-class | accepted only | 22 median+std (scaled) | cv | 0.899 | 0.004 | 0.875 | 0.924 | 0.843 | 24.0 / 40 | 9.2 / 121 |
| A23 | `cnn_1d_hybrid` | supervised (deep learning) | accepted + rejected | trajectories + 23 summary features | cv | 0.891 | 0.010 | 0.882 | 0.901 | 0.835 | 25.4 / 40 | 12.0 / 121 |
| A22 | `cnn_1d` | supervised (deep learning) | accepted + rejected | trajectories 22 ch x 128 time bins | cv | 0.890 | 0.006 | 0.877 | 0.904 | 0.832 | 24.6 / 40 | 11.6 / 121 |
| A15 | `knn_oneclass_k3` | one-class | accepted only | 22 median+std (scaled) | cv | 0.888 | 0.003 | 0.833 | 0.950 | 0.820 | 17.0 / 40 | 6.0 / 121 |
| A12 | `share_any_out` | one-class | accepted only (ranges) | 1: share of samples outside accepted p0.5-p99.5 on any sensor | cv | 0.886 | 0.007 | 0.837 | 0.940 | 0.817 | 17.8 / 40 | 7.2 / 121 |
| A13 | `share_any_out_staged` | one-class | accepted only (ranges) | 1: as A12, range per tenth of batch | cv | 0.885 | 0.007 | 0.833 | 0.945 | 0.816 | 17.0 / 40 | 6.6 / 121 |
| A11 | `teammate_fixed_rule_all` | one-class (contaminated reference) | accepted + rejected, labels ignored | 22 median+std | cv | 0.884 | 0.004 | 0.844 | 0.929 | 0.817 | 19.2 / 40 | 8.6 / 121 |
| A08 | `teammate_fixed_rule_insample` | one-class | accepted only (all 121, scored on same batches) | 22 median+std | in-sample | 0.879 | — | 0.919 | 0.843 | 0.826 | 31.0 / 40 | 19.0 / 121 |
| A14 | `rf_shares_maxstep` | supervised | accepted + rejected | 22 median+std + 13 shares + 11 max-step | cv | 0.876 | 0.011 | 0.812 | 0.950 | 0.798 | 13.4 / 40 | 6.0 / 121 |
| A06 | `pca_limits_simple` | one-class | accepted only | 22 median+std | cv | 0.873 | 0.005 | 0.803 | 0.957 | 0.791 | 11.6 / 40 | 5.2 / 121 |
| A10 | `teammate_fixed_rule_loose` | one-class | accepted only | 22 median+std | cv | 0.871 | 0.008 | 0.919 | 0.828 | 0.816 | 31.2 / 40 | 20.8 / 121 |
| A05 | `random_forest` | supervised | accepted + rejected | 243 level+shape | cv | 0.860 | 0.005 | 0.777 | 0.964 | 0.765 | 6.6 / 40 | 4.4 / 121 |
| A01 | `accept_all` | baseline | none | none | cv | 0.858 | 0.000 | 0.752 | 1.000 | 0.752 | 0.0 / 40 | 0.0 / 121 |
| A28 | `hist_gradient_boosting` | supervised (trees) | accepted + rejected | 22 median+std | cv | 0.858 | 0.008 | 0.780 | 0.954 | 0.763 | 7.4 / 40 | 5.6 / 121 |
| A03 | `pca_mspc` | one-class | accepted only | 243 level+shape | cv | 0.855 | 0.006 | 0.774 | 0.954 | 0.757 | 6.4 / 40 | 5.6 / 121 |
| A07 | `pca_limits_full` | one-class | accepted only | 243 level+shape | cv | 0.855 | 0.006 | 0.774 | 0.954 | 0.757 | 6.4 / 40 | 5.6 / 121 |
| A02 | `feature_limits` | one-class | accepted only | 243 level+shape | cv | 0.854 | 0.000 | 0.750 | 0.992 | 0.745 | 0.0 / 40 | 1.0 / 121 |
| A04 | `logreg` | supervised | accepted + rejected | 243 level+shape | cv | 0.846 | 0.015 | 0.776 | 0.929 | 0.745 | 7.6 / 40 | 8.6 / 121 |
| A09 | `teammate_fixed_rule` | one-class | accepted only | 22 median+std | cv | 0.818 | 0.011 | 0.911 | 0.742 | 0.752 | 31.2 / 40 | 31.2 / 121 |

## Process B — all models (sorted by F1)

| id | model | family | trained on | features | validation | F1 | ± | prec. | recall | acc. | bad caught | good thrown away |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B02 | `pca_mspc` | one-class | accepted only | 393 level+shape | cv | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 5.0 / 5 | 0.0 / 35 |
| B03 | `pca_limits_full` | one-class | accepted only | 393 level+shape | cv | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 5.0 / 5 | 0.0 / 35 |
| B04 | `ocsvm` | one-class | accepted only | 32 median+std | cv | 0.997 | 0.006 | 1.000 | 0.994 | 0.995 | 5.0 / 5 | 0.2 / 35 |
| B05 | `logreg` | supervised | accepted + rejected | 393 level+shape | cv | 0.997 | 0.006 | 0.994 | 1.000 | 0.995 | 4.8 / 5 | 0.0 / 35 |
| B06 | `pca_limits_simple` | one-class | accepted only | 32 median+std | cv | 0.997 | 0.006 | 0.994 | 1.000 | 0.995 | 4.8 / 5 | 0.0 / 35 |
| B07 | `random_forest` | supervised | accepted + rejected | 393 level+shape | cv | 0.992 | 0.007 | 0.983 | 1.000 | 0.985 | 4.4 / 5 | 0.0 / 35 |
| B08 | `ensemble_fixed` | ensemble | accepted + rejected | 32 median+std | cv | 0.975 | 0.013 | 0.961 | 0.989 | 0.955 | 3.6 / 5 | 0.4 / 35 |
| B09 | `svm` | supervised | accepted + rejected | 32 median+std | cv | 0.934 | 0.035 | 0.927 | 0.943 | 0.885 | 2.4 / 5 | 2.0 / 35 |
| B01 | `accept_all` | baseline | none | none | cv | 0.933 | 0.000 | 0.875 | 1.000 | 0.875 | 0.0 / 5 | 0.0 / 35 |
| B10 | `pca_mspc_trained_on_all` | one-class (contaminated reference) | accepted + rejected, labels ignored | 393 level+shape | cv | 0.933 | 0.000 | 0.875 | 1.000 | 0.875 | 0.0 / 5 | 0.0 / 35 |
| B11 | `feature_limits` | one-class | accepted only | 393 level+shape | cv | 0.910 | 0.007 | 0.929 | 0.891 | 0.845 | 2.6 / 5 | 3.8 / 35 |

## Findings in one paragraph each

- **A:** each rejected batch deviates on a different sensor. Limits and PCA catch only about a third of them;
  boundary methods trained on both classes (SVM, kNN) catch about 70–80 %. More features hurt (243 level+shape
  < 22 median+std): with ~97–129 training batches the signal gets diluted. Deep learning on full trajectories
  (1D-CNN) did not beat the summary-feature SVM.
- **B:** all 5 rejected batches share one fault: aeration drops ~30 % between samples 500 and 1250, then
  recovers. Any one-class model trained on accepted batches finds it. Supervised SVM fails (5 examples).
- **Training on all batches** helps the teammate's fixed rule on A (0.818 → 0.884) by widening its limits, but
  breaks PCA on B (1.000 → 0.933): the fault is absorbed into "normal".

## Round of 21 Nov evening: what was tried after the best model (A26–A32)

- **Where in the batch do A faults show?** Split every batch into 20 equal time bins and compared rejected vs
  accepted per sensor and bin (AUC of |deviation from the accepted median|). The signal is flat over time
  (AUC 0.55–0.74; strongest: temperature_1 everywhere, temperature_4 at 20–30 % of the batch). Unlike B, A has
  no fault window. Time-bin features alone reach 0.914 (A26) but add nothing to the ensemble (A29, 0.941 vs 0.943).
- **Gradient boosting** (A28) is close to accept-all (0.858) and drags the ensemble down (A30, 0.918). Too few
  rejected batches (about 32 per training part) for trees to learn from.
- **Majority vote** of the three members (A32) gives 0.937: slightly lower but steadier (± 0.003), and it
  throws away the fewest good batches (4.8 / 121).
- **Conclusion:** A24 (`ensemble`) stays the model for A. Nothing tried beats it by more than the noise (± 0.01).

## Member agreement: a traffic light for the operator (A24)

Each member votes with its own cut-off (chosen in cross-validation). Out-of-fold on the calibration batches:

| members voting "rejected" | share of batches | truly rejected | ensemble correct |
|---|---|---|---|
| 0 of 3 (green) | 70 % | 6 % | 94 % |
| 1 of 3 (amber) | 9 % | 27 % | 78 % |
| 2 of 3 (amber) | 13 % | 81 % | 86 % |
| 3 of 3 (red) | 8 % | 94 % | 94 % |

The amber group (members disagree) is 22 % of batches but holds 45 % of all errors. Suggested use: green →
release, red → reject, amber → send to a human with the top reasons. This does not change the leaderboard
label (that stays the ensemble decision); it is the decision-support layer for round 2.

## Test set: checks before submitting (no labels, nothing submitted)

Models trained on all calibration batches; cut-off = median of the 5 cross-validation cut-offs.
Saved: `outputs/20261121-181628_A-ensemble_B-pca_NOT-SUBMITTED.csv` (+ `_details.csv` with score and reasons).

- **A (161 batches):** 36 rejected (22 %; calibration 25 %, out-of-fold 22 %). Test score distribution matches
  the out-of-fold calibration scores; batch lengths and columns identical; mild widening of temperature_2.
  10 batches lie within ± 0.25 of the cut-off. Traffic light: 115 green, 25 amber, 21 red.
- **B (40 batches):** 5 rejected (12.5 %, same as calibration). The test batches run at **new operating points**
  (aeration and starting volume outside the calibration range). Standard T²/SPE limits would flag 33 of 40.
  The 5 rejected show the known fault (aeration −30 % at samples 500–1250); the other 35 show none.
  Gap in score: highest normal 3.2 vs lowest fault 490,000, so the split is safe.

## How to add your model (template)

Copy a row into the table of its process and add an object to the JSON list:

```
id: A26                      # next free id for the process
model: <short_name>
family: one-class | supervised | ensemble | baseline | deep learning
trained_on: accepted only | accepted + rejected | accepted + rejected, labels ignored
features: <what goes in, e.g. 22 median+std>
decision: <how a batch becomes 'rejected', incl. threshold rule>
validation: cv | nested cv | cv (settings chosen after seeing results) | in-sample
f1_accepted, f1_std, precision_accepted, recall_accepted, accuracy, bad_caught, good_thrown_away
code: <file and function>
notes: <settings, caveats>
```

If your number is in-sample, rerun it with the protocol above before comparing (`python -m brpmm.evaluate`
does it for every method registered in `brpmm.methods.METHODS`).

## Where the code is

- Repo (`src/brpmm/`): `features.py`, `methods.py` (all `methods.py:` entries), `evaluate.py` (protocol),
  `metrics.py`, `report.py`. Run: `BRPMM_DATA_DIR=... python -m brpmm.evaluate --process A B`.
- `scratch/...` entries: experiment scripts on Maria's laptop (not in the repo yet); same splits and seeds.
- Reports with figures: `outputs/report_<timestamp>/report.md` (gitignored). Shared findings doc:
  https://claude.ai/artifact/KX6J8CK1TAMwukKiu17siS

## All rows, machine-readable

Fields as in the tables; `f1_std` is null for in-sample results; counts are averages per repeat.

```json
[
 {
  "id": "A01",
  "model": "accept_all",
  "process": "A",
  "family": "baseline",
  "trained_on": "none",
  "features": "none",
  "decision": "always 'accepted'",
  "validation": "cv",
  "f1_accepted": 0.858,
  "f1_std": 0.0,
  "precision_accepted": 0.752,
  "recall_accepted": 1.0,
  "accuracy": 0.752,
  "good_kept": 121.0,
  "bad_let_through": 40.0,
  "bad_caught": 0.0,
  "good_thrown_away": 0.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: accept_all",
  "notes": "floor every model must beat"
 },
 {
  "id": "A02",
  "model": "feature_limits",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "243 level+shape",
  "decision": "max robust |z| > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.854,
  "f1_std": 0.0,
  "precision_accepted": 0.75,
  "recall_accepted": 0.992,
  "accuracy": 0.745,
  "good_kept": 120.0,
  "bad_let_through": 40.0,
  "bad_caught": 0.0,
  "good_thrown_away": 1.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: feature_limits",
  "notes": "too many features: some always look extreme by chance"
 },
 {
  "id": "A03",
  "model": "pca_mspc",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "243 level+shape",
  "decision": "max(T2/lim, SPE/lim) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.855,
  "f1_std": 0.006,
  "precision_accepted": 0.774,
  "recall_accepted": 0.954,
  "accuracy": 0.757,
  "good_kept": 115.4,
  "bad_let_through": 33.6,
  "bad_caught": 6.4,
  "good_thrown_away": 5.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: pca_mspc",
  "notes": ""
 },
 {
  "id": "A04",
  "model": "logreg",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "243 level+shape",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.846,
  "f1_std": 0.015,
  "precision_accepted": 0.776,
  "recall_accepted": 0.929,
  "accuracy": 0.745,
  "good_kept": 112.4,
  "bad_let_through": 32.4,
  "bad_caught": 7.6,
  "good_thrown_away": 8.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: logreg",
  "notes": "C=0.05, balanced weights"
 },
 {
  "id": "A05",
  "model": "random_forest",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "243 level+shape",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.86,
  "f1_std": 0.005,
  "precision_accepted": 0.777,
  "recall_accepted": 0.964,
  "accuracy": 0.765,
  "good_kept": 116.6,
  "bad_let_through": 33.4,
  "bad_caught": 6.6,
  "good_thrown_away": 4.4,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: random_forest",
  "notes": "300 trees, min_samples_leaf 3"
 },
 {
  "id": "A06",
  "model": "pca_limits_simple",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "22 median+std",
  "decision": "T2/SPE/per-feature limits, single CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.873,
  "f1_std": 0.005,
  "precision_accepted": 0.803,
  "recall_accepted": 0.957,
  "accuracy": 0.791,
  "good_kept": 115.8,
  "bad_let_through": 28.4,
  "bad_caught": 11.6,
  "good_thrown_away": 5.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: pca_limits_simple",
  "notes": "teammate's recipe, cut-off chosen by CV"
 },
 {
  "id": "A07",
  "model": "pca_limits_full",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "243 level+shape",
  "decision": "as A06",
  "validation": "cv",
  "f1_accepted": 0.855,
  "f1_std": 0.006,
  "precision_accepted": 0.774,
  "recall_accepted": 0.954,
  "accuracy": 0.757,
  "good_kept": 115.4,
  "bad_let_through": 33.6,
  "bad_caught": 6.4,
  "good_thrown_away": 5.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: pca_limits_full",
  "notes": ""
 },
 {
  "id": "A08",
  "model": "teammate_fixed_rule_insample",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only (all 121, scored on same batches)",
  "features": "22 median+std",
  "decision": "T2 or SPE > 95th pct, or any |z| > Bonferroni limit",
  "validation": "in-sample",
  "f1_accepted": 0.879,
  "f1_std": null,
  "precision_accepted": 0.919,
  "recall_accepted": 0.843,
  "accuracy": 0.826,
  "good_kept": 102.0,
  "bad_let_through": 9.0,
  "bad_caught": 31.0,
  "good_thrown_away": 19.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/fixed_rule.py",
  "notes": "NOT comparable: scored on training batches; reproduces teammate notebook"
 },
 {
  "id": "A09",
  "model": "teammate_fixed_rule",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "22 median+std",
  "decision": "as A08, limits fixed",
  "validation": "cv",
  "f1_accepted": 0.818,
  "f1_std": 0.011,
  "precision_accepted": 0.911,
  "recall_accepted": 0.742,
  "accuracy": 0.752,
  "good_kept": 89.8,
  "bad_let_through": 8.8,
  "bad_caught": 31.2,
  "good_thrown_away": 31.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/fixed_rule.py",
  "notes": "honest version of A08"
 },
 {
  "id": "A10",
  "model": "teammate_fixed_rule_loose",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "22 median+std",
  "decision": "99.5th pct + Bonferroni 0.005",
  "validation": "cv",
  "f1_accepted": 0.871,
  "f1_std": 0.008,
  "precision_accepted": 0.919,
  "recall_accepted": 0.828,
  "accuracy": 0.816,
  "good_kept": 100.2,
  "bad_let_through": 8.8,
  "bad_caught": 31.2,
  "good_thrown_away": 20.8,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/fixed_rule_loose.py",
  "notes": "looser limits picked after seeing results"
 },
 {
  "id": "A11",
  "model": "teammate_fixed_rule_all",
  "process": "A",
  "family": "one-class (contaminated reference)",
  "trained_on": "accepted + rejected, labels ignored",
  "features": "22 median+std",
  "decision": "as A08",
  "validation": "cv",
  "f1_accepted": 0.884,
  "f1_std": 0.004,
  "precision_accepted": 0.844,
  "recall_accepted": 0.929,
  "accuracy": 0.817,
  "good_kept": 112.4,
  "bad_let_through": 20.8,
  "bad_caught": 19.2,
  "good_thrown_away": 8.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/fixed_rule_all.py",
  "notes": "reference model includes faulty batches"
 },
 {
  "id": "A12",
  "model": "share_any_out",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only (ranges)",
  "features": "1: share of samples outside accepted p0.5-p99.5 on any sensor",
  "decision": "share > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.886,
  "f1_std": 0.007,
  "precision_accepted": 0.837,
  "recall_accepted": 0.94,
  "accuracy": 0.817,
  "good_kept": 113.8,
  "bad_let_through": 22.2,
  "bad_caught": 17.8,
  "good_thrown_away": 7.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/targeted.py, core.shares_fast",
  "notes": "single feature, very explainable"
 },
 {
  "id": "A13",
  "model": "share_any_out_staged",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only (ranges)",
  "features": "1: as A12, range per tenth of batch",
  "decision": "share > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.885,
  "f1_std": 0.007,
  "precision_accepted": 0.833,
  "recall_accepted": 0.945,
  "accuracy": 0.816,
  "good_kept": 114.4,
  "bad_let_through": 23.0,
  "bad_caught": 17.0,
  "good_thrown_away": 6.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/targeted.py",
  "notes": ""
 },
 {
  "id": "A14",
  "model": "rf_shares_maxstep",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std + 13 shares + 11 max-step",
  "decision": "P(rejected) > 0.5",
  "validation": "cv",
  "f1_accepted": 0.876,
  "f1_std": 0.011,
  "precision_accepted": 0.812,
  "recall_accepted": 0.95,
  "accuracy": 0.798,
  "good_kept": 115.0,
  "bad_let_through": 26.6,
  "bad_caught": 13.4,
  "good_thrown_away": 6.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/targeted.py",
  "notes": ""
 },
 {
  "id": "A15",
  "model": "knn_oneclass_k3",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "22 median+std (scaled)",
  "decision": "mean distance to 3 nearest accepted > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.888,
  "f1_std": 0.003,
  "precision_accepted": 0.833,
  "recall_accepted": 0.95,
  "accuracy": 0.82,
  "good_kept": 115.0,
  "bad_let_through": 23.0,
  "bad_caught": 17.0,
  "good_thrown_away": 6.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp1.py",
  "notes": ""
 },
 {
  "id": "A16",
  "model": "knn_classifier_k7",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std (scaled)",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.903,
  "f1_std": 0.009,
  "precision_accepted": 0.863,
  "recall_accepted": 0.947,
  "accuracy": 0.847,
  "good_kept": 114.6,
  "bad_let_through": 18.2,
  "bad_caught": 21.8,
  "good_thrown_away": 6.4,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp1.py",
  "notes": "distance-weighted"
 },
 {
  "id": "A17",
  "model": "pca_knn_classifier_k5",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std -> PCA 90 % var",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.907,
  "f1_std": 0.007,
  "precision_accepted": 0.87,
  "recall_accepted": 0.949,
  "accuracy": 0.855,
  "good_kept": 114.8,
  "bad_let_through": 17.2,
  "bad_caught": 22.8,
  "good_thrown_away": 6.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp1.py",
  "notes": ""
 },
 {
  "id": "A18",
  "model": "ocsvm",
  "process": "A",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "22 median+std (scaled)",
  "decision": "-decision > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.899,
  "f1_std": 0.004,
  "precision_accepted": 0.875,
  "recall_accepted": 0.924,
  "accuracy": 0.843,
  "good_kept": 111.8,
  "bad_let_through": 16.0,
  "bad_caught": 24.0,
  "good_thrown_away": 9.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: ocsvm",
  "notes": "nu 0.05, gamma 0.5/22"
 },
 {
  "id": "A19",
  "model": "svm_fixed",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std (scaled)",
  "decision": "decision > CV cut-off",
  "validation": "cv (settings chosen after seeing results)",
  "f1_accepted": 0.931,
  "f1_std": 0.006,
  "precision_accepted": 0.91,
  "recall_accepted": 0.954,
  "accuracy": 0.894,
  "good_kept": 115.4,
  "bad_let_through": 11.4,
  "bad_caught": 28.6,
  "good_thrown_away": 5.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: svm",
  "notes": "rbf, C 0.3, gamma 2/22, balanced; optimistic"
 },
 {
  "id": "A20",
  "model": "svm_nested",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std (scaled)",
  "decision": "C, gamma and cut-off chosen inside training",
  "validation": "nested cv",
  "f1_accepted": 0.914,
  "f1_std": 0.012,
  "precision_accepted": 0.9,
  "recall_accepted": 0.927,
  "accuracy": 0.868,
  "good_kept": 112.2,
  "bad_let_through": 12.4,
  "bad_caught": 27.6,
  "good_thrown_away": 8.8,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp3.py",
  "notes": "honest estimate for the SVM"
 },
 {
  "id": "A21",
  "model": "mlp",
  "process": "A",
  "family": "supervised (deep learning)",
  "trained_on": "accepted + rejected",
  "features": "22 median+std + log(batch length)",
  "decision": "P(rejected) > 0.5",
  "validation": "cv",
  "f1_accepted": 0.902,
  "f1_std": 0.015,
  "precision_accepted": 0.908,
  "recall_accepted": 0.896,
  "accuracy": 0.853,
  "good_kept": 108.4,
  "bad_let_through": 11.0,
  "bad_caught": 29.0,
  "good_thrown_away": 12.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/dl.py",
  "notes": "JAX, 32-16 hidden, dropout 0.3, 3-seed ensemble, one untuned config"
 },
 {
  "id": "A22",
  "model": "cnn_1d",
  "process": "A",
  "family": "supervised (deep learning)",
  "trained_on": "accepted + rejected",
  "features": "trajectories 22 ch x 128 time bins",
  "decision": "P(rejected) > 0.5",
  "validation": "cv",
  "f1_accepted": 0.89,
  "f1_std": 0.006,
  "precision_accepted": 0.877,
  "recall_accepted": 0.904,
  "accuracy": 0.832,
  "good_kept": 109.4,
  "bad_let_through": 15.4,
  "bad_caught": 24.6,
  "good_thrown_away": 11.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/dl.py",
  "notes": "2 conv layers, global avg+max pooling, augmentation"
 },
 {
  "id": "A23",
  "model": "cnn_1d_hybrid",
  "process": "A",
  "family": "supervised (deep learning)",
  "trained_on": "accepted + rejected",
  "features": "trajectories + 23 summary features",
  "decision": "P(rejected) > 0.5",
  "validation": "cv",
  "f1_accepted": 0.891,
  "f1_std": 0.01,
  "precision_accepted": 0.882,
  "recall_accepted": 0.901,
  "accuracy": 0.835,
  "good_kept": 109.0,
  "bad_let_through": 14.6,
  "bad_caught": 25.4,
  "good_thrown_away": 12.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/dl.py",
  "notes": ""
 },
 {
  "id": "A24",
  "model": "ensemble_fixed",
  "process": "A",
  "family": "ensemble (supervised + one-class)",
  "trained_on": "accepted + rejected (ocsvm member: accepted only)",
  "features": "22 median+std",
  "decision": "mean of standardised scores of svm + pca_knn(k5) + ocsvm > CV cut-off",
  "validation": "cv (settings chosen after seeing results)",
  "f1_accepted": 0.943,
  "f1_std": 0.007,
  "precision_accepted": 0.931,
  "recall_accepted": 0.955,
  "accuracy": 0.913,
  "good_kept": 115.6,
  "bad_let_through": 8.6,
  "bad_caught": 31.4,
  "good_thrown_away": 5.4,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "methods.py: ensemble",
  "notes": "BEST at fixed settings; optimistic"
 },
 {
  "id": "A25",
  "model": "ensemble_nested",
  "process": "A",
  "family": "ensemble (supervised + one-class)",
  "trained_on": "accepted + rejected",
  "features": "22 median+std",
  "decision": "SVM settings, ensemble members and cut-off chosen inside training",
  "validation": "nested cv",
  "f1_accepted": 0.93,
  "f1_std": 0.014,
  "precision_accepted": 0.919,
  "recall_accepted": 0.942,
  "accuracy": 0.894,
  "good_kept": 114.0,
  "bad_let_through": 10.0,
  "bad_caught": 30.0,
  "good_thrown_away": 7.0,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp5.py",
  "notes": "HONEST estimate of the best pipeline"
 },
 {
  "id": "A26",
  "model": "tbin_svm",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "220 time-bin medians (20 bins x 11 sensors), absolute deviation from accepted median",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.914,
  "f1_std": 0.004,
  "precision_accepted": 0.894,
  "recall_accepted": 0.936,
  "accuracy": 0.868,
  "good_kept": 113.2,
  "bad_let_through": 13.4,
  "bad_caught": 26.6,
  "good_thrown_away": 7.8,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": "time-localised features alone"
 },
 {
  "id": "A27",
  "model": "svm_base_plus_tbins",
  "process": "A",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "22 median+std + 220 time-bin medians",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.918,
  "f1_std": 0.004,
  "precision_accepted": 0.894,
  "recall_accepted": 0.944,
  "accuracy": 0.873,
  "good_kept": 114.2,
  "bad_let_through": 13.6,
  "bad_caught": 26.4,
  "good_thrown_away": 6.8,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": "extra features dilute the SVM"
 },
 {
  "id": "A28",
  "model": "hist_gradient_boosting",
  "process": "A",
  "family": "supervised (trees)",
  "trained_on": "accepted + rejected",
  "features": "22 median+std",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.858,
  "f1_std": 0.008,
  "precision_accepted": 0.78,
  "recall_accepted": 0.954,
  "accuracy": 0.763,
  "good_kept": 115.4,
  "bad_let_through": 32.6,
  "bad_caught": 7.4,
  "good_thrown_away": 5.6,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": "close to accept-all: too few rejected batches for boosting"
 },
 {
  "id": "A29",
  "model": "ensemble_plus_tbin_svm",
  "process": "A",
  "family": "ensemble",
  "trained_on": "accepted + rejected (ocsvm member: accepted only)",
  "features": "22 median+std; tbin member 220 time-bin",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.941,
  "f1_std": 0.007,
  "precision_accepted": 0.926,
  "recall_accepted": 0.957,
  "accuracy": 0.911,
  "good_kept": 115.8,
  "bad_let_through": 9.2,
  "bad_caught": 30.8,
  "good_thrown_away": 5.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": "no gain over A24"
 },
 {
  "id": "A30",
  "model": "ensemble_plus_hgb",
  "process": "A",
  "family": "ensemble",
  "trained_on": "accepted + rejected (ocsvm member: accepted only)",
  "features": "22 median+std",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.918,
  "f1_std": 0.009,
  "precision_accepted": 0.907,
  "recall_accepted": 0.931,
  "accuracy": 0.876,
  "good_kept": 112.6,
  "bad_let_through": 11.6,
  "bad_caught": 28.4,
  "good_thrown_away": 8.4,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": "boosting member hurts"
 },
 {
  "id": "A31",
  "model": "ensemble_plus_tbin_svm_plus_hgb",
  "process": "A",
  "family": "ensemble",
  "trained_on": "accepted + rejected (ocsvm member: accepted only)",
  "features": "22 median+std + 220 time-bin",
  "decision": "mean of standardised member scores > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.928,
  "f1_std": 0.012,
  "precision_accepted": 0.917,
  "recall_accepted": 0.94,
  "accuracy": 0.891,
  "good_kept": 113.8,
  "bad_let_through": 10.4,
  "bad_caught": 29.6,
  "good_thrown_away": 7.2,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/exp6.py",
  "notes": ""
 },
 {
  "id": "A32",
  "model": "ensemble_majority_vote",
  "process": "A",
  "family": "ensemble (vote)",
  "trained_on": "accepted + rejected (ocsvm member: accepted only)",
  "features": "22 median+std",
  "decision": "majority vote",
  "validation": "cv",
  "f1_accepted": 0.937,
  "f1_std": 0.003,
  "precision_accepted": 0.915,
  "recall_accepted": 0.96,
  "accuracy": 0.903,
  "good_kept": 116.2,
  "bad_let_through": 10.8,
  "bad_caught": 29.2,
  "good_thrown_away": 4.8,
  "n_accepted": 121,
  "n_rejected": 40,
  "code": "scratch/agree_test.py",
  "notes": "each member votes with its own cut-off; reject if >=2 of 3"
 },
 {
  "id": "B01",
  "model": "accept_all",
  "process": "B",
  "family": "baseline",
  "trained_on": "none",
  "features": "none",
  "decision": "always 'accepted'",
  "validation": "cv",
  "f1_accepted": 0.933,
  "f1_std": 0.0,
  "precision_accepted": 0.875,
  "recall_accepted": 1.0,
  "accuracy": 0.875,
  "good_kept": 35.0,
  "bad_let_through": 5.0,
  "bad_caught": 0.0,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: accept_all",
  "notes": ""
 },
 {
  "id": "B02",
  "model": "pca_mspc",
  "process": "B",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "393 level+shape",
  "decision": "max(T2/lim, SPE/lim) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 1.0,
  "f1_std": 0.0,
  "precision_accepted": 1.0,
  "recall_accepted": 1.0,
  "accuracy": 1.0,
  "good_kept": 35.0,
  "bad_let_through": 0.0,
  "bad_caught": 5.0,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: pca_mspc",
  "notes": "BEST for B; reason given: aeration_rate step"
 },
 {
  "id": "B03",
  "model": "pca_limits_full",
  "process": "B",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "393 level+shape",
  "decision": "as A06",
  "validation": "cv",
  "f1_accepted": 1.0,
  "f1_std": 0.0,
  "precision_accepted": 1.0,
  "recall_accepted": 1.0,
  "accuracy": 1.0,
  "good_kept": 35.0,
  "bad_let_through": 0.0,
  "bad_caught": 5.0,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: pca_limits_full",
  "notes": ""
 },
 {
  "id": "B04",
  "model": "ocsvm",
  "process": "B",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "32 median+std",
  "decision": "-decision > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.997,
  "f1_std": 0.006,
  "precision_accepted": 1.0,
  "recall_accepted": 0.994,
  "accuracy": 0.995,
  "good_kept": 34.8,
  "bad_let_through": 0.0,
  "bad_caught": 5.0,
  "good_thrown_away": 0.2,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: ocsvm",
  "notes": ""
 },
 {
  "id": "B05",
  "model": "logreg",
  "process": "B",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "393 level+shape",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.997,
  "f1_std": 0.006,
  "precision_accepted": 0.994,
  "recall_accepted": 1.0,
  "accuracy": 0.995,
  "good_kept": 35.0,
  "bad_let_through": 0.2,
  "bad_caught": 4.8,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: logreg",
  "notes": ""
 },
 {
  "id": "B06",
  "model": "pca_limits_simple",
  "process": "B",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "32 median+std",
  "decision": "as A06",
  "validation": "cv",
  "f1_accepted": 0.997,
  "f1_std": 0.006,
  "precision_accepted": 0.994,
  "recall_accepted": 1.0,
  "accuracy": 0.995,
  "good_kept": 35.0,
  "bad_let_through": 0.2,
  "bad_caught": 4.8,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: pca_limits_simple",
  "notes": ""
 },
 {
  "id": "B07",
  "model": "random_forest",
  "process": "B",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "393 level+shape",
  "decision": "P(rejected) > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.992,
  "f1_std": 0.007,
  "precision_accepted": 0.983,
  "recall_accepted": 1.0,
  "accuracy": 0.985,
  "good_kept": 35.0,
  "bad_let_through": 0.6,
  "bad_caught": 4.4,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: random_forest",
  "notes": ""
 },
 {
  "id": "B08",
  "model": "ensemble_fixed",
  "process": "B",
  "family": "ensemble",
  "trained_on": "accepted + rejected",
  "features": "32 median+std",
  "decision": "as A24",
  "validation": "cv",
  "f1_accepted": 0.975,
  "f1_std": 0.013,
  "precision_accepted": 0.961,
  "recall_accepted": 0.989,
  "accuracy": 0.955,
  "good_kept": 34.6,
  "bad_let_through": 1.4,
  "bad_caught": 3.6,
  "good_thrown_away": 0.4,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: ensemble",
  "notes": ""
 },
 {
  "id": "B09",
  "model": "svm",
  "process": "B",
  "family": "supervised",
  "trained_on": "accepted + rejected",
  "features": "32 median+std",
  "decision": "decision > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.934,
  "f1_std": 0.035,
  "precision_accepted": 0.927,
  "recall_accepted": 0.943,
  "accuracy": 0.885,
  "good_kept": 33.0,
  "bad_let_through": 2.6,
  "bad_caught": 2.4,
  "good_thrown_away": 2.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: svm",
  "notes": "5 rejected batches too few for a boundary"
 },
 {
  "id": "B10",
  "model": "pca_mspc_trained_on_all",
  "process": "B",
  "family": "one-class (contaminated reference)",
  "trained_on": "accepted + rejected, labels ignored",
  "features": "393 level+shape",
  "decision": "as B02",
  "validation": "cv",
  "f1_accepted": 0.933,
  "f1_std": 0.0,
  "precision_accepted": 0.875,
  "recall_accepted": 1.0,
  "accuracy": 0.875,
  "good_kept": 35.0,
  "bad_let_through": 5.0,
  "bad_caught": 0.0,
  "good_thrown_away": 0.0,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "scratch/train_on_all.py",
  "notes": "fault absorbed into 'normal'"
 },
 {
  "id": "B11",
  "model": "feature_limits",
  "process": "B",
  "family": "one-class",
  "trained_on": "accepted only",
  "features": "393 level+shape",
  "decision": "max robust |z| > CV cut-off",
  "validation": "cv",
  "f1_accepted": 0.91,
  "f1_std": 0.007,
  "precision_accepted": 0.929,
  "recall_accepted": 0.891,
  "accuracy": 0.845,
  "good_kept": 31.2,
  "bad_let_through": 2.4,
  "bad_caught": 2.6,
  "good_thrown_away": 3.8,
  "n_accepted": 35,
  "n_rejected": 5,
  "code": "methods.py: feature_limits",
  "notes": ""
 }
]
```
