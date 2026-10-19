# 10_lgbm_patient_feats

## Question / hypothesis

Do rich per-patient aggregate features (cross-visit statistics, linear score trajectories, rater-bias correction, PK-corrected scores) combined with LightGBM push the Kaggle RMSE well below the current ~7.44?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request 2025-07-04: "chaque patient a 4 à 12 visites dans X_train comme dans X_test — il faut exploiter les AUTRES visites du même patient."
  - Key insight: `on`/`off` are noisy biased measurements; the `target` (true OFF) evolves smoothly per patient. Cross-visit aggregates derived purely from X columns (never from target) provide a much richer signal than single-visit features.
  - 09_lag_features showed prev_target gives CV RMSE 2.62 but is NaN on all test patients — the right approach is X-column-only cross-visit features.
- **Why this matters:** The Kaggle holdout is by patient but test patients have ALL their visits in X_test. Per-patient aggregates (mean, std, trajectory slope) computed across those visits give the model the same patient context without leaking labels.

## Method

### Step 1 — `build_features(df)` applied to train and test separately (no leakage)

Added features (all derived from X columns only, grouped by `patient_id`):
- `tsd = age - age_at_diagnosis`; `tsd_patient_mean = patient-mean(age_at_diagnosis)` for missing-robust version
- Per-patient: `n_visits`, `age_min`, `age_max`, `t_rel = age - age_min`, visit rank
- For `off`, `on`, `ledd`: patient mean, median, std, n_nonmissing; deviation of this visit from patient mean
- For `off` and `on`: per-patient linear regression `value ~ age` (≥2 points); slope, intercept, fitted value at this visit's age (fallback: patient mean)
- For `off` and `on`: previous known value and next known value within patient (sorted by age)
- `off_isna`, `on_isna` binary flags
- `Index` and `patient_id` excluded from features

### Step 2 — In-fold features (computed on train partition only each fold)

Implemented via a custom sklearn `FunctionTransformer` / cross-val loop to avoid target leakage:
- **Rater bias**: per `rater_id`, mean of `(off − target)` and `(on − target)` on train; mapped to all rows; `off_corr = off − bias_off`; patient mean of `off_corr`
- **PK-corrected ON**: `time_since_intake_on` binned `[0, 0.5, 1, 1.5, 2, 3, 4, 6, 30]`; median `on/target` ratio per bin on train; `on_corr = on / ratio`; patient mean of `on_corr`
- `best = off_corr if available else on_corr`; per-patient linear fit of `best ~ age`, evaluated at visit age (`best_lin`)

### Step 3 — Model

`LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=63, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7)` — compared against `06_feature_engineering` baseline on the same `GroupKFold(5)` splits.

### Step 4 — If gain confirmed (RMSE drop > fold std)

- Light tuning of `num_leaves`, `min_child_samples`, `learning_rate`
- Multi-seed average (3–5 seeds)
- Blend LightGBM + HistGradientBoosting
- Level-2: out-of-fold patient-mean predictions re-injected as feature

- **Files touched:** `src/parkinson/features.py`, `experiments/10_lgbm_patient_feats.py`, `submissions/10_lgbm_patient_feats.csv`
- **Hub push:** `project.put("10_lgbm_patient_feats", report)` to `bobathon-esilv`.
- **Cross-validation:** GroupKFold(5) on `patient_id` throughout.

## Risks / things that could invalidate the result

- In-fold features (rater bias, PK ratio) must be computed on train partition only; if they accidentally use the validation rows the CV score is optimistic.
- Per-patient aggregates computed on the full dataset (not per fold) are X-only so they carry no label leakage, but they do carry test-set information into the train features — acceptable since we apply `build_features` separately to train and test.
- LightGBM's early stopping requires a validation set; here we fix `n_estimators=2000` with a low `learning_rate` rather than early stopping to avoid leaking the val fold.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** HGB 4.45 ± 0.05 / LGBM 4.25 ± 0.06 (GroupKFold 5-fold) — LGBM selected; submission → submissions/10_lgbm_patient_feats.csv; hub https://skore.probabl.ai/hack_a_bbl/bobathon-esilv/cross-validations/87907
- **Implication for next iteration:** Per-patient cross-visit aggregates are the dominant signal (7.44 → 4.25, −43 %). Blend HGB+LGBM gave no gain; LGBM alone is best. Next: add in-fold rater-bias / PK correction, or lag prev_on/prev_off only (B8).
