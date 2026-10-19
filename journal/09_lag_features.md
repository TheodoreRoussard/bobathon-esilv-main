# 09_lag_features

## Question / hypothesis

Do per-patient lagged scores (previous-visit target, on, off) capture the temporal progression signal and lower RMSE significantly?

## Motivation

- **Sourcing strategy:** B5 (promoted from Backlog)
- **Source(s):**
  - B5: CONTEXT.md — "temporal progression: capture how motor scores evolve over visits for each patient." All prior models treat visits independently; the best predictor of next-visit true OFF is prior-visit true OFF.
  - EDA: `patient_id` repeats ~8 times per patient, sorted by `age`. The target is a continuous progression — previous scores are the strongest possible feature.
- **Why this matters:** The Kaggle holdout is by patient, not by visit. A model that uses within-patient trajectory is exploiting exactly the structure the competition rewards.

## Method

- **Files touched:** `experiments/09_lag_features.py`, `submissions/09_lag_features.csv`
- **Change versus 06_feature_engineering:** Sort visits by `(patient_id, age)` and compute `prev_target`, `prev_on`, `prev_off` via `groupby("patient_id").shift(1)`. First visit per patient gets NaN (no prior visit) — HGBR handles that natively.
- **Leakage guard:** `shift(1)` only looks backward within the sorted trajectory — it never sees the current or future rows. GroupKFold then holds out entire patients, so a patient's lag values come only from their own past visits, not from the validation patients.
- **Feature set:** all features from 06_feature_engineering plus `prev_target`, `prev_on`, `prev_off`. Missing indicators (B2) not included here to keep changes isolated; the two can be combined later.
- **Cross-validation:** GroupKFold(5) on `patient_id`.
- **Hub push:** `project.put("09_lag_features", report)`.
- **Kaggle file:** For test visits, the lag comes from their last known training visit — a join on `patient_id` sorted by age. Test-set first visits for new patients get NaN.

## Risks / things that could invalidate the result

- `prev_target` is the debiased true OFF from training labels — it is not available at inference time on new patients (test patients never appeared in train). For test rows, `prev_target` will be NaN everywhere (no training target known for those patients). The model must learn to function without it via the NaN path.
- If the model heavily relies on `prev_target` for train CV but it's NaN for all test rows, the Kaggle score may be worse than the CV RMSE suggests. This is the key risk to watch.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 2.62 ± 0.08 (GroupKFold 5-fold) — down from 7.44, a 65 % improvement. prev_target is by far the strongest feature in CV.
- **Kaggle public score:** worse than 06_feature_engineering — B7 risk confirmed. `prev_target` is a training label; it is NaN for every test patient (holdout is by patient), so the model falls back to a poor path at inference time. The CV gain was illusory.
- **Implication for next iteration:** Exclude `prev_target` from any future lag experiment. `prev_on`/`prev_off` are real clinical measurements available at test time (from the last training visit for each patient) and are safe to use.
