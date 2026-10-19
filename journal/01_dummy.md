# 01_dummy

## Question / hypothesis

Does always predicting the training mean give us a useful RMSE floor that all real models must beat?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Lab guide Day 1 § 3: "build the laziest possible model before touching a real one"
- **Why this matters:** The dummy mean RMSE is the reference point. A model that cannot beat it has a bug upstream — bad data load, misaligned rows, or a broken metric — not a modelling problem. It also makes later RMSE numbers interpretable ("beat the dummy by X points").

## Method

- **Files touched:** `src/parkinson/hub.py`, `experiments/01_dummy.py`
- **Change versus baseline:** First experiment; no previous experiment to compare against.
- **Estimator:** `sklearn.dummy.DummyRegressor(strategy="mean")` — always predicts the mean `target` from the training visits.
- **Features:** numeric columns only (`sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, `off`). Categorical and identifier columns (`patient_id`, `cohort`, `gene`, `rater_id`) are excluded — the dummy ignores all features anyway; this is just the shape `skore.evaluate` receives.
- **Cross-validation:** `splitter=0.2` (random 20 % row holdout). This is intentional for Day 1 — the same patient can appear on both sides, which is a data leak. That leak is *the point*: Day 2 fixes it with `GroupKFold`. The dummy is too weak to exploit any leak; this is acceptable for a floor estimate.
- **Hub push:** `project.put("01_dummy", report)` to the `bobathon-esilv` project on the hub.
- **Out of scope:** imputation, feature engineering, grouped CV, submission file (no Kaggle upload for the dummy).

## Risks / things that could invalidate the result

- The random 20 % holdout leaks patient data across train/validation — the dummy can't exploit this, so the floor is valid here, but a real model trained this way would look better than it truly is.
- If `target` is not zero-centered, the mean predictor's RMSE ≈ std(target). Confirming this gives us a sanity-check on the metric.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 16.48 (random 20 % row holdout)
- **Implication for next iteration:** This RMSE ≈ std(target) = 16.5, confirming the sanity check. Any real model must beat 16.48. Next: try Ridge regression on the same holdout to see whether linear features help.
