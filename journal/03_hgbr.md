# 03_hgbr

## Question / hypothesis

Does HistGradientBoostingRegressor, evaluated with patient-grouped CV, reveal that Ridge's row-holdout scores were too optimistic — and does treating missingness natively lower the RMSE further?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Lab guide Day 2 § 3–4: "GroupKFold is a dress rehearsal of the Kaggle split"; "missingness is part of the generative process — do not impute it away"
- **Why this matters:** Two changes at once: honest evaluation (no patient leak) and a model that can use NaN as a signal. This is the first experiment whose score is directly comparable to Kaggle's public leaderboard.

## Method

- **Files touched:** `experiments/03_hgbr.py`
- **Change versus 02_ridge:**
  1. **Splitter:** `GroupKFold(n_splits=5)` on `patient_id` — whole patients go to one fold, never both sides.
  2. **Model:** `HistGradientBoostingRegressor(random_state=0)` — no imputation; NaNs are routed at each split natively.
  3. **Comparison:** dummy, ridge (same `cv_splits`), and hgbr in a single `ComparisonReport`.
- **Hub push:** `project.put("03_hgbr", report)` to the `bobathon-esilv` project.
- **Out of scope:** categorical features (`gene`, `cohort`, `rater_id`), DataOps graph, Kaggle submission file — all in later experiments.

## Risks / things that could invalidate the result

- Ridge on `cv_splits` will likely look worse than on the row holdout from 02_ridge; that is expected and is the point of this experiment.
- HGBR has more hyperparameters (learning rate, max_iter, max_depth). Default settings may not be optimal; but they are a fair first pass for comparison.
- The numeric-only feature set still discards `gene`, `cohort`, and `rater_id`. Any improvement here is a lower bound on what a full-feature model can do.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.76 ± 0.15 (GroupKFold 5-fold) vs ridge 10.60 ± 0.27 on same split — 27 % improvement
- **Implication for next iteration:** Native NaN handling in HGBR is clearly worth it. Adding categorical features (gene, cohort, rater_id) via skrub tabular_pipeline is the natural next step.
