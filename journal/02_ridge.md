# 02_ridge

## Question / hypothesis

Does a Ridge regressor with median imputation beat the dummy mean, and how much of the error does linear signal in the numeric features explain?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Lab guide Day 2 § 1–2: "numeric clinical variables may carry a simple additive signal; Ridge is the first check"
- **Why this matters:** The dummy proved our floor (RMSE 16.48). Ridge tells us whether the numbers (`age`, `on`, `off`, `ledd`, intake delays) carry any linear predictive signal at all. It is also the experiment that gets uploaded to Kaggle, so the hub URL must appear in the submission description.

## Method

- **Files touched:** `experiments/02_ridge.py`, `submissions/02_ridge.csv`
- **Change versus 01_dummy:** Replace `DummyRegressor` with `make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))`. Same feature columns as the dummy (numeric only; identifiers and strings excluded).
- **Cross-validation:** `splitter=0.2` (random 20 % row holdout), same as Day 1. The patient-data leak is known and accepted here; it is fixed in 03_hgbr. The dummy is evaluated on the same split for a head-to-head comparison.
- **Hub push:** `project.put("02_ridge", report)` to the `bobathon-esilv` project.
- **Kaggle file:** `final = clone(ridge).fit(X, y)` on all training visits → predict `X_test[feature_cols]` → write `submissions/02_ridge.csv` with header `Index,target`.
- **Out of scope:** alpha tuning, categorical features, grouped CV — all deferred to later experiments.

## Risks / things that could invalidate the result

- The random row holdout leaks patients across train/validation. Ridge may look better than it truly is; the real score will show up in 03_hgbr with GroupKFold.
- `on` and `off` are ~30–42 % missing; the median imputer fills them with a single constant, which may suppress their true signal. The RMSE improvement here is therefore a lower bound on what honest imputation or native-NaN models could achieve.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 10.49 (random 20 % row holdout) vs dummy 16.48 — 36 % improvement
- **Implication for next iteration:** Linear signal is real but the row holdout leaks patients. GroupKFold will show the true score; HGBR should benefit from keeping NaNs as signal.
