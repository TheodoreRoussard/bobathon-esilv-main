# 06_feature_engineering

## Question / hypothesis

Do `disease_duration` (B1) and a pharmacokinetic levodopa-concentration proxy (B3) lower the GroupKFold RMSE below `04_tabular_pipeline`'s 7.52 ± 0.14?

## Motivation

- **Sourcing strategy:** B1 + B3 (promoted from Backlog)
- **Source(s):**
  - B1: EDA finding — `age` and `age_at_diagnosis` have Pearson r = 0.94; `disease_duration = age − age_at_diagnosis` encodes the clinically meaningful disease stage without redundancy.
  - B3: CONTEXT.md — "levodopa effect is proportional to blood concentration: a fast absorption phase, then a decline toward zero"; standard PK model uses exponential decay with half-life ≈ 90 min.
- **Why this matters:** Both features address the generative process described in CONTEXT.md directly. The current model sees raw timing delays and doses as opaque numbers; these features give it the *physical interpretation*.

## Method

- **Files touched:** `experiments/06_feature_engineering.py`, `submissions/06_feature_engineering.csv`
- **New features added on top of 04_tabular_pipeline:**
  - `disease_duration = age − age_at_diagnosis` (B1)
  - `ledd_conc_on = ledd × exp(−ln(2)/90 × time_since_intake_on)` — estimated levodopa blood level at ON-state assessment (B3)
  - `ledd_conc_off = ledd × exp(−ln(2)/90 × time_since_intake_off)` — same for OFF-state (B3)
- **Model:** `tabular_pipeline("regressor")` (TableVectorizer + HistGradientBoostingRegressor) — identical to 04_tabular_pipeline.
- **Features:** all columns from 04_tabular_pipeline plus the three new ones above; `patient_id` excluded.
- **Cross-validation:** GroupKFold(5) on `patient_id` — same cv_splits as previous experiments.
- **Hub push:** `project.put("06_feature_engineering", report)`.
- **Kaggle file:** fit on all training visits → `submissions/06_feature_engineering.csv`.

## Risks / things that could invalidate the result

- The 90-minute half-life is a population average; patient-level variation and formulation type can shift it substantially. The proxy will carry useful signal in aggregate but noisy for individual visits.
- `ledd_conc_on/off` will be NaN wherever `ledd` or `time_since_intake_*` is NaN (~46–79 % of rows). HGBR handles NaNs natively so this is not a blocker, but the proxy is only informative for the minority of visits with complete timing data.
- `disease_duration` inherits the 5 % missingness of `age_at_diagnosis`.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.44 ± 0.15 (GroupKFold 5-fold) — beats 04_tabular_pipeline by 0.08 RMSE points (~1 %)
- **Implication for next iteration:** Both features add signal. Next logical step: lag features from prior visits (B5) to address temporal progression, or HGBR hyperparameter tuning (B4).
