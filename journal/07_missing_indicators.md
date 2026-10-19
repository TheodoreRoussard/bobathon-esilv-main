# 07_missing_indicators

## Question / hypothesis

Do explicit binary missing-value flags lower RMSE on top of the engineered features from 06_feature_engineering?

## Motivation

- **Sourcing strategy:** B2 (promoted from Backlog)
- **Source(s):**
  - B2: HGBR routes NaN values to a learned branch at each split, but that signal is implicit. Explicit binary flags (`col_missing = col.isna().astype(int)`) make the pattern visible to any downstream model and may give HGBR a more direct route to exploit the missingness structure.
  - EDA: `time_since_intake_off` (78.8 % missing), `off` (42.4 %), `time_since_intake_on` (46.4 %), `ledd` (36.6 %), `on` (29.6 %), `gene` (32.4 %) — all have clinically meaningful absence patterns.
- **Why this matters:** A missing OFF score often means the visit was ON-only (uncomfortable OFF exams are skipped — CONTEXT.md). That absence is signal about visit type and patient state, not random noise.

## Method

- **Files touched:** `experiments/07_missing_indicators.py`
- **Change versus 06_feature_engineering:** Add six binary indicator columns before fitting: `on_missing`, `off_missing`, `ledd_missing`, `gene_missing`, `time_since_intake_on_missing`, `time_since_intake_off_missing`. All other features from 06 kept. Model and CV unchanged.
- **Cross-validation:** GroupKFold(5) on `patient_id`.
- **Hub push:** `project.put("07_missing_indicators", report)`.
- **Out of scope:** Kaggle submission — defer to 08/09 which stack on top.

## Risks / things that could invalidate the result

- HGBR already exploits NaN routing internally; explicit flags may be redundant and add noise without gain.
- `gene` is a string column — `gene_missing` derived from it is straightforward, but `TableVectorizer` will also see the NaN in the original column and may encode it as a separate category. There could be slight overlap in the signal.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.45 ± 0.15 (GroupKFold 5-fold) — flat vs 06_feature_engineering (7.44). Explicit flags add no signal on top of HGBR's native NaN routing.
- **Implication for next iteration:** B2 is resolved: explicit missing indicators are redundant for HGBR. Skip for future tree-based experiments; revisit only if switching to a model that cannot handle NaN natively.
