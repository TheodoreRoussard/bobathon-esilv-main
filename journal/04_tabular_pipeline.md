# 04_tabular_pipeline

## Question / hypothesis

Does including the categorical features (`gene`, `cohort`, `rater_id`) via skrub's `tabular_pipeline` lower the RMSE beyond the fold-to-fold spread seen with numeric-only HGBR?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Lab guide Day 2 § 5: "Ridge and numeric HGBR never saw `gene` or `cohort`; tabular_pipeline turns text into numbers automatically"
  - EDA finding: `cohort` has Cramér V 0.09 and `gene` 0.05 with target — small but non-zero signal
- **Why this matters:** `gene` (LRRK2+, GBA, etc.) encodes disease subtype that clinically affects progression rate and levodopa response. `rater_id` captures systematic scoring differences across 50 clinicians. Discarding these was a deliberate limitation of 03_hgbr; this experiment measures their value.

## Method

- **Files touched:** `experiments/04_tabular_pipeline.py`
- **Change versus 03_hgbr:** Replace the numeric-only `X` with `X_full = visits.drop(columns=["patient_id", "target"])`. Use `skrub.tabular_pipeline("regressor")` which wraps `TableVectorizer` + `HistGradientBoostingRegressor`. `TableVectorizer` auto-picks one-hot for low-cardinality columns and `StringEncoder` for higher-cardinality text.
- **Cross-validation:** same `cv_splits` (GroupKFold 5-fold on patient_id) as 03_hgbr for a fair comparison.
- **Hub push:** `project.put("04_tabular_pipeline", report)` to `bobathon-esilv`.
- **Out of scope:** DataOps graph, Kaggle submission file — those are in 05_dataops.

## Risks / things that could invalidate the result

- `rater_id` has 50 unique values. If some raters appear only in train, their encoding is not meaningful on the test set — though with 50 raters and ~8 visits each this is unlikely to cause complete cold-start.
- `gene` is 32 % missing; `TableVectorizer` will encode the NaN as a distinct category or pass it to HGBR's native handler depending on dtype. Worth checking the fitted transformer.
- Any gain here is additive to 03_hgbr's improvement; the two changes (NaN handling + categoricals) are not independent.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.52 ± 0.14 (GroupKFold 5-fold) vs numeric-only HGBR 7.76 ± 0.15 — marginal improvement from categoricals
- **Implication for next iteration:** gene, cohort, rater_id add small signal (~3 % RMSE drop). DataOps + SkrubLearner is the natural next packaging step for baking the split into the graph.
