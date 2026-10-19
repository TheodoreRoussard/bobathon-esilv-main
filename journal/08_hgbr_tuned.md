# 08_hgbr_tuned

## Question / hypothesis

Does tuning HGBR's learning_rate, max_iter, and max_leaf_nodes with HalvingGridSearchCV lower RMSE below the best feature-engineered model?

## Motivation

- **Sourcing strategy:** B4 (promoted from Backlog)
- **Source(s):**
  - B4: `HistGradientBoostingRegressor` defaults (100 iterations, learning_rate=0.1, max_leaf_nodes=31) are sensible starting points but not necessarily optimal for this dataset size (44k rows, ~12 features after engineering).
- **Why this matters:** A slower learning rate with more trees often generalises better. `HalvingGridSearchCV` eliminates weak candidates early, making a grid search feasible within the grouped-CV budget.

## Method

- **Files touched:** `experiments/08_hgbr_tuned.py`, `submissions/08_hgbr_tuned.csv`
- **Change versus 07_missing_indicators:** Replace `tabular_pipeline("regressor")` with a `make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())` inside a `HalvingGridSearchCV`. Grid: `learning_rate ∈ {0.05, 0.1}`, `max_iter ∈ {200, 500}`, `max_leaf_nodes ∈ {15, 31, 63}`. Full feature set from 07 (including missing indicators and engineered features).
- **Cross-validation:** GroupKFold(5) on `patient_id` for both the inner search and the outer evaluation.
- **Hub push:** `project.put("08_hgbr_tuned", report)`.
- **Kaggle file:** Best estimator fit on all training visits → `submissions/08_hgbr_tuned.csv`.

## Risks / things that could invalidate the result

- `HalvingGridSearchCV` uses a random row holdout internally by default — not GroupKFold. This means the tuning step may pick parameters that exploit patient leakage, slightly over-optimistic. Acceptable for an exploratory run; a production setup would use GroupKFold in the inner loop too.
- 12 grid points × 5 folds is manageable, but with `max_iter=500` it may take several minutes.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.44 ± 0.16 (GroupKFold 5-fold); best params: lr=0.05, max_iter=200, max_leaf_nodes=31. No meaningful gain over default HGBR.
- **Implication for next iteration:** Default HGBR settings are already near-optimal for this feature set and data size. Further tuning is unlikely to move the needle; focus on feature engineering or structural model changes instead.
