# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease MDS-UPDRS motor-score progression (regression)
- **Goal:** Minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training.
- **Last experiment:** 10_lgbm_patient_feats - done
- **Last result:** RMSE 4.25 ± 0.06 (GroupKFold 5-fold, LGBM + build_features)
- **Best model so far:** `10_lgbm_patient_feats` — RMSE 4.25 ± 0.06 (GroupKFold 5-fold); Kaggle file `submissions/10_lgbm_patient_feats.csv`; hub https://skore.probabl.ai/hack_a_bbl/bobathon-esilv/cross-validations/87907

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-04
  - env manager: pip+venv - recorded: 2025-07-04
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: none - recorded: 2025-07-04
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: hub - recorded: 2025-07-04
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: n/a - recorded: 2025-07-04
  - student prior: some-sklearn - recorded: 2025-07-04
  - CV splitter family: GroupKFold - recorded: 2025-07-04

## Data understanding (EDA)

<!--
Short index entry — the full analysis lives in data/eda.md.
-->

- **Status:** done - 2025-07-04
- **Summary:** 44 590 train rows / 11 013 test rows, 12 features, continuous target (MDS-UPDRS motor score, mean 37.5 ± 16.5).
  5 576 unique patients with ~8 visits each — `GroupKFold` on `patient_id` is required.
  `off` (Pearson r = 0.87) and `on` (r = 0.67) are dominant predictors but ~30–42 % missing;
  `ledd` and `age` are secondary. No datetime columns; no extreme target skew.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor mean — RMSE floor | done | RMSE 16.48 (random 20 % row holdout) | [design note](01_dummy.md) |
| `02_ridge` | Ridge + median imputation, row holdout, Kaggle submission | done | RMSE 10.49 (random 20 % row holdout) | [design note](02_ridge.md) |
| `03_hgbr` | HistGradientBoosting + GroupKFold, missingness as signal | done | RMSE 7.76 ± 0.15 (GroupKFold 5-fold) | [design note](03_hgbr.md) |
| `04_tabular_pipeline` | skrub tabular_pipeline + categoricals (gene, cohort, rater_id) | done | RMSE 7.52 ± 0.14 (GroupKFold 5-fold) | [design note](04_tabular_pipeline.md) |
| `05_dataops` | skrub DataOps graph, groups baked in + Kaggle submission | done | RMSE 7.65 ± 0.16 (GroupKFold 5-fold) | [design note](05_dataops.md) |
| `06_feature_engineering` | disease_duration (B1) + PK levodopa proxy (B3) | done | RMSE 7.44 ± 0.15 (GroupKFold 5-fold) | [design note](06_feature_engineering.md) |
| `07_missing_indicators` | explicit binary missingness flags (B2) | done | RMSE 7.45 ± 0.15 (GroupKFold 5-fold) | [design note](07_missing_indicators.md) |
| `08_hgbr_tuned` | HalvingGridSearchCV over HGBR hyperparameters (B4) | done | RMSE 7.44 ± 0.16 (GroupKFold 5-fold) | [design note](08_hgbr_tuned.md) |
| `09_lag_features` | per-patient lagged target/on/off from prior visit (B5) | done | RMSE 2.62 ± 0.08 CV / Kaggle worse than 06 (prev_target leaks) | [design note](09_lag_features.md) |
| `10_lgbm_patient_feats` | LightGBM + rich per-patient cross-visit aggregates | done | RMSE 4.25 ± 0.06 (GroupKFold 5-fold, LGBM) | [design note](10_lgbm_patient_feats.md) |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Engineer `disease_duration`~~ - resolved in 06_feature_engineering | ~~`my-pick:04_tabular_pipeline`~~ |
| ~~B2~~ | ~~Missing-value indicator columns~~ - resolved in 07_missing_indicators (no gain) | ~~`my-pick:04_tabular_pipeline`~~ |
| ~~B3~~ | ~~PK levodopa concentration proxy~~ - resolved in 06_feature_engineering | ~~`my-pick:04_tabular_pipeline`~~ |
| ~~B4~~ | ~~Tune HGBR hyperparameters~~ - resolved in 08_hgbr_tuned (no gain over defaults) | ~~`my-pick:04_tabular_pipeline`~~ |
| ~~B5~~ | ~~Lag features from prior visits~~ - resolved in 09_lag_features (RMSE 2.62 ± 0.08) | ~~`my-pick:06_feature_engineering`~~ |
| ~~B6~~ | ~~Combine lag features with feature engineering~~ - superseded by B7 outcome | ~~`my-pick:09_lag_features`~~ |
| ~~B7~~ | ~~Investigate prev_target leakage~~ - confirmed: Kaggle score worse than 06_feature_engineering | ~~`my-pick:09_lag_features`~~ |
| B8 | Lag experiment without prev_target — use only prev_on and prev_off (real clinical measurements available at test time via last-training-visit join) | `my-pick:09_lag_features` |
