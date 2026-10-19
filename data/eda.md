# EDA: Parkinson's Disease MDS-UPDRS Progression

Interactive reports:
- [data/eda_train.html](eda_train.html) — train split (features + target)
- [data/eda_test.html](eda_test.html) — test split (features only)

---

## Dataset at a glance

| | Train | Test |
|---|---|---|
| Rows | 44 590 | 11 013 |
| Feature columns | 12 | 12 |
| Target column | `target` (continuous) | — |

Each row is one clinical visit for one patient. The dataset tracks
Parkinson's disease progression across repeated visits, using 5 576
unique patients in the train split.

**Features:**

| Column | Type | Null % | # Unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0 % | 5 576 | Repeated across visits — strong group signal |
| `cohort` | string | 0 % | 2 | Binary: A / B |
| `sexM` | int | 0 % | 2 | 0 = female, 1 = male |
| `gene` | string | 32.4 % | 4 | Genetic mutation; substantial missingness |
| `age_at_diagnosis` | float | 5.2 % | 563 | Stable per patient |
| `age` | float | 0 % | 1 150 | Age at visit — varies across visits |
| `ledd` | float | 36.6 % | 1 320 | Levodopa equivalent daily dose; heavy missingness |
| `time_since_intake_on` | float | 46.4 % | 64 | Minutes since last dose (ON state); ~half missing |
| `time_since_intake_off` | float | 78.8 % | 178 | Minutes since last dose (OFF state); very sparse |
| `rater_id` | string | 0 % | 50 | Clinician who performed the rating |
| `on` | float | 29.6 % | 83 | MDS-UPDRS part III score — ON state |
| `off` | float | 42.4 % | 101 | MDS-UPDRS part III score — OFF state |

---

## Target

`target` is a continuous MDS-UPDRS motor score (0–109.5 in this set).

| Statistic | Value |
|---|---|
| Mean | 37.47 |
| Median (p50) | 37.30 |
| Std | 16.50 |
| IQR (p25–p75) | 25.6 – 49.3 |
| Min | 0.0 |
| Max | 109.5 |
| # Unique values | 867 |

Distribution is roughly bell-shaped and centred near 37, with a light
right tail (scores above ~65 are rare). No missing values in the
target. The spread is moderate (IQR ≈ 24 points), so MAE / RMSE are
interpretable in the same units as the score. No extreme skew that
would mandate a log transform.

---

## Per-column findings

**High-missingness columns (action required in the pipeline):**
- `time_since_intake_off` — 78.8 % missing. Almost exclusively absent
  for ON-state visits. Needs imputation or a binary indicator flag.
- `time_since_intake_on` — 46.4 % missing. Similar pattern.
- `ledd` — 36.6 % missing. Drug-dose data not collected at all visits.
- `off` — 42.4 % missing; `on` — 29.6 % missing. Both are prior
  MDS-UPDRS sub-scores that are the strongest predictors (see
  Associations). Their missingness patterns are complementary: many
  visits have only ON or only OFF sub-scores, not both.
- `gene` — 32.4 % missing.

**Cardinality notes:**
- `patient_id` has 5 576 unique values with a unique ratio of 0.125,
  confirming ~8 visits per patient on average — a strong group
  structure that must be respected in cross-validation.
- `rater_id` has 50 unique values; rater effects may introduce
  systematic bias in the target.

---

## Structure signals

**No datetime columns detected** — the temporal ordering of visits is
encoded implicitly through `age` (patient age at visit) and the
`patient_id` repeated-measures structure, not via an explicit date
column.

**Group structure — `patient_id`:** The same patient appears across
multiple rows. A standard `KFold` or `StratifiedKFold` will leak
patient-level information across folds (training on earlier visits
of a patient while validating on later ones). **`GroupKFold` is the
appropriate splitter**, using `patient_id` as the group key.

---

## Associations

Top associations with `target`:

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | 0.669 | 0.368 |
| `ledd` | 0.298 | 0.236 |
| `age` | 0.310 | 0.114 |
| `age_at_diagnosis` | 0.133 | 0.051 |

The `off` and `on` sub-scores are **very strongly correlated with the
target** (Pearson 0.87 and 0.67 respectively). This is expected —
`off` and `on` are prior administrations of the same MDS-UPDRS
assessment. Their high predictive value is legitimate (not leakage),
since the target is a *future* assessment for the same patient.
However, their high missingness (~30–42 %) means a large fraction of
the training set cannot use these features directly; imputation
strategy matters.

**`on` ↔ `off` inter-correlation:** Pearson 0.872, Cramér V 0.325.
The two sub-scores are themselves correlated, so they carry
overlapping information.

**`age_at_diagnosis` ↔ `age`:** Pearson 0.942, Cramér V 0.562 —
near-perfect linear relationship (age at visit is age at diagnosis +
time elapsed). Including both may not add signal and one could be
dropped or the difference computed as a "disease duration" feature.

---

## Modelling implications

1. **CV splitter → `GroupKFold`** with `groups=patient_id`.
   Standard folds would leak patient data across train/validation,
   giving over-optimistic scores. This is the main structural
   constraint.

2. **Metric → RMSE or MAE** (same units as the score). No strong
   skew, so no transformation needed. If the competition metric is
   specified separately, use that; otherwise MAE is more interpretable
   for a clinical motor scale.

3. **Imputation is non-trivial.** Three columns exceed 40 % missing:
   `time_since_intake_off`, `off`, and `time_since_intake_on`. The
   missingness is likely MNAR (depends on visit type). A binary
   indicator alongside an imputed value is safer than imputation alone.

4. **`on` / `off` are the dominant predictors.** A strong baseline
   can be built from just those two columns + `ledd` + `age`.
   A `HistGradientBoostingRegressor` (native missing-value support) or
   `tabular_pipeline` with an imputer will handle them naturally.

5. **`rater_id` may introduce bias.** With 50 raters, target-encoding
   or embedding is reasonable; including it as a feature could help
   or hurt depending on whether the validation set includes raters
   seen in training.

6. **`age_at_diagnosis` + `age` redundancy.** Consider engineering a
   `disease_duration = age - age_at_diagnosis` feature and dropping
   one of the originals to avoid near-collinearity.

---

## Open questions

- **Competition or evaluation metric?** `sample_submission.csv`
  suggests a submission format but does not specify the metric.
  Clarification needed before selecting the loss.
- **What does `cohort` encode?** Two values (A / B) — is this a
  study cohort, a data-collection site, or a patient subgroup?
  Matters for whether cohort-stratified analysis is appropriate.
- **Missing `gene` values** — are they truly unknown (not tested) or
  an encoding artefact? If the former, `None` / `unknown` is a valid
  category; if the latter, imputation from family history or other
  features may be possible.
- **`rater_id` in test set** — do the test rows include rater IDs
  from the training set, or are new raters present? Determines
  whether target-encoding `rater_id` is safe.
