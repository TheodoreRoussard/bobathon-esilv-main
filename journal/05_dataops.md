# 05_dataops

## Question / hypothesis

Does baking GroupKFold and groups directly into the skrub DataOp graph produce the same (or better) RMSE as 04_tabular_pipeline while eliminating the risk of the split drifting from the data?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Lab guide Day 2 § 6: "GroupKFold only works if `groups` is the patient_id of the *same* rows as X — putting that on the graph is how you stop the leakage from creeping back in"
  - Lab guide Day 2 § 7: fit on all training visits → predict X_test → Kaggle CSV
- **Why this matters:** In 04_tabular_pipeline we pass `cv_splits` as a separate list — easy to accidentally pass the wrong groups after a copy-paste. The DataOp graph makes groups an intrinsic part of the recipe, not a side argument. This experiment also produces the best Kaggle submission so far.

## Method

- **Files touched:** `experiments/05_dataops.py`, `submissions/05_dataops.csv`
- **Change versus 04_tabular_pipeline:** Rewrite the pipeline as a skrub DataOp:
  - `skrub.var("visits", visits)` — named input
  - `.drop("target", axis=1).skb.mark_as_X(cv=GroupKFold(5), split_kwargs={"groups": groups})` — features + split baked in
  - `data["target"].skb.mark_as_y()` — target
  - `.skb.apply(TableVectorizer()).skb.apply(HistGradientBoostingRegressor(...), y=y_op)` — pipeline steps
  - `evaluate(pred)` with no `splitter=` reads cv/groups from the graph
- **Kaggle file:** `pred.skb.make_learner()` → `fit({"visits": visits})` → `predict({"visits": X_test})` → `submissions/05_dataops.csv`
- **Hub push:** `project.put("05_dataops", report)` to `bobathon-esilv`.

## Risks / things that could invalidate the result

- `X_test` has no `target` column — `make_learner().predict({"visits": X_test})` must not try to access it. The DataOp graph only references `data["target"]` for `mark_as_y`, which is not used at predict time.
- The scores should match 04_tabular_pipeline closely (same model, same split logic). A large divergence would indicate the groups were being passed incorrectly in one of the two experiments.

## Status

- **State:** done
- **Approved by user on:** 2025-07-04
- **Headline result:** RMSE 7.65 ± 0.16 (GroupKFold 5-fold via DataOps graph); Kaggle file submissions/05_dataops.csv written; hub https://skore.probabl.ai/hack_a_bbl/bobathon-esilv/cross-validations/75346
- **Implication for next iteration:** DataOps successfully bakes the split into the graph. Scores are consistent with 04_tabular_pipeline (within fold variance). Further gains likely require feature engineering: disease_duration, missing-value indicators, or drug-timing features.
