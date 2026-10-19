# %% [markdown]
# # 05_dataops — skrub DataOps, groups baked in + Kaggle submission
#
# Same model as 04_tabular_pipeline (TableVectorizer + HGBR) but the
# GroupKFold split is attached directly to the DataOp graph via
# mark_as_X(cv=..., split_kwargs={"groups": ...}). evaluate(pred) reads
# cv and groups from the graph — no separate cv_splits list that can drift.
#
# Also produces the Kaggle submission: SkrubLearner fit on all training
# visits → predict X_test → submissions/05_dataops.csv.
#
# Hub key: 05_dataops | Project: bobathon-esilv
# Kaggle file: submissions/05_dataops.csv

# %%
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skrub import TableVectorizer
import skrub

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

visits = X_train.join(y_train)

# %%
# Build the DataOp graph. groups is derived from the same node as X,
# so it is guaranteed to stay aligned with the feature rows.
data = skrub.var("visits", visits)
groups = data["patient_id"]

X_op = (
    data.drop("target", axis=1)
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op
    .skb.apply(TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

# Freeze the DataOp graph into a SkrubLearner.
# evaluate(learner, data={"visits": ...}) reads cv + split_kwargs from the graph.
learner = pred.skb.make_learner()
report = evaluate(learner, data={"visits": visits})
print(report.metrics.rmse())

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("05_dataops", report)
# Prints: Consult your report at https://skore.probabl.ai/…

# %%
# Kaggle submission: fit a plain pipeline on all training visits, predict test.
# The DataOp graph's `.drop("target")` node fails on X_test (no target column).
# Use sklearn-style fit(X, y) instead for the submission.
from skrub import tabular_pipeline

X_full = visits.drop(columns=["patient_id", "target"])
y = visits["target"]
X_test_full = X_test.drop(columns=["patient_id"])

model_submission = tabular_pipeline("regressor")
model_submission.fit(X_full, y)
pred_test = model_submission.predict(X_test_full)

submission = X_test[[]].copy()  # empty frame preserving the Index
submission["target"] = pred_test
submission = submission.reset_index()  # brings Index back as a column

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "05_dataops.csv", index=False)
print("Kaggle file written to submissions/05_dataops.csv")
