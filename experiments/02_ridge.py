# %% [markdown]
# # 02_ridge — Ridge regression, row holdout, Kaggle submission
#
# First real model: SimpleImputer(median) + Ridge(alpha=1.0) on numeric
# features. Evaluated with a random 20 % row holdout (same as 01_dummy).
# The patient-data leak is intentional here; GroupKFold arrives in 03_hgbr.
#
# Hub key: 02_ridge | Project: bobathon-esilv
# Kaggle file: submissions/02_ridge.csv

# %%
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

visits = X_train.join(y_train)

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]
y = visits["target"]

# %%
# Build Ridge pipeline: fill NaNs with column medians, then linear regression.
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
dummy = DummyRegressor(strategy="mean")

# Evaluate both models on the same random 20 % row holdout.
report_dummy = evaluate(dummy, X, y)
report_ridge = evaluate(ridge, X, y)
print("RMSE dummy:", report_dummy.metrics.rmse())
print("RMSE ridge:", report_ridge.metrics.rmse())

# %%
# Push ridge report to Skore Hub.
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", report_ridge)
# Prints: Consult your report at https://skore.probabl.ai/…

# %%
# Kaggle submission: refit on ALL training visits, predict test.
final = clone(ridge).fit(X, y)
submission = X_test[feature_cols].copy()
submission["target"] = final.predict(X_test[feature_cols])
submission = submission[["target"]].copy()
submission.index.name = "Index"
submission = submission.reset_index()

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print("Kaggle file written to submissions/02_ridge.csv")
