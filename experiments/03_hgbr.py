# %% [markdown]
# # 03_hgbr — GroupKFold + HistGradientBoosting, missingness as signal
#
# Two changes versus 02_ridge:
#   1. Honest evaluation: GroupKFold(5) on patient_id — whole patients go
#      to one fold, never both sides. This matches the Kaggle holdout.
#   2. HistGradientBoostingRegressor: natively handles NaN, no imputation.
#      A missing OFF score is routed to a specific branch at each tree split,
#      so absence is itself a signal.
#
# Comparison report: dummy vs ridge vs hgbr, all on the same cv_splits.
#
# Hub key: 03_hgbr | Project: bobathon-esilv

# %%
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")

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
# Patient-grouped CV: each patient lands entirely in train or validation,
# never on both sides. This mirrors the Kaggle by-patient holdout.
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

# %%
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)
hgbr = HistGradientBoostingRegressor(random_state=0)

# Compare all three on the same grouped folds for visibility.
comparison = evaluate({"dummy": dummy, "ridge": ridge, "hgbr": hgbr}, X, y, splitter=cv_splits)
print(comparison.metrics.rmse())

# Separate HGBR report for hub push (project.put requires CrossValidationReport).
report_hgbr = evaluate(hgbr, X, y, splitter=cv_splits)

# %%
# Push to Skore Hub.
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("03_hgbr", report_hgbr)
# Prints: Consult your report at https://skore.probabl.ai/…
