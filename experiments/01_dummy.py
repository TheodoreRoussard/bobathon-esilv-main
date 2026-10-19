# %% [markdown]
# # 01_dummy — DummyRegressor mean baseline
#
# The laziest possible model: always predict the mean `target` from the
# training visits. This establishes the RMSE floor that every real model
# must beat. Any model that scores the same as or worse than this dummy
# has an upstream bug (data load, metric, or submission format), not a
# modelling problem.
#
# Evaluation: skore.evaluate with splitter=0.2 (random 20 % row holdout).
# The same patient can appear on both sides — this is a known limitation
# fixed by GroupKFold on Day 2. It is harmless for the dummy.
#
# Hub key: 01_dummy | Project: bobathon-esilv

# %%
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import Project, evaluate, login

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
# Load train visits (features + target joined on Index).
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
# Fit and evaluate the dummy with a random 20 % row holdout.
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
print("RMSE:", report.metrics.rmse())

# %%
# Push the report to Skore Hub.
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# The console prints: Consult your report at https://skore.probabl.ai/…
