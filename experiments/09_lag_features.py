# %% [markdown]
# # 09_lag_features — per-patient lagged scores (B5)
#
# Adds three lag features derived from the previous visit per patient
# (sorted by age within patient):
#   prev_target: the true OFF score at the patient's prior visit
#   prev_on:     the measured ON score at the prior visit
#   prev_off:    the measured OFF score at the prior visit
#
# This directly encodes temporal progression — the strongest prior-model
# signal that all earlier experiments ignored. First visit per patient
# gets NaN; HGBR handles that natively.
#
# Leakage guard: shift(1) only looks backward within the sorted trajectory.
# GroupKFold holds out entire patients, so no cross-patient leakage.
#
# Risk: prev_target is a training label — test patients have no known
# previous target. For test rows, prev_target will be NaN. The model
# must generalise through the NaN path on the test set.
#
# Hub key: 09_lag_features | Project: bobathon-esilv
# Kaggle file: submissions/09_lag_features.csv

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skrub import tabular_pipeline

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

visits = X_train.join(y_train)

# %%
PK_HALF_LIFE = 90.0


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """B1 + B3 engineered features (no lags — applied before lag step)."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    k = np.log(2) / PK_HALF_LIFE
    df["ledd_conc_on"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    df["ledd_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-visit target, on, off per patient (sorted by age)."""
    df = df.sort_values(["patient_id", "age"])
    grp = df.groupby("patient_id")
    df["prev_target"] = grp["target"].shift(1)
    df["prev_on"] = grp["on"].shift(1)
    df["prev_off"] = grp["off"].shift(1)
    return df


# Apply base features, then compute lags on the full train set.
# Lags only look backward (shift(1)), so there is no forward leakage.
visits_fe = add_base_features(visits)
visits_fe = add_lag_features(visits_fe)

X_full = visits_fe.drop(columns=["patient_id", "target"])
y = visits_fe["target"]
groups = visits_fe["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %%
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
print(report.metrics.rmse())

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("09_lag_features", report)
# Prints: Consult your report at https://skore.probabl.ai/…

# %%
# Kaggle submission.
# For test visits: the last known training visit per patient provides the lag.
# New patients (no training history) will have NaN lags — HGBR handles that.
last_train = (
    visits_fe.sort_values(["patient_id", "age"])
    .groupby("patient_id")[["target", "on", "off"]]
    .last()
    .rename(columns={"target": "prev_target", "on": "prev_on", "off": "prev_off"})
)

X_test_fe = add_base_features(X_test)
X_test_fe = X_test_fe.join(last_train, on="patient_id")

X_test_full = X_test_fe.drop(columns=["patient_id"])

model.fit(X_full, y)
preds = model.predict(X_test_full)

submission = X_test[[]].copy()
submission["target"] = preds
submission = submission.reset_index()

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "09_lag_features.csv", index=False)
print("Kaggle file written to submissions/09_lag_features.csv")
