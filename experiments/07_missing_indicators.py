# %% [markdown]
# # 07_missing_indicators — explicit binary missingness flags (B2)
#
# Adds six binary indicator columns on top of 06_feature_engineering:
# on_missing, off_missing, ledd_missing, gene_missing,
# time_since_intake_on_missing, time_since_intake_off_missing.
#
# HGBR already exploits NaN routing internally; these flags make the
# missingness pattern explicit and may give the model a more direct
# handle on visit type (ON-only vs OFF-available visits).
#
# Hub key: 07_missing_indicators | Project: bobathon-esilv

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

visits = X_train.join(y_train)

# %%
PK_HALF_LIFE = 90.0  # minutes
INDICATOR_COLS = ["on", "off", "ledd", "gene", "time_since_intake_on", "time_since_intake_off"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """B1 + B3 engineered features plus B2 missing-value indicators."""
    df = df.copy()
    # B1: disease stage
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    # B3: PK concentration proxy
    k = np.log(2) / PK_HALF_LIFE
    df["ledd_conc_on"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    df["ledd_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    # B2: explicit missing-value indicators
    for col in INDICATOR_COLS:
        df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df


visits_fe = add_features(visits)

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
project.put("07_missing_indicators", report)
