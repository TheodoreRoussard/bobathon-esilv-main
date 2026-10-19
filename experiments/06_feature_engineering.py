# %% [markdown]
# # 06_feature_engineering — disease_duration (B1) + PK proxy (B3)
#
# Two new features on top of 04_tabular_pipeline:
#
# B1: disease_duration = age - age_at_diagnosis
#     Replaces the near-collinear pair (Pearson r=0.94) with the
#     clinically meaningful "years since PD diagnosis".
#
# B3: ledd_conc_on/off = ledd * exp(-ln(2)/90 * time_since_intake_*)
#     Estimated levodopa blood concentration at assessment time,
#     using a population half-life of ~90 minutes. Encodes how much
#     drug was still active, which directly biases the measured score.
#
# Everything else identical to 04_tabular_pipeline:
# tabular_pipeline("regressor") + GroupKFold(5) on patient_id.
#
# Hub key: 06_feature_engineering | Project: bobathon-esilv
# Kaggle file: submissions/06_feature_engineering.csv

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
# Feature engineering — applied identically to train and test.
PK_HALF_LIFE = 90.0  # minutes; published levodopa population average


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add disease_duration (B1) and PK concentration proxies (B3)."""
    df = df.copy()
    # B1: disease stage — years since diagnosis
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    # B3: levodopa blood-level proxy at ON/OFF assessment time
    # NaN propagates where ledd or time_since_intake_* is missing — HGBR handles that.
    k = np.log(2) / PK_HALF_LIFE
    df["ledd_conc_on"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    df["ledd_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    return df


visits_fe = add_features(visits)
X_test_fe = add_features(X_test)

# %%
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
project.put("06_feature_engineering", report)
# Prints: Consult your report at https://skore.probabl.ai/…

# %%
# Kaggle submission: fit on all training visits, predict test.
model.fit(X_full, y)
X_test_full = X_test_fe.drop(columns=["patient_id"])
preds = model.predict(X_test_full)

submission = X_test[[]].copy()
submission["target"] = preds
submission = submission.reset_index()

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "06_feature_engineering.csv", index=False)
print("Kaggle file written to submissions/06_feature_engineering.csv")
