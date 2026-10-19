# %% [markdown]
# # 04_tabular_pipeline — skrub tabular_pipeline with categoricals
#
# Adds gene, cohort, and rater_id to the feature set via
# skrub.tabular_pipeline("regressor"), which wraps TableVectorizer +
# HistGradientBoostingRegressor. TableVectorizer auto-picks encoders:
# one-hot for low-cardinality columns, StringEncoder for text.
#
# Same GroupKFold(5) cv_splits as 03_hgbr for a fair comparison.
#
# Hub key: 04_tabular_pipeline | Project: bobathon-esilv

# %%
from pathlib import Path

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

# Full feature set: drop identifiers and target; keep all clinical columns
# including categoricals (gene, cohort, rater_id).
X_full = visits.drop(columns=["patient_id", "target"])
y = visits["target"]

# %%
# Same grouped CV as 03_hgbr — whole patients in one fold, never split.
groups = visits["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %%
# tabular_pipeline wraps TableVectorizer + HistGradientBoostingRegressor.
# TableVectorizer picks the encoder per column automatically.
model = tabular_pipeline("regressor")
report = evaluate(model, X_full, y, splitter=cv_splits)
print(report.metrics.rmse())

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("04_tabular_pipeline", report)
# Prints: Consult your report at https://skore.probabl.ai/…
