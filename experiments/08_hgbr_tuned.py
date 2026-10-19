# %% [markdown]
# # 08_hgbr_tuned — HalvingGridSearchCV over HGBR hyperparameters (B4)
#
# Takes the full feature set from 07_missing_indicators and searches over
# the three most impactful HGBR knobs:
#   - learning_rate: {0.05, 0.1}
#   - max_iter:      {200, 500}
#   - max_leaf_nodes:{15, 31, 63}
#
# HalvingGridSearchCV eliminates weak candidates early — fast enough for
# 12 grid points on 44k rows. Best estimator is used for the Kaggle file.
#
# Hub key: 08_hgbr_tuned | Project: bobathon-esilv
# Kaggle file: submissions/08_hgbr_tuned.csv

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GroupKFold, HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login
from skrub import TableVectorizer

from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

visits = X_train.join(y_train)

# %%
PK_HALF_LIFE = 90.0
INDICATOR_COLS = ["on", "off", "ledd", "gene", "time_since_intake_on", "time_since_intake_off"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """B1 + B2 + B3 feature engineering."""
    df = df.copy()
    df["disease_duration"] = df["age"] - df["age_at_diagnosis"]
    k = np.log(2) / PK_HALF_LIFE
    df["ledd_conc_on"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    df["ledd_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])
    for col in INDICATOR_COLS:
        df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df


visits_fe = add_features(visits)
X_test_fe = add_features(X_test)

X_full = visits_fe.drop(columns=["patient_id", "target"])
y = visits_fe["target"]
groups = visits_fe["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

# %%
# Build a pipeline and search over HGBR hyperparameters.
pipe = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)
param_grid = {
    "histgradientboostingregressor__learning_rate": [0.05, 0.1],
    "histgradientboostingregressor__max_iter": [200, 500],
    "histgradientboostingregressor__max_leaf_nodes": [15, 31, 63],
}
search = HalvingGridSearchCV(
    pipe,
    param_grid,
    scoring="neg_root_mean_squared_error",
    random_state=0,
    n_jobs=-1,
)
search.fit(X_full, y)
print("Best params:", search.best_params_)
print("Best CV RMSE:", -search.best_score_)

best_model = search.best_estimator_

# %%
# Evaluate the best estimator with proper GroupKFold for a fair score.
report = evaluate(best_model, X_full, y, splitter=cv_splits)
print(report.metrics.rmse())

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("08_hgbr_tuned", report)
# Prints: Consult your report at https://skore.probabl.ai/…

# %%
# Kaggle submission: refit best estimator on all training visits.
best_model.fit(X_full, y)
X_test_full = X_test_fe.drop(columns=["patient_id"])
preds = best_model.predict(X_test_full)

submission = X_test[[]].copy()
submission["target"] = preds
submission = submission.reset_index()

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "08_hgbr_tuned.csv", index=False)
print("Kaggle file written to submissions/08_hgbr_tuned.csv")
