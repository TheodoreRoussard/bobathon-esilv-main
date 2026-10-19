# %% [markdown]
# # EDA: Parkinson's Disease MDS-UPDRS Progression
#
# Exploratory data analysis of the train/test split from the Parkinson's
# disease MDS-UPDRS motor-score dataset, run before designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv`, `X_test.csv`
#   live in `data/`. This file never cleans or modifies them.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# X_train, y_train, and X_test are loaded and joined for the overview so
# the target column is visible alongside the features.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

RAW = X_train.join(y_train)
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview — train (features + target)
#
# Per-table report saved to `data/eda_train.html`, plus a compact
# per-column summary: dtype, fraction missing, number of unique values.

# %%
report = skrub.TableReport(RAW, title="train (X + y)", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Table overview — test (features only)
#
# Per-table report saved to `data/eda_test.html`.

# %%
report_test = skrub.TableReport(X_test, title="test (X only)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
{"n_rows": summary_test.get("n_rows"), "n_columns": len(summary_test.get("columns", []))}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of the continuous `target` (MDS-UPDRS total motor score).
# Shapes the regression metric choice and whether skewing warrants a
# target transform.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (which point to time-based validation) and
# high-cardinality id / group-like columns (which point to grouped
# validation, to avoid leaking an entity across folds).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag to call out explicitly.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
