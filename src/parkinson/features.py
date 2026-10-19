"""Per-patient feature engineering for the Parkinson's MDS-UPDRS progression task.

Two entry points:

- ``build_features(df)``      — applied to the full train or full test set;
                                uses only X columns (patient_id, age, on, off …).
                                Safe to apply to train and test separately before CV.

- ``add_infold_features(X_train_fold, y_train_fold, X_val_fold)``
                              — rater-bias correction and PK ratio correction.
                                Must be called inside each CV fold (train partition
                                only) to avoid target leakage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PK_HALF_LIFE_MIN = 90.0  # levodopa half-life, minutes (population average)
PK_BINS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 30.0]  # hours


# ---------------------------------------------------------------------------
# Helper: per-patient linear regression
# ---------------------------------------------------------------------------

def _patient_linreg(
    df: pd.DataFrame,
    value_col: str,
    age_col: str = "age",
) -> pd.DataFrame:
    """Return slope, intercept, and fitted value at each row's age.

    Rows with NaN in value_col are excluded from the fit. If fewer than 2
    valid points exist for a patient the fitted value falls back to the
    patient mean (also NaN-safe).
    """
    results: list[dict] = []
    for pid, grp in df.groupby("patient_id"):
        valid = grp[[age_col, value_col]].dropna()
        if len(valid) >= 2:
            x = valid[age_col].values
            y = valid[value_col].values
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = np.nan, np.nan

        for idx, row in grp.iterrows():
            if np.isfinite(slope):
                fitted = slope * row[age_col] + intercept
            else:
                mean_val = grp[value_col].mean()
                fitted = mean_val if pd.notna(mean_val) else np.nan
            results.append(
                {
                    "_idx": idx,
                    f"{value_col}_slope": slope,
                    f"{value_col}_intercept": intercept,
                    f"{value_col}_lin": fitted,
                }
            )
    return pd.DataFrame(results).set_index("_idx")


# ---------------------------------------------------------------------------
# Helper: previous / next known value within patient (sorted by age)
# ---------------------------------------------------------------------------

def _prev_next(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return previous and next non-NaN value within patient, sorted by age."""
    out = df.sort_values(["patient_id", "age"]).copy()
    out[f"{col}_prev"] = out.groupby("patient_id")[col].ffill().where(
        out[col].isna() | True  # keep NaN for first row
    )
    # ffill gives current if not NaN; we want strictly previous
    out[f"{col}_prev"] = out.groupby("patient_id")[col].shift(1).groupby(
        out["patient_id"]
    ).ffill()
    out[f"{col}_next"] = out.groupby("patient_id")[col].shift(-1).groupby(
        out["patient_id"]
    ).bfill()
    return out[[f"{col}_prev", f"{col}_next"]]


# ---------------------------------------------------------------------------
# Step 1: build_features — X-only, applied to full train or full test
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient aggregate and trajectory features.

    Parameters
    ----------
    df : DataFrame with at minimum columns:
         patient_id, age, age_at_diagnosis, on, off, ledd, Index (as index or col)

    Returns
    -------
    DataFrame with original columns plus engineered features.
    ``patient_id`` and ``Index`` are preserved but not used as model features.
    """
    df = df.copy()

    # --- disease duration (B1) ---
    df["tsd"] = df["age"] - df["age_at_diagnosis"]
    # robust version: replace missing age_at_diagnosis with patient mean
    pat_mean_diag = df.groupby("patient_id")["age_at_diagnosis"].transform("mean")
    df["tsd_robust"] = df["age"] - df["age_at_diagnosis"].fillna(pat_mean_diag)

    # --- per-patient visit count and age range ---
    df["n_visits"] = df.groupby("patient_id")["age"].transform("count")
    df["age_min"] = df.groupby("patient_id")["age"].transform("min")
    df["age_max"] = df.groupby("patient_id")["age"].transform("max")
    df["t_rel"] = df["age"] - df["age_min"]
    df["visit_rank"] = df.sort_values(["patient_id", "age"]).groupby(
        "patient_id"
    ).cumcount()
    # re-align rank to original index order
    df["visit_rank"] = df.groupby("patient_id")["age"].rank(method="first") - 1

    # --- per-patient aggregates for off, on, ledd ---
    for col in ["off", "on", "ledd"]:
        df[f"{col}_pat_mean"] = df.groupby("patient_id")[col].transform("mean")
        df[f"{col}_pat_median"] = df.groupby("patient_id")[col].transform("median")
        df[f"{col}_pat_std"] = df.groupby("patient_id")[col].transform("std")
        df[f"{col}_pat_n"] = df.groupby("patient_id")[col].transform(
            lambda s: s.notna().sum()
        )
        df[f"{col}_dev"] = df[col] - df[f"{col}_pat_mean"]

    # --- per-patient linear regression for off and on ---
    for col in ["off", "on"]:
        linreg_df = _patient_linreg(df, col)
        df = df.join(linreg_df)

    # --- previous / next known value for off and on ---
    for col in ["off", "on"]:
        pn = _prev_next(df, col)
        df = df.join(pn)

    # --- missingness flags ---
    df["off_isna"] = df["off"].isna().astype("int8")
    df["on_isna"] = df["on"].isna().astype("int8")

    # --- PK proxy (B3) ---
    k = np.log(2) / PK_HALF_LIFE_MIN
    df["ledd_conc_on"] = df["ledd"] * np.exp(-k * df["time_since_intake_on"])
    df["ledd_conc_off"] = df["ledd"] * np.exp(-k * df["time_since_intake_off"])

    return df


# ---------------------------------------------------------------------------
# Step 2: in-fold features — must be called inside each CV fold
# ---------------------------------------------------------------------------

def add_infold_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_apply: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rater-bias and PK-corrected features on X_train, apply to X_apply.

    Parameters
    ----------
    X_train : training fold features (already built_features applied)
    y_train : training fold target (used to compute biases)
    X_apply : fold to transform (val fold or test set; read-only)

    Returns
    -------
    Copy of X_apply with rater-bias and PK-corrected columns added.
    """
    out = X_apply.copy()
    train = X_train.copy()
    train["_target"] = y_train.values

    # --- rater bias: mean(off - target) and mean(on - target) per rater_id ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rater_bias_off = (
            train.dropna(subset=["off"])
            .groupby("rater_id")
            .apply(lambda g: (g["off"] - g["_target"]).mean())
            .rename("rater_bias_off")
        )
        rater_bias_on = (
            train.dropna(subset=["on"])
            .groupby("rater_id")
            .apply(lambda g: (g["on"] - g["_target"]).mean())
            .rename("rater_bias_on")
        )

    global_bias_off = (train["off"] - train["_target"]).mean()
    global_bias_on = (train["on"] - train["_target"]).mean()

    out["rater_bias_off"] = out["rater_id"].map(rater_bias_off).fillna(global_bias_off)
    out["rater_bias_on"] = out["rater_id"].map(rater_bias_on).fillna(global_bias_on)
    out["off_corr"] = out["off"] - out["rater_bias_off"]
    out["on_corr_rater"] = out["on"] - out["rater_bias_on"]

    # --- patient mean of off_corr ---
    # use train to compute patient means, map to out
    train["off_corr_train"] = train["off"] - train["rater_id"].map(rater_bias_off).fillna(global_bias_off)
    off_corr_pat_mean = train.groupby("patient_id")["off_corr_train"].mean().rename("off_corr_pat_mean")
    out["off_corr_pat_mean"] = out["patient_id"].map(off_corr_pat_mean)

    # --- PK ratio correction for ON ---
    pk_df = train.dropna(subset=["on", "_target", "time_since_intake_on"]).copy()
    pk_df["ratio"] = pk_df["on"] / pk_df["_target"].replace(0, np.nan)
    pk_df["bin"] = pd.cut(pk_df["time_since_intake_on"], bins=PK_BINS, right=True)
    pk_ratio = pk_df.groupby("bin", observed=True)["ratio"].median().rename("pk_ratio")
    global_ratio = pk_df["ratio"].median()

    if "time_since_intake_on" in out.columns:
        out["_pk_bin"] = pd.cut(out["time_since_intake_on"], bins=PK_BINS, right=True)
        out["_pk_ratio"] = out["_pk_bin"].map(pk_ratio).astype(float).fillna(global_ratio)
        out["on_corr"] = out["on"] / out["_pk_ratio"].replace(0, np.nan)
        out.drop(columns=["_pk_bin", "_pk_ratio"], inplace=True)
    else:
        out["on_corr"] = np.nan

    # --- patient mean of on_corr (from train) ---
    train["_pk_bin"] = pd.cut(train["time_since_intake_on"], bins=PK_BINS, right=True)
    train["_pk_ratio"] = train["_pk_bin"].map(pk_ratio).astype(float).fillna(global_ratio)
    train["on_corr_train"] = train["on"] / train["_pk_ratio"].replace(0, np.nan)
    on_corr_pat_mean = train.groupby("patient_id")["on_corr_train"].mean().rename("on_corr_pat_mean")
    out["on_corr_pat_mean"] = out["patient_id"].map(on_corr_pat_mean)

    # --- best = off_corr if available else on_corr ---
    out["best"] = np.where(out["off_corr"].notna(), out["off_corr"], out["on_corr"])

    # --- per-patient linear fit of best ~ age, evaluated at visit age ---
    best_linreg = _patient_linreg(out, "best")
    out = out.join(best_linreg[["best_lin"]])

    return out
