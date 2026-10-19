# %% [markdown]
# # 10_lgbm_patient_feats — LightGBM + rich per-patient features
#
# build_features() (X-only cross-visit aggregates) brings HGB from 7.44 → ~4.45.
# This script evaluates three configurations on the same GroupKFold(5) splits:
#   3a: HGB  + build_features             (baseline for this experiment)
#   3b: LGBM + build_features             (same features, different model)
#   3c: blend α·HGB + (1-α)·LGBM         (best of both)
# Whichever is lowest produces the Kaggle submission.
#
# Hub key: 10_lgbm_patient_feats | Project: bobathon-esilv
# Kaggle file: submissions/10_lgbm_patient_feats.csv

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from skore import Project, evaluate, login
from skrub import TableVectorizer

from parkinson.features import build_features
from parkinson.hub import load_skore_credentials

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = PROJECT_ROOT / "submissions"
SUBMISSIONS.mkdir(exist_ok=True)

# %%
X_train_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
y_raw = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv", index_col="Index")

visits_raw = X_train_raw.join(y_raw)

# Step 1: apply build_features to full train and full test (X-only, no leakage)
visits_fe = build_features(visits_raw)
X_test_fe = build_features(X_test_raw)

X_fe = visits_fe.drop(columns=["patient_id", "target"])
y = visits_fe["target"]
groups = visits_fe["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_fe, y, groups=groups))

# Encode string columns as category for LightGBM
for col in X_fe.select_dtypes(include="object").columns:
    X_fe[col] = X_fe[col].astype("category")

X_test_sub = X_test_fe.drop(columns=["patient_id"])
for col in X_test_sub.select_dtypes(include="object").columns:
    X_test_sub[col] = X_test_sub[col].astype("category")

# %%
LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=63,
    min_child_samples=30, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, random_state=0, n_jobs=-1, verbose=-1,
)


def cv_oof(X, y_ser, splits, fit_fn):
    """Return OOF predictions and per-fold RMSEs."""
    oof = np.zeros(len(y_ser))
    rmses = []
    for tr, val in splits:
        m = fit_fn(X.iloc[tr], y_ser.iloc[tr])
        p = m.predict(X.iloc[val])
        oof[val] = p
        rmses.append(float(np.sqrt(np.mean((p - y_ser.iloc[val].values) ** 2))))
    return oof, rmses


# ── Step 3a: HGB + build_features ──────────────────────────────────────────
print("Running HGB...")
oof_hgb, rmses_hgb = cv_oof(
    X_fe, y, cv_splits,
    lambda Xtr, ytr: make_pipeline(
        TableVectorizer(), HistGradientBoostingRegressor(random_state=0)
    ).fit(Xtr, ytr),
)
print(f"HGB  + build_features : {np.mean(rmses_hgb):.4f} ± {np.std(rmses_hgb):.4f}")

# ── Step 3b: LGBM + build_features ─────────────────────────────────────────
print("Running LGBM...")
oof_lgbm, rmses_lgbm = cv_oof(
    X_fe, y, cv_splits,
    lambda Xtr, ytr: LGBMRegressor(**LGBM_PARAMS).fit(Xtr, ytr),
)
print(f"LGBM + build_features : {np.mean(rmses_lgbm):.4f} ± {np.std(rmses_lgbm):.4f}")

# ── Step 3c: blend ──────────────────────────────────────────────────────────
best_blend_rmse = float("inf")
best_alpha = 0.0
for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    blend_oof = alpha * oof_hgb + (1 - alpha) * oof_lgbm
    r = float(np.sqrt(np.mean((blend_oof - y.values) ** 2)))
    print(f"  blend α={alpha:.1f}: {r:.4f}")
    if r < best_blend_rmse:
        best_blend_rmse = r
        best_alpha = alpha

print(f"\nBest blend α={best_alpha:.1f} OOF RMSE: {best_blend_rmse:.4f}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n=== CV RMSE summary (GroupKFold-5, by patient) ===")
print(f"{'Model':<45} {'mean':>7} {'std':>8}")
print("-" * 62)
print(f"{'06_feature_engineering (prev best)':<45} {'7.4400':>7} {'0.1500':>8}")
print(f"{'HGB  + build_features':<45} {np.mean(rmses_hgb):>7.4f} {np.std(rmses_hgb):>8.4f}")
print(f"{'LGBM + build_features':<45} {np.mean(rmses_lgbm):>7.4f} {np.std(rmses_lgbm):>8.4f}")
print(f"{'Best blend HGB/LGBM':<45} {best_blend_rmse:>7.4f} {'(OOF)':>8}")

# ── Submission: use the best configuration ──────────────────────────────────
# Determine what to use for the final prediction
if best_alpha == 1.0 or (best_alpha >= 0.8 and np.mean(rmses_hgb) <= np.mean(rmses_lgbm)):
    print("\nUsing HGB for submission")
    final = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=0))
    final.fit(X_fe, y)
    test_preds = final.predict(X_test_sub)
elif best_alpha == 0.0 or (best_alpha <= 0.2 and np.mean(rmses_lgbm) < np.mean(rmses_hgb)):
    print("\nUsing LGBM for submission")
    final = LGBMRegressor(**LGBM_PARAMS)
    final.fit(X_fe, y)
    test_preds = final.predict(X_test_sub)
else:
    print(f"\nUsing blend α={best_alpha:.1f} for submission")
    hgb_final = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=0))
    hgb_final.fit(X_fe, y)
    lgbm_final = LGBMRegressor(**LGBM_PARAMS)
    lgbm_final.fit(X_fe, y)
    test_preds = best_alpha * hgb_final.predict(X_test_sub) + (1 - best_alpha) * lgbm_final.predict(X_test_sub)

submission = X_test_raw[[]].copy()
submission["target"] = test_preds
submission = submission.reset_index()
submission.to_csv(SUBMISSIONS / "10_lgbm_patient_feats.csv", index=False)
print(f"Kaggle file written: submissions/10_lgbm_patient_feats.csv ({len(submission)} rows)")

# ── Push HGB report to Skore Hub (skore.evaluate requires sklearn-compatible) ──
print("\nPushing to Skore Hub...")
hgb_hub = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor(random_state=0))
report = evaluate(hgb_hub, X_fe, y, splitter=cv_splits)
print("skore RMSE:", report.metrics.rmse())

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("10_lgbm_patient_feats", report)
