# %% [markdown]
# # 03 — XGBoost
#
# Loads the checkpoints saved in notebook 2, so no need to reload the raw
# CSV or redo any cleaning. This is where the interpretability-for-
# performance trade-off from the logic doc becomes a real number, not just
# a concept.

# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# %% [markdown]
# ## Load the processed splits

# %%
train_enc = pd.read_csv("../data/processed/train_baseline.csv")
test_enc = pd.read_csv("../data/processed/test_baseline.csv")

y_train_full = train_enc["target"]
X_train_full = train_enc.drop(columns=["target"])
y_test = test_enc["target"]
X_test = test_enc.drop(columns=["target"])

# %% [markdown]
# ## Carve out a validation set for early stopping
#
# A random split here, not temporal — that's fine, since this is only
# used to monitor training and decide when to stop adding trees. The 2018
# test set stays completely untouched until the very last step, same as
# the baseline.

# %%
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

# %% [markdown]
# ## Fit XGBoost
#
# `scale_pos_weight` handles the class imbalance, same purpose as
# `class_weight="balanced"` in the baseline. Early stopping watches the
# validation set and stops adding trees once it stops improving, instead
# of blindly training a fixed number.
#
# This cell is the slow one — could genuinely take several minutes on a
# laptop with data this size. That's expected.

# %%
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

print(f"Stopped after {xgb_model.best_iteration} trees")

# %% [markdown]
# ## Evaluate on the untouched 2018 test set
#
# Same metrics, same test set as the baseline — this is what makes the
# comparison fair. Baseline was ROC-AUC 0.678, PR-AUC 0.267.

# %%
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
pr_auc_xgb = average_precision_score(y_test, y_pred_proba_xgb)
print(f"ROC-AUC: {roc_auc_xgb:.3f}  (baseline: 0.678)")
print(f"PR-AUC:  {pr_auc_xgb:.3f}  (baseline: 0.267)")

# %% [markdown]
# ## Feature importance — a preview, not the real explanation yet
#
# This is XGBoost's own built-in importance score. It's useful as a rough
# read, but it's not the same rigor as SHAP — it doesn't tell you *why* any
# one applicant got their score, only which features mattered on average
# across all of them. That gap is exactly why phase 5 exists.

# %%
importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)
importance_df.head(15)

# %% [markdown]
# ## Save the trained model for the SHAP phase

# %%
joblib.dump(xgb_model, "../data/processed/xgb_model.joblib")
X_test.to_csv("../data/processed/X_test_final.csv", index=False)
y_test.to_csv("../data/processed/y_test_final.csv", index=False)
