# %% [markdown]
# # 04 — SHAP Explainability
#
# Loads the trained model saved in notebook 3 — no need to retrain anything.
# This phase fixes the exact problem you just ran into: XGBoost's built-in
# feature importance made `term_60 months` look implausibly dominant. SHAP
# explains predictions properly instead of relying on that shortcut.

# %%
import pandas as pd
import numpy as np
import joblib
import shap

# %% [markdown]
# ## Load the trained model and test set

# %%
xgb_model = joblib.load("../data/processed/xgb_model.joblib")
X_test = pd.read_csv("../data/processed/X_test_final.csv")
y_test = pd.read_csv("../data/processed/y_test_final.csv").squeeze()

# %% [markdown]
# ## Sample the test set for the global explanation
#
# Computing SHAP values for every one of the ~240k test rows would be slow
# and isn't necessary — a representative sample gives a reliable read on
# which features matter overall.

# %%
SAMPLE_SIZE = 3000
X_sample = X_test.sample(SAMPLE_SIZE, random_state=42)

# %% [markdown]
# ## Build the explainer and compute SHAP values
#
# `TreeExplainer` is built specifically for tree-based models like XGBoost —
# it computes exact SHAP values efficiently instead of approximating them.
# This cell might take a minute, that's normal.

# %%
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_sample)

# %% [markdown]
# ## Global: which features matter, on average, across applicants
#
# Each dot is one applicant. Position left/right is that feature's actual
# impact on their risk score. Color is the feature's value for that
# applicant (red = high, blue = low).

# %%
shap.summary_plot(shap_values, X_sample)

# %% [markdown]
# ## Compare against XGBoost's built-in importance
#
# This is the payoff. See how differently SHAP ranks things compared to the
# "gain" importance from notebook 3, where `term_60 months` alone looked
# more important than everything else combined.

# %%
mean_abs_shap = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
mean_abs_shap.head(15)

# %% [markdown]
# ## Local: explain two specific applicants
#
# Pulling the single highest-risk and single lowest-risk applicant out of
# the sample, and checking what actually happened to their loan — this is
# the real test of whether the explanation lines up with reality.

# %%
sample_pred_proba = xgb_model.predict_proba(X_sample)[:, 1]
y_sample = y_test.loc[X_sample.index]

highest_risk_pos = np.argmax(sample_pred_proba)
lowest_risk_pos = np.argmin(sample_pred_proba)

actual_highest = "Charged Off" if y_sample.iloc[highest_risk_pos] == 1 else "Fully Paid"
actual_lowest = "Charged Off" if y_sample.iloc[lowest_risk_pos] == 1 else "Fully Paid"

print(
    f"Highest-risk applicant — predicted default probability: {sample_pred_proba[highest_risk_pos]:.3f}, actual outcome: {actual_highest}")
print(
    f"Lowest-risk applicant — predicted default probability: {sample_pred_proba[lowest_risk_pos]:.3f}, actual outcome: {actual_lowest}")

# %% [markdown]
# ### Why did the highest-risk applicant get that score?

# %%
shap.plots.waterfall(shap_values[highest_risk_pos])

# %% [markdown]
# ### And the lowest-risk applicant?

# %%
shap.plots.waterfall(shap_values[lowest_risk_pos])

# %% [markdown]
# ## Save the explainer for the Streamlit app
#
# So the next phase can load it directly instead of rebuilding it.

# %%
joblib.dump(explainer, "../data/processed/shap_explainer.joblib")
