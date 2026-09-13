# %% [markdown]
# # 02 — Clean, Engineer Features & Baseline Model
#
# Self-contained: reloads and re-derives the resolved-loans dataset from
# notebook 1 (condensed, since you've already seen why each step exists),
# then moves into cleaning, feature engineering, and the Logistic Regression
# baseline. This is the first cell that produces a real, defensible number.

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

pd.set_option("display.max_columns", 100)

# %% [markdown]
# ## Reload and re-derive `resolved` (condensed from notebook 1)

# %%
DATA_PATH = "../data/raw/accepted_2007_to_2018Q4.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df = df[df["issue_d"].dt.year >= 2016].copy()

resolved = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
resolved["target"] = (resolved["loan_status"] == "Charged Off").astype(int)
print(resolved.shape)

# %% [markdown]
# ## Select features — application-time only
#
# Everything here is information a lender would have *before* making a
# decision. `grade`, `sub_grade`, and `int_rate` are deliberately excluded —
# they're Lending Club's own risk output, not raw applicant data (see the
# logic doc if you want the full reasoning on why that's leakage).

# %%
FEATURE_COLS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "purpose", "dti", "delinq_2yrs", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "inq_last_6mths",
]

model_df = resolved[FEATURE_COLS + ["addr_state",
                                    "issue_d", "earliest_cr_line", "target"]].copy()
model_df.shape

# %% [markdown]
# ## Engineer credit history length
#
# `earliest_cr_line` is a date, not a usable number on its own. Converting
# it to "years of credit history as of the loan's issue date" turns it into
# the kind of feature a model can actually learn from.

# %%
model_df["earliest_cr_line"] = pd.to_datetime(
    model_df["earliest_cr_line"], format="%b-%Y")
model_df["credit_history_years"] = (
    (model_df["issue_d"] - model_df["earliest_cr_line"]).dt.days / 365.25
)
model_df = model_df.drop(columns=["earliest_cr_line"])

# %% [markdown]
# ## Temporal split — train on 2016–2017, test on 2018
#
# Not a random split. The model only ever sees the past during training,
# and gets evaluated on loans issued after that — the same way a model
# would actually be validated before going into production.

# %%
train = model_df[model_df["issue_d"].dt.year <=
                 2017].drop(columns=["issue_d"]).copy()
test = model_df[model_df["issue_d"].dt.year ==
                2018].drop(columns=["issue_d"]).copy()
print("train:", train.shape, " test:", test.shape)

# %% [markdown]
# ## Clean `emp_length` into a number
#
# It currently looks like "10+ years", "< 1 year", "n/a" — not usable as-is.

# %%
EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
train["emp_length_years"] = train["emp_length"].map(EMP_LENGTH_MAP)
test["emp_length_years"] = test["emp_length"].map(EMP_LENGTH_MAP)
train = train.drop(columns=["emp_length"])
test = test.drop(columns=["emp_length"])

# %% [markdown]
# ## Check what's still missing

# %%
train.isna().mean().sort_values(ascending=False).head(10)

# %% [markdown]
# ## Impute missing values — using only the training set's median
#
# The median gets computed from train, then applied to both train and test.
# Computing it from test data too would be a subtle leak: it would let
# information about the "future" test set quietly influence training.

# %%
NUMERIC_COLS = [
    "annual_inc", "dti", "delinq_2yrs", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "inq_last_6mths", "credit_history_years",
    "emp_length_years",
]
medians = train[NUMERIC_COLS].median()
train[NUMERIC_COLS] = train[NUMERIC_COLS].fillna(medians)
test[NUMERIC_COLS] = test[NUMERIC_COLS].fillna(medians)

# %% [markdown]
# ## One-hot encode the categorical columns
#
# `term`, `home_ownership`, and `purpose` aren't numbers — a model needs
# them turned into separate 0/1 columns first.

# %%
# Hold state aside. It never enters the model, it exists only so the
# fairness check can group predictions by geography afterwards.
train_state = train.pop("addr_state")
test_state = test.pop("addr_state")

CATEGORICAL_COLS = ["term", "home_ownership", "purpose"]
train_enc = pd.get_dummies(train, columns=CATEGORICAL_COLS, drop_first=True)
test_enc = pd.get_dummies(test, columns=CATEGORICAL_COLS, drop_first=True)

# align in case a category shows up in one split but not the other
train_enc, test_enc = train_enc.align(
    test_enc, join="left", axis=1, fill_value=0)
print(train_enc.shape, test_enc.shape)

# %% [markdown]
# ## Save the processed splits
#
# So the next phase (XGBoost) can load these directly instead of redoing
# all the cleaning above from scratch.

# %%
train_enc.to_csv("../data/processed/train_baseline.csv", index=False)
test_enc.to_csv("../data/processed/test_baseline.csv", index=False)

train_state.to_csv("../data/processed/train_state.csv", index=False)
test_state.to_csv("../data/processed/test_state.csv", index=False)


# %% [markdown]
# ## Split into X / y

# %%
y_train = train_enc["target"]
X_train = train_enc.drop(columns=["target"])
y_test = test_enc["target"]
X_test = test_enc.drop(columns=["target"])

# %% [markdown]
# ## Baseline: Logistic Regression
#
# `class_weight="balanced"` matters here — only ~22% of loans defaulted,
# so without this the model would happily predict "fully paid" for nearly
# everyone and still look decent. This tells it to actually weigh the
# minority class properly during training.
#
# This cell may take a minute or two on a laptop — over a million rows,
# that's expected, not a sign anything's wrong.

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline_model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Evaluate — ROC-AUC and PR-AUC, not accuracy
#
# Accuracy lies on imbalanced data (see the logic doc). These two actually
# measure whether the model can tell defaulters apart from non-defaulters.

# %%
y_pred_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC:  {pr_auc:.3f}")

# %% [markdown]
# ## Look at the coefficients
#
# This is the entire point of starting with Logistic Regression — every
# feature's effect is directly readable, no extra explainability tooling
# needed yet.

# %%
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": baseline_model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
coef_df.head(15)
# %%
