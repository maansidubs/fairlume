# %% [markdown]
# # 05 — Fairness check by geography
#
# The model never sees location. That is not the same as being neutral to
# it: ordinary credit variables carry geography inside them, so the model
# can end up treating some places more harshly than their actual repayment
# behaviour justifies, without anyone intending it.
#
# State is a proxy, not a protected class. This does not test for race or
# sex, and it cannot. What it does test is whether the model's errors are
# spread evenly across the country, which is the closest thing this data
# supports and is the standard first pass before anything more serious.

# %%
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import joblib

pd.set_option("display.max_rows", 60)

# %% [markdown]
# ## Load the test set, the outcomes, the state column, and the model

# %%
X_test = pd.read_csv("../data/processed/X_test_final.csv")
y_test = pd.read_csv("../data/processed/y_test_final.csv").squeeze("columns")
state = pd.read_csv("../data/processed/test_state.csv").squeeze("columns")
model = joblib.load("../data/processed/xgb_model.joblib")

assert len(X_test) == len(y_test) == len(state), "row counts do not line up"
print(f"{len(X_test):,} applicants, {state.nunique()} states")

# %%
scores = model.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## Choose a decision threshold
#
# A probability is not a decision. To ask whether anyone is treated
# unfairly there has to be a line, so this sets one policy that declines
# the riskiest 20% of applicants overall. The same line applies to
# everybody, everywhere. Any unevenness that follows comes from the model,
# not from a different rule being used in different places.

# %%
DECLINE_RATE = 0.20
threshold = np.quantile(scores, 1 - DECLINE_RATE)
declined = scores >= threshold
print(
    f"threshold {threshold:.4f}, declining {declined.mean():.1%} of applicants")

# %% [markdown]
# ## The metric that matters
#
# States genuinely differ in how often people default, so a state being
# declined more often is not by itself evidence of anything. The question
# is narrower and harder to explain away: among the people who went on to
# repay in full, how often were they declined anyway.
#
# That is the false positive rate, and it is the rate of a specific harm:
# a creditworthy applicant turned down. If it is materially higher in one
# place than another, the model is making its mistakes unevenly.

# %%
df = pd.DataFrame({
    "state": state.values,
    "actual_default": y_test.values,
    "score": scores,
    "declined": declined,
})

repaid = df[df["actual_default"] == 0]

by_state = pd.DataFrame({
    "applicants": df.groupby("state").size(),
    "default_rate": df.groupby("state")["actual_default"].mean(),
    "decline_rate": df.groupby("state")["declined"].mean(),
    "false_positive_rate": repaid.groupby("state")["declined"].mean(),
})

# Small states produce noisy rates, so hold them out of the comparison
# rather than letting a handful of loans drive the headline number.
MIN_APPLICANTS = 300
big = by_state[by_state["applicants"] >= MIN_APPLICANTS].copy()
print(f"{len(big)} of {len(by_state)} states have at least {MIN_APPLICANTS} applicants")

# %% [markdown]
# ## Results

# %%
overall_fpr = repaid["declined"].mean()
big["fpr_vs_national"] = big["false_positive_rate"] / overall_fpr
big = big.sort_values("false_positive_rate", ascending=False)

print(f"National false positive rate: {overall_fpr:.1%}\n")
print(big.head(10).round(4))
print()
print(big.tail(10).round(4))

# %%
worst = big.iloc[0]
best = big.iloc[-1]
spread = worst["false_positive_rate"] / best["false_positive_rate"]

print(f"Harshest: {worst.name}  FPR {worst['false_positive_rate']:.1%}"
      f"  ({worst['applicants']:,.0f} applicants)")
print(f"Gentlest: {best.name}  FPR {best['false_positive_rate']:.1%}"
      f"  ({best['applicants']:,.0f} applicants)")
print(f"Ratio between them: {spread:.2f}x")

# %% [markdown]
# ## The four-fifths rule
#
# US enforcement agencies have long used a rough screen: if one group is
# selected at less than 80% the rate of the most-favoured group, that is
# treated as a signal worth investigating. It is a guideline rather than a
# law of nature, and it was written for hiring, but it is the closest
# available yardstick and it is what a regulator would reach for first.
#
# Applied here to approval rather than rejection.

# %%
big["approval_rate"] = 1 - big["decline_rate"]
ratio = big["approval_rate"] / big["approval_rate"].max()
flagged = big[ratio < 0.8]

print(f"Highest approval rate: {big['approval_rate'].max():.1%}")
print(f"Lowest approval rate:  {big['approval_rate'].min():.1%}")
print(f"Ratio: {big['approval_rate'].min() / big['approval_rate'].max():.3f}")
print()
if len(flagged) == 0:
    print("No state falls below the four-fifths threshold.")
else:
    print(f"{len(flagged)} state(s) below the four-fifths threshold:")
    print(flagged[["applicants", "approval_rate", "default_rate"]].round(4))

# %% [markdown]
# ## Does the model rank equally well everywhere?
#
# A model can be evenly harsh and still be worse at telling people apart
# in one place than another. Where AUC is low, the score carries less
# information, and the decisions made from it are closer to arbitrary.

# %%


def state_auc(g):
    if g["actual_default"].nunique() < 2:
        return np.nan
    return roc_auc_score(g["actual_default"], g["score"])


aucs = (df[df["state"].isin(big.index)]
        .groupby("state")[["actual_default", "score"]]
        .apply(state_auc)
        .sort_values())

print(f"National ROC-AUC: {roc_auc_score(y_test, scores):.3f}\n")
print("Weakest five states:")
print(aucs.head(5).round(3))
print("\nStrongest five states:")
print(aucs.tail(5).round(3))

# %% [markdown]
# ## Save the table

# %%
big["roc_auc"] = aucs
big.to_csv("../data/processed/fairness_by_state.csv")
print("saved to data/processed/fairness_by_state.csv")

# %% [markdown]
# ## What this does and does not establish
#
# It does not clear the model. Geography is a weak proxy, protected
# characteristics were never in the file to begin with, and a model can
# pass every check here while still producing disparate impact along
# lines this data cannot see.
#
# What it establishes is narrower and still worth having: whether the
# model's mistakes fall evenly across places, measured on the outcome that
# actually harms someone. A result either way is publishable. An even
# spread is evidence the model is not obviously broken in this respect. An
# uneven one is the beginning of a real investigation.
