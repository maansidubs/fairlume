# %% [markdown]
# # 01 — Load & Explore
#
# First look at the Lending Club accepted-loans data: shape, target variable,
# missing values, class balance. Run this top to bottom in VS Code's Python
# Interactive window (each `# %%` is a cell), or convert it to a notebook with
# `jupytext --to notebook 01_load_and_explore.py` if you'd rather work in Jupyter.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
sns.set_theme(style="whitegrid")

# %%
# Update this to match whatever the downloaded file is actually called —
# Kaggle's accepted-loans file name has varied across dataset versions.
DATA_PATH = "../data/raw/accepted_2007_to_2018Q4.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
print(df.shape)
df.head()

# %% [markdown]
# ## Trim to a manageable slice
#
# The full file is 2M+ rows across 150 columns. For a portfolio-scale project,
# a recent, well-populated window keeps every run fast without losing the story.

# %%
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df = df[df["issue_d"].dt.year >= 2016].copy()
print(df.shape)

# %% [markdown]
# ## Define the target
#
# `loan_status` includes several in-progress states (`Current`, `Late`,
# `In Grace Period`) where the final outcome isn't known yet. Keep only loans
# that resolved one way or the other.

# %%
df["loan_status"].value_counts()

# %%
resolved = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
resolved["target"] = (resolved["loan_status"] == "Charged Off").astype(int)
resolved["target"].value_counts(normalize=True)

# %% [markdown]
# ## Class balance
#
# Credit risk data is almost always imbalanced — defaults are the minority
# class. Worth knowing this number now: it'll shape modeling choices later
# (class weights, threshold tuning, etc.) rather than being a surprise then.

# %%
ax = resolved["target"].value_counts().plot(kind="bar")
ax.set_xticklabels(["Fully Paid", "Charged Off"], rotation=0)
ax.set_ylabel("count")
plt.title("Loan outcome distribution (2016–2018)")
plt.show()

# %% [markdown]
# ## Missing values
#
# A first pass — which columns are mostly empty (candidates to drop) vs which
# need imputation later.

# %%
missing = resolved.isna().mean().sort_values(ascending=False)
missing[missing > 0].head(30)
