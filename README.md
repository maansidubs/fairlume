# Fairlume — Credit Risk Scoring Model

An explainable credit risk scoring model trained on Lending Club's historical loan
data: a Logistic Regression baseline, an XGBoost model tuned for lift over it, SHAP
explanations for every prediction, and a Streamlit app to demo it interactively.

This is the model behind the Fairlume product concept.

## Roadmap

1. Data & setup — download the loan history, trim to a workable window, scaffold the repo.
2. Explore & define the target — clean `loan_status`, filter to resolved loans (Fully Paid / Charged Off), engineer basic features.
3. Baseline: Logistic Regression — a simple, interpretable model, end-to-end.
4. XGBoost — tuned gradient boosting for real predictive lift over the baseline.
5. SHAP — global feature importance + per-applicant explanations.
6. Streamlit app — an interactive demo: enter applicant details, get a score and its explanation.

## Data source

Full accepted/rejected loan history (2007–2018): https://www.kaggle.com/datasets/wordsforthewise/lending-club

Smaller, pre-cleaned alternative if you want to skip trimming: https://www.kaggle.com/datasets/utkarshx27/lending-club-loan-dataset

Download the accepted-loans CSV and place it in `data/raw/`. That folder is gitignored —
don't commit raw data.

## Setup

```bash
# from the project root
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then open `notebooks/01_load_and_explore.py` in VS Code (Python Interactive / Jupyter mode —
the `# %%` markers define cells) and run it top to bottom once the CSV is in `data/raw/`.

## Structure

```
credit-risk-model/
├── data/
│   ├── raw/            # downloaded CSVs, gitignored
│   └── processed/      # cleaned/feature-engineered data, gitignored
├── notebooks/           # exploratory work, numbered in order
├── src/                 # reusable pipeline code once things stabilize
├── app/                 # the Streamlit app
├── requirements.txt
└── README.md
```

## Notes

- The dataset is trimmed to a recent issue-date window before modeling — noted here as a
  scoping decision, not a shortcut: it keeps iteration fast without changing the story.
- Loans in transitory states (`Current`, `Late`, `In Grace Period`) are excluded from the
  target, since the outcome isn't known yet.