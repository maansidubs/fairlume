"""
Fairlume — Streamlit demo of the REAL trained credit risk model.

Unlike the illustrative formula on the marketing landing page, this app
runs the actual XGBoost model and SHAP explainer trained in notebooks
02-04, live, on whatever you type in below.

Run with (from the project root, venv activated):
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Paths built from this file's own location — works no matter which
# directory you happen to run the command from.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"

st.set_page_config(page_title="Fairlume — Model Demo",
                   page_icon="🔎", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(DATA_DIR / "xgb_model.joblib")
    explainer = joblib.load(DATA_DIR / "shap_explainer.joblib")
    return model, explainer


model, explainer = load_artifacts()
model_features = model.get_booster().feature_names

st.title("Fairlume — the real model")
st.caption(
    "This runs the actual trained XGBoost model and SHAP explainer from the "
    "notebooks on whatever you enter below — not the illustrative formula "
    "on the marketing site."
)

col_input, col_output = st.columns([1, 1.3])

with col_input:
    st.subheader("Applicant details")

    loan_amnt = st.slider("Loan amount ($)", 1000, 40000, 12000, step=500)
    term = st.selectbox("Term", ["36 months", "60 months"])
    annual_inc = st.number_input(
        "Annual income ($)", min_value=0, value=65000, step=1000)
    emp_length_years = st.slider("Employment length (years)", 0, 10, 5)
    home_ownership = st.selectbox(
        "Home ownership", ["MORTGAGE", "OWN", "RENT"])
    purpose = st.selectbox(
        "Loan purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
         "medical", "small_business", "car", "moving", "vacation", "house",
         "wedding", "renewable_energy", "other"],
    )
    dti = st.slider("Debt-to-income ratio (%)", 0.0, 60.0, 18.0, step=0.5)
    credit_history_years = st.slider(
        "Length of credit history (years)", 0.0, 40.0, 10.0, step=0.5)
    delinq_2yrs = st.number_input(
        "Delinquencies in past 2 years", min_value=0, value=0)
    open_acc = st.number_input("Open credit accounts", min_value=0, value=8)
    total_acc = st.number_input(
        "Total credit accounts (ever)", min_value=0, value=20)
    pub_rec = st.number_input(
        "Public records (bankruptcies etc.)", min_value=0, value=0)
    revol_bal = st.number_input(
        "Revolving balance ($)", min_value=0, value=8000, step=100)
    revol_util = st.slider("Revolving utilization (%)",
                           0.0, 150.0, 35.0, step=0.5)
    inq_last_6mths = st.number_input(
        "Credit inquiries, last 6 months", min_value=0, value=1)

    run = st.button("Score this applicant", type="primary")


def build_feature_row(raw, model_features):
    """Turns the raw form inputs into a single-row DataFrame matching the
    model's exact trained feature schema — including one-hot columns —
    without needing to hardcode which categories survived encoding."""
    row = {
        "loan_amnt": raw["loan_amnt"],
        "annual_inc": raw["annual_inc"],
        "dti": raw["dti"],
        "delinq_2yrs": raw["delinq_2yrs"],
        "open_acc": raw["open_acc"],
        "pub_rec": raw["pub_rec"],
        "revol_bal": raw["revol_bal"],
        "revol_util": raw["revol_util"],
        "total_acc": raw["total_acc"],
        "inq_last_6mths": raw["inq_last_6mths"],
        "credit_history_years": raw["credit_history_years"],
        "emp_length_years": raw["emp_length_years"],
    }

    def set_onehot(prefix, value):
        for feat in model_features:
            if feat.startswith(prefix + "_"):
                row[feat] = 1 if feat == f"{prefix}_{value}" else 0

    set_onehot("term", raw["term"])
    set_onehot("home_ownership", raw["home_ownership"])
    set_onehot("purpose", raw["purpose"])

    return pd.DataFrame([row]).reindex(columns=model_features, fill_value=0)


with col_output:
    st.subheader("Result")
    if run:
        raw = dict(
            loan_amnt=loan_amnt, term=term, annual_inc=annual_inc,
            emp_length_years=emp_length_years, home_ownership=home_ownership,
            purpose=purpose, dti=dti, credit_history_years=credit_history_years,
            delinq_2yrs=delinq_2yrs, open_acc=open_acc, total_acc=total_acc,
            pub_rec=pub_rec, revol_bal=revol_bal, revol_util=revol_util,
            inq_last_6mths=inq_last_6mths,
        )
        input_df = build_feature_row(raw, model_features)

        proba = model.predict_proba(input_df)[0, 1]
        st.metric("Predicted probability of default", f"{proba:.1%}")

        if proba < 0.10:
            st.success("Low predicted risk")
        elif proba < 0.25:
            st.info("Moderate predicted risk")
        else:
            st.warning("High predicted risk")

        st.markdown("**Why this score — SHAP breakdown**")
        shap_values_row = explainer(input_df)
        shap.plots.waterfall(shap_values_row[0], show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info('Fill in the applicant details and click "Score this applicant."')
