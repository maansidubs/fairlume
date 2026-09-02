"""
Fairlume API — serves the REAL trained model over HTTP.

This is what index.html's live demo will eventually call, instead of
running the illustrative JS formula it uses today.

Run locally with (from the project root, venv activated):
    uvicorn api.main:app --reload

Then open http://localhost:8000/docs to test it interactively before
touching any frontend code.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"

model = joblib.load(DATA_DIR / "xgb_model.joblib")
explainer = joblib.load(DATA_DIR / "shap_explainer.joblib")
model_features = model.get_booster().feature_names

app = FastAPI(title="Fairlume API")

# Lets a page on a different domain (like the deployed landing page) call
# this API from the browser. "*" is fine for now — tighten this to your
# actual site's domain once it's deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class Applicant(BaseModel):
    loan_amnt: float = Field(..., ge=1000, le=40000)
    term: Literal["36 months", "60 months"]
    annual_inc: float = Field(..., ge=0)
    emp_length_years: float = Field(..., ge=0, le=10)
    home_ownership: Literal["MORTGAGE", "OWN", "RENT"]
    purpose: str
    dti: float = Field(..., ge=0, le=100)
    credit_history_years: float = Field(..., ge=0)
    delinq_2yrs: int = Field(..., ge=0)
    open_acc: int = Field(..., ge=0)
    total_acc: int = Field(..., ge=0)
    pub_rec: int = Field(..., ge=0)
    revol_bal: float = Field(..., ge=0)
    revol_util: float = Field(..., ge=0)
    inq_last_6mths: int = Field(..., ge=0)


def build_feature_row(applicant: Applicant) -> pd.DataFrame:
    """Same encoding logic as the Streamlit app — turns raw applicant
    fields into the exact one-hot feature row the model was trained on."""
    row = {
        "loan_amnt": applicant.loan_amnt,
        "annual_inc": applicant.annual_inc,
        "dti": applicant.dti,
        "delinq_2yrs": applicant.delinq_2yrs,
        "open_acc": applicant.open_acc,
        "pub_rec": applicant.pub_rec,
        "revol_bal": applicant.revol_bal,
        "revol_util": applicant.revol_util,
        "total_acc": applicant.total_acc,
        "inq_last_6mths": applicant.inq_last_6mths,
        "credit_history_years": applicant.credit_history_years,
        "emp_length_years": applicant.emp_length_years,
    }

    def set_onehot(prefix, value):
        for feat in model_features:
            if feat.startswith(prefix + "_"):
                row[feat] = 1 if feat == f"{prefix}_{value}" else 0

    set_onehot("term", applicant.term)
    set_onehot("home_ownership", applicant.home_ownership)
    set_onehot("purpose", applicant.purpose)

    return pd.DataFrame([row]).reindex(columns=model_features, fill_value=0)


@app.get("/")
def root():
    return {"status": "Fairlume API is running"}


@app.post("/score")
def score(applicant: Applicant):
    input_df = build_feature_row(applicant)

    proba = float(model.predict_proba(input_df)[0, 1])

    shap_row = explainer(input_df)
    factors = [
        {"feature": feat, "shap_value": float(val)}
        for feat, val in zip(model_features, shap_row.values[0])
    ]
    factors.sort(key=lambda f: abs(f["shap_value"]), reverse=True)

    return {
        "default_probability": proba,
        "base_value": float(shap_row.base_values[0]),
        "top_factors": factors[:8],
    }
