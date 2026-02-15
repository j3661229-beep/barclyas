"""
FastAPI backend for Pre-Delinquency Intervention Engine.
Endpoints: GET /customers, GET /customer/{id}, GET /risk/{id}, POST /trigger-intervention/{id}.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running as uvicorn api.main:app
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import DATA_DIR
from src.explainability import load_model_and_features, local_explanation
from src.feature_engineering import FEATURE_NAMES
from src.governance import generate_audit_report, log_intervention_decision, log_explanation_snapshot, log_risk_score
from src.intervention_engine import get_interventions, trigger_intervention
from src.risk_scoring import score_single
from src.retraining import get_current_model_version


app = FastAPI(
    title="Pre-Delinquency Intervention Engine API",
    description="Predict default risk and trigger proactive interventions.",
    version="1.0.0",
)


def _load_features_df() -> pd.DataFrame:
    path = DATA_DIR / "customer_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run pipeline first to generate data and features.")
    return pd.read_csv(path)


def _load_transactions_df() -> pd.DataFrame:
    path = DATA_DIR / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError("Run pipeline first to generate transactions.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@app.get("/")
def root():
    return {"service": "Pre-Delinquency Intervention Engine", "docs": "/docs"}


@app.get("/customers")
def list_customers(risk_tier: str | None = None, limit: int = 100):
    """
    List customers. Optional query: risk_tier (Low/Medium/High) to filter.
    Returns customer_id, risk_score, risk_tier for each.
    """
    try:
        df = _load_features_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    # Score all if we don't have precomputed scores
    from src.risk_scoring import score_batch
    X = df[FEATURE_NAMES].fillna(0)
    scores = score_batch(X)
    out = pd.concat([df[["customer_id"]], scores], axis=1)
    if risk_tier:
        out = out[out["risk_tier"] == risk_tier]
    out = out.head(limit)
    return out.to_dict(orient="records")


@app.get("/customer/{customer_id}")
def get_customer(customer_id: str):
    """Customer detail: features and recent transactions."""
    try:
        features_df = _load_features_df()
        transactions_df = _load_transactions_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if customer_id not in features_df["customer_id"].values:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = features_df[features_df["customer_id"] == customer_id].iloc[0]
    tx = transactions_df[transactions_df["customer_id"] == customer_id]
    tx = tx.sort_values("date", ascending=False).head(100)
    return {
        "customer_id": customer_id,
        "features": row[FEATURE_NAMES].fillna(0).to_dict(),
        "transactions": tx[["date", "transaction_type", "amount", "balance_after"]].to_dict(orient="records"),
    }


@app.get("/risk/{customer_id}")
def get_risk(customer_id: str):
    """
    Risk score, tier, and SHAP top-3 risk factors for the customer.
    """
    try:
        df = _load_features_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if customer_id not in df["customer_id"].values:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = df[df["customer_id"] == customer_id][FEATURE_NAMES].fillna(0)
    score_result = score_single(row.iloc[0])
    model, fn = load_model_and_features()
    X = df[FEATURE_NAMES].fillna(0)
    explanation = local_explanation(model, X, customer_id, fn)
    try:
        version = get_current_model_version()
        log_risk_score(customer_id, version, score_result["risk_score"], score_result["risk_tier"], score_result["default_probability"])
        log_explanation_snapshot(customer_id, {"top_3_risk_factors": explanation["top_3_risk_factors"], "shap_values": explanation["shap_values"]})
    except Exception:
        pass
    return {
        "customer_id": customer_id,
        "risk_score": score_result["risk_score"],
        "risk_tier": score_result["risk_tier"],
        "default_probability": score_result["default_probability"],
        "top_3_risk_factors": explanation["top_3_risk_factors"],
        "shap_values": explanation["shap_values"],
    }


@app.post("/trigger-intervention/{customer_id}")
def trigger_intervention_endpoint(customer_id: str):
    """
    Trigger intervention for customer if they are High risk. Idempotent: creates new log entry.
    """
    try:
        df = _load_features_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if customer_id not in df["customer_id"].values:
        raise HTTPException(status_code=404, detail="Customer not found")
    row = df[df["customer_id"] == customer_id][FEATURE_NAMES].fillna(0)
    score_result = score_single(row.iloc[0])
    if score_result["risk_tier"] != "High":
        raise HTTPException(
            status_code=400,
            detail=f"Intervention only for High risk. Current tier: {score_result['risk_tier']}",
        )
    recorded = trigger_intervention(
        customer_id=customer_id,
        risk_tier=score_result["risk_tier"],
        risk_score=score_result["risk_score"],
    )
    try:
        log_intervention_decision(customer_id, score_result["risk_tier"], score_result["risk_score"], "triggered:" + ",".join(r.get("action_type", "") for r in recorded))
    except Exception:
        pass
    return {"customer_id": customer_id, "interventions_triggered": recorded}


@app.get("/interventions")
def list_interventions(customer_id: str | None = None, limit: int = 100):
    """Intervention logs, optionally filtered by customer_id."""
    logs = get_interventions(customer_id=customer_id, limit=limit)
    return {"interventions": logs}


@app.get("/audit/{customer_id}")
def get_audit_report(customer_id: str):
    """Governance audit report for customer: model versions, risk scores, explanations, interventions."""
    try:
        return generate_audit_report(customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
