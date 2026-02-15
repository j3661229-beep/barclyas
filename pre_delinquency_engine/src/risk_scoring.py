"""
Risk scoring engine for Pre-Delinquency Intervention Engine.
Converts model probability to 0–100 risk score and risk tiers: Low (0–40), Medium (40–70), High (70–100).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, RISK_TIER_HIGH, RISK_TIER_LOW, RISK_TIER_MEDIUM
from .feature_engineering import FEATURE_NAMES
from .model_training import MODEL_FILE_XGB


RiskTier = Literal["Low", "Medium", "High"]


def probability_to_score(prob: float) -> float:
    """Convert default probability to 0–100 risk score."""
    return float(np.clip(prob * 100.0, 0.0, 100.0))


def score_to_tier(score: float) -> RiskTier:
    """Map risk score to tier: Low (0–40), Medium (40–70), High (70–100)."""
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def score_batch(
    X: pd.DataFrame,
    model_path: Path | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score a batch of customers. X must contain feature columns.
    Returns DataFrame with columns: risk_score, risk_tier, default_probability.
    """
    model_path = model_path or (MODELS_DIR / MODEL_FILE_XGB)
    feature_names = feature_names or FEATURE_NAMES
    model = joblib.load(model_path)
    X_ = X[feature_names].fillna(0)
    proba = model.predict_proba(X_)[:, 1]
    scores = np.array([probability_to_score(p) for p in proba])
    tiers = [score_to_tier(s) for s in scores]
    return pd.DataFrame({
        "risk_score": scores,
        "risk_tier": tiers,
        "default_probability": proba,
    })


def score_single(
    features: dict | pd.Series,
    model_path: Path | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Score a single customer. features: dict or Series with feature names as keys.
    Returns dict with risk_score, risk_tier, default_probability.
    """
    feature_names = feature_names or FEATURE_NAMES
    if isinstance(features, dict):
        row = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}])
    else:
        row = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)
    out = score_batch(row, model_path=model_path, feature_names=feature_names)
    return {
        "risk_score": float(out["risk_score"].iloc[0]),
        "risk_tier": out["risk_tier"].iloc[0],
        "default_probability": float(out["default_probability"].iloc[0]),
    }
