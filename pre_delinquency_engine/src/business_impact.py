"""
Business impact simulation for Pre-Delinquency Intervention Engine.
Estimates: portfolio loss without system, loss prevented via early detection, % improvement in recovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, TARGET_COL
from .risk_scoring import score_batch, score_to_tier, probability_to_score


def load_customer_features_and_targets(data_dir: Path | None = None) -> pd.DataFrame:
    """Load feature matrix with default_next_30_days."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "customer_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature engineering first: {path}")
    return pd.read_csv(path)


def estimate_portfolio_loss_without_system(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    avg_emi: float = 15_000.0,
    recovery_rate_without: float = 0.25,
) -> dict[str, Any]:
    """
    Simulate portfolio loss if we do nothing (no early intervention).
    Assumes defaulters cause loss = (1 - recovery_rate) * avg_emi per default (simplified).
    """
    defaults = df[target_col].sum()
    total_exposure = defaults * avg_emi
    loss_without = total_exposure * (1 - recovery_rate_without)
    return {
        "total_defaults": int(defaults),
        "total_exposure_estimate": total_exposure,
        "recovery_rate_without_system": recovery_rate_without,
        "estimated_loss_without_system": loss_without,
    }


def estimate_loss_prevented_and_recovery_improvement(
    df: pd.DataFrame,
    risk_scores: pd.DataFrame,
  avg_emi: float = 15_000.0,
  recovery_rate_without: float = 0.25,
  recovery_rate_with_system: float = 0.45,
  high_risk_intervention_rate: float = 0.70,
) -> dict[str, Any]:
    """
    With early detection we intervene on High-risk; assume some % avoid default.
    Improvement: (recovery_with - recovery_without) and loss prevented.
    """
    high_risk = (risk_scores["risk_tier"] == "High").sum()
    # Assume we intervene on all High-risk; high_risk_intervention_rate of them avoid default
    defaults_avoided = high_risk * high_risk_intervention_rate * 0.3  # 30% of high-risk would have defaulted
    loss_prevented = defaults_avoided * avg_emi * (1 - recovery_rate_without)
    total_defaults = df[TARGET_COL].sum()
    recovery_improvement_pct = (recovery_rate_with_system - recovery_rate_without) * 100
    return {
        "high_risk_customers": int(high_risk),
        "defaults_avoided_estimate": defaults_avoided,
        "loss_prevented_estimate": loss_prevented,
        "recovery_rate_with_system": recovery_rate_with_system,
        "recovery_improvement_pct": recovery_improvement_pct,
    }


def run_business_impact(
    data_dir: Path | None = None,
  feature_df: pd.DataFrame | None = None,
  risk_scores_df: pd.DataFrame | None = None,
  avg_emi: float = 15_000.0,
) -> dict[str, Any]:
    """
    Run full business impact simulation.
    If feature_df/risk_scores_df not provided, loads from data dir and computes scores.
    """
    data_dir = data_dir or DATA_DIR
    if feature_df is None:
        feature_df = load_customer_features_and_targets(data_dir)
    from .feature_engineering import FEATURE_NAMES
    if risk_scores_df is None:
        risk_scores_df = score_batch(feature_df[FEATURE_NAMES].fillna(0))

    without = estimate_portfolio_loss_without_system(
        feature_df, avg_emi=avg_emi, recovery_rate_without=0.25
    )
    with_system = estimate_loss_prevented_and_recovery_improvement(
        feature_df,
        risk_scores_df,
        avg_emi=avg_emi,
        recovery_rate_without=0.25,
        recovery_rate_with_system=0.45,
    )
    return {
        "without_system": without,
        "with_system": with_system,
        "summary": {
            "estimated_portfolio_loss_without_system": without["estimated_loss_without_system"],
            "loss_prevented_via_early_detection": with_system["loss_prevented_estimate"],
            "recovery_rate_improvement_pct": with_system["recovery_improvement_pct"],
        },
    }
