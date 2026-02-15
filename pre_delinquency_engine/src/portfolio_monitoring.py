"""
Portfolio monitoring layer: risk distribution, stress index, segment breakdown, drift detection (KS), alerts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    DATA_DIR,
    DRIFT_KS_THRESHOLD,
    HIGH_RISK_INCREASE_ALERT_PCT,
)

logger = logging.getLogger(__name__)


def risk_distribution_histogram(risk_scores: pd.Series) -> dict[str, Any]:
    """Bin risk scores and return counts for histogram."""
    bins = [0, 20, 40, 60, 80, 100]
    counts, _ = np.histogram(risk_scores.clip(0, 100), bins=bins)
    return {
        "bins": bins,
        "counts": counts.tolist(),
        "labels": ["0-20", "20-40", "40-60", "60-80", "80-100"],
    }


def portfolio_stress_index(risk_scores: pd.Series) -> float:
    """Single 0-100 index: higher = more portfolio stress. Mean of risk scores."""
    return float(risk_scores.clip(0, 100).mean())


def high_risk_pct(risk_tiers: pd.Series) -> float:
    """Percentage of customers in High risk tier."""
    n = len(risk_tiers)
    if n == 0:
        return 0.0
    return 100.0 * (risk_tiers == "High").sum() / n


def segment_wise_risk_breakdown(
    df: pd.DataFrame,
    risk_score_col: str = "risk_score",
    risk_tier_col: str = "risk_tier",
    segment_col: str | None = None,
) -> pd.DataFrame:
    """
    If segment_col provided (e.g. 'segment' or 'risk_tier'), return mean risk and count per segment.
    Otherwise use risk_tier as segment.
    """
    if segment_col and segment_col in df.columns:
        seg = segment_col
    else:
        seg = risk_tier_col
    agg = df.groupby(seg).agg(
        mean_risk_score=(risk_score_col, "mean"),
        count=(risk_score_col, "count"),
        high_risk_pct=(risk_tier_col, lambda s: 100.0 * (s == "High").sum() / len(s) if len(s) else 0),
    ).reset_index()
    return agg


def drift_detection_ks(
    scores_baseline: np.ndarray | pd.Series,
    scores_current: np.ndarray | pd.Series,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    KS test between baseline (e.g. last week) and current risk score distribution.
    Alert if KS statistic exceeds threshold (distribution shift).
    """
    threshold = threshold or DRIFT_KS_THRESHOLD
    scores_baseline = np.asarray(scores_baseline).flatten()
    scores_current = np.asarray(scores_current).flatten()
    stat, p_value = stats.ks_2samp(scores_baseline, scores_current)
    return {
        "ks_statistic": float(stat),
        "ks_p_value": float(p_value),
        "threshold": threshold,
        "alert": stat >= threshold,
        "message": "Significant risk distribution shift detected" if stat >= threshold else "No significant drift",
    }


def check_high_risk_increase_alert(
    prev_high_risk_pct: float,
    current_high_risk_pct: float,
    increase_threshold_pct: float | None = None,
) -> dict[str, Any]:
    """Alert if High-risk population increases by more than X%."""
    increase_threshold_pct = increase_threshold_pct or HIGH_RISK_INCREASE_ALERT_PCT
    change = current_high_risk_pct - prev_high_risk_pct
    alert = change >= increase_threshold_pct
    return {
        "prev_high_risk_pct": prev_high_risk_pct,
        "current_high_risk_pct": current_high_risk_pct,
        "change_pct": change,
        "threshold_pct": increase_threshold_pct,
        "alert": alert,
        "message": f"High-risk % increased by {change:.1f}%" if alert else "Within threshold",
    }


def run_portfolio_monitoring(
    risk_df: pd.DataFrame,
    risk_score_col: str = "risk_score",
    risk_tier_col: str = "risk_tier",
    baseline_scores: pd.Series | None = None,
    segment_col: str | None = None,
) -> dict[str, Any]:
    """
    Full portfolio monitoring snapshot: histogram, stress index, % high risk,
    segment breakdown, drift (if baseline provided), alerts.
    """
    scores = risk_df[risk_score_col]
    tiers = risk_df[risk_tier_col]
    out = {
        "risk_distribution_histogram": risk_distribution_histogram(scores),
        "portfolio_stress_index": portfolio_stress_index(scores),
        "high_risk_pct": high_risk_pct(tiers),
        "segment_breakdown": segment_wise_risk_breakdown(
            risk_df, risk_score_col=risk_score_col, risk_tier_col=risk_tier_col, segment_col=segment_col
        ).to_dict(orient="records"),
        "drift": None,
        "alerts": [],
    }
    if baseline_scores is not None and len(baseline_scores) > 0:
        drift = drift_detection_ks(baseline_scores, scores)
        out["drift"] = drift
        if drift.get("alert"):
            out["alerts"].append({"type": "drift", "detail": drift["message"]})
    return out
