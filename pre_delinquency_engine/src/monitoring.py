"""
Model monitoring & drift detection: feature distribution (train vs scoring), risk score drift, alerts.
Macro stress simulation: increase salary delays across portfolio and observe stress spike.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATA_DIR, DRIFT_KS_THRESHOLD, MODELS_DIR
from .feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

DRIFT_REPORT_FILE = "drift_report.json"


def feature_distribution_comparison(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compare feature distributions between training and scoring data (KS per feature)."""
    feature_names = feature_names or FEATURE_NAMES
    results = {}
    for col in feature_names:
        if col not in X_train.columns or col not in X_score.columns:
            continue
        a, b = X_train[col].dropna().values, X_score[col].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        stat, p_value = stats.ks_2samp(a, b)
        results[col] = {"ks_statistic": float(stat), "ks_p_value": float(p_value)}
    return results


def risk_score_drift_detection(
    scores_train: np.ndarray | pd.Series,
    scores_current: np.ndarray | pd.Series,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Drift in risk score distribution. Alert if KS exceeds threshold."""
    threshold = threshold or DRIFT_KS_THRESHOLD
    stat, p_value = stats.ks_2samp(np.asarray(scores_train).flatten(), np.asarray(scores_current).flatten())
    alert = stat >= threshold
    return {
        "ks_statistic": float(stat),
        "ks_p_value": float(p_value),
        "threshold": threshold,
        "alert": alert,
        "message": "Risk score drift detected" if alert else "No significant drift",
    }


def simulate_macro_stress(
    transactions_df: pd.DataFrame,
    salary_delay_extra_days: int = 5,
) -> pd.DataFrame:
    """
    Simulate macro stress: add extra salary_delay_days to salary_credit rows.
    Returns modified transactions (copy); use to re-run feature engineering and observe stress spike.
    """
    df = transactions_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    mask = df["transaction_type"] == "salary_credit"
    if "salary_delay_days" in df.columns:
        df.loc[mask, "salary_delay_days"] = df.loc[mask, "salary_delay_days"].fillna(0) + salary_delay_extra_days
    else:
        df["salary_delay_days"] = 0
        df.loc[mask, "salary_delay_days"] = salary_delay_extra_days
    if "date" in df.columns and mask.any():
        df.loc[mask, "date"] = df.loc[mask, "date"] + pd.Timedelta(days=salary_delay_extra_days)
    logger.info("Macro stress simulated: +%s days salary delay", salary_delay_extra_days)
    return df


def run_monitoring_checks(
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    scores_train: np.ndarray | pd.Series | None = None,
    scores_current: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Run feature distribution comparison and optional risk score drift; return alerts."""
    out = {
        "feature_drift": feature_distribution_comparison(X_train, X_current),
        "risk_score_drift": None,
        "alerts": [],
    }
    for col, v in out["feature_drift"].items():
        if v.get("ks_statistic", 0) >= DRIFT_KS_THRESHOLD:
            out["alerts"].append({"type": "feature_drift", "feature": col, "ks_statistic": v["ks_statistic"]})
    if scores_train is not None and scores_current is not None:
        out["risk_score_drift"] = risk_score_drift_detection(scores_train, scores_current)
        if out["risk_score_drift"].get("alert"):
            out["alerts"].append({"type": "risk_score_drift", "detail": out["risk_score_drift"]["message"]})
    return out


def save_drift_report(
    report: dict[str, Any],
    models_dir: Path | None = None,
) -> Path:
    """Persist drift snapshot to models/drift_report.json (post-deployment awareness)."""
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_drift": report.get("feature_drift", {}),
        "risk_score_drift": report.get("risk_score_drift"),
        "alerts": report.get("alerts", []),
        "alert_count": len(report.get("alerts", [])),
    }
    path = models_dir / DRIFT_REPORT_FILE
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info("Drift report saved to %s", path)
    return path


def run_monitoring_and_save(
    X_train: pd.DataFrame,
    X_current: pd.DataFrame,
    scores_train: np.ndarray | pd.Series | None = None,
    scores_current: np.ndarray | pd.Series | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Run monitoring checks and save snapshot to models/drift_report.json."""
    report = run_monitoring_checks(X_train, X_current, scores_train, scores_current)
    save_drift_report(report, models_dir=models_dir)
    return report
