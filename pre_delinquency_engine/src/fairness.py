"""
Fairness & bias validation: optional protected attributes (NOT used in model).
Subgroup AUC, Precision/Recall by segment, Disparate Impact Ratio, Statistical Parity Difference.
Output: fairness_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .config import REPORTS_DIR, TARGET_COL
from .feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)


# Optional protected attribute column name (synthetic segment only; not in model features)
PROTECTED_ATTR_COL = "segment"  # e.g. A/B for testing; not used in FEATURE_NAMES


def subgroup_auc_by_segment(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segments: np.ndarray | pd.Series,
) -> dict[str, float]:
    """AUC per segment. Segments identified by unique values."""
    result = {}
    for seg in np.unique(segments):
        mask = (segments == seg)
        if y_true[mask].sum() < 2 or (1 - y_true[mask]).sum() < 2:
            result[str(seg)] = 0.0
            continue
        try:
            result[str(seg)] = float(roc_auc_score(y_true[mask], y_prob[mask]))
        except Exception:
            result[str(seg)] = 0.0
    return result


def precision_recall_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: np.ndarray | pd.Series,
) -> dict[str, dict[str, float]]:
    """Precision and recall per segment."""
    result = {}
    for seg in np.unique(segments):
        mask = (segments == seg)
        if mask.sum() == 0:
            continue
        result[str(seg)] = {
            "precision": float(precision_score(y_true[mask], y_pred[mask], zero_division=0)),
            "recall": float(recall_score(y_true[mask], y_pred[mask], zero_division=0)),
            "count": int(mask.sum()),
        }
    return result


def disparate_impact_ratio(
    y_pred: np.ndarray,
    segments: np.ndarray | pd.Series,
    favored_segment: Any = None,
) -> dict[str, Any]:
    """
    Disparate Impact = (rate of positive outcome in unprivileged) / (rate in privileged).
    We treat positive outcome as predicted High-risk (y_pred==1). DI < 0.8 often flagged.
    """
    uniq = list(np.unique(segments))
    if len(uniq) < 2:
        return {"ratio": 1.0, "by_segment": {}, "message": "Need at least 2 segments"}
    favored = favored_segment if favored_segment is not None else uniq[0]
    rates = {}
    for seg in uniq:
        mask = (segments == seg)
        rates[str(seg)] = float(np.mean(y_pred[mask])) if mask.sum() > 0 else 0.0
    priv_rate = rates.get(str(favored), 0.0)
    if priv_rate == 0:
        return {"ratio": 1.0, "by_segment": rates, "message": "Privileged rate is 0"}
    di = min(rates.values()) / priv_rate if priv_rate > 0 else 1.0
    return {"ratio": float(di), "by_segment": rates, "favored_segment": str(favored)}


def statistical_parity_difference(
    y_pred: np.ndarray,
    segments: np.ndarray | pd.Series,
    favored_segment: Any = None,
) -> dict[str, Any]:
    """SPD = P(positive | unprivileged) - P(positive | privileged). Ideally near 0."""
    uniq = list(np.unique(segments))
    if len(uniq) < 2:
        return {"spd": 0.0, "by_segment": {}, "message": "Need at least 2 segments"}
    favored = favored_segment if favored_segment is not None else uniq[0]
    rates = {}
    for seg in uniq:
        mask = (segments == seg)
        rates[str(seg)] = float(np.mean(y_pred[mask])) if mask.sum() > 0 else 0.0
    unpriv_rates = [rates[s] for s in rates if str(s) != str(favored)]
    unpriv_avg = np.mean(unpriv_rates) if unpriv_rates else 0.0
    priv_rate = rates.get(str(favored), 0.0)
    spd = float(unpriv_avg - priv_rate)
    return {"spd": spd, "by_segment": rates, "favored_segment": str(favored)}


def run_fairness_validation(
    df: pd.DataFrame,
    y_true: np.ndarray | pd.Series | None = None,
    y_prob: np.ndarray | pd.Series | None = None,
    y_pred: np.ndarray | pd.Series | None = None,
    segment_col: str = PROTECTED_ATTR_COL,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run all fairness metrics. If segment_col not in df, add a synthetic segment (A/B) for demo.
    Protected attributes are NOT used in model—only for post-hoc fairness reporting.
    """
    if segment_col not in df.columns:
        # Synthetic segment: assign A/B by index for demonstration
        np.random.seed(42)
        segments = np.where(np.random.rand(len(df)) > 0.5, "A", "B")
    else:
        segments = df[segment_col].values
    y_true = np.asarray(y_true) if y_true is not None else df[TARGET_COL].values
    y_prob = np.asarray(y_prob) if y_prob is not None else None
    y_pred = np.asarray(y_pred) if y_pred is not None else (np.asarray(y_prob) >= 0.5).astype(int) if y_prob is not None else None
    if y_pred is None:
        y_pred = (y_true.copy()).astype(int)  # fallback

    report = {
        "note": "Protected attributes are not used in the model; used only for fairness reporting.",
        "subgroup_auc": subgroup_auc_by_segment(y_true, y_prob if y_prob is not None else y_pred.astype(float), segments),
        "precision_recall_by_segment": precision_recall_by_segment(y_true, y_pred, segments),
        "disparate_impact_ratio": disparate_impact_ratio(y_pred, segments),
        "statistical_parity_difference": statistical_parity_difference(y_pred, segments),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or REPORTS_DIR / "fairness_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Fairness report saved to %s", out_path)
    return report
