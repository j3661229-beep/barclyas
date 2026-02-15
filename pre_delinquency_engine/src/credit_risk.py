"""
Credit risk framework (PD–LGD–EAD) for Pre-Delinquency Intervention Engine.
Expected Loss = PD × LGD × EAD; threshold optimization (AUC vs loss-optimized).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AVG_EMI_DEFAULT,
    DEFAULT_LGD,
    EAD_MULTIPLIER_EMI,
    REPORTS_DIR,
    TARGET_COL,
)
from .feature_engineering import FEATURE_NAMES
from .risk_scoring import score_batch

logger = logging.getLogger(__name__)


def probability_to_pd(prob: float) -> float:
    """Model probability is treated as 30-day PD (Probability of Default)."""
    return float(np.clip(prob, 0.0, 1.0))


def simulate_ead(
    df: pd.DataFrame,
    emi_col: str | None = None,
    avg_emi: float = AVG_EMI_DEFAULT,
    multiplier: float = EAD_MULTIPLIER_EMI,
) -> np.ndarray:
    """
    Simulate Exposure at Default. EAD ≈ outstanding exposure (e.g. 3× monthly EMI).
    If no EMI column, use avg_emi for all.
    """
    if emi_col and emi_col in df.columns:
        emi = df[emi_col].fillna(avg_emi).values
    else:
        emi = np.full(len(df), avg_emi)
    return emi * multiplier


def simulate_lgd(
    n: int,
    lgd_mean: float = DEFAULT_LGD,
    lgd_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate Loss Given Default per customer (e.g. 45% ± noise). Not used in training."""
    rng = rng or np.random.default_rng(42)
    lgd = rng.normal(lgd_mean, lgd_std, n)
    return np.clip(lgd, 0.01, 0.99).astype(float)


def expected_loss_per_customer(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
) -> np.ndarray:
    """Expected Loss = PD × LGD × EAD per customer."""
    return pd_arr * lgd_arr * ead_arr


def threshold_optimization_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find threshold that maximizes Youden (sensitivity + specificity - 1) / AUC-style."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_youden = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1.0
        if youden > best_youden:
            best_youden = youden
            best_thresh = float(t)
    return {"threshold": best_thresh, "criterion": "youden", "value": best_youden}


def threshold_optimization_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    cost_fp: float = 50.0,
    cost_fn: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Find threshold that minimizes expected cost/loss.
    cost_fp: cost of false positive (intervene when no default); cost_fn: weight for missed default.
    Simplified: total_cost = sum(EL where predicted default) + cost_fp * FP + cost_fn * (EL of FN).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    pd_arr = y_prob
    el_arr = pd_arr * lgd_arr * ead_arr
    best_thresh = 0.5
    best_cost = float("inf")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn_el = el_arr[(y_true == 1) & (y_pred == 0)].sum()
        cost = cost_fp * fp + cost_fn * fn_el
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)
    return {"threshold": best_thresh, "criterion": "expected_loss", "value": best_cost}


def run_credit_risk_report(
    feature_df: pd.DataFrame,
    targets: pd.Series | None = None,
    risk_scores_df: pd.DataFrame | None = None,
    avg_emi: float = AVG_EMI_DEFAULT,
    lgd: float = DEFAULT_LGD,
    ead_multiplier: float = EAD_MULTIPLIER_EMI,
    save_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build PD–LGD–EAD, EL, compare AUC-optimized vs loss-optimized threshold; store structured report.
    """
    from .risk_scoring import score_batch
    if risk_scores_df is None:
        X = feature_df[FEATURE_NAMES].fillna(0)
        risk_scores_df = score_batch(X)
    pd_arr = risk_scores_df["default_probability"].values
    n = len(pd_arr)
    ead_arr = simulate_ead(feature_df, emi_col=None, avg_emi=avg_emi, multiplier=ead_multiplier)
    lgd_arr = simulate_lgd(n, lgd_mean=lgd)
    el_arr = expected_loss_per_customer(pd_arr, lgd_arr, ead_arr)

    y_true = None
    if targets is not None:
        y_true = np.asarray(targets).astype(int)
    elif TARGET_COL in feature_df.columns:
        y_true = feature_df[TARGET_COL].dropna().astype(int).values
    if y_true is not None and len(y_true) == n:
        thresh_auc = threshold_optimization_auc(y_true, pd_arr)
        thresh_loss = threshold_optimization_loss(y_true, pd_arr, lgd_arr, ead_arr)
    else:
        thresh_auc = {"threshold": 0.5, "criterion": "youden", "value": None}
        thresh_loss = {"threshold": 0.5, "criterion": "expected_loss", "value": None}

    portfolio_el = float(np.sum(el_arr))
    report = {
        "credit_risk_framework": {
            "PD": "model default probability (30-day)",
            "LGD": lgd,
            "EAD": f"{ead_multiplier}x monthly EMI (avg EMI={avg_emi})",
        },
        "portfolio_expected_loss": portfolio_el,
        "per_customer_EL_mean": float(np.mean(el_arr)),
        "per_customer_EL_std": float(np.std(el_arr)),
        "threshold_optimization": {
            "auc_optimized": thresh_auc,
            "loss_optimized": thresh_loss,
        },
        "comparison": {
            "auc_optimized_threshold": thresh_auc["threshold"],
            "loss_optimized_threshold": thresh_loss["threshold"],
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = save_path or REPORTS_DIR / "credit_risk_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Credit risk report saved to %s", out_path)
    return report
