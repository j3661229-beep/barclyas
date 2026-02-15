"""
Explainability module using SHAP for Pre-Delinquency Intervention Engine.
Global feature importance and local explanations (top 3 risk factors).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODELS_DIR
from .feature_engineering import FEATURE_NAMES
from .model_training import FEATURE_NAMES_FILE, MODEL_FILE_XGB


def load_model_and_features(models_dir: Path | None = None):
    """Load XGBoost model and feature names from disk."""
    try:
        models_dir = models_dir or MODELS_DIR
        model = joblib.load(models_dir / MODEL_FILE_XGB)
        try:
            fn = joblib.load(models_dir / FEATURE_NAMES_FILE)
        except FileNotFoundError:
            fn = FEATURE_NAMES
        return model, fn
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def get_shap_explainer(model, X: pd.DataFrame, feature_names: list[str] | None = None):
    """
    Create a SHAP Explainer for the model using modern API.
    Uses a small background sample for efficiency.
    """
    try:
        feature_names = feature_names or FEATURE_NAMES
        # Use modern SHAP API - shap.Explainer auto-detects model type
        explainer = shap.Explainer(model)
        return explainer
    except Exception as e:
        # Fallback to TreeExplainer without deprecated parameters
        try:
            X_bg = X.sample(min(100, len(X)), random_state=42) if len(X) > 100 else X
            explainer = shap.TreeExplainer(model, X_bg)
            return explainer
        except Exception as e2:
            raise RuntimeError(f"Failed to create SHAP explainer: {str(e2)}")


def global_feature_importance(
    model,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    plot_path: Path | None = None,
) -> pd.DataFrame | dict[str, str]:
    """
    Compute global SHAP feature importance (mean |SHAP| per feature).
    Optionally save a bar plot to plot_path.
    Returns DataFrame on success or error dict on failure.
    """
    try:
        feature_names = feature_names or FEATURE_NAMES
        explainer = get_shap_explainer(model, X, feature_names)
        
        # Use modern SHAP API
        X_sample = X.sample(min(500, len(X)), random_state=42) if len(X) > 500 else X
        shap_values = explainer(X_sample)
        
        # Extract values - handle both old and new SHAP formats
        if hasattr(shap_values, 'values'):
            sv = shap_values.values
        else:
            sv = shap_values
            
        if isinstance(sv, list):
            sv = sv[0]
            
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
        importance_df = mean_abs.sort_values(ascending=False).reset_index()
        importance_df.columns = ["feature", "mean_abs_shap"]

        if plot_path:
            try:
                plot_path = Path(plot_path)
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                import matplotlib.pyplot as plt
                shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()
            except Exception as e:
                # Plot is optional, continue without it
                pass

        return importance_df
    except Exception as e:
        return {"error": f"Global SHAP explanation failed: {str(e)}"}


def local_explanation(
    model,
    X: pd.DataFrame,
    customer_index: int | str,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Local SHAP explanation for a single customer.
    customer_index: integer row index or customer_id (then X must have customer_id column).
    Returns dict with shap_values, base_value, and top_3_risk_factors or error dict.
    """
    try:
        feature_names = feature_names or FEATURE_NAMES
        if isinstance(customer_index, str):
            if "customer_id" not in X.columns:
                return {"error": "X must contain customer_id to use customer_id as index"}
            idx = X.index[X["customer_id"] == customer_index].tolist()
            if not idx:
                return {"error": f"Customer {customer_index} not found"}
            customer_index = idx[0]
            
        row = X.iloc[[customer_index]]
        if "customer_id" in row.columns:
            row = row.drop(columns=["customer_id"])
        row = row[feature_names] if all(c in row.columns for c in feature_names) else row

        explainer = get_shap_explainer(
            model, 
            X[feature_names] if all(c in X.columns for c in feature_names) else X, 
            feature_names
        )
        
        # Use modern SHAP API
        shap_result = explainer(row)
        
        # Extract SHAP values
        if hasattr(shap_result, 'values'):
            sv = shap_result.values[0]
        else:
            sv = shap_result
            if isinstance(sv, list):
                sv = sv[0]
            if len(sv.shape) > 1:
                sv = sv[0]
                
        # Extract base value
        if hasattr(shap_result, 'base_values'):
            base_value = float(shap_result.base_values[0])
        elif hasattr(explainer, "expected_value"):
            base_value = float(explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[0])
        else:
            base_value = 0.0

        # Top 3 risk factors: features with positive SHAP (increase default probability)
        contrib = pd.Series(sv, index=feature_names)
        top_positive = contrib.nlargest(3)
        top_3_risk_factors = [
            {
                "feature": name, 
                "shap_value": float(val), 
                "effect": "increases risk",
                "feature_value": float(row[name].iloc[0]) if name in row.columns else 0.0
            }
            for name, val in top_positive.items()
        ]

        return {
            "shap_values": {feature_names[i]: float(sv[i]) for i in range(len(sv))},
            "base_value": base_value,
            "top_3_risk_factors": top_3_risk_factors,
        }
    except Exception as e:
        return {"error": f"Local SHAP explanation failed: {str(e)}"}


def run_global_explanation(
    X: pd.DataFrame,
    plot_path: Path | None = None,
    models_dir: Path | None = None,
) -> pd.DataFrame | dict[str, str]:
    """
    Load model, compute global SHAP importance, optionally save plot.
    X should contain only feature columns (no customer_id if not in FEATURE_NAMES).
    Returns DataFrame on success or error dict on failure.
    """
    try:
        models_dir = models_dir or MODELS_DIR
        model, fn = load_model_and_features(models_dir)
        X_ = X[fn].copy() if all(c in X.columns for c in fn) else X.copy()
        return global_feature_importance(model, X_, fn, plot_path)
    except Exception as e:
        return {"error": f"Global explanation failed: {str(e)}"}

