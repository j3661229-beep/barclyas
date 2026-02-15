"""
Model retraining pipeline: periodic retraining, model comparison, performance degradation detection.
Old vs new AUC comparison; automatic model version increment.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODELS_DIR
from .governance import log_model_version
from .model_training import (
    load_feature_data,
    prepare_X_y,
    train_test_split_data,
    train_xgboost_model,
    evaluate_model,
    get_current_model_version,
    increment_model_version,
    MODEL_FILE_XGB,
    METRICS_FILE,
)
from .feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)


def run_retraining(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    compare_to_previous: bool = True,
) -> dict[str, Any]:
    """
    Retrain XGBoost, compare AUC to previous model, save new version and metrics.
    If compare_to_previous and new AUC < old AUC by more than 0.02, flag degradation.
    """
    from .config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    old_version = get_current_model_version(models_dir)
    old_auc = None
    existing_metrics = {}
    if (models_dir / METRICS_FILE).exists():
        with open(models_dir / METRICS_FILE) as f:
            existing_metrics = json.load(f)
        if compare_to_previous:
            old_auc = existing_metrics.get("xgboost", {}).get("auc_roc")

    df = load_feature_data(data_dir)
    X, y = prepare_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = train_xgboost_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    new_auc = metrics.get("auc_roc", 0.0)

    new_version = increment_model_version(models_dir)
    joblib.dump(model, models_dir / MODEL_FILE_XGB)
    # Preserve full metrics structure; update only xgboost, model_version, training_timestamp
    full_metrics = dict(existing_metrics)
    full_metrics["model_version"] = new_version
    full_metrics["training_timestamp"] = datetime.now(timezone.utc).isoformat()
    full_metrics["xgboost"] = {**metrics, "f1": metrics.get("f1_score", metrics.get("f1", 0))}
    full_metrics["features_used"] = full_metrics.get("features_used", FEATURE_NAMES)
    if "feature_names" not in full_metrics:
        full_metrics["feature_names"] = FEATURE_NAMES
    with open(models_dir / METRICS_FILE, "w") as f:
        json.dump(full_metrics, f, indent=2)

    log_model_version(
        model_version=new_version,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
        feature_set=FEATURE_NAMES,
        metrics_summary=metrics,
    )

    degradation_detected = False
    if old_auc is not None and new_auc < old_auc - 0.02:
        degradation_detected = True
        logger.warning("Performance degradation: old AUC %.3f vs new AUC %.3f", old_auc, new_auc)

    return {
        "old_version": old_version,
        "new_version": new_version,
        "old_auc": old_auc,
        "new_auc": new_auc,
        "degradation_detected": degradation_detected,
        "metrics": metrics,
    }
