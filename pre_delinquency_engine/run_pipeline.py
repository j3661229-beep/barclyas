"""
End-to-end pipeline for Pre-Delinquency Intervention Engine.
Runs: data generation -> feature engineering -> model training -> credit risk, fairness, governance.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, MODELS_DIR, LOG_LEVEL, LOG_FORMAT
from src.data_generation import run_data_generation
from src.feature_engineering import run_feature_engineering, FEATURE_NAMES
from src.model_training import run_training
from src.risk_scoring import score_batch
from src.explainability import run_global_explanation
from src.business_impact import run_business_impact
from src.credit_risk import run_credit_risk_report
from src.fairness import run_fairness_validation
from src.governance import log_model_version
from src.monitoring import run_monitoring_and_save

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    import pandas as pd
    logger.info("1/6 Generating synthetic data...")
    run_data_generation(save=True)
    logger.info("2/6 Feature engineering...")
    run_feature_engineering()
    logger.info("3/6 Training models (LR + XGBoost)...")
    metrics, xgb_model, lr_model = run_training(save_metrics=True)
    from datetime import datetime, timezone
    log_model_version(
        model_version=metrics.get("model_version", "v1.0.0"),
        training_timestamp=metrics.get("training_timestamp", datetime.now(timezone.utc).isoformat()),
        feature_set=metrics.get("features_used", FEATURE_NAMES),
        metrics_summary=metrics.get("xgboost", {}),
    )
    logger.info("4/6 Credit risk report & SHAP...")
    df = pd.read_csv(DATA_DIR / "customer_features.csv")
    X = df[FEATURE_NAMES].fillna(0)
    scores = score_batch(X)
    run_credit_risk_report(df, risk_scores_df=scores)
    try:
        run_global_explanation(X, plot_path=MODELS_DIR / "shap_global.png")
    except Exception as e:
        logger.warning("SHAP plot skipped: %s", e)
    logger.info("5/6 Business impact & fairness...")
    try:
        impact = run_business_impact(feature_df=df, risk_scores_df=scores)
        logger.info("Business impact: %s", impact["summary"])
    except Exception as e:
        logger.warning("Business impact skipped: %s", e)
    try:
        run_fairness_validation(df, y_true=df.get("default_next_30_days"), y_prob=scores["default_probability"].values, y_pred=(scores["default_probability"] >= 0.5).astype(int).values)
    except Exception as e:
        logger.warning("Fairness validation skipped: %s", e)
    try:
        run_monitoring_and_save(X_train=X, X_current=X, scores_train=scores["default_probability"].values, scores_current=scores["default_probability"].values, models_dir=MODELS_DIR)
    except Exception as e:
        logger.warning("Drift report skipped: %s", e)
    logger.info("6/6 Pipeline complete. Data: data/, models: models/, reports: reports/.")


if __name__ == "__main__":
    main()
