"""
Production-grade ML pipeline for Pre-Delinquency Intervention Engine.
Trains Logistic Regression (baseline) and XGBoost; persists models and structured metrics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb

from .config import (
    AVG_EMI_DEFAULT,
    DATA_DIR,
    DEFAULT_LGD,
    EAD_MULTIPLIER_EMI,
    MODELS_DIR,
    MODEL_VERSION_INITIAL,
    TARGET_COL,
)
from .feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Version and file names
# -----------------------------------------------------------------------------
MODEL_FILE_XGB = "xgb_model.joblib"
MODEL_FILE_LR = "lr_baseline.joblib"
METRICS_FILE = "metrics.json"
VERSION_FILE = "version.json"
FEATURE_NAMES_FILE = "feature_names.joblib"

# -----------------------------------------------------------------------------
# Training constants (production: prefer config or env)
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_CV_FOLDS = 5
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)

# XGBoost hyperparameters (optimized for institutional deployment - targeting AUC ≥ 0.75)
XGB_HYPERPARAMS = {
    "n_estimators": 200,  # Increased from 100 for better learning
    "max_depth": 6,  # Slightly deeper trees for complex patterns
    "learning_rate": 0.05,  # Lower learning rate for better generalization
    "min_child_weight": 3,  # Regularization to prevent overfitting on small segments
    "subsample": 0.8,  # Bootstrap sampling for variance reduction
    "colsample_bytree": 0.8,  # Feature sampling for regularization
    "gamma": 0.1,  # Minimum loss reduction for split (regularization)
    "scale_pos_weight": 1.0,  # Auto-calculated based on class imbalance (updated at runtime)
    "random_state": RANDOM_STATE,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

LR_HYPERPARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}

# Model lifecycle status (production | candidate | shadow | deprecated)
ModelStatus = Literal["production", "candidate", "shadow", "deprecated"]
MODEL_STATUS_DEFAULT: ModelStatus = "production"


def _compute_training_data_hash(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """SHA-256 hash of training data for reproducibility."""
    h = hashlib.sha256()
    h.update(X_train.values.tobytes())
    h.update(y_train.values.astype(np.int8).tobytes())
    return h.hexdigest()


def _ensure_models_dir(models_dir: Path | None = None) -> Path:
    """Create models directory if it does not exist."""
    path = models_dir or MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_current_model_version(models_dir: Path | None = None) -> str:
    """Read model version from version.json; return initial if file missing (e.g. v1.0.0)."""
    path = (models_dir or MODELS_DIR) / VERSION_FILE
    if path.exists():
        try:
            with open(path) as f:
                v = json.load(f).get("model_version", MODEL_VERSION_INITIAL)
                return v if str(v).startswith("v") else f"v{v}"
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read version file: %s. Using initial version.", e)
    initial = MODEL_VERSION_INITIAL
    return initial if str(initial).startswith("v") else f"v{initial}"


def increment_model_version(models_dir: Path | None = None) -> str:
    """Increment patch version (e.g. v1.0.0 -> v1.0.1) and persist. Returns new version."""
    current = get_current_model_version(models_dir)
    current_clean = current.lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", current_clean)
    if match:
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        new_version = f"v{major}.{minor}.{patch + 1}"
    else:
        new_version = "v1.0.1"
    path = (models_dir or MODELS_DIR) / VERSION_FILE
    _ensure_models_dir(path.parent)
    with open(path, "w") as f:
        json.dump({"model_version": new_version}, f, indent=2)
    logger.info("Model version set to %s", new_version)
    return new_version


def _optimal_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> float:
    """Threshold that maximizes Youden index (sensitivity + specificity - 1)."""
    thresholds = thresholds if thresholds is not None else THRESHOLD_GRID
    best_t, best_j = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def _expected_loss_optimized_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ead_arr: np.ndarray,
    lgd_arr: np.ndarray,
    cost_fp: float = 50.0,
    cost_fn: float = 1.0,
) -> float:
    """Threshold that minimizes expected cost (FP cost + FN expected loss)."""
    from .credit_risk import threshold_optimization_loss
    result = threshold_optimization_loss(
        y_true, y_prob, lgd_arr, ead_arr,
        cost_fp=cost_fp, cost_fn=cost_fn,
        thresholds=THRESHOLD_GRID,
    )
    return result["threshold"]


def load_feature_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load customer features (with target) from data directory."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / "customer_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature engineering first. Expected: {path}")
    return pd.read_csv(path)


def prepare_X_y(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target y. Drops rows with missing target."""
    feature_cols = feature_cols or FEATURE_NAMES
    df = df.dropna(subset=[target_col])
    X = df[feature_cols].copy()
    X = X.fillna(0)
    y = df[target_col].astype(int)
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train-test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute AUC-ROC, precision, recall, F1, accuracy, confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc_roc"] = 0.0
    return metrics


def _class_imbalance_ratio(y: np.ndarray | pd.Series) -> float:
    """Ratio of positive class to negative class (e.g. 0.15 = 15% positive)."""
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_neg == 0:
        return 1.0
    return float(n_pos / n_neg)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Train baseline Logistic Regression."""
    model = LogisticRegression(**LR_HYPERPARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> xgb.XGBClassifier:
    """Train primary XGBoost classifier. Uses XGB_HYPERPARAMS by default."""
    # Calculate scale_pos_weight from class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    
    # Merge hyperparameters with dynamic scale_pos_weight
    params = {**XGB_HYPERPARAMS, "scale_pos_weight": scale_pos_weight, **kwargs}
    if random_state != RANDOM_STATE:
        params["random_state"] = random_state
    
    logger.info(f"Training XGBoost with scale_pos_weight={scale_pos_weight:.3f} (class ratio: {n_pos}/{n_neg})")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_models_and_metrics(
    models: dict[str, Any],
    metrics_dict: dict[str, Any],
    models_dir: Path | None = None,
) -> None:
    """
    Persist models and metrics to disk.
    models: dict with keys 'xgb_model' and 'lr_model' (sklearn/xgb objects).
    metrics_dict: full metrics structure (will be written as metrics.json).
    """
    models_dir = _ensure_models_dir(models_dir)
    # Save models
    if "xgb_model" in models:
        path_xgb = models_dir / MODEL_FILE_XGB
        joblib.dump(models["xgb_model"], path_xgb)
        logger.info("XGBoost model saved to %s", path_xgb)
    if "lr_model" in models:
        path_lr = models_dir / MODEL_FILE_LR
        joblib.dump(models["lr_model"], path_lr)
        logger.info("Logistic Regression model saved to %s", path_lr)
    # Feature names
    features_used = metrics_dict.get("features_used", [])
    if features_used:
        joblib.dump(features_used, models_dir / FEATURE_NAMES_FILE)
    # Metrics
    path_metrics = models_dir / METRICS_FILE
    with open(path_metrics, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Metrics saved to %s", path_metrics)


def run_training(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    save_metrics: bool = True,
    increment_version: bool = True,
    model_status: ModelStatus = MODEL_STATUS_DEFAULT,
) -> tuple[dict[str, Any], xgb.XGBClassifier, LogisticRegression]:
    """
    Load features, train-test split, train LR and XGBoost, evaluate, compute thresholds
    and business metrics, then persist models and structured metrics.
    Returns (metrics_dict, xgb_model, lr_model).
    """
    from .credit_risk import simulate_ead, simulate_lgd, expected_loss_per_customer

    data_dir = data_dir or DATA_DIR
    models_dir = _ensure_models_dir(models_dir or MODELS_DIR)

    df = load_feature_data(data_dir)
    X, y = prepare_X_y(df)
    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    train_shape = X_train.shape
    test_shape = X_test.shape

    # Class imbalance
    imbalance_ratio = _class_imbalance_ratio(y_train)
    logger.info("Class imbalance ratio (pos/neg): %.4f", imbalance_ratio)

    # Train models
    lr = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost_model(X_train, y_train)

    # Test-set predictions and probabilities
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    y_test_arr = np.asarray(y_test)

    # Per-model evaluation
    lr_metrics = evaluate_model(y_test_arr, lr_pred, lr_prob)
    xgb_metrics = evaluate_model(y_test_arr, xgb_pred, xgb_prob)

    # Cross-validation mean AUC
    try:
        cv_auc_lr = cross_val_score(
            LogisticRegression(**LR_HYPERPARAMS), X_train, y_train,
            cv=N_CV_FOLDS, scoring="roc_auc", n_jobs=-1
        )
        lr_metrics["cross_val_auc_mean"] = float(np.mean(cv_auc_lr))
        lr_metrics["cross_val_auc_std"] = float(np.std(cv_auc_lr))
    except Exception as e:
        logger.warning("LR cross-validation failed: %s", e)
        lr_metrics["cross_val_auc_mean"] = lr_metrics.get("auc_roc", 0.0)
        lr_metrics["cross_val_auc_std"] = 0.0

    try:
        cv_auc_xgb = cross_val_score(
            xgb.XGBClassifier(**XGB_HYPERPARAMS), X_train, y_train,
            cv=N_CV_FOLDS, scoring="roc_auc", n_jobs=-1
        )
        xgb_metrics["cross_val_auc_mean"] = float(np.mean(cv_auc_xgb))
        xgb_metrics["cross_val_auc_std"] = float(np.std(cv_auc_xgb))
    except Exception as e:
        logger.warning("XGB cross-validation failed: %s", e)
        xgb_metrics["cross_val_auc_mean"] = xgb_metrics.get("auc_roc", 0.0)
        xgb_metrics["cross_val_auc_std"] = 0.0

    # Optimal thresholds
    opt_thresh_lr = _optimal_threshold_youden(y_test_arr, lr_prob)
    opt_thresh_xgb = _optimal_threshold_youden(y_test_arr, xgb_prob)
    lr_metrics["optimal_threshold"] = opt_thresh_lr
    xgb_metrics["optimal_threshold"] = opt_thresh_xgb
    logger.info("Logistic Regression optimal threshold (Youden): %.4f", opt_thresh_lr)
    logger.info("XGBoost optimal threshold (Youden): %.4f", opt_thresh_xgb)

    # EAD/LGD for test set (for expected-loss optimized threshold)
    df_test = df.loc[X_test.index] if hasattr(X_test, "index") else df.iloc[-len(X_test):]
    ead_test = simulate_ead(df_test, emi_col=None, avg_emi=AVG_EMI_DEFAULT, multiplier=EAD_MULTIPLIER_EMI)
    lgd_test = simulate_lgd(len(X_test), lgd_mean=DEFAULT_LGD)
    el_test = expected_loss_per_customer(xgb_prob, lgd_test, ead_test)
    opt_thresh_xgb_el = _expected_loss_optimized_threshold(y_test_arr, xgb_prob, ead_test, lgd_test)
    xgb_metrics["expected_loss_optimized_threshold"] = opt_thresh_xgb_el
    logger.info("XGBoost expected-loss optimized threshold: %.4f", opt_thresh_xgb_el)

    # Business metrics (avg PD and EL on test set; loss reduction vs baseline)
    avg_pd = float(np.mean(xgb_prob))
    portfolio_expected_loss = float(np.sum(el_test))
    # Baseline: constant PD = observed default rate; EL = PD * LGD * EAD per customer
    baseline_pd = float(np.mean(y_test_arr))
    baseline_el = float(baseline_pd * np.sum(lgd_test * ead_test))
    loss_reduction_vs_baseline = float((baseline_el - portfolio_expected_loss) / baseline_el) if baseline_el > 0 else 0.0

    # Model version and timestamp
    version_path = models_dir / VERSION_FILE
    if increment_version and version_path.exists():
        model_version = increment_model_version(models_dir)
    else:
        model_version = get_current_model_version(models_dir)
        if not version_path.exists():
            with open(version_path, "w") as f:
                json.dump({"model_version": model_version}, f, indent=2)
            logger.info("Initial model version written: %s", model_version)
    training_timestamp = datetime.now(timezone.utc).isoformat()

    # Reproducibility: hash of training data
    data_hash = _compute_training_data_hash(X_train, y_train)

    # Model signature (prevents inference mismatch)
    input_features = list(feature_cols)
    feature_count = len(input_features)

    # Structured metrics payload
    metrics_dict = {
        "model_version": model_version,
        "model_status": model_status,
        "training_timestamp": training_timestamp,
        "data_hash": data_hash,
        "input_features": input_features,
        "feature_count": feature_count,
        "features_used": feature_cols,
        "training_data_shape": list(train_shape),
        "test_data_shape": list(test_shape),
        "random_state": RANDOM_STATE,
        "xgboost_hyperparameters": XGB_HYPERPARAMS,
        "logistic_regression_hyperparameters": LR_HYPERPARAMS,
        "class_imbalance_ratio": imbalance_ratio,
        "xgboost": {
            "auc_roc": xgb_metrics["auc_roc"],
            "precision": xgb_metrics["precision"],
            "recall": xgb_metrics["recall"],
            "f1_score": xgb_metrics["f1_score"],
            "f1": xgb_metrics["f1_score"],
            "accuracy": xgb_metrics["accuracy"],
            "optimal_threshold": xgb_metrics["optimal_threshold"],
            "expected_loss_optimized_threshold": xgb_metrics["expected_loss_optimized_threshold"],
            "confusion_matrix": xgb_metrics["confusion_matrix"],
            "cross_val_auc_mean": xgb_metrics["cross_val_auc_mean"],
            "cross_val_auc_std": xgb_metrics["cross_val_auc_std"],
        },
        "logistic_regression": {
            "auc_roc": lr_metrics["auc_roc"],
            "precision": lr_metrics["precision"],
            "recall": lr_metrics["recall"],
            "f1_score": lr_metrics["f1_score"],
            "f1": lr_metrics["f1_score"],
            "accuracy": lr_metrics["accuracy"],
            "optimal_threshold": lr_metrics["optimal_threshold"],
            "confusion_matrix": lr_metrics["confusion_matrix"],
            "cross_val_auc_mean": lr_metrics["cross_val_auc_mean"],
            "cross_val_auc_std": lr_metrics["cross_val_auc_std"],
        },
        "business_metrics": {
            "avg_pd": avg_pd,
            "portfolio_expected_loss": portfolio_expected_loss,
            "loss_reduction_vs_baseline": loss_reduction_vs_baseline,
        },
    }

    models_to_save = {"xgb_model": xgb_model, "lr_model": lr}
    if save_metrics:
        save_models_and_metrics(models_to_save, metrics_dict, models_dir)

    # Executive logging line (demo / ops visibility)
    el_reduction_pct = loss_reduction_vs_baseline * 100.0
    logger.info(
        "Model %s trained — AUC: %.2f — EL Reduction: %.1f%%",
        model_version,
        xgb_metrics["auc_roc"],
        el_reduction_pct,
    )

    return metrics_dict, xgb_model, lr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run_training(save_metrics=True)
    logger.info("Model training complete. Models and metrics in models/")
