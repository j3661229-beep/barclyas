# Model and File Links Reference

Single source of truth for model paths, metrics, and versioning across the Pre-Delinquency Intervention Engine.

## Model and artifact paths (all under `models/`)

| Artifact | Filename | Defined in | Used by |
|----------|----------|------------|---------|
| XGBoost model | `xgb_model.joblib` | `src/model_training.py` (`MODEL_FILE_XGB`) | `risk_scoring`, `explainability`, `api`, `dashboard`, `retraining` |
| Logistic Regression | `lr_baseline.joblib` | `src/model_training.py` (`MODEL_FILE_LR`) | `model_training` (save only) |
| Feature names | `feature_names.joblib` | `src/model_training.py` (`FEATURE_NAMES_FILE`) | `model_training`, `explainability` |
| Metrics | `metrics.json` | `src/model_training.py` (`METRICS_FILE`) | `model_training`, `retraining`, `dashboard` |
| Version | `version.json` | `src/model_training.py` (`VERSION_FILE`) | `model_training`, `retraining`, `api` (via `get_current_model_version`) |

## Config (paths)

- `MODELS_DIR`: `src/config.py` (default: `PROJECT_ROOT / "models"`), overridable via `PDE_MODELS_DIR`.
- `DATA_DIR`, `REPORTS_DIR`: same pattern in `config.py`.

## Version flow

- **First training**: `model_training.run_training()` writes `version.json` with `v1.0.0` (from `MODEL_VERSION_INITIAL` in config).
- **Subsequent training**: `run_training(increment_version=True)` calls `increment_model_version()` → e.g. `v1.0.1`.
- **Retraining**: `retraining.run_retraining()` uses `model_training.increment_model_version()` and updates `metrics.json` while preserving existing keys (e.g. `business_metrics`, `features_used`).

## Who reads/writes what

- **Writes**  
  - `model_training.run_training()`: all of the above (models, feature names, metrics, version).  
  - `retraining.run_retraining()`: XGBoost model, `metrics.json` (merge with existing), `version.json` (via `increment_model_version`).
- **Reads**  
  - `risk_scoring`: `MODELS_DIR / MODEL_FILE_XGB`, `FEATURE_NAMES` (fallback if no joblib).  
  - `explainability`: `MODEL_FILE_XGB`, `FEATURE_NAMES_FILE`.  
  - `api`: `get_current_model_version()` (from retraining, which uses model_training).  
  - `dashboard`: `MODELS_DIR / "metrics.json"`.

## Metrics.json structure (production)

- Top-level: `model_version`, `training_timestamp`, `features_used`, `training_data_shape`, `test_data_shape`, `random_state`, `xgboost_hyperparameters`, `logistic_regression_hyperparameters`, `class_imbalance_ratio`.
- `xgboost` / `logistic_regression`: `auc_roc`, `precision`, `recall`, `f1_score`, `f1`, `accuracy`, `optimal_threshold`, `confusion_matrix`, `cross_val_auc_mean`/`std`; XGB only: `expected_loss_optimized_threshold`.
- `business_metrics`: `avg_pd`, `portfolio_expected_loss`, `loss_reduction_vs_baseline`.
