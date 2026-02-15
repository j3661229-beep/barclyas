"""
Configuration for Pre-Delinquency Intervention Engine.
Centralizes paths, parameters, and environment-based settings.
No hardcoded business parameters in code; use this module.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PDE_DATA_DIR", str(PROJECT_ROOT / "data")))
MODELS_DIR = Path(os.environ.get("PDE_MODELS_DIR", str(PROJECT_ROOT / "models")))
REPORTS_DIR = Path(os.environ.get("PDE_REPORTS_DIR", str(PROJECT_ROOT / "reports")))

# Data generation
N_CUSTOMERS = int(os.environ.get("PDE_N_CUSTOMERS", "1000"))
MONTHS_HISTORY = int(os.environ.get("PDE_MONTHS_HISTORY", "6"))
STRESS_FRACTION = float(os.environ.get("PDE_STRESS_FRACTION", "0.25"))
DEFAULT_RATE = float(os.environ.get("PDE_DEFAULT_RATE", "0.12"))

# Risk tiers (aligned with risk_scoring.py)
RISK_TIER_LOW = (0, 40)
RISK_TIER_MEDIUM = (40, 70)
RISK_TIER_HIGH = (70, 101)

# Credit risk (PD-LGD-EAD)
DEFAULT_LGD = float(os.environ.get("PDE_DEFAULT_LGD", "0.45"))  # 45% LGD
EAD_MULTIPLIER_EMI = float(os.environ.get("PDE_EAD_MULTIPLIER_EMI", "3.0"))  # EAD ≈ 3x monthly EMI
AVG_EMI_DEFAULT = float(os.environ.get("PDE_AVG_EMI", "15000.0"))

# Portfolio monitoring & drift
HIGH_RISK_INCREASE_ALERT_PCT = float(os.environ.get("PDE_HIGH_RISK_ALERT_PCT", "10.0"))  # Alert if High-risk % increases by X%
DRIFT_KS_THRESHOLD = float(os.environ.get("PDE_DRIFT_KS_THRESHOLD", "0.15"))  # KS p-value or statistic threshold
RISK_VELOCITY_SPIKE_THRESHOLD = float(os.environ.get("PDE_RISK_VELOCITY_SPIKE", "0.15"))  # PD change for spike

# Intervention ROI
INTERVENTION_COST_SMS = float(os.environ.get("PDE_COST_SMS", "2.0"))
INTERVENTION_COST_PAYMENT_HOLIDAY = float(os.environ.get("PDE_COST_HOLIDAY", "500.0"))
INTERVENTION_COST_EMI_RESTRUCTURE = float(os.environ.get("PDE_COST_EMI", "200.0"))
INTERVENTION_PREVENT_PROBABILITY = float(os.environ.get("PDE_INTERVENTION_PREVENT_PROB", "0.25"))

# Model & governance
MODEL_VERSION_INITIAL = "1.0.0"
AUDIT_DB_FILENAME = "governance_audit.db"
RISK_HISTORY_DB_FILENAME = "risk_trajectory.db"

# File names
CUSTOMERS_CSV = "customers.csv"
TRANSACTIONS_CSV = "transactions.csv"
CUSTOMER_FEATURES_CSV = "customer_features.csv"
TARGET_COL = "default_next_30_days"

# Logging
LOG_LEVEL = os.environ.get("PDE_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
