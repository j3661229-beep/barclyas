"""
Risk trajectory tracking: daily risk score history, risk velocity, sudden stress spike detection.
risk_velocity = current_PD - 7_day_average_PD
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, RISK_HISTORY_DB_FILENAME, RISK_VELOCITY_SPIKE_THRESHOLD
from .risk_scoring import score_batch
from .feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Risk history stored in SQLite (same pattern as interventions)
RISK_HISTORY_DB = DATA_DIR / RISK_HISTORY_DB_FILENAME


def _get_conn(db_path: Path | None = None) -> Any:
    import sqlite3
    db_path = db_path or RISK_HISTORY_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_score_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            pd REAL NOT NULL,
            risk_score REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_history_customer_date ON risk_score_history(customer_id, as_of_date)")
    conn.commit()
    return conn


def record_risk_snapshot(
    customer_id: str,
    pd_value: float,
    risk_score: float,
    risk_tier: str,
    as_of_date: datetime | str | None = None,
    db_path: Path | None = None,
) -> None:
    """Append one day's risk snapshot for a customer."""
    as_of_date = as_of_date or datetime.now(timezone.utc).date().isoformat()
    if hasattr(as_of_date, "isoformat"):
        as_of_date = as_of_date.isoformat()[:10]
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO risk_score_history (customer_id, as_of_date, pd, risk_score, risk_tier, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (customer_id, as_of_date, pd_value, risk_score, risk_tier, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def record_portfolio_snapshot(
    risk_df: pd.DataFrame,
    customer_id_col: str = "customer_id",
    as_of_date: datetime | str | None = None,
    db_path: Path | None = None,
) -> None:
    """Record risk snapshot for all customers in risk_df (columns: customer_id, default_probability, risk_score, risk_tier)."""
    as_of_date = as_of_date or datetime.now(timezone.utc).date().isoformat()
    if hasattr(as_of_date, "isoformat"):
        as_of_date = as_of_date.isoformat()[:10]
    conn = _get_conn(db_path)
    try:
        for _, row in risk_df.iterrows():
            cid = row[customer_id_col]
            pd_val = row.get("default_probability", row.get("pd", 0.0))
            score = row.get("risk_score", 0.0)
            tier = row.get("risk_tier", "Low")
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO risk_score_history (customer_id, as_of_date, pd, risk_score, risk_tier, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cid, as_of_date, pd_val, score, tier, created_at),
            )
        conn.commit()
    finally:
        conn.close()


def get_risk_history(
    customer_id: str,
    days: int = 30,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Get recent risk history for one customer."""
    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            """SELECT as_of_date, pd, risk_score, risk_tier FROM risk_score_history
               WHERE customer_id = ? ORDER BY as_of_date DESC LIMIT ?""",
            (customer_id, days),
        ).fetchall()
        return pd.DataFrame(rows, columns=["as_of_date", "pd", "risk_score", "risk_tier"])
    finally:
        conn.close()


def compute_risk_velocity(
    current_pd: float,
    history_pds: list[float] | pd.Series,
    window_days: int = 7,
) -> float:
    """risk_velocity = current_PD - 7_day_average_PD. Positive = risk increasing."""
    if not history_pds or len(history_pds) == 0:
        return 0.0
    recent = list(history_pds)[:window_days]
    avg = sum(recent) / len(recent)
    return float(current_pd - avg)


def detect_stress_spike(
    current_pd: float,
    history_pds: list[float] | pd.Series,
    window_days: int = 7,
    threshold: float | None = None,
) -> bool:
    """True if risk_velocity exceeds threshold (sudden stress spike)."""
    thresh = threshold if threshold is not None else RISK_VELOCITY_SPIKE_THRESHOLD
    vel = compute_risk_velocity(current_pd, history_pds, window_days)
    return vel >= thresh


def get_risk_progression_before_default(
    customer_id: str,
    default_date: str | None,
    days_before: int = 30,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Get risk progression in the days before a (simulated) default date. For plotting."""
    df = get_risk_history(customer_id, days=days_before, db_path=db_path)
    if df.empty or default_date is None:
        return df
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    return df.sort_values("as_of_date")


def simulate_risk_history_for_dashboard(
    feature_df: pd.DataFrame,
    risk_scores_df: pd.DataFrame,
    num_days: int = 14,
    db_path: Path | None = None,
) -> None:
    """
    Simulate daily snapshots by adding small noise to current scores so we have history.
    In production, this would be real daily scores.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    base = risk_scores_df["default_probability"].values
    for d in range(num_days):
        as_of = (datetime.now(timezone.utc) - timedelta(days=d)).date().isoformat()
        noise = rng.uniform(-0.02, 0.02, len(base))
        pd_sim = np.clip(base + noise, 0.01, 0.99)
        from .risk_scoring import probability_to_score, score_to_tier
        scores = [probability_to_score(p) for p in pd_sim]
        tiers = [score_to_tier(s) for s in scores]
        sim_df = pd.DataFrame({
            "customer_id": feature_df["customer_id"].values,
            "default_probability": pd_sim,
            "risk_score": scores,
            "risk_tier": tiers,
        })
        record_portfolio_snapshot(sim_df, as_of_date=as_of, db_path=db_path)
    logger.info("Simulated %s days of risk history", num_days)
