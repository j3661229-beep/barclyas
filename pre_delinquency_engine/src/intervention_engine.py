"""
Intervention engine for Pre-Delinquency Intervention Engine.
Triggers proactive interventions for High-risk customers and stores them in SQLite.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR

# Default DB path: data folder
DEFAULT_DB_PATH = DATA_DIR / "interventions.db"


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get SQLite connection; create DB and table if needed."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            risk_tier TEXT NOT NULL,
            risk_score REAL NOT NULL,
            action_type TEXT NOT NULL,
            action_params TEXT,
            status TEXT DEFAULT 'triggered',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# Intervention types for High risk
INTERVENTION_PAYMENT_HOLIDAY = "payment_holiday"
INTERVENTION_EMI_RESTRUCTURE = "emi_restructure_suggestion"
INTERVENTION_SMS_REMINDER = "soft_sms_reminder"


def get_default_actions_for_high_risk() -> list[dict[str, Any]]:
    """
    Default set of interventions when risk tier is High.
    Business logic: offer payment holiday, suggest EMI restructure, send soft SMS.
    """
    return [
        {"action_type": INTERVENTION_PAYMENT_HOLIDAY, "action_params": "1_month"},
        {"action_type": INTERVENTION_EMI_RESTRUCTURE, "action_params": "suggest_extension"},
        {"action_type": INTERVENTION_SMS_REMINDER, "action_params": "upcoming_emi"},
    ]


def trigger_intervention(
    customer_id: str,
    risk_tier: str,
    risk_score: float,
    actions: list[dict[str, Any]] | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    If risk_tier is High, trigger default interventions and persist to SQLite.
    Returns list of recorded intervention rows (each with id, customer_id, action_type, status, created_at).
    """
    if risk_tier != "High":
        return []
    actions = actions or get_default_actions_for_high_risk()
    conn = get_connection(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    recorded = []
    try:
        for a in actions:
            cur = conn.execute(
                """INSERT INTO interventions (customer_id, risk_tier, risk_score, action_type, action_params, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'triggered', ?)""",
                (
                    customer_id,
                    risk_tier,
                    risk_score,
                    a["action_type"],
                    a.get("action_params", ""),
                    created_at,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
            recorded.append({
                "id": row_id,
                "customer_id": customer_id,
                "risk_tier": risk_tier,
                "risk_score": risk_score,
                "action_type": a["action_type"],
                "action_params": a.get("action_params", ""),
                "status": "triggered",
                "created_at": created_at,
            })
    finally:
        conn.close()
    return recorded


def get_interventions(
    customer_id: str | None = None,
    limit: int = 500,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Fetch intervention logs from DB, optionally filtered by customer_id."""
    conn = get_connection(db_path)
    try:
        if customer_id:
            rows = conn.execute(
                "SELECT id, customer_id, risk_tier, risk_score, action_type, action_params, status, created_at "
                "FROM interventions WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
                (customer_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, customer_id, risk_tier, risk_score, action_type, action_params, status, created_at "
                "FROM interventions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "customer_id": r[1],
                "risk_tier": r[2],
                "risk_score": r[3],
                "action_type": r[4],
                "action_params": r[5],
                "status": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]
    finally:
        conn.close()
