"""
Model governance & audit trail: model version, training timestamp, feature set,
risk score timestamp, explanation snapshot, intervention decision log. SQLite-backed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR, AUDIT_DB_FILENAME

logger = logging.getLogger(__name__)

AUDIT_DB_PATH = DATA_DIR / AUDIT_DB_FILENAME


def _get_conn(db_path: Path | None = None):
    import sqlite3
    path = db_path or AUDIT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            training_timestamp TEXT NOT NULL,
            feature_set TEXT NOT NULL,
            metrics_summary TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_risk_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            pd REAL NOT NULL,
            score_timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            score_timestamp TEXT NOT NULL,
            explanation_snapshot TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_intervention_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            risk_tier TEXT NOT NULL,
            risk_score REAL NOT NULL,
            action_taken TEXT NOT NULL,
            decision_timestamp TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_model_version(
    model_version: str,
    training_timestamp: str,
    feature_set: list[str],
    metrics_summary: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> None:
    """Record a new model version and training metadata."""
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO audit_model_versions (model_version, training_timestamp, feature_set, metrics_summary, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                model_version,
                training_timestamp,
                json.dumps(feature_set),
                json.dumps(metrics_summary or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_risk_score(
    customer_id: str,
    model_version: str,
    risk_score: float,
    risk_tier: str,
    pd: float,
    score_timestamp: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Log a risk score event for audit."""
    ts = score_timestamp or datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO audit_risk_scores (customer_id, model_version, risk_score, risk_tier, pd, score_timestamp, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (customer_id, model_version, risk_score, risk_tier, pd, ts, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def log_explanation_snapshot(
    customer_id: str,
    explanation_snapshot: dict[str, Any],
  score_timestamp: str | None = None,
  db_path: Path | None = None,
) -> None:
    """Log SHAP/explanation snapshot for audit."""
    ts = score_timestamp or datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO audit_explanations (customer_id, score_timestamp, explanation_snapshot, created_at)
               VALUES (?, ?, ?, ?)""",
            (customer_id, ts, json.dumps(explanation_snapshot), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def log_intervention_decision(
    customer_id: str,
    risk_tier: str,
    risk_score: float,
    action_taken: str,
  decision_timestamp: str | None = None,
  db_path: Path | None = None,
) -> None:
    """Log intervention decision for audit."""
    ts = decision_timestamp or datetime.now(timezone.utc).isoformat()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            """INSERT INTO audit_intervention_decisions (customer_id, risk_tier, risk_score, action_taken, decision_timestamp, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (customer_id, risk_tier, risk_score, action_taken, ts, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def generate_audit_report(customer_id: str, db_path: Path | None = None) -> dict[str, Any]:
    """Generate full audit report for a customer: model versions, risk scores, explanations, interventions."""
    conn = _get_conn(db_path)
    try:
        versions = conn.execute(
            "SELECT model_version, training_timestamp, feature_set, metrics_summary FROM audit_model_versions ORDER BY id DESC LIMIT 5"
        ).fetchall()
        scores = conn.execute(
            "SELECT model_version, risk_score, risk_tier, pd, score_timestamp FROM audit_risk_scores WHERE customer_id = ? ORDER BY id DESC LIMIT 20",
            (customer_id,),
        ).fetchall()
        explanations = conn.execute(
            "SELECT score_timestamp, explanation_snapshot FROM audit_explanations WHERE customer_id = ? ORDER BY id DESC LIMIT 5",
            (customer_id,),
        ).fetchall()
        decisions = conn.execute(
            "SELECT risk_tier, risk_score, action_taken, decision_timestamp FROM audit_intervention_decisions WHERE customer_id = ? ORDER BY id DESC LIMIT 20",
            (customer_id,),
        ).fetchall()
    finally:
        conn.close()

    def parse_json(s):
        try:
            return json.loads(s) if s else None
        except Exception:
            return s

    return {
        "customer_id": customer_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_versions": [
            {"model_version": r[0], "training_timestamp": r[1], "feature_set": parse_json(r[2]), "metrics_summary": parse_json(r[3])}
            for r in versions
        ],
        "risk_score_history": [
            {"model_version": r[0], "risk_score": r[1], "risk_tier": r[2], "pd": r[3], "score_timestamp": r[4]}
            for r in scores
        ],
        "explanation_snapshots": [
            {"score_timestamp": r[0], "explanation_snapshot": parse_json(r[1])}
            for r in explanations
        ],
        "intervention_decisions": [
            {"risk_tier": r[0], "risk_score": r[1], "action_taken": r[2], "decision_timestamp": r[3]}
            for r in decisions
        ],
    }
