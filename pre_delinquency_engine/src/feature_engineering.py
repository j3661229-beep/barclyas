"""
Feature engineering for Pre-Delinquency Intervention Engine.
Builds behavioral and stress indicators from transaction-level data.
Modular: all features defined here; no protected attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, TARGET_COL, CUSTOMER_FEATURES_CSV


# Feature names used by model and explainability
FEATURE_NAMES = [
    "salary_delay_days",
    "balance_decline_ratio_4w",
    "utility_delay_frequency",
    "lending_app_txn_count",
    "failed_auto_debit_count",
    "atm_withdrawal_spike",
    "discretionary_spending_drop_pct",
]


def _salary_delay_days(transactions: pd.DataFrame, customer_id: str) -> float:
    """
    Average delay (in days) of salary credit from expected date (e.g. 1st of month).
    Business logic: delayed salary is a strong stress signal.
    """
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "salary_credit")
    ].copy()
    if tx.empty:
        return 0.0
    if "salary_delay_days" in tx.columns:
        return float(tx["salary_delay_days"].mean())
    if "is_delayed" in tx.columns and tx["is_delayed"].any():
        return 5.0  # approximate when delayed but no exact days
    return 0.0


def _balance_decline_ratio_4w(transactions: pd.DataFrame, customer_id: str) -> float:
    """
    Ratio: (balance 4 weeks ago - current balance) / |balance 4 weeks ago|.
    Positive = decline. Capped for stability.
    Business logic: week-over-week balance decline indicates cash flow stress.
    """
    tx = transactions[transactions["customer_id"] == customer_id].sort_values("date")
    if len(tx) < 2:
        return 0.0
    tx = tx.dropna(subset=["balance_after"])
    if tx.empty:
        return 0.0
    latest = tx["date"].max()
    four_weeks_ago = latest - pd.Timedelta(days=28)
    past = tx[tx["date"] <= four_weeks_ago]
    balance_4w = past["balance_after"].iloc[-1] if len(past) else tx["balance_after"].iloc[0]
    balance_now = tx["balance_after"].iloc[-1]
    if abs(balance_4w) < 1:
        return 0.0
    ratio = (balance_4w - balance_now) / abs(balance_4w)
    return float(max(-1.0, min(1.0, ratio)))  # cap


def _utility_delay_frequency(transactions: pd.DataFrame, customer_id: str) -> float:
    """
    Fraction of utility payments that were delayed (is_delayed=True).
    """
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "utility_payment")
    ]
    if tx.empty:
        return 0.0
    if "is_delayed" not in tx.columns:
        return 0.0
    return tx["is_delayed"].mean()


def _lending_app_txn_count(transactions: pd.DataFrame, customer_id: str) -> int:
    """Count of lending app transactions in the observation window."""
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "lending_app")
    ]
    return len(tx)


def _failed_auto_debit_count(transactions: pd.DataFrame, customer_id: str) -> int:
    """Count of failed auto-debit attempts."""
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "failed_auto_debit")
    ]
    return len(tx)


def _atm_withdrawal_spike(transactions: pd.DataFrame, customer_id: str) -> float:
    """
    Spike: (ATM count in last 4 weeks) / (avg monthly ATM count) - 1, capped.
    High spike can indicate cash hoarding or stress.
    """
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"] == "atm_withdrawal")
    ]
    if tx.empty:
        return 0.0
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    latest = tx["date"].max()
    last_4w = tx[tx["date"] >= (latest - pd.Timedelta(days=28))]
    count_4w = len(last_4w)
    months = max(1, (tx["date"].max() - tx["date"].min()).days / 30)
    avg_per_month = len(tx) / months
    if avg_per_month < 0.5:
        return 0.0
    spike = (count_4w / (avg_per_month * (28 / 30))) - 1.0
    return float(max(-0.5, min(3.0, spike)))


def _discretionary_spending_drop_pct(transactions: pd.DataFrame, customer_id: str) -> float:
    """
    Percentage drop in discretionary spending (ATM + UPI outflows) in last 4 weeks
    vs previous 4 weeks. Negative = drop.
    """
    tx = transactions[
        (transactions["customer_id"] == customer_id)
        & (transactions["transaction_type"].isin(["atm_withdrawal", "upi_transfer"]))
        & (transactions["amount"] < 0)
    ].copy()
    if len(tx) < 2:
        return 0.0
    tx["date"] = pd.to_datetime(tx["date"])
    latest = tx["date"].max()
    last_4w = tx[tx["date"] >= (latest - pd.Timedelta(days=28))]
    prev_4w = tx[
        (tx["date"] >= (latest - pd.Timedelta(days=56)))
        & (tx["date"] < (latest - pd.Timedelta(days=28)))
    ]
    spend_last = last_4w["amount"].sum()
    spend_prev = prev_4w["amount"].sum()
    if spend_prev >= 0 or spend_prev == 0:
        return 0.0
    # spend_prev is negative; spend_last is negative. Drop = less negative = increase in spending
    # We want "drop %": (prev - last) / |prev| when last > prev (less spending)
    change_pct = (spend_last - spend_prev) / abs(spend_prev)
    # change_pct > 0 means last period spent less (drop)
    return float(max(-1.0, min(1.0, change_pct)))


def build_features_for_customer(
    transactions: pd.DataFrame,
    customer_id: str,
) -> dict[str, Any]:
    """
    Build all engineered features for a single customer.
    Returns a dict with keys matching FEATURE_NAMES.
    """
    return {
        "customer_id": customer_id,
        "salary_delay_days": _salary_delay_days(transactions, customer_id),
        "balance_decline_ratio_4w": _balance_decline_ratio_4w(transactions, customer_id),
        "utility_delay_frequency": _utility_delay_frequency(transactions, customer_id),
        "lending_app_txn_count": _lending_app_txn_count(transactions, customer_id),
        "failed_auto_debit_count": _failed_auto_debit_count(transactions, customer_id),
        "atm_withdrawal_spike": _atm_withdrawal_spike(transactions, customer_id),
        "discretionary_spending_drop_pct": _discretionary_spending_drop_pct(
            transactions, customer_id
        ),
    }


def build_feature_matrix(
    transactions: pd.DataFrame,
    customer_ids: list[str] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for all customers (or given list).
    """
    if customer_ids is None:
        customer_ids = transactions["customer_id"].unique().tolist()
    if hasattr(customer_ids, "tolist"):
        customer_ids = customer_ids.tolist()
    rows = [build_features_for_customer(transactions, cid) for cid in customer_ids]
    return pd.DataFrame(rows)


def run_feature_engineering(
    data_dir: Path | None = None,
    transactions_path: Path | None = None,
    targets_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load transactions (and optionally targets), build features, merge target, save.
    Returns the feature DataFrame with default_next_30_days if targets provided.
    """
    data_dir = data_dir or DATA_DIR
    transactions_path = transactions_path or data_dir / "transactions.csv"
    targets_path = targets_path or data_dir / "targets.csv"
    output_path = output_path or data_dir / CUSTOMER_FEATURES_CSV

    transactions = pd.read_csv(transactions_path)
    transactions["date"] = pd.to_datetime(transactions["date"])

    feature_df = build_feature_matrix(transactions)
    if targets_path.exists():
        targets = pd.read_csv(targets_path)
        feature_df = feature_df.merge(targets, on="customer_id", how="left")
    feature_df.to_csv(output_path, index=False)
    return feature_df


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering complete. Output: data/customer_features.csv")
