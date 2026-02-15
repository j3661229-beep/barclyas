"""
Synthetic banking data generation for Pre-Delinquency Intervention Engine.
Simulates 1000 customers, 6 months of transactions, and injects financial stress patterns.
No protected attributes (gender, religion, etc.) are used.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    DEFAULT_RATE,
    MONTHS_HISTORY,
    N_CUSTOMERS,
    STRESS_FRACTION,
    CUSTOMERS_CSV,
    TRANSACTIONS_CSV,
    TARGET_COL,
)


def _ensure_data_dir() -> Path:
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _generate_customer_ids(n: int) -> list[str]:
    """Generate unique customer identifiers."""
    return [f"C{i:06d}" for i in range(1, n + 1)]


def _generate_base_amounts(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate base salary, EMI, and utility amounts per customer (realistic ranges)."""
    # Salary: 30k–150k (in rupees)
    salary = rng.uniform(30_000, 150_000, n).round(2)
    # EMI: 15–40% of salary
    emi_ratio = rng.uniform(0.15, 0.40, n)
    emi = (salary * emi_ratio).round(2)
    # Monthly utility: 2–8k
    utility = rng.uniform(2_000, 8_000, n).round(2)
    return {"salary": salary, "emi": emi, "utility": utility}


def _month_range(months: int, end_date: datetime | None = None) -> list[datetime]:
    """Last day of each month for the last `months` months."""
    end = end_date or datetime.now()
    out: list[datetime] = []
    for i in range(months - 1, -1, -1):
        # Last day of month
        d = end.replace(day=1) - timedelta(days=1) - timedelta(days=30 * i)
        # Approximate month end
        month_end = d.replace(day=28) + timedelta(days=4)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        out.append(month_end)
    return sorted(out)


def _generate_transactions_for_customer(
    customer_id: str,
    salary: float,
    emi: float,
    utility: float,
    is_stressed: bool,
    rng: np.random.Generator,
    end_date: datetime,
) -> list[dict[str, Any]]:
    """
    Generate 6 months of transactions for one customer.
    Stress: salary delay 3–7 days, balance decline, more lending app usage,
    late utility, reduced discretionary (fewer UPI/ATM for non-essential).
    """
    rows: list[dict[str, Any]] = []
    # Start balance: 0.5–3x monthly salary
    balance = rng.uniform(salary * 0.5, salary * 3.0)
    month_ends = _month_range(MONTHS_HISTORY, end_date)
    # Track salary credit day per month for EMI timing
    salary_delays: list[int] = []
    utility_delays: list[int] = []

    for m, month_end in enumerate(month_ends):
        # --- Salary credit (one per month) ---
        salary_delay_days = 0
        if is_stressed and rng.random() < 0.7:
            salary_delay_days = int(rng.integers(3, 8))
        salary_delays.append(salary_delay_days)
        # Expected salary: 1st of the same month as month_end
        expected_salary_date = month_end.replace(day=1)
        salary_date = expected_salary_date + timedelta(days=salary_delay_days)
        salary_date = min(salary_date, month_end)
        balance += salary
        rows.append({
            "customer_id": customer_id,
            "date": salary_date,
            "transaction_type": "salary_credit",
            "amount": salary,
            "balance_after": balance,
            "is_delayed": salary_delay_days > 0,
            "salary_delay_days": salary_delay_days,  # for feature engineering
        })

        # --- EMI debit (same day or a few days after salary for stressed might be before salary) ---
        emi_day = min(salary_date.day + 2, 28)
        emi_date = salary_date.replace(day=emi_day) if emi_day <= 28 else salary_date + timedelta(days=2)
        if balance >= emi:
            balance -= emi
            rows.append({
                "customer_id": customer_id,
                "date": emi_date,
                "transaction_type": "emi_debit",
                "amount": -emi,
                "balance_after": balance,
                "is_delayed": False,
            })
        else:
            # Failed EMI (record as failed_auto_debit)
            rows.append({
                "customer_id": customer_id,
                "date": emi_date,
                "transaction_type": "failed_auto_debit",
                "amount": -emi,
                "balance_after": balance,
                "is_delayed": True,
            })

        # --- Utility (monthly; stressed = late sometimes) ---
        util_delay = 0
        if is_stressed and rng.random() < 0.5:
            util_delay = int(rng.integers(5, 15))
        utility_delays.append(util_delay)
        util_date = month_end - timedelta(days=25) + timedelta(days=util_delay)
        util_date = min(util_date, month_end)
        if balance >= utility:
            balance -= utility
            rows.append({
                "customer_id": customer_id,
                "date": util_date,
                "transaction_type": "utility_payment",
                "amount": -utility,
                "balance_after": balance,
                "is_delayed": util_delay > 0,
            })

        # --- ATM withdrawals (discretionary; stressed = fewer) ---
        n_atm = rng.integers(4, 12) if not is_stressed else rng.integers(1, 5)
        for _ in range(n_atm):
            amt = -float(rng.integers(500, 5000, 1)[0])
            d = month_end - timedelta(days=int(rng.integers(0, 28)))
            if balance + abs(amt) >= 0:
                balance += amt
                rows.append({
                    "customer_id": customer_id,
                    "date": d,
                    "transaction_type": "atm_withdrawal",
                    "amount": amt,
                    "balance_after": balance,
                    "is_delayed": False,
                })

        # --- UPI (discretionary; stressed = fewer) ---
        n_upi = rng.integers(8, 25) if not is_stressed else rng.integers(2, 10)
        for _ in range(n_upi):
            amt = -float(rng.integers(100, 3000, 1)[0])
            d = month_end - timedelta(days=int(rng.integers(0, 28)))
            if balance + abs(amt) >= 0:
                balance += amt
                rows.append({
                    "customer_id": customer_id,
                    "date": d,
                    "transaction_type": "upi_transfer",
                    "amount": amt,
                    "balance_after": balance,
                    "is_delayed": False,
                })

        # --- Lending app (stressed = more usage) ---
        n_lending = rng.integers(0, 2) if not is_stressed else rng.integers(2, 8)
        for _ in range(n_lending):
            amt = float(rng.integers(2000, 15000, 1)[0])  # credit
            d = month_end - timedelta(days=int(rng.integers(0, 28)))
            balance += amt
            rows.append({
                "customer_id": customer_id,
                "date": d,
                "transaction_type": "lending_app",
                "amount": amt,
                "balance_after": balance,
                "is_delayed": False,
            })
    # Sort by date
    rows.sort(key=lambda x: x["date"])
    return rows


def _assign_default_labels(
    customer_ids: list[str],
    transactions_df: pd.DataFrame,
    stress_flags: dict[str, bool],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute binary target default_next_30_days (0/1).
    Higher probability for stressed customers and those with failed_auto_debit / low balance.
    """
    default_probs: list[float] = []
    for cid in customer_ids:
        tx = transactions_df[transactions_df["customer_id"] == cid]
        failed = (tx["transaction_type"] == "failed_auto_debit").sum()
        lending = (tx["transaction_type"] == "lending_app").sum()
        balance_series = tx["balance_after"]
        min_balance = balance_series.min() if len(balance_series) else 0
        stress = stress_flags.get(cid, False)
        # Base probability
        p = DEFAULT_RATE
        if stress:
            p += 0.35
        p += min(0.2, failed * 0.08)
        p += min(0.15, lending * 0.03)
        if min_balance < 0:
            p += 0.2
        p = min(0.95, p)
        default_probs.append(p)
    probs = np.array(default_probs)
    defaults = (rng.random(len(customer_ids)) < probs).astype(int)
    return pd.DataFrame({"customer_id": customer_ids, TARGET_COL: defaults})


def generate_synthetic_data(
    n_customers: int = N_CUSTOMERS,
    months: int = MONTHS_HISTORY,
    stress_fraction: float = STRESS_FRACTION,
    seed: int | None = 42,
    end_date: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic customers, transactions, and default labels.

    Returns:
        customers_df: customer_id (and any non-PII metadata if needed).
        transactions_df: customer_id, date, transaction_type, amount, balance_after, is_delayed.
        targets_df: customer_id, default_next_30_days.
    """
    rng = np.random.default_rng(seed)
    end_date = end_date or datetime.now()
    customer_ids = _generate_customer_ids(n_customers)
    n_stress = int(n_customers * stress_fraction)
    stress_customers = set(rng.choice(customer_ids, size=n_stress, replace=False))
    stress_flags = {cid: (cid in stress_customers) for cid in customer_ids}

    amounts = _generate_base_amounts(n_customers, rng)

    all_tx: list[dict[str, Any]] = []
    for i, cid in enumerate(customer_ids):
        rows = _generate_transactions_for_customer(
            customer_id=cid,
            salary=float(amounts["salary"][i]),
            emi=float(amounts["emi"][i]),
            utility=float(amounts["utility"][i]),
            is_stressed=(cid in stress_customers),
            rng=rng,
            end_date=end_date,
        )
        all_tx.extend(rows)

    transactions_df = pd.DataFrame(all_tx)
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "stress_injected": [stress_flags[cid] for cid in customer_ids],
    })

    targets_df = _assign_default_labels(customer_ids, transactions_df, stress_flags, rng)
    return customers_df, transactions_df, targets_df


def save_datasets(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    data_dir: Path | None = None,
) -> None:
    """Save generated DataFrames to CSV in data directory."""
    data_dir = data_dir or _ensure_data_dir()
    customers_df.to_csv(data_dir / CUSTOMERS_CSV, index=False)
    transactions_df.to_csv(data_dir / TRANSACTIONS_CSV, index=False)
    # Merge targets into a single customer-level dataset for downstream feature engineering
    customer_with_target = customers_df.merge(targets_df, on="customer_id")
    customer_with_target.to_csv(data_dir / "customer_targets.csv", index=False)
    # Keep targets separate for feature_engineering to join
    targets_df.to_csv(data_dir / "targets.csv", index=False)


def run_data_generation(
    n_customers: int = N_CUSTOMERS,
    months: int = MONTHS_HISTORY,
    stress_fraction: float = STRESS_FRACTION,
    seed: int | None = 42,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry: generate and optionally save synthetic data.

    Returns:
        customers_df, transactions_df, targets_df
    """
    customers_df, transactions_df, targets_df = generate_synthetic_data(
        n_customers=n_customers,
        months=months,
        stress_fraction=stress_fraction,
        seed=seed,
    )
    if save:
        save_datasets(customers_df, transactions_df, targets_df)
    return customers_df, transactions_df, targets_df


if __name__ == "__main__":
    run_data_generation(save=True)
    print("Data generation complete. Check pre_delinquency_engine/data/")
