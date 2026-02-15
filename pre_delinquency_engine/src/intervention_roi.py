"""
Intervention ROI optimization: P(intervention prevents default), cost per intervention, net benefit, ROI per type.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    AVG_EMI_DEFAULT,
    DEFAULT_LGD,
    INTERVENTION_COST_EMI_RESTRUCTURE,
    INTERVENTION_COST_PAYMENT_HOLIDAY,
    INTERVENTION_COST_SMS,
    INTERVENTION_PREVENT_PROBABILITY,
)
from .intervention_engine import (
    INTERVENTION_EMI_RESTRUCTURE,
    INTERVENTION_PAYMENT_HOLIDAY,
    INTERVENTION_SMS_REMINDER,
)


# Cost per intervention type (config-driven)
INTERVENTION_COSTS = {
    INTERVENTION_SMS_REMINDER: INTERVENTION_COST_SMS,
    INTERVENTION_PAYMENT_HOLIDAY: INTERVENTION_COST_PAYMENT_HOLIDAY,
    INTERVENTION_EMI_RESTRUCTURE: INTERVENTION_COST_EMI_RESTRUCTURE,
}


def cost_per_intervention(action_type: str) -> float:
    """Return cost for a single intervention of given type."""
    return INTERVENTION_COSTS.get(action_type, 50.0)


def expected_benefit_if_prevented(
    avg_emi: float = AVG_EMI_DEFAULT,
    lgd: float = DEFAULT_LGD,
) -> float:
    """Benefit per default prevented = exposure * (1 - recovery) ≈ EMI * 3 * LGD (simplified)."""
    return avg_emi * 3.0 * lgd


def net_benefit_per_customer(
    prob_prevent: float = INTERVENTION_PREVENT_PROBABILITY,
    cost: float = 100.0,
    benefit_if_prevented: float | None = None,
) -> float:
    """Net benefit = E[benefit] - cost = prob_prevent * benefit_if_prevented - cost."""
    benefit_if_prevented = benefit_if_prevented or expected_benefit_if_prevented()
    return prob_prevent * benefit_if_prevented - cost


def roi_per_intervention_type(
    prob_prevent: float = INTERVENTION_PREVENT_PROBABILITY,
    avg_emi: float = AVG_EMI_DEFAULT,
    lgd: float = DEFAULT_LGD,
) -> list[dict[str, Any]]:
    """ROI per intervention type: cost, expected benefit, net benefit, ROI %."""
    benefit = expected_benefit_if_prevented(avg_emi=avg_emi, lgd=lgd)
    out = []
    for action_type, cost in INTERVENTION_COSTS.items():
        exp_benefit = prob_prevent * benefit
        net = exp_benefit - cost
        roi_pct = (net / cost * 100) if cost > 0 else 0.0
        out.append({
            "action_type": action_type,
            "cost": cost,
            "expected_benefit": exp_benefit,
            "net_benefit": net,
            "roi_pct": roi_pct,
        })
    return out


def portfolio_intervention_roi_summary(
    interventions_df: pd.DataFrame,
    action_type_col: str = "action_type",
    prob_prevent: float = INTERVENTION_PREVENT_PROBABILITY,
) -> dict[str, Any]:
    """Aggregate ROI across interventions (e.g. from get_interventions)."""
    if interventions_df.empty:
        return {"total_cost": 0.0, "total_expected_benefit": 0.0, "total_net_benefit": 0.0, "by_type": []}
    benefit = expected_benefit_if_prevented()
    total_cost = 0.0
    total_expected_benefit = 0.0
    by_type = []
    for at in interventions_df[action_type_col].unique():
        count = (interventions_df[action_type_col] == at).sum()
        cost = count * cost_per_intervention(at)
        exp_ben = count * prob_prevent * benefit
        total_cost += cost
        total_expected_benefit += exp_ben
        by_type.append({
            "action_type": at,
            "count": int(count),
            "cost": cost,
            "expected_benefit": exp_ben,
            "net_benefit": exp_ben - cost,
        })
    return {
        "total_cost": total_cost,
        "total_expected_benefit": total_expected_benefit,
        "total_net_benefit": total_expected_benefit - total_cost,
        "by_type": by_type,
    }
