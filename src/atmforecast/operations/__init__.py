"""Cash-inventory optimisation built on top of the demand forecast."""

from .cash_optimizer import (  # noqa: F401
    CashPlan,
    newsvendor_critical_ratio,
    normal_quantile,
    recommend_cash_load,
    simulate_policy,
)
