r"""
Cash-inventory optimisation  (pure standard library).
=====================================================

A demand forecast is only useful if it drives a *decision*: how much cash to
load into each ATM, and when to send a replenishment van. This module maps the
probabilistic forecast onto that decision using classical inventory theory.

Newsvendor critical ratio
--------------------------
Let :math:`C_u` be the under-stock (stock-out) penalty per rupee short and
:math:`C_o` the over-stock (holding / opportunity) cost per rupee idle. The
cost-optimal service level is the newsvendor critical ratio

    .. math:: p^\* = \frac{C_u}{C_u + C_o},

and the optimal cash to load for a day with demand :math:`D\sim\mathcal N(\mu,\sigma^2)`
is the quantile :math:`\mu + z_{p^\*}\,\sigma` (a *base-stock* level). The term
:math:`z_{p^\*}\sigma` is the **safety stock** that buffers demand uncertainty.

For a multi-day replenishment cycle of length :math:`L` the relevant quantity
is cycle demand :math:`\sum_{t=1}^{L} D_t`; assuming independence its variance
adds, so the safety stock scales like :math:`z_{p^\*}\,\sigma\sqrt{L}`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Gaussian helpers (no scipy)
# ---------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    r"""Inverse standard-normal CDF :math:`z_p` (Acklam's rational approximation).

    Accurate to ~1e-9 over the open interval (0,1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")

    # coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def newsvendor_critical_ratio(stockout_penalty: float, holding_cost: float) -> float:
    r""":math:`p^\*=C_u/(C_u+C_o)` -- the cost-optimal service level."""
    if stockout_penalty <= 0 or holding_cost < 0:
        raise ValueError("penalty must be > 0 and holding cost >= 0")
    return stockout_penalty / (stockout_penalty + holding_cost)


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------

@dataclass
class CashPlan:
    service_level: float
    z: float
    horizon: int
    per_day_load: List[float] = field(default_factory=list)
    cycle_load: float = 0.0
    safety_stock: float = 0.0
    expected_stockout_prob: float = 0.0


def recommend_cash_load(
    forecast: Sequence[float],
    residual_std: float,
    service_level: float | None = None,
    stockout_penalty: float | None = None,
    holding_cost: float | None = None,
) -> CashPlan:
    r"""Recommend the base-stock cash load for a replenishment cycle.

    Either pass an explicit ``service_level`` **or** the two costs (from which
    the newsvendor optimal level is derived). ``forecast`` is the per-day mean
    demand over the cycle; ``residual_std`` the one-day forecast-error std.
    """
    if service_level is None:
        if stockout_penalty is None or holding_cost is None:
            raise ValueError("provide service_level or (stockout_penalty, holding_cost)")
        service_level = newsvendor_critical_ratio(stockout_penalty, holding_cost)

    z = normal_quantile(service_level)
    horizon = len(forecast)

    # per-day base-stock levels
    per_day = [mu + z * residual_std for mu in forecast]

    # cycle (cover the whole horizon between visits): variance adds over days
    cycle_mean = sum(forecast)
    cycle_sigma = residual_std * math.sqrt(horizon)
    cycle_load = cycle_mean + z * cycle_sigma
    safety_stock = z * cycle_sigma

    # probability the *cycle* demand exceeds the loaded amount
    if cycle_sigma > 0:
        stockout_prob = 1.0 - normal_cdf((cycle_load - cycle_mean) / cycle_sigma)
    else:
        stockout_prob = 0.0

    return CashPlan(
        service_level=service_level,
        z=z,
        horizon=horizon,
        per_day_load=per_day,
        cycle_load=cycle_load,
        safety_stock=safety_stock,
        expected_stockout_prob=stockout_prob,
    )


def simulate_policy(
    actual_demand: Sequence[float],
    cycle_load: float,
    replenish_cost: float,
    stockout_penalty: float,
    holding_rate_daily: float,
) -> dict:
    r"""Cost a base-stock policy against a realised demand path.

    One replenishment loads ``cycle_load`` at day 0; we then draw down against
    ``actual_demand`` and tally: 1 replenishment cost, a stock-out penalty on
    every day cash runs out, and a holding cost on average idle cash.
    """
    balance = cycle_load
    total_holding = 0.0
    stockout_days = 0
    served = 0.0
    for d in actual_demand:
        start = balance
        dispensed = min(d, max(balance, 0.0))
        served += dispensed
        if d > start + 1e-9:
            stockout_days += 1
        balance = start - dispensed
        total_holding += holding_rate_daily * max(balance, 0.0)

    cost = (
        replenish_cost
        + stockout_penalty * stockout_days
        + total_holding
    )
    return {
        "cycle_load": cycle_load,
        "total_cost": cost,
        "replenish_cost": replenish_cost,
        "stockout_days": stockout_days,
        "stockout_penalty_total": stockout_penalty * stockout_days,
        "holding_cost_total": total_holding,
        "cash_served": served,
        "fill_rate": served / sum(actual_demand) if sum(actual_demand) else 1.0,
    }
