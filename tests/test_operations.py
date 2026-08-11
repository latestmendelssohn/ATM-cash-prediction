"""Tests for the pure-Python cash-inventory optimiser."""
import math

import pytest

from atmforecast.operations.cash_optimizer import (
    newsvendor_critical_ratio,
    normal_cdf,
    normal_quantile,
    recommend_cash_load,
    simulate_policy,
)


def test_normal_quantile_known_values():
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-6)
    assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert normal_quantile(0.95) == pytest.approx(1.644854, abs=1e-4)


def test_normal_cdf_quantile_are_inverses():
    for p in (0.1, 0.3, 0.5, 0.8, 0.99):
        assert normal_cdf(normal_quantile(p)) == pytest.approx(p, abs=1e-6)


def test_critical_ratio():
    # penalty 5000, holding 500 -> 5000/5500
    assert newsvendor_critical_ratio(5000, 500) == pytest.approx(10 / 11)


def test_higher_service_level_loads_more_cash():
    fc = [100.0] * 7
    low = recommend_cash_load(fc, residual_std=20.0, service_level=0.80)
    high = recommend_cash_load(fc, residual_std=20.0, service_level=0.99)
    assert high.cycle_load > low.cycle_load
    assert high.safety_stock > low.safety_stock


def test_safety_stock_scales_with_sqrt_horizon():
    fc = [100.0] * 4
    plan = recommend_cash_load(fc, residual_std=10.0, service_level=0.95)
    z = normal_quantile(0.95)
    assert plan.safety_stock == pytest.approx(z * 10.0 * math.sqrt(4))


def test_stockout_prob_matches_target():
    fc = [100.0] * 7
    plan = recommend_cash_load(fc, residual_std=15.0, service_level=0.95)
    # loading to the 95% quantile => residual stockout prob ~ 5%
    assert plan.expected_stockout_prob == pytest.approx(0.05, abs=1e-3)


def test_simulate_policy_no_stockout_when_overloaded():
    demand = [100.0, 120.0, 90.0]
    res = simulate_policy(demand, cycle_load=1000.0, replenish_cost=500,
                          stockout_penalty=5000, holding_rate_daily=0.0001)
    assert res["stockout_days"] == 0
    assert res["fill_rate"] == pytest.approx(1.0)


def test_simulate_policy_penalises_stockout():
    demand = [100.0, 100.0, 100.0]
    res = simulate_policy(demand, cycle_load=150.0, replenish_cost=500,
                          stockout_penalty=5000, holding_rate_daily=0.0)
    assert res["stockout_days"] >= 1
    assert res["total_cost"] > 500
