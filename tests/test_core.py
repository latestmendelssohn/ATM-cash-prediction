"""Core tests for the pure-Python forecasting + cash-planning logic.

Run:  pytest        (needs no third-party packages beyond pytest)
"""
import math
from statistics import NormalDist

import pytest

import data
import models as M
from analyst import build_prompt, cash_plan_report, forecast_report, inr

CSV = data.DEFAULT_CSV


# --------------------------- metrics ---------------------------

def test_metrics_perfect_forecast():
    y = [10.0, 20.0, 30.0]
    assert M.mae(y, y) == 0 and M.rmse(y, y) == 0 and M.mape(y, y) == 0


def test_mae_rmse_known():
    assert M.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert M.rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(math.sqrt(5 / 3))


def test_mase_below_one_means_beats_naive():
    train = [10 + 0.5 * w + v for w in range(4) for v in (0, 2, 4, 1, 0, 3, 5)]
    assert M.mase([10.0], [10.0], train, m=7) == 0.0


def test_backtest_rejects_series_shorter_than_one_window():
    with pytest.raises(ValueError, match="too short"):
        M.rolling_origin_backtest(list(range(100)), lambda: M.SeasonalNaive(7),
                                  horizon=14, min_train=365)


# --------------------------- models ---------------------------

def test_seasonal_naive_repeats_week():
    m = M.SeasonalNaive(7).fit([1, 2, 3, 4, 5, 6, 7] * 3)
    assert m.predict(9) == [1, 2, 3, 4, 5, 6, 7, 1, 2]


def test_holt_winters_recovers_weekly_pattern():
    pat = [10, -5, 0, 3, -2, 8, -14]
    y = [100 + pat[t % 7] for t in range(70)]
    fc = M.HoltWinters(7, None, "add").fit(y).predict(7)
    for k in range(7):
        assert abs(fc[k] - (100 + pat[k % 7])) < 3.0


def test_holt_winters_beats_mean_on_seasonal_series():
    _, y = data.load_series(CSV, "ATM001")
    board = M.leaderboard(y)
    assert board[0]["model"] == "holt_winters"
    assert board[0]["MASE"] < 1.0


def test_interval_contains_point_and_widens():
    y = [100 + 2 * t + 5 * math.sin(t) for t in range(60)]
    p, lo, hi = M.HoltWinters(7, "add", "add").fit(y).predict_interval(10)
    assert all(a <= b <= c for a, b, c in zip(lo, p, hi))
    assert (hi[-1] - lo[-1]) >= (hi[0] - lo[0])


# --------------------------- cash planning ---------------------------

def test_normal_quantile_known_values():
    assert M.normal_quantile(0.5) == pytest.approx(0.0, abs=1e-6)
    assert M.normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-4)


def test_normal_quantile_matches_stdlib():
    """The hand-rolled Acklam approximation is kept on purpose, so pin its accuracy."""
    for p in (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999):
        assert M.normal_quantile(p) == pytest.approx(NormalDist().inv_cdf(p), abs=1e-6)


def test_z_for_level_is_exact_not_snapped_to_a_table():
    assert M.z_for_level(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert M.z_for_level(0.5) == pytest.approx(0.674490, abs=1e-5)
    with pytest.raises(ValueError):
        M.z_for_level(1.0)


def test_explicit_zero_smoothing_parameter_is_respected():
    y = [100 + (t % 7) for t in range(60)]
    m = M.HoltWinters(7, "add", "add", optimize=False,
                      alpha=0.0, beta=0.2, gamma=0.2).fit(y)
    assert m.alpha == 0.0


def test_costs_set_the_service_level():
    """Newsvendor critical ratio: p* = cu / (cu + co)."""
    plan = M.recommend_cash_load([100.0] * 7, 10.0, cu=9.0, co=1.0)
    assert plan["service_level"] == pytest.approx(0.9)
    dearer = M.recommend_cash_load([100.0] * 7, 10.0, cu=99.0, co=1.0)
    assert dearer["cycle_load"] > plan["cycle_load"]


def test_measured_cycle_sigma_is_used_when_given():
    fc = [100.0] * 14
    assumed = M.recommend_cash_load(fc, 10.0, 0.95)
    measured = M.recommend_cash_load(fc, 10.0, 0.95, cycle_sigma=500.0)
    assert measured["cycle_sigma"] == 500.0
    assert measured["safety_stock"] > assumed["safety_stock"]


def test_cycle_sigma_from_backtest_is_positive_on_real_data():
    _, y = data.load_series(CSV, "ATM001")
    s = M.cycle_sigma_from_backtest(y, lambda: M.SeasonalNaive(7), horizon=14)
    assert s > 0


def test_higher_service_level_loads_more():
    fc = [100.0] * 7
    lo = M.recommend_cash_load(fc, 20.0, 0.80)
    hi = M.recommend_cash_load(fc, 20.0, 0.99)
    assert hi["cycle_load"] > lo["cycle_load"]


def test_stockout_prob_matches_target():
    plan = M.recommend_cash_load([100.0] * 7, 15.0, 0.95)
    assert plan["expected_stockout_prob"] == pytest.approx(0.05, abs=1e-3)


def test_topup_computed():
    plan = M.recommend_cash_load([100.0] * 5, 10.0, 0.95, current_balance=300.0)
    assert plan["suggested_topup"] == pytest.approx(max(0.0, plan["cycle_load"] - 300.0))


# --------------------------- reports / data ---------------------------

def test_inr_formatting():
    assert inr(15_000_000) == "Rs 1.50 Cr"
    assert inr(250_000) == "Rs 2.50 L"


def test_forecast_report_metadata():
    r = forecast_report("ATM001", "holt_winters", "2024-01-01", [1e5, 1.2e5])
    assert r["metadata"]["kind"] == "forecast" and r["metadata"]["atm_id"] == "ATM001"


def test_build_prompt_grounds_context():
    p = build_prompt("how much?", [{"text": "load Rs 5 Cr", "metadata": {"kind": "cash_plan"}}])
    assert "CONTEXT:" in p and "QUESTION: how much?" in p and "Rs 5 Cr" in p


def test_dataset_loads():
    atms = data.list_atms(CSV)
    assert atms == ["ATM001", "ATM002", "ATM003", "ATM004", "ATM005"]
    d, y = data.load_series(CSV, "ATM001")
    assert len(y) == 1095 and len(d) == 1095
