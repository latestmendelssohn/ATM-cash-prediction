"""Core tests for the pure-Python forecasting + cash-planning logic.

Run:  pytest        (needs no third-party packages)
"""
import math

import pytest

import data
import models as M
from analyst import build_prompt, cash_plan_report, forecast_report, inr


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
    _, y = data.load_series("data/atm_transactions.csv", "ATM001")
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
    atms = data.list_atms("data/atm_transactions.csv")
    assert atms == ["ATM001", "ATM002", "ATM003", "ATM004", "ATM005"]
    d, y = data.load_series("data/atm_transactions.csv", "ATM001")
    assert len(y) == 1095 and len(d) == 1095
